"""Numerical gates for the opt-in architecture-independent DSA fallback."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetK, GetS
from sglang.kernels.ops.attention.dsa.triton_mqa_logits_sm80 import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    triton_sparse_mla_fwd,
)
from sglang.srt.layers.attention.dsa.torch_dsa_fallback import (
    FP8_DTYPE,
    act_quant_torch_dsa,
    fp8_paged_mqa_logits_torch_dsa,
    fp8_ragged_mqa_logits_torch_dsa,
    sparse_mla_torch_dsa,
)
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ragged_reference(q, k, scale, weight, starts, ends):
    result = torch.full((q.shape[0], k.shape[0]), -torch.inf, dtype=torch.float64)
    q, k, scale, weight = (value.cpu().double() for value in (q, k, scale, weight))
    for row, (start, end) in enumerate(zip(starts.cpu(), ends.cpu())):
        for column in range(int(start), int(end)):
            result[row, column] = scale[column] * sum(
                weight[row, head] * max(torch.dot(q[row, head], k[column]).item(), 0.0)
                for head in range(q.shape[1])
            )
    return result.cuda().float()


def _make_paged_cache(num_pages: int, head_dim: int = 128):
    page_size = 64
    raw = torch.zeros(
        (num_pages, page_size * (head_dim + 4)), dtype=torch.uint8, device="cuda"
    )
    values = (
        torch.randn(num_pages, page_size, head_dim, device="cuda")
        .clamp(-2, 2)
        .to(FP8_DTYPE)
    )
    scales = torch.rand(num_pages, page_size, device="cuda") * 0.2 + 0.05
    raw[:, : page_size * head_dim] = values.reshape(num_pages, -1).view(torch.uint8)
    raw[:, page_size * head_dim :] = scales.contiguous().view(torch.uint8)
    return raw.view(num_pages, page_size, 1, head_dim + 4), values, scales


@pytest.mark.parametrize("head_dim", [64, 128])
def test_ragged_indexer_matches_independent_loop(head_dim):
    torch.manual_seed(19)
    q = torch.randn(4, 8, head_dim, device="cuda").clamp(-2, 2).to(FP8_DTYPE)
    k = torch.randn(11, head_dim, device="cuda").clamp(-2, 2).to(FP8_DTYPE)
    scale = torch.rand(11, device="cuda") * 0.2 + 0.05
    weight = torch.rand(4, 8, device="cuda") - 0.3
    starts = torch.tensor([0, 2, 4, 1], device="cuda")
    ends = torch.tensor([6, 9, 11, 4], device="cuda")
    expected = _ragged_reference(q, k, scale, weight, starts, ends)
    actual = fp8_ragged_mqa_logits_torch_dsa(
        q, (k, scale), weight, starts, ends, chunk_rows=2, clean_logits=False
    )
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(
        actual[torch.isfinite(actual)],
        expected[torch.isfinite(expected)],
        atol=5e-4,
        rtol=1e-5,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_paged_indexer_obeys_packed_layout_and_masks(head_dim):
    torch.manual_seed(23)
    cache, values, scales = _make_paged_cache(6, head_dim)
    q = torch.randn(2, 1, 8, head_dim, device="cuda").clamp(-2, 2).to(FP8_DTYPE)
    weight = torch.rand(2, 8, device="cuda") - 0.2
    lengths = torch.tensor([70, 121], dtype=torch.int32, device="cuda")
    pages = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda")
    actual = fp8_paged_mqa_logits_torch_dsa(
        q,
        cache,
        weight,
        lengths,
        pages,
        None,
        128,
        kv_chunk_tokens=64,
        clean_logits=False,
    )
    for batch in range(2):
        keys = torch.cat([values[page] for page in pages[batch].tolist()]).float()
        expected = torch.relu(keys @ q[batch, 0].float().T)
        expected = (expected * weight[batch].unsqueeze(0)).sum(1)
        expected *= torch.cat([scales[page] for page in pages[batch].tolist()])
        length = int(lengths[batch])
        torch.testing.assert_close(
            actual[batch, :length], expected[:length], atol=5e-4, rtol=1e-5
        )
        assert torch.isneginf(actual[batch, length:]).all()


@pytest.mark.parametrize("head_dim", [64, 128])
def test_quantized_index_cache_roundtrip_with_partial_reordered_pages(head_dim):
    torch.manual_seed(59)
    # MLA quantization remains 128-wide even when index keys are 64-wide.
    pool = SimpleNamespace(
        page_size=64,
        index_head_dim=head_dim,
        quant_block_size=128,
        indexer_quant_block_size=head_dim,
        custom_mem_pool=None,
        index_k_with_scale_buffer_dtype=torch.uint8,
        device="cuda",
        layer_num=2,
        start_layer=0,
        skip_topk_layers=[False, True],
        layer_transfer_counter=None,
    )
    cache = IndexKeyCache(pool, index_buf_size=256)
    buf = cache.get_buffer(0)
    assert buf.shape == (5, 64 * (head_dim + 4))
    assert cache.get_buffer(1).shape == (0, 64 * (head_dim + 4))
    keys = torch.randn(5, head_dim, dtype=torch.bfloat16, device="cuda")
    quantized, scales = act_quant_torch_dsa(keys, block_size=head_dim)
    assert scales.shape == (5, 1) and scales.dtype == torch.float32
    torch.testing.assert_close(
        quantized.float() * scales, keys.float(), atol=0.125, rtol=0.07
    )
    loc = torch.tensor([193, 0, 127, 64, 255], dtype=torch.int64, device="cuda")
    cache.store_quantized(0, loc, quantized, scales)
    # Independent byte-layout oracle also catches accidental writes to padding.
    expected = torch.zeros_like(buf)
    for row, slot in enumerate(loc.tolist()):
        page, offset = divmod(slot, 64)
        expected[page, offset * head_dim : (offset + 1) * head_dim] = quantized[
            row
        ].view(torch.uint8)
        expected[
            page, 64 * head_dim + offset * 4 : 64 * head_dim + (offset + 1) * 4
        ] = scales[row].view(torch.uint8)
    assert torch.equal(buf, expected)
    pages = torch.tensor([[3, 0], [1, 3]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([67, 65], dtype=torch.int32, device="cuda")
    expected_k, expected_s = [], []
    for page_ids, length in zip(pages.tolist(), lengths.tolist()):
        expected_k.append(
            torch.cat(
                [expected[p, : 64 * head_dim].view(64, head_dim) for p in page_ids]
            )[:length]
        )
        expected_s.append(
            torch.cat([expected[p, 64 * head_dim :].view(64, 4) for p in page_ids])[
                :length
            ]
        )
    got_k, got_s = cache.get_k_and_scale(0, lengths, pages, 132, 67)
    assert torch.equal(got_k, torch.cat(expected_k))
    assert torch.equal(got_s, torch.cat(expected_s))
    for gather, oracle in [(GetK, expected_k[0]), (GetS, expected_s[0])]:
        assert torch.equal(gather.triton(pool, buf, 67, pages[0]), oracle)
        assert torch.equal(gather.torch_fast(pool, buf, 67, pages[0]), oracle)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_indexer_quantization_scales_and_reconstruction(head_dim, scale_fmt):
    torch.manual_seed(61)
    values = torch.randn(3, 2, head_dim, dtype=torch.bfloat16, device="cuda")
    values[0, 0] = 0
    values[0, 1] = 1e-6
    quantized, scales = act_quant_torch_dsa(values, head_dim, scale_fmt)
    expected_scales = []
    for row in values.cpu().reshape(-1, head_dim).double():
        scale = max(max(abs(float(x)) for x in row), 1e-4) / 448
        if scale_fmt is not None:
            scale = 2 ** math.ceil(math.log2(scale))
        expected_scales.append(scale)
    expected_scales = torch.tensor(expected_scales, device="cuda").view(3, 2, 1)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=1e-6)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert torch.isfinite(quantized.float()).all()
    assert torch.equal(quantized[0, 0].float(), torch.zeros_like(values[0, 0]).float())
    torch.testing.assert_close(
        quantized.float() * scales, values.float(), atol=1e-8, rtol=0.063
    )


def test_triton_ragged_indexer_matches_torch_oracle_at_glm_shape():
    torch.manual_seed(37)
    q = torch.randn(5, 32, 128, device="cuda").clamp(-2, 2).to(FP8_DTYPE)
    k = torch.randn(257, 128, device="cuda").clamp(-2, 2).to(FP8_DTYPE)
    scale = torch.rand(257, device="cuda") * 0.2 + 0.05
    weight = torch.rand(5, 32, device="cuda") - 0.4
    starts = torch.tensor([0, 3, 64, 128, 17], dtype=torch.int32, device="cuda")
    ends = torch.tensor([257, 220, 129, 256, 33], dtype=torch.int32, device="cuda")
    expected = fp8_ragged_mqa_logits_torch_dsa(
        q, (k, scale), weight, starts, ends, chunk_rows=2, clean_logits=False
    )
    actual = fp8_mqa_logits_triton(
        q, (k, scale), weight, starts, ends, clean_logits=True
    )
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-2, rtol=2e-3)

    expected_topk = torch.topk(expected, 16, dim=-1).indices
    actual_topk = torch.topk(actual, 16, dim=-1).indices
    assert torch.equal(actual_topk, expected_topk)


def test_triton_paged_indexer_matches_torch_oracle_at_glm_shape():
    torch.manual_seed(41)
    cache, _, _ = _make_paged_cache(8)
    q = torch.randn(2, 1, 32, 128, device="cuda").clamp(-2, 2).to(FP8_DTYPE)
    weight = torch.rand(2, 32, device="cuda") - 0.4
    lengths = torch.tensor([[129], [251]], dtype=torch.int32, device="cuda")
    pages = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int32, device="cuda")
    expected = fp8_paged_mqa_logits_torch_dsa(
        q,
        cache,
        weight,
        lengths,
        pages,
        None,
        256,
        kv_chunk_tokens=128,
        clean_logits=False,
    )
    actual = fp8_paged_mqa_logits_triton(
        q, cache, weight, lengths, pages, 256, clean_logits=True
    )
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-2, rtol=2e-3)


def _sparse_reference(query, cache, indices, scale, value_dim):
    output_device, output_dtype = query.device, query.dtype
    result = torch.zeros(query.shape[0], query.shape[1], value_dim, dtype=torch.float64)
    query, cache, indices = query.cpu().double(), cache.cpu().double(), indices.cpu()
    for row in range(query.shape[0]):
        selected = indices[row][indices[row] >= 0].tolist()
        for head in range(query.shape[1]):
            logits = torch.tensor(
                [torch.dot(query[row, head], cache[key]) * scale for key in selected]
            )
            probabilities = torch.softmax(logits, dim=0)
            for probability, key in zip(probabilities, selected):
                result[row, head] += probability * cache[key, :value_dim]
    return result.to(device=output_device, dtype=output_dtype)


def test_sparse_mla_matches_loop_and_tp2_partition():
    torch.manual_seed(29)
    query = torch.randn(4, 8, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(19, 1, 16, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor(
        [[0, 2, 7, -1], [3, 3, 9, 1], [18, 4, -1, -1], [6, -1, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    scale = 1 / math.sqrt(query.shape[-1])
    expected = _sparse_reference(query, cache[:, 0], indices, scale, 12)
    actual = sparse_mla_torch_dsa(
        query,
        cache,
        indices,
        softmax_scale=scale,
        value_dim=12,
        chunk_rows=2,
    )
    torch.testing.assert_close(actual, expected, atol=2**-6, rtol=2**-6)
    partitioned = torch.cat(
        [
            sparse_mla_torch_dsa(
                part,
                cache,
                indices,
                softmax_scale=scale,
                value_dim=12,
                chunk_rows=2,
            )
            for part in query.chunk(2, dim=1)
        ],
        dim=1,
    )
    assert torch.equal(actual, partitioned)


def test_sparse_mla_all_invalid_selection_is_zero_and_finite():
    torch.manual_seed(31)
    query = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(7, 1, 16, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([[-1, -1, -1], [1, -1, 5]], dtype=torch.int32, device="cuda")
    result = sparse_mla_torch_dsa(
        query,
        cache,
        indices,
        softmax_scale=1 / math.sqrt(query.shape[-1]),
        value_dim=12,
        chunk_rows=1,
    )
    assert torch.isfinite(result).all()
    assert torch.equal(result[0], torch.zeros_like(result[0]))


def _triton_sparse(query, cache, indices, scale):
    return triton_sparse_mla_fwd(
        q_nope=query[..., :512],
        q_rope=query[..., 512:],
        kv=cache,
        indices=indices.unsqueeze(1),
        sm_scale=scale,
        d_v=512,
    )[0]


def test_triton_sparse_mla_matches_torch_oracle_and_tp2_at_glm_shape():
    torch.manual_seed(43)
    query = torch.randn(8, 32, 576, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(257, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((8, 2048), -1, dtype=torch.int32, device="cuda")
    valid_lengths = [0, 1, 7, 16, 63, 128, 256, 2048]
    for row, length in enumerate(valid_lengths):
        if length:
            indices[row, :length] = torch.randint(
                0, cache.shape[0], (length,), dtype=torch.int32, device="cuda"
            )
    scale = 1 / math.sqrt(query.shape[-1])
    expected = sparse_mla_torch_dsa(
        query,
        cache,
        indices,
        softmax_scale=scale,
        value_dim=512,
        chunk_rows=2,
    )
    actual = _triton_sparse(query, cache, indices, scale)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual, expected, atol=2**-9, rtol=2**-7)

    partitioned = torch.cat(
        [_triton_sparse(part, cache, indices, scale) for part in query.chunk(2, dim=1)],
        dim=1,
    )
    assert torch.equal(actual, partitioned)
