"""TileLang paged MQA logits for indexer head counts TileLang's GEMM cannot split (Flash-8B: 4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

hopper_only = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9,
    reason="the TileLang paged MQA logits kernel serves SM90 (Hopper) only",
)

PAGE_SIZE = 64
HEAD_DIM = 128
SEQ_LENS = (300, 64, 129, 320)


def _inputs(num_heads: int, pages_per_seq: int = 5):
    """Keys and their fp32 scales are page-major, as indexer_k_quant_and_cache writes them."""
    batch_size = len(SEQ_LENS)
    num_pages = batch_size * pages_per_seq
    raw = torch.zeros(
        (num_pages, PAGE_SIZE * (HEAD_DIM + 4)), dtype=torch.uint8, device="cuda"
    )
    keys = (
        torch.randn(num_pages, PAGE_SIZE, HEAD_DIM, device="cuda")
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.rand(num_pages, PAGE_SIZE, device="cuda") * 0.2 + 0.05
    raw[:, : PAGE_SIZE * HEAD_DIM] = keys.reshape(num_pages, -1).view(torch.uint8)
    raw[:, PAGE_SIZE * HEAD_DIM :] = scales.contiguous().view(torch.uint8)
    kv = raw.view(num_pages, PAGE_SIZE, 1, HEAD_DIM + 4)
    q = (
        torch.randn(batch_size, 1, num_heads, HEAD_DIM, device="cuda")
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    weights = torch.randn(batch_size, num_heads, device="cuda")
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32, device="cuda")
    page_table = (
        torch.randperm(num_pages, device="cuda")
        .reshape(batch_size, pages_per_seq)
        .to(torch.int32)
    )
    return q, kv, weights, seq_lens, page_table, pages_per_seq * PAGE_SIZE


@hopper_only
@pytest.mark.parametrize("num_heads", [4, 8, 12, 32])
def test_tilelang_paged_mqa_logits_matches_triton(num_heads):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        tilelang_fp8_paged_mqa_logits,
    )
    from sglang.kernels.ops.attention.dsa.triton_mqa_logits_sm80 import (
        fp8_paged_mqa_logits_triton,
    )

    torch.manual_seed(num_heads)
    q, kv, weights, seq_lens, page_table, max_len = _inputs(num_heads)
    expected = fp8_paged_mqa_logits_triton(
        q, kv, weights, seq_lens, page_table, max_len, clean_logits=False
    )
    actual = tilelang_fp8_paged_mqa_logits(
        q, kv, weights, seq_lens, page_table, None, max_len, clean_logits=False
    )
    for row, length in enumerate(SEQ_LENS):
        torch.testing.assert_close(
            actual[row, :length], expected[row, :length], rtol=2e-3, atol=1e-4
        )


@hopper_only
def test_padded_heads_add_exactly_nothing():
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        tilelang_fp8_paged_mqa_logits,
    )

    torch.manual_seed(4)
    q, kv, weights, seq_lens, page_table, max_len = _inputs(4)
    four = tilelang_fp8_paged_mqa_logits(
        q, kv, weights, seq_lens, page_table, None, max_len, clean_logits=False
    )
    q_eight = F.pad(q.view(torch.uint8), (0, 0, 0, 4)).view(torch.float8_e4m3fn)
    eight = tilelang_fp8_paged_mqa_logits(
        q_eight.contiguous(),
        kv,
        F.pad(weights, (0, 4)).contiguous(),
        seq_lens,
        page_table,
        None,
        max_len,
        clean_logits=False,
    )
    for row, length in enumerate(SEQ_LENS):
        assert torch.equal(four[row, :length], eight[row, :length])


@pytest.mark.parametrize("num_heads", [4, 32])
def test_fixture_writes_the_page_major_cache_layout(num_heads):
    """Keys packed per token instead of per page still decode, so the Hopper comparison above needs its own check."""
    from sglang.kernels.ops.attention.dsa.triton_mqa_logits_sm80 import (
        fp8_paged_mqa_logits_triton,
    )
    from sglang.srt.layers.attention.dsa.torch_dsa_fallback import (
        fp8_paged_mqa_logits_torch_dsa,
    )

    torch.manual_seed(num_heads)
    q, kv, weights, seq_lens, page_table, max_len = _inputs(num_heads)
    triton_logits = fp8_paged_mqa_logits_triton(
        q, kv, weights, seq_lens, page_table, max_len, clean_logits=False
    )
    torch_logits = fp8_paged_mqa_logits_torch_dsa(
        q,
        kv,
        weights,
        seq_lens,
        page_table,
        None,
        max_len,
        kv_chunk_tokens=4096,
        clean_logits=False,
    )
    for row, length in enumerate(SEQ_LENS):
        torch.testing.assert_close(
            triton_logits[row, :length],
            torch_logits[row, :length],
            rtol=1e-5,
            atol=1e-5,
        )
