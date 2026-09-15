"""TileLang paged MQA logits for indexer head counts TileLang's GEMM cannot split (Flash-8B: 4)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the TileLang paged MQA logits kernel serves SM90 (Hopper) only",
)

PAGE_SIZE = 64
HEAD_DIM = 128


def _inputs(num_heads: int, batch_size: int = 4, pages_per_seq: int = 5):
    num_pages = batch_size * pages_per_seq
    kv = torch.zeros(
        (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.uint8, device="cuda"
    )
    keys = torch.randn(num_pages, PAGE_SIZE, 1, HEAD_DIM, device="cuda").clamp(-4, 4)
    kv[..., :HEAD_DIM] = keys.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(num_pages, PAGE_SIZE, 1, 1, device="cuda") * 0.2 + 0.05
    kv[..., HEAD_DIM:] = (
        scales.contiguous().view(torch.uint8).reshape(num_pages, PAGE_SIZE, 1, 4)
    )
    q = (
        torch.randn(batch_size, 1, num_heads, HEAD_DIM, device="cuda")
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    weights = torch.randn(batch_size, num_heads, device="cuda")
    seq_lens = torch.tensor([300, 64, 129, 320], dtype=torch.int32, device="cuda")
    page_table = (
        torch.randperm(num_pages, device="cuda")
        .reshape(batch_size, pages_per_seq)
        .to(torch.int32)
    )
    return q, kv, weights, seq_lens, page_table, pages_per_seq * PAGE_SIZE


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
    for row, length in enumerate(seq_lens.tolist()):
        torch.testing.assert_close(
            actual[row, :length], expected[row, :length], rtol=2e-3, atol=1e-4
        )


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
    for row, length in enumerate(seq_lens.tolist()):
        assert torch.equal(four[row, :length], eight[row, :length])
