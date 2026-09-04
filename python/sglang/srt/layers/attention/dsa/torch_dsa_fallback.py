"""Architecture-independent reference kernels for DeepSeek sparse attention.

These kernels are deliberately opt-in.  They provide a correctness path on
CUDA architectures, such as SM80, which cannot execute the native FP8 E4M3
DeepGEMM indexer or the fused sparse-MLA kernels.  Float32 is used for logits
and softmax accumulation; row chunking bounds temporary memory.
"""

from __future__ import annotations

from typing import Any

import torch
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def fp8_ragged_mqa_logits_torch_dsa(
    q_fp8: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    row_start: torch.Tensor,
    row_end: torch.Tensor,
    *,
    chunk_rows: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute DSA indexer logits for a flattened ragged KV cache.

    The formula matches ``deep_gemm.fp8_mqa_logits``:

    ``scale[k] * sum_h(weight[q,h] * relu(dot(q[q,h], k[k])))``.
    """
    k_fp8, k_scale = kv_fp8
    num_queries, num_heads, head_dim = q_fp8.shape
    num_keys, key_dim = k_fp8.shape
    assert head_dim == key_dim == 128
    assert weight.shape == (num_queries, num_heads)
    assert k_scale.shape == (num_keys,)
    assert row_start.shape == row_end.shape == (num_queries,)
    assert chunk_rows > 0
    assert clean_logits is False, "masking is completed by topk_transform"

    result = torch.empty(
        (num_queries, num_keys), dtype=torch.float32, device=q_fp8.device
    )
    key = k_fp8.float()
    positions = torch.arange(num_keys, device=q_fp8.device)
    for begin in range(0, num_queries, chunk_rows):
        end = min(begin + chunk_rows, num_queries)
        logits = torch.matmul(q_fp8[begin:end].float(), key.T)
        logits.relu_()
        logits.mul_(weight[begin:end].float().unsqueeze(-1))
        reduced = logits.sum(dim=1)
        reduced.mul_(k_scale.float().unsqueeze(0))
        valid = (positions.unsqueeze(0) >= row_start[begin:end].unsqueeze(1)) & (
            positions.unsqueeze(0) < row_end[begin:end].unsqueeze(1)
        )
        reduced.masked_fill_(~valid, -torch.inf)
        result[begin:end] = reduced
    return result


def fp8_paged_mqa_logits_torch_dsa(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    schedule_metadata: Any,
    max_seq_len: int,
    *,
    kv_chunk_tokens: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute DSA indexer logits from SGLang's 64-token packed page layout."""
    del schedule_metadata
    batch_size, next_tokens, num_heads, head_dim = q_fp8.shape
    page_size = kvcache_fp8.shape[1]
    assert next_tokens == 1
    assert head_dim == 128
    assert page_size == 64
    assert kvcache_fp8.shape[1:] == (page_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert clean_logits is False, "masking is completed by topk_transform"
    if seq_lens.ndim == 2:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert kv_chunk_tokens > 0

    max_pages = (max_seq_len + page_size - 1) // page_size
    page_bytes = page_size * (head_dim + 4)
    value_bytes = page_size * head_dim

    cache_bytes = kvcache_fp8.view(torch.uint8).reshape(-1, page_bytes)
    query = q_fp8[:, 0].float()
    result = query.new_full((batch_size, max_seq_len), -torch.inf)
    pages_per_chunk = max(1, kv_chunk_tokens // page_size)
    for page_begin in range(0, max_pages, pages_per_chunk):
        page_end = min(page_begin + pages_per_chunk, max_pages)
        page_ids = page_table[:, page_begin:page_end].clamp_min(0)
        gathered = cache_bytes[page_ids]
        num_tokens = (page_end - page_begin) * page_size
        key = (
            gathered[..., :value_bytes]
            .contiguous()
            .view(FP8_DTYPE)
            .float()
            .view(batch_size, num_tokens, head_dim)
        )
        key_scale = (
            gathered[..., value_bytes:]
            .contiguous()
            .view(torch.float32)
            .view(batch_size, num_tokens)
        )
        logits = torch.bmm(key, query.transpose(1, 2))
        logits.relu_()
        logits.mul_(weight.float().unsqueeze(1))
        reduced = logits.sum(dim=2)
        reduced.mul_(key_scale)
        token_begin = page_begin * page_size
        token_end = min(token_begin + num_tokens, max_seq_len)
        result[:, token_begin:token_end] = reduced[:, : token_end - token_begin]
    positions = torch.arange(max_seq_len, device=q_fp8.device)
    result.masked_fill_(positions.unsqueeze(0) >= seq_lens.unsqueeze(1), -torch.inf)
    return result


def sparse_mla_torch_dsa(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: float,
    value_dim: int,
    chunk_rows: int,
) -> torch.Tensor:
    """Selected-KV MLA with bounded temporaries and FP32 score accumulation."""
    num_queries, _, query_dim = query.shape
    assert indices.ndim == 2 and indices.shape[0] == num_queries
    assert chunk_rows > 0
    assert 0 < value_dim <= query_dim
    if kv_cache.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "The torch DSA fallback requires an unquantized KV cache; set "
            f"kv_cache_dtype=bfloat16 (got {kv_cache.dtype})."
        )
    cache = kv_cache.reshape(-1, kv_cache.shape[-1])
    assert cache.shape[1] == query_dim

    outputs: list[torch.Tensor] = []
    for begin in range(0, num_queries, chunk_rows):
        end = min(begin + chunk_rows, num_queries)
        selected = indices[begin:end]
        valid = selected >= 0
        row_has_valid_key = valid.any(dim=1)
        selected_kv = cache[selected.clamp_min(0).long()].float()
        logits = torch.einsum("qhd,qkd->qhk", query[begin:end].float(), selected_kv)
        logits.mul_(softmax_scale)
        logits.masked_fill_(~valid.unsqueeze(1), -torch.inf)
        # FlashMLA's reference semantics for an all-masked selection are a
        # zero output.  Give softmax one finite sentinel to avoid NaNs, then
        # remove every invalid probability below.
        if not torch.all(row_has_valid_key):
            logits[~row_has_valid_key, :, 0] = 0.0
        probabilities = torch.softmax(logits, dim=-1)
        probabilities.masked_fill_(~valid.unsqueeze(1), 0.0)
        output = torch.einsum(
            "qhk,qkd->qhd", probabilities, selected_kv[..., :value_dim]
        )
        outputs.append(output.to(query.dtype))
    return torch.cat(outputs, dim=0)
