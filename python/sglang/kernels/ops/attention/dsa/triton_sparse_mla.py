"""Triton sparse-MLA forward for selected GLM-5/DeepSeek KV rows.

A per-query online-softmax kernel over the indexer-selected top-k KV.  The
same math is used for the existing FP8 ROCm prefill path and for the explicit
BF16 CUDA SM80 fallback.  The latter avoids Hopper-only barriers in the
TileLang v2 kernel while retaining BF16 tensor-core compute and FP32 softmax
state.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0


def _prune_configs(configs, named_args, **kwargs):
    """Drop configs whose KV tile exceeds topk (pure waste)."""
    topk = named_args["topk"]
    keep = [c for c in configs if c.kwargs["BLOCK_N"] <= topk]
    return keep or [configs[0]]


# The best (BLOCK_N, num_warps, num_stages) is shape- and arch-sensitive, so
# autotune over a grid keyed on the attention shape. Benchmarked once per key
# (a one-time stall on the first prefill of each new shape), then cached.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns)
    for bn in (32, 64, 128)
    for w in (1, 2, 4)
    for ns in (1, 2)
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["topk", "H", "DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sparse_mla_fwd_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    probability_scale,
    topk,
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    h = tl.arange(0, H)
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    # q is read as two separate tensors (q_nope width D_V, q_rope width D_TAIL):
    # the upstream concat into a single [.., DIM] tensor is skipped since this
    # kernel splits q into main/tail anyway.
    q_main = tl.load(q_nope_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :]).to(
        q_nope_ptr.dtype.element_ty
    )  # [H, D_V]
    q_tail = tl.load(
        q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :]
    ).to(q_nope_ptr.dtype.element_ty)  # [H, D_TAIL]

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0)
        kbase = kv_ptr + page[:, None] * DIM
        kv_main = tl.load(kbase + dv[None, :], mask=valid[:, None], other=0.0).to(
            q_nope_ptr.dtype.element_ty
        )  # [BLOCK_N, D_V] -- reused as V
        kv_tail = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(q_nope_ptr.dtype.element_ty)  # [BLOCK_N, D_TAIL]

        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # Guard an all-masked row (m_new == -inf): shift by 0 instead so that
        # exp(-inf - 0) = 0 rather than exp(-inf + inf) = NaN. Identical to
        # m_new whenever the row has >=1 valid key.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])  # [H, BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_compute = (p * probability_scale).to(q_nope_ptr.dtype.element_ty)
        pv = tl.dot(p_compute, kv_main).to(tl.float32) * (1.0 / probability_scale)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """q_nope: [seq, H, d_v], q_rope: [seq, H, dim-d_v],
    kv: [num_pages, 1, dim], indices: [seq, 1, topk].

    Q and KV must have the same BF16 or FP8 dtype. Reads Q from the two
    un-concatenated tensors directly and returns ``[1, seq, H, d_v]`` BF16 to
    match ``tilelang_sparse_fwd``.
    """
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    if q_nope.dtype != q_rope.dtype or q_nope.dtype != kv.dtype:
        raise ValueError(
            "Triton sparse MLA requires matching Q/KV dtypes, got "
            f"q_nope={q_nope.dtype}, q_rope={q_rope.dtype}, kv={kv.dtype}."
        )
    if q_nope.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise ValueError(f"Unsupported Triton sparse-MLA dtype: {q_nope.dtype}")
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    # BLOCK_N / num_warps / num_stages are chosen by @triton.autotune.
    probability_scale = _FP8_MAX if q_nope.dtype != torch.bfloat16 else 1.0
    _sparse_mla_fwd_kernel[(seq,)](
        q_nope,
        q_rope,
        kv,
        indices,
        out,
        sm_scale,
        probability_scale,
        topk,
        H=H,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
    )
    return out.unsqueeze(0)
