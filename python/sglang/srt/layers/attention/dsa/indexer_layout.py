"""The index cache stores one FP32 scale for each FP8 key vector."""


def indexer_quant_block_size(head_dim: int) -> int:
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        raise ValueError("DSA index_head_dim must be an integer")
    if head_dim not in (64, 128):
        raise ValueError(f"DSA index_head_dim must be 64 or 128; got {head_dim}")
    return head_dim


def check_indexer_backend(head_dim: int, backend: str) -> None:
    indexer_quant_block_size(head_dim)
    backend = getattr(backend, "value", backend)
    if head_dim != 128 and backend != "torch":
        raise ValueError(
            f"DSA index_head_dim={head_dim} requires "
            "dsa_paged_mqa_logits_backend='torch'; optimized indexer kernels "
            "require index_head_dim=128"
        )
