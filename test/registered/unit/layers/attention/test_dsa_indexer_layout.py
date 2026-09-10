"""Reject unsupported kernel/layout combinations before allocating the cache."""

from enum import Enum

import pytest
from sglang.srt.layers.attention.dsa.indexer_layout import (
    check_indexer_backend,
    indexer_quant_block_size,
)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_indexer_has_one_scale_per_key(head_dim):
    assert head_dim // indexer_quant_block_size(head_dim) == 1
    check_indexer_backend(head_dim, "torch")


@pytest.mark.parametrize("head_dim", [True, 0, -64, 32, 96, 256, 64.0])
def test_unsupported_layout_is_rejected(head_dim):
    with pytest.raises(ValueError, match="index_head_dim"):
        indexer_quant_block_size(head_dim)


@pytest.mark.parametrize("backend", ["auto", "deepgemm", "triton", "cutedsl", "aiter"])
def test_tiny_indexer_requires_explicit_torch_backend(backend):
    with pytest.raises(ValueError, match="requires.*torch"):
        check_indexer_backend(64, backend)
    check_indexer_backend(128, backend)


def test_backend_enum_is_supported():
    class Backend(Enum):
        TORCH = "torch"

    check_indexer_backend(64, Backend.TORCH)
