"""The portable indexer must not enter an SM90-only metadata kernel."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)


def _backend(name):
    backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
    backend.paged_mqa_logits_backend = DSAPagedMQALogitsBackend(name)
    return backend


@pytest.mark.parametrize("name", ["torch", "triton"])
def test_portable_indexer_does_not_call_deepgemm(name, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("DeepGEMM must not be used"))
    monkeypatch.setattr(
        dsa_backend,
        "deep_gemm",
        SimpleNamespace(get_paged_mqa_logits_metadata=forbidden, get_num_sms=forbidden),
        raising=False,
    )
    backend = _backend(name)
    lengths = torch.tensor([[21], [64]], dtype=torch.int32)
    assert backend._build_paged_mqa_schedule_metadata(lengths) is None
    metadata = SimpleNamespace(paged_mqa_schedule_metadata=None)
    backend._refresh_paged_mqa_schedule_metadata(metadata, lengths)
    assert metadata.paged_mqa_schedule_metadata is None
    forbidden.assert_not_called()


@pytest.mark.parametrize("name", ["deepgemm", "cutedsl"])
def test_native_schedule_is_built_and_refreshed_in_place(name, monkeypatch):
    build = Mock(return_value=torch.tensor([2, 3], dtype=torch.int32))
    monkeypatch.setattr(
        dsa_backend,
        "deep_gemm",
        SimpleNamespace(get_paged_mqa_logits_metadata=build, get_num_sms=lambda: 108),
        raising=False,
    )
    backend = _backend(name)
    lengths = torch.tensor([[21], [64]], dtype=torch.int32)
    metadata = SimpleNamespace(paged_mqa_schedule_metadata=None)
    backend._refresh_paged_mqa_schedule_metadata(metadata, lengths)
    build.assert_called_once_with(lengths, 64, 108)
    pointer = metadata.paged_mqa_schedule_metadata.data_ptr()
    build.return_value = torch.tensor([4, 5], dtype=torch.int32)
    backend._refresh_paged_mqa_schedule_metadata(metadata, lengths)
    assert metadata.paged_mqa_schedule_metadata.data_ptr() == pointer
    assert metadata.paged_mqa_schedule_metadata.tolist() == [4, 5]
