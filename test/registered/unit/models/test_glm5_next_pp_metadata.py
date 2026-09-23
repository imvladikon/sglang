from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_pp_nonfirst_rank_both_inputs_none_does_not_crash():
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    residual = torch.ones_like(hidden)
    proxy = PPProxyTensors({"hidden_states": hidden, "residual": residual})
    batch = SimpleNamespace(can_run_tbo=False)
    calls = []

    def layer(positions, values, forward_batch, skip, *args, **kwargs):
        calls.append((positions, forward_batch))
        return values + 1, skip + 2, None

    model = SimpleNamespace(
        start_layer=1,
        end_layer=2,
        pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
        dflash_capture=False,
        layers_to_capture=[],
        layers={1: layer},
        config=SimpleNamespace(mhc=False),
    )
    # Exercise the current language-model entry point. The removed model-local
    # CP preparation hook no longer owns this pipeline hand-off.
    with patch(
        "sglang.srt.models.glm5_next.check_cuda_graph_backend", return_value=True
    ):
        result = Glm5NextModel.forward(model, None, None, batch, pp_proxy_tensors=proxy)

    assert calls == [(None, batch)]
    torch.testing.assert_close(result["hidden_states"], hidden + 1)
    torch.testing.assert_close(result["residual"], residual + 2)


def test_pp_nonfirst_rank_drops_the_proxy_residual_under_mhc():
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    proxy = PPProxyTensors(
        {"hidden_states": hidden, "residual": torch.ones_like(hidden)}
    )
    batch = SimpleNamespace(can_run_tbo=False)
    seen = []

    def layer(positions, values, forward_batch, skip, *args, **kwargs):
        seen.append(skip)
        return values + 1, torch.zeros_like(values), None

    model = SimpleNamespace(
        start_layer=1,
        end_layer=2,
        pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
        dflash_capture=False,
        layers_to_capture=[],
        layers={1: layer},
        config=SimpleNamespace(mhc=True),
    )
    with patch(
        "sglang.srt.models.glm5_next.check_cuda_graph_backend", return_value=True
    ):
        Glm5NextModel.forward(model, None, None, batch, pp_proxy_tensors=proxy)

    # mHC carries its residual streams inside hidden_states, so reusing the
    # proxy residual would double-count them.
    assert seen == [None]
