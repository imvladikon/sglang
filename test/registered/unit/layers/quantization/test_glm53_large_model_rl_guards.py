"""CPU regression tests for GLM-5.3 Flash large-model RL hot paths."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

# Import the quantization package first to follow SGLang's normal import order
# and avoid the flashinfer_trtllm <-> fp8 registration cycle.
import sglang.srt.layers.quantization.fp8  # noqa: F401
import torch
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    align_fp8_moe_weights_for_flashinfer_trtllm,
    get_activation_type,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
from sglang.srt.utils import torch_memory_saver_adapter as tms_adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Activation:
    Swiglu = SimpleNamespace(value=1)
    Geglu = SimpleNamespace(value=2)
    Silu = SimpleNamespace(value=3)
    Gelu = SimpleNamespace(value=4)
    Relu2 = SimpleNamespace(value=5)


class TestGlm53LargeModelRlGuards(CustomTestCase):
    def test_glm53_keeps_deepseek_dsa_family_defaults_after_override_split(self):
        from sglang.srt.arg_groups.model_override_base import _MODEL_OVERRIDE_FNS
        from sglang.srt.arg_groups.model_overrides.deepseek_v2 import (
            _deepseek_family_overrides,
        )

        architecture = "Glm5NextForConditionalGeneration"
        self.assertIn(_deepseek_family_overrides, _MODEL_OVERRIDE_FNS[architecture])

        args = SimpleNamespace(
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            enable_prefill_cp=False,
        )
        platform = SimpleNamespace(is_npu=False, is_xpu=False, is_hip=False)
        with (
            patch(
                "sglang.srt.configs.model_config.is_deepseek_dsa",
                return_value=True,
            ),
            patch(
                "sglang.srt.arg_groups.model_overrides.deepseek_v2.get_platform",
                return_value=platform,
            ),
        ):
            self.assertEqual(
                _deepseek_family_overrides(args, SimpleNamespace()),
                {"attention_backend": "dsa", "page_size": 64},
            )

    def test_gated_silu_does_not_touch_nonexistent_situ_enum(self):
        flashinfer = ModuleType("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        core = ModuleType("flashinfer.fused_moe.core")
        core.ActivationType = _Activation
        with patch.dict(
            sys.modules,
            {
                "flashinfer": flashinfer,
                "flashinfer.fused_moe": fused_moe,
                "flashinfer.fused_moe.core": core,
            },
        ):
            self.assertEqual(get_activation_type("silu", is_gated=True), 1)

    def test_plain_trtllm_moe_restores_canonical_bf16_shape(self):
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.use_flashinfer_trtllm_moe = True
        layer = SimpleNamespace(
            num_local_experts=2,
            intermediate_size_per_partition=4,
            hidden_size=3,
            moe_runner_config=SimpleNamespace(is_gated=True),
        )
        param = torch.nn.Parameter(torch.zeros(2, 1, 6, 4), requires_grad=False)

        method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer, param, "model.layers.0.mlp.experts.w13_weight"
        )

        self.assertEqual(tuple(param.shape), (2, 8, 3))

    def test_trtllm_fp8_alignment_preserves_expert_weight_loaders(self):
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=True),
            w13_weight=torch.nn.Parameter(
                torch.zeros(2, 32, 16, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros(2, 16, 16, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            w13_input_scale=torch.nn.Parameter(torch.ones(2), requires_grad=False),
            w2_input_scale=torch.nn.Parameter(torch.ones(2), requires_grad=False),
            w13_weight_scale=torch.nn.Parameter(torch.ones(2), requires_grad=False),
            w2_weight_scale=torch.nn.Parameter(torch.ones(2), requires_grad=False),
        )
        loader = MagicMock()
        layer.w13_weight.weight_loader = loader
        layer.w2_weight.weight_loader = loader
        original_w13 = layer.w13_weight
        original_w2 = layer.w2_weight

        flashinfer = ModuleType("flashinfer")
        flashinfer.shuffle_matrix_a = lambda weight, _tile: weight
        flashinfer.reorder_rows_for_gated_act_gemm = lambda weight: weight
        with patch.dict(sys.modules, {"flashinfer": flashinfer}):
            align_fp8_moe_weights_for_flashinfer_trtllm(layer)

        self.assertIs(layer.w13_weight, original_w13)
        self.assertIs(layer.w2_weight, original_w2)
        self.assertIs(layer.w13_weight.weight_loader, loader)
        self.assertIs(layer.w2_weight.weight_loader, loader)

    def test_idle_empty_cache_is_suppressed_while_engine_paused(self):
        sleeper = IdleSleeper.__new__(IdleSleeper)
        sleeper.poller = MagicMock()
        sleeper.last_empty_time = 0.0
        sleeper.empty_cache_interval = 1.0
        sleeper.can_empty_cache = lambda: False

        with (
            patch(
                "sglang.srt.managers.scheduler_components.idle_sleeper.real_time",
                return_value=2.0,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.idle_sleeper.current_platform.empty_cache"
            ) as empty_cache,
        ):
            sleeper.maybe_sleep()

        empty_cache.assert_not_called()

    def test_model_runner_boundary_restores_tms_preload_mode(self):
        memory_saver = SimpleNamespace(hook_mode="torch")
        with patch.object(tms_adapter, "_memory_saver", memory_saver):
            tms_adapter._TorchMemorySaverAdapterReal().configure_current_process()
        self.assertEqual(memory_saver.hook_mode, "preload")


if __name__ == "__main__":
    unittest.main()
