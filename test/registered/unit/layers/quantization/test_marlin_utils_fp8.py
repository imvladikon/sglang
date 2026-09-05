"""Tests for FP8 Marlin utilities."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.quantization import marlin_utils, marlin_utils_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFp8MarlinBias(CustomTestCase):
    def test_dense_bias_remains_in_logical_output_order(self):
        size_k = 32
        size_n = 32

        layer = torch.nn.Module()
        layer.input_size_per_partition = size_k
        layer.output_size_per_partition = size_n
        layer.orig_dtype = torch.float16
        layer.weight_block_size = None
        layer.weight = torch.nn.Parameter(
            torch.zeros((size_k, size_n), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones((size_n,), dtype=torch.float32), requires_grad=False
        )
        original_bias = torch.arange(size_n, dtype=torch.float16)
        layer.bias = torch.nn.Parameter(original_bias.clone(), requires_grad=False)

        with (
            patch.object(
                marlin_utils_fp8,
                "marlin_make_workspace",
                return_value=torch.empty(0, dtype=torch.int32),
            ),
            patch.object(
                marlin_utils_fp8,
                "gptq_marlin_repack",
                return_value=torch.empty(0, dtype=torch.int32),
                create=True,
            ),
        ):
            marlin_utils_fp8.prepare_fp8_layer_for_marlin(layer)

        torch.testing.assert_close(layer.bias, original_bias)

    def test_workspace_reuses_storage_and_rejects_incompatible_layout(self):
        device = torch.device("cpu")
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=4),
        ):
            workspace = marlin_utils.marlin_make_workspace(device)
            workspace.fill_(9)
            reused = marlin_utils.marlin_make_workspace(device, existing=workspace)

            self.assertIs(reused, workspace)
            self.assertTrue(torch.all(reused == 0))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                marlin_utils.marlin_make_workspace(
                    device, max_blocks_per_sm=4, existing=workspace
                )

    def test_dense_prepare_preserves_workspace_across_repack(self):
        size_k = 128
        size_n = 64
        layer = torch.nn.Module()
        layer.input_size_per_partition = size_k
        layer.output_size_per_partition = size_n
        layer.orig_dtype = torch.float16
        layer.weight_block_size = [128, 128]

        def load_checkpoint_format():
            layer.weight = torch.nn.Parameter(
                torch.zeros((size_k, size_n), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale_inv = torch.nn.Parameter(
                torch.ones((1, 1), dtype=torch.float32), requires_grad=False
            )

        def fake_repack(**kwargs):
            return torch.zeros(
                kwargs["size_k"] // 16,
                kwargs["size_n"] * 2,
                dtype=torch.int32,
            )

        with (
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=4),
            ),
            patch.object(
                marlin_utils_fp8,
                "gptq_marlin_repack",
                side_effect=fake_repack,
                create=True,
            ),
        ):
            load_checkpoint_format()
            marlin_utils_fp8.prepare_fp8_layer_for_marlin(layer)
            workspace = layer.workspace
            workspace_ptr = workspace.data_ptr()

            load_checkpoint_format()
            marlin_utils_fp8.prepare_fp8_layer_for_marlin(layer)

        self.assertIs(layer.workspace, workspace)
        self.assertEqual(layer.workspace.data_ptr(), workspace_ptr)

    def test_moe_prepare_uses_local_expert_count_under_ep(self):
        hidden_size = 128
        intermediate_size = 128
        local_experts = 2
        layer = torch.nn.Module()
        layer.num_experts = 8
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size
        layer.orig_dtype = torch.bfloat16
        layer.weight_block_size = [128, 128]
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(
                local_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(
                local_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            torch.ones(local_experts, 2, 1), requires_grad=False
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            torch.ones(local_experts, 1, 1), requires_grad=False
        )

        def fake_repack(**kwargs):
            return torch.zeros(
                kwargs["size_k"] // 16,
                kwargs["size_n"] * 2,
                dtype=torch.int32,
            )

        with (
            patch.object(
                marlin_utils_fp8,
                "marlin_make_workspace",
                return_value=torch.empty(0, dtype=torch.int32),
            ),
            patch.object(
                marlin_utils_fp8,
                "gptq_marlin_repack",
                side_effect=fake_repack,
                create=True,
            ) as repack,
        ):
            marlin_utils_fp8.prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)

        self.assertEqual(repack.call_count, local_experts * 2)
        self.assertEqual(layer.w13_weight.shape[0], local_experts)
        self.assertEqual(layer.w2_weight.shape[0], local_experts)


if __name__ == "__main__":
    unittest.main(verbosity=3)
