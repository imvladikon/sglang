"""CPU tests for checkpoint-format reloads of FP8 Marlin layers."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.marlin_reload import (
    abort_marlin_reload,
    begin_marlin_reload,
    finalize_marlin_reload,
    finalize_marlin_reload_metadata,
    has_marlin_reload_metadata,
    record_marlin_reload_metadata,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fp8(values, shape):
    return (
        torch.tensor(values, dtype=torch.float32).reshape(shape).to(torch.float8_e4m3fn)
    )


def _stacked_loader(param, loaded_weight, shard_id):
    offset = 0 if shard_id == 0 else loaded_weight.shape[0]
    param.data[offset : offset + loaded_weight.shape[0]].copy_(loaded_weight)


def _expert_loader(param, loaded_weight, weight_name, shard_id, expert_id):
    if param.ndim == 3:
        offset = 0 if shard_id == 0 else loaded_weight.shape[0]
        param.data[expert_id, offset : offset + loaded_weight.shape[0]].copy_(
            loaded_weight
        )
    else:
        offset = 0 if shard_id == 0 else loaded_weight.shape[0]
        param.data[expert_id, offset : offset + loaded_weight.shape[0]].copy_(
            loaded_weight
        )


def _pack_fp8(weight):
    """Stand-in for Marlin's shape- and dtype-changing CUDA repack."""
    raw = weight.detach().contiguous().view(torch.uint8)
    return raw.view(torch.int32).reshape(
        *weight.shape[:-2], weight.shape[-2] // 4, weight.shape[-1]
    )


def _replace_parameter(layer, old_name, new_name, value):
    if hasattr(layer, old_name):
        delattr(layer, old_name)
    layer.register_parameter(
        new_name, torch.nn.Parameter(value.contiguous(), requires_grad=False)
    )


def _workspace(layer, size):
    existing = getattr(layer, "workspace", None)
    if existing is None:
        layer.workspace = torch.zeros(size, dtype=torch.int32)
    else:
        if existing.dtype != torch.int32 or existing.numel() != size:
            raise ValueError("incompatible synthetic Marlin workspace")
        existing.zero_()


class _SyntheticMarlinMethod(QuantizeMethodBase):
    use_marlin = True

    def apply(self, layer, *args, **kwargs):  # pragma: no cover - reload-only test
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        if hasattr(layer, "w13_weight"):
            _workspace(layer, 8)
            _replace_parameter(
                layer, "w13_weight", "w13_weight", _pack_fp8(layer.w13_weight)
            )
            _replace_parameter(
                layer, "w2_weight", "w2_weight", _pack_fp8(layer.w2_weight)
            )
            _replace_parameter(
                layer,
                "w13_weight_scale_inv",
                "w13_weight_scale",
                layer.w13_weight_scale_inv.detach().clone(),
            )
            _replace_parameter(
                layer,
                "w2_weight_scale_inv",
                "w2_weight_scale",
                layer.w2_weight_scale_inv.detach().clone(),
            )
            return

        _workspace(layer, 4)
        _replace_parameter(layer, "weight", "weight", _pack_fp8(layer.weight))
        _replace_parameter(
            layer,
            "weight_scale_inv",
            "weight_scale",
            layer.weight_scale_inv.detach().clone(),
        )


class _SyntheticNativeFp8Method(_SyntheticMarlinMethod):
    use_marlin = False

    def process_weights_after_loading(self, layer):
        return


class _SyntheticDerivedScaleMethod(_SyntheticMarlinMethod):
    """Mimic a native backend with a post-processed scale/kernel layout."""

    use_marlin = False


class _DenseGateUp(torch.nn.Module):
    def __init__(self, quant_method=None):
        super().__init__()
        self.quant_method = quant_method or _SyntheticMarlinMethod()
        self.weight = torch.nn.Parameter(
            torch.zeros((8, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.weight.weight_loader = _stacked_loader
        self.weight_scale_inv = torch.nn.Parameter(
            torch.zeros((2, 1), dtype=torch.float32), requires_grad=False
        )
        self.weight_scale_inv.weight_loader = _stacked_loader


class _MoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method = _SyntheticMarlinMethod()
        self.w13_weight = torch.nn.Parameter(
            torch.zeros((2, 8, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros((2, 4, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.zeros((2, 2, 1), dtype=torch.float32), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.zeros((2, 1, 1), dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters(recurse=False):
            param.weight_loader = _expert_loader


class _SyntheticGlmBlock(torch.nn.Module):
    """Small GLM-shaped dense + MoE checkpoint/model-format contract."""

    def __init__(self, native_dense=False):
        super().__init__()
        dense_method = _SyntheticNativeFp8Method() if native_dense else None
        self.dense = _DenseGateUp(dense_method)
        self.moe = _MoeExperts()

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith("dense."):
                source = "gate_proj" if ".gate_proj." in name else "up_proj"
                target = (
                    "dense.weight_scale_inv"
                    if name.endswith("scale_inv")
                    else "dense.weight"
                )
                shard_id = 0 if source == "gate_proj" else 1
                param = params[target]
                param.weight_loader(param, loaded_weight, shard_id)
                continue

            parts = name.split(".")
            expert_id = int(parts[2])
            source = parts[3]
            is_scale = parts[4] == "weight_scale_inv"
            if source in ("gate_proj", "up_proj"):
                target = "moe.w13_weight_scale_inv" if is_scale else "moe.w13_weight"
                shard_id = 0 if source == "gate_proj" else 1
            else:
                target = "moe.w2_weight_scale_inv" if is_scale else "moe.w2_weight"
                shard_id = 0
            param = params[target]
            param.weight_loader(
                param,
                loaded_weight,
                name,
                shard_id=shard_id,
                expert_id=expert_id,
            )


class _SyntheticGlmPostLoadBlock(_SyntheticGlmBlock):
    """Mimic GLM MLA deriving runtime tensors inside model.load_weights."""

    def __init__(self):
        super().__init__()
        self.derived_dense_weight = None

    def load_weights(self, weights):
        weights = list(weights)
        super().load_weights(weights)
        if any(name.startswith("dense.") for name, _ in weights):
            self.derived_dense_weight = self.dense.weight.detach().clone()


def _checkpoint(seed):
    base = seed * 16
    weights = [
        ("dense.gate_proj.weight", _fp8(range(base + 1, base + 17), (4, 4))),
        ("dense.gate_proj.weight_scale_inv", torch.tensor([[seed + 0.25]])),
        ("dense.up_proj.weight", _fp8(range(base + 17, base + 33), (4, 4))),
        ("dense.up_proj.weight_scale_inv", torch.tensor([[seed + 0.5]])),
    ]
    for expert_id in range(2):
        expert_base = base + 32 + expert_id * 48
        for offset, source in enumerate(("gate_proj", "up_proj", "down_proj")):
            values = range(
                expert_base + offset * 16 + 1, expert_base + (offset + 1) * 16 + 1
            )
            weights.append(
                (f"moe.experts.{expert_id}.{source}.weight", _fp8(values, (4, 4)))
            )
            weights.append(
                (
                    f"moe.experts.{expert_id}.{source}.weight_scale_inv",
                    torch.tensor([[seed + expert_id + offset / 4 + 1.0]]),
                )
            )
    return weights


def _cold_kernel(checkpoint):
    model = _SyntheticGlmBlock()
    model.load_weights(checkpoint)
    model.dense.quant_method.process_weights_after_loading(model.dense)
    model.moe.quant_method.process_weights_after_loading(model.moe)
    return model


class TestMarlinReload(CustomTestCase):
    def test_model_post_load_observes_model_format_before_marlin_repack(self):
        model = _SyntheticGlmPostLoadBlock()
        initial = _checkpoint(0)
        model.load_weights(initial)
        record_marlin_reload_metadata(model, torch.device("cpu"))
        model.dense.quant_method.process_weights_after_loading(model.dense)
        model.moe.quant_method.process_weights_after_loading(model.moe)
        updated = _checkpoint(1)
        expected = _cold_kernel(updated)
        runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: runner,
        )

        success, _ = updater.update_weights_from_tensor(updated, finalize=True)

        self.assertTrue(success)
        self.assertEqual(model.derived_dense_weight.shape, (8, 4))
        expected_model_weight = torch.cat((updated[0][1], updated[2][1]), dim=0)
        torch.testing.assert_close(model.derived_dense_weight, expected_model_weight)
        torch.testing.assert_close(model.dense.weight, expected.dense.weight)

    def test_weight_updater_keeps_transaction_across_buckets(self):
        model = _SyntheticGlmBlock()
        model.load_weights(_checkpoint(0))
        record_marlin_reload_metadata(model, torch.device("cpu"))
        model.dense.quant_method.process_weights_after_loading(model.dense)
        model.moe.quant_method.process_weights_after_loading(model.moe)
        updated = _checkpoint(1)
        expected = _cold_kernel(updated)
        runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: runner,
        )

        for start in range(0, len(updated), 3):
            success, _ = updater.update_weights_from_tensor(
                updated[start : start + 3], finalize=False
            )
            self.assertTrue(success)
        success, _ = updater.update_weights_from_tensor([], finalize=True)

        self.assertTrue(success)
        torch.testing.assert_close(model.dense.weight, expected.dense.weight)
        torch.testing.assert_close(model.moe.w13_weight, expected.moe.w13_weight)

    def test_non_marlin_transformed_layout_is_detected_after_postprocess(self):
        model = _SyntheticGlmBlock()
        model.dense.quant_method = _SyntheticDerivedScaleMethod()
        model.moe.quant_method = _SyntheticNativeFp8Method()
        model.load_weights(_checkpoint(0))
        record_marlin_reload_metadata(model, torch.device("cpu"))
        self.assertFalse(has_marlin_reload_metadata(model))
        model.dense.quant_method.process_weights_after_loading(model.dense)
        model.moe.quant_method.process_weights_after_loading(model.moe)
        finalize_marlin_reload_metadata(model)

        self.assertTrue(has_marlin_reload_metadata(model))
        self.assertTrue(begin_marlin_reload(model))
        model.load_weights(_checkpoint(1))
        finalize_marlin_reload(model)

    def test_sglang_parameter_properties_and_scalar_fill_are_supported(self):
        layer = torch.nn.Module()
        layer.quant_method = _SyntheticMarlinMethod()
        layer.weight = ModelWeightParameter(
            data=torch.zeros((8, 4), dtype=torch.float8_e4m3fn),
            input_dim=1,
            output_dim=0,
            weight_loader=default_weight_loader,
        )
        layer.weight_scale_inv = BlockQuantScaleParameter(
            data=torch.ones((1, 1), dtype=torch.float32),
            input_dim=1,
            output_dim=0,
            weight_loader=default_weight_loader,
        )
        first = _fp8(range(1, 33), (8, 4))
        layer.weight.weight_loader(layer.weight, first)
        record_marlin_reload_metadata(layer, torch.device("cpu"))
        layer.quant_method.process_weights_after_loading(layer)
        packed_weight = layer.weight

        second = _fp8(range(33, 65), (8, 4))
        self.assertTrue(begin_marlin_reload(layer))
        layer.weight.weight_loader(layer.weight, second)
        # default_weight_loader uses fill_ rather than copy_ for one element.
        layer.weight_scale_inv.weight_loader(
            layer.weight_scale_inv, torch.tensor([[0.25]])
        )
        finalize_marlin_reload(layer)

        self.assertIs(layer.weight, packed_weight)
        torch.testing.assert_close(layer.weight, _pack_fp8(second))

    def test_glm_dense_and_moe_bucketed_reload_matches_cold_kernel(self):
        model = _SyntheticGlmBlock()
        model.load_weights(_checkpoint(0))
        record_marlin_reload_metadata(model, torch.device("cpu"))
        model.dense.quant_method.process_weights_after_loading(model.dense)
        model.moe.quant_method.process_weights_after_loading(model.moe)

        dense_weight = model.dense.weight
        dense_scale = model.dense.weight_scale
        dense_workspace_ptr = model.dense.workspace.data_ptr()
        moe_w13 = model.moe.w13_weight
        moe_w2 = model.moe.w2_weight
        moe_workspace_ptr = model.moe.workspace.data_ptr()

        updated = _checkpoint(1)
        expected = _cold_kernel(updated)
        self.assertTrue(begin_marlin_reload(model))
        for start in range(0, len(updated), 3):
            model.load_weights(updated[start : start + 3])
        finalize_marlin_reload(model)

        self.assertIs(model.dense.weight, dense_weight)
        self.assertIs(model.dense.weight_scale, dense_scale)
        self.assertIs(model.moe.w13_weight, moe_w13)
        self.assertIs(model.moe.w2_weight, moe_w2)
        self.assertEqual(model.dense.workspace.data_ptr(), dense_workspace_ptr)
        self.assertEqual(model.moe.workspace.data_ptr(), moe_workspace_ptr)
        torch.testing.assert_close(model.dense.weight, expected.dense.weight)
        torch.testing.assert_close(
            model.dense.weight_scale, expected.dense.weight_scale
        )
        torch.testing.assert_close(model.moe.w13_weight, expected.moe.w13_weight)
        torch.testing.assert_close(model.moe.w2_weight, expected.moe.w2_weight)
        torch.testing.assert_close(
            model.moe.w13_weight_scale, expected.moe.w13_weight_scale
        )
        torch.testing.assert_close(
            model.moe.w2_weight_scale, expected.moe.w2_weight_scale
        )

        # A second update verifies that restore metadata was not mutated by the first.
        updated_again = _checkpoint(2)
        expected_again = _cold_kernel(updated_again)
        self.assertTrue(begin_marlin_reload(model))
        model.load_weights(updated_again)
        finalize_marlin_reload(model)
        self.assertIs(model.dense.weight, dense_weight)
        self.assertIs(model.moe.w13_weight, moe_w13)
        torch.testing.assert_close(model.dense.weight, expected_again.dense.weight)
        torch.testing.assert_close(model.moe.w13_weight, expected_again.moe.w13_weight)

    def test_incomplete_layer_is_rejected_and_kernel_storage_is_restored(self):
        model = _SyntheticGlmBlock()
        model.load_weights(_checkpoint(0))
        record_marlin_reload_metadata(model, torch.device("cpu"))
        model.dense.quant_method.process_weights_after_loading(model.dense)
        model.moe.quant_method.process_weights_after_loading(model.moe)
        original_weight = model.dense.weight
        original_value = model.dense.weight.detach().clone()

        self.assertTrue(begin_marlin_reload(model))
        model.load_weights(_checkpoint(1)[:1])
        with self.assertRaisesRegex(RuntimeError, "Incomplete FP8 Marlin"):
            finalize_marlin_reload(model)

        self.assertIs(model.dense.weight, original_weight)
        torch.testing.assert_close(model.dense.weight, original_value)
        self.assertFalse(model._sglang_marlin_reload_active)

    def test_abort_restores_untouched_kernel(self):
        model = _cold_kernel(_checkpoint(0))
        # A cold post-process alone does not opt a model into reloads.
        self.assertFalse(has_marlin_reload_metadata(model))
        abort_marlin_reload(model)

    def test_native_fp8_layer_is_not_enrolled_in_marlin_restore(self):
        model = _SyntheticGlmBlock(native_dense=True)
        record_marlin_reload_metadata(model.dense, torch.device("cpu"))
        self.assertFalse(has_marlin_reload_metadata(model.dense))
        self.assertFalse(begin_marlin_reload(model.dense))


if __name__ == "__main__":
    unittest.main(verbosity=3)
