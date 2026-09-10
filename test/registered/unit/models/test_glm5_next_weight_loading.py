import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class _RecordingParam:
    def __init__(self):
        self.calls = []

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.calls.append((loaded_weight.clone(), args, kwargs))


def _checkpoint_model(parameters, num_experts=2):
    return SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=num_experts, num_hidden_layers=24),
        num_fused_shared_experts=0,
        quant_config=None,
        named_parameters=lambda: iter(parameters.items()),
    )


class TestGlm5NextWeightLoading(unittest.TestCase):
    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_transformers_hyperconnection_and_forget_gate_names(self, post_load):
        aliases = [
            (f"{source}.{suffix}", f"{target}_{suffix}")
            for source, target in [("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")]
            for suffix in ["base", "scale", "fn"]
        ] + [
            (f"self_attn.forget_gate.{name}", f"self_attn.{name}")
            for name in ["A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"]
        ]
        for source, target in aliases:
            with self.subTest(source=source):
                param = _RecordingParam()
                model = _checkpoint_model({f"model.layers.0.{target}": param})
                value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                Glm5NextForConditionalGeneration.load_weights(
                    model, [(f"model.language_model.layers.0.{source}", value)]
                )
                self.assertEqual(len(param.calls), 1)
                torch.testing.assert_close(param.calls[0][0], value, rtol=0, atol=0)

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_transformers_combined_convolution_preserves_qkv_order(self, post_load):
        param = _RecordingParam()
        model = _checkpoint_model({"model.layers.0.self_attn.qkv_conv1d.weight": param})
        value = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4)
        Glm5NextForConditionalGeneration.load_weights(
            model,
            [("model.language_model.layers.0.self_attn.conv1d.weight", value)],
        )
        self.assertEqual(len(param.calls), 3)
        for index, (loaded, args, kwargs) in enumerate(param.calls):
            self.assertEqual(args, (index,))
            self.assertEqual(kwargs, {})
            torch.testing.assert_close(loaded, value[index * 2 : (index + 1) * 2])

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_transformers_packed_experts_preserve_expert_and_gate_up_order(
        self, post_load
    ):
        gate_up, down = _RecordingParam(), _RecordingParam()
        model = _checkpoint_model(
            {
                "model.layers.1.mlp.experts.w13_weight": gate_up,
                "model.layers.1.mlp.experts.w2_weight": down,
            }
        )
        packed_gate_up = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
        packed_down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 100
        Glm5NextForConditionalGeneration.load_weights(
            model,
            [
                (
                    "model.language_model.layers.1.mlp.experts.gate_up_proj",
                    packed_gate_up,
                ),
                ("model.language_model.layers.1.mlp.experts.down_proj", packed_down),
            ],
        )
        self.assertEqual(len(gate_up.calls), 4)
        self.assertEqual(len(down.calls), 2)
        for expert in range(2):
            for offset, shard in [(0, "w1"), (1, "w3")]:
                loaded, _, kwargs = gate_up.calls[2 * expert + offset]
                self.assertEqual(kwargs, {"shard_id": shard, "expert_id": expert})
                torch.testing.assert_close(
                    loaded, packed_gate_up[expert, offset * 3 : (offset + 1) * 3]
                )
            loaded, _, kwargs = down.calls[expert]
            self.assertEqual(kwargs, {"shard_id": "w2", "expert_id": expert})
            torch.testing.assert_close(loaded, packed_down[expert])

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_malformed_transformers_packed_weights_are_rejected(self, post_load):
        cases = [
            ("self_attn.conv1d.weight", (5, 1, 4)),
            ("self_attn.conv1d.weight", (6, 2, 4)),
            ("mlp.experts.gate_up_proj", (2, 5, 4)),
            ("mlp.experts.gate_up_proj", (3, 6, 4)),
            ("mlp.experts.down_proj", (2, 4)),
        ]
        for suffix, shape in cases:
            with (
                self.subTest(suffix=suffix, shape=shape),
                self.assertRaisesRegex(ValueError, "GLM.*checkpoint"),
            ):
                Glm5NextForConditionalGeneration.load_weights(
                    _checkpoint_model({}),
                    [
                        (
                            f"model.language_model.layers.0.{suffix}",
                            torch.zeros(shape),
                        )
                    ],
                )

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_quark_block_fp8_weight_scale_loads_scale_inv(self, post_load):
        scale_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=0,
                num_hidden_layers=45,
                num_nextn_predict_layers=1,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            named_parameters=lambda: iter(
                [("model.layers.0.mlp.down_proj.weight_scale_inv", scale_param)]
            ),
        )
        loaded_scale = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        Glm5NextForConditionalGeneration.load_weights(
            model,
            [
                (
                    "model.language_model.layers.0.mlp.down_proj.weight_scale",
                    loaded_scale,
                )
            ],
        )

        self.assertIs(scale_param.loaded, loaded_scale)
        post_load.assert_called_once()

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_fused_mla_inputs_survive_streamed_bucket_boundary(self, post_load):
        fused_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=0,
                num_hidden_layers=10,
                num_nextn_predict_layers=1,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            fuse_qkv_a_proj=True,
            _weight_update_a_proj_cache=None,
            named_parameters=lambda: iter(
                [
                    (
                        "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa.weight",
                        fused_param,
                    )
                ]
            ),
        )
        q_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        kv_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100

        Glm5NextForConditionalGeneration.begin_weight_update_transaction(model)
        Glm5NextForConditionalGeneration.load_weights(
            model,
            [("model.language_model.layers.3.self_attn.q_a_proj.weight", q_weight)],
        )
        self.assertIsNone(fused_param.loaded)
        Glm5NextForConditionalGeneration.load_weights(
            model,
            [
                (
                    "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
                    kv_weight,
                )
            ],
        )
        Glm5NextForConditionalGeneration.finalize_weight_update_transaction(model)

        torch.testing.assert_close(
            fused_param.loaded, torch.cat((q_weight, kv_weight), dim=0)
        )
        self.assertIsNone(model._weight_update_a_proj_cache)
        self.assertEqual(post_load.call_count, 2)

    def test_incomplete_fused_mla_transaction_is_rejected_and_cleared(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=0,
                num_hidden_layers=10,
                num_nextn_predict_layers=1,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            fuse_qkv_a_proj=True,
            _weight_update_a_proj_cache=None,
            named_parameters=lambda: iter([]),
        )

        Glm5NextForConditionalGeneration.begin_weight_update_transaction(model)
        model._weight_update_a_proj_cache[
            "model.layers.3.self_attn.q_a_proj.weight"
        ] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "Incomplete GLM fused MLA"):
            Glm5NextForConditionalGeneration.finalize_weight_update_transaction(model)
        self.assertIsNone(model._weight_update_a_proj_cache)


if __name__ == "__main__":
    unittest.main()
