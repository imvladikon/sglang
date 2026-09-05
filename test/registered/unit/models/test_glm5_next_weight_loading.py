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


class TestGlm5NextWeightLoading(unittest.TestCase):
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
        model._weight_update_a_proj_cache["model.layers.3.self_attn.q_a_proj.weight"] = (
            torch.ones(1)
        )
        with self.assertRaisesRegex(RuntimeError, "Incomplete GLM fused MLA"):
            Glm5NextForConditionalGeneration.finalize_weight_update_transaction(model)
        self.assertIsNone(model._weight_update_a_proj_cache)


if __name__ == "__main__":
    unittest.main()
