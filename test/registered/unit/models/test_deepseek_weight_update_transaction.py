import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, _param, loaded_weight):
        self.loaded = loaded_weight


class _LoaderHarness(DeepseekV2WeightLoaderMixin):
    def __init__(self, params):
        self.config = SimpleNamespace(
            q_lora_rank=3,
            n_routed_experts=0,
            num_hidden_layers=1,
        )
        self.quant_config = None
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.num_fused_shared_experts = 0
        self.model = SimpleNamespace(start_layer=0, end_layer=1)
        self.fuse_qkv_a_proj = True
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(self, weights, _modules, _nextn_conf):
        return weights

    def post_load_weights(self, is_nextn=False, weight_names=None):
        pass


class TestDeepseekWeightUpdateTransaction(unittest.TestCase):
    def test_non_streamed_load_keeps_existing_fusion_behavior(self):
        fused_param = _FakeParam()
        model = _LoaderHarness(
            {"model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight": fused_param}
        )
        q_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        kv_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100

        model.do_load_weights(
            [
                ("model.layers.0.self_attn.q_a_proj.weight", q_weight),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    kv_weight,
                ),
            ]
        )

        torch.testing.assert_close(
            fused_param.loaded, torch.cat((q_weight, kv_weight), dim=0)
        )
        self.assertIsNone(model._weight_update_a_proj_cache)

    def test_fused_mla_inputs_survive_bucket_boundary(self):
        fused_param = _FakeParam()
        model = _LoaderHarness(
            {"model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight": fused_param}
        )
        q_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        expected_q = q_weight.clone()
        kv_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100

        model.begin_weight_update_transaction()
        model.do_load_weights([("model.layers.0.self_attn.q_a_proj.weight", q_weight)])
        self.assertIsNone(fused_param.loaded)
        q_weight.fill_(-1)
        model.do_load_weights(
            [
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    kv_weight,
                )
            ]
        )
        model.finalize_weight_update_transaction()

        torch.testing.assert_close(
            fused_param.loaded, torch.cat((expected_q, kv_weight), dim=0)
        )
        self.assertIsNone(model._weight_update_a_proj_cache)

    @patch(
        "sglang.srt.models.deepseek_common.deepseek_weight_loader."
        "_load_fused_indexer_wk"
    )
    def test_fused_dsa_inputs_survive_bucket_boundary(self, load_indexer):
        model = _LoaderHarness(
            {
                "model.layers.0.self_attn.indexer.wk_weights_proj.weight": (
                    torch.nn.Parameter(
                        torch.zeros(3, 4, dtype=torch.bfloat16),
                        requires_grad=False,
                    )
                )
            }
        )
        observed_pending_ids = []

        def consume(name, loaded_weight, _params, pending, _quant_config):
            observed_pending_ids.append(id(pending))
            entry = pending.setdefault("fused-indexer", {})
            entry["scale" if name.endswith("scale_inv") else "weight"] = loaded_weight
            if entry.keys() == {"weight", "scale"}:
                pending.pop("fused-indexer")
            return True

        load_indexer.side_effect = consume
        model.begin_weight_update_transaction()
        model.do_load_weights(
            [("model.layers.0.self_attn.indexer.wk.weight", torch.ones(2, 4))]
        )
        model.do_load_weights(
            [
                (
                    "model.layers.0.self_attn.indexer.wk.weight_scale_inv",
                    torch.ones(1, 1),
                )
            ]
        )
        model.finalize_weight_update_transaction()

        self.assertEqual(len(set(observed_pending_ids)), 1)
        self.assertIsNone(model._weight_update_indexer_wk_cache)

    def test_incomplete_fused_sources_are_rejected_and_cleared(self):
        model = _LoaderHarness({})
        model.begin_weight_update_transaction()
        model._weight_update_a_proj_cache["layer.q_a_proj.weight"] = torch.ones(1)
        model._weight_update_indexer_wk_cache["layer.indexer.wk"] = {
            "weight": torch.ones(1)
        }

        with self.assertRaisesRegex(
            RuntimeError, "Incomplete DeepSeek/GLM fused weight update"
        ):
            model.finalize_weight_update_transaction()

        self.assertIsNone(model._weight_update_a_proj_cache)
        self.assertIsNone(model._weight_update_indexer_wk_cache)


if __name__ == "__main__":
    unittest.main()
