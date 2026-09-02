import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.events = []

    def begin_weight_update_transaction(self):
        self.events.append("begin")

    def load_weights(self, weights):
        self.events.append(("load", [name for name, _ in weights]))

    def finalize_weight_update_transaction(self):
        self.events.append("finalize")

    def abort_weight_update_transaction(self):
        self.events.append("abort")


class _FailingHookModel(_HookModel):
    def load_weights(self, weights):
        super().load_weights(weights)
        raise RuntimeError("synthetic load failure")


class _FinalizeFailingHookModel(_HookModel):
    def finalize_weight_update_transaction(self):
        self.events.append("finalize")
        raise RuntimeError("synthetic finalize failure")


class TestWeightUpdateTransaction(unittest.TestCase):
    def make_updater(self, model):
        runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
        return WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=SimpleNamespace(),
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: runner,
        )

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    def test_transaction_spans_buckets_and_finalizes_once(self, _unsupported):
        model = _HookModel()
        updater = self.make_updater(model)

        updater.update_weights_from_tensor([("first", torch.ones(1))], finalize=False)
        updater.update_weights_from_tensor([("second", torch.ones(1))], finalize=False)
        updater.update_weights_from_tensor([], finalize=True)

        self.assertEqual(
            model.events,
            [
                "begin",
                ("load", ["first"]),
                ("load", ["second"]),
                ("load", []),
                "finalize",
            ],
        )
        self.assertFalse(model._sglang_weight_update_transaction_active)

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    def test_failed_bucket_aborts_model_transaction(self, _unsupported):
        model = _FailingHookModel()
        updater = self.make_updater(model)

        with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
            updater.update_weights_from_tensor(
                [("broken", torch.ones(1))], finalize=False
            )

        self.assertEqual(model.events, ["begin", ("load", ["broken"]), "abort"])
        self.assertFalse(model._sglang_weight_update_transaction_active)

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    def test_failed_finalize_aborts_model_transaction(self, _unsupported):
        model = _FinalizeFailingHookModel()
        updater = self.make_updater(model)

        with self.assertRaisesRegex(RuntimeError, "synthetic finalize failure"):
            updater.update_weights_from_tensor([("last", torch.ones(1))], finalize=True)

        self.assertEqual(
            model.events,
            ["begin", ("load", ["last"]), "finalize", "abort"],
        )
        self.assertFalse(model._sglang_weight_update_transaction_active)


if __name__ == "__main__":
    unittest.main()
