"""DSA reference construction must not initialize an unused DeepGEMM runtime."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer, dsa_indexer_kpool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestIndexerRuntimeInitialization(unittest.TestCase):
    def _construct(self, module, backend, *, cuda):
        with ExitStack() as stack:
            deep_gemm = MagicMock()
            deep_gemm.get_num_sms.return_value = 72
            if backend == "torch":
                deep_gemm.get_num_sms.side_effect = RuntimeError(
                    "unused DeepGEMM runtime"
                )
            stack.enter_context(
                patch.object(module, "deep_gemm", deep_gemm, create=True)
            )
            stack.enter_context(
                patch.object(
                    module,
                    "get_exec",
                    return_value=SimpleNamespace(
                        kernel=SimpleNamespace(dsa_paged_mqa_logits_backend=backend)
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "ReplicatedLinear",
                    side_effect=lambda *a, **k: torch.nn.Identity(),
                )
            )
            stack.enter_context(
                patch.object(
                    module, "LayerNorm", side_effect=lambda *a, **k: torch.nn.Identity()
                )
            )
            stack.enter_context(
                patch.object(
                    module, "get_rope_wrapper", return_value=torch.nn.Identity()
                )
            )
            stack.enter_context(
                patch.object(
                    module, "get_device", return_value=SimpleNamespace(device="cpu")
                )
            )
            current = stack.enter_context(
                patch.object(torch.cuda, "current_device", return_value=3)
            )
            props = stack.enter_context(
                patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(multi_processor_count=108),
                )
            )
            if module is dsa_indexer:
                cls = dsa_indexer.Indexer
                stack.enter_context(patch.object(module, "_is_cuda", cuda))
                stack.enter_context(
                    patch.object(module, "is_dsa_enable_prefill_cp", return_value=False)
                )
                stack.enter_context(
                    patch.object(
                        module, "get_parallel", return_value=SimpleNamespace(pp_size=1)
                    )
                )
            else:
                cls = dsa_indexer_kpool.IndexerKPool
                stack.enter_context(patch.object(module, "is_cuda", return_value=cuda))
            instance = cls(
                hidden_size=256,
                index_n_heads=4,
                index_head_dim=128,
                rope_head_dim=64,
                index_topk=64,
                q_lora_rank=128,
                max_position_embeddings=256,
                rope_theta=10000.0,
                layer_id=0,
                scale_fmt=None,
                config=SimpleNamespace(
                    index_kpool=4,
                    index_kpool_always_select_tail=True,
                    index_kpool_compress=True,
                ),
            )
            if not cuda:
                deep_gemm.get_num_sms.assert_not_called()
                current.assert_not_called()
                props.assert_not_called()
            elif backend == "torch":
                self.assertEqual(instance.sm_count, 108)
                self.assertEqual(instance.half_device_sm_count, 56)
                current.assert_called_once_with()
                props.assert_called_once_with(3)
                deep_gemm.get_num_sms.assert_not_called()
            else:
                self.assertEqual(instance.sm_count, 72)
                self.assertEqual(instance.half_device_sm_count, 40)
                deep_gemm.get_num_sms.assert_called_once_with()
                current.assert_not_called()
                props.assert_not_called()

    def test_torch_indexers_do_not_initialize_deep_gemm(self):
        for module in (dsa_indexer, dsa_indexer_kpool):
            with self.subTest(indexer=module.__name__):
                self._construct(module, "torch", cuda=True)

    def test_deep_gemm_indexers_keep_configured_sm_budget(self):
        for module in (dsa_indexer, dsa_indexer_kpool):
            with self.subTest(indexer=module.__name__):
                self._construct(module, "deepgemm", cuda=True)

    def test_cpu_indexers_do_not_query_cuda(self):
        for module in (dsa_indexer, dsa_indexer_kpool):
            with self.subTest(indexer=module.__name__):
                self._construct(module, "torch", cuda=False)


if __name__ == "__main__":
    unittest.main()
