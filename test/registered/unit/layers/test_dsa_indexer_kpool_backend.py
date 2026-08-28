import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
    _topk_transform_torch,
)
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    _topk_from_pooled_history_logits_unfused,
    _torch_hadamard_index_head,
    _torch_softmax_rotate_quantize,
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKPoolMqaBackend(CustomTestCase):
    @staticmethod
    def _fallback_backend():
        return SimpleNamespace(
            dsa_index_kpool=4,
            dsa_index_topk=8,
            dsa_topk_backend=DSATopKBackend.TORCH,
            dsa_prefill_impl="torch",
            dsa_decode_impl="torch",
        )

    @staticmethod
    def _forward_batch(seq_len):
        return SimpleNamespace(
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: True,
                is_decode_or_idle=lambda: False,
            ),
        )

    def test_short_torch_fallback_trims_only_empty_kpool_tail(self):
        backend = self._fallback_backend()
        indices = torch.tensor(
            [[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1]], dtype=torch.int32
        )

        result = DeepseekSparseAttnBackend._trim_empty_kpool_tail_for_torch_fallback(
            backend,
            indices,
            "flashinfer_sparse_mla",
            self._forward_batch(seq_len=5),
        )

        torch.testing.assert_close(result, indices[:, :8])
        DeepseekSparseAttnBackend._check_kpool_tail_backend(
            backend, result, "flashinfer_sparse_mla", "prefill"
        )

    def test_torch_reference_skips_deep_gemm_schedule_at_long_context(self):
        backend = self._fallback_backend()
        self.assertTrue(
            DeepseekSparseAttnBackend._skip_short_torch_indexer_schedule(
                backend, self._forward_batch(seq_len=128)
            )
        )

    def test_long_torch_fallback_keeps_kpool_backend_guard(self):
        backend = self._fallback_backend()
        indices = torch.arange(11, dtype=torch.int32).unsqueeze(0)

        result = DeepseekSparseAttnBackend._trim_empty_kpool_tail_for_torch_fallback(
            backend,
            indices,
            "flashinfer_sparse_mla",
            self._forward_batch(seq_len=9),
        )

        self.assertIs(result, indices)
        with self.assertRaisesRegex(NotImplementedError, "index_kpool > 1"):
            DeepseekSparseAttnBackend._check_kpool_tail_backend(
                backend, result, "flashinfer_sparse_mla", "prefill"
            )

    def test_torch_sparse_mla_matches_direct_attention(self):
        q = torch.tensor(
            [[[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]], dtype=torch.bfloat16
        )
        kv = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0]],
            dtype=torch.bfloat16,
        )
        indices = torch.tensor([[2, 0, -1]], dtype=torch.int32)

        result = DeepseekSparseAttnBackend._forward_torch_sparse_mla(
            q_all=q,
            kv_cache=kv,
            page_table_1=indices,
            sm_scale=0.5,
            v_head_dim=2,
        )

        selected = kv[[2, 0]].float()
        scores = torch.einsum("hd,kd->hk", q[0].float(), selected) * 0.5
        expected = torch.einsum(
            "hk,kd->hd", torch.softmax(scores, dim=-1), selected[:, :2]
        ).unsqueeze(0)
        torch.testing.assert_close(result.float(), expected, atol=5e-3, rtol=5e-3)

    def test_torch_sparse_mla_rejects_empty_index_rows(self):
        with self.assertRaisesRegex(ValueError, "no valid KV indices"):
            DeepseekSparseAttnBackend._forward_torch_sparse_mla(
                q_all=torch.zeros(1, 1, 3),
                kv_cache=torch.zeros(2, 3),
                page_table_1=torch.full((1, 2), -1, dtype=torch.int32),
                sm_scale=1.0,
                v_head_dim=2,
            )

    def test_torch_paged_topk_returns_physical_slots(self):
        metadata = SimpleNamespace(
            page_table_1=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
            )
        )
        result = _topk_transform_torch(
            logits=torch.tensor(
                [[0.1, 0.9, 0.8, 50.0], [0.7, 0.6, 50.0, 40.0]],
                dtype=torch.float32,
            ),
            lengths=torch.tensor([3, 2], dtype=torch.int32),
            topk=3,
            topk_transform_method=TopkTransformMethod.PAGED,
            attn_metadata=metadata,
            cu_seqlens_q_topk=None,
            topk_indices_offset=None,
            row_starts=None,
            batch_idx_list=None,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[11, 12, 10], [20, 21, -1]], dtype=torch.int32),
        )

    def test_torch_ragged_topk_applies_row_relative_offsets(self):
        result = _topk_transform_torch(
            logits=torch.tensor([[50.0, 0.2, 0.8, 40.0]], dtype=torch.float32),
            lengths=torch.tensor([2], dtype=torch.int32),
            topk=3,
            topk_transform_method=TopkTransformMethod.RAGGED,
            attn_metadata=SimpleNamespace(),
            cu_seqlens_q_topk=None,
            topk_indices_offset=torch.tensor([100], dtype=torch.int32),
            row_starts=torch.tensor([1], dtype=torch.int32),
            batch_idx_list=None,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[101, 100, -1]], dtype=torch.int32),
        )

    def test_compact_hadamard_is_orthonormal(self):
        identity = torch.eye(64, dtype=torch.float32)
        rotated = _torch_hadamard_index_head(identity)

        torch.testing.assert_close(rotated @ rotated.T, identity)

    def test_reference_kpool_quantization_supports_compact_head(self):
        slot_k = torch.arange(2 * 4 * 64, dtype=torch.float32).reshape(2, 4, 64)
        slot_score = (
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=torch.float32
            )
            .unsqueeze(-1)
            .expand_as(slot_k)
        )
        ape = torch.zeros(4, 64, dtype=torch.float32)

        quantized, scale = _torch_softmax_rotate_quantize(
            slot_k=slot_k,
            slot_score=slot_score,
            ape=ape,
            round_scale=True,
        )

        self.assertEqual(quantized.shape, (2, 64))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.all(scale > 0))
        torch.testing.assert_close(torch.log2(scale), torch.log2(scale).round())

    def test_compact_torch_reference_selects_pools_and_visible_tail(self):
        indexer = SimpleNamespace(
            index_kpool=4,
            index_topk=8,
            head_dim=2,
            softmax_scale=2**-0.5,
            index_kpool_compress_ape=torch.zeros(4, 2),
        )
        keys = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ]
        )
        result = dsa_indexer_kpool.IndexerKPool._torch_reference_topk_single(
            indexer,
            query=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            keys=keys,
            gate_scores=torch.zeros_like(keys),
            head_weights=torch.tensor([1.0, -1.0]),
        )

        # Pool 0 wins; pool 1 remains because the budget holds both complete
        # pools, and token 8 is appended as the visible incomplete tail.
        self.assertEqual(set(result[:8].tolist()), set(range(8)))
        self.assertEqual(result[8].item(), 8)
        self.assertTrue(torch.all(result[9:] == -1))

    def test_cuda_tilelang_selector_reads_heads_from_unexpanded_query(self):
        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            self.assertFalse(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(
                    torch.empty(1, 32, 128)
                )
            )
            self.assertTrue(
                dsa_indexer_kpool.IndexerKPool._should_use_tilelang_paged_mqa_logits(
                    torch.empty(1, 16, 128)
                )
            )

    def test_rocm_uses_aiter_mqa_logits(self):
        marker = object()
        aiter_impl = MagicMock(return_value=marker)
        module = ModuleType("aiter.ops.triton.fp8_mqa_logits")
        module.fp8_mqa_logits = aiter_impl

        args = tuple(object() for _ in range(6))
        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=True),
            patch.dict(
                sys.modules,
                {"aiter.ops.triton.fp8_mqa_logits": module},
            ),
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                *args, clean_logits=False
            )

        self.assertIs(result, marker)
        aiter_impl.assert_called_once_with(*args, clean_logits=False)

    def test_cuda_keeps_deep_gemm_mqa_logits(self):
        marker = object()
        deep_gemm = MagicMock()
        deep_gemm.fp8_mqa_logits.return_value = marker
        q_fp8, k_fp8, k_scale, weights, starts, ends = (object() for _ in range(6))

        with (
            patch.object(dsa_indexer_kpool, "is_hip", return_value=False),
            patch.object(dsa_indexer_kpool, "deep_gemm", deep_gemm, create=True),
        ):
            result = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=True,
            )

        self.assertIs(result, marker)
        deep_gemm.fp8_mqa_logits.assert_called_once_with(
            q_fp8,
            (k_fp8, k_scale),
            weights,
            starts,
            ends,
            clean_logits=True,
        )

    def test_portable_topk_masks_invalid_groups_and_expands(self):
        logits = torch.tensor([[0.1, 0.9, 0.8, 50.0]], dtype=torch.float32)
        result = _topk_from_pooled_history_logits_unfused(
            logits=logits,
            group_lengths=torch.tensor([3], dtype=torch.int32),
            pool_size=2,
            topk=4,
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
        )

    def test_rocm_uses_fused_kpool_topk_for_supported_group_count(self):
        logits = MagicMock()
        logits.ndim = 2
        logits.shape = (1, 512)
        logits.is_cuda = True
        logits.dtype = torch.float32
        group_lengths = torch.tensor([256], dtype=torch.int32)
        marker = MagicMock()
        marker.shape = (1, 2048)

        with (
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.moe.kpool_topk_transform.fast_kpool_topk_transform_fused",
                return_value=marker,
            ) as fused,
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index._topk_from_pooled_history_logits_unfused"
            ) as unfused,
        ):
            result = topk_from_pooled_history_logits(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=4,
                topk=2048,
            )

        self.assertIs(result, marker)
        fused.assert_called_once()
        unfused.assert_not_called()

    def test_rocm_keeps_2048_group_topk_on_unfused_path(self):
        logits = MagicMock()
        logits.ndim = 2
        logits.shape = (1, 2048)
        logits.is_cuda = True
        logits.dtype = torch.float32
        group_lengths = torch.tensor([2048], dtype=torch.int32)
        marker = object()

        with (
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index.is_hip",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index._topk_from_pooled_history_logits_unfused",
                return_value=marker,
            ) as unfused,
        ):
            result = topk_from_pooled_history_logits(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=4,
                topk=8192,
            )

        self.assertIs(result, marker)
        unfused.assert_called_once()

    def test_cuda_keeps_supported_group_count_on_fused_path(self):
        logits = MagicMock()
        logits.ndim = 2
        logits.shape = (1, 512)
        logits.is_cuda = True
        logits.dtype = torch.float32
        group_lengths = torch.tensor([256], dtype=torch.int32)
        marker = MagicMock()
        marker.shape = (1, 2048)

        with (
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index.is_hip",
                return_value=False,
            ),
            patch(
                "sglang.kernels.ops.moe.kpool_topk_transform.fast_kpool_topk_transform_fused",
                return_value=marker,
            ) as fused,
        ):
            result = topk_from_pooled_history_logits(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=4,
                topk=2048,
            )

        self.assertIs(result, marker)
        fused.assert_called_once()


if __name__ == "__main__":
    unittest.main()
