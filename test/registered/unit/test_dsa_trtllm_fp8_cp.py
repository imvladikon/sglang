"""CP-v2 FP8 cache writes must gather only rank-local, RoPE-transformed keys."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSATRTLLMFP8CP(unittest.TestCase):
    def test_fp8_cache_gathers_only_after_fused_rope(self):
        for fused_rope, cp_enabled, save_cache in (
            (False, True, True),
            (True, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with self.subTest(rope=fused_rope, cp=cp_enabled, save=save_cache):
                backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
                    dsa_backend.DeepseekSparseAttnBackend
                )
                backend.forward_metadata = SimpleNamespace()
                backend.kv_cache_dtype = torch.float8_e4m3fn
                backend.qk_rope_head_dim = 2
                backend.kv_lora_rank = 4
                backend.token_to_kv_pool = MagicMock()
                # Stop immediately after the cache write, before GPU-only decode.
                backend.token_to_kv_pool.get_key_buffer.side_effect = RuntimeError(
                    "cache write observed"
                )
                q, k, k_rope = (torch.randn(1, 1, 4) for _ in range(3))
                gathered_k, gathered_rope = (
                    torch.randn_like(k),
                    torch.randn_like(k_rope),
                )
                strategy = MagicMock()
                strategy.all_gather_dsa_trtllm_fp8_kv.return_value = (
                    gathered_k,
                    gathered_rope,
                )
                batch = SimpleNamespace(
                    positions=torch.tensor([0]), out_cache_loc="loc"
                )
                layer = SimpleNamespace(is_cross_attention=False, layer_id=0)
                with (
                    patch.object(
                        dsa_backend, "dsa_use_prefill_cp", return_value=cp_enabled
                    ),
                    patch.object(dsa_backend, "get_cp_strategy", return_value=strategy),
                    patch.object(
                        dsa_backend,
                        "mla_quantize_for_fp8_no_rope",
                        return_value=(q, k, k_rope),
                    ),
                    patch.object(
                        dsa_backend,
                        "mla_quantize_and_rope_for_fp8",
                        return_value=(q, k, k_rope),
                    ),
                    self.assertRaisesRegex(RuntimeError, "cache write observed"),
                ):
                    backend._forward_trtllm(
                        q,
                        k,
                        k,
                        layer,
                        batch,
                        torch.tensor([1]),
                        save_kv_cache=save_cache,
                        q_rope=k_rope,
                        k_rope=k_rope,
                        cos_sin_cache=torch.empty(0) if fused_rope else None,
                    )
                gather = strategy.all_gather_dsa_trtllm_fp8_kv
                cache_write = backend.token_to_kv_pool.set_mla_kv_buffer
                if fused_rope and cp_enabled and save_cache:
                    gather.assert_called_once_with(batch, k, k_rope)
                    cache_write.assert_called_once_with(
                        layer, "loc", gathered_k, gathered_rope
                    )
                else:
                    gather.assert_not_called()
                    if save_cache:
                        cache_write.assert_called_once_with(layer, "loc", k, k_rope)
                    else:
                        cache_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
