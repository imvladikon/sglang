"""Exercise empty DP-attention ranks through the real Triton LoRA backend."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestIdleLoRABatchInfo(CustomTestCase):
    def test_idle_decode_idle_rebuilds_dense_and_moe_metadata(self):
        for is_moe in (False, True):
            with self.subTest(is_moe=is_moe):
                backend = TritonLoRABackend(
                    max_loras_per_batch=2, device=torch.device("cuda")
                )
                backend.is_moe_lora = is_moe
                for mode, indices in (
                    (ForwardMode.IDLE, []),
                    (ForwardMode.DECODE, [0, 1]),
                    (ForwardMode.IDLE, []),
                ):
                    backend.prepare_lora_batch(
                        SimpleNamespace(forward_mode=mode, batch_size=len(indices)),
                        weight_indices=indices,
                        lora_ranks=[4, 4],
                        scalings=[1.0, 1.0],
                        use_cuda_graph=False,
                    )
                    info = backend.batch_info
                    self.assertEqual(info.bs, len(indices))
                    self.assertEqual(info.num_segments, len(indices))
                    self.assertEqual(info.max_len, 1 if indices else 0)
                    self.assertEqual(info.seg_lens.tolist(), [1] * len(indices))
                    self.assertEqual(
                        info.seg_indptr.tolist(), list(range(len(indices) + 1))
                    )
                    self.assertEqual(info.weight_indices.tolist(), indices)
                    self.assertEqual(info.seg_lens.dtype, torch.int32)
                    self.assertEqual(info.seg_lens.device.type, "cuda")
                    if is_moe:
                        self.assertEqual(
                            info.moe_lora_info.token_lora_mapping.tolist(), indices
                        )
                        self.assertEqual(
                            info.moe_lora_info.adapter_enabled.tolist(),
                            [1, 1] if indices else [0, 0],
                        )


if __name__ == "__main__":
    unittest.main()
