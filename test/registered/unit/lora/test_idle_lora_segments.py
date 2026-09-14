# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Idle DP-attention ranks have no requests or LoRA token segments."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.lora.utils import generate_sequence_lengths, get_batch_token_counts
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIdleLoRASegments(unittest.TestCase):
    def test_idle_counts_without_extend_or_speculative_metadata(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE, batch_size=0)
        self.assertEqual(get_batch_token_counts(batch), (0, 0))

    def test_idle_segments_are_empty_int32_on_requested_device(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE, batch_size=0)
        segments = generate_sequence_lengths(batch, device=torch.device("cpu"))
        self.assertEqual(segments.shape, (0,))
        self.assertEqual(segments.dtype, torch.int32)
        self.assertEqual(segments.device, torch.device("cpu"))
        indptr = torch.cat((torch.zeros(1, dtype=torch.int32), segments.cumsum(0)))
        self.assertEqual(indptr.tolist(), [0])

    def test_active_modes_keep_request_segments_and_total_tokens(self):
        cases = (
            (ForwardMode.DECODE, {}, [1, 1]),
            (
                ForwardMode.TARGET_VERIFY,
                {"spec_info": SimpleNamespace(draft_token_num=4)},
                [4, 4],
            ),
            (
                ForwardMode.EXTEND,
                {
                    "extend_num_tokens": 5,
                    "extend_seq_lens_cpu": [2, 3],
                    "extend_seq_lens": torch.tensor([2, 3], dtype=torch.int32),
                },
                [2, 3],
            ),
        )
        for mode, metadata, expected in cases:
            with self.subTest(mode=mode):
                batch = SimpleNamespace(forward_mode=mode, batch_size=2, **metadata)
                segments = generate_sequence_lengths(batch, device=torch.device("cpu"))
                self.assertEqual(segments.tolist(), expected)
                self.assertEqual(segments.dtype, torch.int32)
                self.assertEqual(
                    get_batch_token_counts(batch), (sum(expected), max(expected))
                )


if __name__ == "__main__":
    unittest.main()
