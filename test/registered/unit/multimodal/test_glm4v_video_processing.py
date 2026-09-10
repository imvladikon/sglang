import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.glm4v import (  # noqa: E402
    _glm_effective_presize_budget,
    glm_processor_video_config,
    glm_sample_and_decode_sync,
    preprocess_video_frames_sync,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeVideoReader:
    avg_fps = 4.0

    def __len__(self):
        return 8

    def get_frames_at(self, indices):
        return np.stack(
            [np.full((4, 6, 3), index, dtype=np.uint8) for index in indices]
        )


class TestGlmVideoProcessing(CustomTestCase):
    def test_processor_budget_survives_public_video_options(self):
        processor = SimpleNamespace(
            fps=2.0,
            max_image_tokens=1024,
            min_image_tokens=4,
            patch_size=14,
            merge_size=2,
            patch_expand_factor=4,
            temporal_patch_size=2,
            resize_mode="resize",
        )

        config = glm_processor_video_config(processor)

        self.assertEqual(config["fps"], 2.0)
        self.assertEqual(config["_presize_budget"]["pixels_per_token"], 1568)
        effective = _glm_effective_presize_budget(config, 256)
        self.assertEqual(effective["_presize_budget"]["max_pixels"], 256 * 1568)
        self.assertEqual(config["_presize_budget"]["max_pixels"], 1024 * 1568)

    def test_model_sampler_drives_decoding_and_metadata(self):
        video_processor = SimpleNamespace(sample_frames=Mock(return_value=[1, 3, 7]))

        frames, metadata = glm_sample_and_decode_sync(
            _FakeVideoReader(), {"fps": 1.5}, video_processor
        )

        np.testing.assert_array_equal(frames[:, 0, 0, 0], [1, 3, 7])
        self.assertEqual(metadata["frames_indices"], [1, 3, 7])
        self.assertEqual(metadata["total_num_frames"], 8)
        self.assertEqual(metadata["fps"], 4.0)
        video_processor.sample_frames.assert_called_once()
        self.assertEqual(video_processor.sample_frames.call_args.kwargs["fps"], 1.5)

    def test_frame_list_keeps_timing_metadata(self):
        frame_list = [
            {
                "frame_image": np.zeros((3, 4, 3), dtype=np.uint8),
                "timestamp": 0.0,
                "detail": json.dumps({"video_duration": 2.0}),
            },
            {
                "frame_image": np.ones((3, 4, 3), dtype=np.uint8),
                "timestamp": 1.0,
            },
        ]

        frames, metadata = preprocess_video_frames_sync(frame_list)

        self.assertEqual(len(frames), 2)
        self.assertEqual(metadata["duration"], 2.0)
        self.assertEqual(metadata["fps"], 1.0)
        self.assertEqual(metadata["frames_indices"], [0, 1])


if __name__ == "__main__":
    import unittest

    unittest.main()
