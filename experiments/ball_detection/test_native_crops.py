import unittest

import numpy as np
import torch

from neural_candidates import StreamingDetector


class CaptureModel:
    def __call__(self, frames):
        self.input = frames
        heatmap = torch.zeros((1, 1, 288, 512))
        heatmap[0, 0, 100:103, 200:203] = .9
        return heatmap, None, None


class NativeCropTests(unittest.TestCase):
    def detector(self):
        detector = object.__new__(StreamingDetector)
        detector.model = CaptureModel()
        detector.zoom_background = np.zeros((576, 1024, 3), dtype=np.float32)
        detector.candidate_floor = .15
        detector.calls = detector.crop_calls = 0
        detector.inference_seconds = 0
        return detector

    def test_original_quarter_crop_keeps_full_model_input_and_maps_back(self):
        for width, height in [(1280, 720), (3840, 2160)]:
            detector = self.detector()
            image = np.full((height, width, 3), 128, dtype=np.uint8)
            candidates = detector.detect_native_crop([image]*4, [.5, .5])
            self.assertEqual(tuple(detector.model.input.shape), (1, 15, 288, 512))
            self.assertEqual(len(candidates), 1)
            self.assertAlmostEqual(candidates[0]['x'], .25 + .5*201/512)
            self.assertAlmostEqual(candidates[0]['y'], .25 + .5*101/288)
            self.assertAlmostEqual(float(detector.model.input[0, 3, 100, 100]), 128/255, places=6)

    def test_edge_crop_is_clamped_inside_source_image(self):
        detector = self.detector()
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        candidates = detector.detect_native_crop([image]*4, [0, 0])
        self.assertAlmostEqual(candidates[0]['x'], .5*201/512)
        self.assertAlmostEqual(candidates[0]['y'], .5*101/288)
        self.assertEqual(detector.crop_calls, 1)


if __name__ == '__main__':
    unittest.main()
