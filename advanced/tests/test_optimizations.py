import unittest

import numpy as np

from advanced.detection_metrics import score_frame, summarize
from common.optimizations import (
    LightingController, MotionDetector, encode_jpeg, enhance_clahe, enhance_gamma, resize_to_width,
)


def frame(value=100, size=(480, 640)):
    return np.full((*size, 3), value, np.uint8)


class ResizeAndJpegTests(unittest.TestCase):
    def test_resize_keeps_aspect_ratio_and_never_upscales(self):
        self.assertEqual(resize_to_width(frame(size=(1080, 1920)), 960).shape[:2], (540, 960))
        self.assertEqual(resize_to_width(frame(), 1280).shape[:2], (480, 640))

    def test_lower_quality_gives_smaller_jpeg(self):
        noisy = np.random.default_rng(0).integers(0, 255, (240, 320, 3), np.uint8)
        self.assertLess(len(encode_jpeg(noisy, 30)), len(encode_jpeg(noisy, 85)))
        with self.assertRaises(ValueError):
            encode_jpeg(noisy, 0)


class MotionDetectorTests(unittest.TestCase):
    def test_sends_first_frame_skips_static_and_sends_changes(self):
        detector = MotionDetector(min_changed=0.01, max_skipped=100)
        static = frame()
        moved = static.copy()
        moved[100:200, 100:200] = 250  # a bright object appears

        self.assertTrue(detector.should_send(static))
        self.assertFalse(detector.should_send(static.copy()))
        self.assertTrue(detector.should_send(moved))
        self.assertFalse(detector.should_send(moved.copy()))

    def test_forces_a_frame_after_max_skipped(self):
        detector = MotionDetector(max_skipped=2)
        sent = [detector.should_send(frame()) for _ in range(5)]
        self.assertEqual(sent, [True, False, False, True, False])

    def test_changes_outside_the_region_are_ignored(self):
        detector = MotionDetector(min_changed=0.01, region=(0.0, 0.5, 1.0, 1.0), max_skipped=100)
        top_changed = frame()
        top_changed[0:200] = 250  # e.g. the sea at the top of the frame
        detector.should_send(frame())
        self.assertFalse(detector.should_send(top_changed))


class EnhancementTests(unittest.TestCase):
    def test_gamma_below_one_brightens_dark_tones_and_keeps_extremes(self):
        dark = frame(40)
        self.assertGreater(enhance_gamma(dark, 0.6).mean(), dark.mean())
        self.assertEqual(enhance_gamma(frame(0), 0.6).max(), 0)
        self.assertEqual(enhance_gamma(frame(255), 0.6).min(), 255)

    def test_clahe_keeps_shape_and_raises_local_contrast(self):
        # Dark, low-contrast texture, like a person in the shade.
        texture = np.random.default_rng(0).choice(np.array([50, 60], np.uint8), (480, 640))
        image = np.dstack([texture] * 3)
        out = enhance_clahe(image)
        self.assertEqual(out.shape, image.shape)
        self.assertGreater(out.std(), image.std())


class LightingControllerTests(unittest.TestCase):
    def test_turns_on_at_once_and_off_after_hold_empty_frames(self):
        controller = LightingController(hold_frames=3)
        states = [controller.update(n) for n in (0, 1, 0, 0, 2, 0, 0, 0, 0)]
        self.assertEqual(states, [False, True, True, True, True, True, True, False, False])

    def test_hold_of_one_frame_is_the_raw_decision(self):
        controller = LightingController(hold_frames=1)
        self.assertEqual([controller.update(n) for n in (1, 0, 1)], [True, False, True])
        with self.assertRaises(ValueError):
            LightingController(hold_frames=0)


class DetectionMetricsTests(unittest.TestCase):
    def test_greedy_matching_counts_each_person_once(self):
        truth = [(0, 0, 10, 20), (100, 100, 110, 120)]
        detections = [(1, 1, 10, 20), (0, 0, 11, 21), (300, 300, 310, 320)]
        score = score_frame(detections, truth)
        self.assertEqual((score.true_positives, score.detections, score.people), (1, 3, 2))
        metrics = summarize([score])
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["count_mae"], 1)


if __name__ == "__main__":
    unittest.main()
