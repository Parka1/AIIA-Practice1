import unittest

import numpy as np

from local_vm.vm_inference import detect_people, draw_detections, letterbox


class FakeNet:
    def __init__(self, prediction):
        self.prediction = prediction
        self.input = None

    def setInput(self, blob):
        self.input = blob

    def forward(self):
        return self.prediction


def yolo_output():
    output = np.zeros((1, 84, 3), dtype=np.float32)
    output[0, :4, 0] = (50, 310, 30, 50)
    output[0, 4, 0] = 0.9
    output[0, :4, 1] = (51, 310, 30, 50)
    output[0, 4, 1] = 0.8
    output[0, :4, 2] = (200, 200, 60, 80)
    output[0, 4, 2] = 0.1
    output[0, 5, 2] = 0.99
    return output


class LetterboxTests(unittest.TestCase):
    def test_preserves_aspect_ratio_and_centers_padding(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(frame, 960)

        self.assertEqual(padded.shape, (960, 960, 3))
        self.assertEqual(scale, 0.5)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 210)
        self.assertEqual(int(padded[0, 0, 0]), 114)


class DetectionTests(unittest.TestCase):
    def test_counts_persons_and_suppresses_duplicate_boxes(self):
        net = FakeNet(yolo_output())
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        detections, inference_ms = detect_people(net, frame, image_size=960)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0][:4], (70, 150, 60, 100))
        self.assertAlmostEqual(detections[0][4], 0.9)
        self.assertGreaterEqual(inference_ms, 0)
        self.assertEqual(net.input.shape, (1, 3, 960, 960))

    def test_rejects_unexpected_model_output_shape(self):
        net = FakeNet(np.zeros((1, 80, 4), dtype=np.float32))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "84 values per box"):
            detect_people(net, frame, image_size=64)

    def test_draws_detection_labels_without_mutating_the_input(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)

        annotated = draw_detections(
            frame,
            [(100, 100, 40, 80, 0.9), (135, 100, 40, 80, 0.8)],
        )

        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated, frame))
        self.assertFalse(np.any(frame))


if __name__ == "__main__":
    unittest.main()
