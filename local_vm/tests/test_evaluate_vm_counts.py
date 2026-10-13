import unittest

from local_vm.evaluate_vm_counts import evaluate_rows, evaluate_rows_by_camera


class EvaluateCountTests(unittest.TestCase):
    def test_reports_count_accuracy_and_error_metrics(self):
        metrics = evaluate_rows(
            [
                {"frame_id": "frame-1", "cam_id": "camera-a", "ground_truth_count": "2", "yolo_count": "2"},
                {"frame_id": "frame-2", "cam_id": "camera-a", "ground_truth_count": "1", "yolo_count": "0"},
                {"frame_id": "frame-3", "cam_id": "camera-b", "ground_truth_count": "0", "yolo_count": "1"},
            ]
        )

        self.assertEqual(metrics["frames"], 3)
        self.assertAlmostEqual(metrics["exact_count_accuracy_percent"], 100 / 3)
        self.assertAlmostEqual(metrics["mean_absolute_count_error"], 2 / 3)
        self.assertAlmostEqual(metrics["root_mean_square_count_error"], (2 / 3) ** 0.5)
        self.assertEqual(metrics["mean_count_bias"], 0)
        self.assertEqual(metrics["under_count_frames"], 1)
        self.assertEqual(metrics["over_count_frames"], 1)

    def test_requires_ground_truth_labels_and_unique_frame_ids(self):
        with self.assertRaisesRegex(ValueError, "fill in ground_truth_count"):
            evaluate_rows(
                [
                    {
                        "frame_id": "frame-1",
                        "cam_id": "camera-a",
                        "ground_truth_count": "",
                        "yolo_count": "0",
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "duplicate frame_id"):
            evaluate_rows(
                [
                    {"frame_id": "frame-1", "cam_id": "camera-a", "ground_truth_count": "0", "yolo_count": "0"},
                    {"frame_id": "frame-1", "cam_id": "camera-a", "ground_truth_count": "1", "yolo_count": "0"},
                ]
            )

    def test_groups_metrics_by_camera(self):
        rows = [
            {"frame_id": "sp-1", "cam_id": "santa_pola", "ground_truth_count": "4", "yolo_count": "0"},
            {"frame_id": "sp-2", "cam_id": "santa_pola", "ground_truth_count": "6", "yolo_count": "2"},
            {"frame_id": "gr-1", "cam_id": "grazalema", "ground_truth_count": "2", "yolo_count": "2"},
            {"frame_id": "gr-2", "cam_id": "grazalema", "ground_truth_count": "4", "yolo_count": "2"},
        ]

        metrics = evaluate_rows_by_camera(rows)

        self.assertEqual(metrics["santa_pola"]["frames"], 2)
        self.assertEqual(metrics["santa_pola"]["exact_count_accuracy_percent"], 0)
        self.assertEqual(metrics["santa_pola"]["mean_absolute_count_error"], 4)
        self.assertEqual(metrics["grazalema"]["frames"], 2)
        self.assertEqual(metrics["grazalema"]["exact_count_accuracy_percent"], 50)
        self.assertEqual(metrics["grazalema"]["mean_absolute_count_error"], 1)


if __name__ == "__main__":
    unittest.main()
