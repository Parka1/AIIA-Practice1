import csv
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from common.capture import CAMERAS, LiveFrameReader
from common.results import make_frame_id, now_local
from local_vm.send_to_vm import run_camera, upload_one_frame


class SendToVmTests(unittest.TestCase):
    def test_supported_multi_camera_sources_are_registered(self):
        self.assertIn("santa_pola", CAMERAS)
        self.assertIn("grazalema", CAMERAS)
        self.assertIn("benidorm_poniente", CAMERAS)
        self.assertIn("benidorm_levante", CAMERAS)
        self.assertIn("madrid_alcala", CAMERAS)
        self.assertIn("benalmadena", CAMERAS)
        self.assertIn("la_orotava", CAMERAS)
        self.assertIn("villafranca_del_bierzo", CAMERAS)
        self.assertIn("calafell", CAMERAS)
        self.assertIn("alicante_almadraba", CAMERAS)
        self.assertIn("barcelona_port", CAMERAS)
        self.assertIn("sevilla_plaza", CAMERAS)
        self.assertIn("formentera_es_pujols", CAMERAS)
        self.assertIn("cristosende_preview", CAMERAS)
        self.assertIn("alhambra_preview", CAMERAS)

    def test_live_frame_reader_keeps_the_latest_stream_frame(self):
        class FakeCapture:
            def __init__(self, *_args):
                pass

            def isOpened(self):
                return True

            def read(self):
                return True, np.full((2, 3, 3), 7, dtype=np.uint8)

            def release(self):
                pass

        with patch("common.capture.cv2.VideoCapture", FakeCapture):
            reader = LiveFrameReader("https://example.test/live.m3u8")
            reader.start()
            try:
                frame = reader.latest_frame(timeout=1)
            finally:
                reader.close()

        self.assertEqual(frame.shape, (2, 3, 3))
        self.assertTrue(np.all(frame == 7))

    def test_upload_records_camera_results_in_common_csv(self):
        captured_at = now_local()
        frame_id = make_frame_id("benidorm_poniente", captured_at)
        response = Mock(
            status_code=201,
            json=lambda: {
                "result": {
                    "timestamp": captured_at.isoformat(timespec="milliseconds"),
                    "architecture": "vm",
                    "cam_id": "benidorm_poniente",
                    "frame_id": frame_id,
                    "n_people": 3,
                    "response_time_ms": 750.0,
                    "inference_ms": 410.0,
                    "frame_bytes": 6312,
                }
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_csv = Path(temporary_directory) / "results.csv"
            frame_directory = Path(temporary_directory) / "frames"
            with (
                patch("local_vm.send_to_vm.get_frame", return_value=np.zeros((20, 30, 3), dtype=np.uint8)),
                patch("local_vm.send_to_vm.now_local", return_value=captured_at),
                patch("local_vm.send_to_vm.requests.post", return_value=response) as post,
            ):
                success, message = upload_one_frame(
                    cam="benidorm_poniente",
                    endpoint="http://vm/api/frames",
                    token="unit-test-token",
                    quality=85,
                    save_frames_dir=frame_directory,
                    results_csv=results_csv,
                )

            self.assertTrue(success)
            self.assertIn(frame_id, message)
            self.assertTrue((frame_directory / f"{frame_id}.jpg").is_file())
            post.assert_called_once()
            self.assertEqual(
                post.call_args.kwargs["headers"]["Authorization"],
                "Bearer unit-test-token",
            )
            with results_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["architecture"], "vm")
            self.assertEqual(rows[0]["cam_id"], "benidorm_poniente")
            self.assertEqual(rows[0]["frame_id"], frame_id)
            self.assertEqual(rows[0]["n_people"], "3")

    def test_upload_can_replay_a_saved_camera_frame(self):
        captured_at = now_local()
        frame_id = make_frame_id("benidorm_poniente", captured_at)
        response = Mock(
            status_code=201,
            json=lambda: {
                "result": {
                    "timestamp": captured_at.isoformat(timespec="milliseconds"),
                    "architecture": "vm",
                    "cam_id": "benidorm_poniente",
                    "frame_id": frame_id,
                    "n_people": 2,
                    "response_time_ms": 700.0,
                    "inference_ms": 400.0,
                    "frame_bytes": 6,
                }
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            replay_directory = Path(temporary_directory)
            replay_image = b"saved-jpeg"
            (replay_directory / "benidorm_poniente.jpg").write_bytes(replay_image)
            with (
                patch("local_vm.send_to_vm.get_frame") as capture,
                patch("local_vm.send_to_vm.now_local", return_value=captured_at),
                patch("local_vm.send_to_vm.requests.post", return_value=response) as post,
            ):
                success, message = upload_one_frame(
                    cam="benidorm_poniente",
                    endpoint="http://vm/api/frames",
                    token="unit-test-token",
                    quality=85,
                    save_frames_dir=None,
                    results_csv=None,
                    replay_dir=replay_directory,
                )

            self.assertTrue(success)
            self.assertIn(frame_id, message)
            capture.assert_not_called()
            self.assertEqual(
                post.call_args.kwargs["files"]["frame"][1],
                replay_image,
            )

    def test_upload_uses_the_sampled_stream_frame_and_capture_timestamp(self):
        captured_at = now_local()
        frame_id = make_frame_id("santa_pola", captured_at)
        response = Mock(
            status_code=201,
            json=lambda: {
                "result": {
                    "timestamp": captured_at.isoformat(timespec="milliseconds"),
                    "architecture": "vm",
                    "cam_id": "santa_pola",
                    "frame_id": frame_id,
                    "n_people": 1,
                    "response_time_ms": 500.0,
                    "inference_ms": 300.0,
                    "frame_bytes": 6,
                }
            },
        )
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        with (
            patch("local_vm.send_to_vm.get_frame") as capture,
            patch("local_vm.send_to_vm.requests.post", return_value=response) as post,
        ):
            success, _ = upload_one_frame(
                cam="santa_pola",
                endpoint="http://vm/api/frames",
                token="unit-test-token",
                quality=85,
                save_frames_dir=None,
                results_csv=None,
                live_frame=frame,
                live_capture_ms=2.0,
                live_captured_at=captured_at,
            )

        self.assertTrue(success)
        capture.assert_not_called()
        self.assertEqual(
            post.call_args.kwargs["data"]["timestamp"],
            captured_at.isoformat(timespec="milliseconds"),
        )

    def test_camera_workers_upload_independently_at_their_own_interval(self):
        completed = threading.Barrier(2)

        def upload_frame(cam):
            completed.wait(timeout=2)
            return True, f"{cam} uploaded", time.monotonic()

        stop = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            santa = executor.submit(run_camera, "santa_pola", 5.0, 1, stop, upload_frame)
            grazalema = executor.submit(run_camera, "grazalema", 5.0, 1, stop, upload_frame)
            self.assertEqual(santa.result(timeout=3), 1)
            self.assertEqual(grazalema.result(timeout=3), 1)

    def test_camera_worker_counts_successful_uploads_not_failures(self):
        calls = []

        def upload_frame(cam):
            calls.append(cam)
            return (
                len(calls) == 2,
                f"{cam} attempt {len(calls)}",
                time.monotonic(),
            )

        uploaded = run_camera(
            "madrid_alcala",
            interval=0.001,
            count=1,
            stop=threading.Event(),
            upload_frame=upload_frame,
        )

        self.assertEqual(uploaded, 1)
        self.assertEqual(calls, ["madrid_alcala", "madrid_alcala"])

    def test_camera_worker_waits_five_seconds_from_each_sample(self):
        sampled_at = []

        def upload_frame(_cam):
            sample_time = time.monotonic()
            sampled_at.append(sample_time)
            return True, "uploaded", sample_time

        run_camera(
            "santa_pola",
            interval=0.03,
            count=2,
            stop=threading.Event(),
            upload_frame=upload_frame,
        )

        self.assertGreaterEqual(sampled_at[1] - sampled_at[0], 0.03)

    def test_camera_worker_stops_after_three_consecutive_failures(self):
        with self.assertRaisesRegex(RuntimeError, "3 consecutive"):
            run_camera(
                "grazalema",
                interval=0.001,
                count=1,
                stop=threading.Event(),
                upload_frame=lambda _cam: (
                    False,
                    "camera unavailable",
                    time.monotonic(),
                ),
            )


if __name__ == "__main__":
    unittest.main()
