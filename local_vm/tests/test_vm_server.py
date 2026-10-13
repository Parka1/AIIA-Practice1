import csv
import io
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import psutil

from common.results import now_local
from local_vm.tests.test_vm_inference import FakeNet, yolo_output
from local_vm.vm_server import create_app


class VmServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.data_dir = Path(self.temporary_directory.name)
        self.app = create_app(
            data_dir=self.data_dir,
            upload_token="test-only-token",
            hourly_rate_usd=0.0114,
            detector=FakeNet(yolo_output()),
        )
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        ok, image = cv2.imencode(
            ".jpg",
            np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        self.assertTrue(ok)
        self.image_bytes = image.tobytes()

    def upload(self, timestamp=None, token="test-only-token", payload=None):
        return self.client.post(
            "/api/frames",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "cam_id": "santa_pola",
                "timestamp": (timestamp or now_local()).isoformat(timespec="milliseconds"),
                "frame": (
                    io.BytesIO(self.image_bytes if payload is None else payload),
                    "frame.jpg",
                    "image/jpeg",
                ),
            },
        )

    def test_rejects_missing_or_invalid_upload_token(self):
        response = self.upload(token="wrong")

        self.assertEqual(response.status_code, 401)

    def test_processes_frame_and_persists_common_result_and_metrics(self):
        response = self.upload()

        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["result"]["architecture"], "vm")
        self.assertEqual(body["result"]["n_people"], 1)
        self.assertEqual(body["result"]["frame_bytes"], len(self.image_bytes))
        self.assertTrue((self.data_dir / "latest.jpg").is_file())
        with (self.data_dir / "results.csv").open(newline="", encoding="utf-8") as stream:
            result = next(csv.DictReader(stream))
        self.assertEqual(result["n_people"], "1")
        self.assertEqual(result["cam_id"], "santa_pola")
        with (self.data_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            metrics = next(csv.DictReader(stream))
        self.assertEqual(int(metrics["upload_bytes"]), len(self.image_bytes))
        self.assertIn("cpu_percent", metrics)
        self.assertIn("process_rss_mb", metrics)

        dashboard = self.client.get("/api/dashboard").get_json()
        self.assertEqual(dashboard["total_frames"], 1)
        self.assertEqual(dashboard["upload_bytes"], len(self.image_bytes))
        self.assertGreaterEqual(dashboard["estimated_vm_cost_usd"], 0)
        self.assertAlmostEqual(
            dashboard["vm_runtime_seconds"],
            time.time() - psutil.boot_time(),
            delta=1,
        )

    def test_rejects_invalid_image_and_stale_timestamp(self):
        invalid_image = self.upload(payload=b"not an image")
        old_timestamp = self.upload(timestamp=now_local() - timedelta(hours=1))

        self.assertEqual(invalid_image.status_code, 400)
        self.assertEqual(old_timestamp.status_code, 400)

    def test_upload_bandwidth_uses_capture_to_completion_window(self):
        start = now_local() - timedelta(seconds=10)
        self.assertEqual(self.upload(timestamp=start).status_code, 201)
        self.assertEqual(
            self.upload(timestamp=start + timedelta(seconds=5)).status_code,
            201,
        )

        dashboard = self.client.get("/api/dashboard").get_json()
        latest = dashboard["results"][-1]
        window_seconds = (
            (now_local() - start).total_seconds()
            if len(dashboard["results"]) == 1
            else (5 + float(latest["response_time_ms"]) / 1000)
        )
        expected_kib_per_second = (
            2 * len(self.image_bytes) / 1024 / window_seconds
        )
        self.assertAlmostEqual(
            dashboard["upload_kib_per_second"],
            expected_kib_per_second,
            places=3,
        )

    def test_upload_bandwidth_excludes_gaps_between_separate_runs(self):
        old_start = now_local() - timedelta(seconds=120)
        self.assertEqual(self.upload(timestamp=old_start).status_code, 201)
        self.assertEqual(
            self.upload(timestamp=old_start + timedelta(seconds=5)).status_code,
            201,
        )
        latest_start = now_local()
        self.assertEqual(self.upload(timestamp=latest_start).status_code, 201)

        dashboard = self.client.get("/api/dashboard").get_json()
        latest = dashboard["results"][-1]
        expected_rate = (
            len(self.image_bytes)
            / 1024
            / (float(latest["response_time_ms"]) / 1000)
        )
        self.assertAlmostEqual(
            dashboard["upload_kib_per_second"],
            expected_rate,
            places=1,
        )

    def test_health_dashboard_and_preview_routes(self):
        self.assertEqual(self.client.get("/api/health").get_json()["status"], "ok")
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/latest.jpg").status_code, 404)


if __name__ == "__main__":
    unittest.main()
