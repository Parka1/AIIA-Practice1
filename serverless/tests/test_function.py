import csv
import io
import sys
import unittest
from datetime import timedelta
from pathlib import Path

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from PIL import Image

FUNCTION_DIR = Path(__file__).resolve().parents[1] / "function_app"
sys.path.insert(0, str(FUNCTION_DIR))

import frame_processing  # noqa: E402
from frame_processing import METRICS_BLOB, RESULTS_BLOB, Timer, process_frame  # noqa: E402
from results import FIELDNAMES, parse_frame_id  # noqa: E402
from vision import Analysis, Person, VisionError  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class FakeBlob:
    def __init__(self, container, name):
        self.container, self.name = container, name

    def append_block(self, data, appendpos_condition=None):
        if self.name not in self.container.blobs:
            raise ResourceNotFoundError("missing")
        if appendpos_condition is not None:
            assert len(self.container.blobs[self.name]) == appendpos_condition
        self.container.blobs[self.name] += data

    def create_append_blob(self, content_settings=None, if_none_match=None):
        if self.name in self.container.blobs:
            raise ResourceExistsError("exists")
        self.container.blobs[self.name] = b""


class FakeContainer:
    def get_blob_client(self, name):
        return FakeBlob(self, name)

    def __init__(self):
        self.blobs = {}

    def upload_blob(self, name, data, overwrite=False, content_settings=None):
        self.blobs[name] = data


def jpeg(width=64, height=48):
    out = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(out, format="JPEG")
    return out.getvalue()


def rows(container, name):
    return list(csv.DictReader(io.StringIO(container.blobs[name].decode("utf-8"))))


class ProcessFrameTests(unittest.TestCase):
    frame_id = "santa_pola_20261013T193044917"

    def run_frame(self, web, analyze):
        _, captured_at = parse_frame_id(self.frame_id)
        timer = Timer(started_at=captured_at + timedelta(milliseconds=800), start=0.0)
        return process_frame(
            f"{self.frame_id}.jpg", jpeg(), web, "https://endpoint", "key",
            min_confidence=0.5, timer=timer, cold_start=True, analyze=analyze,
        )

    def test_stores_annotated_frame_and_one_row_in_each_csv(self):
        people = [Person(5, 5, 10, 20, 0.66), Person(1, 1, 2, 2, 0.1)]
        analysis = Analysis(people, 64, 48, inference_ms=950.0, service_ms=420.0, retries=1)
        web = FakeContainer()

        result = self.run_frame(web, lambda image, endpoint, key: analysis)
        self.run_frame(web, lambda image, endpoint, key: analysis)

        self.assertEqual(result.n_people, 1)
        self.assertEqual(result.architecture, "serverless")
        self.assertEqual(result.cam_id, "santa_pola")
        self.assertIn(f"frames/{self.frame_id}.jpg", web.blobs)
        self.assertEqual(web.blobs[RESULTS_BLOB].decode().splitlines()[0], ",".join(FIELDNAMES))
        results = rows(web, RESULTS_BLOB)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["frame_id"], self.frame_id)
        self.assertEqual(results[0]["inference_ms"], "950.0")
        metrics = rows(web, METRICS_BLOB)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0]["status"], "ok")
        self.assertEqual(metrics[0]["start_delay_ms"], "800.0")
        self.assertEqual(metrics[0]["boxes"], "2")
        self.assertEqual(metrics[0]["vision_retries"], "1")
        self.assertEqual(metrics[0]["cold_start"], "1")

    def test_vision_error_is_recorded_without_a_result_row(self):
        def failing(image, endpoint, key):
            raise VisionError("AI Vision returned HTTP 429")

        web = FakeContainer()
        with self.assertLogs(level="ERROR"):
            self.assertIsNone(self.run_frame(web, failing))

        self.assertNotIn(RESULTS_BLOB, web.blobs)
        self.assertIn("429", rows(web, METRICS_BLOB)[0]["status"])

    def test_nothing_is_written_to_the_raw_container(self):
        web = FakeContainer()
        self.run_frame(web, lambda image, endpoint, key: Analysis([], 64, 48, 1.0, None, 0))

        self.assertTrue(all(not name.startswith(frame_processing.RAW_CONTAINER) for name in web.blobs))
        self.assertNotEqual(frame_processing.RAW_CONTAINER, frame_processing.WEB_CONTAINER)


class SharedModuleCopyTests(unittest.TestCase):
    def test_function_copies_match_the_originals(self):
        # The function app is deployed on its own, so it carries copies of these modules.
        for copy, original in (
            ("results.py", "common/results.py"),
            ("vision.py", "serverless/vision.py"),
        ):
            self.assertEqual(
                (FUNCTION_DIR / copy).read_bytes(), (REPO / original).read_bytes(),
                f"serverless/function_app/{copy} is out of date: copy {original} again",
            )


if __name__ == "__main__":
    unittest.main()
