"""Processing of one raw frame: AI Vision people count, annotated image and results.

Kept apart from function_app.py so it can be tested without the Functions runtime.

Everything is written to the $web container of the static website, never to the
raw container: a write there would trigger the function again (recursion).

    $web/frames/<frame_id>.jpg   frame with the detected people drawn
    $web/data/results.csv        common results format (common/results.py)
    $web/data/metrics.csv        timings of every stage, for Task 3.6

Both CSV files are append blobs, so concurrent function instances can add rows
without overwriting each other.
"""

import csv
import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Callable, Optional

from azure.core.exceptions import HttpResponseError, ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import ContainerClient, ContentSettings
from PIL import Image, ImageDraw, ImageFont

from results import FIELDNAMES, Result, format_csv, now_local, parse_frame_id
from vision import Analysis, VisionError, analyze_people, confident_people

RAW_CONTAINER = "raw-frames"
WEB_CONTAINER = "$web"
assert RAW_CONTAINER != WEB_CONTAINER, "the function would trigger itself recursively"

RESULTS_BLOB = "data/results.csv"
METRICS_BLOB = "data/metrics.csv"
FRAME_BLOB = "frames/{frame_id}.jpg"
METRICS_FIELDS = [
    "frame_id", "cam_id", "captured_at", "cold_start", "status",
    "start_delay_ms",  # capture -> function start: upload + Event Grid + cold start
    "inference_ms", "service_ms", "vision_retries", "draw_ms", "store_ms",
    "function_ms",  # function start -> results stored
    "response_time_ms", "boxes", "n_people", "min_confidence", "frame_bytes",
]

BOX_COLOR = (0, 200, 0)
JPEG_QUALITY = 85
NO_CACHE = "no-cache"  # the dashboard always needs the latest version


@dataclass(frozen=True)
class Timer:
    """Function start, both as wall-clock time (to compare with the capture time) and as perf_counter."""

    started_at: datetime
    start: float

    @classmethod
    def now(cls) -> "Timer":
        return cls(now_local(), time.perf_counter())

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000


def process_frame(
    blob_name: str,
    image: bytes,
    web: ContainerClient,
    endpoint: str,
    key: str,
    min_confidence: float,
    timer: Timer,
    cold_start: bool,
    analyze: Callable[..., Analysis] = analyze_people,
) -> Optional[Result]:
    """Count the people in a raw frame and store the results in $web.

    Returns None (and records the error in metrics.csv) if AI Vision fails, so
    the frame is not retried and does not spend more of the AI Vision quota.
    """
    frame_id = PurePosixPath(blob_name).stem
    cam_id, captured_at = parse_frame_id(frame_id)
    metrics = {
        "frame_id": frame_id,
        "cam_id": cam_id,
        "captured_at": captured_at.isoformat(timespec="milliseconds"),
        "cold_start": int(cold_start),
        "start_delay_ms": _ms_between(captured_at, timer.started_at),
        "min_confidence": min_confidence,
        "frame_bytes": len(image),
    }

    try:
        analysis = analyze(image, endpoint, key)
    except VisionError as e:
        logging.error("AI Vision failed for %s: %s", frame_id, e)
        metrics.update(status=f"vision error: {e}"[:200], function_ms=round(timer.elapsed_ms(), 1))
        append_csv_row(web, METRICS_BLOB, METRICS_FIELDS, metrics)
        return None

    people = confident_people(analysis.people, min_confidence)
    draw_start = time.perf_counter()
    annotated = draw_people(image, people)
    draw_ms = (time.perf_counter() - draw_start) * 1000

    store_start = time.perf_counter()
    web.upload_blob(
        FRAME_BLOB.format(frame_id=frame_id),
        annotated,
        overwrite=True,
        content_settings=ContentSettings(content_type="image/jpeg"),
    )
    # Response time ends when the result is stored: measured just before appending the row.
    result = Result(
        timestamp=captured_at,
        architecture="serverless",
        cam_id=cam_id,
        frame_id=frame_id,
        n_people=len(people),
        response_time_ms=_ms_between(captured_at, now_local()),
        inference_ms=analysis.inference_ms,
        frame_bytes=len(image),
    )
    append_csv_text(web, RESULTS_BLOB, ",".join(FIELDNAMES) + "\n", format_csv(result))
    store_ms = (time.perf_counter() - store_start) * 1000

    metrics.update(
        status="ok",
        inference_ms=round(analysis.inference_ms, 1),
        service_ms=analysis.service_ms if analysis.service_ms is not None else "",
        vision_retries=analysis.retries,
        draw_ms=round(draw_ms, 1),
        store_ms=round(store_ms, 1),
        function_ms=round(timer.elapsed_ms(), 1),
        response_time_ms=round(result.response_time_ms, 1),
        boxes=len(analysis.people),
        n_people=len(people),
    )
    append_csv_row(web, METRICS_BLOB, METRICS_FIELDS, metrics)
    return result


def draw_people(image: bytes, people) -> bytes:
    """Return the JPEG with a box and confidence label for every person."""
    with Image.open(io.BytesIO(image)) as opened:
        frame = opened.convert("RGB")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=22)
    for person in people:
        draw.rectangle([person.x, person.y, person.x + person.w, person.y + person.h], outline=BOX_COLOR, width=3)
        draw.text((person.x, max(person.y - 26, 0)), f"person {person.confidence:.2f}", fill=BOX_COLOR, font=font)
    out = io.BytesIO()
    frame.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def append_csv_row(web: ContainerClient, blob_name: str, fieldnames: list[str], row: dict) -> None:
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n").writerow(row)
    append_csv_text(web, blob_name, ",".join(fieldnames) + "\n", buf.getvalue())


def append_csv_text(web: ContainerClient, blob_name: str, header: str, text: str) -> None:
    """Append text to a CSV append blob, creating it with its header the first time."""
    blob = web.get_blob_client(blob_name)
    try:
        blob.append_block(text.encode("utf-8"))
    except ResourceNotFoundError:
        create_csv_blob(web, blob_name, header)
        blob.append_block(text.encode("utf-8"))


def create_csv_blob(web: ContainerClient, blob_name: str, header: str) -> None:
    """Create an empty CSV append blob with its header, unless it already exists."""
    blob = web.get_blob_client(blob_name)
    try:
        blob.create_append_blob(
            content_settings=ContentSettings(content_type="text/csv", cache_control=NO_CACHE),
            if_none_match="*",
        )
    except (ResourceExistsError, HttpResponseError) as e:
        # Another instance created it first: 409 or 412 (precondition failed).
        if getattr(e, "status_code", None) not in (409, 412):
            raise
        return
    # Only the instance that created the blob writes the header, at offset 0.
    blob.append_block(header.encode("utf-8"), appendpos_condition=0)


def _ms_between(start: datetime, end: datetime) -> float:
    return round((end - start).total_seconds() * 1000, 1)
