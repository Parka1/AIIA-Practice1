"""Capture webcam frames locally and upload them to the raw-frames Blob Storage container.

Each upload will trigger the Part 3 Azure Function. The blob is named
<frame_id>.jpg and carries the camera and capture time as metadata, so the
function can compute the end-to-end response time.

The raw container must never be the $web container of the static website:
the function writes its results there, and a trigger on $web would make the
function trigger itself recursively.

Every upload is also logged locally (capture and upload times) for Task 3.6.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient, ContentSettings

from common.capture import CAMERAS, CaptureError, get_frame
from common.results import make_frame_id, now_local
from serverless.credentials import load_credentials

DEFAULT_CONTAINER = "raw-frames"
WEB_CONTAINER = "$web"
# AI Vision free tier (F0) limit; every uploaded frame is one call.
VISION_CALLS_PER_MINUTE = 20
UPLOAD_LOG_FIELDS = ["frame_id", "cam_id", "captured_at", "capture_ms", "upload_ms", "frame_bytes"]


def append_upload_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UPLOAD_LOG_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cam", default="grazalema", choices=CAMERAS, help="camera id (default: grazalema)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames (default: 5)")
    parser.add_argument("--count", type=int, default=0, help="frames to upload, 0 = until Ctrl+C")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (default: 85, as in Part 2)")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help=f"raw container (default: {DEFAULT_CONTAINER})")
    parser.add_argument("--log", type=Path, default=Path("results/serverless_uploads.csv"))
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.container == WEB_CONTAINER:
        parser.error("frames must not be uploaded to $web: the function would trigger itself recursively")
    if 60 / args.interval > VISION_CALLS_PER_MINUTE:
        print(
            f"warning: {60 / args.interval:.0f} frames/min exceeds the {VISION_CALLS_PER_MINUTE} calls/min "
            "of the AI Vision free tier; the function will get HTTP 429"
        )

    (connection_string,) = load_credentials("storage_connection_string")
    container = BlobServiceClient.from_connection_string(connection_string).get_container_client(args.container)
    if not container.exists():
        raise SystemExit(f"container {args.container!r} does not exist in the storage account")

    uploaded = 0
    next_at = time.monotonic()
    try:
        while args.count == 0 or uploaded < args.count:
            start = time.perf_counter()
            try:
                frame = get_frame(CAMERAS[args.cam])
            except CaptureError as e:
                print(f"[{now_local():%H:%M:%S}] capture error: {e}")
            else:
                captured_at = now_local()
                capture_ms = (time.perf_counter() - start) * 1000
                image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])[1].tobytes()
                frame_id = make_frame_id(args.cam, captured_at)
                upload_start = time.perf_counter()
                try:
                    container.upload_blob(
                        f"{frame_id}.jpg",
                        image,
                        content_settings=ContentSettings(content_type="image/jpeg"),
                        metadata={
                            "cam_id": args.cam,
                            "frame_id": frame_id,
                            "captured_at": captured_at.isoformat(timespec="milliseconds"),
                        },
                    )
                except AzureError as e:
                    print(f"[{captured_at:%H:%M:%S}] {frame_id} upload error: {e}")
                else:
                    upload_ms = (time.perf_counter() - upload_start) * 1000
                    uploaded += 1
                    append_upload_log(args.log, {
                        "frame_id": frame_id,
                        "cam_id": args.cam,
                        "captured_at": captured_at.isoformat(timespec="milliseconds"),
                        "capture_ms": round(capture_ms, 1),
                        "upload_ms": round(upload_ms, 1),
                        "frame_bytes": len(image),
                    })
                    print(
                        f"[{captured_at:%H:%M:%S}] {args.container}/{frame_id}.jpg "
                        f"capture={capture_ms:.0f} ms upload={upload_ms:.0f} ms size={len(image) / 1024:.1f} KiB"
                    )
            if args.count and uploaded >= args.count:
                break
            # Keep a fixed rate; if capturing took longer than the interval, don't try to catch up.
            next_at = max(next_at + args.interval, time.monotonic())
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"{uploaded} frames uploaded to {args.container}")


if __name__ == "__main__":
    main()
