"""Capture webcam frames locally and upload them to the raw-frames Blob Storage container.

Each upload will trigger the Part 3 Azure Function. The blob is named
<frame_id>.jpg and carries the camera and capture time as metadata, so the
function can compute the end-to-end response time.

The raw container must never be the $web container of the static website:
the function writes its results there, and a trigger on $web would make the
function trigger itself recursively.

Every upload is also logged locally (capture and upload times) for Task 3.6.

Optional client-side optimizations (Advanced task 3, common/optimizations.py):
--width lowers the resolution and --motion skips frames without motion, so
fewer bytes are uploaded and fewer AI Vision calls are made.

Several cameras (Part 4, Task 4.3): --cams starts one capture/upload thread per
camera, like local_vm/send_to_vm.py. Pair it with --persistent-streams to keep
each HLS stream open and sample its latest frame, which keeps the five-second
cadence (see common/capture.py).
"""

import argparse
import csv
import threading
import time
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings

from common.capture import CAMERAS, CaptureError, LiveFrameReader, get_frame
from common.optimizations import MOTION_REGIONS, MotionDetector, encode_jpeg, resize_to_width
from common.results import make_frame_id, now_local
from serverless.credentials import load_credentials

DEFAULT_CONTAINER = "raw-frames"
WEB_CONTAINER = "$web"
# AI Vision free tier (F0) limit; every uploaded frame is one call.
VISION_CALLS_PER_MINUTE = 20
UPLOAD_LOG_FIELDS = ["frame_id", "cam_id", "captured_at", "capture_ms", "upload_ms", "frame_bytes"]
MAX_CONSECUTIVE_FAILURES = 3  # per camera, then the run stops instead of hanging on a dead stream

_print_lock = threading.Lock()
_log_lock = threading.Lock()


def append_upload_log(path: Path, row: dict, fieldnames: list[str] = UPLOAD_LOG_FIELDS) -> None:
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def say(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def raw_container(name: str = DEFAULT_CONTAINER) -> ContainerClient:
    if name == WEB_CONTAINER:
        raise ValueError("frames must not be uploaded to $web: the function would trigger itself recursively")
    (connection_string,) = load_credentials("storage_connection_string")
    container = BlobServiceClient.from_connection_string(connection_string).get_container_client(name)
    if not container.exists():
        raise SystemExit(f"container {name!r} does not exist in the storage account")
    return container


def upload_cameras(
    container: ContainerClient,
    cams: list[str],
    *,
    interval: float = 5.0,
    count: int = 0,
    quality: int = 85,
    width: int = 0,
    motion: float = 0.0,
    log: Path = Path("results/serverless_uploads.csv"),
    stage: str = "",
    persistent_streams: bool = False,
    save_frames_dir: Path | None = None,
) -> dict[str, int]:
    """Upload `count` frames per camera (0 = until Ctrl+C), one thread per camera.

    With `stage`, a "stage" column is added to the upload log, so the runs of a
    multi-camera staircase can be told apart. Returns the frames uploaded per camera.
    """
    fieldnames = UPLOAD_LOG_FIELDS + (["stage"] if stage else [])
    stop = threading.Event()
    readers: dict[str, LiveFrameReader] = {}
    uploaded = {cam: 0 for cam in cams}
    errors: list[BaseException] = []

    def run_camera(cam: str) -> None:
        detector = MotionDetector(min_changed=motion, region=MOTION_REGIONS.get(cam)) if motion else None
        failures = skipped = 0
        next_at = time.monotonic()
        while not stop.is_set() and (count == 0 or uploaded[cam] < count):
            start = time.perf_counter()
            try:
                frame = readers[cam].latest_frame() if persistent_streams else get_frame(CAMERAS[cam])
            except CaptureError as e:
                say(f"[{now_local():%H:%M:%S}] {cam} capture error: {e}")
                failures += 1
            else:
                captured_at = now_local()
                capture_ms = (time.perf_counter() - start) * 1000
                if detector and not detector.should_send(frame):
                    skipped += 1
                    say(f"[{captured_at:%H:%M:%S}] {cam} no motion, frame skipped")
                else:
                    if width:
                        frame = resize_to_width(frame, width)
                    image = encode_jpeg(frame, quality)
                    frame_id = make_frame_id(cam, captured_at)
                    if save_frames_dir:
                        save_frames_dir.mkdir(parents=True, exist_ok=True)
                        (save_frames_dir / f"{frame_id}.jpg").write_bytes(image)
                    upload_start = time.perf_counter()
                    try:
                        container.upload_blob(
                            f"{frame_id}.jpg",
                            image,
                            content_settings=ContentSettings(content_type="image/jpeg"),
                            metadata={
                                "cam_id": cam,
                                "frame_id": frame_id,
                                "captured_at": captured_at.isoformat(timespec="milliseconds"),
                            },
                        )
                    except AzureError as e:
                        say(f"[{captured_at:%H:%M:%S}] {frame_id} upload error: {e}")
                        failures += 1
                    else:
                        upload_ms = (time.perf_counter() - upload_start) * 1000
                        uploaded[cam] += 1
                        failures = 0
                        row = {
                            "frame_id": frame_id,
                            "cam_id": cam,
                            "captured_at": captured_at.isoformat(timespec="milliseconds"),
                            "capture_ms": round(capture_ms, 1),
                            "upload_ms": round(upload_ms, 1),
                            "frame_bytes": len(image),
                        }
                        append_upload_log(log, row | ({"stage": stage} if stage else {}), fieldnames)
                        say(
                            f"[{captured_at:%H:%M:%S}] {container.container_name}/{frame_id}.jpg "
                            f"capture={capture_ms:.0f} ms upload={upload_ms:.0f} ms size={len(image) / 1024:.1f} KiB"
                        )
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"{cam} failed {failures} consecutive captures or uploads")
            if count and uploaded[cam] >= count:
                break
            # Keep a fixed rate; if capturing took longer than the interval, don't try to catch up.
            next_at = max(next_at + interval, time.monotonic())
            stop.wait(max(0.0, next_at - time.monotonic()))
        if detector:
            say(f"{cam}: {skipped} frames skipped without motion")

    def worker(cam: str) -> None:
        try:
            run_camera(cam)
        except BaseException as e:  # stop the other cameras too
            errors.append(e)
            stop.set()

    try:
        if persistent_streams:
            readers = {cam: LiveFrameReader(CAMERAS[cam]) for cam in cams}
            for reader in readers.values():
                reader.start()
            for reader in readers.values():
                reader.latest_frame(timeout=30, max_age_s=10)
        threads = [threading.Thread(target=worker, args=(cam,), daemon=True) for cam in cams]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        for reader in readers.values():
            reader.close()
    if errors:
        raise errors[0]
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cam", default="santa_pola", choices=CAMERAS, help="camera id (default: santa_pola)")
    parser.add_argument("--cams", nargs="+", choices=CAMERAS, help="several cameras at once, one thread each")
    parser.add_argument("--persistent-streams", action="store_true",
                        help="keep each HLS stream open and sample its latest frame (recommended with --cams)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames (default: 5)")
    parser.add_argument("--count", type=int, default=0, help="frames to upload per camera, 0 = until Ctrl+C")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (default: 85, as in Part 2)")
    parser.add_argument("--width", type=int, default=0, help="resize frames to this width in px (default: keep 1920)")
    parser.add_argument("--motion", type=float, default=0.0, metavar="FRACTION",
                        help="only upload frames with at least this fraction of changed pixels, e.g. 0.005 (default: off)")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help=f"raw container (default: {DEFAULT_CONTAINER})")
    parser.add_argument("--log", type=Path, default=Path("results/serverless_uploads.csv"))
    parser.add_argument("--save-frames-dir", type=Path, help="also save the uploaded JPEGs locally")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.container == WEB_CONTAINER:
        parser.error("frames must not be uploaded to $web: the function would trigger itself recursively")
    cams = args.cams or [args.cam]
    if len(cams) != len(set(cams)):
        parser.error("each camera may appear only once in --cams")
    calls_per_minute = len(cams) * 60 / args.interval
    if calls_per_minute > VISION_CALLS_PER_MINUTE:
        print(
            f"warning: {calls_per_minute:.0f} frames/min exceeds the {VISION_CALLS_PER_MINUTE} calls/min "
            "of the AI Vision free tier; the function will get HTTP 429"
        )

    container = raw_container(args.container)
    uploaded = upload_cameras(
        container, cams, interval=args.interval, count=args.count, quality=args.quality, width=args.width,
        motion=args.motion, log=args.log, persistent_streams=args.persistent_streams,
        save_frames_dir=args.save_frames_dir,
    )
    print(f"{sum(uploaded.values())} frames uploaded to {args.container}: "
          + ", ".join(f"{cam} {n}" for cam, n in uploaded.items()))


if __name__ == "__main__":
    main()
