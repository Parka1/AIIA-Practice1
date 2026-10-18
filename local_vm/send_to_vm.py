"""Capture webcam frames locally and send them to the Part 2 VM."""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from common.capture import CAMERAS, CaptureError, LiveFrameReader, get_frame
from common.results import Result, append_result, make_frame_id, now_local

DEFAULT_TOKEN_FILE = Path.home() / ".config/aiia-practice1/vm-upload-token"
MAX_CONSECUTIVE_UPLOAD_FAILURES = 3
PRINT_LOCK = threading.Lock()
CSV_LOCK = threading.Lock()


def bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return parse


def upload_one_frame(
    cam: str,
    endpoint: str,
    token: str,
    quality: int,
    save_frames_dir: Path | None,
    results_csv: Path | None,
    replay_dir: Path | None = None,
    live_frame: np.ndarray | None = None,
    live_capture_ms: float | None = None,
    live_captured_at: datetime | None = None,
) -> tuple[bool, str]:
    capture_started = time.perf_counter()
    if replay_dir:
        image = (replay_dir / f"{cam}.jpg").read_bytes()
    else:
        if live_frame is None:
            try:
                frame = get_frame(CAMERAS[cam])
            except CaptureError as error:
                return False, f"[{now_local():%H:%M:%S}] {cam} capture error: {error}"
        else:
            frame = live_frame
        capture_ms = (
            live_capture_ms
            if live_capture_ms is not None
            else (time.perf_counter() - capture_started) * 1000
        )
        captured_at = live_captured_at or now_local()
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            raise RuntimeError(f"OpenCV could not encode the {cam} frame")
        image = encoded.tobytes()

    if replay_dir:
        capture_ms = (time.perf_counter() - capture_started) * 1000
        captured_at = now_local()
    frame_id = make_frame_id(cam, captured_at)
    if save_frames_dir:
        save_frames_dir.mkdir(parents=True, exist_ok=True)
        (save_frames_dir / f"{frame_id}.jpg").write_bytes(image)

    started = time.perf_counter()
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            data={
                "cam_id": cam,
                "timestamp": captured_at.isoformat(timespec="milliseconds"),
            },
            files={"frame": (f"{frame_id}.jpg", image, "image/jpeg")},
            timeout=120,
        )
    except requests.exceptions.RequestException as error:
        return False, f"[{captured_at:%H:%M:%S}] {cam} upload error: {error}"

    if response.status_code in (401, 403):
        raise RuntimeError(
            "VM rejected the upload token; check the local token file "
            "and remote service configuration"
        )
    if response.status_code >= 400:
        return (
            False,
            f"[{captured_at:%H:%M:%S}] {cam} VM returned "
            f"{response.status_code}: {response.text[:500]}",
        )

    payload = response.json()["result"]
    if results_csv:
        with CSV_LOCK:
            append_result(
                results_csv,
                Result(
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    architecture=payload["architecture"],
                    cam_id=payload["cam_id"],
                    frame_id=payload["frame_id"],
                    n_people=int(payload["n_people"]),
                    response_time_ms=float(payload["response_time_ms"]),
                    inference_ms=float(payload["inference_ms"]),
                    frame_bytes=int(payload["frame_bytes"]),
                ),
            )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return (
        True,
        f"[{captured_at:%H:%M:%S}] {payload['frame_id']} "
        f"capture={capture_ms:.1f} ms "
        f"people={payload['n_people']} "
        f"inference={payload['inference_ms']:.1f} ms "
        f"response={payload['response_time_ms']:.1f} ms "
        f"upload={len(image) / 1024:.1f} KiB HTTP={elapsed_ms:.1f} ms",
    )


def run_camera(
    cam: str,
    interval: float,
    count: int,
    stop: threading.Event,
    upload_frame,
) -> int:
    uploaded = 0
    consecutive_failures = 0
    while not stop.is_set() and (count == 0 or uploaded < count):
        success, message, sampled_at = upload_frame(cam)
        with PRINT_LOCK:
            print(message)
        if success:
            uploaded += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_UPLOAD_FAILURES:
                raise RuntimeError(
                    f"{cam} failed {consecutive_failures} consecutive captures or uploads"
                )
        if count and uploaded >= count:
            break
        next_at = max(sampled_at + interval, time.monotonic())
        if stop.wait(max(0.0, next_at - time.monotonic())):
            break
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture frames and upload them to the YOLOv8 Azure VM."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("AIIA_VM_URL"),
        help="public VM base URL (or set AIIA_VM_URL)",
    )
    parser.add_argument("--cam", choices=CAMERAS, help="single camera (default: santa_pola)")
    parser.add_argument(
        "--cams",
        nargs="+",
        choices=CAMERAS,
        help="capture cameras concurrently, each at the configured interval",
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="successful uploads per camera; 0 runs continuously",
    )
    parser.add_argument("--quality", type=bounded_int(1, 100), default=85)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="optional local copy of successful results in the shared CSV format",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=Path,
        help="optionally save captured JPEGs locally for detection review",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        help="replay one saved <cam_id>.jpg per camera instead of capturing live frames",
    )
    parser.add_argument(
        "--persistent-streams",
        action="store_true",
        help="keep each HLS stream open and sample its latest frame for every upload",
    )
    args = parser.parse_args()
    if not args.url:
        parser.error("provide --url or set AIIA_VM_URL")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.count < 0:
        parser.error("--count must be zero or greater")
    if args.cam and args.cams:
        parser.error("--cam and --cams cannot be combined")
    cameras = args.cams or [args.cam or "santa_pola"]
    if len(cameras) != len(set(cameras)):
        parser.error("each camera may appear only once in --cams")
    if args.replay_dir:
        missing = [
            cam for cam in cameras if not (args.replay_dir / f"{cam}.jpg").is_file()
        ]
        if missing:
            parser.error(
                f"--replay-dir is missing JPEGs for: {', '.join(missing)}"
            )
    if args.persistent_streams and args.replay_dir:
        parser.error("--persistent-streams cannot be combined with --replay-dir")
    if args.persistent_streams:
        snapshots = [
            cam for cam in cameras
            if Path(CAMERAS[cam].split("?", 1)[0]).suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        ]
        if snapshots:
            parser.error(
                "--persistent-streams requires video streams; snapshot sources: "
                f"{', '.join(snapshots)}"
            )

    token = os.getenv("AIIA_VM_UPLOAD_TOKEN")
    if not token and args.token_file.is_file():
        token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        parser.error(
            "set AIIA_VM_UPLOAD_TOKEN or create the token file at "
            f"{args.token_file}"
        )

    endpoint = f"{args.url.rstrip('/')}/api/frames"
    stop = threading.Event()
    readers: dict[str, LiveFrameReader] = {}

    def upload_frame(cam: str) -> tuple[bool, str, float]:
        if args.replay_dir:
            sampled_at = time.monotonic()
            success, message = upload_one_frame(
                cam=cam,
                endpoint=endpoint,
                token=token,
                quality=args.quality,
                save_frames_dir=args.save_frames_dir,
                results_csv=args.results_csv,
                replay_dir=args.replay_dir,
            )
            return success, message, sampled_at

        capture_started = time.perf_counter()
        try:
            frame = (
                readers[cam].latest_frame()
                if args.persistent_streams
                else get_frame(CAMERAS[cam])
            )
        except CaptureError as error:
            return (
                False,
                f"[{now_local():%H:%M:%S}] {cam} capture error: {error}",
                time.monotonic(),
            )
        live_capture_ms = (time.perf_counter() - capture_started) * 1000
        live_captured_at = now_local()
        sampled_at = time.monotonic()
        success, message = upload_one_frame(
            cam=cam,
            endpoint=endpoint,
            token=token,
            quality=args.quality,
            save_frames_dir=args.save_frames_dir,
            results_csv=args.results_csv,
            replay_dir=args.replay_dir,
            live_frame=frame,
            live_capture_ms=live_capture_ms,
            live_captured_at=live_captured_at,
        )
        return success, message, sampled_at

    try:
        if args.persistent_streams:
            readers = {
                cam: LiveFrameReader(CAMERAS[cam])
                for cam in cameras
            }
            for reader in readers.values():
                reader.start()
            for reader in readers.values():
                reader.latest_frame(timeout=30, max_age_s=10)
        with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            futures = {
                executor.submit(
                    run_camera,
                    cam,
                    args.interval,
                    args.count,
                    stop,
                    upload_frame,
                ): cam
                for cam in cameras
            }
            for future in as_completed(futures):
                cam = futures[future]
                try:
                    uploaded = future.result()
                except Exception:
                    stop.set()
                    raise
                print(f"{cam}: {uploaded} frames uploaded")
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        for reader in readers.values():
            reader.close()
    print(f"Uploads completed to {args.url}")


if __name__ == "__main__":
    main()
