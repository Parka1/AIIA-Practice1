"""Capture webcam frames locally and send them to the Part 2 VM."""

import argparse
import os
import time
from pathlib import Path

import cv2
import requests

from common.capture import CAMERAS, CaptureError, get_frame
from common.results import make_frame_id, now_local

DEFAULT_TOKEN_FILE = Path.home() / ".config/aiia-practice1/vm-upload-token"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture frames and upload them to the YOLOv8 Azure VM."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("AIIA_VM_URL"),
        help="public VM base URL (or set AIIA_VM_URL)",
    )
    parser.add_argument("--cam", default="santa_pola", choices=CAMERAS)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--count", type=int, default=0, help="successful uploads; 0 runs continuously")
    parser.add_argument("--quality", type=bounded_int(1, 100), default=85)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument(
        "--save-frames-dir",
        type=Path,
        help="optionally save captured JPEGs locally for detection review",
    )
    args = parser.parse_args()
    if not args.url:
        parser.error("provide --url or set AIIA_VM_URL")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.count < 0:
        parser.error("--count must be zero or greater")

    token = os.getenv("AIIA_VM_UPLOAD_TOKEN")
    if not token and args.token_file.is_file():
        token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        parser.error(
            "set AIIA_VM_UPLOAD_TOKEN or create the token file at "
            f"{args.token_file}"
        )

    endpoint = f"{args.url.rstrip('/')}/api/frames"
    uploaded = 0
    next_at = time.monotonic()
    try:
        while args.count == 0 or uploaded < args.count:
            try:
                frame = get_frame(CAMERAS[args.cam])
            except CaptureError as error:
                print(f"[{now_local():%H:%M:%S}] capture error: {error}")
            else:
                captured_at = now_local()
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV could not encode the captured frame")
                image = encoded.tobytes()
                frame_id = make_frame_id(args.cam, captured_at)
                if args.save_frames_dir:
                    args.save_frames_dir.mkdir(parents=True, exist_ok=True)
                    (args.save_frames_dir / f"{frame_id}.jpg").write_bytes(image)
                started = time.perf_counter()
                try:
                    response = requests.post(
                        endpoint,
                        headers={"Authorization": f"Bearer {token}"},
                        data={
                            "cam_id": args.cam,
                            "timestamp": captured_at.isoformat(timespec="milliseconds"),
                        },
                        files={"frame": (f"{frame_id}.jpg", image, "image/jpeg")},
                        timeout=120,
                    )
                except requests.exceptions.RequestException as error:
                    print(f"[{captured_at:%H:%M:%S}] upload error: {error}")
                else:
                    if response.status_code in (401, 403):
                        raise RuntimeError(
                            "VM rejected the upload token; check the local token file "
                            "and remote service configuration"
                        )
                    if response.status_code >= 400:
                        print(
                            f"[{captured_at:%H:%M:%S}] VM returned "
                            f"{response.status_code}: {response.text[:500]}"
                        )
                    else:
                        payload = response.json()["result"]
                        uploaded += 1
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        print(
                            f"[{captured_at:%H:%M:%S}] {payload['frame_id']} "
                            f"people={payload['n_people']} "
                            f"inference={payload['inference_ms']:.1f} ms "
                            f"response={payload['response_time_ms']:.1f} ms "
                            f"upload={len(image) / 1024:.1f} KiB "
                            f"HTTP={elapsed_ms:.1f} ms"
                        )

            if args.count and uploaded >= args.count:
                break
            next_at = max(next_at + args.interval, time.monotonic())
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"{uploaded} frames uploaded to {args.url}")


if __name__ == "__main__":
    main()
