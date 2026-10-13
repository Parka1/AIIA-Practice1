"""Capture webcam frames locally and count people with Azure AI Vision.

First step of Part 3: checks the AI Vision resource (feature, region, quota) and
measures capture and inference times before the Azure Function exists. Results
are only printed; the serverless CSV is written by the function.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common.capture import CAMERAS, CaptureError, get_frame
from common.results import make_frame_id, now_local
from serverless.credentials import load_credentials
from serverless.vision import DEFAULT_MIN_CONFIDENCE, VisionError, analyze_people, confident_people

BOX_COLOR = (0, 200, 0)  # BGR


def draw_people(frame: np.ndarray, people, min_confidence: float) -> np.ndarray:
    """Return a copy of the frame with a box and confidence for every confident person."""
    annotated = frame.copy()
    for person in confident_people(people, min_confidence):
        cv2.rectangle(annotated, (person.x, person.y),(person.x + person.w, person.y + person.h), BOX_COLOR, 2)
        cv2.putText(
            annotated, f"person {person.confidence:.2f}", (person.x, max(person.y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2,
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cam", default="grazalema", choices=CAMERAS, help="camera id (default: grazalema)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames (default: 5)")
    parser.add_argument("--count", type=int, default=0, help="frames to analyze, 0 = until Ctrl+C")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (default: 85, as in Part 2)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--save-dir", type=Path, help="save the frames with the detected people drawn")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    endpoint, key = load_credentials("vision_endpoint", "vision_key")
    analyzed = 0
    next_at = time.monotonic()
    try:
        while args.count == 0 or analyzed < args.count:
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
                try:
                    analysis = analyze_people(image, endpoint, key)
                except VisionError as e:
                    print(f"[{captured_at:%H:%M:%S}] {frame_id} vision error: {e}")
                else:
                    analyzed += 1
                    people = confident_people(analysis.people, args.min_confidence)
                    service = f"{analysis.service_ms:.0f} ms" if analysis.service_ms is not None else "n/a"
                    print(
                        f"[{captured_at:%H:%M:%S}] {frame_id} people={len(people)} "
                        f"(boxes={len(analysis.people)}) capture={capture_ms:.0f} ms "
                        f"inference={analysis.inference_ms:.0f} ms service={service} "
                        f"retries={analysis.retries} upload={len(image) / 1024:.1f} KiB"
                    )
                    if args.save_dir:
                        args.save_dir.mkdir(parents=True, exist_ok=True)
                        annotated = draw_people(frame, analysis.people, args.min_confidence)
                        # cv2.imwrite fails on non-ASCII paths on Windows, so encode and write the bytes.
                        (args.save_dir / f"{frame_id}.jpg").write_bytes(cv2.imencode(".jpg", annotated)[1].tobytes())
            if args.count and analyzed >= args.count:
                break
            # Keep a fixed rate; if capturing took longer than the interval, don't try to catch up.
            next_at = max(next_at + args.interval, time.monotonic())
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"{analyzed} frames analyzed")


if __name__ == "__main__":
    main()
