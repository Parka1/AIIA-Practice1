"""Capture webcam frames and count people locally with Ultralytics YOLO."""

import argparse
import math
import time
from pathlib import Path

from common.results import Result, append_result, now_local

DEFAULT_CSV = Path("results/local.csv")


def positive_float(value: str) -> float:
    """Parse a strictly positive interval."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative frame count."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def count_people(model, frame, confidence: float, image_size: int) -> tuple[int, float]:
    """Return the number of person detections and model inference time in ms."""
    started = time.perf_counter()
    predictions = model.predict(
        source=frame,
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - started) * 1000
    if len(predictions) != 1:
        raise RuntimeError(f"expected one YOLO result for a frame, got {len(predictions)}")
    boxes = predictions[0].boxes
    return (0 if boxes is None else len(boxes)), inference_ms


def run(args) -> int:
    """Capture and process frames until the requested count or Ctrl+C."""
    from common.capture import CAMERAS, CaptureError, get_frame
    from ultralytics import YOLO

    model = YOLO(args.model)
    saved = 0
    next_at = time.monotonic()
    try:
        while args.count == 0 or saved < args.count:
            try:
                frame = get_frame(CAMERAS[args.cam])
            except CaptureError as error:
                print(f"[{now_local():%H:%M:%S}] capture error: {error}")
            else:
                captured_at = now_local()
                processing_started = time.perf_counter()
                n_people, inference_ms = count_people(
                    model, frame, args.confidence, args.image_size
                )
                response_time_ms = (time.perf_counter() - processing_started) * 1000
                result = Result(
                    timestamp=captured_at,
                    architecture="local",
                    cam_id=args.cam,
                    n_people=n_people,
                    response_time_ms=response_time_ms,
                    inference_ms=inference_ms,
                )
                append_result(args.csv, result)
                saved += 1
                print(
                    f"[{captured_at:%H:%M:%S}] {result.frame_id} "
                    f"people={n_people} inference={inference_ms:.1f} ms "
                    f"response={response_time_ms:.1f} ms"
                )

            if args.count and saved >= args.count:
                break
            next_at = max(next_at + args.interval, time.monotonic())
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"{saved} frame results written to {args.csv}")
    return saved


def main() -> None:
    from common.capture import CAMERAS

    parser = argparse.ArgumentParser(
        description="Count people in webcam frames locally with YOLOv8."
    )
    parser.add_argument("--cam", default="santa_pola", choices=CAMERAS)
    parser.add_argument("--interval", type=positive_float, default=5.0)
    parser.add_argument("--count", type=non_negative_int, default=0)
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model or weights path")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--image-size",
        type=positive_int,
        default=960,
        help="YOLO inference image size in pixels (default: 960)",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
