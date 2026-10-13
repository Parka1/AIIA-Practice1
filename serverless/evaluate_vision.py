"""Count people with Azure AI Vision in the labelled Part 2 frames and compare with YOLO.

Every frame is sent once and all the detections are saved to a JSON file, so the
confidence threshold can be changed later without calling the service again
(use --reuse). The output CSV extends the Part 2 ground truth with the AI Vision
count, so both models are evaluated on exactly the same frames.

The confidence threshold (0.5 by default) was fixed before running this
evaluation; the threshold sweep is only a sensitivity check, since choosing the
best threshold on these 20 frames would overfit them.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from local_vm.evaluate_vm_counts import evaluate_rows, evaluate_rows_by_camera
from serverless.credentials import load_credentials
from serverless.probe_vision import draw_people
from serverless.vision import DEFAULT_MIN_CONFIDENCE, Person, analyze_people, count_people

SWEEP_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
# The free tier (F0) allows 20 calls per minute.
DEFAULT_PAUSE_S = 3.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ground_truth", nargs="?", type=Path, default=Path("report/data/part2_vm_ground_truth.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/data/part3_vision_ground_truth.csv"))
    parser.add_argument("--detections", type=Path, default=Path("report/data/part3_vision_detections.json"))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_S, help="seconds between AI Vision calls")
    parser.add_argument("--reuse", action="store_true", help="use the saved detections instead of calling AI Vision")
    parser.add_argument("--annotate-dir", type=Path, help="save the frames with the detected people drawn")
    args = parser.parse_args()

    with args.ground_truth.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.reuse:
        detections = json.loads(args.detections.read_text(encoding="utf-8"))
    else:
        detections = analyze_frames(rows, args.ground_truth.parent, args.pause)
        args.detections.parent.mkdir(parents=True, exist_ok=True)
        args.detections.write_text(json.dumps(detections, indent=1), encoding="utf-8")
        print(f"detections saved to {args.detections}")

    people = {frame_id: [Person(**p) for p in d["people"]] for frame_id, d in detections.items()}
    for row in rows:
        row["vision_count"] = str(count_people(people[row["frame_id"]], args.min_confidence))
        row["vision_inference_ms"] = round(detections[row["frame_id"]]["inference_ms"], 1)
    write_rows(args.out, rows)
    print(f"counts saved to {args.out}")

    if args.annotate_dir:
        args.annotate_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            frame_path = args.ground_truth.parent / row["frame_file"]
            frame = cv2.imdecode(np.fromfile(frame_path, np.uint8), cv2.IMREAD_COLOR)
            annotated = draw_people(frame, people[row["frame_id"]], args.min_confidence)
            (args.annotate_dir / frame_path.name).write_bytes(cv2.imencode(".jpg", annotated)[1].tobytes())

    print_report(rows, people, args.min_confidence)


def analyze_frames(rows: list[dict], base_dir: Path, pause_s: float) -> dict:
    endpoint, key = load_credentials("vision_endpoint", "vision_key")
    detections = {}
    for i, row in enumerate(rows):
        if i:
            time.sleep(pause_s)
        image = (base_dir / row["frame_file"]).read_bytes()
        analysis = analyze_people(image, endpoint, key)
        detections[row["frame_id"]] = {
            "inference_ms": analysis.inference_ms,
            "service_ms": analysis.service_ms,
            "retries": analysis.retries,
            "frame_bytes": len(image),
            "people": [asdict(p) for p in analysis.people],
        }
        print(
            f"{row['frame_id']} truth={row['ground_truth_count']} yolo={row['yolo_count']} "
            f"vision={count_people(analysis.people)} (boxes={len(analysis.people)}) "
            f"inference={analysis.inference_ms:.0f} ms retries={analysis.retries}"
        )
    return detections


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows: list[dict], people: dict, min_confidence: float) -> None:
    header = f"{'':<22}{'frames':>7}{'exact %':>9}{'MAE':>7}{'RMSE':>7}{'bias':>7}{'under':>7}{'over':>6}"

    def line(label: str, m: dict) -> str:
        return (
            f"{label:<22}{m['frames']:>7}{m['exact_count_accuracy_percent']:>9.1f}"
            f"{m['mean_absolute_count_error']:>7.2f}{m['root_mean_square_count_error']:>7.2f}"
            f"{m['mean_count_bias']:>7.2f}{m['under_count_frames']:>7}{m['over_count_frames']:>6}"
        )

    print(f"\nYOLO vs AI Vision (confidence >= {min_confidence}):")
    print(header)
    for column, model in (("yolo_count", "YOLO"), ("vision_count", "AI Vision")):
        print(line(f"{model} all", evaluate_rows(rows, column)))
        for cam_id, metrics in evaluate_rows_by_camera(rows, column).items():
            print(line(f"{model} {cam_id}", metrics))

    print("\nAI Vision threshold sensitivity (all frames):")
    print(header)
    for threshold in SWEEP_THRESHOLDS:
        sweep_rows = [
            {**row, "vision_count": str(count_people(people[row["frame_id"]], threshold))} for row in rows
        ]
        print(line(f"confidence >= {threshold}", evaluate_rows(sweep_rows, "vision_count")))


if __name__ == "__main__":
    main()
