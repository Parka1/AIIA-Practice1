"""Evaluate VM people counts against manually labelled frame counts."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _count_errors(
    rows: list[dict[str, str]], prediction_column: str = "yolo_count"
) -> list[int]:
    if not rows:
        raise ValueError("at least one annotated frame is required")

    frame_ids = set()
    errors = []
    for line_number, row in enumerate(rows, start=2):
        frame_id = row.get("frame_id", "").strip()
        if not frame_id:
            raise ValueError(f"row {line_number}: frame_id is required")
        if frame_id in frame_ids:
            raise ValueError(f"row {line_number}: duplicate frame_id {frame_id!r}")
        frame_ids.add(frame_id)
        camera_id = row.get("cam_id", "").strip()
        if not camera_id:
            raise ValueError(f"row {line_number}: cam_id is required")
        truth_text = row.get("ground_truth_count", "").strip()
        prediction_text = row.get(prediction_column, "").strip()
        if not truth_text:
            raise ValueError(
                f"row {line_number}: fill in ground_truth_count before evaluation"
            )
        try:
            truth = int(truth_text)
            prediction = int(prediction_text)
        except ValueError as error:
            raise ValueError(
                f"row {line_number}: counts must be non-negative integers"
            ) from error
        if truth < 0 or prediction < 0:
            raise ValueError(f"row {line_number}: counts must be non-negative integers")
        errors.append(prediction - truth)
    return errors


def _summarize_errors(errors: list[int]) -> dict[str, float | int]:
    absolute_errors = [abs(error) for error in errors]
    return {
        "frames": len(errors),
        "exact_count_accuracy_percent": 100 * sum(error == 0 for error in errors) / len(errors),
        "mean_absolute_count_error": sum(absolute_errors) / len(errors),
        "root_mean_square_count_error": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "mean_count_bias": sum(errors) / len(errors),
        "under_count_frames": sum(error < 0 for error in errors),
        "over_count_frames": sum(error > 0 for error in errors),
    }


def evaluate_rows(
    rows: list[dict[str, str]], prediction_column: str = "yolo_count"
) -> dict[str, float | int]:
    return _summarize_errors(_count_errors(rows, prediction_column))


def evaluate_rows_by_camera(
    rows: list[dict[str, str]], prediction_column: str = "yolo_count"
) -> dict[str, dict[str, float | int]]:
    _count_errors(rows, prediction_column)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["cam_id"].strip()].append(row)
    return {
        camera_id: _summarize_errors(_count_errors(camera_rows, prediction_column))
        for camera_id, camera_rows in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("report/data/part2_vm_ground_truth.csv"),
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print("Overall:")
    for metric, value in evaluate_rows(rows).items():
        print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
    print("By camera:")
    for camera_id, metrics in evaluate_rows_by_camera(rows).items():
        print(f"  {camera_id}:")
        for metric, value in metrics.items():
            print(
                f"    {metric}: {value:.3f}"
                if isinstance(value, float)
                else f"    {metric}: {value}"
            )


if __name__ == "__main__":
    main()
