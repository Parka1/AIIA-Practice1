"""Advanced task 3: minimum resolution and JPEG compression that keep the detection.

    python -m advanced.evaluate_bandwidth            # YOLO only (local, free)
    python -m advanced.evaluate_bandwidth --vision   # also AI Vision (uses F0 quota once; cached)

Every variant is produced from the 20 frames of the serverless run (1920x1080,
already JPEG quality 85, so 85 is the highest quality that can be evaluated) and
evaluated against the manual ground-truth boxes of Part 3:

    resolution sweep: width 1920 / 1600 / 1280 / 960 / 640 px, quality 85
    quality sweep:    quality 85 / 70 / 50 / 30 / 15, width 1920 px

YOLO uses the Part 1 and 2 settings (YOLOv8n, 960-px input, confidence 0.25) on
the decoded variant, as the VM would receive it. AI Vision receives the encoded
bytes, as the function would. Its responses are cached in
report/data/advanced3_vision_detections.json, so re-running the script does not
call the service again.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from advanced.detection_metrics import scale_boxes, score_frame, summarize
from common.optimizations import encode_jpeg, resize_to_width

REPORT = Path("report")
FRAMES = REPORT / "part3_frames"
GROUND_TRUTH = REPORT / "data" / "part3_ground_truth_boxes.json"
VISION_CACHE = REPORT / "data" / "advanced3_vision_detections.json"
OUT_FRAMES_CSV = REPORT / "data" / "advanced3_bandwidth_frames.csv"
OUT_SUMMARY_CSV = REPORT / "data" / "advanced3_bandwidth_summary.csv"
FIGURE = REPORT / "images" / "Advanced_3" / "advanced3_bandwidth.png"

FULL_WIDTH = 1920
BASELINE_QUALITY = 85
WIDTHS = (1920, 1600, 1280, 960, 640)
QUALITIES = (85, 70, 50, 30, 15)
FRAMES_PER_SECOND = 1 / 5  # one frame every 5 s per camera
VISION_PAUSE_S = 3.5  # AI Vision F0 allows 20 calls per minute
YOLO_COLOR, VISION_COLOR = "#2864dc", "#d9741c"


def variants() -> list[tuple[str, str, int, int]]:
    """(sweep, name, width, quality); the baseline belongs to both sweeps."""
    out = [("resolution", f"w{w}_q{BASELINE_QUALITY}", w, BASELINE_QUALITY) for w in WIDTHS]
    out += [("quality", f"w{FULL_WIDTH}_q{q}", FULL_WIDTH, q) for q in QUALITIES]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vision", action="store_true", help="also evaluate AI Vision (cached calls)")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    frames = {stem: cv2.imread(str(FRAMES / f"{stem}.jpg")) for stem in sorted(ground_truth)}

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.predict(next(iter(frames.values())), imgsz=960, conf=0.25, classes=[0], verbose=False)  # warm-up

    vision = VisionCache(VISION_CACHE) if args.vision else None
    rows = []
    for sweep, name, width, quality in variants():
        for stem, frame in frames.items():
            image = encode_jpeg(resize_to_width(frame, width), quality)
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            factor = FULL_WIDTH / decoded.shape[1]

            start = time.perf_counter()
            result = model.predict(decoded, imgsz=960, conf=0.25, classes=[0], verbose=False)[0]
            yolo_ms = (time.perf_counter() - start) * 1000
            yolo_boxes = scale_boxes(result.boxes.xyxy.tolist(), factor)
            rows.append(frame_row(sweep, name, width, quality, stem, "yolo", image, yolo_boxes, ground_truth[stem], yolo_ms))

            if vision:
                boxes, vision_ms = vision.detect(f"{name}|{stem}", image)
                rows.append(frame_row(sweep, name, width, quality, stem, "ai_vision", image,
                                      scale_boxes(boxes, factor), ground_truth[stem], vision_ms))
        print(f"{name}: done")

    write_csv(OUT_FRAMES_CSV, [{k: v for k, v in r.items() if k != "score"} for r in rows])
    summary = summarize_rows(rows)
    write_csv(OUT_SUMMARY_CSV, summary)
    print_summary(summary)
    plot(summary)


class VisionCache:
    """AI Vision detections (confidence >= 0.5) per variant and frame, stored on disk."""

    def __init__(self, path: Path):
        from serverless.credentials import load_credentials

        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.endpoint, self.key = load_credentials("vision_endpoint", "vision_key")
        self._last_call = 0.0

    def detect(self, key: str, image: bytes) -> tuple[list, float]:
        if key not in self.data:
            from serverless.vision import analyze_people, confident_people

            time.sleep(max(0.0, VISION_PAUSE_S - (time.monotonic() - self._last_call)))
            analysis = analyze_people(image, self.endpoint, self.key)
            self._last_call = time.monotonic()
            self.data[key] = {
                "inference_ms": analysis.inference_ms,
                "boxes": [[p.x, p.y, p.x + p.w, p.y + p.h] for p in confident_people(analysis.people)],
            }
            self.path.write_text(json.dumps(self.data), encoding="utf-8")
        entry = self.data[key]
        return entry["boxes"], entry["inference_ms"]


def frame_row(sweep, name, width, quality, stem, model, image, boxes, truth, inference_ms) -> dict:
    score = score_frame(boxes, truth)
    return {
        "sweep": sweep, "variant": name, "width": width, "quality": quality,
        "frame_id": stem, "cam_id": stem.rsplit("_", 1)[0], "model": model,
        "frame_bytes": len(image), "people": score.people, "detected": score.detections,
        "true_positives": score.true_positives, "inference_ms": round(inference_ms, 1), "score": score,
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["sweep"], r["variant"], r["width"], r["quality"], r["model"]) for r in rows},
                  key=lambda k: (k[0], -k[2], -k[3], k[4]))
    for sweep, name, width, quality, model in keys:
        for cam in ("all", "santa_pola", "grazalema"):
            group = [r for r in rows if (r["sweep"], r["variant"], r["model"]) == (sweep, name, model)
                     and cam in ("all", r["cam_id"])]
            metrics = summarize([r["score"] for r in group])
            kib = mean(r["frame_bytes"] for r in group) / 1024
            out.append({
                "sweep": sweep, "variant": name, "width": width, "quality": quality, "model": model, "camera": cam,
                "frames": metrics["frames"], "kib_per_frame": round(kib, 1),
                "kbit_per_s": round(kib * 1024 * 8 * FRAMES_PER_SECOND / 1000, 1),
                "recall": round(metrics["recall"], 3), "precision": round(metrics["precision"], 3),
                "count_mae": round(metrics["count_mae"], 2), "count_bias": round(metrics["count_bias"], 2),
                "mean_detected": round(metrics["mean_detected"], 2),
                "inference_ms": round(mean(r["inference_ms"] for r in group), 1),
            })
    return out


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'sweep':<11}{'variant':<12}{'model':<10}{'KiB':>7}{'kbit/s':>8}{'recall':>8}{'prec':>7}"
          f"{'MAE':>7}{'det':>6}{'ms':>8}   recall SP / GZ")
    for row in summary:
        if row["camera"] != "all":
            continue
        per_cam = {r["camera"]: r["recall"] for r in summary
                   if (r["sweep"], r["variant"], r["model"]) == (row["sweep"], row["variant"], row["model"])}
        print(f"{row['sweep']:<11}{row['variant']:<12}{row['model']:<10}{row['kib_per_frame']:>7}{row['kbit_per_s']:>8}"
              f"{row['recall']:>8.3f}{row['precision']:>7.2f}{row['count_mae']:>7.2f}{row['mean_detected']:>6.1f}"
              f"{row['inference_ms']:>8.0f}   {per_cam['santa_pola']:.3f} / {per_cam['grazalema']:.3f}")


def plot(summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in summary if r["camera"] == "all"]
    models = [m for m in ("yolo", "ai_vision") if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sweep, label in ((axes[0], "resolution", "{} px"), (axes[1], "quality", "q{}")):
        for model in models:
            points = [r for r in rows if r["sweep"] == sweep and r["model"] == model]
            x = [r["kib_per_frame"] for r in points]
            y = [r["recall"] for r in points]
            color = YOLO_COLOR if model == "yolo" else VISION_COLOR
            ax.plot(x, y, "-o", color=color, lw=2, ms=7, label="YOLOv8n" if model == "yolo" else "AI Vision")
            if model == models[0]:
                for r in points:
                    value = r["width"] if sweep == "resolution" else r["quality"]
                    ax.annotate(label.format(value), (r["kib_per_frame"], r["recall"]), textcoords="offset points",
                                xytext=(0, -16), ha="center", fontsize=8, color="#444")
        ax.set_title("Resolution (JPEG quality 85)" if sweep == "resolution" else "JPEG quality (1920 px)", fontsize=11)
        ax.set_xlabel("KiB per frame")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall (people found)")
    axes[0].set_ylim(0, 0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"\nsaved {FIGURE}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
