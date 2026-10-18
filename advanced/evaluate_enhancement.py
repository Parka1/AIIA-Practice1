"""Shadow enhancement (CLAHE / gamma) before detection.

    python -m advanced.evaluate_enhancement            # YOLO only
    python -m advanced.evaluate_enhancement --vision   # also AI Vision (cached)

Many of the people missed in Parts 2 and 3 sit in the shade of awnings and
umbrellas. Each enhancement is applied to the 20 frames with manual boxes
(1920 px, encoded again at quality 85) and both detectors are scored as in
advanced.evaluate_bandwidth. The figure also shows a shaded crop before and after.
"""

import argparse
import json

import cv2
import numpy as np

from advanced.detection_metrics import scale_boxes
from advanced.evaluate_bandwidth import (
    FRAMES, GROUND_TRUTH, REPORT, VISION_CACHE, VISION_COLOR, YOLO_COLOR, VisionCache, frame_row, summarize_rows,
    write_csv,
)
from common.optimizations import encode_jpeg, enhance_clahe, enhance_gamma

OUT_SUMMARY_CSV = REPORT / "data" / "advanced3_enhancement_summary.csv"
FIGURE = REPORT / "images" / "Advanced_3" / "advanced3_enhancement.png"
EXAMPLE_FIGURE = REPORT / "images" / "Advanced_3" / "advanced3_enhancement_example.jpg"
# name -> (label, function); "w1920_q85" is the baseline already cached by evaluate_bandwidth.
VARIANTS = {
    "w1920_q85": ("None", lambda f: f),
    "clahe2": ("CLAHE 2", lambda f: enhance_clahe(f, 2.0)),
    "clahe4": ("CLAHE 4", lambda f: enhance_clahe(f, 4.0)),
    "gamma0.6": ("Gamma 0.6", lambda f: enhance_gamma(f, 0.6)),
}
# Shaded areas shown in the example figure: (frame, x1, y1, x2, y2).
EXAMPLES = [
    ("grazalema_20261014T171015027", 1000, 400, 1260, 560),
    ("santa_pola_20261014T170850943", 990, 450, 1250, 610),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    frames = {stem: cv2.imread(str(FRAMES / f"{stem}.jpg")) for stem in sorted(ground_truth)}

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.predict(next(iter(frames.values())), imgsz=960, conf=0.25, classes=[0], verbose=False)
    vision = VisionCache(VISION_CACHE) if args.vision else None

    rows = []
    for name, (_, enhance) in VARIANTS.items():
        for stem, frame in frames.items():
            image = encode_jpeg(enhance(frame), 85)
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            result = model.predict(decoded, imgsz=960, conf=0.25, classes=[0], verbose=False)[0]
            rows.append(frame_row("enhancement", name, 1920, 85, stem, "yolo", image,
                                  result.boxes.xyxy.tolist(), ground_truth[stem], 0.0))
            if vision:
                boxes, ms = vision.detect(f"{name}|{stem}", image)
                rows.append(frame_row("enhancement", name, 1920, 85, stem, "ai_vision", image,
                                      scale_boxes(boxes, 1.0), ground_truth[stem], ms))
        print(f"{name}: done")

    summary = summarize_rows(rows)
    write_csv(OUT_SUMMARY_CSV, summary)
    print(f"\n{'variant':<11}{'model':<10}{'camera':<11}{'recall':>8}{'prec':>7}{'MAE':>7}{'det':>6}{'KiB':>7}")
    for r in summary:
        print(f"{r['variant']:<11}{r['model']:<10}{r['camera']:<11}{r['recall']:>8.3f}{r['precision']:>7.2f}"
              f"{r['count_mae']:>7.2f}{r['mean_detected']:>6.1f}{r['kib_per_frame']:>7}")
    plot(summary)
    example(frames)


def plot(summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    names = list(VARIANTS)
    models = [m for m in ("yolo", "ai_vision") if any(r["model"] == m for r in summary)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, cam, title in ((axes[0], "grazalema", "Grazalema"), (axes[1], "santa_pola", "Santa Pola")):
        width = 0.38
        for k, model in enumerate(models):
            values = [next(r["recall"] for r in summary if (r["variant"], r["model"], r["camera"]) == (n, model, cam))
                      for n in names]
            x = [i + (k - (len(models) - 1) / 2) * width for i in range(len(names))]
            ax.bar(x, values, width * 0.92, color=YOLO_COLOR if model == "yolo" else VISION_COLOR,
                   label="YOLOv8n" if model == "yolo" else "AI Vision")
            for xi, v in zip(x, values):
                ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(range(len(names)), [VARIANTS[n][0] for n in names])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall (people found)")
    axes[0].set_ylim(0, 0.55)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"saved {FIGURE}")


def example(frames: dict) -> None:
    """Shaded crops: original, CLAHE 2 and gamma 0.6 side by side."""
    rows = []
    for stem, x1, y1, x2, y2 in EXAMPLES:
        frame = frames[stem]
        crops = [f(frame)[y1:y2, x1:x2] for f in (lambda f: f, VARIANTS["clahe2"][1], VARIANTS["gamma0.6"][1])]
        crops = [cv2.resize(c, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC) for c in crops]
        gap = np.full((crops[0].shape[0], 12, 3), 255, np.uint8)
        rows.append(np.hstack([crops[0], gap, crops[1], gap, crops[2]]))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 12, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(str(EXAMPLE_FIGURE), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"saved {EXAMPLE_FIGURE}")


if __name__ == "__main__":
    main()
