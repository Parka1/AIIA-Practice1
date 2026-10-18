"""Advanced task 3: skip frames without motion before sending them (MotionDetector).

    python -m advanced.evaluate_motion

Uses continuous local captures (one frame every 5 s, nothing uploaded) in
report/advanced3_sequences/<day|night>/<camera>/. A folder may hold several
captures; they are replayed as separate runs (advanced.sequences.split_runs). Every frame is first counted with
YOLOv8n (Part 1/2 settings); that count is taken as what the system would report if
the frame were processed. Then the MotionDetector is replayed over the sequence for
several thresholds, with and without the camera's motion region:

    sent             frames that would be sent (AI Vision calls / uploads)
    stale            frames whose reported count (the one of the last frame sent)
                     differs from the count of the frame itself
    missed occupied  frames with people detected but reported as empty, which for
                     the lighting system means lights off with people present

Frames are always sent at least once a minute (max_skipped = 12).
"""

import csv
from pathlib import Path

import cv2

from advanced.sequences import CAMERA_COLORS, PERIOD_STYLES, sequence_dirs, split_runs
from common.optimizations import MOTION_REGIONS, MotionDetector

COUNTS_CSV = Path("report/data/advanced3_sequence_counts.csv")
OUT_CSV = Path("report/data/advanced3_motion.csv")
FIGURE = Path("report/images/Advanced_3/advanced3_motion.png")
THRESHOLDS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def main() -> None:
    sequences = sequence_dirs()
    counts = yolo_counts(sequences)

    rows = []
    for seq in sequences:
        period, cam = seq.parent.name, seq.name
        runs = [[(cv2.imread(str(f)), counts[f.stem]) for f in run] for run in split_runs(list(seq.glob("*.jpg")))]
        regions = [("full frame", None)]
        if cam in MOTION_REGIONS:
            regions.append(("motion region", MOTION_REGIONS[cam]))
        for region_name, region in regions:
            for threshold in THRESHOLDS:
                rows.append(replay(period, cam, region_name, region, threshold, runs))

    write_csv(OUT_CSV, rows)
    print(f"{'sequence':<20}{'region':<15}{'thr':>7}{'frames':>7}{'sent %':>8}{'stale %':>9}{'missed occ.':>12}")
    for r in rows:
        print(f"{r['period'] + '/' + r['camera']:<20}{r['region']:<15}{r['threshold']:>7}{r['frames']:>7}"
              f"{r['sent_percent']:>8.1f}{r['stale_percent']:>9.1f}{r['missed_occupied']:>12}")
    plot(rows)


def replay(period, cam, region_name, region, threshold, runs) -> dict:
    sent = stale = missed = 0
    for run in runs:
        detector = MotionDetector(min_changed=threshold, region=region)
        reported = None
        for image, count in run:
            if detector.should_send(image):
                sent += 1
                reported = count
            stale += reported != count
            missed += count > 0 and reported == 0
    actual = [count for run in runs for _, count in run]
    n = len(actual)
    return {
        "period": period, "camera": cam, "region": region_name, "threshold": threshold,
        "runs": len(runs), "frames": n,
        "sent": sent, "sent_percent": round(100 * sent / n, 1),
        "stale": stale, "stale_percent": round(100 * stale / n, 1),
        "missed_occupied": missed, "occupied_frames": sum(c > 0 for c in actual),
    }


def yolo_counts(sequences: list[Path]) -> dict[str, int]:
    """YOLOv8n person count per frame, cached in COUNTS_CSV."""
    counts = {}
    if COUNTS_CSV.exists():
        with COUNTS_CSV.open(encoding="utf-8") as f:
            counts = {r["frame_id"]: int(r["yolo_count"]) for r in csv.DictReader(f)}
    missing = [f for seq in sequences for f in sorted(seq.glob("*.jpg")) if f.stem not in counts]
    if missing:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        for f in missing:
            counts[f.stem] = len(model.predict(str(f), imgsz=960, conf=0.25, classes=[0], verbose=False)[0].boxes)
        write_csv(COUNTS_CSV, [{"frame_id": k, "yolo_count": v} for k, v in sorted(counts.items())])
    return counts


def plot(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    # Full frame only: the motion-region variants are in the CSV and discussed in the report.
    series = sorted({(r["camera"], r["period"]) for r in rows}, key=lambda k: (k[0] != "santa_pola", k[1]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    for cam, period in series:
        points = [r for r in rows if (r["camera"], r["period"], r["region"]) == (cam, period, "full frame")]
        label = f"{cam.replace('_', ' ').title()}, {period}"
        style, color = PERIOD_STYLES[period], CAMERA_COLORS[cam]
        x = [r["threshold"] * 100 for r in points]
        axes[0].plot(x, [r["sent_percent"] for r in points], style, marker="o", color=color, lw=2, ms=6, label=label)
        axes[1].plot(x, [r["stale_percent"] for r in points], style, marker="o", color=color, lw=2, ms=6, label=label)
    for ax, title in ((axes[0], "Frames sent (AI Vision calls)"), (axes[1], "Frames with an outdated count")):
        ax.set_xscale("log")
        ax.set_xticks([t * 100 for t in THRESHOLDS], [f"{t * 100:g}" for t in THRESHOLDS])
        ax.minorticks_off()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Motion threshold (% of pixels changed)")
        ax.set_ylabel("% of frames")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"saved {FIGURE}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
