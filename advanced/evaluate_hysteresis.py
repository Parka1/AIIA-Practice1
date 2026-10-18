"""Lighting decision with hysteresis (LightingController) on continuous sequences.

    python -m advanced.evaluate_hysteresis

For the Smart Lighting System the exact count does not matter, only whether the
zone is occupied. The YOLOv8n count of every frame of the sequences used for the
motion detector (report/data/advanced3_sequence_counts.csv) is fed to the
controller with several hold times:

    lights on %   fraction of frames with the lights on
    switches      number of on/off changes (flicker)
    dark while occupied   frames with the lights off although the zone is occupied;
                  only for the daytime sequences: in Santa Pola people were visible in
                  every inspected frame, and the daytime Grazalema frames (from Parts 2
                  and 3) all have manual labels with 2-16 people

Separate captures in the same folder are replayed as separate runs, starting
with the lights off (advanced.sequences.split_runs).
"""

import csv
from pathlib import Path

from advanced.sequences import sequence_dirs, split_runs
from common.optimizations import LightingController

COUNTS_CSV = Path("report/data/advanced3_sequence_counts.csv")
OUT_CSV = Path("report/data/advanced3_hysteresis.csv")
FIGURE = Path("report/images/Advanced_3/advanced3_hysteresis.png")
HOLDS = (1, 2, 3, 6, 12)  # frames; one frame every 5 s
ALWAYS_OCCUPIED = {("day", "santa_pola"), ("day", "grazalema")}
PLOT_HOLDS = (1, 3, 12)


def main() -> None:
    with COUNTS_CSV.open(encoding="utf-8") as f:
        counts = {r["frame_id"]: int(r["yolo_count"]) for r in csv.DictReader(f)}

    rows, timelines = [], {}
    for seq in sequence_dirs():
        period, cam = seq.parent.name, seq.name
        runs = [[counts[f.stem] for f in run] for run in split_runs(list(seq.glob("*.jpg")))]
        series = [n for run in runs for n in run]
        for hold in HOLDS:
            states, switches = [], 0
            for run in runs:
                controller = LightingController(hold_frames=hold)
                run_states = [controller.update(n) for n in run]
                switches += sum(a != b for a, b in zip(run_states, run_states[1:]))
                states += run_states
            occupied = (period, cam) in ALWAYS_OCCUPIED
            rows.append({
                "period": period, "camera": cam, "runs": len(runs), "hold_frames": hold, "hold_seconds": hold * 5,
                "frames": len(states), "lights_on_percent": round(100 * sum(states) / len(states), 1),
                "switches": switches,
                "dark_while_occupied": sum(not s for s in states) if occupied else "",
            })
            timelines[(period, cam, hold)] = states
        timelines[(period, cam, "counts")] = series
        timelines[(period, cam, "breaks")] = [sum(len(r) for r in runs[:i]) for i in range(1, len(runs))]

    write_csv(OUT_CSV, rows)
    print(f"{'sequence':<20}{'hold':>6}{'on %':>8}{'switches':>10}{'dark while occupied':>21}")
    for r in rows:
        print(f"{r['period'] + '/' + r['camera']:<20}{r['hold_seconds']:>5}s{r['lights_on_percent']:>8}"
              f"{r['switches']:>10}{str(r['dark_while_occupied']):>21}")
    plot(timelines)


def plot(timelines: dict) -> None:
    import matplotlib.pyplot as plt

    sequences = sorted({(k[0], k[1]) for k in timelines}, key=lambda k: (k[0], k[1] != "santa_pola"))
    fig, axes = plt.subplots(len(sequences), 1, figsize=(10, 2.1 * len(sequences)), sharex=True)
    for ax, (period, cam) in zip(axes, sequences):
        counts = timelines[(period, cam, "counts")]
        x = range(len(counts))
        ax.bar(x, counts, width=0.8, color="#9aa5b4", label="People detected (YOLOv8n)")
        top = max(max(counts), 1) + 0.6
        step = 0.75 * max(top, 4.6) / 4.6  # band spacing in data units, same size on every panel
        for i, hold in enumerate(PLOT_HOLDS):
            states = timelines[(period, cam, hold)]
            y = top + step * (i + 0.5)
            spans = runs(states)
            ax.broken_barh(spans, (y - 0.37 * step, 0.74 * step), color="#e0a10b")
            ax.text(-1.5, y, "raw" if hold == 1 else f"hold {hold * 5} s", ha="right", va="center", fontsize=8)
        for b in timelines[(period, cam, "breaks")]:
            ax.axvline(b - 0.5, color="#555", lw=1, ls="--")
            ax.text(b + 0.6, top - 0.3, "new capture", fontsize=7, color="#555", va="top")
        ax.set_ylim(0, top + step * (len(PLOT_HOLDS) + 0.3))
        ax.set_yticks(range(0, int(top) + 1, max(1, round(top / 4))))
        ax.set_title(f"{cam.replace('_', ' ').title()}, {period}", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.5, 119.5)
    axes[-1].set_xlabel("Frame (one every 5 s)")
    fig.text(0.01, 0.5, "People detected", rotation=90, va="center", fontsize=9)
    fig.tight_layout(rect=(0.06, 0, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"saved {FIGURE}")


def runs(states: list[bool]) -> list[tuple[float, float]]:
    """(start, length) of every run of frames with the lights on, for broken_barh."""
    spans, start = [], None
    for i, on in enumerate(states + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start - 0.5, i - start))
            start = None
    return spans


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
