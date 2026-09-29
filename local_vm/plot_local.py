"""Plot the local person detections stored in the common results CSV."""

import argparse
from pathlib import Path

from common.results import load_results


def create_plot(csv_path: Path, output_path: Path) -> None:
    """Create a time series of the person count for each processed frame."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    results = load_results(csv_path)
    if results.empty:
        raise ValueError(f"no detection results found in {csv_path}")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(results["timestamp"], results["n_people"], marker="o", linewidth=1.5)
    axis.set(
        title="People detected per frame - local YOLOv8",
        xlabel="Capture time",
        ylabel="Number of people",
    )
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    timezone = results["timestamp"].dt.tz
    axis.xaxis.set_major_locator(
        mdates.SecondLocator(bysecond=range(0, 60, 5), tz=timezone)
    )
    tick_positions = list(axis.get_xticks())
    last_frame = results["timestamp"].iloc[-1].to_pydatetime()
    last_tick = mdates.date2num(last_frame)
    tick_positions.append(last_tick)
    tick_positions.sort()
    labels = []
    for tick in tick_positions:
        tick_time = mdates.num2date(tick, tz=timezone)
        if abs(tick - last_tick) < 0.001 / 86400:
            labels.append(tick_time.strftime("%H:%M:%S.%f")[:-3])
        else:
            labels.append(tick_time.strftime("%H:%M:%S"))
    axis.set_xticks(tick_positions, labels)
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local YOLOv8 people detections over time.")
    parser.add_argument("--csv", type=Path, default=Path("results/local.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/local_people_over_time.png"))
    args = parser.parse_args()
    create_plot(args.csv, args.out)


if __name__ == "__main__":
    main()
