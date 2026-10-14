"""Task 3.6: response time, detection and cost of the serverless run.

    python -m serverless.analyze_serverless --fetch   # download the run data and frames into report/
    python -m serverless.analyze_serverless --yolo    # count the same frames with YOLOv8 (Part 1/2 settings)
    python -m serverless.analyze_serverless           # print the tables and save the figures

--fetch copies data/results.csv and data/metrics.csv from the $web container, the
local upload log and the raw frames, so the analysis can be reproduced offline.

Response time is measured from the local capture time (frame_id) until the result
is stored by the function; like in Part 2, this assumes the local and Azure clocks
are synchronized. It is split into:
    upload    local PC -> raw-frames (upload log)
    trigger   blob stored -> function start: Event Grid delivery and instance start
              (start_delay_ms - upload_ms; also includes the ~30 ms JPEG encoding)
    vision    AI Vision call from the function
    other     drawing the boxes and storing the image and the CSV row
"""

import argparse
import csv
import shutil
from pathlib import Path
from statistics import mean, median

REPORT = Path("report")
DATA = REPORT / "data"
IMAGES = REPORT / "images"
FRAMES = REPORT / "part3_frames"
RESULTS_CSV = DATA / "part3_serverless_results.csv"
METRICS_CSV = DATA / "part3_serverless_metrics.csv"
UPLOADS_CSV = DATA / "part3_serverless_uploads.csv"
COUNTS_CSV = DATA / "part3_same_frame_counts.csv"
VISION_GT_CSV = DATA / "part3_vision_ground_truth.csv"
CAMERAS = ("santa_pola", "grazalema")

# Retail prices from the Azure Retail Prices API and the Azure Functions pricing page (October 2026).
PRICE = {
    "vision_call": 1.00 / 1000,  # AI Vision S1, Image Analysis Group 1, first 1M calls/month (France Central)
    "function_gb_s": 0.000034,  # Flex Consumption on-demand execution time (Belgium Central)
    "function_execution": 0.000005 / 10,  # Flex Consumption on-demand executions
    "function_free_gb_s": 100_000,  # monthly free grant
    "function_free_executions": 250_000,
    "storage_write": 0.054 / 10_000,  # Hot LRS write operations (Belgium Central)
    "storage_read": 0.0043 / 10_000,
    "storage_gb_month": 0.0196,
    "event_grid_operation": 0.60 / 1_000_000,  # after 100,000 free operations/month
    "event_grid_free": 100_000,
    "inter_region_gb": 0.02,  # function (Belgium Central) -> AI Vision (France Central)
    "vm_hour": 0.0114,  # Part 2 VM, Standard B2ts v2 (Table 2)
}
INSTANCE_GB = 0.5  # 512 MB instances
# Per frame: raw upload, annotated frame, results row, metrics row / blob read by the trigger.
STORAGE_WRITES_PER_FRAME = 4
STORAGE_READS_PER_FRAME = 1
FRAMES_PER_MONTH = 30 * 24 * 3600 // 5  # one camera, one frame every 5 s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="download the run data and frames into report/")
    parser.add_argument("--yolo", action="store_true", help="count people in the run frames with YOLOv8")
    args = parser.parse_args()

    if args.fetch:
        fetch()
    if args.yolo:
        count_with_yolo()

    metrics = read_csv(METRICS_CSV)
    uploads = {row["frame_id"]: row for row in read_csv(UPLOADS_CSV)}
    frames = [frame_timings(m, uploads[m["frame_id"]]) for m in metrics if m["status"] == "ok"]

    print_response_times(frames)
    print_capture(uploads)
    print_detection()
    print_costs(frames)
    plot_response_breakdown(frames)
    if COUNTS_CSV.exists():
        plot_same_frame_counts()


def fetch() -> None:
    from azure.storage.blob import BlobServiceClient

    from serverless.credentials import load_credentials

    (connection_string,) = load_credentials("storage_connection_string")
    service = BlobServiceClient.from_connection_string(connection_string)
    web, raw = service.get_container_client("$web"), service.get_container_client("raw-frames")
    DATA.mkdir(parents=True, exist_ok=True)
    for blob, path in (("data/results.csv", RESULTS_CSV), ("data/metrics.csv", METRICS_CSV)):
        path.write_bytes(web.get_blob_client(blob).download_blob().readall())
    shutil.copyfile("results/serverless_uploads.csv", UPLOADS_CSV)
    FRAMES.mkdir(exist_ok=True)
    for row in read_csv(METRICS_CSV):
        name = f"{row['frame_id']}.jpg"
        (FRAMES / name).write_bytes(raw.get_blob_client(name).download_blob().readall())
    print(f"saved {RESULTS_CSV}, {METRICS_CSV}, {UPLOADS_CSV} and the frames in {FRAMES}")


def count_with_yolo() -> None:
    """Same model and settings as Parts 1 and 2: YOLOv8n, 960 px, person class, confidence 0.25."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    previous = {row["frame_id"]: row for row in read_csv(COUNTS_CSV)} if COUNTS_CSV.exists() else {}
    rows = []
    for result in read_csv(RESULTS_CSV):
        frame = FRAMES / f"{result['frame_id']}.jpg"
        prediction = model.predict(str(frame), imgsz=960, conf=0.25, classes=[0], verbose=False)[0]
        rows.append({
            "frame_id": result["frame_id"],
            "cam_id": result["cam_id"],
            "frame_file": f"../part3_frames/{frame.name}",
            # Manual labels are kept if the file was already labelled.
            "ground_truth_count": previous.get(result["frame_id"], {}).get("ground_truth_count", ""),
            "vision_count": result["n_people"],
            "yolo_count": len(prediction.boxes),
        })
    rows.sort(key=lambda row: row["frame_id"])
    write_csv(COUNTS_CSV, rows)
    print(f"saved {COUNTS_CSV}")


def frame_timings(metric: dict, upload: dict) -> dict:
    start_delay, upload_ms = float(metric["start_delay_ms"]), float(upload["upload_ms"])
    response, vision = float(metric["response_time_ms"]), float(metric["inference_ms"])
    return {
        "frame_id": metric["frame_id"],
        "cam_id": metric["cam_id"],
        "cold_start": metric["cold_start"] == "1",
        "capture": float(upload["capture_ms"]),
        "upload": upload_ms,
        "trigger": start_delay - upload_ms,
        "vision": vision,
        "service": float(metric["service_ms"]),
        "other": response - start_delay - vision,
        "response": response,
        "function_ms": float(metric["function_ms"]),
        "frame_bytes": int(metric["frame_bytes"]),
    }


def print_response_times(frames: list[dict]) -> None:
    print("Response time (ms)                frames   mean  median    max | upload trigger vision  other")
    groups = [(f"{cam} warm", [f for f in frames if f["cam_id"] == cam and not f["cold_start"]]) for cam in CAMERAS]
    groups += [
        ("all warm", [f for f in frames if not f["cold_start"]]),
        ("cold start", [f for f in frames if f["cold_start"]]),
        ("all", frames),
    ]
    for label, group in groups:
        if not group:
            continue
        r = [f["response"] for f in group]
        print(
            f"  {label:<30}{len(group):>7}{mean(r):>7.0f}{median(r):>8.0f}{max(r):>7.0f} |"
            + "".join(f"{mean(f[k] for f in group):>7.0f}" for k in ("upload", "trigger", "vision", "other"))
        )
    warm = [f for f in frames if not f["cold_start"]]
    print(f"  AI Vision service time (warm): mean {mean(f['service'] for f in warm):.0f} ms")
    print(f"  response - upload (warm): mean {mean(f['response'] - f['upload'] for f in warm):.0f} ms")
    print(f"  JPEG size: santa_pola {mean(f['frame_bytes'] for f in frames if f['cam_id'] == 'santa_pola') / 1024:.1f} KiB, "
          f"grazalema {mean(f['frame_bytes'] for f in frames if f['cam_id'] == 'grazalema') / 1024:.1f} KiB")
    # Part 2 VM, Table 2 (Santa Pola, ten frames, 12 October): mean 792.8 ms, median 732.0 ms, max 990.3 ms,
    # inference 413.0 ms. Quoted from the report, since part2_vm_results.csv also holds other test batches.


def print_capture(uploads: dict) -> None:
    print("\nLocal capture (ms)                 frames   mean  median    max   interval s")
    for cam in CAMERAS:
        rows = sorted((r for r in uploads.values() if r["cam_id"] == cam), key=lambda r: r["captured_at"])
        capture = [float(r["capture_ms"]) for r in rows]
        times = [load_time(r["captured_at"]) for r in rows]
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        print(f"  {cam:<30}{len(rows):>7}{mean(capture):>7.0f}{median(capture):>8.0f}{max(capture):>7.0f}"
              f"   {mean(gaps):.1f} mean, {max(gaps):.1f} max")


def print_detection() -> None:
    from local_vm.evaluate_vm_counts import evaluate_rows, evaluate_rows_by_camera

    def line(label, m):
        print(f"  {label:<22}{m['frames']:>7}{m['exact_count_accuracy_percent']:>9.1f}"
              f"{m['mean_absolute_count_error']:>7.2f}{m['root_mean_square_count_error']:>7.2f}"
              f"{m['mean_count_bias']:>7.2f}{m['under_count_frames']:>7}{m['over_count_frames']:>6}")

    header = f"  {'':<22}{'frames':>7}{'exact %':>9}{'MAE':>7}{'RMSE':>7}{'bias':>7}{'under':>7}{'over':>6}"
    rows = read_csv(VISION_GT_CSV)
    print("\nDetection vs manual labels (20 labelled Part 2 frames)")
    print(header)
    for column, model in (("yolo_count", "YOLO"), ("vision_count", "AI Vision")):
        line(f"{model} all", evaluate_rows(rows, column))
        for cam, m in evaluate_rows_by_camera(rows, column).items():
            line(f"{model} {cam}", m)

    if not COUNTS_CSV.exists():
        return
    rows = read_csv(COUNTS_CSV)
    print("\nSame-frame counts in the serverless run (AI Vision conf >= 0.5, YOLO conf >= 0.25)")
    print("  camera         frames  vision mean  yolo mean  agree  vision>yolo  vision<yolo")
    for cam in CAMERAS:
        group = [r for r in rows if r["cam_id"] == cam]
        v = [int(r["vision_count"]) for r in group]
        y = [int(r["yolo_count"]) for r in group]
        print(f"  {cam:<15}{len(group):>6}{mean(v):>13.1f}{mean(y):>11.1f}{sum(a == b for a, b in zip(v, y)):>7}"
              f"{sum(a > b for a, b in zip(v, y)):>13}{sum(a < b for a, b in zip(v, y)):>13}")
    if all(r["ground_truth_count"] for r in rows):
        print("Same frames vs manual labels")
        print(header)
        for column, model in (("yolo_count", "YOLO"), ("vision_count", "AI Vision")):
            line(f"{model} all", evaluate_rows(rows, column))
            for cam, m in evaluate_rows_by_camera(rows, column).items():
                line(f"{model} {cam}", m)
    else:
        print(f"  (fill ground_truth_count in {COUNTS_CSV} to compute accuracy on these frames)")


def print_costs(frames: list[dict]) -> None:
    n = len(frames)
    gb = mean(f["frame_bytes"] for f in frames) / 1024**3
    # Flex Consumption bills at least 1 s per execution, then rounds up to 100 ms.
    billed_s = mean(max(1.0, -(-f["function_ms"] // 100) / 10) for f in frames)
    per_frame = {
        "AI Vision (S1)": PRICE["vision_call"],
        "Functions execution time": billed_s * INSTANCE_GB * PRICE["function_gb_s"],
        "Functions executions": PRICE["function_execution"],
        "Storage operations": STORAGE_WRITES_PER_FRAME * PRICE["storage_write"] + STORAGE_READS_PER_FRAME * PRICE["storage_read"],
        "Event Grid": PRICE["event_grid_operation"],
        "Inter-region transfer": gb * PRICE["inter_region_gb"],
    }
    total = sum(per_frame.values())
    print(f"\nCost per frame (no free grants), billed function time {billed_s:.1f} s")
    for item, cost in per_frame.items():
        print(f"  {item:<26}${cost:.7f}  ({100 * cost / total:.1f}%)")
    print(f"  {'total':<26}${total:.7f} -> ${total * 1000:.3f} per 1,000 frames; this run ({n} frames) ${total * n:.4f}")
    print(f"  this run with AI Vision F0 and the free grants: $0 (AI Vision free tier, within Functions/Event Grid grants)")

    m = FRAMES_PER_MONTH
    monthly = {
        "AI Vision (S1)": m * PRICE["vision_call"],
        "Functions": max(0, m * billed_s * INSTANCE_GB - PRICE["function_free_gb_s"]) * PRICE["function_gb_s"]
        + max(0, m - PRICE["function_free_executions"]) * PRICE["function_execution"],
        "Storage operations": m * per_frame["Storage operations"],
        "Event Grid": max(0, m - PRICE["event_grid_free"]) * PRICE["event_grid_operation"],
        "Inter-region transfer": m * per_frame["Inter-region transfer"],
    }
    print(f"\nOne camera, one frame every 5 s for 30 days ({m:,} frames), with free grants")
    for item, cost in monthly.items():
        print(f"  {item:<26}${cost:,.2f}")
    print(f"  {'total':<26}${sum(monthly.values()):,.2f}")
    print(f"  stored images if kept (raw + annotated): {m * gb * 2:,.0f} GB -> ${m * gb * 2 * PRICE['storage_gb_month']:,.2f}/month")
    print(f"  AI Vision F0 allows 5,000 calls/month = {5000 * 5 / 3600:.1f} h of one camera")
    print(f"  Part 2 VM for comparison: ${PRICE['vm_hour'] * 24 * 30:,.2f}/month (compute only)")


def plot_response_breakdown(frames: list[dict]) -> None:
    import matplotlib.pyplot as plt

    frames = sorted(frames, key=lambda f: f["frame_id"])
    frames = [f for cam in CAMERAS for f in frames if f["cam_id"] == cam]
    parts = [("upload", "Upload to Blob Storage", "#9fb3cf"), ("trigger", "Event Grid + function start", "#5b7aa6"),
             ("vision", "AI Vision call", "#d9741c"), ("other", "Draw + store", "#2864dc")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = range(len(frames))
    bottom = [0.0] * len(frames)
    for key, label, color in parts:
        values = [f[key] / 1000 for f in frames]
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.7)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(list(x), [f"{'SP' if f['cam_id'] == 'santa_pola' else 'GZ'}{i + 1}{'*' if f['cold_start'] else ''}"
                            for i, f in enumerate(frames)], fontsize=8)
    ax.axvline(len([f for f in frames if f["cam_id"] == "santa_pola"]) - 0.5, color="#888", lw=0.8, ls="--")
    ax.set_ylabel("Seconds from capture")
    ax.set_title("Serverless response time per frame (SP = Santa Pola, GZ = Grazalema, * = cold start)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / "part3_response_breakdown.png", dpi=150)
    print(f"\nsaved {IMAGES / 'part3_response_breakdown.png'}")


def plot_same_frame_counts() -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(COUNTS_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, cam in zip(axes, CAMERAS):
        group = [r for r in rows if r["cam_id"] == cam]
        x = range(1, len(group) + 1)
        ax.bar([i - 0.2 for i in x], [int(r["vision_count"]) for r in group], 0.4, color="#d9741c", label="AI Vision")
        ax.bar([i + 0.2 for i in x], [int(r["yolo_count"]) for r in group], 0.4, color="#2864dc", label="YOLOv8n")
        if all(r["ground_truth_count"] for r in group):
            ax.plot(list(x), [int(r["ground_truth_count"]) for r in group], "k_", ms=14, mew=2, label="Manual count")
        ax.set_title(cam.replace("_", " ").title())
        ax.set_xlabel("Frame")
        ax.set_xticks(list(x))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("People detected")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "part3_same_frame_counts.png", dpi=150)
    print(f"saved {IMAGES / 'part3_same_frame_counts.png'}")


def load_time(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
