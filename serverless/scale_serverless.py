"""Task 4.3: how many simultaneous cameras the serverless pipeline can handle.

    python -m serverless.scale_serverless             # run the staircase, then fetch and analyze it
    python -m serverless.scale_serverless --analyze   # only fetch and analyze a finished run

The staircase uses the five cadence-qualified cameras of Task 4.1, added in the
same order as in the VM test (Section 6.1): stage n sends --count frames from
each of the first n cameras, one every 5 s, with persistent HLS streams. After
each stage the script waits until the function has recorded every frame of it,
and then one more minute, so the next stage starts with an empty AI Vision
F0 rate window (20 calls/min).

Outputs, for the report:
    report/data/part4_serverless_uploads.csv   upload log with the stage of every frame
    report/data/part4_serverless_metrics.csv   function metrics of those frames ($web/data/metrics.csv)
    report/data/part4_serverless_results.csv   their rows in the common results format
    report/data/part4_serverless_counts.csv    AI Vision and YOLOv8n counts on the same frames
    report/data/part4_serverless_summary.csv   one row per stage (the tables printed below)
    report/part4_serverless_frames/            the uploaded JPEGs
    report/images/Section_4/part4_serverless_scaling.png

Response time is measured as in Part 3 (capture -> result stored, synchronized
clocks). The AI Vision time is split into the successful call and the time
spent waiting after HTTP 429 responses (function time - call - draw - store).
Costs use the Part 3 prices (serverless/analyze_serverless.py).
"""

import argparse
import csv
import io
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np

from serverless.analyze_serverless import FRAMES_PER_MONTH, PRICE, frame_costs, monthly_costs
from serverless.upload_frames import VISION_CALLS_PER_MINUTE, raw_container, upload_cameras

# Task 4.1 cadence-qualified cameras, in the order of the VM staircase (Table 10).
CAMS = ("santa_pola", "benidorm_poniente", "madrid_alcala", "benalmadena", "calafell")
INTERVAL_S = 5.0
VISION_S1_CALLS_PER_MINUTE = 10 * 60  # S1 default rate limit: 10 transactions per second
VISION_F0_CALLS_PER_MONTH = 5000
VM_MONTH = PRICE["vm_hour"] * 24 * 30
WAIT_TIMEOUT_S = 300
PAUSE_S = 60

REPORT = Path("report")
DATA = REPORT / "data"
UPLOADS_CSV = DATA / "part4_serverless_uploads.csv"
METRICS_CSV = DATA / "part4_serverless_metrics.csv"
RESULTS_CSV = DATA / "part4_serverless_results.csv"
COUNTS_CSV = DATA / "part4_serverless_counts.csv"
SUMMARY_CSV = DATA / "part4_serverless_summary.csv"
FRAMES = REPORT / "part4_serverless_frames"
FIGURE = REPORT / "images" / "Section_4" / "part4_serverless_scaling.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analyze", action="store_true", help="skip the run: fetch and analyze the logged frames")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="cameras per stage")
    parser.add_argument("--count", type=int, default=10, help="frames per camera and stage (default: 10)")
    parser.add_argument("--overwrite", action="store_true", help="discard the log and frames of a previous run")
    parser.add_argument("--no-yolo", action="store_true", help="do not count the frames with YOLOv8n")
    args = parser.parse_args()
    if any(not 1 <= n <= len(CAMS) for n in args.stages):
        parser.error(f"stages must be between 1 and {len(CAMS)} cameras")

    if not args.analyze:
        if UPLOADS_CSV.exists() and not args.overwrite:
            parser.error(f"{UPLOADS_CSV} already exists: use --analyze, or --overwrite to start again")
        run(args.stages, args.count)
    fetch()
    if not args.no_yolo:
        count_with_yolo()
    analyze()


# --- Run -------------------------------------------------------------------------------------------

def run(stages: list[int], count: int) -> None:
    UPLOADS_CSV.unlink(missing_ok=True)
    for old in FRAMES.glob("*.jpg"):
        old.unlink()
    container = raw_container()
    web = web_container()
    calls = count * sum(stages)
    print(f"Staircase {stages}: {calls} frames = {calls} AI Vision calls "
          f"(F0 quota: {VISION_F0_CALLS_PER_MONTH} per month)")
    for i, n in enumerate(stages):
        cams = list(CAMS[:n])
        print(f"\n=== Stage {n}: {', '.join(cams)} ({n * 60 / INTERVAL_S:.0f} frames/min) ===")
        upload_cameras(
            container, cams, interval=INTERVAL_S, count=count, log=UPLOADS_CSV, stage=str(n),
            persistent_streams=True, save_frames_dir=FRAMES,
        )
        frame_ids = {r["frame_id"] for r in read_csv(UPLOADS_CSV) if r["stage"] == str(n)}
        wait_for_function(web, frame_ids)
        if i < len(stages) - 1:
            print(f"pausing {PAUSE_S} s so the AI Vision rate window is empty")
            time.sleep(PAUSE_S)


def web_container():
    from azure.storage.blob import BlobServiceClient

    from serverless.credentials import load_credentials

    (connection_string,) = load_credentials("storage_connection_string")
    return BlobServiceClient.from_connection_string(connection_string).get_container_client("$web")


def download_csv(web, blob: str) -> list[dict]:
    text = web.get_blob_client(blob).download_blob().readall().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def wait_for_function(web, frame_ids: set[str]) -> None:
    """Wait until the function has written a metrics row (ok or error) for every frame."""
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while True:
        done = {r["frame_id"] for r in download_csv(web, "data/metrics.csv")} & frame_ids
        print(f"function finished {len(done)}/{len(frame_ids)} frames")
        if len(done) == len(frame_ids):
            return
        if time.monotonic() > deadline:
            print(f"timeout: {len(frame_ids) - len(done)} frames were never processed")
            return
        time.sleep(10)


# --- Data ------------------------------------------------------------------------------------------

def fetch() -> None:
    """Copy the function's rows for the logged frames from $web into report/data."""
    frame_ids = {r["frame_id"] for r in read_csv(UPLOADS_CSV)}
    web = web_container()
    for blob, path in (("data/metrics.csv", METRICS_CSV), ("data/results.csv", RESULTS_CSV)):
        rows = [r for r in download_csv(web, blob) if r["frame_id"] in frame_ids]
        write_csv(path, rows)
    print(f"saved {METRICS_CSV} and {RESULTS_CSV}")


def count_with_yolo() -> None:
    """YOLOv8n with the Part 1 and 2 settings (960 px, person class, confidence 0.25) on the same frames."""
    metrics = {r["frame_id"]: r for r in read_csv(METRICS_CSV) if r["status"] == "ok"}
    done = {r["frame_id"]: r for r in read_csv(COUNTS_CSV)} if COUNTS_CSV.exists() else {}
    missing = [f for f in metrics if f not in done]
    if missing:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        for frame_id in missing:
            prediction = model.predict(str(FRAMES / f"{frame_id}.jpg"), imgsz=960, conf=0.25, classes=[0],
                                       verbose=False)[0]
            done[frame_id] = {"frame_id": frame_id, "yolo_count": len(prediction.boxes)}
    stage = {r["frame_id"]: r["stage"] for r in read_csv(UPLOADS_CSV)}
    rows = [{
        "frame_id": f, "cam_id": m["cam_id"], "stage": stage[f],
        "vision_count": m["n_people"], "yolo_count": done[f]["yolo_count"],
    } for f, m in sorted(metrics.items())]
    write_csv(COUNTS_CSV, rows)
    print(f"saved {COUNTS_CSV}")


# --- Analysis --------------------------------------------------------------------------------------

def analyze() -> None:
    uploads = read_csv(UPLOADS_CSV)
    metrics = {r["frame_id"]: r for r in read_csv(METRICS_CSV)}
    counts = {r["frame_id"]: r for r in read_csv(COUNTS_CSV)} if COUNTS_CSV.exists() else {}
    stages = sorted({int(r["stage"]) for r in uploads})
    summary = [stage_summary(n, [r for r in uploads if r["stage"] == str(n)], metrics, counts) for n in stages]
    write_csv(SUMMARY_CSV, summary)

    print("\nLoad and AI Vision rate limit (F0: 20 calls/min)")
    print("  cams  uploaded  calls/min  max gap s |   ok  ok after 429  failed 429  lost |  ok %")
    for s in summary:
        print(f"  {s['cameras']:>4}{s['uploaded']:>10}{s['offered_calls_per_min']:>11}{s['max_capture_gap_s']:>11} |"
              f"{s['ok']:>5}{s['ok_after_429']:>13}{s['failed_429']:>12}{s['not_processed']:>6} |{s['ok_percent']:>6}")

    print("\nResponse time of the processed frames (s, capture -> result stored)")
    print("  cams   mean    P95    max | upload trigger  AI Vision  429 wait | cold starts")
    for s in summary:
        print(f"  {s['cameras']:>4}{s['response_mean_s']:>7}{s['response_p95_s']:>7}{s['response_max_s']:>7} |"
              f"{s['upload_mean_s']:>7}{s['trigger_mean_s']:>8}{s['vision_call_mean_s']:>11}{s['retry_wait_mean_s']:>10} |"
              f"{s['cold_starts']:>12}")

    print("\nDetection: people per processed frame (no manual labels for the new cameras)")
    print("  cams  AI Vision  YOLOv8n  same count")
    for s in summary:
        print(f"  {s['cameras']:>4}{s['vision_mean']:>11}{s['yolo_mean']:>9}{s['same_count_percent']:>11}")
    print_per_camera(metrics, counts)

    print("\nCost (Part 3 prices; the run itself used AI Vision F0 and the free grants: $0)")
    for s in summary:
        print(f"  stage {s['cameras']}: {s['executions']} executions, ${s['stage_cost_s1_usd']} at S1 prices")
    print_limits_and_monthly(metrics)
    plot(summary)


def stage_summary(n: int, uploads: list[dict], metrics: dict, counts: dict) -> dict:
    rows = [(u, metrics.get(u["frame_id"])) for u in uploads]
    executed = [(u, m) for u, m in rows if m]
    ok = [(u, m) for u, m in executed if m["status"] == "ok"]
    failed = [m for _, m in executed if m["status"] != "ok"]
    times = sorted(load_time(u["captured_at"]) for u in uploads)
    duration_s = (times[-1] - times[0]).total_seconds() + INTERVAL_S
    gaps = []
    for cam in {u["cam_id"] for u in uploads}:
        cam_times = sorted(load_time(u["captured_at"]) for u in uploads if u["cam_id"] == cam)
        gaps += [(b - a).total_seconds() for a, b in zip(cam_times, cam_times[1:])]

    def stage_mean(values) -> float | str:
        values = list(values)
        return round(mean(values), 2) if values else ""

    response = [num(m["response_time_ms"]) / 1000 for _, m in ok]
    with_counts = [counts[m["frame_id"]] for _, m in ok if m["frame_id"] in counts]
    cost_frames = [{"function_ms": num(m["function_ms"]), "frame_bytes": int(m["frame_bytes"])} for _, m in executed]
    stage_cost = ""
    if cost_frames:
        per_frame, _ = frame_costs(cost_frames)
        not_vision = sum(per_frame.values()) - per_frame["AI Vision (S1)"]
        stage_cost = round(len(executed) * not_vision + len(ok) * per_frame["AI Vision (S1)"], 4)
    return {
        "cameras": n,
        "uploaded": len(uploads),
        "duration_s": round(duration_s, 1),
        "offered_calls_per_min": round(len(uploads) / duration_s * 60, 1),
        "max_capture_gap_s": round(max(gaps), 2) if gaps else "",
        "executions": len(executed),
        "ok": len(ok),
        "ok_after_429": sum(int(m["vision_retries"] or 0) > 0 for _, m in ok),
        "failed_429": sum("429" in m["status"] for m in failed),
        "failed_other": sum("429" not in m["status"] for m in failed),
        "not_processed": len(uploads) - len(executed),
        "ok_percent": round(100 * len(ok) / len(uploads), 1),
        "retries": sum(int(m["vision_retries"] or 0) for _, m in ok),
        "response_mean_s": stage_mean(response),
        "response_p95_s": round(float(np.percentile(response, 95)), 2) if response else "",
        "response_max_s": round(max(response), 2) if response else "",
        "upload_mean_s": stage_mean(num(u["upload_ms"]) / 1000 for u, _ in ok),
        "trigger_mean_s": stage_mean((num(m["start_delay_ms"]) - num(u["upload_ms"])) / 1000 for u, m in ok),
        "vision_call_mean_s": stage_mean(num(m["inference_ms"]) / 1000 for _, m in ok),
        "retry_wait_mean_s": stage_mean(retry_wait_ms(m) / 1000 for _, m in ok),
        "cold_starts": sum(m["cold_start"] == "1" for _, m in executed),
        "vision_mean": stage_mean(int(c["vision_count"]) for c in with_counts),
        "yolo_mean": stage_mean(int(c["yolo_count"]) for c in with_counts),
        "same_count_percent": round(100 * mean(c["vision_count"] == c["yolo_count"] for c in with_counts), 1)
        if with_counts else "",
        "stage_cost_s1_usd": stage_cost,
    }


def retry_wait_ms(m: dict) -> float:
    """Function time not spent in the successful AI Vision call, drawing or storing: waits after HTTP 429."""
    return max(0.0, num(m["function_ms"]) - num(m["inference_ms"]) - num(m["draw_ms"]) - num(m["store_ms"]))


def print_per_camera(metrics: dict, counts: dict) -> None:
    print("  camera               frames  AI Vision  YOLOv8n")
    for cam in CAMS:
        rows = [c for c in counts.values() if c["cam_id"] == cam]
        if rows:
            print(f"  {cam:<22}{len(rows):>5}{mean(int(c['vision_count']) for c in rows):>11.2f}"
                  f"{mean(int(c['yolo_count']) for c in rows):>9.2f}")


def print_limits_and_monthly(metrics: dict) -> None:
    per_camera_per_min = 60 / INTERVAL_S
    print(f"\nLimits at one frame every {INTERVAL_S:.0f} s per camera ({per_camera_per_min:.0f} AI Vision calls/min each)")
    print(f"  AI Vision F0, {VISION_CALLS_PER_MINUTE} calls/min: {int(VISION_CALLS_PER_MINUTE // per_camera_per_min)} camera"
          f" ({VISION_CALLS_PER_MINUTE / per_camera_per_min:.2f} in theory); the 5,000 calls/month last "
          f"{VISION_F0_CALLS_PER_MONTH * INTERVAL_S / 3600:.1f} h of one camera")
    print(f"  AI Vision S1, 10 calls/s: {int(VISION_S1_CALLS_PER_MINUTE // per_camera_per_min)} cameras")

    ok = [m for m in metrics.values() if m["status"] == "ok" and m["vision_retries"] == "0" and m["cold_start"] == "0"]
    if not ok:
        return
    per_frame, billed_s = frame_costs([{"function_ms": num(m["function_ms"]), "frame_bytes": int(m["frame_bytes"])}
                                       for m in ok])
    print(f"\nMonthly cost at S1 prices, 30 days, with free grants (warm frames without 429: {billed_s:.1f} s billed)")
    print(f"  cameras  serverless   VM (B2ts v2, compute only, ${VM_MONTH:.2f}/month for up to 5 live / 8 replay cameras)")
    for n in (1, 2, 5, 8, 50):
        monthly = sum(monthly_costs(n * FRAMES_PER_MONTH, per_frame, billed_s).values())
        print(f"  {n:>7}  ${monthly:>10,.2f}")


def plot(summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    cams = [s["cameras"] for s in summary]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4))
    with_response = [s for s in summary if s["response_mean_s"] != ""]
    left.plot([s["cameras"] for s in with_response], [s["response_mean_s"] for s in with_response],
              "o-", color="#2864dc", label="Mean")
    left.plot([s["cameras"] for s in with_response], [s["response_p95_s"] for s in with_response],
              "s--", color="#d9741c", label="P95")
    left.axhline(INTERVAL_S, color="#888", lw=0.8, ls=":", label="5 s interval")
    left.set_xlabel("Simultaneous cameras")
    left.set_ylabel("Response time of processed frames (s)")
    left.set_xticks(cams)
    left.set_ylim(bottom=0)
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    parts = [("ok", "Processed", "#2a9d5c"), ("ok_after_429", "Processed after HTTP 429", "#9fd3b4"),
             ("failed_429", "Failed: HTTP 429", "#d9741c"), ("not_processed", "Not processed", "#888")]
    bottom = [0.0] * len(summary)
    for key, label, color in parts:
        # "ok" includes the frames processed after a 429; draw them separately.
        values = [100 * ((s["ok"] - s["ok_after_429"]) if key == "ok" else s[key]) / s["uploaded"] for s in summary]
        right.bar(cams, values, bottom=bottom, color=color, label=label, width=0.6)
        bottom = [b + v for b, v in zip(bottom, values)]
    right.set_xlabel("Simultaneous cameras")
    right.set_ylabel("Frames uploaded (%)")
    right.set_xticks(cams)
    right.set_ylim(0, 100)
    right.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Serverless pipeline with AI Vision F0 (20 calls/min), one frame every 5 s per camera", fontsize=10)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    print(f"\nsaved {FIGURE} and {SUMMARY_CSV}")


def num(value: str) -> float:
    return float(value) if value else 0.0


def load_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
