"""HTTP inference service and dashboard for Part 2's Azure VM."""

import csv
import hmac
import io
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

from common.results import FIELDNAMES, Result, append_result, make_frame_id, now_local
from local_vm.vm_inference import detect_people, draw_detections, load_model

METRIC_FIELDS = (
    "timestamp",
    "cpu_percent",
    "process_rss_mb",
    "system_ram_percent",
    "upload_bytes",
)
MAX_FRAME_BYTES = 12 * 1024 * 1024
CAM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
UPLOAD_RUN_GAP_SECONDS = 60


def _append_metrics(path: Path, row: dict) -> None:
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def create_app(
    model_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    upload_token: str | None = None,
    hourly_rate_usd: float | None = None,
    detector=None,
) -> Flask:
    """Create the Flask app; detector can be injected for tests."""
    root = Path(__file__).resolve().parent
    model_path = Path(model_path or os.getenv("AIIA_MODEL_PATH", root / "models/yolov8n_960.onnx"))
    data_dir = Path(data_dir or os.getenv("AIIA_DATA_DIR", "/var/lib/aiia-part2"))
    upload_token = upload_token or os.getenv("AIIA_VM_UPLOAD_TOKEN")
    if not upload_token:
        raise RuntimeError("AIIA_VM_UPLOAD_TOKEN must be configured before the service can start")

    hourly_rate_usd = (
        float(hourly_rate_usd)
        if hourly_rate_usd is not None
        else float(os.getenv("AIIA_VM_HOURLY_RATE_USD", "0.0114"))
    )
    if hourly_rate_usd < 0:
        raise ValueError("VM hourly cost must not be negative")

    data_dir.mkdir(parents=True, exist_ok=True)
    latest_path = data_dir / "latest.jpg"
    results_path = data_dir / "results.csv"
    metrics_path = data_dir / "metrics.csv"
    net_at_start = psutil.net_io_counters()
    process = psutil.Process()
    model = detector if detector is not None else load_model(str(model_path))
    inference_lock = threading.Lock()

    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_FRAME_BYTES,
        UPLOAD_TOKEN=upload_token,
    )

    @app.get("/")
    def dashboard():
        return render_template("vm_dashboard.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model": "YOLOv8n ONNX", "inference_device": "CPU"})

    @app.get("/api/latest.jpg")
    def latest_image():
        if not latest_path.is_file():
            return jsonify({"error": "No frame has been processed yet"}), 404
        return send_file(latest_path, mimetype="image/jpeg", max_age=0)

    @app.get("/api/dashboard")
    def dashboard_data():
        results = _read_csv(results_path)
        metrics = _read_csv(metrics_path)
        sent = sum(int(row["upload_bytes"]) for row in metrics)
        net_now = psutil.net_io_counters()
        first_capture = min(
            (datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc) for row in results),
            default=None,
        )
        last_result = max(
            results,
            key=lambda row: datetime.fromisoformat(row["timestamp"]),
            default=None,
        )
        vm_boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        vm_runtime = max(
            0.0,
            (now_local().astimezone(timezone.utc) - vm_boot).total_seconds(),
        )
        ordered_results = sorted(
            results,
            key=lambda row: datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc),
        )
        latest_run = []
        for row in ordered_results:
            captured_at = datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc)
            if latest_run:
                previous_at = datetime.fromisoformat(
                    latest_run[-1]["timestamp"]
                ).astimezone(timezone.utc)
                if (captured_at - previous_at).total_seconds() > UPLOAD_RUN_GAP_SECONDS:
                    latest_run = []
            latest_run.append(row)
        latest_run_bytes = sum(int(row["frame_bytes"] or 0) for row in latest_run)
        upload_window = 0.0
        if latest_run:
            run_start = datetime.fromisoformat(latest_run[0]["timestamp"]).astimezone(
                timezone.utc
            )
            latest_completion = datetime.fromisoformat(
                latest_run[-1]["timestamp"]
            ).astimezone(timezone.utc) + timedelta(
                seconds=float(latest_run[-1]["response_time_ms"]) / 1000
            )
            upload_window = max(0.0, (latest_completion - run_start).total_seconds())
        recent_metrics = metrics[-100:]
        cpu_values = [float(row["cpu_percent"]) for row in recent_metrics]
        return jsonify(
            {
                "results": results[-500:],
                "metrics": recent_metrics,
                "total_frames": len(results),
                "upload_bytes": sent,
                "upload_kib_per_frame": (sent / len(results) / 1024) if results else 0,
                "upload_kib_per_second": (
                    latest_run_bytes / 1024 / upload_window if upload_window else 0
                ),
                "cpu_percent_latest": cpu_values[-1] if cpu_values else 0,
                "cpu_percent_average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "system_ram_percent_latest": (
                    float(recent_metrics[-1]["system_ram_percent"]) if recent_metrics else 0
                ),
                "process_rss_mb_latest": (
                    float(recent_metrics[-1]["process_rss_mb"]) if recent_metrics else 0
                ),
                "network_received_since_start": max(0, net_now.bytes_recv - net_at_start.bytes_recv),
                "network_sent_since_start": max(0, net_now.bytes_sent - net_at_start.bytes_sent),
                "vm_runtime_seconds": vm_runtime,
                "estimated_vm_cost_usd": hourly_rate_usd * vm_runtime / 3600,
                "hourly_rate_usd": hourly_rate_usd,
                "cost_basis": (
                    "Azure public Linux retail rate multiplied by VM uptime since boot; "
                    "excludes disk, egress, taxes and discounts"
                ),
            }
        )

    @app.post("/api/frames")
    def process_frame():
        authorization = request.headers.get("Authorization", "")
        supplied_token = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied_token, app.config["UPLOAD_TOKEN"]):
            return jsonify({"error": "Unauthorized"}), 401

        cam_id = request.form.get("cam_id", "")
        if not CAM_ID_PATTERN.fullmatch(cam_id):
            return jsonify({"error": "cam_id must contain 1-64 letters, digits, underscores or hyphens"}), 400
        try:
            captured_at = datetime.fromisoformat(request.form.get("timestamp", ""))
        except ValueError:
            return jsonify({"error": "timestamp must be ISO 8601"}), 400
        if captured_at.tzinfo is None:
            return jsonify({"error": "timestamp must include a timezone offset"}), 400
        raw_frame = request.files.get("frame")
        if raw_frame is None:
            return jsonify({"error": "multipart field 'frame' is required"}), 400
        image_bytes = raw_frame.read(MAX_FRAME_BYTES + 1)
        if not image_bytes or len(image_bytes) > MAX_FRAME_BYTES:
            return jsonify({"error": "frame must be a non-empty image no larger than 12 MiB"}), 413
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format != "JPEG":
                    return jsonify({"error": "frame must be a JPEG image"}), 400
                if image.width > 8192 or image.height > 8192:
                    return jsonify({"error": "frame dimensions must not exceed 8192 pixels"}), 400
                image.verify()
        except (UnidentifiedImageError, OSError):
            return jsonify({"error": "frame is not a decodable JPEG image"}), 400
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            return jsonify({"error": "frame is not a decodable image"}), 400

        with inference_lock:
            started_at = time.perf_counter()
            cpu_before = process.cpu_times()
            detections, inference_ms = detect_people(model, frame)
            annotated = draw_detections(frame, detections)
            completed_at = now_local()
            response_time_ms = (completed_at - captured_at).total_seconds() * 1000
            if response_time_ms < -120_000 or response_time_ms > 30 * 60 * 1000:
                return jsonify({"error": "capture timestamp is outside the accepted clock-skew window"}), 400
            response_time_ms = max(0.0, response_time_ms)
            wall_seconds = max(time.perf_counter() - started_at, 1e-9)
            cpu_after = process.cpu_times()
            cpu_seconds = (
                cpu_after.user - cpu_before.user + cpu_after.system - cpu_before.system
            )
            cpu_percent = 100 * cpu_seconds / wall_seconds / max(psutil.cpu_count() or 1, 1)
            memory = psutil.virtual_memory()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            frame_id = make_frame_id(cam_id, captured_at)
            result = Result(
                timestamp=captured_at,
                architecture="vm",
                cam_id=cam_id,
                frame_id=frame_id,
                n_people=len(detections),
                response_time_ms=response_time_ms,
                inference_ms=inference_ms,
                frame_bytes=len(image_bytes),
            )
            ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if not ok:
                raise RuntimeError("OpenCV could not encode the annotated frame")
            temporary_path = latest_path.with_suffix(".jpg.tmp")
            temporary_path.write_bytes(encoded.tobytes())
            temporary_path.replace(latest_path)
            append_result(results_path, result)
            _append_metrics(
                metrics_path,
                {
                    "timestamp": completed_at.isoformat(timespec="milliseconds"),
                    "cpu_percent": round(cpu_percent, 2),
                    "process_rss_mb": round(rss_mb, 2),
                    "system_ram_percent": round(memory.percent, 2),
                    "upload_bytes": len(image_bytes),
                },
            )
        return (
            jsonify(
                {
                    "result": {
                        "timestamp": result.timestamp.isoformat(timespec="milliseconds"),
                        "architecture": result.architecture,
                        "cam_id": result.cam_id,
                        "frame_id": result.frame_id,
                        "n_people": result.n_people,
                        "response_time_ms": result.response_time_ms,
                        "inference_ms": result.inference_ms,
                        "frame_bytes": result.frame_bytes,
                    },
                    "metrics": {
                        "cpu_percent": round(cpu_percent, 2),
                        "process_rss_mb": round(rss_mb, 2),
                        "system_ram_percent": round(memory.percent, 2),
                    },
                }
            ),
            201,
        )

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "request body exceeds the 12 MiB limit"}), 413

    return app


app = create_app() if os.getenv("AIIA_VM_UPLOAD_TOKEN") else Flask(__name__)

if __name__ == "__main__":
    if not app.config.get("UPLOAD_TOKEN"):
        app = create_app()
    app.run(host="127.0.0.1", port=8765, threaded=True)
