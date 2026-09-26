"""Common CSV format for the detection results of every architecture.

Every architecture (local, vm, serverless) writes rows with the same columns,
so a single script can load all the CSV files and compare them:

    column            meaning
    ----------------  ---------------------------------------------------------
    timestamp         frame capture time, ISO 8601 in Spanish time (Europe/Madrid)
                      with its UTC offset (e.g. 2026-09-24T20:10:05.123+02:00)
    architecture      local | vm | serverless
    cam_id            camera key in capture.CAMERAS (e.g. santa_pola)
    frame_id          <cam_id>_<capture time in Spanish time>, e.g. santa_pola_20260924T201005123.
                      Same frame -> same id in every architecture, so results can
                      be joined with each other and with the ground truth.
    n_people          people detected in the frame
    n_vehicles        vehicles detected (empty if not computed)
    response_time_ms  end-to-end time: frame captured -> result stored
    inference_ms      time spent in the model / AI Vision call only (empty if unknown)
    frame_bytes       size of the encoded image sent or processed (empty if n/a)

Usage:

    from results import Result, append_result, now_local

    captured_at = now_local()
    ...
    append_result("results/local.csv", Result(
        timestamp=captured_at, architecture="local", cam_id="santa_pola",
        n_people=4, n_vehicles=12, response_time_ms=850.3, inference_ms=410.2,
        frame_bytes=312455,
    ))

Times always use Europe/Madrid, not the machine's time zone, because the VM
and the Azure Function run in UTC.

This module only uses the standard library (pandas is imported lazily by
load_results), so it can be copied as is to the VM or to the Azure Function.
On Windows, zoneinfo also needs the tzdata package (pip install tzdata).
"""

import csv
import io
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

ARCHITECTURES = ("local", "vm", "serverless")
LOCAL_TZ = ZoneInfo("Europe/Madrid")


@dataclass(kw_only=True)
class Result:
    """One processed frame. Field order is the CSV column order."""

    timestamp: datetime  # capture time, timezone-aware
    architecture: str
    cam_id: str
    frame_id: str = ""  # derived from cam_id + timestamp when empty
    n_people: int
    n_vehicles: Optional[int] = None
    response_time_ms: float
    inference_ms: Optional[float] = None
    frame_bytes: Optional[int] = None

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {self.architecture!r}")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware, use now_local()")
        if not self.frame_id:
            self.frame_id = make_frame_id(self.cam_id, self.timestamp)


FIELDNAMES = [f.name for f in fields(Result)]


def now_local() -> datetime:
    """Current time in Spain as a timezone-aware datetime."""
    return datetime.now(LOCAL_TZ)


def make_frame_id(cam_id: str, captured_at: datetime) -> str:
    """Build the frame id, also used as file / blob name: santa_pola_20260924T201005123."""
    local = captured_at.astimezone(LOCAL_TZ)
    return f"{cam_id}_{local:%Y%m%dT%H%M%S}{local.microsecond // 1000:03d}"


def parse_frame_id(frame_id: str) -> tuple[str, datetime]:
    """Inverse of make_frame_id: returns (cam_id, capture time in Spanish time)."""
    cam_id, stamp = frame_id.rsplit("_", 1)
    captured_at = datetime.strptime(stamp, "%Y%m%dT%H%M%S%f").replace(tzinfo=LOCAL_TZ)
    return cam_id, captured_at


def append_result(csv_path, result: Result) -> None:
    """Append a result to a CSV file, writing the header if the file is new."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(_to_row(result))


def format_csv(result: Result, header: bool = False) -> str:
    """Return the result as CSV text, e.g. to append it to an Azure append blob."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES, lineterminator="\n")
    if header:
        writer.writeheader()
    writer.writerow(_to_row(result))
    return buf.getvalue()


def load_results(*csv_paths):
    """Load one or more result CSV files into a single pandas DataFrame.

    Timestamps are converted to LOCAL_TZ, in case some rows were written with another offset.
    """
    import pandas as pd

    dtypes = {"n_people": "Int64", "n_vehicles": "Int64", "frame_bytes": "Int64"}
    df = pd.concat(
        [pd.read_csv(p, usecols=FIELDNAMES, dtype=dtypes) for p in csv_paths],
        ignore_index=True,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(LOCAL_TZ.key)
    return df


def _to_row(result: Result) -> dict:
    row = asdict(result)
    row["timestamp"] = result.timestamp.astimezone(LOCAL_TZ).isoformat(timespec="milliseconds")
    for key in ("response_time_ms", "inference_ms"):
        if row[key] is not None:
            row[key] = round(row[key], 1)
    return {key: "" if value is None else value for key, value in row.items()}
