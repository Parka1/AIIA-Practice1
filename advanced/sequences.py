"""Frame sequences used by the motion detection and hysteresis evaluations.

A sequence folder (report/advanced3_sequences/<day|night>/<camera>) may hold
several captures taken at different times, e.g. the daytime Grazalema frames of
Parts 2 and 3. They are split into continuous runs wherever two consecutive
frames are more than MAX_GAP_S apart, so the detector and the lighting controller
start again at every run instead of treating separate captures as continuous.
"""

from pathlib import Path

from common.results import parse_frame_id

ROOT = Path("report/advanced3_sequences")
MAX_GAP_S = 60
CAMERA_COLORS = {"santa_pola": "#2864dc", "grazalema": "#d9741c"}
PERIOD_STYLES = {"day": "-", "night": ":"}


def sequence_dirs() -> list[Path]:
    return sorted(p for p in ROOT.glob("*/*") if p.is_dir() and any(p.glob("*.jpg")))


def split_runs(frames: list[Path]) -> list[list[Path]]:
    """Group frames (sorted by name, i.e. by capture time) into continuous runs."""
    runs: list[list[Path]] = []
    previous = None
    for frame in sorted(frames):
        _, captured_at = parse_frame_id(frame.stem)
        if previous is None or (captured_at - previous).total_seconds() > MAX_GAP_S:
            runs.append([])
        runs[-1].append(frame)
        previous = captured_at
    return runs
