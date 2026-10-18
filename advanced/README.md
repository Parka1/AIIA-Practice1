# Advanced task 3: client-side optimizations

The optimizations live in `common/optimizations.py`, so every architecture can
use them before sending a frame:

- `resize_to_width(frame, width)`: lower the resolution.
- `encode_jpeg(frame, quality)`: JPEG compression.
- `MotionDetector(min_changed, region).should_send(frame)`: skip frames without
  motion. `MOTION_REGIONS` holds the area of each camera where people move.
- `enhance_clahe(frame)` / `enhance_gamma(frame)`: brighten shaded areas.
- `LightingController(hold_frames).update(n_people)`: lighting decision with hysteresis.

The serverless upload script exposes them as options:

```sh
.venv/Scripts/python -m serverless.upload_frames --width 960 --quality 70 --motion 0.005
```

## Evaluation (offline, on saved frames)

Run from the repository root:

```sh
# Resolution and JPEG quality sweeps on the 20 frames with manual boxes (report/part3_frames)
.venv/Scripts/python -m advanced.evaluate_bandwidth            # YOLOv8n only
.venv/Scripts/python -m advanced.evaluate_bandwidth --vision   # also AI Vision (responses cached)

# Shadow enhancement on the same 20 frames
.venv/Scripts/python -m advanced.evaluate_enhancement --vision

# Motion detection and lighting hysteresis on continuous captures
# (report/advanced3_sequences/<day|night>/<camera>)
.venv/Scripts/python -m advanced.evaluate_motion
.venv/Scripts/python -m advanced.evaluate_hysteresis
```

Sequences are captured locally without uploading anything:

```sh
cd common
../.venv/Scripts/python capture.py --cam santa_pola --interval 5 --count 120 --out ../report/advanced3_sequences/day/santa_pola
```

Detection is scored against `report/data/part3_ground_truth_boxes.json`: a
detection matches a person when IoU >= 0.3 (`advanced/detection_metrics.py`).
Results are written to `report/data/advanced3_*.csv` and the figures to
`report/images/Advanced_3/`.

```sh
.venv/Scripts/python -m unittest advanced.tests.test_optimizations
```
