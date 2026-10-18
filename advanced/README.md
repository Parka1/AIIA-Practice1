# Advanced task 3: optimizations

Client-side improvements in `common/optimizations.py`, applied before a frame
is sent, so they work for any architecture:

| Function | What it does | Result in the report |
|---|---|---|
| `resize_to_width`, `encode_jpeg` | Lower resolution / JPEG quality | 960 px: -67% bytes, recall -2 points |
| `MotionDetector.should_send` | Skips frames without motion (sends one per minute anyway) | 38% of frames skipped by day, ~70% at night |
| `enhance_clahe`, `enhance_gamma` | Brighten shaded areas | No real gain; not used |
| `LightingController` | Lights off only after N frames without people (hysteresis) | 30 s hold: no flicker, lights on ~99% of the occupied time |

In the serverless uploader:
`python -m serverless.upload_frames --width 960 --quality 85 --motion 0.002`

## Evaluation (offline, on saved frames)

```sh
python -m advanced.evaluate_bandwidth --vision     # resolution and JPEG sweeps
python -m advanced.evaluate_enhancement --vision   # CLAHE / gamma
python -m advanced.evaluate_motion                 # motion detection thresholds
python -m advanced.evaluate_hysteresis             # lighting hold times
```

- The first two scripts score detections against the manual boxes of the 20 frames of the serverless run (`report/data/part3_ground_truth_boxes.json`, IoU >= 0.3). Without `--vision` only YOLO is used. AI Vision responses are cached, so repeating a script costs no quota.
- Motion and hysteresis use 10-minute sequences captured with `python common/capture.py --count 120 --out report/advanced3_sequences/<day|night>/<camera>`.
- The results go to `report/data/advanced3_*.csv` and the figures to `report/images/Advanced_3/`.

Tests: `python -m unittest advanced.tests.test_optimizations`
