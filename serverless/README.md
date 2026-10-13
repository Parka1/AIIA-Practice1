# Serverless processing with Azure AI Vision (Part 3)

People are detected with the Image Analysis 4.0 `people` feature of Azure AI
Vision (`vision.py`). Only boxes with confidence >= 0.5 are counted: the service
also returns dozens of low-confidence boxes per frame.

Credentials are read from environment variables or from `credentials.env` at
the repository root (git-ignored), see `credentials.py`:

| Value | Environment variable | `credentials.env` |
|---|---|---|
| AI Vision endpoint | `AZURE_VISION_ENDPOINT` | `Endpoint` under `# Computer Vision resource` |
| AI Vision key | `AZURE_VISION_KEY` | `Key 1` under `# Computer Vision resource` |
| Storage connection string | `AZURE_STORAGE_CONNECTION_STRING` | `ConnectionString` under `# Storage account` |

Run everything from the repository root (Windows paths shown; on Linux/macOS
use `.venv/bin/python`). The dependencies are the common ones:

```sh
.venv/Scripts/python -m pip install -r serverless/requirements.txt
```

## Live check of the AI Vision resource

Capture a frame every 5 seconds, send it to AI Vision and print the count and
the capture and inference times. Nothing is uploaded to storage:

```sh
.venv/Scripts/python -m serverless.probe_vision --cam grazalema --interval 5 --count 10 --save-dir frames/vision
```

`--save-dir` stores the frames with the detected people drawn.

## Uploading frames to the raw container

Capture a frame every 5 seconds and upload it to the private `raw-frames`
container, which triggers the Azure Function:

```sh
.venv/Scripts/python -m serverless.upload_frames --cam grazalema --interval 5 --count 10
```

Blobs are named `<frame_id>.jpg` and carry `cam_id`, `frame_id` and
`captured_at` as metadata. Capture and upload times are appended to
`results/serverless_uploads.csv` (`--log`). The script refuses `--container
'$web'`, because the function writes its results there and would trigger
itself recursively. It warns when the rate exceeds the 20 calls/min of the
AI Vision free tier: every uploaded frame is one AI Vision call.

Each SkylineWebcams capture opens the HLS stream, reads one frame and closes it
(~1.5 s), so intervals much shorter than that are not reachable.

## Detection efficiency against the labelled frames

Send the 20 frames labelled in Part 2 to AI Vision and compare its counts with
the ground truth and with YOLO:

```sh
.venv/Scripts/python -m serverless.evaluate_vision
```

It writes `report/data/part3_vision_ground_truth.csv` (the Part 2 ground truth
plus `vision_count`) and saves every detection in
`report/data/part3_vision_detections.json`. Add `--reuse` to recompute the
counts from that JSON, e.g. with another `--min-confidence`, without calling
the service again.

## Unit tests

```sh
.venv/Scripts/python -m unittest serverless.tests.test_vision
```
