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
.venv/Scripts/python -m serverless.probe_vision --interval 5 --count 10 --save-dir frames/vision
```

`--save-dir` stores the frames with the detected people drawn.

## Uploading frames to the raw container

Capture a frame every 5 seconds and upload it to the private `raw-frames`
container, which triggers the Azure Function:

```sh
.venv/Scripts/python -m serverless.upload_frames --interval 5 --count 10
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

## Azure Function (Tasks 3.3 and 3.5)

`function_app/` is a Python v2 Azure Functions project for the Flex Consumption
plan. `process_raw_frame` is a blob trigger on `raw-frames/{name}` whose source
is an Event Grid subscription (the only blob trigger in Flex Consumption). For
every frame it calls AI Vision, draws the people with Pillow and writes to the
`$web` container only:

| Blob in `$web` | Content |
|---|---|
| `frames/<frame_id>.jpg` | frame with the detected people drawn |
| `data/results.csv` | common results format, `architecture=serverless` |
| `data/metrics.csv` | per-frame timings: start delay (upload + Event Grid + cold start), AI Vision call, drawing, storage, cold start flag, HTTP 429 retries |

Both CSV files are append blobs, so concurrent instances never overwrite each
other's rows. AI Vision errors (e.g. HTTP 429 after the retries) are recorded in
`metrics.csv` and the frame is not retried.

The project is deployed on its own, so it carries copies of
`common/results.py` and `serverless/vision.py`; `tests/test_function.py` fails
if they get out of date. App settings (see `local.settings.example.json`):
`FRAMES_STORAGE`, `AZURE_VISION_ENDPOINT`, `AZURE_VISION_KEY` and, optionally,
`VISION_MIN_CONFIDENCE`. The real `local.settings.json` is git-ignored.

**Recursion guard.** The function only listens to `raw-frames`, the event
subscription only forwards `Blob Created` events whose subject begins with
`/blobServices/default/containers/raw-frames/blobs/` and ends with `.jpg`,
and the function only writes to `$web`.

## Static website (Task 3.1)

`web/index.html` is the dashboard, served from the `$web` container at
<https://aiiap1serverlessb.z50.web.core.windows.net/>. It reads
`data/results.csv` and `data/metrics.csv` every 5 seconds. Publish it (and
create the empty CSV files) with:

```sh
.venv/Scripts/python -m serverless.deploy_web
```

`--reset-data` deletes the results and processed frames first, for a clean
measurement run.

## Multi-camera capacity (Task 4.3)

`upload_frames.py` accepts several cameras (`--cams`, one thread each); use
`--persistent-streams` with them, as in the VM test of Part 4, to keep each HLS
stream open and the five-second cadence. The staircase of the report runs the
five Task 4.1 cameras from 1 to 5 at a time, ten frames per camera, waits for
the function after each stage, and then fetches, analyzes and plots the run:

```sh
.venv/Scripts/python -m serverless.scale_serverless
```

That is 150 AI Vision calls. Use `--analyze` to repeat only the analysis, and
`--stages`/`--count` for a shorter run. With the F0 tier (20 calls/min) one
camera uses 12 calls/min, so from two cameras on the function gets HTTP 429:
the run measures how many frames are still processed and how much later.
