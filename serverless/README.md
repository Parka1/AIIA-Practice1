# Serverless with Azure Functions and AI Vision (Parts 3 and 4.3)

```
local PC --upload--> raw-frames (Blob) --Event Grid--> Azure Function --> AI Vision (people)
                                                            |
                                                            +--> $web: annotated frame + CSV --> static website
```

- AI Vision (F0, France Central): Image Analysis 4.0 `people` feature; only boxes with confidence >= 0.5 are counted.
- Function App `aiia-p1-frames-rba` (Flex Consumption, Python 3.12): blob trigger on `raw-frames` with an Event Grid source. It is deployed from VS Code (Azure Functions extension) from `function_app/`.
- Website: <https://aiiap1serverlessb.z50.web.core.windows.net/>

**Recursion guard:** the function only listens to `raw-frames` and only writes
to `$web`, the Event Grid subscription only forwards `.jpg` files in
`raw-frames`, and `upload_frames.py` refuses `--container '$web'`.

## Credentials

`credentials.env` in the repository root (git-ignored), or environment
variables: `AZURE_VISION_ENDPOINT`, `AZURE_VISION_KEY`,
`AZURE_STORAGE_CONNECTION_STRING`. The function reads the same values as app
settings (`FRAMES_STORAGE`, see `function_app/local.settings.example.json`).

## Part 3

```sh
python -m serverless.deploy_web --reset-data           # publish the website, empty the results
python -m serverless.upload_frames --cam grazalema --interval 5 --count 10
python -m serverless.analyze_serverless --fetch --yolo # Task 3.6: response time, detection, cost
```

- The function writes `$web/frames/<frame_id>.jpg`, `$web/data/results.csv` (common format) and `$web/data/metrics.csv` (time of every stage, cold starts, HTTP 429 retries). Both CSV files are append blobs, so concurrent instances don't overwrite each other.
- `python -m serverless.probe_vision --count 5` tests AI Vision from the local PC without uploading anything.
- `python -m serverless.evaluate_vision` compares AI Vision with YOLO and with the manual labels of Part 2.

## Part 4.3: several cameras

```sh
python -m serverless.scale_serverless       # staircase from 1 to 5 cameras, 10 frames each, then analysis
python -m serverless.scale_serverless --analyze
```

Uses 150 AI Vision calls. It writes `report/data/part4_serverless_*.csv` and
`report/images/Section_4/part4_serverless_scaling.png`. The F0 tier allows 20
calls/min and one camera uses 12, so from two cameras on the function may get
HTTP 429. `upload_frames.py --cams A B ... --persistent-streams` uploads
several cameras by hand.

Tests: `python -m unittest serverless.tests.test_vision serverless.tests.test_function`
(`test_function` also checks that the copies of `results.py` and `vision.py` in
`function_app/` are up to date).
