# common

Code shared by every architecture (local, VM, serverless).

```
pip install -r common/requirements.txt
```

## capture.py: getting frames

### Saving frames to disk

Run from the repository root:

```
python common/capture.py
```

This saves one frame from the default camera (`santa_pola`) every 5 seconds into `frames/` until you press `Ctrl+C`. The alternative `grazalema` camera is available for the Part 2 comparison. Each frame is printed as it is saved:

```
[20:29:15] santa_pola_20260924T202915035.jpg 1920x1080 (1588 ms)
[20:29:19] santa_pola_20260924T202919889.jpg 1920x1080 (1436 ms)
```

The time in brackets is Spanish time and the value in parentheses is how long the capture took. Files are named `<frame_id>.jpg`, the same `frame_id` used in the CSV, so saved frames can be matched with the results (e.g. for the ground truth). Each frame is a JPEG of about 750 KB.

Arguments (all optional):

| argument | default | what it does |
|---|---|---|
| `--cam` | `santa_pola` | Camera to capture from. Must be one of the ids in `CAMERAS`. |
| `--interval` | `5` | Seconds between frames. A capture takes ~1.5 s, so values below that just capture as fast as possible. If a capture takes longer than the interval, the next one starts immediately and the missed slots are skipped, not caught up. |
| `--count` | `0` | Number of frames to save before stopping. `0` means keep going until `Ctrl+C`. |
| `--out` | `frames` | Folder where the JPEG files are saved. It is created if it doesn't exist. Relative paths are relative to the folder you run the command from. |

Examples:

```
# 50 frames, one every 5 s (~4 minutes), e.g. to label the ground truth
python common/capture.py --count 50 --out ground_truth

# capture from the Grazalema plaza webcam instead of Santa Pola
python common/capture.py --cam grazalema --count 10 --out frames-grazalema

# one frame every 30 s until Ctrl+C
python common/capture.py --interval 30
```

If a capture fails (network error, camera down), the error is printed and the loop goes on with the next frame. `python common/capture.py --help` shows the argument list.

`frames/` is in `.gitignore`, so saved frames are not committed. Use another folder with `--out` for frames that should be shared, such as the ground truth set.

### Using it from code

```python
import sys
sys.path.append("../common")  # path to this folder

from capture import CAMERAS, CaptureError, get_frame

frame = get_frame(CAMERAS["santa_pola"])  # BGR numpy array, 1920x1080
```

- `get_frame(cam_url)` opens the camera, reads one frame and closes it (~1.5 s for the Skyline stream). It raises `CaptureError` if the camera can't be read, so wrap it in `try/except` inside capture loops.
- `CAMERAS` maps a camera id to its URL. It currently contains `santa_pola` (default) and `grazalema` (comparison camera); add further Part 4 cameras there. The id is the `cam_id` used in the CSV.
- Supported URLs: SkylineWebcams pages, snapshot images (`.jpg`, `.jpeg`, `.png`) and anything OpenCV can open directly (MJPEG, RTSP, `.m3u8`, video files).

### How the SkylineWebcams capture works

Skyline doesn't serve an MJPEG stream like the teaching team's `grabber.py` example. It serves an HLS stream (video split into ~3 s chunks) protected by a temporary token:

- The page contains the stream address with a token: `livee.m3u8?a=<token>`. `get_frame` downloads the page, extracts the token and opens `https://hd-auth.skylinewebcams.com/live.m3u8?a=<token>` with OpenCV.
- **Decoy URL:** the address in the page (`livee.m3u8`, with two e's) is a trap. Requesting it invalidates the token and from then on the stream only shows the Skyline logo. The code never requests it and uses `live.m3u8` instead.
- **Token lifetime:** tokens stop working after 2-4 minutes. The token is cached for 60 s and a new one is fetched whenever the stream fails.
- **Logo detection:** if a frame is the Skyline logo (more than 60% white pixels; the logo is ~87% white, real frames ~5%), it is discarded and the capture is retried once with a new token.
- The page also has a thumbnail image (`live4828.jpg`), but it is only 344x193, too small to detect people.

### Limitations

- Each call opens the stream, reads one frame and closes it, which takes ~1.5 s. That is plenty for one frame every 5 s, but it can't reach the 0.5 s rate given as an example in Part 3. That would need a reader that keeps the stream open in a background thread.
- The frame returned is a few seconds behind real time, because of how HLS streams work. It doesn't matter for counting people.
- When capturing several cameras at the same time (Part 4), call `get_frame` from one thread per camera, or the calls add up (~1.5 s each).

## results.py: the common CSV format

| column | meaning |
|---|---|
| `timestamp` | frame capture time, ISO 8601 in Spanish time (Europe/Madrid) with its UTC offset (`2026-09-24T20:10:05.123+02:00`) |
| `architecture` | `local`, `vm` or `serverless` |
| `cam_id` | camera id in `CAMERAS` |
| `frame_id` | `<cam_id>_<capture time in Spanish time>`, e.g. `santa_pola_20260924T201005123`. It is the same for the same frame in every architecture |
| `n_people` | people detected |
| `n_vehicles` | vehicles detected (empty if not computed) |
| `response_time_ms` | end-to-end: frame captured → result stored |
| `inference_ms` | model / AI Vision call only (empty if unknown) |
| `frame_bytes` | size of the encoded image sent or processed (empty if n/a) |

Take the capture time right after `get_frame()` returns, so capture time is left out of `response_time_ms`. It's the same for every architecture.

```python
from results import Result, append_result, now_local

frame = get_frame(CAMERAS["santa_pola"])
captured_at = now_local()
# ... detect ...
append_result("results/local.csv", Result(
    timestamp=captured_at, architecture="local", cam_id="santa_pola",
    n_people=4, n_vehicles=12, response_time_ms=850.3, inference_ms=410.2,
    frame_bytes=312455,
))
```

- `frame_id` is filled in from `cam_id` and `timestamp`. When the capture time must travel with the image (VM request, blob name), send `make_frame_id(...)` and recover the time on the other side with `parse_frame_id(...)`.
- `format_csv(result, header=False)` returns the row as text, e.g. to append it to an Azure append blob.
- `load_results("results/local.csv", "results/vm.csv", ...)` merges several CSV files into one pandas DataFrame, for the report plots.
- All times use Spanish time (Europe/Madrid), not the machine's time zone, so the VM and the Azure Function (which run in UTC) write the same times as the local machine.
- `results.py` only uses the standard library, so it can be copied as is to the VM or the Azure Function. On Windows it also needs `tzdata` (included in `requirements.txt`).
