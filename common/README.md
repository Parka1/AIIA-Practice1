# common

Code shared by every architecture.

## capture.py: webcam frames

```sh
python common/capture.py                                   # santa_pola, one frame every 5 s into frames/
python common/capture.py --cam grazalema --count 10 --out frames-grazalema
```

Options: `--cam`, `--interval` (s), `--count` (0 = until Ctrl+C), `--out`.
Camera ids are the keys of `CAMERAS`; the video streams are `santa_pola`,
`grazalema`, `benidorm_poniente`, `benidorm_levante`, `madrid_alcala`,
`benalmadena`, `la_orotava`, `villafranca_del_bierzo` and `calafell`. The
`*_preview` and other `.jpg` sources are low-resolution snapshots, only for
load tests.

From code:

```python
from common.capture import CAMERAS, get_frame, LiveFrameReader
frame = get_frame(CAMERAS["santa_pola"])     # one BGR frame (~1.5 s: opens and closes the stream)
reader = LiveFrameReader(CAMERAS["santa_pola"]); reader.start()
frame = reader.latest_frame()                # keeps the stream open: use it to hold a 5 s cadence
```

SkylineWebcams serves HLS behind a temporary token: `get_frame` reads the token
from the camera page, caches it for 60 s and retries with a new one when the
stream only shows the Skyline logo. Never request the `livee.m3u8` URL of the
page: it is a decoy that invalidates the token.

## results.py: common CSV format

`timestamp, architecture, cam_id, frame_id, n_people, n_vehicles,
response_time_ms, inference_ms, frame_bytes`, one row per frame.
`frame_id` is `<cam_id>_<capture time>` (e.g. `santa_pola_20260924T201005123`)
and identifies the same frame in every architecture. Times are in Spanish time.
`response_time_ms` goes from capture to result stored.

```python
from common.results import Result, append_result, now_local
append_result("results/local.csv", Result(timestamp=now_local(), architecture="local",
              cam_id="santa_pola", n_people=3, response_time_ms=850.0))
```

It only uses the standard library, so it is copied as is to the VM and the
Azure Function.

## optimizations.py

Resize, JPEG compression, motion detection and lighting hysteresis, applied on
the client before sending a frame. See [`advanced/`](../advanced/README.md).
