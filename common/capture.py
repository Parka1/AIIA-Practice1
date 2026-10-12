"""Frame capture from public webcams, shared by every architecture.

Usage as a library (with the common/ folder on sys.path):

    from capture import CAMERAS, get_frame

    frame = get_frame(CAMERAS["santa_pola"])  # BGR numpy array (OpenCV format)

Usage as a script, saving one JPEG every 5 seconds until Ctrl+C:

    python capture.py --cam santa_pola --interval 5 --out frames

Supported camera URLs:
- SkylineWebcams pages: the HLS stream URL is extracted from the page (see below).
- Snapshot URLs ending in .jpg / .jpeg / .png: downloaded and decoded.
- Anything else OpenCV/FFmpeg can open directly: MJPEG, HLS (.m3u8), RTSP, video files.

SkylineWebcams notes:
- The page embeds 'livee.m3u8?a=<token>'. That URL is a decoy: requesting it
  invalidates the token and the stream turns into the Skyline logo. The real
  stream is https://hd-auth.skylinewebcams.com/live.m3u8?a=<token>.
- Tokens expire after 2-4 minutes, so they are cached for TOKEN_TTL_S seconds
  and refreshed whenever the stream fails or shows the logo.
"""

import argparse
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .results import make_frame_id, now_local
else:
    from results import make_frame_id, now_local

# Camera id -> URL. The first one is the main camera; add the Part 4 ones here.
CAMERAS = {
    "santa_pola": "https://www.skylinewebcams.com/es/webcam/espana/comunidad-valenciana/santa-pola/paseo-maritimo.html",
    "grazalema": "https://www.skylinewebcams.com/es/webcam/espana/andalucia/cadiz/grazalema.html",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
HTTP_TIMEOUT_S = 10
STREAM_TIMEOUT_MS = 15000

SKYLINE_TOKEN_RE = re.compile(r"livee?\.m3u8\?a=([A-Za-z0-9]+)")
SKYLINE_STREAM_URL = "https://hd-auth.skylinewebcams.com/live.m3u8?a={token}"
TOKEN_TTL_S = 60

_stream_cache = {}  # Skyline page URL -> (stream URL, time.monotonic() when resolved)


class CaptureError(RuntimeError):
    """No valid frame could be obtained from a camera."""


def get_frame(cam_url: str) -> np.ndarray:
    """Return the current frame of a webcam as a BGR image (OpenCV format).

    Raises CaptureError if the camera cannot be read.
    """
    if "skylinewebcams.com" in cam_url and cam_url.endswith(".html"):
        return _get_skyline_frame(cam_url)
    if urllib.parse.urlparse(cam_url).path.lower().endswith((".jpg", ".jpeg", ".png")):
        return _get_snapshot(cam_url)
    return _read_stream(cam_url)


def _get_skyline_frame(page_url: str) -> np.ndarray:
    # First try with the cached token, then once more with a fresh one.
    for refresh in (False, True):
        try:
            frame = _read_stream(_skyline_stream_url(page_url, refresh))
        except CaptureError as e:
            reason = str(e)
            continue
        if not _is_skyline_logo(frame):
            return frame
        reason = "the stream only shows the Skyline logo"
    raise CaptureError(f"no valid frame from {page_url}: {reason}")


def _skyline_stream_url(page_url: str, refresh: bool) -> str:
    cached = _stream_cache.get(page_url)
    if cached and not refresh and time.monotonic() - cached[1] < TOKEN_TTL_S:
        return cached[0]
    html = _http_get(page_url).decode("utf-8", errors="replace")
    match = SKYLINE_TOKEN_RE.search(html)
    if not match:
        raise CaptureError(f"stream token not found in {page_url}")
    stream_url = SKYLINE_STREAM_URL.format(token=match.group(1))
    _stream_cache[page_url] = (stream_url, time.monotonic())
    return stream_url


def _is_skyline_logo(frame: np.ndarray) -> bool:
    """The logo shown with an invalid token is ~87% white; real frames are ~5%."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(gray > 235) > 0.6 * gray.size


def _read_stream(url: str) -> np.ndarray:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
    ])
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
    finally:
        cap.release()
    if not ok:
        raise CaptureError(f"could not read a frame from {url}")
    return frame


def _get_snapshot(url: str) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(_http_get(url), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise CaptureError(f"could not decode the image at {url}")
    return frame


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            return response.read()
    except OSError as e:  # URLError, HTTPError and timeouts are all OSError
        raise CaptureError(f"GET {url} failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Save webcam frames as JPEG files at a fixed interval.")
    parser.add_argument("--cam", default="santa_pola", choices=CAMERAS, help="camera id (default: santa_pola)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between frames (default: 5)")
    parser.add_argument("--count", type=int, default=0, help="number of frames to save, 0 = until Ctrl+C")
    parser.add_argument("--out", type=Path, default=Path("frames"), help="output folder (default: frames)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    saved = 0
    next_at = time.monotonic()
    try:
        while args.count == 0 or saved < args.count:
            start = time.perf_counter()
            try:
                frame = get_frame(CAMERAS[args.cam])
            except CaptureError as e:
                print(f"[{now_local():%H:%M:%S}] error: {e}")
            else:
                captured_at = now_local()
                elapsed_ms = (time.perf_counter() - start) * 1000
                # cv2.imwrite fails on non-ASCII paths on Windows, so encode and write the bytes.
                path = args.out / f"{make_frame_id(args.cam, captured_at)}.jpg"
                path.write_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                saved += 1
                height, width = frame.shape[:2]
                print(f"[{captured_at:%H:%M:%S}] {path.name} {width}x{height} ({elapsed_ms:.0f} ms)")
            # Keep a fixed rate; if capturing took longer than the interval, don't try to catch up.
            next_at = max(next_at + args.interval, time.monotonic())
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"{saved} frames saved in {args.out}")


if __name__ == "__main__":
    main()
