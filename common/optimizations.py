"""Client-side optimizations shared by every architecture (Advanced task 3).

They run on the local machine before a frame is sent, so the VM (Part 2) and the
serverless pipeline (Part 3) can use them unchanged:

    from optimizations import MotionDetector, encode_jpeg, resize_to_width

    motion = MotionDetector(region=MOTION_REGIONS.get(cam_id))
    if motion.should_send(frame):
        image = encode_jpeg(resize_to_width(frame, 1280), quality=70)
        ...  # upload image

- resize_to_width: lower the resolution (fewer bytes, smaller people).
- encode_jpeg: JPEG compression with a given quality.
- MotionDetector: skip frames in which nothing has changed since the last frame sent.
- enhance_clahe / enhance_gamma: brighten shaded areas before detection.
- LightingController: switch the lights with hysteresis instead of frame by frame.
"""

import cv2
import numpy as np

# Part of each frame where people and vehicles move, as fractions (x1, y1, x2, y2) of
# the frame size. In Santa Pola the sea, the boats and the sky move continuously and
# would always count as motion, so only the promenade and the street are used.
MOTION_REGIONS = {
    "santa_pola": (0.0, 0.40, 1.0, 1.0),
}


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Scale the frame to `width` pixels keeping the aspect ratio (never upscales)."""
    height, current = frame.shape[:2]
    if width >= current:
        return frame
    return cv2.resize(frame, (width, round(height * width / current)), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG with the given quality (1-100)."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("OpenCV could not encode the frame")
    return encoded.tobytes()


def enhance_clahe(frame: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Local contrast enhancement (CLAHE) of the lightness channel only, to lift shaded areas."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_gamma(frame: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Global gamma correction; gamma < 1 brightens the dark tones more than the bright ones."""
    table = (255 * (np.arange(256) / 255) ** gamma).round().astype(np.uint8)
    return cv2.LUT(frame, table)


class LightingController:
    """Lighting decision with hysteresis: turn on as soon as someone is detected and
    only turn off after `hold_frames` consecutive frames without anyone.

    The detectors miss people from one frame to the next (Part 2 and 3), so switching
    the lights with the raw count of each frame would make them flicker; the hold
    keeps them on through isolated misses. hold_frames=1 is the raw decision.
    """

    def __init__(self, hold_frames: int = 3, min_people: int = 1):
        if hold_frames < 1:
            raise ValueError("hold_frames must be at least 1")
        self.hold_frames = hold_frames
        self.min_people = min_people
        self.on = False
        self._empty = 0

    def update(self, n_people: int) -> bool:
        """Feed the count of a new frame and return whether the lights are on."""
        if n_people >= self.min_people:
            self.on = True
            self._empty = 0
        else:
            self._empty += 1
            if self._empty >= self.hold_frames:
                self.on = False
        return self.on


class MotionDetector:
    """Decide whether a frame differs enough from the last frame sent to be worth sending.

    Both frames are converted to small blurred grayscale images (so that JPEG noise
    and small camera vibrations are ignored) and compared pixel by pixel. The frame is
    sent when the fraction of pixels whose brightness changed by more than
    `pixel_threshold` reaches `min_changed`. Comparing with the last frame *sent*,
    not with the previous one, makes slow changes add up until they are sent.

    `max_skipped` forces a frame through after that many skipped frames, so the
    results are refreshed at least once in a while even in a static scene.
    """

    def __init__(
        self,
        min_changed: float = 0.005,
        pixel_threshold: int = 25,
        region: tuple[float, float, float, float] | None = None,
        max_skipped: int = 12,
        width: int = 320,
    ):
        self.min_changed = min_changed
        self.pixel_threshold = pixel_threshold
        self.region = region
        self.max_skipped = max_skipped
        self.width = width
        self._reference: np.ndarray | None = None
        self._skipped = 0

    def changed_fraction(self, frame: np.ndarray) -> float:
        """Fraction of changed pixels with respect to the last frame sent (1.0 if none yet)."""
        small = self._prepare(frame)
        if self._reference is None or self._reference.shape != small.shape:
            return 1.0
        diff = cv2.absdiff(small, self._reference)
        return float(np.count_nonzero(diff > self.pixel_threshold)) / diff.size

    def should_send(self, frame: np.ndarray) -> bool:
        send = self.changed_fraction(frame) >= self.min_changed or self._skipped >= self.max_skipped
        if send:
            self._reference = self._prepare(frame)
            self._skipped = 0
        else:
            self._skipped += 1
        return send

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if self.region:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = self.region
            frame = frame[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
        gray = cv2.cvtColor(resize_to_width(frame, self.width), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)
