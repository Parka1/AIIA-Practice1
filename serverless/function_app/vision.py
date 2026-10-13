"""People detection with Azure AI Vision, shared by the local scripts and the Azure Function.

Uses the Image Analysis 4.0 REST API with the `people` feature, which returns a
bounding box and a confidence for every person. The `tags` feature only says
that there are people in the image, and `objects` missed the people in the test
frames, so neither can be used to count them.

The service also returns many low-confidence boxes (34 in a frame with 2 real
people), so detections are counted only above DEFAULT_MIN_CONFIDENCE.

Usage:

    from vision import analyze_people, count_people

    analysis = analyze_people(jpeg_bytes, endpoint, key)
    n_people = count_people(analysis.people)

Only the standard library is used, so this module can be copied as is to the
Azure Function (like common/results.py).
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

API_VERSION = "2024-02-01"
ANALYZE_PATH = "/computervision/imageanalysis:analyze?api-version={version}&features=people"
DEFAULT_MIN_CONFIDENCE = 0.5
HTTP_TIMEOUT_S = 30
MAX_RETRIES = 3  # retries after HTTP 429 (rate limit of the pricing tier)
DEFAULT_RETRY_AFTER_S = 2.0


class VisionError(RuntimeError):
    """The AI Vision request failed or returned an unexpected response."""


@dataclass(frozen=True)
class Person:
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass(frozen=True)
class Analysis:
    people: list[Person]  # every detection, including the low-confidence ones
    width: int
    height: int
    inference_ms: float  # round trip of the successful request, network included
    service_ms: Optional[float]  # processing time reported by the service, if any
    retries: int  # requests rejected with HTTP 429 before the successful one


def analyze_people(image: bytes, endpoint: str, key: str, timeout: float = HTTP_TIMEOUT_S) -> Analysis:
    """Send an encoded image (JPEG/PNG) to AI Vision and return the detected people."""
    request = urllib.request.Request(
        endpoint.rstrip("/") + ANALYZE_PATH.format(version=API_VERSION),
        data=image,
        method="POST",
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"},
    )
    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                service_time = response.headers.get("x-envoy-upstream-service-time")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                time.sleep(_retry_after_s(e.headers.get("Retry-After")))
                continue
            detail = e.read()[:300].decode("utf-8", errors="replace")
            raise VisionError(f"AI Vision returned HTTP {e.code}: {detail}") from e
        except OSError as e:  # URLError and timeouts
            raise VisionError(f"AI Vision request failed: {e}") from e
        inference_ms = (time.perf_counter() - start) * 1000
        break

    try:
        analysis = parse_analysis(json.loads(body))
    except (ValueError, KeyError, TypeError) as e:
        raise VisionError(f"unexpected AI Vision response: {body[:300]!r}") from e
    return Analysis(
        people=analysis.people,
        width=analysis.width,
        height=analysis.height,
        inference_ms=inference_ms,
        service_ms=float(service_time) if service_time else None,
        retries=attempt,
    )


def parse_analysis(data: dict) -> Analysis:
    """Build an Analysis (without timings) from the JSON body of the analyze call."""
    people = [
        Person(
            x=value["boundingBox"]["x"],
            y=value["boundingBox"]["y"],
            w=value["boundingBox"]["w"],
            h=value["boundingBox"]["h"],
            confidence=value["confidence"],
        )
        for value in data["peopleResult"]["values"]
    ]
    return Analysis(
        people=people,
        width=data["metadata"]["width"],
        height=data["metadata"]["height"],
        inference_ms=0.0,
        service_ms=None,
        retries=0,
    )


def confident_people(people: list[Person], min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[Person]:
    return [person for person in people if person.confidence >= min_confidence]


def count_people(people: list[Person], min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> int:
    return len(confident_people(people, min_confidence))


def _retry_after_s(header: Optional[str]) -> float:
    try:
        return max(float(header), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_RETRY_AFTER_S
