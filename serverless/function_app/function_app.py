"""Azure Function (Part 3): count the people in every frame uploaded to the raw container.

The blob trigger uses an Event Grid event subscription as its source (the only
blob trigger supported by the Flex Consumption plan). It listens only to
raw-frames; the results go to the $web container (see frame_processing.py), so
the function can never trigger itself.

App settings:
    FRAMES_STORAGE           connection string of the storage account (raw-frames and $web)
    AZURE_VISION_ENDPOINT    AI Vision endpoint
    AZURE_VISION_KEY         AI Vision key
    VISION_MIN_CONFIDENCE    optional, default 0.5
"""

import logging
import os

import azure.functions as func
from azure.storage.blob import BlobServiceClient, ContainerClient

from frame_processing import RAW_CONTAINER, WEB_CONTAINER, Timer, process_frame
from vision import DEFAULT_MIN_CONFIDENCE

app = func.FunctionApp()

_web: ContainerClient | None = None  # reused by every invocation in this worker
_invocations = 0  # 0 -> the next invocation is the first one (cold start) of this worker


def web_container() -> ContainerClient:
    global _web
    if _web is None:
        service = BlobServiceClient.from_connection_string(os.environ["FRAMES_STORAGE"])
        _web = service.get_container_client(WEB_CONTAINER)
    return _web


@app.blob_trigger(
    arg_name="blob",
    path=RAW_CONTAINER + "/{name}",
    connection="FRAMES_STORAGE",
    source=func.BlobSource.EVENT_GRID,
)
def process_raw_frame(blob: func.InputStream) -> None:
    global _invocations
    timer = Timer.now()
    cold_start = _invocations == 0
    _invocations += 1

    # blob.name is "raw-frames/<frame_id>.jpg"
    container, _, blob_name = blob.name.partition("/")
    if container != RAW_CONTAINER or not blob_name.endswith(".jpg"):
        logging.warning("Ignoring %s: not a .jpg in %s", blob.name, RAW_CONTAINER)
        return

    try:
        result = process_frame(
            blob_name,
            blob.read(),
            web_container(),
            endpoint=os.environ["AZURE_VISION_ENDPOINT"],
            key=os.environ["AZURE_VISION_KEY"],
            min_confidence=float(os.getenv("VISION_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)),
            timer=timer,
            cold_start=cold_start,
        )
    except ValueError as e:  # blob name is not a frame id
        logging.warning("Ignoring %s: %s", blob.name, e)
        return
    if result:
        logging.info(
            "%s: %d people, inference %.0f ms, response %.0f ms, cold start %s",
            result.frame_id, result.n_people, result.inference_ms, result.response_time_ms, cold_start,
        )
