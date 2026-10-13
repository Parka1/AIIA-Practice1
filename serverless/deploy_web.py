"""Publish the serverless dashboard to the $web container of the static website.

Uploads serverless/web/index.html and creates the result CSV files (empty, with
their header) that the Azure Function appends to. The teaching team's demo page
that was there before is kept as button_demo.html.

--reset-data deletes the results and processed frames, to start a clean
measurement run; it never touches the raw container.
"""

import argparse
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

from common.results import FIELDNAMES
from serverless.credentials import load_credentials

FUNCTION_DIR = Path(__file__).resolve().parent / "function_app"
sys.path.insert(0, str(FUNCTION_DIR))
from frame_processing import (  # noqa: E402
    METRICS_BLOB, METRICS_FIELDS, NO_CACHE, RESULTS_BLOB, WEB_CONTAINER, create_csv_blob,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
DEMO_BACKUP = "button_demo.html"
DASHBOARD_MARKER = b"Smart Lighting Monitor"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset-data", action="store_true", help="delete data/*.csv and frames/* in $web first")
    args = parser.parse_args()

    (connection_string,) = load_credentials("storage_connection_string")
    service = BlobServiceClient.from_connection_string(connection_string)
    web = service.get_container_client(WEB_CONTAINER)

    index = web.get_blob_client("index.html")
    if index.exists():
        current = index.download_blob().readall()
        if DASHBOARD_MARKER not in current and not web.get_blob_client(DEMO_BACKUP).exists():
            web.upload_blob(DEMO_BACKUP, current, content_settings=ContentSettings(content_type="text/html"))
            print(f"previous index.html kept as {DEMO_BACKUP}")

    web.upload_blob(
        "index.html",
        (WEB_DIR / "index.html").read_bytes(),
        overwrite=True,
        content_settings=ContentSettings(content_type="text/html; charset=utf-8", cache_control=NO_CACHE),
    )
    print("uploaded index.html")

    if args.reset_data:
        deleted = 0
        for prefix in ("data/", "frames/"):
            for blob in web.list_blobs(name_starts_with=prefix):
                web.delete_blob(blob.name)
                deleted += 1
        print(f"deleted {deleted} result blobs")

    create_csv_blob(web, RESULTS_BLOB, ",".join(FIELDNAMES) + "\n")
    create_csv_blob(web, METRICS_BLOB, ",".join(METRICS_FIELDS) + "\n")
    print(f"result files ready: {RESULTS_BLOB}, {METRICS_BLOB}")

    static_website = service.get_service_properties()["static_website"]
    if not static_website.enabled:
        print("warning: static website hosting is disabled in the storage account")


if __name__ == "__main__":
    main()
