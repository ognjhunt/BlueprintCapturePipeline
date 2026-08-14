"""Fetch a public checkpoint on the worker over plain HTTPS, with no SDK.

The Isaac container is not a Google Cloud image and cannot be assumed to carry
gcloud; the same is true of the Hugging Face CLI before the policy venv exists.
Both are avoidable: ``storage.googleapis.com`` and ``huggingface.co`` both serve
these public artifacts over ordinary HTTPS, which was verified by listing each
one with no Authorization header at all.

So this uses urllib and nothing else.  It runs before the policy environment is
usable, needs no third-party package, and forwards no credential -- which is
also the point: every frozen candidate is public, and a token appearing here
would mean something other than the frozen public artifact was fetched.

Retries are bounded and only for transport errors.  A wrong byte count is never
retried, because repeating a fetch that produced the wrong bytes produces the
wrong bytes again.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCHEMA_VERSION = "adp009d_checkpoint_fetch.v1"
GCS_LIST_ENDPOINT = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_MEDIA_ENDPOINT = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}"
TRANSPORT_RETRY_ATTEMPTS = 3
TRANSPORT_RETRY_SLEEP_SECONDS = 5.0
READ_CHUNK_BYTES = 8 * 1024 * 1024


def _request_json(url: str) -> dict:
    with urllib.request.urlopen(  # nosec B310 - URL is built from fixed public GCS HTTPS origin
        url, timeout=120
    ) as response:
        return json.load(response)


def list_gcs_objects(bucket: str, prefix: str) -> list[dict]:
    """Every object under a prefix, following pagination to the end.

    Pagination matters: a checkpoint of this size spans several pages, and
    stopping at the first would silently fetch a partial model.
    """

    objects: list[dict] = []
    page_token = None
    while True:
        query = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if page_token:
            query["pageToken"] = page_token
        url = GCS_LIST_ENDPOINT.format(bucket=bucket) + "?" + urllib.parse.urlencode(query)
        payload = _request_json(url)
        objects.extend(payload.get("items") or [])
        page_token = payload.get("nextPageToken")
        if not page_token:
            return objects


def download_object(bucket: str, name: str, destination: Path) -> int:
    """Stream one object to disk, retrying transport failures only."""

    url = GCS_MEDIA_ENDPOINT.format(
        bucket=bucket, name=urllib.parse.quote(name, safe="")
    ) + "?alt=media"
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(TRANSPORT_RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(  # nosec B310 - URL is built from fixed public GCS HTTPS origin
                url, timeout=600
            ) as response:
                written = 0
                with destination.open("wb") as handle:
                    while True:
                        chunk = response.read(READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
            return written
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < TRANSPORT_RETRY_ATTEMPTS:
                time.sleep(TRANSPORT_RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"checkpoint_object_download_failed:{name}:{last_error}")


def fetch_gcs_prefix(*, uri: str, destination_root: Path) -> dict:
    """Fetch every object under a gs:// prefix and report exactly what landed."""

    remainder = uri.removeprefix("gs://")
    bucket, _, prefix = remainder.partition("/")
    prefix = prefix.rstrip("/") + "/"
    objects = list_gcs_objects(bucket, prefix)
    if not objects:
        raise RuntimeError(f"checkpoint_prefix_empty:{uri}")

    total = 0
    rows = []
    for entry in objects:
        name = str(entry["name"])
        relative = name[len(prefix) :]
        if not relative:
            continue
        written = download_object(bucket, name, destination_root / relative)
        declared = int(entry.get("size", 0))
        if written != declared:
            # Never retried: a fetch that produced the wrong bytes will produce
            # the wrong bytes again, and a partial model must not look complete.
            raise RuntimeError(
                f"checkpoint_object_size_mismatch:{name}:{written}!={declared}"
            )
        rows.append({"name": name, "size_bytes": written})
        total += written

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fetched",
        "uri": uri,
        "object_count": len(rows),
        "total_bytes": total,
        "credentials_used": False,
        "objects": sorted(rows, key=lambda row: row["name"]),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        raise SystemExit(
            "usage: adp009d_checkpoint_fetch_worker.py GS_URI DESTINATION RECEIPT"
        )
    uri, destination, receipt_path = arguments
    receipt = fetch_gcs_prefix(uri=uri, destination_root=Path(destination))
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(receipt_path).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"BLUEPRINT_ADP009D_CHECKPOINT_FETCHED:"
        f"{receipt['object_count']}:{receipt['total_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
