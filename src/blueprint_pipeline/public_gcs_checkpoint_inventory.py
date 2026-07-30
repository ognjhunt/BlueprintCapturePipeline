"""Freeze public GCS checkpoint objects without downloading multi-gigabyte weights."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "public_gcs_checkpoint_inventory.v1"


def build_public_gcs_checkpoint_inventory(
    *, source_uri: str, payload: Any, observed_at: str
) -> dict[str, Any]:
    """Normalize a ``gcloud storage ls --json`` response to immutable identities."""

    if not source_uri.startswith("gs://") or ".." in source_uri:
        raise ValueError("public_gcs_checkpoint_uri_invalid")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("public_gcs_checkpoint_listing_invalid")
    objects: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, Mapping) or row.get("type") != "cloud_object":
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("public_gcs_checkpoint_object_metadata_invalid")
        try:
            size = int(metadata.get("size") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("public_gcs_checkpoint_object_size_invalid") from exc
        if size <= 0:
            continue
        object_row = {
            "name": str(metadata.get("name") or ""),
            "size_bytes": size,
            "generation": str(metadata.get("generation") or ""),
            "metageneration": str(metadata.get("metageneration") or ""),
            "md5_base64": str(metadata.get("md5Hash") or ""),
            "crc32c_base64": str(metadata.get("crc32c") or ""),
            "updated": str(metadata.get("updated") or ""),
        }
        if not all(
            object_row[key]
            for key in ("name", "generation", "md5_base64", "crc32c_base64", "updated")
        ):
            raise ValueError("public_gcs_checkpoint_object_identity_incomplete")
        objects.append(object_row)
    objects.sort(key=lambda row: row["name"])
    if not objects:
        raise ValueError("public_gcs_checkpoint_objects_missing")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "observed_at": observed_at,
        "source_uri": source_uri.rstrip("/"),
        "object_count": len(objects),
        "total_bytes": sum(int(row["size_bytes"]) for row in objects),
        "latest_updated": max(str(row["updated"]) for row in objects),
        "objects": objects,
        "raw_secret_values_recorded": False,
    }
    result["object_inventory_sha256"] = canonical_sha256(objects)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def freeze_public_gcs_checkpoint_inventory(
    *, source_uri: str, output_path: str | Path
) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603
        ["gcloud", "storage", "ls", "--json", "--recursive", source_uri],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout)
    result = build_public_gcs_checkpoint_inventory(
        source_uri=source_uri,
        payload=payload,
        observed_at=utc_now_iso(),
    )
    write_json(Path(output_path).expanduser().resolve(), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = freeze_public_gcs_checkpoint_inventory(
        source_uri=args.source_uri, output_path=args.output
    )
    print(
        json.dumps(
            {
                "source_uri": result["source_uri"],
                "object_count": result["object_count"],
                "total_bytes": result["total_bytes"],
                "object_inventory_sha256": result["object_inventory_sha256"],
                "manifest_sha256": result["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
