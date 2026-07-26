"""Freeze public OpenPI GCS checkpoint objects without downloading weights."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "openpi_checkpoint_inventory.v1"
GCS_JSON_API = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
OBJECT_FIELDS = (
    "name",
    "size",
    "md5Hash",
    "crc32c",
    "generation",
    "metageneration",
    "updated",
)


def legacy_object_manifest_sha256(objects: Sequence[Mapping[str, Any]]) -> str:
    """Reproduce the pre-generation TSV manifest used by the frozen cohort."""
    lines = sorted(
        "\t".join(
            (
                str(row.get("name", "")),
                str(row.get("size", "")),
                str(row.get("md5Hash", "")),
                str(row.get("crc32c", "")),
            )
        )
        for row in objects
    )
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def generation_manifest_sha256(objects: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {key: str(row.get(key, "")) for key in OBJECT_FIELDS}
        for row in sorted(objects, key=lambda item: str(item.get("name", "")))
    ]
    return canonical_sha256(normalized)


def _fetch_objects(
    prefix: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        query = {
            "prefix": prefix,
            "fields": f"items({','.join(OBJECT_FIELDS)}),nextPageToken",
        }
        if page_token:
            query["pageToken"] = page_token
        url = f"{GCS_JSON_API}?{urllib.parse.urlencode(query)}"
        with opener(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("items", [])
        if not isinstance(rows, list):
            raise ValueError("gcs_inventory_items_not_list")
        objects.extend(dict(row) for row in rows)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def build_checkpoint_inventory(
    cohort_path: str | Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    cohort_file = Path(cohort_path).expanduser().resolve()
    cohort = json.loads(cohort_file.read_text(encoding="utf-8"))
    if cohort.get("schema_version") != "policy_ranking_warehouse_policy_cohort.v2":
        raise ValueError("unsupported_policy_cohort_schema")
    entries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for candidate in cohort.get("primary_cohort", []):
        checkpoint_uri = str(candidate["checkpoint"])
        bucket_prefix = "gs://openpi-assets/"
        if not checkpoint_uri.startswith(bucket_prefix):
            blockers.append(f"unsupported_checkpoint_uri:{candidate['policy_id']}")
            continue
        prefix = checkpoint_uri.removeprefix(bucket_prefix).rstrip("/") + "/"
        objects = _fetch_objects(prefix, opener=opener)
        legacy_hash = legacy_object_manifest_sha256(objects)
        generation_hash = generation_manifest_sha256(objects)
        expected_legacy = str(candidate["public_object_manifest_sha256"])
        if legacy_hash != expected_legacy:
            blockers.append(f"legacy_manifest_changed:{candidate['policy_id']}")
        expected_count = int(candidate["checkpoint_object_count"])
        expected_bytes = int(candidate["checkpoint_size_bytes"])
        byte_count = sum(int(row["size"]) for row in objects)
        if len(objects) != expected_count:
            blockers.append(f"object_count_changed:{candidate['policy_id']}")
        if byte_count != expected_bytes:
            blockers.append(f"byte_count_changed:{candidate['policy_id']}")
        entries.append(
            {
                "policy_id": str(candidate["policy_id"]),
                "checkpoint_uri": checkpoint_uri,
                "object_count": len(objects),
                "size_bytes": byte_count,
                "legacy_object_manifest_sha256": legacy_hash,
                "generation_manifest_sha256": generation_hash,
                "objects": [
                    {key: str(row.get(key, "")) for key in OBJECT_FIELDS}
                    for row in sorted(objects, key=lambda item: str(item.get("name", "")))
                ],
            }
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "frozen",
        "queried_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source": GCS_JSON_API,
        "cohort_path": str(cohort_file),
        "openpi_revision": str(cohort["openpi_revision"]),
        "entries": entries,
        "blockers": blockers,
        "claim_boundary": {
            "public_object_identity_frozen": not blockers,
            "checkpoint_downloaded": False,
            "checkpoint_loaded": False,
            "learned_policy_executed": False,
            "checkpoint_license_resolved": False,
        },
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_checkpoint_inventory(args.cohort)
    write_json(Path(args.output), result)
    return 0 if result["status"] == "frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_checkpoint_inventory",
    "generation_manifest_sha256",
    "legacy_object_manifest_sha256",
]
