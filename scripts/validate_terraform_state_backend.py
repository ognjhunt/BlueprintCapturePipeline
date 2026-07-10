#!/usr/bin/env python3
"""Validate the remote GCS Terraform state bucket before init/apply."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


MINIMUM_RETENTION_SECONDS = 30 * 24 * 60 * 60


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_bucket(
    payload: Mapping[str, Any], *, expected_bucket: str, expected_kms_key: str
) -> list[str]:
    blockers: list[str] = []
    expected_name = expected_bucket.removeprefix("gs://").rstrip("/")
    actual_name = str(payload.get("name") or "").removeprefix("gs://").rstrip("/")
    if not expected_name or actual_name != expected_name:
        blockers.append("terraform_state_bucket_name_mismatch")
    location = str(payload.get("location") or "").upper()
    if not location.startswith("US"):
        blockers.append("terraform_state_bucket_not_us")
    iam = _mapping(payload.get("iamConfiguration"))
    uniform = _mapping(iam.get("uniformBucketLevelAccess"))
    if uniform.get("enabled") is not True:
        blockers.append("terraform_state_uniform_bucket_access_not_enabled")
    if str(iam.get("publicAccessPrevention") or "").lower() != "enforced":
        blockers.append("terraform_state_public_access_prevention_not_enforced")
    if _mapping(payload.get("versioning")).get("enabled") is not True:
        blockers.append("terraform_state_versioning_not_enabled")
    retention = _mapping(payload.get("retentionPolicy"))
    try:
        retention_seconds = int(retention.get("retentionPeriod") or 0)
    except (TypeError, ValueError):
        retention_seconds = 0
    if retention_seconds < MINIMUM_RETENTION_SECONDS:
        blockers.append("terraform_state_retention_below_30_days")
    configured_kms = str(
        _mapping(payload.get("encryption")).get("defaultKmsKeyName") or ""
    )
    if not expected_kms_key or configured_kms != expected_kms_key:
        blockers.append("terraform_state_cmek_mismatch")
    return blockers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="Bucket metadata JSON; stdin if omitted")
    parser.add_argument("--expected-bucket", required=True)
    parser.add_argument("--expected-kms-key", required=True)
    args = parser.parse_args(argv)
    try:
        text = (
            args.metadata.read_text(encoding="utf-8")
            if args.metadata is not None
            else sys.stdin.read()
        )
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("bucket_metadata_not_object")
        blockers = validate_bucket(
            payload,
            expected_bucket=args.expected_bucket,
            expected_kms_key=args.expected_kms_key,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"[terraform-state] ERROR {exc}", file=sys.stderr)
        return 1
    if blockers:
        for blocker in blockers:
            print(f"[terraform-state] ERROR {blocker}", file=sys.stderr)
        return 1
    print("[terraform-state] remote locked CMEK backend verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
