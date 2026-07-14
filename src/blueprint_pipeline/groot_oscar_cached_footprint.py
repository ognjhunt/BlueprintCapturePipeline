"""Audit measured host-image plus external-model cached worker footprint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json

SCHEMA_VERSION = "groot_oscar_cached_worker_footprint.v1"
DEFAULT_TARGET_BYTES = 30 * 1024**3
_DIGEST_REF = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")


def build_cached_footprint_audit(
    *,
    image_evidence: Mapping[str, Any],
    model_cache_verification: Mapping[str, Any],
    expected_release_ref: str,
    target_bytes: int = DEFAULT_TARGET_BYTES,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not _DIGEST_REF.fullmatch(expected_release_ref):
        blockers.append("expected_release_ref_must_be_digest_pinned")
    observed_ref = str(
        image_evidence.get("resolved_digest_ref")
        or image_evidence.get("image_ref")
        or ""
    )
    if observed_ref != expected_release_ref:
        blockers.append("cached_worker_release_ref_mismatch")
    image_bytes = image_evidence.get("local_uncompressed_size_bytes")
    if type(image_bytes) is not int or image_bytes <= 0:
        blockers.append("local_uncompressed_worker_image_size_missing")
        image_bytes = None
    if (
        model_cache_verification.get("schema_version")
        != "groot_oscar_external_model_cache_verification.v2"
        or model_cache_verification.get("status") != "passed"
        or model_cache_verification.get("checks", {}).get("models_cached_offline")
        is not True
    ):
        blockers.append("external_model_cache_not_verified")
    model_bytes = model_cache_verification.get("verified_size_bytes")
    if type(model_bytes) is not int or model_bytes <= 0:
        blockers.append("verified_external_model_cache_size_missing")
        model_bytes = None
    total = (
        int(image_bytes) + int(model_bytes)
        if image_bytes is not None and model_bytes is not None
        else None
    )
    target_met = total is not None and total <= int(target_bytes)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "blocked"
            if blockers
            else "target_met"
            if target_met
            else "measured_above_target"
        ),
        "blockers": blockers,
        "release_image_ref": expected_release_ref,
        "model_manifest_digest": model_cache_verification.get(
            "model_manifest_digest"
        ),
        "local_uncompressed_worker_image_size_bytes": image_bytes,
        "verified_external_model_cache_size_bytes": model_bytes,
        "total_cached_worker_footprint_bytes": total,
        "target_max_bytes": int(target_bytes),
        "target_met": target_met,
        "claim_boundary": {
            "registry_compressed_size_is_not_used_as_host_disk_size": True,
            "model_manifest_size_is_bound_to_verified_files": True,
            "cached_footprint_is_not_startup_latency_proof": True,
            "cached_footprint_is_not_task_success": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-evidence", required=True)
    parser.add_argument("--model-cache-verification", required=True)
    parser.add_argument("--expected-release-ref", required=True)
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    image_evidence = json.loads(Path(args.image_evidence).read_text(encoding="utf-8"))
    model_verification = json.loads(
        Path(args.model_cache_verification).read_text(encoding="utf-8")
    )
    result = build_cached_footprint_audit(
        image_evidence=image_evidence,
        model_cache_verification=model_verification,
        expected_release_ref=args.expected_release_ref,
        target_bytes=args.target_bytes,
    )
    write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "target_met" else 2


if __name__ == "__main__":
    raise SystemExit(main())
