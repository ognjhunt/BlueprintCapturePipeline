"""Validation for exact-digest live worker-image evidence used by G1 bundles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json


SCHEMA_VERSION = "g1_kitchen_worker_image_runtime_evidence.v1"


def assemble_worker_image_runtime_evidence(
    *,
    image_digest: str,
    source_commit: str,
    source_dirty_patch_sha256: str,
    build_healthcheck: Mapping[str, Any],
    fast_canary: Mapping[str, Any],
    review_canary: Mapping[str, Any],
    teardown: Mapping[str, Any],
    final_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(build_healthcheck.get("runtime_metadata") or {})
    metadata["build_time_healthcheck_passed"] = build_healthcheck.get("status") == "passed"
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed",
        "image_digest": str(image_digest),
        "source_commit": str(source_commit),
        "source_dirty_patch_sha256": str(source_dirty_patch_sha256),
        "runtime_metadata": metadata,
        "fast_canary": dict(fast_canary),
        "review_canary": dict(review_canary),
        "teardown": dict(teardown),
        "final_inventory": dict(final_inventory),
        "claim_boundary": {
            "worker_image_runtime_proven": True,
            "task_success_proven": False,
            "semantic_success_proven": False,
            "physical_robot_readiness_proven": False,
        },
    }
    validation = validate_worker_image_runtime_evidence(
        evidence,
        expected_image_digest=image_digest,
        expected_source_commit=source_commit,
        expected_dirty_patch_sha256=source_dirty_patch_sha256,
    )
    evidence["status"] = validation["status"]
    evidence["blockers"] = validation["blockers"]
    return evidence


def validate_worker_image_runtime_evidence(
    value: Any,
    *,
    expected_image_digest: str | None = None,
    expected_source_commit: str | None = None,
    expected_dirty_patch_sha256: str | None = None,
) -> dict[str, Any]:
    evidence = dict(value) if isinstance(value, Mapping) else {}
    blockers: list[str] = []
    if evidence.get("schema_version") != SCHEMA_VERSION:
        blockers.append("worker_image_evidence_schema_mismatch")
    if evidence.get("status") != "passed":
        blockers.append("worker_image_evidence_not_passed")
    image_digest = str(evidence.get("image_digest") or "").lower()
    if expected_image_digest and image_digest != str(expected_image_digest).lower():
        blockers.append("worker_image_evidence_image_digest_mismatch")
    if expected_source_commit and evidence.get("source_commit") != expected_source_commit:
        blockers.append("worker_image_evidence_source_commit_mismatch")
    if (
        expected_dirty_patch_sha256
        and evidence.get("source_dirty_patch_sha256") != expected_dirty_patch_sha256
    ):
        blockers.append("worker_image_evidence_dirty_patch_mismatch")
    metadata = dict(evidence.get("runtime_metadata") or {})
    if metadata.get("image_family") != "isaac-eval-worker":
        blockers.append("worker_image_evidence_image_family_mismatch")
    if metadata.get("simulator_family") != "isaac_sim":
        blockers.append("worker_image_evidence_simulator_family_mismatch")
    if metadata.get("simulator_major_version") != 6:
        blockers.append("worker_image_evidence_simulator_major_version_mismatch")
    for field in (
        "blueprint_pipeline_imported",
        "configured_g1_asset_binding_valid",
        "build_time_healthcheck_passed",
    ):
        if metadata.get(field) is not True:
            blockers.append(f"worker_image_evidence_{field}_missing")
    fast = dict(evidence.get("fast_canary") or {})
    review = dict(evidence.get("review_canary") or {})
    for name, canary in (("fast", fast), ("review", review)):
        if canary.get("status") != "passed":
            blockers.append(f"worker_image_evidence_{name}_canary_not_passed")
        if str(canary.get("image_digest") or "").lower() != image_digest:
            blockers.append(f"worker_image_evidence_{name}_canary_digest_mismatch")
    for field in ("provider_allocation_id", "launch_nonce"):
        if not fast.get(field) or fast.get(field) != review.get(field):
            blockers.append(f"worker_image_evidence_same_allocation_{field}_mismatch")
    if review.get("width") != 640 or review.get("height") != 480:
        blockers.append("worker_image_evidence_review_canary_resolution_mismatch")
    teardown = dict(evidence.get("teardown") or {})
    if teardown.get("api_confirmed") is not True or teardown.get("terminal_state") not in {
        "not_found",
        "deleted",
        "terminated",
    }:
        blockers.append("worker_image_evidence_teardown_not_api_confirmed")
    inventory = dict(evidence.get("final_inventory") or {})
    if (
        inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
    ):
        blockers.append("worker_image_evidence_final_inventory_not_zero")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "image_digest": image_digest or None,
        "source_commit": evidence.get("source_commit"),
        "source_dirty_patch_sha256": evidence.get("source_dirty_patch_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-dirty-patch-sha256", required=True)
    parser.add_argument("--build-healthcheck", required=True, type=Path)
    parser.add_argument("--fast-canary", required=True, type=Path)
    parser.add_argument("--review-canary", required=True, type=Path)
    parser.add_argument("--teardown", required=True, type=Path)
    parser.add_argument("--final-inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    def load(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else {}

    result = assemble_worker_image_runtime_evidence(
        image_digest=args.image_digest,
        source_commit=args.source_commit,
        source_dirty_patch_sha256=args.source_dirty_patch_sha256,
        build_healthcheck=load(args.build_healthcheck),
        fast_canary=load(args.fast_canary),
        review_canary=load(args.review_canary),
        teardown=load(args.teardown),
        final_inventory=load(args.final_inventory),
    )
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
