"""Provider-neutral, fail-closed build packet for the reconstruction worker.

The packet is an admission artifact.  It never launches a builder; paid launch
must enter through ``blueprint_pipeline.paid_resource_allocator cpu-build``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import build_worker_stack_manifest


SCHEMA_VERSION = "reconstruction_worker_build_packet.v1"
ALLOCATOR_ENTRYPOINT = [
    "python",
    "-m",
    "blueprint_pipeline.paid_resource_allocator",
    "cpu-build",
]

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSIONED_IMAGE = re.compile(r"^[^\s@]+:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class ReconstructionWorkerBuildPacketError(ValueError):
    pass


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReconstructionWorkerBuildPacketError("packet_not_json_serializable") from exc


def prepare_reconstruction_worker_build_packet(
    *,
    worker_stack_manifest: Mapping[str, Any],
    image_ref: str,
    source_commit_sha: str,
    source_tree_digest: str,
    source_worktree_dirty: bool,
    build_recipe_digest: str | None,
    dependency_lock_digest: str | None,
    license_review_receipt_digest: str | None,
    max_spend_usd: float | None,
    ttl_seconds: int | None,
    retry_cap: int | None,
    authority_id: str | None,
    timestamp: str,
) -> dict[str, Any]:
    """Compile immutable build admission without performing paid execution."""

    manifest = build_worker_stack_manifest(worker_stack_manifest)
    blockers: list[str] = []
    if _COMMIT.fullmatch(source_commit_sha) is None:
        blockers.append("worker_build_source_commit_invalid")
    if manifest["source_commit_sha"] != source_commit_sha:
        blockers.append("worker_build_stack_source_commit_mismatch")
    if _DIGEST.fullmatch(source_tree_digest) is None:
        blockers.append("worker_build_source_tree_digest_invalid")
    if source_worktree_dirty:
        blockers.append("worker_build_requires_clean_immutable_commit")
    if _VERSIONED_IMAGE.fullmatch(image_ref) is None or image_ref.rsplit(":", 1)[-1] in {
        "latest",
        "dev",
        "test",
        "local",
    }:
        blockers.append("worker_build_image_ref_not_versioned")
    for value, blocker in (
        (build_recipe_digest, "worker_build_recipe_digest_missing"),
        (dependency_lock_digest, "worker_dependency_lock_digest_missing"),
        (license_review_receipt_digest, "worker_license_review_receipt_missing"),
    ):
        if _DIGEST.fullmatch(str(value or "")) is None:
            blockers.append(blocker)
    if (
        isinstance(max_spend_usd, bool)
        or not isinstance(max_spend_usd, (int, float))
        or not math.isfinite(float(max_spend_usd))
        or float(max_spend_usd) <= 0
    ):
        blockers.append("worker_build_explicit_budget_missing")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        blockers.append("worker_build_explicit_ttl_missing")
    if isinstance(retry_cap, bool) or not isinstance(retry_cap, int) or retry_cap < 0:
        blockers.append("worker_build_explicit_retry_cap_missing")
    if not isinstance(authority_id, str) or not authority_id.strip():
        blockers.append("worker_build_paid_authority_missing")
    packet = {
        "schema_version": SCHEMA_VERSION,
        "packet_kind": "reconstruction_worker_image",
        "status": "ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "worker_stack_manifest_digest": manifest["worker_stack_manifest_digest"],
        "image_ref": image_ref,
        "source_commit_sha": source_commit_sha,
        "source_tree_digest": source_tree_digest,
        "source_worktree_dirty": source_worktree_dirty,
        "build_recipe_digest": build_recipe_digest,
        "dependency_lock_digest": dependency_lock_digest,
        "license_review_receipt_digest": license_review_receipt_digest,
        "allocator_entrypoint": ALLOCATOR_ENTRYPOINT,
        "canonical_paid_resource_seam_required": True,
        "direct_provider_launcher_allowed": False,
        "provider_identity": None,
        "max_spend_usd": max_spend_usd,
        "ttl_seconds": ttl_seconds,
        "retry_cap": retry_cap,
        "authority_id": authority_id,
        "required_outputs": [
            "reconstruction_worker_build_receipt.v1",
            "reconstruction_worker_smoke_test_receipt.v1",
            "provider_teardown_receipt.v1",
            "provider_zero_verification.v1",
        ],
        "paid_execution_started": False,
        "allocation_success_is_scientific_success": False,
        "build_success_is_scientific_success": False,
        "timestamp": timestamp,
    }
    packet["build_packet_digest"] = canonical_digest(
        packet, digest_field="build_packet_digest"
    )
    return _clone(packet)


__all__ = [
    "ALLOCATOR_ENTRYPOINT",
    "ReconstructionWorkerBuildPacketError",
    "SCHEMA_VERSION",
    "prepare_reconstruction_worker_build_packet",
]
