"""Normalize immutable Isaac registry metadata for reconstruction canaries."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from .isaac_worker_image_manifest import SCHEMA_VERSION as IMAGE_MANIFEST_SCHEMA
from .isaac_worker_source_overlay import BUILD_METHOD as SOURCE_OVERLAY_BUILD_METHOD


SCHEMA_VERSION = "reconstruction_isaac_image_release.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ReconstructionIsaacImageReleaseError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def build_reconstruction_isaac_image_release(
    *, image_manifest: Mapping[str, Any], expected_source_commit: str
) -> dict[str, Any]:
    """Bind one resolved Isaac image to exact clean source without runtime claims."""

    manifest = json.loads(json.dumps(dict(image_manifest)))
    expected = str(expected_source_commit or "").strip().lower()
    identity_value = manifest.get("worker_build_identity")
    identity = dict(identity_value) if isinstance(identity_value, Mapping) else {}
    resolved = str(manifest.get("resolved_digest_ref") or "").strip()
    blockers: list[str] = []
    if manifest.get("schema_version") != IMAGE_MANIFEST_SCHEMA:
        blockers.append("reconstruction_isaac_image_manifest_schema_invalid")
    if manifest.get("status") != "completed":
        blockers.append("reconstruction_isaac_image_manifest_not_completed")
    if _IMAGE.fullmatch(resolved) is None:
        blockers.append("reconstruction_isaac_image_not_digest_pinned")
    if manifest.get("runnable_platform") != "linux/amd64":
        blockers.append("reconstruction_isaac_image_platform_invalid")
    if manifest.get("raw_secret_values_recorded") is not False:
        blockers.append("reconstruction_isaac_image_manifest_secret_boundary_invalid")
    if _COMMIT.fullmatch(expected) is None:
        blockers.append("reconstruction_isaac_image_expected_commit_invalid")
    if identity.get("status") != "verified":
        blockers.append("reconstruction_isaac_image_build_identity_unverified")
    if identity.get("source_commit") != expected:
        blockers.append("reconstruction_isaac_image_source_commit_mismatch")
    if (
        identity.get("source_dirty_patch_sha256")
        != CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
    ):
        blockers.append("reconstruction_isaac_image_dirty_overlay_forbidden")
    if identity.get("worker_image_family") != "isaac-eval-worker":
        blockers.append("reconstruction_isaac_image_family_invalid")
    if identity.get("isaac_sim_major_version") != 6:
        blockers.append("reconstruction_isaac_image_isaac_major_invalid")
    if identity.get("build_method") == SOURCE_OVERLAY_BUILD_METHOD and (
        identity.get("source_layer_matches_last_registry_layer") is not True
        or _IMAGE.fullmatch(str(identity.get("base_image_digest") or "")) is None
        or _DIGEST.fullmatch(str(identity.get("source_manifest_sha256") or "")) is None
        or _DIGEST.fullmatch(str(identity.get("source_layer_digest") or "")) is None
    ):
        blockers.append("reconstruction_isaac_image_source_overlay_lineage_invalid")
    if identity.get("isaac_sim_version") != "6.0.1":
        blockers.append("reconstruction_isaac_image_isaac_version_invalid")
    release = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "resolved_image_digest": resolved or None,
        "source_commit_sha": identity.get("source_commit"),
        "source_dirty_patch_sha256": identity.get("source_dirty_patch_sha256"),
        "runnable_platform": manifest.get("runnable_platform"),
        "worker_image_family": identity.get("worker_image_family"),
        "isaac_sim_major_version": identity.get("isaac_sim_major_version"),
        "isaac_sim_version": identity.get("isaac_sim_version"),
        "image_manifest": manifest,
        "image_manifest_digest": canonical_digest(manifest),
        "build_receipt_status": "registry_config_exact_source_verified",
        "runtime_smoke_completed": False,
        "provider_startup_proven": False,
        "isaac_compatibility_proven": False,
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_isaac_image_release_engineering_only",
    }
    release["image_release_digest"] = canonical_digest(
        release, digest_field="image_release_digest"
    )
    return release


def validate_reconstruction_isaac_image_release(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the receipt from embedded registry evidence and reject drift."""

    supplied = json.loads(json.dumps(dict(value)))
    manifest = supplied.get("image_manifest")
    if not isinstance(manifest, Mapping):
        raise ReconstructionIsaacImageReleaseError(
            ["reconstruction_isaac_image_release_manifest_missing"]
        )
    expected = str(supplied.get("source_commit_sha") or "")
    rebuilt = build_reconstruction_isaac_image_release(
        image_manifest=manifest, expected_source_commit=expected
    )
    if supplied != rebuilt:
        raise ReconstructionIsaacImageReleaseError(
            ["reconstruction_isaac_image_release_replay_mismatch"]
        )
    if rebuilt["status"] != "passed":
        raise ReconstructionIsaacImageReleaseError(
            [
                "reconstruction_isaac_image_release_blocked",
                *[str(code) for code in rebuilt["blockers"]],
            ]
        )
    return rebuilt


__all__ = [
    "ReconstructionIsaacImageReleaseError",
    "SCHEMA_VERSION",
    "build_reconstruction_isaac_image_release",
    "validate_reconstruction_isaac_image_release",
]
