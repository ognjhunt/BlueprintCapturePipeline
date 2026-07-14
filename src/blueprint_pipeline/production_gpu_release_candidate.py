"""Immutable release-candidate closure for the production GPU golden path.

The build-input fingerprint deliberately excludes build time, pushed digest,
and evidence locations.  Rebuilding the same declared closure therefore yields
the same input identity even though the OCI output and attestations are new.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "production_gpu_release_candidate.v1"
_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in sorted(value.items())}


def build_release_candidate_manifest(
    *,
    source_sha: str,
    clean_worktree: bool,
    build_context_digest: str,
    dockerfile_digest: str,
    base_image_ref: str,
    dependency_digests: Mapping[str, str],
    worker_source_digests: Mapping[str, str],
    model_asset_revisions: Mapping[str, str],
    runtime_contract: Mapping[str, Any],
    build_command: Sequence[str],
    image_tag: str,
    pushed_image_ref: str,
    build_timestamp: str,
    builder_identity: str,
    sbom_ref: str,
    provenance_ref: str,
) -> dict[str, Any]:
    """Bind source, build closure, OCI output, and attestations fail-closed."""

    dependencies = _mapping(dependency_digests)
    worker_sources = _mapping(worker_source_digests)
    models = _mapping(model_asset_revisions)
    command = [str(part) for part in build_command]
    build_inputs = {
        "source_sha": source_sha,
        "clean_worktree": clean_worktree is True,
        "build_context_digest": build_context_digest,
        "dockerfile_digest": dockerfile_digest,
        "base_image_ref": base_image_ref,
        "dependency_digests": dependencies,
        "worker_source_digests": worker_sources,
        "model_asset_revisions": models,
        "runtime_contract": dict(runtime_contract),
        "build_command": command,
    }
    fingerprint = "sha256:" + hashlib.sha256(_canonical(build_inputs)).hexdigest()
    blockers: list[str] = []
    if not _SHA.fullmatch(source_sha):
        blockers.append("source_sha_invalid")
    if clean_worktree is not True:
        blockers.append("build_worktree_not_clean")
    for name, value in (
        ("build_context_digest", build_context_digest),
        ("dockerfile_digest", dockerfile_digest),
    ):
        if not _DIGEST.fullmatch(value):
            blockers.append(f"{name}_invalid")
    if not _IMAGE.fullmatch(base_image_ref):
        blockers.append("base_image_not_digest_pinned")
    if not dependencies:
        blockers.append("dependency_digests_missing")
    if not worker_sources:
        blockers.append("worker_source_digests_missing")
    if not models or any(not value.strip() for value in models.values()):
        blockers.append("model_asset_revisions_missing")
    required_runtime = {
        "runtime_user",
        "uid",
        "gid",
        "supplementary_groups",
        "entrypoint",
        "command",
        "required_environment_names",
        "oci_runtime",
        "gpu_access",
    }
    missing_runtime = sorted(required_runtime - set(runtime_contract))
    blockers.extend(f"runtime_contract_missing:{name}" for name in missing_runtime)
    if not command:
        blockers.append("build_command_missing")
    if not image_tag or "@sha256:" in image_tag or image_tag.endswith(":latest"):
        blockers.append("immutable_versioned_image_tag_required")
    if not _IMAGE.fullmatch(pushed_image_ref):
        blockers.append("pushed_image_ref_not_digest_pinned")
    for name, value in (
        ("build_timestamp", build_timestamp),
        ("builder_identity", builder_identity),
        ("sbom_ref", sbom_ref),
        ("provenance_ref", provenance_ref),
    ):
        if not str(value).strip():
            blockers.append(f"{name}_missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed" if not blockers else "blocked",
        "build_input_fingerprint": fingerprint,
        "build_inputs": build_inputs,
        "image_tag": image_tag or None,
        "pushed_image_ref": pushed_image_ref or None,
        "build_timestamp": build_timestamp or None,
        "builder_identity": builder_identity or None,
        "attestations": {"sbom_ref": sbom_ref or None, "provenance_ref": provenance_ref or None},
        "blockers": blockers,
        "claim_boundary": {
            "sealed_manifest_is_not_live_runtime_proof": True,
            "pushed_digest_is_campaign_execution_identity": True,
            "later_source_change_requires_new_release_candidate": True,
        },
    }
