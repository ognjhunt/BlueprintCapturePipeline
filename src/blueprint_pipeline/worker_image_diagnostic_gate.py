"""Fail-closed registry-diagnostic consumption for paid worker-image launches.

The 2026-07-11 live canary kept a 900s startup cutoff because the runner read
the stale generic default diagnostic instead of the diagnostic for the exact
selected digest, so the evidence-derived 1800s recommendation never applied.
This module binds a selected worker image to its registry manifest diagnostic
explicitly and validates it before any staging or provider create call.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from . import isaac_worker_image_manifest as image_manifest

ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV = "BLUEPRINT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC"
DEFAULT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC = "output/isaac_worker_image_manifest_diagnostic.json"
#: Upper sanity bound for a diagnostic's startup-timeout recommendation.
MAX_DIAGNOSTIC_RECOMMENDED_STARTUP_TIMEOUT_SECONDS = 24 * 3600
_DIGEST_IMAGE_RE = re.compile(r"^.+@(?P<digest>sha256:[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def isaac_worker_image_size_diagnostic(
    image_ref: str, explicit_path: str | Path | None = None
) -> dict:
    """Load the registry manifest diagnostic that describes ``image_ref``.

    ``explicit_path`` (the ``--worker-image-manifest-diagnostic`` CLI argument)
    takes precedence over the env override and the mutable default output path,
    so a paid launch of an exact digest can bind its evidence file explicitly.
    """
    explicit_arg = _string(str(explicit_path)) if explicit_path else ""
    explicit_env = _string(os.getenv(ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV))
    selected = explicit_arg or explicit_env or DEFAULT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC
    path = Path(selected).expanduser().resolve()
    base = {
        "env_var": ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV,
        "path": str(path),
        "path_source": (
            "cli_argument" if explicit_arg else "env" if explicit_env else "default_output_path"
        ),
        "path_present": path.is_file(),
        "raw_secret_values_recorded": False,
    }
    if not path.is_file():
        return {
            **base,
            "status": "missing",
            "metadata_available_for_selected_image": False,
        }
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "unreadable",
            "metadata_available_for_selected_image": False,
            "error_type": type(exc).__name__,
        }
    base["diagnostic_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    base["diagnostic_bytes"] = len(raw_bytes)
    if not isinstance(payload, dict):
        return {
            **base,
            "status": "malformed",
            "metadata_available_for_selected_image": False,
        }
    manifest = dict(payload)
    manifest_image_ref = _string(manifest.get("image_ref"))
    resolved_digest_ref = _string(manifest.get("resolved_digest_ref"))
    matching_refs = {ref for ref in (manifest_image_ref, resolved_digest_ref) if ref}
    if image_ref and not matching_refs:
        # A diagnostic that names no image cannot be evidence for any image.
        return {
            **base,
            "status": "ignored_image_ref_unverifiable",
            "metadata_available_for_selected_image": False,
            "selected_image_ref": image_ref,
        }
    if matching_refs and image_ref and image_ref not in matching_refs:
        return {
            **base,
            "status": "ignored_image_ref_mismatch",
            "metadata_available_for_selected_image": False,
            "manifest_image_ref": manifest_image_ref,
            "manifest_resolved_digest_ref": resolved_digest_ref or None,
            "selected_image_ref": image_ref,
        }
    if "@sha256:" in image_ref and image_ref != resolved_digest_ref:
        # An exact digest selection must bind to the diagnostic's resolved
        # digest identity, not merely to a same-string tag field.
        return {
            **base,
            "status": "ignored_image_ref_mismatch",
            "metadata_available_for_selected_image": False,
            "manifest_image_ref": manifest_image_ref,
            "manifest_resolved_digest_ref": resolved_digest_ref or None,
            "selected_image_ref": image_ref,
        }
    return {
        **base,
        "status": _string(manifest.get("status")) or None,
        "schema_version": _string(manifest.get("schema_version")) or None,
        "metadata_available_for_selected_image": True,
        "image_ref": manifest_image_ref or image_ref,
        "resolved_digest": manifest.get("resolved_digest"),
        "resolved_digest_ref": resolved_digest_ref or None,
        "runnable_platform": _string(manifest.get("runnable_platform")) or None,
        "layer_count": manifest.get("layer_count"),
        "total_compressed_size_bytes": manifest.get("total_compressed_size_bytes"),
        "largest_layer_size_bytes": manifest.get("largest_layer_size_bytes"),
        "layers_over_1gb": manifest.get("layers_over_1gb"),
        "large_image_pull_risk": manifest.get("large_image_pull_risk"),
        "split_layer_layout_suitable": manifest.get("split_layer_layout_suitable"),
        "recommended_startup_no_runtime_timeout_seconds": manifest.get(
            "recommended_startup_no_runtime_timeout_seconds"
        ),
        "proof_boundary": (
            "Worker image manifest metadata only. This does not prove container "
            "startup, Isaac Sim execution, rendered RGB quality, WAM quality, or "
            "robot readiness."
        ),
    }


def worker_image_diagnostic_paid_validation(
    *,
    selected_image: str,
    diagnostic: Mapping[str, Any],
    allow_paid: bool,
    image_startup_canary: bool,
) -> dict:
    """Fail-closed validation of the registry diagnostic for paid launches.

    A paid launch of an exact ``@sha256:`` image, and every paid image-startup
    canary, must consume a completed diagnostic for exactly the selected image
    so the evidence-derived startup timeout floor is actually enforced. A stale
    tag-only artifact, mismatched digest, blocked/malformed diagnostic, missing
    size metadata, or unknown schema blocks before staging or provider create.
    """
    selected_image_text = str(selected_image or "").strip()
    digest_match = _DIGEST_IMAGE_RE.fullmatch(selected_image_text)
    digest_pinned = digest_match is not None
    selected_digest = digest_match.group("digest") if digest_match else ""
    required = bool(allow_paid and (digest_pinned or image_startup_canary))
    blockers: list[str] = []
    if required:
        if diagnostic.get("path_source") != "cli_argument":
            blockers.append("worker_image_diagnostic_explicit_path_required")
        if image_startup_canary and not digest_pinned:
            blockers.append("paid_image_startup_canary_requires_digest_pinned_image")
        status = _string(diagnostic.get("status"))
        if diagnostic.get("metadata_available_for_selected_image") is not True:
            if status == "missing":
                blockers.append("worker_image_diagnostic_missing_for_paid_launch")
            elif status == "unreadable":
                blockers.append("worker_image_diagnostic_unreadable")
            elif status == "malformed":
                blockers.append("worker_image_diagnostic_malformed")
            elif status in {"ignored_image_ref_mismatch", "ignored_image_ref_unverifiable"}:
                blockers.append("worker_image_diagnostic_image_ref_mismatch")
            else:
                blockers.append("worker_image_diagnostic_unavailable_for_selected_image")
        else:
            if diagnostic.get("schema_version") not in image_manifest.SUPPORTED_SCHEMA_VERSIONS:
                blockers.append("worker_image_diagnostic_schema_unsupported")
            if status != "completed":
                blockers.append("worker_image_diagnostic_status_not_completed")
            if digest_pinned and _string(diagnostic.get("resolved_digest_ref")) != (
                selected_image_text
            ):
                blockers.append("worker_image_diagnostic_image_ref_mismatch")
            if digest_pinned and _string(diagnostic.get("resolved_digest")) != selected_digest:
                blockers.append("worker_image_diagnostic_resolved_digest_mismatch")
            if (
                _string(diagnostic.get("runnable_platform"))
                != image_manifest.RUNNABLE_PLATFORM
            ):
                blockers.append("worker_image_diagnostic_runnable_platform_unverified")
            layer_count = _positive_int(diagnostic.get("layer_count"))
            total_bytes = _positive_int(
                diagnostic.get("total_compressed_size_bytes")
            )
            largest_bytes = _positive_int(
                diagnostic.get("largest_layer_size_bytes")
            )
            layers_over_1gb = diagnostic.get("layers_over_1gb")
            layer_count_metadata_present = bool(
                type(layers_over_1gb) is int and layers_over_1gb >= 0
            )
            layer_counts_valid = bool(
                layer_count_metadata_present
                and layer_count is not None
                and layers_over_1gb <= layer_count
            )
            if not all((layer_count, total_bytes, largest_bytes)) or not layer_count_metadata_present:
                blockers.append("worker_image_diagnostic_size_metadata_missing")
            elif not layer_counts_valid or largest_bytes > total_bytes:
                blockers.append("worker_image_diagnostic_size_policy_inconsistent")
            recommended = _positive_int(
                diagnostic.get("recommended_startup_no_runtime_timeout_seconds")
            )
            if (
                recommended is None
                or recommended <= 0
                or recommended > MAX_DIAGNOSTIC_RECOMMENDED_STARTUP_TIMEOUT_SECONDS
            ):
                blockers.append("worker_image_diagnostic_timeout_recommendation_invalid")
            if total_bytes and largest_bytes and layer_counts_valid:
                derived = image_manifest.derive_manifest_size_policy(
                    total_compressed_size_bytes=total_bytes,
                    largest_layer_size_bytes=largest_bytes,
                    layers_over_1gb=layers_over_1gb,
                )
                if any(
                    diagnostic.get(key) != expected
                    for key, expected in derived.items()
                ):
                    blockers.append("worker_image_diagnostic_size_policy_inconsistent")
            if (
                not _string(diagnostic.get("path"))
                or _positive_int(diagnostic.get("diagnostic_bytes")) is None
                or not _SHA256_RE.fullmatch(
                    _string(diagnostic.get("diagnostic_sha256"))
                )
            ):
                blockers.append("worker_image_diagnostic_evidence_identity_missing")
    return {
        "required": required,
        "digest_pinned_image": digest_pinned,
        "image_startup_canary": bool(image_startup_canary),
        "status": (
            "passed" if required and not blockers else "blocked"
            if blockers
            else "not_required"
        ),
        "blockers": blockers,
        "debug_override_supported": False,
        "diagnostic_path": diagnostic.get("path"),
        "diagnostic_sha256": diagnostic.get("diagnostic_sha256"),
        "diagnostic_bytes": diagnostic.get("diagnostic_bytes"),
        "diagnostic_resolved_digest": diagnostic.get("resolved_digest"),
        "claim_boundary": (
            "Diagnostic validation proves registry metadata identity only. It does "
            "not prove the provider host, CUDA runtime, Isaac, rendering, or task "
            "execution is healthy."
        ),
    }


def effective_startup_no_runtime_timeout(
    requested_seconds: int,
    image_size_diagnostic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the validated registry recommendation as a non-bypassable floor."""
    requested = max(0, int(requested_seconds or 0))
    diagnostic = (
        image_size_diagnostic
        if isinstance(image_size_diagnostic, Mapping)
        else {}
    )
    derived = image_manifest.derive_manifest_size_policy(
        total_compressed_size_bytes=diagnostic.get("total_compressed_size_bytes"),
        largest_layer_size_bytes=diagnostic.get("largest_layer_size_bytes"),
        layers_over_1gb=diagnostic.get("layers_over_1gb"),
    )
    derived_recommended = derived["recommended_startup_no_runtime_timeout_seconds"]
    declared_recommended = diagnostic.get(
        "recommended_startup_no_runtime_timeout_seconds"
    )
    recommended = (
        derived_recommended
        if type(derived_recommended) is int
        else declared_recommended
        if type(declared_recommended) is int and declared_recommended > 0
        else 0
    )
    effective = max(requested, recommended)
    return {
        "requested_seconds": requested,
        "image_manifest_recommended_seconds": recommended or None,
        "effective_seconds": effective,
        "raised_to_image_manifest_floor": effective > requested,
        "image_manifest_diagnostic_path": diagnostic.get("path"),
        "image_manifest_diagnostic_sha256": diagnostic.get("diagnostic_sha256"),
        "image_manifest_resolved_digest": diagnostic.get("resolved_digest"),
        "disabled": effective == 0,
        "claim_boundary": (
            "This timeout protects large-image startup from premature teardown. It does "
            "not prove the provider host, CUDA runtime, Isaac, rendering, or task "
            "execution is healthy."
        ),
    }
