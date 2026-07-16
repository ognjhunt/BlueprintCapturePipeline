"""Validate copied GR00T + OSCAR image-build result sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REMOTE_BUILD_REQUIRED_RESULTS = (
    "groot_oscar_thin_remote_build_result.json",
    "foundation_buildx_metadata.json",
    "release_buildx_metadata.json",
    "foundation_registry_diagnostic.json",
    "release_registry_diagnostic.json",
    "release_sbom.spdx.json",
    "release_provenance.json",
    "release_layer_report.json",
    "release_buildkit_sbom_attestation.json",
    "release_buildkit_provenance_attestation.json",
    "release_buildkit_attestation_index.json",
    "release_supply_chain_manifest.json",
    "release_supply_chain_disk_admission.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_remote_build_results(results_dir: Path) -> dict[str, Any]:
    """Require copied results appropriate to the foundation build mode."""

    blockers: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    try:
        result_preview = _load_object(
            results_dir / "groot_oscar_thin_remote_build_result.json"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        result_preview = {}
    try:
        foundation_diagnostic = _load_object(
            results_dir / "foundation_registry_diagnostic.json"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        foundation_diagnostic = {}
    serverless = result_preview.get("serverless_worker_contract")
    serverless = serverless if isinstance(serverless, Mapping) else {}
    thin_release = result_preview.get("thin_release_contract")
    thin_release = thin_release if isinstance(thin_release, Mapping) else {}
    foundation_ref = str(result_preview.get("foundation_image_ref") or "")
    digest = foundation_ref.rsplit("@sha256:", 1)[-1]
    reused_digest_foundation = bool(
        result_preview.get("status") == "completed"
        and serverless.get("status") == "passed"
        and serverless.get("worker_source_packaged") is True
        and serverless.get("worker_command_packaged") is True
        and serverless.get("runpod_sdk_exactly_pinned") is True
        and serverless.get("models_externalized") is True
        and thin_release.get("status") == "passed"
        and thin_release.get("release_delta_budget_passed") is True
        and thin_release.get("models_externalized") is True
        and thin_release.get("foundation_image_ref") == foundation_ref
        and foundation_diagnostic.get("status") == "completed"
        and not foundation_diagnostic.get("blockers")
        and foundation_diagnostic.get("image_ref") == foundation_ref
        and foundation_diagnostic.get("resolved_digest_ref") == foundation_ref
        and "@sha256:" in foundation_ref
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )
    required_results = tuple(
        name
        for name in REMOTE_BUILD_REQUIRED_RESULTS
        if name != "foundation_buildx_metadata.json" or not reused_digest_foundation
    )
    for name in required_results:
        path = results_dir / name
        if not path.is_file():
            blockers.append(f"remote_build_result_missing:{name}")
            continue
        try:
            payloads[name] = _load_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append(f"remote_build_result_invalid:{name}")
    result = payloads.get("groot_oscar_thin_remote_build_result.json")
    if result is not None and result.get("status") != "completed":
        blockers.append("remote_build_result_not_completed")
    supply_chain = payloads.get("release_supply_chain_manifest.json")
    if supply_chain is not None and supply_chain.get("status") != "passed":
        blockers.append("remote_build_supply_chain_not_passed")
    disk_admission = payloads.get("release_supply_chain_disk_admission.json")
    if disk_admission is not None and disk_admission.get("status") != "passed":
        blockers.append("remote_build_supply_chain_disk_admission_not_passed")
    return {
        "schema_version": "groot_oscar_remote_build_results_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "required_results": list(required_results),
        "digest_pinned_foundation_reused": reused_digest_foundation,
    }
