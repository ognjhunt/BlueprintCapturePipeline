"""Typed verification for immutable image-build receipts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .openpi_policy_ranking_remote_build_packet import RESULT_NAME as OPENPI_RESULT_NAME


CARRIER_RESULT_NAME = "groot_oscar_carrier_remote_build_result.json"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_remote_carrier_result(
    results_dir: Path, *, packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the carrier build receipt to the exact packet and registry digest."""

    blockers: list[str] = []
    path = results_dir / CARRIER_RESULT_NAME
    payload = _load_object(path) if path.is_file() else {}
    resolved = str(payload.get("resolved_digest_ref") or "")
    expected_tag = str(packet.get("carrier_image_ref") or "")
    expected_base = str(packet.get("carrier_base_image_ref") or "")
    expected_dockerfile = str(packet.get("carrier_dockerfile_sha256") or "")
    if payload.get("schema_version") != "groot_oscar_carrier_remote_build_result.v1":
        blockers.append("carrier_remote_build_result_schema_invalid")
    if payload.get("status") != "completed" or payload.get("blockers") not in ([], ()):
        blockers.append("carrier_remote_build_not_completed")
    if payload.get("image_ref") != expected_tag:
        blockers.append("carrier_remote_build_image_ref_mismatch")
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", resolved):
        blockers.append("carrier_remote_build_digest_ref_invalid")
    else:
        expected_repository = expected_tag.split("@", 1)[0]
        if ":" in expected_repository.rsplit("/", 1)[-1]:
            expected_repository = expected_repository.rsplit(":", 1)[0]
        if resolved.split("@", 1)[0] != expected_repository:
            blockers.append("carrier_remote_build_digest_repository_mismatch")
    if payload.get("base_image_ref") != expected_base:
        blockers.append("carrier_remote_build_base_ref_mismatch")
    if payload.get("dockerfile_sha256") != expected_dockerfile:
        blockers.append("carrier_remote_build_dockerfile_sha256_mismatch")
    if payload.get("source_commit") != packet.get("source_commit"):
        blockers.append("carrier_remote_build_source_commit_mismatch")
    if payload.get("platform") != "linux/amd64":
        blockers.append("carrier_remote_build_platform_invalid")
    if payload.get("raw_secret_values_recorded") is not False:
        blockers.append("carrier_remote_build_secret_boundary_invalid")
    return {
        "schema_version": "groot_oscar_carrier_remote_build_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "resolved_digest_ref": resolved or None,
        "raw_secret_values_recorded": False,
    }


def validate_remote_openpi_result(
    results_dir: Path, *, packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the OpenPI release receipt to the exact packet and registry digest."""

    blockers: list[str] = []
    path = results_dir / OPENPI_RESULT_NAME
    payload = _load_object(path) if path.is_file() else {}
    resolved = str(payload.get("resolved_digest_ref") or "")
    expected_tag = str(packet.get("image_ref") or "")
    if payload.get("schema_version") != "openpi_policy_ranking_gpu_release.v1":
        blockers.append("openpi_remote_build_result_schema_invalid")
    if payload.get("status") != "passed" or payload.get("blockers") not in ([], ()):
        blockers.append("openpi_remote_build_not_completed")
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", resolved):
        blockers.append("openpi_remote_build_digest_ref_invalid")
    else:
        expected_repository = (
            expected_tag.rsplit("/", 1)[0] + "/" + expected_tag.rsplit("/", 1)[-1].rsplit(":", 1)[0]
        )
        if resolved.split("@", 1)[0] != expected_repository:
            blockers.append("openpi_remote_build_digest_repository_mismatch")
    bindings = {
        "source_commit": "openpi_remote_build_source_commit_mismatch",
        "dockerfile_sha256": "openpi_remote_build_dockerfile_sha256_mismatch",
        "context_manifest_sha256": "openpi_remote_build_context_manifest_sha256_mismatch",
        "openpi_revision": "openpi_remote_build_openpi_revision_mismatch",
        "menagerie_revision": "openpi_remote_build_menagerie_revision_mismatch",
    }
    for field, blocker in bindings.items():
        if payload.get(field) != packet.get(field):
            blockers.append(blocker)
    if payload.get("runnable_platform") != "linux/amd64":
        blockers.append("openpi_remote_build_platform_invalid")
    if payload.get("checkpoint_bytes_embedded") not in {0, False}:
        blockers.append("openpi_remote_build_checkpoint_boundary_invalid")
    if payload.get("interiorgs_assets_embedded") is not False:
        blockers.append("openpi_remote_build_interiorgs_boundary_invalid")
    if payload.get("raw_secret_values_recorded") is not False:
        blockers.append("openpi_remote_build_secret_boundary_invalid")
    return {
        "schema_version": "openpi_policy_ranking_remote_build_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "resolved_digest_ref": resolved or None,
        "release_evidence_path": str(path),
        "raw_secret_values_recorded": False,
    }
