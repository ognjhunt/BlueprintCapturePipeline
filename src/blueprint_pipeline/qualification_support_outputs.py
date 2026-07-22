"""Explicit writer and projections for optional qualification trust outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .alpha_readiness import write_alpha_readiness_summary
from .core.common import write_json


_SUPPORT_ARTIFACTS = {
    "qualification_summary": "qualification_summary.json",
    "capture_quality_summary": "capture_quality_summary.json",
    "rights_and_compliance_summary": "rights_and_compliance_summary.json",
    "buyer_trust_score": "buyer_trust_score.json",
    "recapture_requirements": "recapture_requirements.json",
    "provider_preview_status": "provider_preview_status.json",
}


def write_qualification_support_outputs(
    *,
    pipeline_dir: Path,
    launch_bundle: Mapping[str, Any],
    enabled: bool,
) -> dict[str, str]:
    """Write optional trust/readiness artifacts only on an admitted support edge."""

    if not enabled:
        return {}
    paths: dict[str, str] = {}
    for artifact_id, filename in _SUPPORT_ARTIFACTS.items():
        payload = launch_bundle.get(artifact_id)
        if not isinstance(payload, Mapping):
            raise ValueError(f"qualification_support_payload_invalid:{artifact_id}")
        path = pipeline_dir / filename
        write_json(path, payload)
        paths[artifact_id] = str(path)
    return paths


def qualification_support_artifact_uris(
    *,
    bucket: str,
    pipeline_prefix: str,
    enabled: bool,
) -> dict[str, str]:
    """Project only URIs whose optional support artifacts were admitted and written."""

    if not enabled:
        return {}
    return {
        f"{artifact_id}_uri": f"gs://{bucket}/{pipeline_prefix}/{filename}"
        for artifact_id, filename in _SUPPORT_ARTIFACTS.items()
    }


def qualification_rights_support_artifact_uris(
    bucket: str,
    pipeline_prefix: str,
    enabled: bool,
) -> dict[str, str]:
    """Project the optional compliance-summary URI into the core rights review."""

    if not enabled:
        return {}
    return {
        "rights_and_compliance_summary_uri": (
            f"gs://{bucket}/{pipeline_prefix}/rights_and_compliance_summary.json"
        )
    }


def write_alpha_readiness_summary_if_enabled(
    capture_root: Path,
    enabled: bool,
) -> None:
    if enabled:
        write_alpha_readiness_summary(capture_root=capture_root)


def qualification_support_webapp_projection(
    *,
    buyer_trust_score: Mapping[str, Any],
    launch_bundle: Mapping[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    """Return optional inline trust fields without polluting the default product sync."""

    if not enabled:
        return {}
    recapture = launch_bundle.get("recapture_requirements")
    recapture_payload = dict(recapture) if isinstance(recapture, Mapping) else {}
    return {
        "buyer_trust_score": dict(buyer_trust_score),
        "qualification_summary": dict(launch_bundle.get("qualification_summary") or {}),
        "capture_quality_summary": dict(launch_bundle.get("capture_quality_summary") or {}),
        "rights_and_compliance": dict(
            launch_bundle.get("rights_and_compliance_summary") or {}
        ),
        "missing_evidence": list(recapture_payload.get("missing_evidence") or []),
        "recapture_required": bool(recapture_payload.get("required")),
        "recapture_recommendations": list(
            recapture_payload.get("recommendations") or []
        ),
        "preview_status": launch_bundle.get("preview_status"),
    }
