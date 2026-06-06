"""Deterministic QA for privacy-safe provider-preview packets.

The validator proves local artifact lineage and production raw-path policy. It
does not call providers, upload media, sync WebApp, or upgrade hosted/runtime
claims.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import PipelineError, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


PROVIDER_PREVIEW_QA_SCHEMA_VERSION = "provider_preview_qa_manifest.v1"
WEBAPP_UPSTREAM_ID_FIELDS = (
    "site_submission_id",
    "request_id",
    "buyer_request_id",
    "capture_job_id",
)

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "local_provider_preview_packet_qa_only",
    "local_repo_proof_only": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "hosted_session_proven": False,
    "simulator_execution_proven": False,
    "owner_gpu_simulator_execution_proven": False,
    "robot_readiness_proven": False,
    "public_claim_upgrade_allowed": False,
}

PRIVACY_SAFE_SOURCE_IDS = {
    "worldlabs_input_video_uri",
    "world_model_video_uri",
    "privacy_processed_video_uri",
    "privacy_safe_world_model_input",
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _uri_endswith(uri: str, suffixes: Sequence[str]) -> bool:
    normalized = uri.rstrip("/")
    return any(normalized.endswith(suffix) for suffix in suffixes)


def _status_is_complete(value: Any) -> bool:
    return _string(value).lower() in {
        "ready",
        "passed",
        "completed",
        "complete",
        "person_removed",
        "no_people_detected",
        "verified",
        "succeeded",
    }


def _append_unique(target: List[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _artifact_status(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.is_file()}


def _canonical_rgb_uri(canonical_package: Mapping[str, Any]) -> str:
    conditioning = _mapping(canonical_package.get("conditioning"))
    rgb_video = _mapping(conditioning.get("rgb_video"))
    privacy_safe = _mapping(rgb_video.get("privacy_safe_world_model_input"))
    return _string(privacy_safe.get("uri"))


def _adapter_rgb_uri(adapter_input: Mapping[str, Any]) -> str:
    conditioning = _mapping(adapter_input.get("conditioning_inputs"))
    rgb_video = _mapping(conditioning.get("rgb_video"))
    return _string(rgb_video.get("uri"))


def _latest_webapp_sync_stage(webapp_sync: Mapping[str, Any]) -> Dict[str, Any]:
    syncs = webapp_sync.get("syncs")
    if isinstance(syncs, Mapping) and syncs:
        latest_stage = _string(webapp_sync.get("latest_stage"))
        if latest_stage and isinstance(syncs.get(latest_stage), Mapping):
            return dict(syncs[latest_stage])
        for value in reversed(list(syncs.values())):
            if isinstance(value, Mapping):
                return dict(value)
    return dict(webapp_sync)


def _webapp_attachment_payload(
    *,
    webapp_sync: Mapping[str, Any],
    provider_status: Mapping[str, Any],
    provider_run: Mapping[str, Any],
    fallback_sources: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    latest_sync = _latest_webapp_sync_stage(webapp_sync)
    attachment_payload = latest_sync.get("attachment_payload")
    if isinstance(attachment_payload, Mapping):
        return dict(attachment_payload)
    payload: Dict[str, Any] = {}
    for field in WEBAPP_UPSTREAM_ID_FIELDS:
        value: Any = provider_status.get(field) or provider_run.get(field)
        for source in fallback_sources:
            upstream_handoff = source.get("upstream_handoff")
            value = (
                value
                or source.get(field)
                or (
                    upstream_handoff.get(field)
                    if isinstance(upstream_handoff, Mapping)
                    else None
                )
            )
        payload[field] = value
    if not _string(payload.get("request_id")) and _string(payload.get("site_submission_id")):
        payload["request_id"] = payload["site_submission_id"]
    return payload


def _placeholder_upstream_id(value: Any) -> bool:
    text = _string(value).lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "dummy",
            "example",
            "mock-",
            "placeholder",
            "replace_me",
            "test-",
        )
    )


def validate_provider_preview_packet(
    *,
    capture_root: str | Path,
    mode: str = "production",
    require_webapp_sync: bool = False,
) -> Dict[str, Any]:
    normalized_mode = _string(mode).lower() or "production"
    if normalized_mode not in {"production", "advisory"}:
        raise ValueError("mode must be production or advisory")

    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    generated_at = utc_now_iso()

    paths = {
        "privacy_final_walkthrough_mov": context.capture_root / "privacy" / "final_walkthrough.mov",
        "privacy_final_walkthrough_mp4": context.capture_root / "privacy" / "final_walkthrough.mp4",
        "privacy_processing_manifest": pipeline_dir / "privacy_processing_manifest.json",
        "privacy_verification_report": pipeline_dir / "privacy_verification_report.json",
        "worldlabs_input_manifest": (
            pipeline_dir / "worldlabs_input" / "worldlabs_input_manifest.json"
        ),
        "worldlabs_input_audit": pipeline_dir / "worldlabs_input_audit.json",
        "canonical_site_package": (
            pipeline_dir / "site_package" / "canonical_site_package.json"
        ),
        "provider_adapter_input": (
            pipeline_dir / "site_package" / "provider_adapter_inputs" / "world_labs_marble.json"
        ),
        "worldlabs_request_manifest": pipeline_dir / "worldlabs_request_manifest.json",
        "provider_preview_status": pipeline_dir / "provider_preview_status.json",
        "provider_run_manifest": pipeline_dir / "provider_run_manifest.json",
        "webapp_sync_result": pipeline_dir / "webapp_sync_result.json",
        "worldlabs_operation_manifest": pipeline_dir / "worldlabs_operation_manifest.json",
        "worldlabs_world_manifest": pipeline_dir / "worldlabs_world_manifest.json",
        "geometry_summary": pipeline_dir / "geometry" / "geometry_summary.json",
        "geometry_provider_result": pipeline_dir / "geometry" / "logs" / "provider_result.json",
    }

    privacy_manifest = _read_optional_mapping(paths["privacy_processing_manifest"])
    privacy_verification = _read_optional_mapping(paths["privacy_verification_report"])
    input_manifest = _read_optional_mapping(paths["worldlabs_input_manifest"])
    input_audit = _read_optional_mapping(paths["worldlabs_input_audit"])
    canonical_package = _read_optional_mapping(paths["canonical_site_package"])
    adapter_input = _read_optional_mapping(paths["provider_adapter_input"])
    request_manifest = _read_optional_mapping(paths["worldlabs_request_manifest"])
    provider_status = _read_optional_mapping(paths["provider_preview_status"])
    provider_run = _read_optional_mapping(paths["provider_run_manifest"])
    webapp_sync = _read_optional_mapping(paths["webapp_sync_result"])
    operation_manifest = _read_optional_mapping(paths["worldlabs_operation_manifest"])
    world_manifest = _read_optional_mapping(paths["worldlabs_world_manifest"])
    geometry_summary = _read_optional_mapping(paths["geometry_summary"])
    geometry_provider_result = _read_optional_mapping(paths["geometry_provider_result"])
    capture_descriptor = _read_optional_mapping(context.descriptor_path)
    opportunity_handoff = _read_optional_mapping(pipeline_dir / "opportunity_handoff.json")
    raw_manifest = _read_optional_mapping(context.raw_root / "manifest.json")

    blockers: List[str] = []
    warnings: List[str] = []

    final_walkthrough_exists = paths["privacy_final_walkthrough_mov"].is_file() or paths[
        "privacy_final_walkthrough_mp4"
    ].is_file()
    if not final_walkthrough_exists:
        blockers.append("missing_privacy_final_walkthrough")

    privacy_status = _string(privacy_manifest.get("status")).lower()
    verification_status = _string(privacy_verification.get("status")).lower()
    privacy_fail_closed = bool(privacy_manifest.get("fail_closed"))
    privacy_video_uri = _string(
        privacy_manifest.get("privacy_processed_video_uri")
        or privacy_manifest.get("world_model_video_uri")
    )
    verification_matches = not verification_status or verification_status in {
        privacy_status,
        "ready",
        "passed",
        "verified",
        "completed",
        "succeeded",
    }
    privacy_completed = (
        bool(privacy_manifest)
        and bool(privacy_verification)
        and privacy_fail_closed
        and _status_is_complete(privacy_status)
        and verification_matches
    )
    if not privacy_completed:
        blockers.append("privacy_manifest_or_verification_not_complete")
    if not _uri_endswith(
        privacy_video_uri,
        ("/privacy/final_walkthrough.mov", "/privacy/final_walkthrough.mp4"),
    ):
        blockers.append("privacy_output_not_final_walkthrough")

    audit_output_uri = _string(input_audit.get("output_video_uri"))
    audit_source_id = _string(input_audit.get("selected_video_source_id"))
    audit_source_is_final_walkthrough = bool(input_audit.get("source_is_final_walkthrough"))
    audit_derivative_of_final_walkthrough = bool(input_audit.get("derivative_of_final_walkthrough"))
    input_output_uri = _string(input_manifest.get("output_video_uri"))
    input_source_id = _string(input_manifest.get("selected_video_source_id"))
    request_selected_uri = _string(request_manifest.get("selected_video_uri"))
    request_audit_uri = _string(request_manifest.get("worldlabs_input_audit_uri"))
    request_source_manifest_uri = _string(
        request_manifest.get("source_manifest_uri")
        or _mapping(request_manifest.get("input_audit")).get("source_manifest_uri")
    )
    selected_checksum = _string(request_manifest.get("selected_input_checksum_sha256"))
    audit_output_checksum = _string(input_audit.get("output_checksum_sha256"))
    raw_bypass_used = bool(
        input_audit.get("raw_video_bypass_used")
        or _mapping(input_manifest.get("input_labeling")).get("raw_video_bypass_used")
        or _mapping(request_manifest.get("input_labeling")).get("raw_video_bypass_used")
    )
    privacy_safe_input = bool(
        input_audit.get("privacy_safe_input")
        and request_manifest.get("privacy_safe_input")
        and not raw_bypass_used
    )

    if bool(os.getenv("BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS")) and normalized_mode == "production":
        blockers.append("raw_worldlabs_bypass_env_enabled_in_production")
    if input_source_id and input_source_id not in PRIVACY_SAFE_SOURCE_IDS:
        blockers.append("worldlabs_input_selected_source_not_privacy_safe")
    if audit_source_id and audit_source_id not in PRIVACY_SAFE_SOURCE_IDS:
        blockers.append("worldlabs_audit_selected_source_not_privacy_safe")
    if raw_bypass_used:
        blockers.append("raw_video_bypass_used")
    if not privacy_safe_input:
        blockers.append("worldlabs_request_not_privacy_safe")
    if not (audit_source_is_final_walkthrough or audit_derivative_of_final_walkthrough):
        blockers.append("worldlabs_input_not_final_walkthrough_derivative")
    if not audit_output_uri or not request_selected_uri or audit_output_uri != request_selected_uri:
        blockers.append("worldlabs_audit_output_uri_mismatch")
    if input_output_uri and audit_output_uri and input_output_uri != audit_output_uri:
        blockers.append("worldlabs_input_manifest_output_uri_mismatch")
    if not request_audit_uri or not request_audit_uri.endswith("/pipeline/worldlabs_input_audit.json"):
        blockers.append("missing_worldlabs_input_audit_uri")
    if not request_source_manifest_uri:
        blockers.append("missing_worldlabs_source_manifest_uri")
    if audit_output_checksum and selected_checksum and audit_output_checksum != selected_checksum:
        blockers.append("worldlabs_selected_checksum_mismatch")
    if not audit_output_checksum or not selected_checksum:
        blockers.append("missing_worldlabs_input_checksum")

    canonical_rgb_uri = _canonical_rgb_uri(canonical_package)
    adapter_rgb_uri = _adapter_rgb_uri(adapter_input)
    if not canonical_rgb_uri:
        blockers.append("canonical_package_missing_privacy_safe_rgb_video")
    elif canonical_rgb_uri != request_selected_uri:
        blockers.append("canonical_package_rgb_video_mismatch")
    if not adapter_rgb_uri:
        blockers.append("provider_adapter_missing_rgb_video")
    elif adapter_rgb_uri != request_selected_uri:
        blockers.append("provider_adapter_rgb_video_mismatch")

    geometry_live_ready = bool(
        geometry_summary.get("geometry_live_ready")
        or geometry_summary.get("ready_for_world_model")
        or geometry_provider_result.get("geometry_live_ready")
        or geometry_provider_result.get("ready_for_world_model")
    )
    geometry_source = _string(geometry_summary.get("geometry_source") or geometry_provider_result.get("geometry_source"))
    if geometry_live_ready and geometry_source != "video_to_world":
        blockers.append("fallback_geometry_marked_live_ready")
    elif geometry_summary and not geometry_live_ready:
        warnings.append("geometry_present_but_not_live_ready")

    latest_webapp_sync = _latest_webapp_sync_stage(webapp_sync)
    webapp_attachment_payload = _webapp_attachment_payload(
        webapp_sync=webapp_sync,
        provider_status=provider_status,
        provider_run=provider_run,
        fallback_sources=(capture_descriptor, opportunity_handoff, raw_manifest),
    )
    webapp_upstream_ids = {
        field: _string(webapp_attachment_payload.get(field))
        for field in WEBAPP_UPSTREAM_ID_FIELDS
    }
    missing_webapp_upstream_ids = [
        field for field, value in webapp_upstream_ids.items() if not value
    ]
    placeholder_webapp_upstream_ids = [
        field
        for field, value in webapp_upstream_ids.items()
        if _placeholder_upstream_id(value)
    ]
    webapp_upstream_links_verified = bool(
        webapp_attachment_payload.get("upstream_links_verified")
    ) and not missing_webapp_upstream_ids and not placeholder_webapp_upstream_ids
    webapp_sync_status = _string(
        latest_webapp_sync.get("status") or webapp_sync.get("status")
    ).lower()
    webapp_sync_succeeded = webapp_sync_status == "succeeded"

    if require_webapp_sync:
        if not webapp_sync:
            blockers.append("missing_webapp_sync_result")
        if webapp_sync_status == "failed":
            blockers.append("webapp_sync_failed")
        if webapp_sync and not webapp_sync_succeeded:
            blockers.append("webapp_sync_not_succeeded")
        if missing_webapp_upstream_ids:
            blockers.append("webapp_sync_missing_real_upstream_ids")
            _append_unique(
                blockers,
                [f"missing_webapp_{field}" for field in missing_webapp_upstream_ids],
            )
        if placeholder_webapp_upstream_ids:
            blockers.append("webapp_sync_placeholder_upstream_ids")
            _append_unique(
                blockers,
                [f"placeholder_webapp_{field}" for field in placeholder_webapp_upstream_ids],
            )
        if webapp_sync and not webapp_upstream_links_verified:
            blockers.append("webapp_sync_upstream_links_not_verified")
        if webapp_sync_status == "skipped":
            blockers.append("webapp_sync_skipped_not_live")
    elif not webapp_sync:
        warnings.append("webapp_sync_not_required_or_not_present")

    provider_operation_proof = {
        "operation_manifest_present": bool(operation_manifest),
        "world_manifest_present": bool(world_manifest),
        "world_id": _string(world_manifest.get("world_id") or provider_status.get("world_id")),
        "status": "proven" if operation_manifest and world_manifest else "pending",
    }
    hosted_proof = {
        "status": "pending",
        "reason": "hosted_session_access_runtime_entitlement_or_export_evidence_not_checked_by_local_qa",
    }
    if provider_operation_proof["status"] == "pending":
        warnings.append("live_worldlabs_operation_or_world_manifest_pending")
    _append_unique(warnings, _string_list(provider_status.get("blockers")))

    unique_blockers: List[str] = []
    _append_unique(unique_blockers, blockers)
    status = "passed" if not unique_blockers else "blocked"
    claim_ceiling = "provider_proof_pending"
    if provider_operation_proof["status"] == "proven":
        claim_ceiling = "local_repo_proof"
    if unique_blockers:
        claim_ceiling = "local_repo_blocked"

    manifest = {
        "schema_version": PROVIDER_PREVIEW_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "mode": normalized_mode,
        "status": status,
        "claim_ceiling": claim_ceiling,
        "privacy_artifacts": {
            "final_walkthrough_exists": final_walkthrough_exists,
            "processing_manifest": _artifact_status(paths["privacy_processing_manifest"]),
            "verification_report": _artifact_status(paths["privacy_verification_report"]),
            "privacy_status": privacy_status or None,
            "verification_status": verification_status or None,
            "fail_closed": privacy_fail_closed,
        },
        "redaction_proof": {
            "privacy_completed": privacy_completed,
            "privacy_output_uri": privacy_video_uri or None,
            "privacy_output_is_final_walkthrough": "privacy_output_not_final_walkthrough"
            not in unique_blockers,
            "depth_source": privacy_manifest.get("depth_source")
            or _mapping(privacy_manifest.get("depth_conditioning")).get("source"),
        },
        "worldlabs_input_lineage": {
            "input_manifest": _artifact_status(paths["worldlabs_input_manifest"]),
            "input_audit": _artifact_status(paths["worldlabs_input_audit"]),
            "selected_video_source_id": input_source_id or audit_source_id or None,
            "selected_video_uri": request_selected_uri or None,
            "audit_output_video_uri": audit_output_uri or None,
            "source_is_final_walkthrough": audit_source_is_final_walkthrough,
            "derivative_of_final_walkthrough": audit_derivative_of_final_walkthrough,
            "audit_matches_request": bool(audit_output_uri and audit_output_uri == request_selected_uri),
            "selected_input_checksum_sha256": selected_checksum or None,
            "audit_output_checksum_sha256": audit_output_checksum or None,
        },
        "raw_path_policy": {
            "raw_video_bypass_env_enabled": bool(os.getenv("BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS")),
            "raw_video_bypass_used": raw_bypass_used,
            "privacy_safe_input": privacy_safe_input,
            "production_mode_blocks_raw_bypass": normalized_mode == "production",
        },
        "canonical_package_match": {
            "path": str(paths["canonical_site_package"]),
            "exists": bool(canonical_package),
            "rgb_video_uri": canonical_rgb_uri or None,
            "matches_request": bool(canonical_rgb_uri and canonical_rgb_uri == request_selected_uri),
        },
        "provider_adapter_match": {
            "path": str(paths["provider_adapter_input"]),
            "exists": bool(adapter_input),
            "rgb_video_uri": adapter_rgb_uri or None,
            "matches_request": bool(adapter_rgb_uri and adapter_rgb_uri == request_selected_uri),
            "adapter_status": adapter_input.get("status"),
        },
        "request_manifest_validation": {
            "path": str(paths["worldlabs_request_manifest"]),
            "exists": bool(request_manifest),
            "status": request_manifest.get("status"),
            "worldlabs_input_audit_uri": request_audit_uri or None,
            "source_manifest_uri": request_source_manifest_uri or None,
            "privacy_safe_input": bool(request_manifest.get("privacy_safe_input")),
        },
        "geometry_labels": {
            "geometry_summary_present": bool(geometry_summary),
            "geometry_source": geometry_source or None,
            "geometry_live_ready": geometry_live_ready,
            "live_ready_claim_allowed": geometry_live_ready and geometry_source == "video_to_world",
        },
        "webapp_sync_projection": {
            "required": require_webapp_sync,
            "present": bool(webapp_sync),
            "status": webapp_sync_status or None,
            "sync_succeeded": webapp_sync_succeeded,
            "latest_stage": webapp_sync.get("latest_stage"),
            "upstream_links_verified": webapp_upstream_links_verified,
            "missing_upstream_ids": missing_webapp_upstream_ids,
            "placeholder_upstream_ids": placeholder_webapp_upstream_ids,
            "upstream_ids": webapp_upstream_ids,
            "reason": latest_webapp_sync.get("reason") or webapp_sync.get("reason"),
            "blocker": latest_webapp_sync.get("blocker") or webapp_sync.get("blocker"),
        },
        "provider_operation_proof": provider_operation_proof,
        "hosted_proof": hosted_proof,
        "blocked_claims": [
            "raw_walkthrough_as_production_worldlabs_input"
            if raw_bypass_used
            else "",
            "hosted_session_proven",
            "simulator_execution_completed",
            "robot_ready",
            "deployment_ready",
        ],
        "blockers": unique_blockers,
        "warnings": warnings,
        "next_required_live_gates": [
            "explicit_operator_approval_for_worldlabs_generation"
            if provider_operation_proof["status"] == "pending"
            else "",
            "worldlabs_operation_and_world_manifest"
            if provider_operation_proof["status"] == "pending"
            else "",
            "hosted_session_access_runtime_entitlement_or_export_evidence",
            "owner_gpu_simulator_execution_proof",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["blocked_claims"] = [item for item in manifest["blocked_claims"] if item]
    manifest["next_required_live_gates"] = [
        item for item in manifest["next_required_live_gates"] if item
    ]
    write_json(pipeline_dir / "provider_preview_qa_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a provider preview packet locally")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--mode", choices=("production", "advisory"), default="production")
    parser.add_argument("--require-webapp-sync", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_provider_preview_packet(
            capture_root=args.capture_root,
            mode=args.mode,
            require_webapp_sync=args.require_webapp_sync,
        )
    except (PipelineError, ValueError) as exc:
        print(f"[provider-preview-qa] status=error reason={exc}")
        return 2
    print(
        "[provider-preview-qa] manifest="
        f"{Path(args.capture_root) / 'pipeline' / 'provider_preview_qa_manifest.json'}"
    )
    print(f"[provider-preview-qa] status={result['status']}")
    if result["status"] == "blocked" and args.mode == "production":
        print(f"[provider-preview-qa] blockers={','.join(result.get('blockers') or [])}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
