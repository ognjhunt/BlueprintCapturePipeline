"""Repo-local same-capture lineage packet builder.

The packet is a local truth surface for one capture root. It does not call live
providers, WebApp, payment systems, or hardware. Runtime and hardware proof stay
separate from repo-local lineage proof.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "same_capture_lineage_packet.v1"
WEBAPP_ID_FIELDS = ("site_submission_id", "request_id", "buyer_request_id", "capture_job_id")
PLACEHOLDER_ID_MARKERS = ("example", "placeholder", "replace_me", "sample", "todo", "tbd", "<", ">", "your-")
ANDROID_XR_PROFILE_ID = "android_xr_glasses"
ANDROID_XR_VIDEO_ONLY_MODALITY = "android_xr_video_only"


def build_same_capture_lineage_packet(
    *,
    capture_root: Path,
    paperclip_issue_id: str | None = None,
    paperclip_issue_url: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed packet for one capture root without external calls."""

    root = Path(capture_root).expanduser().resolve()
    raw_root = root / "raw"
    pipeline_root = root / "pipeline"

    manifest = _read_json(raw_root / "manifest.json")
    capture_context = _read_json(raw_root / "capture_context.json")
    upload_completion = _read_json(raw_root / "capture_upload_complete.json")
    descriptor = _read_json(root / "capture_descriptor.json")
    qa_report = _read_json(root / "qa_report.json")
    pipeline_handoff = _read_json(root / "pipeline_handoff.json")
    opportunity_handoff = _read_json(pipeline_root / "opportunity_handoff.json")
    qualification_summary = _read_json(pipeline_root / "qualification_summary.json")
    geometry_summary = _read_json(pipeline_root / "geometry" / "geometry_summary.json")
    webapp_sync = _read_json(pipeline_root / "webapp_sync_result.json")

    path_scene_id, path_capture_id = _identity_from_capture_root(root)
    identity_sources = [
        _identity_source("path", {"scene_id": path_scene_id, "capture_id": path_capture_id}),
        _identity_source("raw.manifest", manifest),
        _identity_source("raw.capture_context", capture_context),
        _identity_source("raw.capture_upload_complete", upload_completion),
        _identity_source("bridge.capture_descriptor", descriptor),
        _identity_source("bridge.qa_report", qa_report),
        _identity_source("bridge.pipeline_handoff", pipeline_handoff),
        _identity_source("pipeline.opportunity_handoff", opportunity_handoff),
        _identity_source("pipeline.qualification_summary", qualification_summary),
        _identity_source("pipeline.geometry_summary", geometry_summary),
    ]
    primary_scene_id = _first_non_empty(source["scene_id"] for source in identity_sources)
    primary_capture_id = _first_non_empty(source["capture_id"] for source in identity_sources)
    identity_blockers = _identity_blockers(
        identity_sources=identity_sources,
        primary_scene_id=primary_scene_id,
        primary_capture_id=primary_capture_id,
    )

    raw_bundle = _raw_bundle_packet(
        raw_root=raw_root,
        manifest=manifest,
        capture_context=capture_context,
        upload_completion=upload_completion,
        primary_scene_id=primary_scene_id,
        primary_capture_id=primary_capture_id,
    )
    bridge_handoff = _bridge_handoff_packet(
        descriptor=descriptor,
        qa_report=qa_report,
        pipeline_handoff=pipeline_handoff,
        primary_scene_id=primary_scene_id,
        primary_capture_id=primary_capture_id,
    )
    pipeline_result = _pipeline_result_packet(
        pipeline_root=pipeline_root,
        opportunity_handoff=opportunity_handoff,
        qualification_summary=qualification_summary,
        geometry_summary=geometry_summary,
        primary_scene_id=primary_scene_id,
        primary_capture_id=primary_capture_id,
    )
    webapp_upstream_ids = _webapp_upstream_ids_packet(
        webapp_sync=webapp_sync,
        opportunity_handoff=opportunity_handoff,
        descriptor=descriptor,
        manifest=manifest,
        primary_scene_id=primary_scene_id,
        primary_capture_id=primary_capture_id,
    )
    profile = _capture_profile(manifest=manifest, capture_context=capture_context, descriptor=descriptor)
    geometry_claim = _geometry_claim_allowed(pipeline_result["geometry"])
    android_xr_video_only = (
        profile["capture_profile_id"] == ANDROID_XR_PROFILE_ID
        and profile["capture_modality"] == ANDROID_XR_VIDEO_ONLY_MODALITY
    )

    paperclip_issue = {
        "issue_id": _string(paperclip_issue_id),
        "url": _string(paperclip_issue_url),
        "blockers": [] if _string(paperclip_issue_id) else ["missing_paperclip_issue"],
    }

    repo_blockers = _dedupe(
        [
            *identity_blockers,
            *raw_bundle["blockers"],
            *bridge_handoff["blockers"],
            *pipeline_result["blockers"],
            *webapp_upstream_ids["blockers"],
            *paperclip_issue["blockers"],
        ]
    )
    if android_xr_video_only:
        repo_blockers.append("android_xr_video_only_requires_explicit_geometry_contract")
    repo_blockers = _dedupe(repo_blockers)

    remaining_hardware_gaps = _remaining_hardware_gaps(profile=profile, manifest=manifest, descriptor=descriptor)
    remaining_runtime_gaps = [
        "live_provider_runtime_payment_proof_not_in_repo_packet",
    ]

    repo_same_capture_chain = not repo_blockers
    hosted_review_claim_allowed = bool(repo_same_capture_chain and webapp_upstream_ids["upstream_links_verified"])
    world_model_ready_claim_allowed = bool(repo_same_capture_chain and geometry_claim)
    public_readiness_claim_allowed = False
    android_xr_public_readiness_claim_allowed = False if android_xr_video_only else public_readiness_claim_allowed
    claims = {
        "repo_same_capture_chain_proven": repo_same_capture_chain,
        "hosted_review_claim_allowed": hosted_review_claim_allowed,
        "world_model_ready_claim_allowed": world_model_ready_claim_allowed,
        "launch_claim_allowed": False,
        "public_readiness_claim_allowed": public_readiness_claim_allowed,
        "android_xr_public_readiness_claim_allowed": android_xr_public_readiness_claim_allowed,
        "claim_ceiling": (
            "repo_same_capture_lineage_only"
            if repo_same_capture_chain
            else "blocked_until_repo_lineage_gaps_close"
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "repo_proven" if repo_same_capture_chain else "blocked",
        "capture_root": str(root),
        "scene_id": primary_scene_id,
        "capture_id": primary_capture_id,
        "capture_profile": profile,
        "identity_sources": identity_sources,
        "raw_bundle": raw_bundle,
        "bridge_handoff": bridge_handoff,
        "pipeline_result": pipeline_result,
        "webapp_upstream_ids": webapp_upstream_ids,
        "paperclip_issue": paperclip_issue,
        "claims": claims,
        "repo_blockers": repo_blockers,
        "remaining_hardware_gaps": remaining_hardware_gaps,
        "remaining_runtime_gaps": remaining_runtime_gaps,
    }


def validate_same_capture_lineage_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate packet shape and return a local status summary."""

    blockers: list[str] = []
    if packet.get("schema_version") != SCHEMA_VERSION:
        blockers.append("same_capture_lineage_schema_version_invalid")
    if str(packet.get("status") or "") not in {"repo_proven", "blocked"}:
        blockers.append("same_capture_lineage_status_invalid")
    if not _string(packet.get("capture_id")):
        blockers.append("same_capture_lineage_capture_id_missing")
    if not isinstance(packet.get("repo_blockers"), list):
        blockers.append("same_capture_lineage_repo_blockers_missing")
    claims = packet.get("claims") if isinstance(packet.get("claims"), Mapping) else {}
    if claims.get("launch_claim_allowed") is not False:
        blockers.append("same_capture_lineage_launch_claim_must_remain_false")
    if claims.get("world_model_ready_claim_allowed") is True:
        geometry = (
            (packet.get("pipeline_result") or {}).get("geometry")
            if isinstance(packet.get("pipeline_result"), Mapping)
            else {}
        )
        if not _geometry_claim_allowed(geometry if isinstance(geometry, Mapping) else {}):
            blockers.append("same_capture_lineage_world_model_claim_not_backed_by_live_geometry")
    return {
        "status": "valid" if not blockers else "blocked",
        "blockers": blockers,
    }


def write_same_capture_lineage_packet(
    *,
    capture_root: Path,
    paperclip_issue_id: str | None = None,
    paperclip_issue_url: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build and write the packet under `pipeline/` unless an output path is supplied."""

    root = Path(capture_root).expanduser().resolve()
    path = Path(output_path).expanduser().resolve() if output_path else root / "pipeline" / "same_capture_lineage_packet.json"
    packet = build_same_capture_lineage_packet(
        capture_root=root,
        paperclip_issue_id=paperclip_issue_id,
        paperclip_issue_url=paperclip_issue_url,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _identity_from_capture_root(capture_root: Path) -> tuple[str, str]:
    parts = capture_root.parts
    scene_id = ""
    capture_id = ""
    if "scenes" in parts:
        index = parts.index("scenes")
        if index + 1 < len(parts):
            scene_id = parts[index + 1]
    if "captures" in parts:
        index = parts.index("captures")
        if index + 1 < len(parts):
            capture_id = parts[index + 1]
    return scene_id, capture_id


def _identity_source(name: str, payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "name": name,
        "scene_id": _string(payload.get("scene_id")),
        "capture_id": _string(payload.get("capture_id")),
    }


def _identity_blockers(
    *,
    identity_sources: list[dict[str, str]],
    primary_scene_id: str,
    primary_capture_id: str,
) -> list[str]:
    blockers: list[str] = []
    if not primary_scene_id:
        blockers.append("same_capture_scene_id_missing")
    if not primary_capture_id:
        blockers.append("same_capture_capture_id_missing")
    for source in identity_sources:
        scene_id = source["scene_id"]
        capture_id = source["capture_id"]
        if scene_id and primary_scene_id and scene_id != primary_scene_id:
            blockers.append(f"same_capture_scene_id_mismatch:{source['name']}")
        if capture_id and primary_capture_id and capture_id != primary_capture_id:
            blockers.append(f"same_capture_capture_id_mismatch:{source['name']}")
    return blockers


def _raw_bundle_packet(
    *,
    raw_root: Path,
    manifest: Mapping[str, Any],
    capture_context: Mapping[str, Any],
    upload_completion: Mapping[str, Any],
    primary_scene_id: str,
    primary_capture_id: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    manifest_exists = bool(manifest)
    capture_context_exists = bool(capture_context)
    upload_completion_exists = bool(upload_completion)
    if not manifest_exists:
        blockers.append("raw_manifest_missing")
    if not capture_context_exists:
        blockers.append("raw_capture_context_missing")
    if not upload_completion_exists:
        blockers.append("raw_capture_upload_completion_missing")
    if upload_completion_exists:
        if _string(upload_completion.get("capture_id")) != primary_capture_id:
            blockers.append("raw_capture_upload_completion_capture_id_mismatch")
        if _string(upload_completion.get("scene_id")) != primary_scene_id:
            blockers.append("raw_capture_upload_completion_scene_id_mismatch")
    return {
        "path": str(raw_root),
        "manifest_exists": manifest_exists,
        "capture_context_exists": capture_context_exists,
        "upload_completion": {
            "exists": upload_completion_exists,
            "scene_id": _string(upload_completion.get("scene_id")),
            "capture_id": _string(upload_completion.get("capture_id")),
            "status": _string(upload_completion.get("status")),
            "completed_at": _string(upload_completion.get("completed_at") or upload_completion.get("uploaded_at")),
        },
        "identity_consistent": not blockers,
        "blockers": blockers,
    }


def _bridge_handoff_packet(
    *,
    descriptor: Mapping[str, Any],
    qa_report: Mapping[str, Any],
    pipeline_handoff: Mapping[str, Any],
    primary_scene_id: str,
    primary_capture_id: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    descriptor_exists = bool(descriptor)
    qa_report_exists = bool(qa_report)
    pipeline_handoff_exists = bool(pipeline_handoff)
    if not descriptor_exists:
        blockers.append("bridge_capture_descriptor_missing")
    if not qa_report_exists:
        blockers.append("bridge_qa_report_missing")
    if not pipeline_handoff_exists:
        blockers.append("bridge_pipeline_handoff_missing")
    for name, payload in (
        ("capture_descriptor", descriptor),
        ("qa_report", qa_report),
        ("pipeline_handoff", pipeline_handoff),
    ):
        if payload and _string(payload.get("capture_id")) != primary_capture_id:
            blockers.append(f"bridge_{name}_capture_id_mismatch")
        if payload and _string(payload.get("scene_id")) != primary_scene_id:
            blockers.append(f"bridge_{name}_scene_id_mismatch")
    return {
        "capture_descriptor_exists": descriptor_exists,
        "qa_report_exists": qa_report_exists,
        "pipeline_handoff_exists": pipeline_handoff_exists,
        "same_capture": not blockers,
        "blockers": blockers,
    }


def _pipeline_result_packet(
    *,
    pipeline_root: Path,
    opportunity_handoff: Mapping[str, Any],
    qualification_summary: Mapping[str, Any],
    geometry_summary: Mapping[str, Any],
    primary_scene_id: str,
    primary_capture_id: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    opportunity_handoff_exists = bool(opportunity_handoff)
    qualification_summary_exists = bool(qualification_summary)
    completion_marker_exists = (pipeline_root / ".qualification_pipeline_complete").is_file()
    if not opportunity_handoff_exists:
        blockers.append("pipeline_opportunity_handoff_missing")
    if not qualification_summary_exists:
        blockers.append("pipeline_qualification_summary_missing")
    if not completion_marker_exists:
        blockers.append("pipeline_completion_marker_missing")
    for name, payload in (
        ("opportunity_handoff", opportunity_handoff),
        ("qualification_summary", qualification_summary),
        ("geometry_summary", geometry_summary),
    ):
        if payload and _string(payload.get("capture_id")) != primary_capture_id:
            blockers.append(f"pipeline_{name}_capture_id_mismatch")
        if payload and _string(payload.get("scene_id")) != primary_scene_id:
            blockers.append(f"pipeline_{name}_scene_id_mismatch")
    geometry = _geometry_packet(geometry_summary)
    if geometry["fallback_used"]:
        blockers.append("fallback_geometry_not_world_model_ready")
    return {
        "status": "succeeded" if not blockers else "blocked",
        "opportunity_handoff_exists": opportunity_handoff_exists,
        "qualification_summary_exists": qualification_summary_exists,
        "completion_marker_exists": completion_marker_exists,
        "same_capture": not [
            blocker
            for blocker in blockers
            if blocker.endswith("_capture_id_mismatch") or blocker.endswith("_scene_id_mismatch")
        ],
        "geometry": geometry,
        "blockers": blockers,
    }


def _geometry_packet(geometry_summary: Mapping[str, Any]) -> dict[str, Any]:
    geometry_source = _string(geometry_summary.get("geometry_source") or "missing")
    fallback_used = bool(geometry_summary.get("fallback_used")) or geometry_source == "fallback_geometry"
    launch_blockers = _string_list(geometry_summary.get("launch_blockers"))
    return {
        "exists": bool(geometry_summary),
        "geometry_source": geometry_source,
        "fallback_used": fallback_used,
        "provider_native_result": bool(geometry_summary.get("provider_native_result")),
        "site_frame_available": bool(geometry_summary.get("site_frame_available")),
        "scale_resolved": bool(geometry_summary.get("scale_resolved")),
        "ready_for_world_model": bool(geometry_summary.get("ready_for_world_model")),
        "geometry_live_ready": bool(geometry_summary.get("geometry_live_ready")),
        "launch_blockers": launch_blockers,
    }


def _webapp_upstream_ids_packet(
    *,
    webapp_sync: Mapping[str, Any],
    opportunity_handoff: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    manifest: Mapping[str, Any],
    primary_scene_id: str,
    primary_capture_id: str,
) -> dict[str, Any]:
    sync_payload, source = _latest_webapp_sync_payload(webapp_sync)
    upstream_handoff = manifest.get("upstream_handoff") if isinstance(manifest.get("upstream_handoff"), Mapping) else {}
    candidates: list[tuple[str, Mapping[str, Any]]] = [
        (source, sync_payload),
        ("pipeline.opportunity_handoff", opportunity_handoff),
        ("bridge.capture_descriptor", descriptor),
        ("raw.manifest.upstream_handoff", upstream_handoff),
    ]
    ids: dict[str, str] = {}
    id_sources: dict[str, str] = {}
    for field in WEBAPP_ID_FIELDS:
        for candidate_source, payload in candidates:
            value = _string(payload.get(field))
            if value:
                ids[field] = value
                id_sources[field] = candidate_source
                break
        ids.setdefault(field, "")
        id_sources.setdefault(field, "")

    blockers: list[str] = []
    for field in WEBAPP_ID_FIELDS:
        value = ids[field]
        if not value:
            blockers.append(f"missing_webapp_{field}")
        elif _contains_placeholder_id(value):
            blockers.append(f"placeholder_webapp_{field}")
        elif value in {primary_capture_id, f"{primary_scene_id}:{primary_capture_id}", f"{primary_scene_id}/{primary_capture_id}"}:
            blockers.append(f"generated_capture_id_used_for_webapp_{field}")
    sync_capture_id = _string(sync_payload.get("capture_id"))
    sync_scene_id = _string(sync_payload.get("scene_id"))
    if sync_capture_id and sync_capture_id != primary_capture_id:
        blockers.append("webapp_sync_capture_id_mismatch")
    if sync_scene_id and sync_scene_id != primary_scene_id:
        blockers.append("webapp_sync_scene_id_mismatch")
    upstream_links_verified = not blockers
    if sync_payload and sync_payload.get("upstream_links_verified") is False:
        upstream_links_verified = False
        missing = _string_list(sync_payload.get("missing_upstream_links"))
        blockers.extend(f"webapp_sync_missing_{field}" for field in missing)
    return {
        "source": source,
        "ids": ids,
        "id_sources": id_sources,
        "sync_status": _string(webapp_sync.get("status")),
        "latest_stage": _string(webapp_sync.get("latest_stage")),
        "upstream_links_verified": upstream_links_verified,
        "blockers": _dedupe(blockers),
    }


def _latest_webapp_sync_payload(webapp_sync: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    syncs = webapp_sync.get("syncs") if isinstance(webapp_sync.get("syncs"), Mapping) else {}
    latest_stage = _string(webapp_sync.get("latest_stage"))
    latest = syncs.get(latest_stage) if latest_stage and isinstance(syncs.get(latest_stage), Mapping) else None
    if latest is None and syncs:
        latest = next((value for value in reversed(list(syncs.values())) if isinstance(value, Mapping)), None)
    if latest is None:
        latest = webapp_sync
        source = "pipeline.webapp_sync_result"
    else:
        source = f"pipeline.webapp_sync_result:{latest_stage or 'latest'}"
    payload = latest.get("attachment_payload") if isinstance(latest.get("attachment_payload"), Mapping) else latest
    return dict(payload) if isinstance(payload, Mapping) else {}, source


def _capture_profile(
    *,
    manifest: Mapping[str, Any],
    capture_context: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "capture_profile_id": _first_non_empty(
            (
                _string(manifest.get("capture_profile_id")),
                _string(capture_context.get("capture_profile_id")),
                _string(descriptor.get("capture_profile_id")),
            )
        ),
        "capture_modality": _first_non_empty(
            (
                _string(manifest.get("capture_modality")),
                _string(capture_context.get("capture_modality")),
                _string(descriptor.get("capture_modality")),
            )
        ),
        "capture_source": _first_non_empty(
            (
                _string(manifest.get("capture_source")),
                _string(capture_context.get("capture_source")),
                _string(descriptor.get("capture_source")),
            )
        ),
    }


def _geometry_claim_allowed(geometry: Mapping[str, Any]) -> bool:
    return bool(
        geometry.get("exists")
        and geometry.get("geometry_source") == "video_to_world"
        and not bool(geometry.get("fallback_used"))
        and geometry.get("provider_native_result") is True
        and geometry.get("site_frame_available") is True
        and geometry.get("scale_resolved") is True
        and geometry.get("ready_for_world_model") is True
        and geometry.get("geometry_live_ready") is True
    )


def _remaining_hardware_gaps(
    *,
    profile: Mapping[str, str],
    manifest: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> list[str]:
    proof = manifest.get("physical_device_proof")
    if not isinstance(proof, Mapping):
        metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
        proof = metadata.get("physical_device_proof") if isinstance(metadata.get("physical_device_proof"), Mapping) else {}
    if proof.get("physical_device_smoke_passed") is True and _string(proof.get("evidence_uri")):
        return []
    profile_id = _string(profile.get("capture_profile_id"))
    modality = _string(profile.get("capture_modality"))
    source = _string(profile.get("capture_source"))
    if profile_id == ANDROID_XR_PROFILE_ID or modality == ANDROID_XR_VIDEO_ONLY_MODALITY:
        return ["android_xr_video_only_requires_physical_hardware_proof"]
    if "glasses" in profile_id or "glasses" in source or "meta" in source:
        return ["glasses_physical_device_proof_not_in_repo_packet"]
    return ["physical_device_capture_proof_not_in_repo_packet"]


def _contains_placeholder_id(value: str) -> bool:
    normalized = value.strip().lower()
    return any(marker in normalized for marker in PLACEHOLDER_ID_MARKERS)


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_string(item) for item in value if _string(item)]
    if isinstance(value, tuple):
        return [_string(item) for item in value if _string(item)]
    return []


def _first_non_empty(values: object) -> str:
    for value in values:  # type: ignore[union-attr]
        text = _string(value)
        if text:
            return text
    return ""


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
