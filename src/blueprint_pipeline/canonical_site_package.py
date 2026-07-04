"""Canonical provider-agnostic site package assembly."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import ensure_dir, read_json, relative_scene_path, utc_now_iso, write_json
from .launch_proof_policy import production_launch_mode


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _gs_uri_for_path(*, bucket: str, storage_root: Path, path: Path) -> str:
    return f"gs://{bucket}/{relative_scene_path(path, storage_root)}"


def _gs_uri_if_file(*, bucket: str, storage_root: Path, path: Path) -> str | None:
    if not path.is_file():
        return None
    return _gs_uri_for_path(bucket=bucket, storage_root=storage_root, path=path)


def _json_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        text = _string(value)
        if text:
            return text
    return None


def _capture_grounded_ref(uri: str | None, *, source: str, required: bool = False) -> Dict[str, Any]:
    return {
        "uri": uri,
        "source": source,
        "truth_label": "capture_grounded_input",
        "required": bool(required),
        "available": bool(uri),
    }


def _derived_ref(uri: str | None, *, source: str, required: bool = False) -> Dict[str, Any]:
    return {
        "uri": uri,
        "source": source,
        "truth_label": "derived_pipeline_output",
        "required": bool(required),
        "available": bool(uri),
    }


def _semantic_keywords(text: str, words: Sequence[str]) -> list[str]:
    lowered = text.lower()
    return [word for word in words if word in lowered]


def _target_objects(task_targets_payload: Mapping[str, Any], object_index_entries: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    raw_targets = task_targets_payload.get("targets") if isinstance(task_targets_payload.get("targets"), list) else []
    for item in raw_targets:
        if not isinstance(item, Mapping):
            continue
        label = _first_nonempty(item.get("label"), item.get("object_label"), item.get("object_id"))
        if label:
            targets.append({"label": label, "source": "task_targets", "raw": dict(item)})
    if targets:
        return targets
    for item in object_index_entries:
        label = _first_nonempty(item.get("label"), item.get("object_id"), item.get("id"))
        if label:
            targets.append({"label": label, "source": "object_index"})
    return targets


def _geometry_readiness(geometry_summary: Mapping[str, Any]) -> Dict[str, Any]:
    geometry_source = _string(geometry_summary.get("geometry_source") or "missing")
    fallback_used = bool(geometry_summary.get("fallback_used"))
    geometry_live_ready = bool(geometry_summary.get("geometry_live_ready"))
    blockers = _string_list(geometry_summary.get("launch_blockers"))
    if geometry_summary and fallback_used:
        blockers.append("fallback_geometry_not_live_video_to_world")
    if geometry_summary and geometry_source != "video_to_world":
        blockers.append(f"geometry_source_not_video_to_world:{geometry_source or 'missing'}")
    if geometry_summary and not geometry_live_ready:
        blockers.append("geometry_not_live_video_to_world")
    blockers = list(dict.fromkeys(blockers))
    site_faithful_ready = bool(
        geometry_summary
        and geometry_source == "video_to_world"
        and not fallback_used
        and geometry_live_ready
    )
    return {
        "status": "ready" if site_faithful_ready else "blocked" if geometry_summary else "missing",
        "site_faithful_provider_ready": site_faithful_ready,
        "geometry_source": geometry_source,
        "fallback_used": fallback_used,
        "fallback_kind": geometry_summary.get("fallback_kind"),
        "geometry_live_ready": geometry_live_ready,
        "ready_for_world_model": bool(geometry_summary.get("ready_for_world_model")),
        "blockers": blockers,
    }


def _world_labs_readiness(
    *,
    worldlabs_input: Mapping[str, Any],
    privacy_processing: Mapping[str, Any],
    rights_review: Mapping[str, Any],
    provenance_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    output_video_uri = _string(worldlabs_input.get("output_video_uri"))
    status = _string(worldlabs_input.get("status")).lower()
    audit = _as_dict(worldlabs_input.get("audit_payload"))
    labeling = _as_dict(worldlabs_input.get("input_labeling"))
    raw_bypass_used = bool(audit.get("raw_video_bypass_used") or labeling.get("raw_video_bypass_used"))
    # Raw-bypass exemptions exist only for labeled non-production previews. In
    # production launch mode the bypass itself is a blocker and never exempts
    # privacy/rights failures, so unredacted media can't reach a ready package.
    bypass_exempts = raw_bypass_used and not production_launch_mode()
    if not output_video_uri:
        blockers.append("missing_privacy_safe_world_model_input")
    if status and status != "ready":
        blockers.append(f"worldlabs_input_status:{status}")
    if not bool(audit.get("privacy_safe_input") or labeling.get("privacy_safe_input")) and not bypass_exempts:
        blockers.append("privacy_safe_world_model_input_not_verified")
    if raw_bypass_used and production_launch_mode():
        blockers.append("raw_video_bypass_used_in_production")
    elif raw_bypass_used:
        warnings.append("raw_video_bypass_input_non_production")
    privacy_status = _string(privacy_processing.get("status")).lower()
    if privacy_status == "failed_closed" and not bypass_exempts:
        blockers.append("privacy_processing_failed_closed")
    elif privacy_status == "failed_closed":
        warnings.append("privacy_processing_failed_closed_raw_bypass")
    rights_status = _string(rights_review.get("status")).lower()
    if rights_status == "blocked" and not bypass_exempts:
        blockers.append("rights_provenance_review_blocked")
    elif rights_status == "blocked":
        warnings.append("rights_provenance_review_blocked_raw_bypass")
    elif rights_status and rights_status != "cleared":
        warnings.append(f"rights_provenance_review:{rights_status}")
    provenance_status = _string(provenance_summary.get("status")).lower()
    if provenance_status and provenance_status != "grounded":
        warnings.append(f"provenance_summary:{provenance_status}")
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    return {
        "status": "blocked" if blockers else "review_required" if warnings else "ready",
        "required_inputs": [
            "privacy_safe_world_model_input",
            "rights_or_consent_clearance",
            "privacy_processing_clearance",
            "provenance_summary",
        ],
        "blockers": blockers,
        "warnings": warnings,
    }


def _future_adapter_mapping(adapter_id: str, *, canonical_site_package_uri: str) -> Dict[str, Any]:
    return {
        "status": "not_configured",
        "canonical_site_package_uri": canonical_site_package_uri,
        "provider_adapter_input_uri": None,
        "required_adapter": adapter_id,
        "swappable_contract": "ProviderAdapterInput",
        "blockers": [f"{adapter_id}_adapter_not_configured"],
    }


def build_world_labs_marble_adapter_input(
    *,
    package: Mapping[str, Any],
    canonical_site_package_uri: str,
    provider_adapter_input_uri: str,
    source_package_checksum_sha256: str,
) -> Dict[str, Any]:
    readiness = _as_dict(_as_dict(package.get("provider_readiness")).get("world_labs_marble"))
    conditioning = _as_dict(package.get("conditioning"))
    rgb_video = _as_dict(conditioning.get("rgb_video"))
    privacy_input = _as_dict(rgb_video.get("privacy_safe_world_model_input"))
    semantic_task_context = _as_dict(package.get("semantic_task_context"))
    identity = _as_dict(package.get("identity"))
    site_identity = _as_dict(package.get("site_identity_topology")).get("site_identity") or {}
    display_name = (
        _string(site_identity.get("site_name") if isinstance(site_identity, Mapping) else "")
        or _string(identity.get("site_id"))
        or _string(identity.get("capture_id"))
    )
    task_prompt = _string(semantic_task_context.get("robot_team_task_prompt")) or _string(
        semantic_task_context.get("task_statement")
    )
    tags = [
        value
        for value in [
            _string(identity.get("scene_id")),
            _string(identity.get("capture_id")),
            _string(identity.get("site_submission_id")),
            "provider-adapter-input",
            "canonical-site-package",
        ]
        if value
    ]
    return {
        "schema_version": "v1",
        "adapter_input_type": "ProviderAdapterInput",
        "provider": "world_labs",
        "adapter": "marble",
        "status": readiness.get("status") or "blocked",
        "blockers": _string_list(readiness.get("blockers")),
        "warnings": _string_list(readiness.get("warnings")),
        "canonical_site_package_uri": canonical_site_package_uri,
        "provider_adapter_input_uri": provider_adapter_input_uri,
        "source": {
            "canonical_site_package_uri": canonical_site_package_uri,
            "canonical_site_package_checksum_sha256": source_package_checksum_sha256,
            "provider_adapter_input_uri": provider_adapter_input_uri,
        },
        "conditioning_inputs": {
            "rgb_video": {
                "uri": privacy_input.get("uri"),
                "source_id": "privacy_safe_world_model_input",
                "privacy_safe": bool(privacy_input.get("privacy_safe")),
                "checksum_sha256": privacy_input.get("checksum_sha256"),
                "source_checksum_sha256": privacy_input.get("source_checksum_sha256"),
                "source_manifest_uri": privacy_input.get("source_manifest_uri"),
                "input_audit_uri": privacy_input.get("input_audit_uri"),
                "labeling": dict(privacy_input.get("labeling") or {}),
            },
            "frames": conditioning.get("frames") or {},
            "camera": conditioning.get("camera") or {},
            "depth_confidence": conditioning.get("depth_confidence") or {},
            "geometry": conditioning.get("geometry") or {},
            "semantic_task_context": semantic_task_context,
        },
        "generation": {
            "display_name": display_name or f"Blueprint {identity.get('capture_id')}",
            "text_prompt": task_prompt,
            "tags": tags,
        },
        "labeling": {
            "capture_grounded": True,
            "generated_output": False,
            "raw_capture_authoritative": True,
            "provider_output_authoritative": False,
        },
    }


def build_blueprint_canonical_site_package(
    *,
    descriptor: Mapping[str, Any],
    capture_root: Path,
    pipeline_dir: Path,
    bucket: str,
    storage_root: Path,
    pipeline_prefix: str,
    descriptor_uri: str,
    qa_report_uri: str,
    qa_report: Mapping[str, Any],
    privacy_processing: Mapping[str, Any],
    worldlabs_input: Mapping[str, Any],
    geometry_artifacts: Mapping[str, Any],
    provenance_summary: Mapping[str, Any],
    rights_provenance_review: Mapping[str, Any],
    site_intake: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    task_targets_payload: Mapping[str, Any],
    task_hypothesis_report: Mapping[str, Any],
    object_index_entries: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    metadata = _as_dict(descriptor.get("metadata"))
    raw_manifest = _read_optional_json(capture_root / "raw" / "manifest.json")
    raw_context = _read_optional_json(capture_root / "raw" / "capture_context.json")
    geometry_summary = _as_dict(geometry_artifacts.get("summary"))
    geometry_readiness = _geometry_readiness(geometry_summary)
    worldlabs_audit = _as_dict(worldlabs_input.get("audit_payload"))
    worldlabs_labeling = _as_dict(worldlabs_input.get("input_labeling"))
    privacy_depth = _as_dict(privacy_processing.get("depth_conditioning"))
    site_identity = _as_dict(metadata.get("site_identity"))
    capture_topology = _as_dict(metadata.get("capture_topology"))
    route_anchors = _as_dict(metadata.get("route_anchors"))
    checkpoint_events = _as_dict(metadata.get("checkpoint_events"))
    site_intake_task_context = _as_dict(site_intake.get("task_context"))
    task_statement = _first_nonempty(
        site_intake_task_context.get("task_statement"),
        metadata.get("task_statement"),
        _as_dict(task_hypothesis_report.get("normalized_task_hypothesis")).get("workflow_name"),
    )
    workflow_context = _first_nonempty(
        site_intake_task_context.get("workflow_context"),
        metadata.get("workflow_context"),
    )
    task_zone = (
        site_intake_task_context.get("task_zone")
        if isinstance(site_intake_task_context.get("task_zone"), Mapping)
        else metadata.get("task_zone")
        if isinstance(metadata.get("task_zone"), Mapping)
        else {}
    )
    semantic_text = json.dumps(
        {
            "task_statement": task_statement,
            "workflow_context": workflow_context,
            "task_zone": task_zone,
            "metadata": metadata,
            "task_hypothesis": task_hypothesis_report,
        },
        sort_keys=True,
    )
    raw_walkthrough = _capture_grounded_ref(
        _string(descriptor.get("raw_video_uri")),
        source="capture_descriptor.raw_video_uri",
        required=True,
    )
    privacy_safe_world_model_input = _derived_ref(
        _string(worldlabs_input.get("output_video_uri")) or _string(descriptor.get("world_model_video_uri")),
        source="worldlabs_input_or_privacy_processing",
        required=True,
    )
    privacy_safe_world_model_input.update(
        {
            "privacy_safe": bool(
                worldlabs_audit.get("privacy_safe_input")
                or worldlabs_labeling.get("privacy_safe_input")
            ),
            "checksum_sha256": worldlabs_audit.get("output_checksum_sha256")
            or worldlabs_input.get("output_checksum_sha256"),
            "source_checksum_sha256": worldlabs_audit.get("source_checksum_sha256")
            or worldlabs_input.get("input_checksum_sha256"),
            "source_manifest_uri": worldlabs_audit.get("source_manifest_uri")
            or privacy_processing.get("privacy_manifest_uri"),
            "input_audit_uri": worldlabs_input.get("audit_uri"),
            "labeling": dict(worldlabs_labeling),
        }
    )
    conditioning = {
        "rgb_video": {
            "raw_walkthrough": raw_walkthrough,
            "privacy_processed_video": _derived_ref(
                _string(privacy_processing.get("privacy_processed_video_uri"))
                or _string(descriptor.get("privacy_processed_video_uri")),
                source="privacy_processing",
            ),
            "privacy_safe_world_model_input": privacy_safe_world_model_input,
            "worldlabs_input_manifest_uri": worldlabs_input.get("manifest_uri"),
            "worldlabs_input_audit_uri": worldlabs_input.get("audit_uri"),
        },
        "frames": {
            "frame_index_uri": _string(descriptor.get("frames_index_uri")),
            "keyframe_uri": _string(descriptor.get("keyframe_uri")) or None,
            "extracted_frame_index_truth": "capture_grounded",
        },
        "temporal_alignment": {
            "frames_index_uri": _string(descriptor.get("frames_index_uri")),
            "motion_log_uri": _string(descriptor.get("motion_log_uri")) or None,
            "arkit_frames_uri": _string(descriptor.get("arkit_frames_uri")) or None,
            "capture_start_epoch_ms": raw_manifest.get("capture_start_epoch_ms"),
            "fps_source": raw_manifest.get("fps_source"),
            "timestamp_truth": "capture_grounded_when_present",
        },
        "camera": {
            "poses_uri": _first_nonempty(
                descriptor.get("arkit_poses_uri"),
                _gs_uri_if_file(
                    bucket=bucket,
                    storage_root=storage_root,
                    path=pipeline_dir / "geometry" / "camera" / "poses.jsonl",
                ),
            ),
            "intrinsics_uri": _first_nonempty(
                descriptor.get("arkit_intrinsics_uri"),
                _gs_uri_if_file(
                    bucket=bucket,
                    storage_root=storage_root,
                    path=pipeline_dir / "geometry" / "camera" / "intrinsics.json",
                ),
            ),
            "trajectory_uri": _gs_uri_if_file(
                bucket=bucket,
                storage_root=storage_root,
                path=pipeline_dir / "geometry" / "camera" / "trajectory_summary.json",
            ),
            "coordinate_frame_session_id": descriptor.get("coordinate_frame_session_id"),
        },
        "depth_confidence": {
            "capture_depth": {
                "depth_prefix_uri": descriptor.get("arkit_depth_prefix_uri"),
                "confidence_prefix_uri": descriptor.get("arkit_confidence_prefix_uri"),
                "source": "arkit" if descriptor.get("arkit_depth_prefix_uri") else None,
            },
            "privacy_depth": {
                "depth_manifest_uri": privacy_depth.get("depth_manifest_uri"),
                "confidence_manifest_uri": privacy_depth.get("confidence_manifest_uri"),
                "depth_prefix_uri": privacy_depth.get("depth_prefix_uri"),
                "confidence_prefix_uri": privacy_depth.get("confidence_prefix_uri"),
                "source": privacy_depth.get("source"),
                "provider": privacy_depth.get("provider"),
            },
            "geometry_depth": {
                "depth_manifest_uri": geometry_artifacts.get("depth_manifest_uri"),
                "confidence_manifest_uri": geometry_artifacts.get("confidence_manifest_uri"),
            },
        },
        "geometry": {
            "manifest_uri": geometry_artifacts.get("geometry_manifest_uri"),
            "summary_uri": geometry_artifacts.get("geometry_summary_uri"),
            "poses_uri": geometry_artifacts.get("camera_poses_uri"),
            "intrinsics_uri": geometry_artifacts.get("camera_intrinsics_uri"),
            "depth_manifest_uri": geometry_artifacts.get("depth_manifest_uri"),
            "confidence_manifest_uri": geometry_artifacts.get("confidence_manifest_uri"),
            "summary": dict(geometry_summary),
            **geometry_readiness,
        },
        "rough_geometry": {
            "geometry_manifest_uri": geometry_artifacts.get("geometry_manifest_uri"),
            "point_cloud_uri": _gs_uri_if_file(
                bucket=bucket,
                storage_root=storage_root,
                path=pipeline_dir / "geometry" / "point_cloud.jsonl",
            ),
            "advanced_geometry_bundle_uri": _gs_uri_if_file(
                bucket=bucket,
                storage_root=storage_root,
                path=pipeline_dir / "advanced_geometry" / "advanced_geometry_bundle.json",
            ),
            "splat_or_mesh_uri": _first_nonempty(
                _gs_uri_if_file(
                    bucket=bucket,
                    storage_root=storage_root,
                    path=pipeline_dir / "advanced_geometry" / "3dgs_compressed.ply",
                ),
                _gs_uri_if_file(
                    bucket=bucket,
                    storage_root=storage_root,
                    path=pipeline_dir / "geometry" / "mesh.glb",
                ),
            ),
        },
    }
    world_labs_readiness = _world_labs_readiness(
        worldlabs_input=worldlabs_input,
        privacy_processing=privacy_processing,
        rights_review=rights_provenance_review,
        provenance_summary=provenance_summary,
    )
    missing_fields: list[str] = []
    if not raw_walkthrough.get("uri"):
        missing_fields.append("conditioning.rgb_video.raw_walkthrough.uri")
    if not privacy_safe_world_model_input.get("uri"):
        missing_fields.append("conditioning.rgb_video.privacy_safe_world_model_input.uri")
    if not conditioning["frames"]["frame_index_uri"]:
        missing_fields.append("conditioning.frames.frame_index_uri")
    if not conditioning["camera"]["poses_uri"]:
        missing_fields.append("conditioning.camera.poses_uri")
    if not conditioning["camera"]["intrinsics_uri"]:
        missing_fields.append("conditioning.camera.intrinsics_uri")
    if not (
        conditioning["depth_confidence"]["capture_depth"].get("depth_prefix_uri")
        or conditioning["depth_confidence"]["privacy_depth"].get("depth_manifest_uri")
        or conditioning["depth_confidence"]["geometry_depth"].get("depth_manifest_uri")
    ):
        missing_fields.append("conditioning.depth_confidence.depth")
    package_uri = _gs_uri_for_path(
        bucket=bucket,
        storage_root=storage_root,
        path=pipeline_dir / "site_package" / "canonical_site_package.json",
    )
    return {
        "schema_version": "v1",
        "package_type": "BlueprintCanonicalSitePackage",
        "generated_at": utc_now_iso(),
        "identity": {
            "scene_id": descriptor.get("scene_id"),
            "capture_id": descriptor.get("capture_id"),
            "site_submission_id": descriptor.get("site_submission_id") or metadata.get("site_submission_id"),
            "buyer_request_id": descriptor.get("buyer_request_id"),
            "capture_job_id": descriptor.get("capture_job_id"),
            "site_id": site_identity.get("site_id") or descriptor.get("scene_id"),
        },
        "canonical_site_package_uri": package_uri,
        "descriptor_uri": descriptor_uri,
        "qa_report_uri": qa_report_uri,
        "pipeline_prefix": pipeline_prefix,
        "conditioning": conditioning,
        "semantic_task_context": {
            "task_statement": task_statement,
            "workflow_context": workflow_context,
            "success_criteria": _string_list(site_intake_task_context.get("success_criteria"))
            or _string_list(metadata.get("success_criteria")),
            "task_zone": task_zone,
            "zones": [value for value in [_string(_as_dict(task_zone).get("label")), site_identity.get("zone_id")] if value],
            "aisles": _semantic_keywords(semantic_text, ["aisle", "lane"]),
            "docks": _semantic_keywords(semantic_text, ["dock", "loading dock"]),
            "doors": _semantic_keywords(semantic_text, ["door", "doorway"]),
            "restricted_areas": _string_list(metadata.get("privacy_restrictions"))
            + _string_list(metadata.get("security_restrictions"))
            + _string_list(metadata.get("capture_restrictions")),
            "target_objects": _target_objects(task_targets_payload, list(object_index_entries or [])),
            "route_anchors": route_anchors,
            "checkpoint_events": checkpoint_events,
            "robot_team_task_prompt": task_statement,
            "task_hypothesis_report_uri": f"gs://{bucket}/{pipeline_prefix}/task_hypothesis_report.json",
            "task_targets_uri": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
        },
        "site_identity_topology": {
            "site_identity": site_identity,
            "capture_topology": capture_topology,
            "capture_orientation": descriptor.get("capture_orientation") or metadata.get("capture_orientation"),
            "raw_manifest_hints": raw_manifest,
        },
        "device_modality": {
            "capture_source": descriptor.get("capture_source"),
            "capture_tier": descriptor.get("capture_tier"),
            "capture_modality": descriptor.get("capture_modality"),
            "evidence_tier": descriptor.get("evidence_tier"),
            "device_model": raw_manifest.get("device_model") or raw_context.get("deviceModel"),
            "os_version": raw_manifest.get("os_version") or raw_context.get("osVersion"),
            "capture_profile_id": descriptor.get("capture_profile_id") or raw_manifest.get("capture_profile_id"),
            "capture_capabilities": metadata.get("capture_capabilities") or raw_manifest.get("capture_capabilities") or {},
            "sensor_availability": _as_dict(_as_dict(metadata.get("scene_memory_capture")).get("sensor_availability")),
            "arkit": {
                "poses_uri": descriptor.get("arkit_poses_uri"),
                "intrinsics_uri": descriptor.get("arkit_intrinsics_uri"),
                "frames_uri": descriptor.get("arkit_frames_uri"),
                "depth_prefix_uri": descriptor.get("arkit_depth_prefix_uri"),
                "confidence_prefix_uri": descriptor.get("arkit_confidence_prefix_uri"),
            },
            "arcore": {
                "poses_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "arcore" / "poses.jsonl"),
                "intrinsics_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "arcore" / "session_intrinsics.json"),
                "depth_manifest_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "arcore" / "depth_manifest.json"),
                "confidence_manifest_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "arcore" / "confidence_manifest.json"),
            },
            "companion_phone": {
                "poses_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "companion_phone" / "poses.jsonl"),
                "intrinsics_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "companion_phone" / "session_intrinsics.json"),
                "calibration_uri": _gs_uri_if_file(bucket=bucket, storage_root=storage_root, path=capture_root / "raw" / "companion_phone" / "calibration.json"),
            },
        },
        "rights_privacy_provenance": {
            "rights": _as_dict(rights_provenance_review.get("rights")),
            "privacy": {
                **_as_dict(rights_provenance_review.get("privacy")),
                "status": privacy_processing.get("status"),
                "mode": privacy_processing.get("mode"),
                "privacy_manifest_uri": privacy_processing.get("privacy_manifest_uri"),
            },
            "provenance": _as_dict(rights_provenance_review.get("provenance")),
            "rights_review_status": rights_provenance_review.get("status"),
            "rights_review_blockers": _string_list(rights_provenance_review.get("blockers")),
            "provenance_summary": dict(provenance_summary),
        },
        "source_checksums": {
            "worldlabs_input": {
                "output_checksum_sha256": worldlabs_audit.get("output_checksum_sha256"),
                "source_checksum_sha256": worldlabs_audit.get("source_checksum_sha256"),
                "source_manifest_uri": worldlabs_audit.get("source_manifest_uri"),
            },
            "qa_report": qa_report.get("checksum_sha256"),
            "raw_manifest": raw_manifest.get("checksum_sha256"),
        },
        "truth_labels": {
            "raw_capture_authoritative": True,
            "capture_grounded_inputs": [
                "raw_walkthrough",
                "frames_index",
                "camera_poses",
                "camera_intrinsics",
                "capture_depth_or_confidence",
                "site_identity",
                "capture_topology",
                "rights_and_consent",
            ],
            "generated_or_derived_outputs": [
                "privacy_processed_video",
                "worldlabs_input_video",
                "pipeline_geometry",
                "scene_memory",
                "provider_world_output",
                "hosted_review",
            ],
            "provider_outputs_non_authoritative": True,
        },
        "provider_readiness": {
            "world_labs_marble": world_labs_readiness,
            "geometry": geometry_readiness,
        },
        "missing_fields": missing_fields,
        "blockers": list(dict.fromkeys(world_labs_readiness["blockers"] + geometry_readiness["blockers"])),
        "qualification_context": {
            "qa_report_status": qa_report.get("status"),
            "completeness_status": scorecard.get("completeness_status"),
            "readiness_state": qualification_record.get("readiness_state"),
        },
        "adapter_mappings": {
            "world_labs_marble": {
                "status": world_labs_readiness["status"],
                "canonical_site_package_uri": package_uri,
                "provider_adapter_input_uri": None,
                "swappable_contract": "ProviderAdapterInput",
            },
            "marble": {
                "status": world_labs_readiness["status"],
                "canonical_site_package_uri": package_uri,
                "provider_adapter_input_uri": None,
                "swappable_contract": "ProviderAdapterInput",
            },
            "cosmos_omniverse_isaac": _future_adapter_mapping(
                "cosmos_omniverse_isaac",
                canonical_site_package_uri=package_uri,
            ),
            "runway": _future_adapter_mapping("runway", canonical_site_package_uri=package_uri),
            "genie_like": _future_adapter_mapping("genie_like", canonical_site_package_uri=package_uri),
            "internal_viewer": _future_adapter_mapping("internal_viewer", canonical_site_package_uri=package_uri),
        },
    }


def write_blueprint_canonical_site_package(
    *,
    descriptor: Mapping[str, Any],
    capture_root: Path,
    pipeline_dir: Path,
    bucket: str,
    storage_root: Path,
    pipeline_prefix: str,
    descriptor_uri: str,
    qa_report_uri: str,
    qa_report: Mapping[str, Any],
    privacy_processing: Mapping[str, Any],
    worldlabs_input: Mapping[str, Any],
    geometry_artifacts: Mapping[str, Any],
    provenance_summary: Mapping[str, Any],
    rights_provenance_review: Mapping[str, Any],
    site_intake: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    task_targets_payload: Mapping[str, Any],
    task_hypothesis_report: Mapping[str, Any],
    object_index_entries: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    site_package_dir = pipeline_dir / "site_package"
    adapter_dir = site_package_dir / "provider_adapter_inputs"
    ensure_dir(adapter_dir)
    package_path = site_package_dir / "canonical_site_package.json"
    adapter_path = adapter_dir / "world_labs_marble.json"
    package_uri = _gs_uri_for_path(bucket=bucket, storage_root=storage_root, path=package_path)
    adapter_uri = _gs_uri_for_path(bucket=bucket, storage_root=storage_root, path=adapter_path)
    package = build_blueprint_canonical_site_package(
        descriptor=descriptor,
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        bucket=bucket,
        storage_root=storage_root,
        pipeline_prefix=pipeline_prefix,
        descriptor_uri=descriptor_uri,
        qa_report_uri=qa_report_uri,
        qa_report=qa_report,
        privacy_processing=privacy_processing,
        worldlabs_input=worldlabs_input,
        geometry_artifacts=geometry_artifacts,
        provenance_summary=provenance_summary,
        rights_provenance_review=rights_provenance_review,
        site_intake=site_intake,
        scorecard=scorecard,
        qualification_record=qualification_record,
        task_targets_payload=task_targets_payload,
        task_hypothesis_report=task_hypothesis_report,
        object_index_entries=object_index_entries,
    )
    package["adapter_mappings"]["world_labs_marble"]["provider_adapter_input_uri"] = adapter_uri
    package["adapter_mappings"]["marble"]["provider_adapter_input_uri"] = adapter_uri
    package_checksum = _json_checksum(package)
    adapter_input = build_world_labs_marble_adapter_input(
        package=package,
        canonical_site_package_uri=package_uri,
        provider_adapter_input_uri=adapter_uri,
        source_package_checksum_sha256=package_checksum,
    )
    write_json(package_path, package)
    write_json(adapter_path, adapter_input)
    return {
        "canonical_site_package": package,
        "canonical_site_package_uri": package_uri,
        "canonical_site_package_path": str(package_path),
        "provider_adapter_inputs": {"world_labs_marble": adapter_input},
        "provider_adapter_input_uris": {"world_labs_marble": adapter_uri},
        "provider_adapter_input_paths": {"world_labs_marble": str(adapter_path)},
    }
