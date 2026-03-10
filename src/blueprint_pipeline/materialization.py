"""Materialize qualification inputs from raw capture uploads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import (
    ensure_dir,
    join_gs_uri,
    parse_bool,
    read_json,
    resolve_gs_uri_to_path,
    try_parse_float,
    utc_now_iso,
    write_json,
)


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    out: List[str] = []
    for item in values:
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _dict_float(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, float] = {}
    for key, item in value.items():
        text = str(key).strip()
        if not text:
            continue
        try:
            out[text] = float(item)
        except (TypeError, ValueError):
            continue
    return out


def _raw_video_candidates(raw_root: Path) -> List[str]:
    names = [
        "walkthrough.mov",
        "walkthrough.mp4",
        "recording.mov",
        "recording.mp4",
    ]
    out: List[str] = []
    for name in names:
        path = raw_root / name
        if path.is_file():
            out.append(name)
    return out


def _capture_source(manifest: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    for candidate in (
        str(manifest.get("capture_source") or "").strip().lower(),
        str(context.get("captureSource") or "").strip().lower(),
    ):
        if candidate in {"iphone", "glasses"}:
            return candidate
        if candidate == "iphonevideo":
            return "iphone"
        if candidate == "metaglasses":
            return "glasses"
    return "iphone"


def _capture_tier(source: str, manifest: Mapping[str, Any]) -> str:
    tier = str(manifest.get("capture_tier_hint") or "").strip()
    if tier:
        return tier
    return "tier2_glasses" if source == "glasses" else "tier1_iphone"


def _capture_modality(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    source: str,
    scaffolding_used: List[str],
) -> str:
    explicit = str(context.get("captureModality") or manifest.get("capture_modality") or "").strip().lower()
    if explicit in {"iphone_arkit_lidar", "glasses_video_only", "glasses_plus_scaffolding"}:
        return explicit
    if source == "iphone" and parse_bool(manifest.get("has_lidar"), default=False):
        return "iphone_arkit_lidar"
    if source == "glasses" and scaffolding_used:
        return "glasses_plus_scaffolding"
    if source == "glasses":
        return "glasses_video_only"
    return "iphone_arkit_lidar"


def _has_minimum_intake(intake: Mapping[str, Any]) -> bool:
    return bool(
        str(intake.get("workflowName") or "").strip()
        and _string_list(intake.get("taskSteps"))
        and (
            str(intake.get("zone") or "").strip()
            or str(intake.get("owner") or "").strip()
        )
    )


def _evidence_tier(
    *,
    source: str,
    modality: str,
    intake_complete: bool,
    calibration_assets: List[str],
    scaffolding_validation: Mapping[str, Any],
) -> str:
    if source == "glasses":
        if (
            modality == "glasses_plus_scaffolding"
            and intake_complete
            and calibration_assets
            and parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False)
        ):
            return "glasses_with_validated_scaffolding"
        return "pre_screen_video"
    if modality == "iphone_arkit_lidar" and intake_complete:
        return "qualified_metric_capture"
    return "pre_screen_video"


def materialize_capture_bundle(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    gcs_root: Path,
    raw_prefix_uri: Optional[str] = None,
) -> Dict[str, Any]:
    raw_prefix_uri = raw_prefix_uri or f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw"
    raw_root = resolve_gs_uri_to_path(raw_prefix_uri, gcs_root)
    capture_root = raw_root.parent

    manifest_path = raw_root / "manifest.json"
    intake_path = raw_root / "intake_packet.json"
    context_path = raw_root / "capture_context.json"

    manifest = _read_optional_json(manifest_path)
    intake = _read_optional_json(intake_path)
    context = _read_optional_json(context_path)

    source = _capture_source(manifest, context)
    tier = _capture_tier(source, manifest)
    scaffolding_used = _string_list(context.get("scaffoldingUsed") or manifest.get("scaffolding_used"))
    coverage_plan = _string_list(context.get("coveragePlan") or manifest.get("coverage_plan"))
    calibration_assets = _string_list(context.get("calibrationAssets") or manifest.get("calibration_assets"))
    uncertainty_priors = _dict_float(context.get("uncertaintyPriors") or manifest.get("uncertainty_priors"))
    modality = _capture_modality(manifest, context, source, scaffolding_used)

    arkit_root = raw_root / "arkit"
    arkit_poses_uri = None
    arkit_intrinsics_uri = None
    arkit_depth_prefix_uri = None
    arkit_confidence_prefix_uri = None
    if (arkit_root / "poses.jsonl").is_file():
        arkit_poses_uri = join_gs_uri(raw_prefix_uri, "arkit/poses.jsonl")
    if (arkit_root / "intrinsics.json").is_file():
        arkit_intrinsics_uri = join_gs_uri(raw_prefix_uri, "arkit/intrinsics.json")
    if (arkit_root / "depth").is_dir():
        arkit_depth_prefix_uri = join_gs_uri(raw_prefix_uri, "arkit/depth")
    if (arkit_root / "confidence").is_dir():
        arkit_confidence_prefix_uri = join_gs_uri(raw_prefix_uri, "arkit/confidence")

    frames_index_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/frames/index.jsonl"
    frames_dir = capture_root / "frames"
    ensure_dir(frames_dir)
    frames_path = frames_dir / "index.jsonl"
    frame_index_payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "raw_prefix_uri": raw_prefix_uri,
        "video_candidates": _raw_video_candidates(raw_root),
        "generated_at": utc_now_iso(),
    }
    frames_path.write_text(json.dumps(frame_index_payload) + "\n", encoding="utf-8")

    video_candidates = _raw_video_candidates(raw_root)
    raw_video_uri = join_gs_uri(raw_prefix_uri, video_candidates[0]) if video_candidates else str(manifest.get("video_uri") or "").strip() or None
    intake_packet_uri = join_gs_uri(raw_prefix_uri, "intake_packet.json") if intake_path.is_file() else None
    intake_complete = _has_minimum_intake(intake)
    validated_scale_raw = context.get("validatedScaleMeters") or manifest.get("validated_scale_m")
    validated_scale_m = None
    if validated_scale_raw is not None:
        validated_scale_m = try_parse_float(validated_scale_raw, 0.0)
    validated_pose_coverage = try_parse_float(
        context.get("validatedPoseCoverage") or manifest.get("validated_pose_coverage"),
        0.0,
    )
    hidden_zone_bound = try_parse_float(
        context.get("hiddenZoneBound") or manifest.get("hidden_zone_bound"),
        1.0,
    )
    scale_anchor_count = len(_string_list(context.get("scaleAnchorAssets") or manifest.get("scale_anchor_assets")))
    checkpoint_count = len(_string_list(context.get("checkpointAssets") or manifest.get("checkpoint_assets")))
    scaffolding_validation = {
        "scale_anchor_count": scale_anchor_count,
        "checkpoint_count": checkpoint_count,
        "validated_scale_m": validated_scale_m,
        "validated_pose_coverage": round(float(validated_pose_coverage or 0.0), 4),
        "hidden_zone_bound": round(float(hidden_zone_bound or 1.0), 4),
        "validated_metric_bundle": bool(
            modality == "glasses_plus_scaffolding"
            and calibration_assets
            and validated_scale_m is not None
            and float(validated_pose_coverage or 0.0) >= 0.7
            and float(hidden_zone_bound or 1.0) <= 0.35
            and scale_anchor_count > 0
            and checkpoint_count > 0
        ),
    }
    evidence_tier = _evidence_tier(
        source=source,
        modality=modality,
        intake_complete=intake_complete,
        calibration_assets=calibration_assets,
        scaffolding_validation=scaffolding_validation,
    )

    metadata: Dict[str, Any] = {
        "site_submission_id": f"{scene_id}:{capture_id}",
        "opportunity_id": scene_id,
        "task_statement": str(intake.get("workflowName") or manifest.get("scene_id") or scene_id),
        "workflow_context": " | ".join(_string_list(intake.get("taskSteps"))),
        "success_criteria": [str(intake.get("targetKPI") or "").strip()] if str(intake.get("targetKPI") or "").strip() else [],
        "task_zone": {"label": str(intake.get("zone") or "").strip()} if str(intake.get("zone") or "").strip() else {},
        "operating_constraints": [value for value in [str(intake.get("shift") or "").strip()] if value],
        "privacy_restrictions": _string_list(intake.get("privacySecurityLimits")),
        "security_restrictions": _string_list(intake.get("captureRestrictions")),
        "known_blockers": _string_list(intake.get("knownBlockers")),
        "owner": str(intake.get("owner") or "").strip() or None,
        "adjacent_systems": _string_list(intake.get("adjacentSystems")),
        "non_routine_modes": _string_list(intake.get("nonRoutineModes")),
        "people_traffic_notes": _string_list(intake.get("peopleTrafficNotes")),
        "capture_restrictions": _string_list(intake.get("captureRestrictions")),
        "capture_modality": modality,
        "evidence_tier": evidence_tier,
        "scaffolding_used": scaffolding_used,
        "coverage_plan": coverage_plan,
        "calibration_assets": calibration_assets,
        "uncertainty_priors": uncertainty_priors,
        "scaffolding_validation": scaffolding_validation,
    }

    descriptor = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_source": source,
        "capture_tier": tier,
        "capture_modality": modality,
        "evidence_tier": evidence_tier,
        "raw_prefix_uri": raw_prefix_uri,
        "frames_index_uri": frames_index_uri,
        "raw_video_uri": raw_video_uri,
        "arkit_poses_uri": arkit_poses_uri,
        "arkit_intrinsics_uri": arkit_intrinsics_uri,
        "arkit_depth_prefix_uri": arkit_depth_prefix_uri,
        "arkit_confidence_prefix_uri": arkit_confidence_prefix_uri,
        "qa_report_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/qa_report.json",
        "qa_status": None,
        "intended_space_type": str(manifest.get("intended_space_type") or "default"),
        "scaffolding_used": scaffolding_used,
        "intake_packet_uri": intake_packet_uri,
        "coverage_plan": coverage_plan,
        "calibration_assets": calibration_assets,
        "scaffolding_validation": scaffolding_validation,
        "uncertainty_priors": uncertainty_priors,
        "requested_lanes": (
            ["qualification", "advanced_geometry"]
            if evidence_tier in {"qualified_metric_capture", "glasses_with_validated_scaffolding"}
            else ["qualification"]
        ),
        "quality": {
            "pose_match_rate": try_parse_float(manifest.get("pose_match_rate"), 0.95 if modality == "iphone_arkit_lidar" else 0.35),
            "has_metric_geometry": evidence_tier in {"qualified_metric_capture", "glasses_with_validated_scaffolding"},
            "intake_complete": intake_complete,
        },
        "metadata": metadata,
    }

    hidden_zone_score = min(
        1.0,
        0.2 * len(_string_list(intake.get("captureRestrictions")))
        + 0.15 * len(_string_list(intake.get("privacySecurityLimits"))),
    )
    uncertainty_score = 0.15 if modality == "iphone_arkit_lidar" else 0.45
    if modality == "glasses_video_only":
        uncertainty_score = 0.78
    if not intake_complete:
        uncertainty_score = min(1.0, uncertainty_score + 0.15)
    if not raw_video_uri:
        uncertainty_score = min(1.0, uncertainty_score + 0.25)
    if modality == "glasses_plus_scaffolding" and not parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False):
        uncertainty_score = min(1.0, uncertainty_score + 0.2)
    if hidden_zone_bound is not None:
        uncertainty_score = min(1.0, uncertainty_score + max(0.0, float(hidden_zone_bound) - 0.2) * 0.4)

    checks = [
        {
            "name": "raw_manifest_present",
            "passed": manifest_path.is_file(),
            "detail": "raw manifest present" if manifest_path.is_file() else "raw manifest missing",
        },
        {
            "name": "raw_video_present",
            "passed": bool(raw_video_uri),
            "detail": raw_video_uri or "raw video missing",
        },
        {
            "name": "intake_present",
            "passed": intake_path.is_file(),
            "detail": "intake packet present" if intake_path.is_file() else "intake packet missing",
        },
        {
            "name": "intake_complete",
            "passed": intake_complete,
            "detail": "intake has workflow, steps, and zone/owner" if intake_complete else "intake missing workflow, steps, or zone/owner",
        },
        {
            "name": "metric_geometry_present",
            "passed": evidence_tier in {"qualified_metric_capture", "glasses_with_validated_scaffolding"},
            "detail": "validated metric evidence present" if evidence_tier in {"qualified_metric_capture", "glasses_with_validated_scaffolding"} else "metric geometry not present",
        },
        {
            "name": "scaffolding_validated",
            "passed": modality != "glasses_plus_scaffolding" or parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False),
            "detail": "scaffolding validated for metric checks" if modality != "glasses_plus_scaffolding" or parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False) else "glasses scaffolding lacks validated scale/pose coverage",
        },
    ]

    if evidence_tier == "qualified_metric_capture":
        status = "passed" if manifest_path.is_file() and raw_video_uri and intake_complete else "degraded"
    elif evidence_tier == "glasses_with_validated_scaffolding":
        status = "passed" if manifest_path.is_file() and raw_video_uri and intake_complete else "degraded"
    else:
        status = "degraded"

    qa_report = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_modality": modality,
        "evidence_tier": evidence_tier,
        "uncertainty_score": round(uncertainty_score, 4),
        "hidden_zone_score": round(hidden_zone_score, 4),
        "hidden_zone_bound": round(float(hidden_zone_bound or 1.0), 4),
        "scaffolding_validation": scaffolding_validation,
        "checks": checks,
        "escalation_recommendation": {
            "recommended_lane": "advanced_geometry" if status == "passed" and descriptor["requested_lanes"] == ["qualification", "advanced_geometry"] else "qualification",
            "human_review_required": evidence_tier != "qualified_metric_capture" or uncertainty_score >= 0.3,
            "reason": (
                "validated metric capture supports geometry-backed automation"
                if evidence_tier in {"qualified_metric_capture", "glasses_with_validated_scaffolding"}
                else "capture remains pre-screen only because metric evidence is incomplete"
            ),
        },
    }

    write_json(capture_root / "capture_descriptor.json", descriptor)
    write_json(capture_root / "qa_report.json", qa_report)

    return {
        "descriptor_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/capture_descriptor.json",
        "qa_report_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/qa_report.json",
        "descriptor": descriptor,
        "qa_report": qa_report,
    }
