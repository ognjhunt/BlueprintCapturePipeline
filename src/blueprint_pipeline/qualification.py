"""Qualification-first orchestration helpers and artifact builders."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import (
    PipelineError,
    StageError,
    ensure_dir,
    has_nonempty_file,
    infer_storage_root_from_scene_path,
    parse_gs_uri,
    read_json,
    relative_scene_path,
    resolve_gs_uri_to_path,
    to_pipeline_prefix,
    utc_now_iso,
    write_json,
)
from .ios_manifest import IOSManifest, load_object_index, load_raw_manifest, resolve_object_index_uri
from .task_targets import infer_task_targets, write_task_targets


@dataclass
class QualificationGate:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _local_file_pointer(path: Path) -> Dict[str, Any]:
    pointer: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return pointer
    stat = path.stat()
    pointer["size_bytes"] = int(stat.st_size)
    pointer["modified_at"] = utc_now_iso()
    return pointer


def _default_qa_report_uri(descriptor_uri: str) -> str:
    parsed = parse_gs_uri(descriptor_uri)
    if parsed.key.endswith("capture_descriptor.json"):
        qa_key = parsed.key[: -len("capture_descriptor.json")] + "qa_report.json"
    else:
        qa_key = f"{parsed.key.rstrip('/')}/qa_report.json"
    return f"gs://{parsed.bucket}/{qa_key}"


def _safe_path_exists(uri: Optional[str], storage_root: Path) -> bool:
    if not uri:
        return False
    try:
        return resolve_gs_uri_to_path(uri, storage_root).exists()
    except Exception:
        return False


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return read_json(path)
    except Exception:
        return None


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _relative_path_from(base_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, base_dir)).as_posix()


def _normalize_local_path(path_value: Any, *, base_dir: Path) -> Optional[str]:
    raw = str(path_value or "").strip()
    if not raw or raw.startswith("gs://"):
        return None
    candidate = Path(raw)
    resolved = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
    return _relative_path_from(base_dir, resolved)


def _metadata_scene_package_path(metadata: Mapping[str, Any], *, base_dir: Path) -> Optional[str]:
    scene_package = (
        metadata.get("scene_package")
        if isinstance(metadata.get("scene_package"), Mapping)
        else {}
    )
    for value in (
        scene_package.get("scene_package_path"),
        scene_package.get("root_path"),
        scene_package.get("bundle_path"),
        metadata.get("scene_package_path"),
    ):
        normalized = _normalize_local_path(value, base_dir=base_dir)
        if normalized:
            return normalized
    return None


def attach_handoff_package_paths(
    handoff_payload: Mapping[str, Any],
    *,
    pipeline_dir: Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(handoff_payload)
    handoff_dir = pipeline_dir
    advanced_dir = pipeline_dir / "advanced_geometry"
    bundle_manifest = advanced_dir / "advanced_geometry_bundle.json"

    if bundle_manifest.is_file():
        geometry_package: Dict[str, Any] = {"bundle_path": _relative_path_from(handoff_dir, advanced_dir)}
        optional_files = {
            "ply_path": advanced_dir / "3dgs_compressed.ply",
            "labels_path": advanced_dir / "labels.json",
            "structure_path": advanced_dir / "structure.json",
            "task_hints_path": advanced_dir / "task_targets.synthetic.json",
        }
        for key, path in optional_files.items():
            if path.is_file():
                geometry_package[key] = _relative_path_from(handoff_dir, path)
        payload["geometry_package"] = geometry_package
    else:
        payload.pop("geometry_package", None)

    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    scene_package_path = _metadata_scene_package_path(metadata_mapping, base_dir=handoff_dir)
    if scene_package_path:
        payload["scene_package"] = {"scene_package_path": scene_package_path}
    else:
        payload.pop("scene_package", None)

    return payload


def _disabled_task_targets(scene_id: str, capture_id: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "inference_mode": "disabled",
        "video_analysis": {"external_inference": {"status": "skipped", "reason": reason}},
        "manipulation_candidates": [],
        "articulation_hints": [],
        "target_object_ids": [],
        "articulation_required_ids": [],
        "tasks": [],
    }


def _build_runtime_preflight_report(
    *,
    descriptor_path: Path,
    qa_report_path: Path,
    manifest_path: Optional[Path],
    object_index_path: Optional[Path],
    gcs_root: Path,
) -> Dict[str, Any]:
    checks = [
        QualificationGate(
            "gcs_root",
            gcs_root.exists() and gcs_root.is_dir(),
            f"found {gcs_root}" if gcs_root.exists() and gcs_root.is_dir() else f"missing {gcs_root}",
        ),
        QualificationGate(
            "descriptor_access",
            descriptor_path.is_file(),
            f"found {descriptor_path}" if descriptor_path.is_file() else f"missing {descriptor_path}",
        ),
        QualificationGate(
            "qa_report_access",
            qa_report_path.is_file(),
            f"found {qa_report_path}" if qa_report_path.is_file() else f"missing {qa_report_path}",
        ),
        QualificationGate(
            "raw_manifest_access",
            bool(manifest_path and manifest_path.is_file()),
            f"found {manifest_path}" if manifest_path and manifest_path.is_file() else "missing raw manifest",
        ),
        QualificationGate(
            "object_index_access",
            bool(object_index_path and object_index_path.is_file()),
            f"found {object_index_path}" if object_index_path and object_index_path.is_file() else "missing object index",
        ),
    ]
    status = "passed" if all(check.passed for check in checks) else "degraded"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "status": status,
        "generated_at": utc_now_iso(),
        "checks": [check.to_dict() for check in checks],
    }


def _build_capture_package_manifest(
    *,
    descriptor: CaptureDescriptor,
    descriptor_uri: str,
    qa_report_uri: str,
    manifest_uri: Optional[str],
    object_index_uri: Optional[str],
    storage_root: Path,
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    evidence_items = [
        {"name": "descriptor", "uri": descriptor_uri, "exists": _safe_path_exists(descriptor_uri, storage_root)},
        {"name": "qa_report", "uri": qa_report_uri, "exists": _safe_path_exists(qa_report_uri, storage_root)},
        {"name": "raw_manifest", "uri": manifest_uri, "exists": _safe_path_exists(manifest_uri, storage_root)},
        {
            "name": "object_index",
            "uri": object_index_uri,
            "exists": _safe_path_exists(object_index_uri, storage_root),
            "object_count": len(object_index_entries),
        },
        {
            "name": "frames_index",
            "uri": descriptor.frames_index_uri,
            "exists": _safe_path_exists(descriptor.frames_index_uri, storage_root),
        },
        {
            "name": "raw_video",
            "uri": descriptor.raw_video_uri,
            "exists": _safe_path_exists(descriptor.raw_video_uri, storage_root),
        },
        {
            "name": "keyframe",
            "uri": descriptor.keyframe_uri,
            "exists": _safe_path_exists(descriptor.keyframe_uri, storage_root),
        },
        {
            "name": "arkit_poses",
            "uri": descriptor.arkit_poses_uri,
            "exists": _safe_path_exists(descriptor.arkit_poses_uri, storage_root),
        },
        {
            "name": "arkit_intrinsics",
            "uri": descriptor.arkit_intrinsics_uri,
            "exists": _safe_path_exists(descriptor.arkit_intrinsics_uri, storage_root),
        },
    ]
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "requested_lanes": list(descriptor.requested_lanes),
        "descriptor_uri": descriptor_uri,
        "qa_report_uri": qa_report_uri,
        "raw_prefix_uri": descriptor.raw_prefix_uri,
        "raw_manifest_uri": manifest_uri,
        "object_index_uri": object_index_uri,
        "evidence_items": evidence_items,
        "counts": {
            "object_index_count": len(object_index_entries),
            "manipulation_candidates": len(descriptor.manipulation_candidates),
            "articulation_hints": len(descriptor.articulation_hints),
        },
    }


def _build_completeness_scorecard(
    *,
    descriptor: CaptureDescriptor,
    qa_report: Mapping[str, Any],
    manifest: Optional[IOSManifest],
    object_index_uri: Optional[str],
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    qa_status = str(qa_report.get("status") or "missing").strip().lower()
    checks = [
        QualificationGate(
            "qa_report_present",
            qa_status != "missing",
            f"qa_status={qa_status}",
        ),
        QualificationGate(
            "qa_usable",
            qa_status == "passed",
            "qa passed" if qa_status == "passed" else f"qa status is {qa_status}",
        ),
        QualificationGate(
            "raw_manifest_present",
            manifest is not None,
            "raw manifest resolved" if manifest is not None else "raw manifest missing",
        ),
        QualificationGate(
            "object_index_present",
            bool(object_index_uri),
            "object index resolved" if object_index_uri else "object index missing",
        ),
        QualificationGate(
            "object_index_populated",
            len(object_index_entries) > 0,
            f"{len(object_index_entries)} indexed objects",
        ),
        QualificationGate(
            "task_evidence_present",
            bool(descriptor.frames_index_uri or descriptor.raw_video_uri or descriptor.keyframe_uri),
            "capture evidence URIs present"
            if descriptor.frames_index_uri or descriptor.raw_video_uri or descriptor.keyframe_uri
            else "missing capture evidence URIs",
        ),
    ]
    follow_ups = [check.detail for check in checks if not check.passed]
    passed_count = sum(1 for check in checks if check.passed)
    score = round(passed_count / float(len(checks) or 1), 4)
    completeness_status = "sufficient" if all(check.passed for check in checks[1:]) else "need_more_evidence"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "qa_status": qa_status,
        "completeness_status": completeness_status,
        "score": score,
        "checks": [check.to_dict() for check in checks],
        "follow_ups": follow_ups,
    }


def _build_task_scope_record(
    *,
    descriptor: CaptureDescriptor,
    task_targets_payload: Mapping[str, Any],
    completeness_status: str,
) -> Dict[str, Any]:
    target_object_ids = [
        str(value)
        for value in task_targets_payload.get("target_object_ids", [])
        if str(value).strip()
    ]
    articulation_required_ids = [
        str(value)
        for value in task_targets_payload.get("articulation_required_ids", [])
        if str(value).strip()
    ]
    tasks = [
        dict(item)
        for item in task_targets_payload.get("tasks", [])
        if isinstance(item, Mapping)
    ]
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    assumptions = [
        str(item).strip()
        for item in metadata.get("assumptions", [])
        if str(item).strip()
    ] if isinstance(metadata.get("assumptions"), list) else []
    blockers = [
        str(item).strip()
        for item in metadata.get("known_blockers", [])
        if str(item).strip()
    ] if isinstance(metadata.get("known_blockers"), list) else []
    if not target_object_ids and not tasks:
        blockers.append("Task zone is not yet well scoped from the available evidence.")
    if completeness_status != "sufficient":
        blockers.append("Additional evidence is required before scope can be locked.")
    success_criteria = [
        str(item).strip()
        for item in metadata.get("success_criteria", [])
        if str(item).strip()
    ] if isinstance(metadata.get("success_criteria"), list) else []
    if not success_criteria:
        success_criteria = ["Identify the task zone, key objects, and blockers well enough for buyer review."]
    scope_status = "scoped" if target_object_ids or tasks else "needs_clarification"
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "scope_status": scope_status,
        "task_statement": str(metadata.get("task_statement") or metadata.get("workflow_context") or "unknown_task"),
        "buyer_type": str(metadata.get("buyer_type") or "unknown"),
        "task_zone": metadata.get("task_zone") if isinstance(metadata.get("task_zone"), Mapping) else {},
        "adjacent_workflow_context": str(metadata.get("workflow_context") or "unknown"),
        "target_object_ids": target_object_ids,
        "articulation_required_ids": articulation_required_ids,
        "tasks": tasks,
        "manipulation_candidates": [
            dict(item)
            for item in task_targets_payload.get("manipulation_candidates", [])
            if isinstance(item, Mapping)
        ],
        "articulation_hints": [
            dict(item)
            for item in task_targets_payload.get("articulation_hints", [])
            if isinstance(item, Mapping)
        ],
        "assumptions": assumptions,
        "blockers": blockers,
        "success_criteria": success_criteria,
    }


def _score_bucket(value: float, detail: str) -> Dict[str, Any]:
    bounded = max(0.0, min(1.0, float(value)))
    return {"score": round(bounded, 4), "detail": detail}


def _build_qualification_record(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    completeness_status = str(scorecard.get("completeness_status") or "need_more_evidence")
    qa_status = str(scorecard.get("qa_status") or "missing")
    target_object_ids = scope_record.get("target_object_ids", [])
    articulation_required_ids = scope_record.get("articulation_required_ids", [])
    scope_status = str(scope_record.get("scope_status") or "needs_clarification")

    privacy_restricted = bool(descriptor.metadata.get("privacy_restrictions")) if isinstance(descriptor.metadata, Mapping) else False
    safety_concerns = descriptor.metadata.get("safety_concerns") if isinstance(descriptor.metadata, Mapping) else []
    safety_count = len(safety_concerns) if isinstance(safety_concerns, list) else 0

    rubric = {
        "physical_access": _score_bucket(
            0.8 if object_index_entries else 0.35,
            "Object index coverage provides a usable view of the task zone."
            if object_index_entries
            else "Physical clearances remain uncertain because the object index is missing.",
        ),
        "task_repeatability": _score_bucket(
            0.75 if target_object_ids else 0.4,
            "Task targets were inferred from the capture package."
            if target_object_ids
            else "No stable task targets were inferred from the current evidence.",
        ),
        "environmental_conditions": _score_bucket(
            0.8 if qa_status == "passed" else 0.3,
            "Capture QA passed."
            if qa_status == "passed"
            else "Capture QA did not pass, so the environment evidence is unreliable.",
        ),
        "safety_process_constraints": _score_bucket(
            0.45 if safety_count else 0.75,
            "Safety or process constraints were flagged in metadata."
            if safety_count
            else "No explicit safety blockers were supplied in metadata.",
        ),
        "integration_friction": _score_bucket(
            0.55 if articulation_required_ids else 0.8,
            "Articulated targets suggest extra integration complexity."
            if articulation_required_ids
            else "No articulated manipulation requirement was inferred.",
        ),
        "evidence_completeness": _score_bucket(
            float(scorecard.get("score") or 0.0),
            f"Completeness status is {completeness_status}.",
        ),
    }

    risks: List[Dict[str, Any]] = []
    if completeness_status != "sufficient":
        risks.append(
            {
                "id": "need_more_evidence",
                "severity": "high",
                "category": "evidence",
                "detail": "Additional capture evidence is required before the opportunity can be routed.",
            }
        )
    if scope_status != "scoped":
        risks.append(
            {
                "id": "scope_ambiguity",
                "severity": "medium",
                "category": "scoping",
                "detail": "Task scope remains ambiguous from the available capture evidence.",
            }
        )
    if articulation_required_ids:
        risks.append(
            {
                "id": "articulation_complexity",
                "severity": "medium",
                "category": "integration",
                "detail": "Articulated targets indicate a more complex manipulation environment.",
            }
        )
    if privacy_restricted:
        risks.append(
            {
                "id": "privacy_restrictions",
                "severity": "medium",
                "category": "access",
                "detail": "Privacy restrictions may hide decision-critical areas.",
            }
        )

    blockers = [
        str(item).strip()
        for item in scope_record.get("blockers", [])
        if str(item).strip()
    ]

    rubric_scores = [float(item.get("score") or 0.0) for item in rubric.values()]
    confidence = round(sum(rubric_scores) / float(len(rubric_scores) or 1), 4)
    if scope_status == "scoped":
        confidence = round(min(1.0, confidence + 0.05), 4)

    if completeness_status != "sufficient":
        readiness_state = "not_ready_yet"
    elif confidence >= 0.78 and not any(risk["severity"] == "high" for risk in risks):
        readiness_state = "ready"
    else:
        readiness_state = "risky"

    advanced_geometry_recommended = completeness_status == "sufficient" and bool(
        target_object_ids or articulation_required_ids
    )

    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "readiness_state": readiness_state,
        "confidence": confidence,
        "advanced_geometry_recommended": advanced_geometry_recommended,
        "rubric": rubric,
        "risks": risks,
        "blockers": blockers,
        "escalation": {
            "required": readiness_state != "ready" or advanced_geometry_recommended,
            "reason": "Need more evidence" if readiness_state == "not_ready_yet" else "Advanced geometry is recommended",
        },
    }


def _build_qualification_brief(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
) -> Dict[str, Any]:
    readiness_state = str(qualification_record.get("readiness_state") or "not_ready_yet")
    completeness_status = str(scorecard.get("completeness_status") or "need_more_evidence")
    risks = qualification_record.get("risks", [])
    risk_summaries = [str(item.get("detail") or "") for item in risks if isinstance(item, Mapping)]
    next_steps: List[str] = []
    if completeness_status != "sufficient":
        next_steps.append("Collect another capture pass that closes evidence gaps called out in the QA scorecard.")
    if str(scope_record.get("scope_status") or "") != "scoped":
        next_steps.append("Clarify the task zone, workflow, and target objects before routing.")
    if bool(qualification_record.get("advanced_geometry_recommended")):
        next_steps.append("Escalate to the advanced geometry lane for richer object-localized geometry outputs.")
    if not next_steps:
        next_steps.append("Route the opportunity handoff to robot-team review.")
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "headline": f"{descriptor.scene_id} is {readiness_state.replace('_', ' ')} for qualification routing.",
        "readiness_state": readiness_state,
        "completeness_status": completeness_status,
        "confidence": float(qualification_record.get("confidence") or 0.0),
        "task_statement": scope_record.get("task_statement"),
        "top_risks": risk_summaries[:3],
        "next_steps": next_steps,
    }


def _build_opportunity_handoff(
    *,
    descriptor: CaptureDescriptor,
    scorecard: Mapping[str, Any],
    scope_record: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
    brief: Mapping[str, Any],
    config: Any,
    pipeline_dir: Path,
) -> Dict[str, Any]:
    metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    readiness_state = str(qualification_record.get("readiness_state") or "not_ready_yet")
    completeness_status = str(scorecard.get("completeness_status") or "need_more_evidence")
    confidence = float(qualification_record.get("confidence") or 0.0)
    match_ready = completeness_status == "sufficient" and readiness_state == "ready" and confidence >= 0.78
    recommended_lane = (
        "advanced_geometry"
        if bool(qualification_record.get("advanced_geometry_recommended"))
        else "qualification"
    )
    site_submission_id = str(metadata.get("site_submission_id") or "").strip() or (
        f"{descriptor.scene_id}:{descriptor.capture_id}"
    )
    opportunity_id = str(metadata.get("opportunity_id") or "").strip() or site_submission_id
    task_statement = (
        str(metadata.get("task_statement") or "").strip()
        or str(metadata.get("workflow_context") or "").strip()
        or f"Qualification review for {descriptor.scene_id}/{descriptor.capture_id}"
    )
    operating_constraints = _string_list(metadata.get("operating_constraints"))
    if not operating_constraints:
        operating_hours = str(metadata.get("operating_hours") or "").strip()
        if operating_hours:
            operating_constraints = [operating_hours]
    if not operating_constraints:
        operating_constraints = ["Not provided in intake metadata"]

    privacy_security_constraints = _string_list(metadata.get("privacy_restrictions"))
    privacy_security_constraints.extend(
        value
        for value in _string_list(metadata.get("security_restrictions"))
        if value not in privacy_security_constraints
    )
    if not privacy_security_constraints:
        privacy_security_constraints = ["Not provided in intake metadata"]

    known_blockers = _string_list(scope_record.get("blockers"))
    if not known_blockers:
        known_blockers = _string_list(metadata.get("known_blockers"))
    if not known_blockers:
        known_blockers = ["No known blockers supplied"]

    task_zone = scope_record.get("task_zone") if isinstance(scope_record.get("task_zone"), Mapping) else {}
    target_object_ids = _string_list(scope_record.get("target_object_ids"))
    in_scope_zone: Any
    if task_zone:
        in_scope_zone = dict(task_zone)
    elif target_object_ids:
        in_scope_zone = target_object_ids
    else:
        in_scope_zone = descriptor.environment_type_hint or descriptor.scene_id

    success_criteria = _string_list(scope_record.get("success_criteria"))
    target_robot_team = (
        metadata.get("target_robot_team")
        if isinstance(metadata.get("target_robot_team"), Mapping)
        else {}
    )
    robot_platform = (
        str(target_robot_team.get("robot_platform") or "").strip()
        or str(metadata.get("robot_platform") or "").strip()
        or str(getattr(config, "robot_type", "") or "").strip()
        or "franka"
    )
    handoff = {
        "schema_version": "v1",
        "lane": "qualification",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "generated_at": utc_now_iso(),
        "site_submission_id": site_submission_id,
        "opportunity_id": opportunity_id,
        "qualification_state": readiness_state,
        "downstream_evaluation_eligibility": readiness_state == "ready",
        "operator_approved_summary": (
            str(metadata.get("operator_approved_summary") or "").strip()
            or str(brief.get("headline") or "").strip()
        ),
        "scoped_task_definition": {
            "task_id": str(metadata.get("task_id") or "").strip() or opportunity_id,
            "scoped_task_statement": task_statement,
            "success_criteria": success_criteria,
            "in_scope_zone": in_scope_zone,
        },
        "site_constraints": {
            "operating_constraints": operating_constraints,
            "privacy_security_constraints": privacy_security_constraints,
            "known_blockers": known_blockers,
        },
        "target_robot_team": {
            "team_name_or_id": (
                str(target_robot_team.get("team_name_or_id") or "").strip()
                or str(metadata.get("team_name_or_id") or "").strip()
                or "default_robot_team"
            ),
            "robot_platform": robot_platform,
            "embodiment_notes": (
                str(target_robot_team.get("embodiment_notes") or "").strip()
                or str(metadata.get("embodiment_notes") or "").strip()
                or f"Qualification-default targeting for {robot_platform}"
            ),
        },
        "match_ready": match_ready,
        "recommended_lane": recommended_lane,
        "readiness_state": readiness_state,
        "confidence": confidence,
        "risks": qualification_record.get("risks", []),
    }
    return attach_handoff_package_paths(handoff, pipeline_dir=pipeline_dir, metadata=metadata)


def _build_pipeline_summary(
    *,
    bucket: str,
    descriptor_uri: str,
    qa_report_uri: str,
    object_index_uri: Optional[str],
    pipeline_prefix: str,
    pipeline_dir: Path,
    task_targets_payload: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    qualification_record: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "lane": "qualification",
        "generated_at": utc_now_iso(),
        "pipeline_prefix": pipeline_prefix,
        "source_uris": {
            "descriptor_uri": descriptor_uri,
            "qa_report_uri": qa_report_uri,
            "object_index_uri": object_index_uri,
            "task_targets_uri": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
            "site_intake_uri": f"gs://{bucket}/{pipeline_prefix}/site_intake.json",
            "capture_package_manifest_uri": f"gs://{bucket}/{pipeline_prefix}/capture_package_manifest.json",
            "capture_qa_scorecard_uri": f"gs://{bucket}/{pipeline_prefix}/capture_qa_scorecard.json",
            "task_scope_record_uri": f"gs://{bucket}/{pipeline_prefix}/task_scope_record.json",
            "qualification_record_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "qualification_brief_uri": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
            "opportunity_handoff_uri": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
            "runtime_preflight_report_uri": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
        },
        "source_files": {
            "runtime_preflight_report": _local_file_pointer(pipeline_dir / "runtime_preflight_report.json"),
            "task_targets": _local_file_pointer(pipeline_dir / "task_targets.json"),
            "site_intake": _local_file_pointer(pipeline_dir / "site_intake.json"),
            "capture_package_manifest": _local_file_pointer(pipeline_dir / "capture_package_manifest.json"),
            "capture_qa_scorecard": _local_file_pointer(pipeline_dir / "capture_qa_scorecard.json"),
            "task_scope_record": _local_file_pointer(pipeline_dir / "task_scope_record.json"),
            "qualification_record": _local_file_pointer(pipeline_dir / "qualification_record.json"),
            "qualification_brief": _local_file_pointer(pipeline_dir / "qualification_brief.json"),
            "opportunity_handoff": _local_file_pointer(pipeline_dir / "opportunity_handoff.json"),
        },
        "metrics": {
            "completeness_status": str(scorecard.get("completeness_status") or "need_more_evidence"),
            "readiness_state": str(qualification_record.get("readiness_state") or "not_ready_yet"),
            "confidence": float(qualification_record.get("confidence") or 0.0),
            "task_target_count": len(task_targets_payload.get("target_object_ids", []))
            if isinstance(task_targets_payload.get("target_object_ids"), list)
            else 0,
            "risk_count": len(qualification_record.get("risks", []))
            if isinstance(qualification_record.get("risks"), list)
            else 0,
            "object_index_count": len(task_targets_payload.get("object_index_entries", []))
            if isinstance(task_targets_payload.get("object_index_entries"), list)
            else 0,
        },
    }


def _write_failure(
    *,
    pipeline_dir: Path,
    descriptor_uri: str,
    stage: str,
    error: Exception,
    gates: List[QualificationGate],
) -> None:
    payload = {
        "schema_version": "v1",
        "lane": "qualification",
        "status": "failed",
        "stage": stage,
        "descriptor_uri": descriptor_uri,
        "error": str(error),
        "failed_at": utc_now_iso(),
        "gates": [gate.to_dict() for gate in gates],
    }
    write_json(pipeline_dir / ".swap_pipeline_failed.json", payload)


def run_qualification_pipeline(
    *,
    descriptor_gcs_uri: str,
    config: Any,
) -> Dict[str, Any]:
    stage = "intake"
    gates: List[QualificationGate] = []
    scene_id = "unknown_scene"
    capture_id = "unknown_capture"
    pipeline_prefix = "_pipeline_failures"
    pipeline_dir = config.gcs_root / pipeline_prefix
    ensure_dir(pipeline_dir)

    try:
        parsed_uri = parse_gs_uri(descriptor_gcs_uri)
        bucket = parsed_uri.bucket
        descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, config.gcs_root)
        storage_root = infer_storage_root_from_scene_path(descriptor_path)
        descriptor = CaptureDescriptor.from_file(descriptor_path)
        scene_id = descriptor.scene_id
        capture_id = descriptor.capture_id
        pipeline_prefix = to_pipeline_prefix(scene_id, capture_id)
        pipeline_dir = storage_root / pipeline_prefix
        ensure_dir(pipeline_dir)

        qa_report_uri = descriptor.qa_report_uri or _default_qa_report_uri(descriptor_gcs_uri)
        qa_report_path = resolve_gs_uri_to_path(qa_report_uri, storage_root)
        qa_report = _try_read_json(qa_report_path) or {"schema_version": "v1", "status": "missing"}

        manifest_uri = None
        manifest: Optional[IOSManifest] = None
        manifest_path: Optional[Path] = None
        object_index_uri: Optional[str] = None
        object_index_path: Optional[Path] = None
        object_index_entries: List[Mapping[str, Any]] = []

        try:
            manifest = load_raw_manifest(descriptor.raw_prefix_uri, gcs_root=storage_root)
            manifest_uri = f"{descriptor.raw_prefix_uri.rstrip('/')}/manifest.json"
            manifest_path = resolve_gs_uri_to_path(manifest_uri, storage_root)
            object_index_uri = resolve_object_index_uri(descriptor.raw_prefix_uri, manifest)
            if object_index_uri:
                object_index_path = resolve_gs_uri_to_path(object_index_uri, storage_root)
                object_index_entries = load_object_index(object_index_uri, gcs_root=storage_root)
        except Exception:
            manifest = None
            object_index_uri = None
            object_index_entries = []

        stage = "runtime_preflight"
        if getattr(config, "runtime_preflight_enabled", True):
            runtime_preflight_report = _build_runtime_preflight_report(
                descriptor_path=descriptor_path,
                qa_report_path=qa_report_path,
                manifest_path=manifest_path,
                object_index_path=object_index_path,
                gcs_root=storage_root,
            )
        else:
            runtime_preflight_report = {
                "schema_version": "v1",
                "lane": "qualification",
                "status": "skipped",
                "generated_at": utc_now_iso(),
                "detail": "runtime preflight disabled by configuration",
            }
        write_json(pipeline_dir / "runtime_preflight_report.json", runtime_preflight_report)
        gates.append(
            QualificationGate(
                "runtime_preflight_gate",
                runtime_preflight_report.get("status") in {"passed", "skipped", "degraded"},
                f"status={runtime_preflight_report.get('status')}",
            )
        )

        stage = "intake"
        site_intake = {
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "generated_at": utc_now_iso(),
            "descriptor": descriptor.to_dict(),
            "descriptor_uri": descriptor_gcs_uri,
            "qa_report_uri": qa_report_uri,
            "site_identity": {
                "scene_id": descriptor.scene_id,
                "capture_id": descriptor.capture_id,
                "environment_type_hint": descriptor.environment_type_hint or "unknown",
            },
            "task_context": {
                "buyer_type": descriptor.metadata.get("buyer_type") if isinstance(descriptor.metadata, Mapping) else None,
                "task_statement": descriptor.metadata.get("task_statement") if isinstance(descriptor.metadata, Mapping) else None,
                "workflow_context": descriptor.metadata.get("workflow_context") if isinstance(descriptor.metadata, Mapping) else None,
                "operating_hours": descriptor.metadata.get("operating_hours") if isinstance(descriptor.metadata, Mapping) else None,
            },
            "constraints": {
                "privacy_restrictions": descriptor.metadata.get("privacy_restrictions") if isinstance(descriptor.metadata, Mapping) else None,
                "security_restrictions": descriptor.metadata.get("security_restrictions") if isinstance(descriptor.metadata, Mapping) else None,
                "known_blockers": descriptor.metadata.get("known_blockers") if isinstance(descriptor.metadata, Mapping) else [],
                "safety_concerns": descriptor.metadata.get("safety_concerns") if isinstance(descriptor.metadata, Mapping) else [],
            },
        }
        write_json(pipeline_dir / "site_intake.json", site_intake)
        gates.append(QualificationGate("intake_gate", True, "descriptor parsed and intake record written"))

        stage = "capture_package_manifest"
        capture_package_manifest = _build_capture_package_manifest(
            descriptor=descriptor,
            descriptor_uri=descriptor_gcs_uri,
            qa_report_uri=qa_report_uri,
            manifest_uri=manifest_uri,
            object_index_uri=object_index_uri,
            storage_root=storage_root,
            object_index_entries=object_index_entries,
        )
        write_json(pipeline_dir / "capture_package_manifest.json", capture_package_manifest)

        stage = "task_targets"
        if manifest is not None and object_index_uri:
            task_targets_payload = infer_task_targets(
                descriptor=descriptor,
                manifest=manifest,
                object_index_entries=object_index_entries,
                object_index_uri=object_index_uri,
                storage_root=storage_root,
                max_targets=max(1, int(getattr(config, "task_target_max_objects", 24) or 24)),
            )
        else:
            task_targets_payload = _disabled_task_targets(
                descriptor.scene_id,
                descriptor.capture_id,
                "missing_manifest_or_object_index",
            )
        task_targets_with_index = dict(task_targets_payload)
        task_targets_with_index["object_index_entries"] = [dict(item) for item in object_index_entries]
        write_task_targets(pipeline_dir / "task_targets.json", task_targets_with_index)

        stage = "completeness"
        scorecard = _build_completeness_scorecard(
            descriptor=descriptor,
            qa_report=qa_report,
            manifest=manifest,
            object_index_uri=object_index_uri,
            object_index_entries=object_index_entries,
        )
        write_json(pipeline_dir / "capture_qa_scorecard.json", scorecard)
        gates.append(
            QualificationGate(
                "completeness_gate",
                str(scorecard.get("completeness_status")) == "sufficient",
                f"completeness_status={scorecard.get('completeness_status')}",
            )
        )

        stage = "scoping"
        scope_record = _build_task_scope_record(
            descriptor=descriptor,
            task_targets_payload=task_targets_payload,
            completeness_status=str(scorecard.get("completeness_status") or "need_more_evidence"),
        )
        write_json(pipeline_dir / "task_scope_record.json", scope_record)
        gates.append(
            QualificationGate(
                "scoping_gate",
                str(scope_record.get("scope_status")) == "scoped",
                f"scope_status={scope_record.get('scope_status')}",
            )
        )

        stage = "qualification"
        qualification_record = _build_qualification_record(
            descriptor=descriptor,
            scorecard=scorecard,
            scope_record=scope_record,
            object_index_entries=object_index_entries,
        )
        qualification_brief = _build_qualification_brief(
            descriptor=descriptor,
            scorecard=scorecard,
            scope_record=scope_record,
            qualification_record=qualification_record,
        )
        opportunity_handoff = _build_opportunity_handoff(
            descriptor=descriptor,
            scorecard=scorecard,
            scope_record=scope_record,
            qualification_record=qualification_record,
            brief=qualification_brief,
            config=config,
            pipeline_dir=pipeline_dir,
        )
        write_json(pipeline_dir / "qualification_record.json", qualification_record)
        write_json(pipeline_dir / "qualification_brief.json", qualification_brief)
        write_json(pipeline_dir / "opportunity_handoff.json", opportunity_handoff)
        gates.append(
            QualificationGate(
                "qualification_gate",
                True,
                f"readiness_state={qualification_record.get('readiness_state')}",
            )
        )

        stage = "completion"
        pipeline_summary = _build_pipeline_summary(
            bucket=bucket,
            descriptor_uri=descriptor_gcs_uri,
            qa_report_uri=qa_report_uri,
            object_index_uri=object_index_uri,
            pipeline_prefix=pipeline_prefix,
            pipeline_dir=pipeline_dir,
            task_targets_payload=task_targets_with_index,
            scorecard=scorecard,
            qualification_record=qualification_record,
        )
        write_json(pipeline_dir / "pipeline_summary.json", pipeline_summary)

        quality_report = {
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "status": "passed",
            "generated_at": utc_now_iso(),
            "readiness_state": qualification_record.get("readiness_state"),
            "completeness_status": scorecard.get("completeness_status"),
            "gates": [gate.to_dict() for gate in gates],
            "artifacts": {
                "descriptor_uri": descriptor_gcs_uri,
                "qa_report_uri": qa_report_uri,
                "task_targets": f"gs://{bucket}/{pipeline_prefix}/task_targets.json",
                "runtime_preflight_report": f"gs://{bucket}/{pipeline_prefix}/runtime_preflight_report.json",
                "site_intake": f"gs://{bucket}/{pipeline_prefix}/site_intake.json",
                "capture_package_manifest": f"gs://{bucket}/{pipeline_prefix}/capture_package_manifest.json",
                "capture_qa_scorecard": f"gs://{bucket}/{pipeline_prefix}/capture_qa_scorecard.json",
                "task_scope_record": f"gs://{bucket}/{pipeline_prefix}/task_scope_record.json",
                "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
                "qualification_brief": f"gs://{bucket}/{pipeline_prefix}/qualification_brief.json",
                "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
                "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
            },
        }
        write_json(pipeline_dir / "swap_quality_report.json", quality_report)

        completion_payload = {
            "schema_version": "v1",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "status": "completed",
            "completed_at": utc_now_iso(),
            "quality_report": f"gs://{bucket}/{pipeline_prefix}/swap_quality_report.json",
            "pipeline_summary": f"gs://{bucket}/{pipeline_prefix}/pipeline_summary.json",
            "qualification_record": f"gs://{bucket}/{pipeline_prefix}/qualification_record.json",
            "opportunity_handoff": f"gs://{bucket}/{pipeline_prefix}/opportunity_handoff.json",
        }
        write_json(pipeline_dir / ".swap_pipeline_complete", completion_payload)

        return {
            "status": "completed",
            "lane": "qualification",
            "scene_id": descriptor.scene_id,
            "capture_id": descriptor.capture_id,
            "pipeline_prefix": pipeline_prefix,
            "readiness_state": qualification_record.get("readiness_state"),
            "completeness_status": scorecard.get("completeness_status"),
            "match_ready": opportunity_handoff.get("match_ready"),
        }

    except Exception as exc:
        if isinstance(exc, StageError):
            stage = exc.stage
        ensure_dir(pipeline_dir)
        try:
            _write_failure(
                pipeline_dir=pipeline_dir,
                descriptor_uri=descriptor_gcs_uri,
                stage=stage,
                error=exc,
                gates=gates,
            )
            write_json(
                pipeline_dir / "swap_quality_report.json",
                {
                    "schema_version": "v1",
                    "lane": "qualification",
                    "scene_id": scene_id,
                    "capture_id": capture_id,
                    "status": "failed",
                    "generated_at": utc_now_iso(),
                    "failed_stage": stage,
                    "error": str(exc),
                    "gates": [gate.to_dict() for gate in gates],
                },
            )
        except Exception:
            pass
        raise PipelineError(str(exc)) from exc
