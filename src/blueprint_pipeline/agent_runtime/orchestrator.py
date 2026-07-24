"""Deterministic local agent review orchestrator."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..common import MAXIMUM_HIDDEN_ZONE_BOUND, ensure_dir, parse_bool, write_json, write_text
from .artifacts import PipelineReviewArtifacts, load_pipeline_review_artifacts
from .contracts import AgentReviewBundle, ReviewOutputFile, ReviewStepResult
from .openai_phase2 import OpenAIPhase2Config, build_openai_skill_runner
from .providers import (
    ClaudeAgentProvider,
    LocalDeterministicAgentProvider,
    OpenAIAgentProvider,
)
from .skill_sync import sync_skill_pack


# Route edges below this confidence are surfaced as low-confidence in the
# evidence audit, and the recapture acceptance prose cites the same bar so the
# two stay in lock-step.
MINIMUM_ROUTE_EDGE_CONFIDENCE = 0.7

_DEFAULT_HUMAN_ACTIONS = [
    "Confirm workflow boundary and success criteria.",
    "Confirm the in-scope zone and accountable site owner.",
    "Review non-routine modes and safety/EHS constraints.",
    "Confirm hidden or restricted areas were adequately captured.",
    "Approve recapture when evidence is incomplete.",
    "Make the final readiness signoff.",
    "Choose the OEM, integrator, or target robot platform for downstream evaluation.",
]

_LLM_OVERRIDE_SKILLS = {
    "humanoid_site_readiness_reviewer",
    "humanoid_workcell_risk_reviewer",
    "humanoid_route_access_reviewer",
    "oem_handoff_writer",
    "recapture_planner",
    "readiness_report_writer",
}

_RECAPTURE_KEYWORDS = (
    "hidden zone",
    "hidden-zone",
    "clearance",
    "width",
    "route",
    "reach",
    "occlusion",
    "coverage",
    "measurement",
    "metric",
    "geometry",
    "missing evidence",
    "missing view",
)

_ACCESS_KEYWORDS = (
    "restricted",
    "escort",
    "badge",
    "access",
    "permission",
)

_RECAPTURE_EQUIPMENT_BY_CATEGORY = {
    "floor": ["digital inclinometer", "phone camera"],
    "machine_interface": ["structured light scanner", "calipers", "tape measure"],
}


def _lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _zone_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "label", "zone", "title", "id"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _resolve_blocker_resolution_path(
    *,
    category: Any,
    detail: Any,
    severity: Any = "",
    resolution_path: Any = "",
) -> str:
    existing = _lower_text(resolution_path)
    if existing in {"recapture", "scope_change", "site_modification", "human_review", "platform_change", "oem_consultation", "not_resolvable"}:
        return existing

    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    severity_text = _lower_text(severity)

    if normalized_category in {"geometry", "capture_quality", "capture_coverage", "evidence", "hidden_zone"}:
        return "recapture"
    if normalized_category in {"scoping", "privacy_security", "non_routine"}:
        return "human_review"
    if normalized_category == "access":
        return "human_review"
    if normalized_category in {"integration", "machine_interface"}:
        return "oem_consultation"
    if normalized_category in {"task_fit"}:
        return "platform_change"
    if normalized_category in {"runtime"}:
        return "platform_change"
    if normalized_category in {"traffic_shared", "traffic_pedestrian"}:
        return "site_modification"
    if normalized_category in {"workflow_ambiguity"}:
        return "scope_change"
    if normalized_category == "automation_gap":
        if any(keyword in normalized_detail for keyword in _RECAPTURE_KEYWORDS):
            return "recapture"
        if any(keyword in normalized_detail for keyword in _ACCESS_KEYWORDS):
            return "human_review"
        return "recapture" if severity_text in {"high", "hard_blocker"} else "human_review"
    if any(keyword in normalized_detail for keyword in _RECAPTURE_KEYWORDS):
        return "recapture"
    if any(keyword in normalized_detail for keyword in _ACCESS_KEYWORDS):
        return "human_review"
    return "human_review"


def _recapture_priority(severity: Any) -> str:
    severity_text = _lower_text(severity)
    if severity_text == "hard_blocker":
        return "P0"
    if severity_text == "high":
        return "P1"
    if severity_text == "medium":
        return "P2"
    if severity_text == "low":
        return "P3"
    return "P4"


def _recapture_priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(priority, 5)


def _recapture_equipment(category: str, detail: str) -> List[str]:
    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    if "floor" in normalized_category or any(keyword in normalized_detail for keyword in ("floor", "grade", "slope")):
        return ["digital inclinometer", "phone camera"]
    if "machine_interface" in normalized_category or any(
        keyword in normalized_detail for keyword in ("machine interface", "button", "handle", "fixture", "door force")
    ):
        return ["structured light scanner", "calipers", "tape measure"]
    if normalized_category in {"hidden_zone", "capture_coverage", "geometry", "evidence", "capture_quality"} or any(
        keyword in normalized_detail
        for keyword in ("width", "clearance", "route", "reach", "hidden zone", "hidden-zone", "occlusion", "coverage", "geometry")
    ):
        return ["LiDAR scanner", "laser range finder", "phone camera"]
    if "capture_quality" in normalized_category:
        return ["tripod", "phone camera"]
    return ["phone camera"]


def _preferred_capture_mode(category: str, detail: str, fallback: str) -> str:
    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    if normalized_category in {"geometry", "capture_coverage", "capture_quality", "evidence", "hidden_zone"}:
        return "iphone_arkit_lidar"
    if any(keyword in normalized_detail for keyword in ("width", "clearance", "route", "reach", "hidden zone", "hidden-zone", "occlusion", "coverage", "metric")):
        return "iphone_arkit_lidar"
    return fallback or "iphone_arkit_lidar"


def _recapture_access(detail: str, category: str) -> str:
    normalized_detail = _lower_text(detail)
    normalized_category = _lower_text(category)
    if normalized_category in {"access", "privacy_security"}:
        return "restricted"
    if any(keyword in normalized_detail for keyword in ("restricted", "escort", "badge", "permission")):
        return "restricted; escort required"
    return "open"


def _recapture_timing(category: str, detail: str) -> str:
    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    if normalized_category in {"traffic_shared", "traffic_pedestrian"} or any(keyword in normalized_detail for keyword in ("traffic", "shift", "operations")):
        return "during operations"
    if _recapture_access(detail, category) != "open":
        return "scheduled access"
    return "off-hours"


def _recapture_instructions(category: str, detail: str, access: str, preferred_mode: str) -> List[str]:
    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    steps = [
        f"Re-capture the affected zone with `{preferred_mode}` and preserve the original capture provenance.",
        "Include a scale reference or metric measurement so the blocker can be resolved quantitatively.",
    ]
    if any(keyword in normalized_detail for keyword in ("hidden zone", "coverage", "occlusion")):
        steps.append("Cover the previously hidden or occluded area in a full pass, including the adjacent transition.")
    if any(keyword in normalized_detail for keyword in ("width", "clearance", "route", "reach")):
        steps.append("Measure the narrowest point and record the exact clearance from the same standing position used for qualification.")
    if "floor" in normalized_category or any(keyword in normalized_detail for keyword in ("floor", "grade", "slope")):
        steps.append("Capture floor condition and slope from multiple points along the affected path.")
    if access != "open":
        steps.append("Confirm the access requirement before capture and document the escort or authorization used.")
    return steps


def _recapture_acceptance_criteria(category: str, detail: str) -> List[str]:
    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    criteria = []
    if any(keyword in normalized_detail for keyword in ("width", "clearance", "route", "reach")):
        criteria.extend(
            [
                "Metric measurement is recorded at the narrowest point.",
                f"Confidence is at least {MINIMUM_ROUTE_EDGE_CONFIDENCE} or an equivalent metric-grade source is cited.",
            ]
        )
    elif any(keyword in normalized_detail for keyword in ("hidden zone", "coverage", "occlusion")):
        criteria.extend(
            [
                "The previously uncovered area is visible in the new capture pass.",
                "The blocker detail is no longer uncited by the evidence bundle.",
            ]
        )
    elif "floor" in normalized_category or any(keyword in normalized_detail for keyword in ("floor", "grade", "slope")):
        criteria.extend(
            [
                "Slope or grade is measured with a calibrated tool.",
                "A reference image or scan includes scale context.",
            ]
        )
    else:
        criteria.extend(
            [
                "The blocker detail is resolved with cited capture evidence.",
                "The evidence bundle links back to the affected zone.",
            ]
        )
    return criteria


def _recapture_effort_minutes(category: str, detail: str, access: str) -> int:
    normalized_category = _lower_text(category)
    normalized_detail = _lower_text(detail)
    if access != "open":
        return 30
    if "machine_interface" in normalized_category or any(keyword in normalized_detail for keyword in ("machine interface", "button", "fixture", "door force")):
        return 30
    if "floor" in normalized_category or any(keyword in normalized_detail for keyword in ("floor", "grade", "slope")):
        return 25
    if any(keyword in normalized_detail for keyword in ("width", "clearance", "route", "reach", "hidden zone", "hidden-zone", "occlusion", "coverage")):
        return 20
    return 15


def _resolution_detail(resolution_path: str, category: str, detail: str) -> str:
    normalized_detail = _lower_text(detail)
    normalized_category = _lower_text(category)
    if resolution_path == "recapture":
        if any(keyword in normalized_detail for keyword in ("width", "clearance", "route", "reach")):
            return "Re-capture the affected zone with metric evidence and a scale reference."
        if any(keyword in normalized_detail for keyword in ("hidden zone", "coverage", "occlusion")) or "capture_coverage" in normalized_category:
            return "Re-capture the uncovered area with complete coverage and preserve provenance."
        if "floor" in normalized_category or any(keyword in normalized_detail for keyword in ("floor", "grade", "slope")):
            return "Re-capture the floor condition with calibrated metric evidence."
        return "Re-capture the missing evidence for this blocker."
    if resolution_path == "human_review":
        return "Capture alone cannot close this blocker. Escalate to human review."
    if resolution_path == "scope_change":
        return "Adjust the qualification scope before another capture pass will help."
    if resolution_path == "site_modification":
        return "Requires a site change before another capture pass will help."
    if resolution_path == "oem_consultation":
        return "Confirm OEM or integrator constraints before recapture."
    if resolution_path == "platform_change":
        return "The current platform or configuration appears insufficient; reassess the platform."
    if resolution_path == "not_resolvable":
        return "No capture-only resolution path was identified."
    return "Capture-only follow-up is not yet determined."


def _enrich_blocker_entry(
    blocker: Mapping[str, Any],
    *,
    default_zone: str,
    source_artifact: str,
) -> Dict[str, Any]:
    category = str(blocker.get("category") or "general").strip()
    detail = str(blocker.get("detail") or "").strip()
    severity = str(blocker.get("severity") or "medium").strip()
    resolution_path = _resolve_blocker_resolution_path(
        category=category,
        detail=detail,
        severity=severity,
        resolution_path=blocker.get("resolution_path"),
    )
    zone = _zone_text(blocker.get("zone") or default_zone)
    source_artifacts = blocker.get("source_artifacts")
    normalized_source_artifacts = (
        _normalize_strings(source_artifacts)
        if isinstance(source_artifacts, list)
        else []
    )
    if source_artifact not in normalized_source_artifacts:
        normalized_source_artifacts.append(source_artifact)
    enriched = dict(blocker)
    enriched["id"] = str(blocker.get("id") or blocker.get("blocker_id") or blocker.get("category") or "blocker").strip()
    enriched["category"] = category
    enriched["severity"] = severity
    enriched["detail"] = detail
    enriched["zone"] = zone
    enriched["resolution_path"] = resolution_path
    enriched["resolution_detail"] = str(blocker.get("resolution_detail") or "").strip() or _resolution_detail(
        resolution_path,
        category,
        detail,
    )
    enriched["source_artifacts"] = normalized_source_artifacts
    return enriched


def _build_recapture_step(
    blocker: Mapping[str, Any],
    *,
    fallback_capture_modality: str,
    default_zone: str,
) -> Optional[Dict[str, Any]]:
    resolution_path = _lower_text(blocker.get("resolution_path"))
    if resolution_path != "recapture":
        return None

    category = str(blocker.get("category") or "capture_evidence").strip()
    detail = str(blocker.get("detail") or "").strip()
    severity = str(blocker.get("severity") or "medium").strip()
    zone = _zone_text(blocker.get("zone") or default_zone)
    preferred_capture_mode = _preferred_capture_mode(category, detail, fallback_capture_modality)
    access = _recapture_access(detail, category)
    equipment = _recapture_equipment(category, detail)
    timing = _recapture_timing(category, detail)
    priority = _recapture_priority(severity)
    blocker_id = str(blocker.get("id") or blocker.get("blocker_id") or "").strip()
    source_artifacts = blocker.get("source_artifacts")
    normalized_source_artifacts = (
        _normalize_strings(source_artifacts)
        if isinstance(source_artifacts, list)
        else []
    )
    estimated_effort_minutes = _recapture_effort_minutes(category, detail, access)
    return {
        "blocker_id": blocker_id,
        "category": category,
        "detail": detail,
        "zone": zone,
        "priority": priority,
        "resolution_path": "recapture",
        "preferred_capture_mode": preferred_capture_mode,
        "equipment": equipment,
        "instructions": _recapture_instructions(category, detail, access, preferred_capture_mode),
        "acceptance_criteria": _recapture_acceptance_criteria(category, detail),
        "access": access,
        "timing": timing,
        "estimated_effort_minutes": estimated_effort_minutes,
        "justification": "Metric capture is preferred when blockers affect geometry, hidden zones, or coverage.",
        "source_artifacts": normalized_source_artifacts,
        "resolution_detail": str(blocker.get("resolution_detail") or "").strip() or _resolution_detail(
            "recapture",
            category,
            detail,
        ),
        "source_blockers": [blocker_id] if blocker_id else [],
    }


def _group_recapture_sessions(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    session_index: Dict[tuple[str, str, tuple[str, ...]], Dict[str, Any]] = {}
    for step in steps:
        equipment = tuple(str(item) for item in step.get("equipment", []) if str(item).strip())
        key = (str(step.get("zone") or ""), str(step.get("access") or ""), equipment)
        session = session_index.get(key)
        if session is None:
            session = {
                "order": len(session_index) + 1,
                "zone": step.get("zone"),
                "access": step.get("access"),
                "timing": step.get("timing"),
                "equipment": list(equipment),
                "items": [],
                "estimated_effort_minutes": 0,
            }
            session_index[key] = session
            sessions.append(session)
        session["items"].append(
            {
                "order": step.get("order"),
                "blocker_id": step.get("blocker_id"),
                "detail": step.get("detail"),
                "priority": step.get("priority"),
            }
        )
        session["estimated_effort_minutes"] += int(step.get("estimated_effort_minutes") or 0)
    return sessions


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_curated_standards() -> List[Dict[str, Any]]:
    corpus_path = (
        _repo_root()
        / "skillpacks"
        / "industrial_readiness"
        / "references"
        / "curated_standards.json"
    )
    if not corpus_path.is_file():
        return []
    import json

    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, Mapping) else []
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def _normalize_strings(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _existing_human_actions(artifacts: PipelineReviewArtifacts) -> List[Dict[str, Any]]:
    entries = artifacts.human_actions_required.get("actions", [])
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def _normalized_intake(artifacts: PipelineReviewArtifacts) -> Dict[str, Any]:
    task_context = artifacts.site_intake.get("task_context", {})
    constraints = artifacts.site_intake.get("constraints", {})
    missing = []
    for key, value in (
        ("workflow", task_context.get("task_statement") or task_context.get("workflow_decomposition")),
        ("task_zone", task_context.get("task_zone")),
        ("success_criteria", task_context.get("success_criteria")),
    ):
        if not value:
            missing.append(key)
    status = "normalized" if not missing else "needs_human_completion"
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "status": status,
        "capture_modality": artifacts.descriptor.capture_modality,
        "workflow": task_context.get("task_statement") or task_context.get("workflow_decomposition"),
        "zone": task_context.get("task_zone"),
        "owner": task_context.get("owner"),
        "success_criteria": task_context.get("success_criteria"),
        "adjacent_systems": task_context.get("adjacent_systems"),
        "non_routine_modes": task_context.get("non_routine_modes"),
        "people_traffic_notes": task_context.get("people_traffic_notes"),
        "privacy_restrictions": constraints.get("privacy_restrictions"),
        "security_restrictions": constraints.get("security_restrictions"),
        "known_blockers": constraints.get("known_blockers"),
        "missing_required_fields": missing,
    }


def _evidence_audit(artifacts: PipelineReviewArtifacts) -> Dict[str, Any]:
    scorecard = artifacts.capture_qa_scorecard
    geometry = artifacts.geometry_evidence
    route_edges = artifacts.route_graph.get("edges", [])
    low_confidence_edges = []
    for edge in route_edges:
        if not isinstance(edge, Mapping):
            continue
        confidence = float(edge.get("confidence") or 0.0)
        if confidence < MINIMUM_ROUTE_EDGE_CONFIDENCE:
            low_confidence_edges.append(
                {
                    "edge_id": edge.get("id") or edge.get("to") or edge.get("target"),
                    "confidence": round(confidence, 4),
                    "detail": "Route edge remains low confidence for downstream autonomy planning.",
                }
            )
    evidence_gaps = []
    for detail in scorecard.get("follow_ups", []):
        text = str(detail).strip()
        if text:
            evidence_gaps.append(
                {
                    "category": "capture_evidence",
                    "severity": "high" if "missing" in text.lower() else "medium",
                    "detail": text,
                    "source_artifacts": ["capture_qa_scorecard.json"],
                }
            )
    hidden_zone_bound = float(geometry.get("hidden_zone_bound") or 1.0)
    if hidden_zone_bound > MAXIMUM_HIDDEN_ZONE_BOUND:
        evidence_gaps.append(
            {
                "category": "hidden_zone",
                "severity": "high",
                "detail": (
                    f"Hidden-zone bound {round(hidden_zone_bound, 4)} exceeds the readiness envelope."
                ),
                "source_artifacts": ["geometry_evidence.json"],
            }
        )
    if artifacts.supplemental_geometry:
        supplemental = [item["path"] for item in artifacts.supplemental_geometry if item.get("path")]
    else:
        supplemental = []
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "status": "grounded" if not evidence_gaps else "needs_more_evidence",
        "evidence_gaps": evidence_gaps,
        "low_confidence_route_edges": low_confidence_edges,
        "hidden_zone_bound": hidden_zone_bound,
        "metric_ready": bool(geometry.get("metric_ready")),
        "supplemental_geometry": supplemental,
    }


def _agent_blocker_register(
    artifacts: PipelineReviewArtifacts,
    evidence_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    default_zone = _zone_text(artifacts.task_scope_record.get("task_zone"))
    entries = [
        _enrich_blocker_entry(item, default_zone=default_zone, source_artifact="blocker_register.json")
        for item in artifacts.blocker_register.get("entries", [])
        if isinstance(item, Mapping)
    ]
    existing_details = {str(item.get("detail") or "").strip() for item in entries}
    for gap in evidence_audit.get("evidence_gaps", []):
        if not isinstance(gap, Mapping):
            continue
        detail = str(gap.get("detail") or "").strip()
        if not detail or detail in existing_details:
            continue
        gap_entry = {
            "id": str(gap.get("id") or gap.get("blocker_id") or f"evidence_gap_{len(entries) + 1}"),
            "severity": gap.get("severity", "medium"),
            "category": gap.get("category", "evidence"),
            "detail": detail,
            "source_artifacts": gap.get("source_artifacts", []),
        }
        entries.append(
            _enrich_blocker_entry(
                gap_entry,
                default_zone=default_zone,
                source_artifact="evidence_audit.json",
            )
        )
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "entries": entries,
    }


def _capability_envelope(
    artifacts: PipelineReviewArtifacts,
    evidence_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = [dict(item) for item in artifacts.capability_checks.get("checks", []) if isinstance(item, Mapping)]
    measurements = artifacts.qualification_record.get("measurements", {})
    statements = []
    for check in checks:
        detail = str(check.get("detail") or "").strip()
        status = str(check.get("status") or check.get("passed") or "").strip()
        if detail:
            statements.append(f"{check.get('name', 'check')}: {status or 'unknown'} - {detail}")
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "metric_ready": bool(artifacts.geometry_evidence.get("metric_ready")),
        "measurements": dict(measurements) if isinstance(measurements, Mapping) else {},
        "bounded_claims": statements,
        "evidence_gaps": evidence_audit.get("evidence_gaps", []),
    }


def _standards_notes(
    artifacts: PipelineReviewArtifacts,
    blocker_register: Mapping[str, Any],
) -> Dict[str, Any]:
    corpus = _load_curated_standards()
    categories = {
        str(item.get("category") or "").strip().lower()
        for item in blocker_register.get("entries", [])
        if isinstance(item, Mapping)
    }
    if artifacts.descriptor.capture_modality in {"glasses_video_only", "android_video_only"}:
        categories.add("capture_quality")
    selected = []
    for entry in corpus:
        entry_categories = {
            str(value).strip().lower()
            for value in entry.get("categories", [])
            if str(value).strip()
        }
        if categories.intersection(entry_categories):
            selected.append(entry)
    if not selected:
        selected = corpus[:3]
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "source": "curated_local_corpus",
        "entries": selected,
    }


def _recapture_plan(
    artifacts: PipelineReviewArtifacts,
    blocker_register: Mapping[str, Any],
    *,
    route_access_review: Optional[Mapping[str, Any]] = None,
    workcell_risk_review: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    default_zone = _zone_text(artifacts.task_scope_record.get("task_zone"))
    blockers = [
        _enrich_blocker_entry(item, default_zone=default_zone, source_artifact="blocker_register.json")
        for item in blocker_register.get("entries", [])
        if isinstance(item, Mapping)
    ]
    steps: List[Dict[str, Any]] = []
    for source_order, blocker in enumerate(blockers, start=1):
        step = _build_recapture_step(
            blocker,
            fallback_capture_modality=artifacts.descriptor.capture_modality,
            default_zone=default_zone,
        )
        if step is None:
            continue
        step["source_order"] = source_order
        steps.append(step)

    steps.sort(key=lambda item: (_recapture_priority_rank(str(item.get("priority") or "P4")), int(item.get("source_order") or 0)))
    for index, step in enumerate(steps, start=1):
        step["order"] = index

    capture_sessions = _group_recapture_sessions(steps)
    priority_distribution = dict(Counter(str(step.get("priority") or "P4") for step in steps))
    equipment_list: List[str] = []
    access_requirements_summary: List[str] = []
    estimated_total_effort_minutes = 0
    expected_impact: List[Dict[str, Any]] = []
    for step in steps:
        estimated_total_effort_minutes += int(step.get("estimated_effort_minutes") or 0)
        expected_impact.append(
            {
                "blocker_id": step.get("blocker_id"),
                "detail": step.get("detail"),
                "resolution_path": step.get("resolution_path"),
            }
        )
        for item in step.get("equipment", []):
            text = str(item).strip()
            if text and text not in equipment_list:
                equipment_list.append(text)
        access_text = str(step.get("access") or "").strip()
        if access_text and access_text not in access_requirements_summary and access_text != "open":
            access_requirements_summary.append(access_text)

    access_pending = bool(steps) and all(str(step.get("access") or "").strip() not in {"", "open"} for step in steps)
    required = bool(steps)
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "required": required,
        "access_pending": access_pending,
        "steps": steps,
        "capture_sessions": capture_sessions,
        "priority_distribution": priority_distribution,
        "equipment_list": equipment_list,
        "access_requirements_summary": access_requirements_summary,
        "estimated_total_recapture_effort_minutes": estimated_total_effort_minutes,
        "expected_impact": expected_impact,
        "route_access_review": dict(route_access_review) if isinstance(route_access_review, Mapping) else {},
        "workcell_risk_review": dict(workcell_risk_review) if isinstance(workcell_risk_review, Mapping) else {},
    }


def _humanoid_site_review(
    artifacts: PipelineReviewArtifacts,
    standards_notes: Mapping[str, Any],
) -> Dict[str, Any]:
    blocker_count = len(artifacts.blocker_register.get("entries", []))
    standards_count = len(standards_notes.get("entries", []))
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "summary": (
            f"Site review remains {artifacts.readiness_decision.get('status')} with "
            f"{blocker_count} blocker entries and {standards_count} curated guidance notes."
        ),
        "focus_areas": [
            "shared human-robot operating space",
            "route clearances and choke points",
            "hidden conditions near workcells and task zones",
        ],
    }


def _humanoid_workcell_risk_review(artifacts: PipelineReviewArtifacts) -> Dict[str, Any]:
    risks = [
        str(item.get("detail") or "").strip()
        for item in artifacts.qualification_record.get("risks", [])
        if isinstance(item, Mapping)
    ]
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "summary": "Workcell risk review compiled from qualification risks and hidden-zone evidence.",
        "risks": risks[:6],
    }


def _humanoid_route_access_review(artifacts: PipelineReviewArtifacts) -> Dict[str, Any]:
    measurements = artifacts.qualification_record.get("measurements", {})
    min_width = measurements.get("minimum_route_width_m") if isinstance(measurements, Mapping) else None
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "summary": "Route-access review constrained to measured route width and route graph confidence.",
        "minimum_route_width_m": min_width,
        "downstream_evaluation_eligibility": artifacts.opportunity_handoff.get(
            "downstream_evaluation_eligibility"
        ),
    }


# The oem_handoff_writer skill names these as required inputs. A handoff that
# silently omits them is not a partial handoff -- it is an OEM being asked to
# assess platform fit without the evidence the assessment depends on.
_OEM_HANDOFF_REQUIRED_INPUTS: tuple[tuple[str, str], ...] = (
    ("opportunity_handoff", "opportunity_handoff"),
    ("readiness_decision", "readiness_decision"),
    ("capability_envelope", "capability_checks"),
    ("blocker_register", "blocker_register"),
    ("human_actions_required", "human_actions_required"),
    ("normalized_intake", "site_intake"),
    ("geometry_evidence", "geometry_evidence"),
    ("task_scope_record", "task_scope_record"),
)


def _oem_handoff_summary(artifacts: PipelineReviewArtifacts) -> Dict[str, Any]:
    """Report the handoff's completeness rather than narrating around gaps.

    The previous summary degraded to a one-line string when the skill returned
    nothing, so a handoff missing most of its required evidence was
    indistinguishable from a complete one.
    """

    target_robot_team = artifacts.opportunity_handoff.get("target_robot_team", {})
    present: list[str] = []
    missing: list[str] = []
    for input_name, attribute in _OEM_HANDOFF_REQUIRED_INPUTS:
        value = getattr(artifacts, attribute, None)
        (present if value else missing).append(input_name)

    blockers = [f"oem_handoff_required_input_missing:{name}" for name in missing]
    if not target_robot_team:
        blockers.append("oem_handoff_target_robot_team_unselected")

    return {
        "schema_version": "v2",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "recommended_lane": artifacts.opportunity_handoff.get("recommended_lane"),
        "target_robot_team": target_robot_team,
        "status": "complete" if not blockers else "incomplete",
        "required_inputs": [name for name, _ in _OEM_HANDOFF_REQUIRED_INPUTS],
        "present_inputs": present,
        "missing_inputs": missing,
        "required_input_coverage": (
            round(len(present) / len(_OEM_HANDOFF_REQUIRED_INPUTS), 4)
            if _OEM_HANDOFF_REQUIRED_INPUTS
            else None
        ),
        "blockers": sorted(set(blockers)),
        "summary": (
            f"Prepared OEM-facing handoff for {target_robot_team.get('robot_platform')}."
            if not blockers
            else (
                f"Handoff incomplete: {len(missing)} of "
                f"{len(_OEM_HANDOFF_REQUIRED_INPUTS)} required inputs missing"
                + ("; downstream robot platform not yet selected." if not target_robot_team else ".")
            )
        ),
        "claim_boundary": {
            "handoff_completeness_is_not_site_readiness": True,
            "complete_means_inputs_present_not_platform_fit_established": True,
        },
    }


def _render_agent_memo(
    artifacts: PipelineReviewArtifacts,
    normalized_intake: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
    standards_notes: Mapping[str, Any],
    recapture_plan: Mapping[str, Any],
    human_actions: List[Mapping[str, Any]],
) -> str:
    lines = [
        f"# Agent Review Memo: {artifacts.descriptor.scene_id}/{artifacts.descriptor.capture_id}",
        "",
        f"- Readiness: `{artifacts.readiness_decision.get('status', 'not_ready_yet')}`",
        f"- Confidence: `{artifacts.readiness_decision.get('confidence', 0.0)}`",
        f"- Capture modality: `{artifacts.descriptor.capture_modality}`",
        f"- Evidence tier: `{artifacts.descriptor.evidence_tier}`",
        "",
        "## Intake Normalization",
        f"- Status: `{normalized_intake.get('status', 'needs_human_completion')}`",
    ]
    missing_fields = normalized_intake.get("missing_required_fields", [])
    if missing_fields:
        lines.append("- Missing required fields: " + ", ".join(str(item) for item in missing_fields))
    else:
        lines.append("- Required workflow, zone, and success criteria are present.")

    lines.extend(["", "## Evidence Audit"])
    gaps = evidence_audit.get("evidence_gaps", [])
    if not gaps:
        lines.append("- No new evidence gaps were added by the agent review.")
    else:
        for gap in gaps[:8]:
            if not isinstance(gap, Mapping):
                continue
            lines.append(f"- [{gap.get('severity', 'medium')}] {gap.get('detail', '')}")

    lines.extend(["", "## Standards Notes"])
    for entry in standards_notes.get("entries", [])[:5]:
        if not isinstance(entry, Mapping):
            continue
        citation = str(entry.get("citation") or entry.get("source") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        lines.append(f"- {entry.get('title', 'Guidance')}: {summary} ({citation})")

    lines.extend(["", "## Human Actions Required"])
    for action in human_actions:
        if not isinstance(action, Mapping):
            continue
        lines.append(f"- {action.get('action', '')}")

    lines.extend(["", "## Recapture"])
    if not parse_bool(recapture_plan.get("required"), default=False):
        lines.append("- No recapture plan was generated.")
    else:
        if parse_bool(recapture_plan.get("access_pending"), default=False):
            lines.append("- Access pending: restricted-zone authorization is still required.")
        for step in recapture_plan.get("steps", [])[:8]:
            if not isinstance(step, Mapping):
                continue
            lines.append(
                f"- Step {step.get('order')}: {step.get('detail')} "
                f"(zone: {step.get('zone') or 'unknown'}, preferred mode: {step.get('preferred_capture_mode')}, access: {step.get('access') or 'open'})"
            )

    return "\n".join(lines) + "\n"


def _provider_from_name(
    provider: str,
    *,
    repo_root: Path,
    skill_runner=None,
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
):
    normalized = provider.strip().lower()
    if normalized in {"local", "deterministic", "no_llm", "no-llm"}:
        return LocalDeterministicAgentProvider(repo_root=repo_root)
    if normalized == "claude":
        return ClaudeAgentProvider(skill_runner=skill_runner, repo_root=repo_root)
    if normalized == "openai":
        resolved_runner = skill_runner or build_openai_skill_runner(
            repo_root=repo_root,
            config=openai_phase2_config,
        )
        return OpenAIAgentProvider(skill_runner=resolved_runner, repo_root=repo_root)
    raise ValueError(f"Unsupported agent provider: {provider}")


def _write_step_output(
    pipeline_dir: Path,
    filename: str,
    payload: Mapping[str, Any],
) -> str:
    path = pipeline_dir / filename
    write_json(path, payload)
    return str(path)


def run_agent_review(
    *,
    capture_root: str | Path,
    provider_name: str,
    mode: str = "qualification",
    skill_runner=None,
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
) -> Dict[str, Any]:
    if mode != "qualification":
        raise ValueError(f"Unsupported agent review mode: {mode}")

    repo_root = _repo_root()
    sync_skill_pack(repo_root)
    artifacts = load_pipeline_review_artifacts(capture_root)
    ensure_dir(artifacts.pipeline_dir)
    provider = _provider_from_name(
        provider_name,
        repo_root=repo_root,
        skill_runner=skill_runner,
        openai_phase2_config=openai_phase2_config,
    )

    outputs: List[ReviewOutputFile] = []
    steps: List[ReviewStepResult] = []

    def run_step(skill_name: str, filename: str, local_builder, payload: Mapping[str, Any]) -> Dict[str, Any]:
        override = provider.invoke_skill(skill_name, payload) if skill_name in _LLM_OVERRIDE_SKILLS else None
        if override is None:
            result = local_builder()
            source = "local_deterministic"
        else:
            result = dict(override)
            source = "provider_override"
        output_path = _write_step_output(artifacts.pipeline_dir, filename, result)
        outputs.append(ReviewOutputFile(name=skill_name, path=output_path))
        steps.append(
            ReviewStepResult(
                skill_name=skill_name,
                output_path=output_path,
                source=source,
                provider_metadata=provider.skill_metadata(skill_name),
            )
        )
        return result

    normalized_intake = run_step(
        "intake_normalizer",
        "normalized_site_intake.json",
        lambda: _normalized_intake(artifacts),
        {"site_intake": artifacts.site_intake, "capture_package_manifest": artifacts.capture_package_manifest},
    )
    evidence_audit = run_step(
        "evidence_auditor",
        "evidence_audit.json",
        lambda: _evidence_audit(artifacts),
        {
            "capture_qa_scorecard": artifacts.capture_qa_scorecard,
            "geometry_evidence": artifacts.geometry_evidence,
            "scene_graph": artifacts.scene_graph,
            "route_graph": artifacts.route_graph,
        },
    )
    agent_blocker_register = run_step(
        "blocker_taxonomist",
        "agent_blocker_register.json",
        lambda: _agent_blocker_register(artifacts, evidence_audit),
        {
            "blocker_register": artifacts.blocker_register,
            "evidence_audit": evidence_audit,
            "site_intake": artifacts.site_intake,
        },
    )
    capability_envelope = run_step(
        "capability_envelope_writer",
        "capability_envelope.json",
        lambda: _capability_envelope(artifacts, evidence_audit),
        {
            "capability_checks": artifacts.capability_checks,
            "geometry_evidence": artifacts.geometry_evidence,
            "task_scope_record": artifacts.task_scope_record,
        },
    )
    standards_notes = run_step(
        "standards_retriever",
        "standards_notes.json",
        lambda: _standards_notes(artifacts, agent_blocker_register),
        {"site_intake": artifacts.site_intake, "blocker_register": agent_blocker_register},
    )
    run_step(
        "humanoid_site_readiness_reviewer",
        "humanoid_site_readiness_review.json",
        lambda: _humanoid_site_review(artifacts, standards_notes),
        {"readiness_decision": artifacts.readiness_decision, "standards_notes": standards_notes},
    )
    humanoid_workcell_review = run_step(
        "humanoid_workcell_risk_reviewer",
        "humanoid_workcell_risk_review.json",
        lambda: _humanoid_workcell_risk_review(artifacts),
        {"qualification_record": artifacts.qualification_record, "geometry_evidence": artifacts.geometry_evidence},
    )
    humanoid_route_review = run_step(
        "humanoid_route_access_reviewer",
        "humanoid_route_access_review.json",
        lambda: _humanoid_route_access_review(artifacts),
        {"route_graph": artifacts.route_graph, "qualification_record": artifacts.qualification_record},
    )
    run_step(
        "oem_handoff_writer",
        "oem_handoff_summary.json",
        lambda: _oem_handoff_summary(artifacts),
        {"opportunity_handoff": artifacts.opportunity_handoff},
    )
    recapture_plan = run_step(
        "recapture_planner",
        "recapture_plan.json",
        lambda: _recapture_plan(
            artifacts,
            agent_blocker_register,
            route_access_review=humanoid_route_review,
            workcell_risk_review=humanoid_workcell_review,
        ),
        {
            "normalized_intake": normalized_intake,
            "capture_qa_scorecard": artifacts.capture_qa_scorecard,
            "geometry_evidence": artifacts.geometry_evidence,
            "blocker_register": agent_blocker_register,
            "route_access_review": humanoid_route_review,
            "workcell_risk_review": humanoid_workcell_review,
        },
    )

    human_actions = _existing_human_actions(artifacts)
    if not human_actions:
        human_actions = [
            {"action": action, "required": True, "owner": "human_reviewer"}
            for action in _DEFAULT_HUMAN_ACTIONS
        ]

    memo_override = provider.invoke_skill(
        "readiness_report_writer",
        {
            "readiness_decision": artifacts.readiness_decision,
            "blocker_register": agent_blocker_register,
            "capability_envelope": capability_envelope,
            "standards_notes": standards_notes,
            "human_actions_required": human_actions,
            "recapture_plan": recapture_plan,
        },
    )
    memo_content = (
        str(memo_override.get("memo_markdown") or "")
        if isinstance(memo_override, Mapping)
        else ""
    )
    memo_source = "provider_override" if memo_content else "local_deterministic"
    if not memo_content:
        memo_content = _render_agent_memo(
            artifacts,
            normalized_intake=normalized_intake,
            evidence_audit=evidence_audit,
            standards_notes=standards_notes,
            recapture_plan=recapture_plan,
            human_actions=human_actions,
        )
    memo_path = artifacts.pipeline_dir / "agent_readiness_memo.md"
    write_text(memo_path, memo_content)
    outputs.append(ReviewOutputFile(name="readiness_report_writer", path=str(memo_path)))
    steps.append(
        ReviewStepResult(
            skill_name="readiness_report_writer",
            output_path=str(memo_path),
            source=memo_source,
            provider_metadata=provider.skill_metadata("readiness_report_writer"),
        )
    )

    bundle_path = artifacts.pipeline_dir / "agent_review_bundle.json"
    bundle = AgentReviewBundle(
        scene_id=artifacts.descriptor.scene_id,
        capture_id=artifacts.descriptor.capture_id,
        provider=provider.name,
        readiness_state=str(artifacts.readiness_decision.get("status") or "not_ready_yet"),
        final_memo_path=str(memo_path),
        final_bundle_path=str(bundle_path),
        human_actions_required_path=str(artifacts.pipeline_dir / "human_actions_required.json"),
        outputs=outputs,
        steps=steps,
        runtime={
            **provider.runtime_metadata(),
            "mode": mode,
            "supplemental_geometry": artifacts.supplemental_geometry,
        },
    )
    payload = bundle.to_dict()
    payload["artifacts"] = {
        "readiness_decision": str(artifacts.pipeline_dir / "readiness_decision.json"),
        "readiness_report": str(artifacts.pipeline_dir / "readiness_report.md"),
        "human_actions_required": str(artifacts.pipeline_dir / "human_actions_required.json"),
        "task_hypothesis_report": str(artifacts.pipeline_dir / "task_hypothesis_report.json"),
        "normalized_task_hypothesis": str(artifacts.pipeline_dir / "normalized_task_hypothesis.json"),
        "blocker_register": str(artifacts.pipeline_dir / "blocker_register.json"),
        "agent_blocker_register": str(artifacts.pipeline_dir / "agent_blocker_register.json"),
        "standards_notes": str(artifacts.pipeline_dir / "standards_notes.json"),
        "recapture_plan": str(artifacts.pipeline_dir / "recapture_plan.json"),
        "final_operator_summary": str(memo_path),
    }
    payload["specialized_reviews"] = {
        "humanoid_site_readiness_review": str(artifacts.pipeline_dir / "humanoid_site_readiness_review.json"),
        "humanoid_workcell_risk_review": str(artifacts.pipeline_dir / "humanoid_workcell_risk_review.json"),
        "humanoid_route_access_review": str(artifacts.pipeline_dir / "humanoid_route_access_review.json"),
        "oem_handoff_summary": str(artifacts.pipeline_dir / "oem_handoff_summary.json"),
    }
    write_json(bundle_path, payload)
    return payload
