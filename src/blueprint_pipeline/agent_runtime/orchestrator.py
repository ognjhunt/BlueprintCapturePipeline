"""Deterministic local agent review orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..common import ensure_dir, parse_bool, write_json, write_text
from .artifacts import PipelineReviewArtifacts, load_pipeline_review_artifacts
from .contracts import AgentReviewBundle, ReviewOutputFile, ReviewStepResult
from .openai_phase2 import OpenAIPhase2Config, build_openai_skill_runner
from .providers import ClaudeAgentProvider, OpenAIAgentProvider
from .skill_sync import sync_skill_pack


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
        if confidence < 0.7:
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
    if hidden_zone_bound > 0.35:
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
    entries = [dict(item) for item in artifacts.blocker_register.get("entries", []) if isinstance(item, Mapping)]
    existing_details = {str(item.get("detail") or "").strip() for item in entries}
    for gap in evidence_audit.get("evidence_gaps", []):
        if not isinstance(gap, Mapping):
            continue
        detail = str(gap.get("detail") or "").strip()
        if not detail or detail in existing_details:
            continue
        entries.append(
            {
                "severity": gap.get("severity", "medium"),
                "category": gap.get("category", "evidence"),
                "detail": detail,
                "source_artifacts": gap.get("source_artifacts", []),
            }
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
    if artifacts.descriptor.capture_modality == "glasses_video_only":
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
    evidence_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    readiness_state = str(artifacts.readiness_decision.get("status") or "not_ready_yet")
    steps = []
    for index, gap in enumerate(evidence_audit.get("evidence_gaps", []), start=1):
        if not isinstance(gap, Mapping):
            continue
        category = str(gap.get("category") or "capture_evidence")
        preferred_mode = "iphone_arkit_lidar"
        if category not in {"hidden_zone", "capture_evidence"}:
            preferred_mode = artifacts.descriptor.capture_modality
        steps.append(
            {
                "order": index,
                "category": category,
                "detail": gap.get("detail"),
                "preferred_capture_mode": preferred_mode,
                "justification": "Metric capture is preferred when blockers affect geometry or hidden zones.",
            }
        )
    required = readiness_state != "ready" or bool(steps)
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "required": required,
        "steps": steps,
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


def _oem_handoff_summary(artifacts: PipelineReviewArtifacts) -> Dict[str, Any]:
    target_robot_team = artifacts.opportunity_handoff.get("target_robot_team", {})
    return {
        "schema_version": "v1",
        "scene_id": artifacts.descriptor.scene_id,
        "capture_id": artifacts.descriptor.capture_id,
        "recommended_lane": artifacts.opportunity_handoff.get("recommended_lane"),
        "target_robot_team": target_robot_team,
        "summary": (
            "Human still needs to choose the downstream robot platform or integrator."
            if not target_robot_team
            else f"Prepared OEM-facing handoff summary for {target_robot_team.get('robot_platform')}."
        ),
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
        for step in recapture_plan.get("steps", [])[:8]:
            if not isinstance(step, Mapping):
                continue
            lines.append(
                f"- Step {step.get('order')}: {step.get('detail')} "
                f"(preferred mode: {step.get('preferred_capture_mode')})"
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
    humanoid_site_review = run_step(
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
    oem_handoff = run_step(
        "oem_handoff_writer",
        "oem_handoff_summary.json",
        lambda: _oem_handoff_summary(artifacts),
        {"opportunity_handoff": artifacts.opportunity_handoff},
    )
    recapture_plan = run_step(
        "recapture_planner",
        "recapture_plan.json",
        lambda: _recapture_plan(artifacts, evidence_audit),
        {
            "capture_qa_scorecard": artifacts.capture_qa_scorecard,
            "geometry_evidence": artifacts.geometry_evidence,
            "blocker_register": agent_blocker_register,
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
