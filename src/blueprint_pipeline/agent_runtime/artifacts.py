"""Load qualification artifacts for agent review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..capture_bridge import CaptureDescriptor
from ..common import PipelineError, optional_read_json, read_json
from ..local_capture import LocalCaptureContext, resolve_local_capture_context


@dataclass(frozen=True)
class PipelineReviewArtifacts:
    context: LocalCaptureContext
    descriptor: CaptureDescriptor
    qa_report: Dict[str, Any]
    site_intake: Dict[str, Any]
    capture_package_manifest: Dict[str, Any]
    capture_qa_scorecard: Dict[str, Any]
    task_scope_record: Dict[str, Any]
    qualification_record: Dict[str, Any]
    qualification_brief: Dict[str, Any]
    scene_graph: Dict[str, Any]
    route_graph: Dict[str, Any]
    geometry_evidence: Dict[str, Any]
    supplemental_geometry: list[Dict[str, Any]]
    capability_checks: Dict[str, Any]
    blocker_register: Dict[str, Any]
    readiness_decision: Dict[str, Any]
    readiness_report: str
    opportunity_handoff: Dict[str, Any]
    human_actions_required: Dict[str, Any]
    task_hypothesis_report: Dict[str, Any]
    normalized_task_hypothesis: Dict[str, Any]
    @property
    def pipeline_dir(self) -> Path:
        return self.context.pipeline_root


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _read_required_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"Missing required pipeline artifact: {label} at {path}")
    return read_json(path)


def _supplemental_geometry_artifacts(pipeline_root: Path) -> list[Dict[str, Any]]:
    candidates = (
        ("geometry_summary", pipeline_root / "geometry" / "geometry_summary.json"),
        ("geometry_manifest", pipeline_root / "geometry" / "geometry_manifest.json"),
        ("advanced_geometry_bundle", pipeline_root / "advanced_geometry" / "advanced_geometry_bundle.json"),
        ("worldlabs_export_manifest", pipeline_root / "worldlabs_export_manifest.json"),
        (
            "worldlabs_materialized_assets",
            pipeline_root / "worldlabs_assets" / "materialized_assets_manifest.json",
        ),
    )
    artifacts: list[Dict[str, Any]] = []
    for label, path in candidates:
        if path.is_file():
            artifacts.append({"label": label, "path": str(path), "exists": True})
    return artifacts


def load_pipeline_review_artifacts(capture_root: str | Path) -> PipelineReviewArtifacts:
    context = resolve_local_capture_context(capture_root)
    descriptor = CaptureDescriptor.from_file(context.descriptor_path)
    qa_report_path = context.capture_root / "qa_report.json"
    return PipelineReviewArtifacts(
        context=context,
        descriptor=descriptor,
        qa_report=optional_read_json(qa_report_path) or {},
        site_intake=_read_required_json(context.pipeline_root / "site_intake.json", "site_intake"),
        capture_package_manifest=_read_required_json(
            context.pipeline_root / "capture_package_manifest.json",
            "capture_package_manifest",
        ),
        capture_qa_scorecard=_read_required_json(
            context.pipeline_root / "capture_qa_scorecard.json",
            "capture_qa_scorecard",
        ),
        task_scope_record=_read_required_json(
            context.pipeline_root / "task_scope_record.json",
            "task_scope_record",
        ),
        qualification_record=_read_required_json(
            context.pipeline_root / "qualification_record.json",
            "qualification_record",
        ),
        qualification_brief=_read_required_json(
            context.pipeline_root / "qualification_brief.json",
            "qualification_brief",
        ),
        scene_graph=_read_required_json(context.pipeline_root / "scene_graph.json", "scene_graph"),
        route_graph=_read_required_json(context.pipeline_root / "route_graph.json", "route_graph"),
        geometry_evidence=_read_required_json(
            context.pipeline_root / "geometry_evidence.json",
            "geometry_evidence",
        ),
        supplemental_geometry=_supplemental_geometry_artifacts(context.pipeline_root),
        capability_checks=_read_required_json(
            context.pipeline_root / "capability_checks.json",
            "capability_checks",
        ),
        blocker_register=_read_required_json(
            context.pipeline_root / "blocker_register.json",
            "blocker_register",
        ),
        readiness_decision=_read_required_json(
            context.pipeline_root / "readiness_decision.json",
            "readiness_decision",
        ),
        readiness_report=_read_text(context.pipeline_root / "readiness_report.md"),
        opportunity_handoff=_read_required_json(
            context.pipeline_root / "opportunity_handoff.json",
            "opportunity_handoff",
        ),
        human_actions_required=_read_required_json(
            context.pipeline_root / "human_actions_required.json",
            "human_actions_required",
        ),
        task_hypothesis_report=_read_required_json(
            context.pipeline_root / "task_hypothesis_report.json",
            "task_hypothesis_report",
        ),
        normalized_task_hypothesis=_read_required_json(
            context.pipeline_root / "normalized_task_hypothesis.json",
            "normalized_task_hypothesis",
        ),
    )
