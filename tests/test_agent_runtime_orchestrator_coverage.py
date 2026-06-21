from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_runtime import orchestrator as orch


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline = capture_root / "pipeline"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "capture_source": "iphone",
            "capture_tier": "tier2_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
            "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/frames.jsonl",
            "capture_modality": "iphone_arkit_lidar",
            "evidence_tier": "qualified_metric_capture",
            "quality": {"metric_ready": True},
        },
    )
    _write_json(
        pipeline / "site_intake.json",
        {
            "task_context": {
                "task_statement": "Move totes from receiving to the return rack",
                "task_zone": "Receiving aisle",
                "success_criteria": "Totes arrive undamaged",
                "owner": "ops",
                "adjacent_systems": ["dock door"],
            },
            "constraints": {
                "privacy_restrictions": "none",
                "security_restrictions": "badge escort",
                "known_blockers": ["narrow route"],
            },
        },
    )
    _write_json(pipeline / "capture_package_manifest.json", {"schema_version": "v1"})
    _write_json(
        pipeline / "capture_qa_scorecard.json",
        {"follow_ups": ["Missing view of route turn", "Confirm route clearance"]},
    )
    _write_json(pipeline / "task_scope_record.json", {"task_zone": {"name": "Receiving aisle"}})
    _write_json(
        pipeline / "qualification_record.json",
        {
            "measurements": {"minimum_route_width_m": 0.82},
            "risks": [{"detail": "Operator traffic crosses the route"}, "bad"],
        },
    )
    _write_json(pipeline / "qualification_brief.json", {"summary": "brief"})
    _write_json(pipeline / "scene_graph.json", {"nodes": []})
    _write_json(
        pipeline / "route_graph.json",
        {"edges": ["bad", {"id": "edge-a", "confidence": 0.4}, {"id": "edge-b", "confidence": 0.95}]},
    )
    _write_json(pipeline / "geometry_evidence.json", {"hidden_zone_bound": 0.5, "metric_ready": True})
    _write_json(pipeline / "geometry" / "geometry_summary.json", {"exists": True})
    _write_json(
        pipeline / "capability_checks.json",
        {"checks": ["bad", {"name": "reach", "status": "failed", "detail": "Reach envelope unclear"}]},
    )
    _write_json(
        pipeline / "blocker_register.json",
        {
            "entries": [
                {
                    "id": "route_width",
                    "category": "geometry",
                    "severity": "high",
                    "detail": "Route width needs metric confirmation",
                    "zone": {"name": "Receiving aisle"},
                },
                {
                    "id": "ambiguous_scope",
                    "category": "workflow_ambiguity",
                    "severity": "medium",
                    "detail": "Task boundary is ambiguous",
                },
            ]
        },
    )
    _write_json(pipeline / "readiness_decision.json", {"status": "not_ready_yet", "confidence": 0.42})
    (pipeline / "readiness_report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(pipeline / "opportunity_handoff.json", {"recommended_lane": "pilot_eval"})
    _write_json(pipeline / "human_actions_required.json", {"actions": []})
    _write_json(pipeline / "task_hypothesis_report.json", {"hypothesis": "move totes"})
    _write_json(pipeline / "normalized_task_hypothesis.json", {"task": "move_totes"})
    return capture_root


def _artifacts(**overrides):
    base = {
        "descriptor": SimpleNamespace(
            scene_id="scene-a",
            capture_id="cap-a",
            capture_modality="android_video_only",
            evidence_tier="pre_screen_video",
        ),
        "site_intake": {"task_context": {}, "constraints": {}},
        "capture_package_manifest": {},
        "capture_qa_scorecard": {},
        "task_scope_record": {"task_zone": "Zone A"},
        "qualification_record": {"measurements": {}, "risks": []},
        "qualification_brief": {},
        "scene_graph": {},
        "route_graph": {},
        "geometry_evidence": {},
        "supplemental_geometry": [],
        "capability_checks": {},
        "blocker_register": {"entries": []},
        "readiness_decision": {"status": "not_ready_yet", "confidence": 0.0},
        "readiness_report": "",
        "opportunity_handoff": {},
        "human_actions_required": {"actions": []},
        "task_hypothesis_report": {},
        "normalized_task_hypothesis": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeProvider:
    name = "fake"

    def __init__(self, overrides: dict[str, dict[str, object]] | None = None) -> None:
        self.overrides = overrides or {}

    def invoke_skill(self, skill_name: str, _payload):
        value = self.overrides.get(skill_name)
        return dict(value) if isinstance(value, dict) else None

    def skill_metadata(self, skill_name: str) -> dict[str, object]:
        return {"skill": skill_name, "fake": True}

    def runtime_metadata(self) -> dict[str, object]:
        return {"provider": self.name, "fake_runtime": True}


def test_agent_orchestrator_resolution_and_recapture_helper_edges() -> None:
    assert orch._zone_text({"label": "Dock"}) == "Dock"
    assert orch._zone_text({"unknown": "x"}) == ""
    assert orch._resolve_blocker_resolution_path(category="runtime", detail="", severity="") == "platform_change"
    assert orch._resolve_blocker_resolution_path(category="access", detail="", severity="") == "human_review"
    assert orch._resolve_blocker_resolution_path(category="integration", detail="", severity="") == "oem_consultation"
    assert orch._resolve_blocker_resolution_path(category="task_fit", detail="", severity="") == "platform_change"
    assert orch._resolve_blocker_resolution_path(category="traffic_shared", detail="", severity="") == "site_modification"
    assert orch._resolve_blocker_resolution_path(category="workflow_ambiguity", detail="", severity="") == "scope_change"
    assert orch._resolve_blocker_resolution_path(category="automation_gap", detail="Badge access needed", severity="low") == "human_review"
    assert orch._resolve_blocker_resolution_path(category="automation_gap", detail="Needs geometry", severity="low") == "recapture"
    assert orch._resolve_blocker_resolution_path(category="automation_gap", detail="Unclear", severity="high") == "recapture"
    assert orch._resolve_blocker_resolution_path(category="other", detail="Restricted escort needed", severity="") == "human_review"
    assert orch._resolve_blocker_resolution_path(category="other", detail="Plain blocker", severity="") == "human_review"

    assert orch._recapture_priority("hard_blocker") == "P0"
    assert orch._recapture_priority("low") == "P3"
    assert orch._recapture_priority("unknown") == "P4"
    assert orch._recapture_priority_rank("PX") == 5
    assert orch._recapture_equipment("floor", "") == ["digital inclinometer", "phone camera"]
    assert "calipers" in orch._recapture_equipment("machine_interface", "")
    assert orch._recapture_equipment("bad_capture_quality_detail", "") == ["tripod", "phone camera"]
    assert orch._recapture_equipment("misc", "plain")[0] == "phone camera"
    assert orch._preferred_capture_mode("misc", "route width", "android_video_only") == "iphone_arkit_lidar"
    assert orch._preferred_capture_mode("misc", "plain", "android_video_only") == "android_video_only"
    assert orch._recapture_access("", "access") == "restricted"
    assert orch._recapture_access("badge permission required", "misc") == "restricted; escort required"
    assert orch._recapture_timing("traffic_shared", "") == "during operations"
    assert orch._recapture_timing("misc", "badge required") == "scheduled access"

    hidden_steps = orch._recapture_instructions("hidden_zone", "hidden zone occlusion", "open", "iphone")
    width_steps = orch._recapture_instructions("misc", "route width clearance", "open", "iphone")
    floor_steps = orch._recapture_instructions("floor", "floor slope", "restricted", "iphone")
    assert any("previously hidden" in step for step in hidden_steps)
    assert any("narrowest point" in step for step in width_steps)
    assert any("floor condition" in step for step in floor_steps)
    assert any("access requirement" in step for step in floor_steps)

    assert "previously uncovered" in orch._recapture_acceptance_criteria("hidden_zone", "coverage")[0]
    assert "Slope" in orch._recapture_acceptance_criteria("floor", "grade")[0]
    assert "resolved" in orch._recapture_acceptance_criteria("misc", "plain")[0]
    assert orch._recapture_effort_minutes("misc", "plain", "restricted") == 30
    assert orch._recapture_effort_minutes("machine_interface", "", "open") == 30
    assert orch._recapture_effort_minutes("floor", "", "open") == 25
    assert orch._recapture_effort_minutes("misc", "route width", "open") == 20
    assert orch._recapture_effort_minutes("misc", "plain", "open") == 15

    assert orch._resolution_detail("recapture", "capture_coverage", "hidden coverage") == (
        "Re-capture the uncovered area with complete coverage and preserve provenance."
    )
    assert orch._resolution_detail("recapture", "floor", "slope") == "Re-capture the floor condition with calibrated metric evidence."
    assert orch._resolution_detail("scope_change", "", "") == "Adjust the qualification scope before another capture pass will help."
    assert orch._resolution_detail("site_modification", "", "") == "Requires a site change before another capture pass will help."
    assert orch._resolution_detail("oem_consultation", "", "") == "Confirm OEM or integrator constraints before recapture."
    assert orch._resolution_detail("platform_change", "", "").startswith("The current platform")
    assert orch._resolution_detail("not_resolvable", "", "").startswith("No capture-only")
    assert orch._resolution_detail("unknown", "", "").startswith("Capture-only")

    assert orch._build_recapture_step({"resolution_path": "human_review"}, fallback_capture_modality="iphone", default_zone="Zone") is None
    grouped = orch._group_recapture_sessions(
        [
            {"order": 1, "zone": "A", "access": "open", "equipment": ["phone"], "estimated_effort_minutes": 10},
            {"order": 2, "zone": "A", "access": "open", "equipment": ["phone"], "estimated_effort_minutes": 5},
        ]
    )
    assert grouped[0]["estimated_effort_minutes"] == 15


def test_agent_orchestrator_artifact_builders_and_memo_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orch,
        "_load_curated_standards",
        lambda: [
            {"title": "Route", "categories": ["geometry"], "summary": "Measure routes", "citation": "local"},
            {"title": "Fallback", "categories": ["other"], "summary": "Fallback note", "source": "local"},
        ],
    )
    artifacts = _artifacts(
        site_intake={
            "task_context": {"task_statement": "Move tote", "task_zone": "A", "success_criteria": "done"},
            "constraints": {"privacy_restrictions": "none"},
        },
        capture_qa_scorecard={"follow_ups": ["Missing route view", "Check glare"]},
        geometry_evidence={"hidden_zone_bound": 0.5, "metric_ready": True},
        route_graph={"edges": ["bad", {"id": "e1", "confidence": 0.3}]},
        supplemental_geometry=[{"path": "geom.json"}, {}],
        blocker_register={"entries": [{"id": "b1", "category": "geometry", "severity": "high", "detail": "Route width"}]},
        capability_checks={"checks": ["bad", {"name": "reach", "passed": False, "detail": "Reach unclear"}]},
        qualification_record={"measurements": {"minimum_route_width_m": 0.8}, "risks": [{"detail": "Pinch point"}, "bad"]},
        opportunity_handoff={
            "recommended_lane": "pilot",
            "target_robot_team": {"robot_platform": "G1"},
            "downstream_evaluation_eligibility": "blocked",
        },
        human_actions_required={"actions": [{"action": "Review route"}, "bad"]},
    )

    complete_intake = orch._normalized_intake(artifacts)
    missing_intake = orch._normalized_intake(_artifacts())
    assert complete_intake["status"] == "normalized"
    assert missing_intake["missing_required_fields"] == ["workflow", "task_zone", "success_criteria"]

    evidence = orch._evidence_audit(artifacts)
    empty_supplemental_evidence = orch._evidence_audit(_artifacts())
    assert evidence["status"] == "needs_more_evidence"
    assert evidence["low_confidence_route_edges"][0]["edge_id"] == "e1"
    assert evidence["supplemental_geometry"] == ["geom.json"]
    assert empty_supplemental_evidence["supplemental_geometry"] == []

    blocker_register = orch._agent_blocker_register(
        artifacts,
        {
            "evidence_gaps": [
                {"detail": "Route width", "category": "geometry"},
                "bad",
                {"detail": "", "category": "geometry"},
                {"detail": "New gap", "category": "capture_quality", "severity": "low"},
            ]
        },
    )
    assert {entry["detail"] for entry in blocker_register["entries"]} == {"Route width", "New gap"}

    capability = orch._capability_envelope(artifacts, evidence)
    assert capability["bounded_claims"][0].startswith("reach:")
    standards = orch._standards_notes(artifacts, blocker_register)
    fallback_standards = orch._standards_notes(_artifacts(), {"entries": []})
    assert standards["entries"][0]["title"] == "Route"
    assert len(fallback_standards["entries"]) == 2
    assert orch._existing_human_actions(artifacts) == [{"action": "Review route"}]

    recapture = orch._recapture_plan(artifacts, blocker_register)
    site_review = orch._humanoid_site_review(artifacts, standards)
    workcell = orch._humanoid_workcell_risk_review(artifacts)
    route = orch._humanoid_route_access_review(artifacts)
    handoff = orch._oem_handoff_summary(artifacts)
    assert "curated guidance" in site_review["summary"]
    assert workcell["risks"] == ["Pinch point"]
    assert route["minimum_route_width_m"] == 0.8
    assert "G1" in handoff["summary"]

    memo = orch._render_agent_memo(
        artifacts,
        normalized_intake=missing_intake,
        evidence_audit=evidence,
        standards_notes={"entries": [standards["entries"][0], "bad"]},
        recapture_plan=recapture,
        human_actions=[{"action": "Review route"}, "bad"],
    )
    assert "Missing required fields" in memo
    assert "Access pending" not in memo
    no_recapture_memo = orch._render_agent_memo(
        artifacts,
        normalized_intake=complete_intake,
        evidence_audit={"evidence_gaps": []},
        standards_notes={"entries": []},
        recapture_plan={"required": False},
        human_actions=[],
    )
    assert "No new evidence gaps" in no_recapture_memo
    assert "No recapture plan" in no_recapture_memo
    access_pending_memo = orch._render_agent_memo(
        artifacts,
        normalized_intake=complete_intake,
        evidence_audit={"evidence_gaps": ["bad", {"severity": "low", "detail": "Gap"}]},
        standards_notes={"entries": []},
        recapture_plan={
            "required": True,
            "access_pending": True,
            "steps": ["bad", {"order": 1, "detail": "Recapture route", "zone": "A", "preferred_capture_mode": "iphone"}],
        },
        human_actions=[],
    )
    assert "Access pending" in access_pending_memo
    assert "Recapture route" in access_pending_memo


def test_agent_orchestrator_load_curated_standards_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orch, "_repo_root", lambda: tmp_path)
    assert orch._load_curated_standards() == []
    corpus = tmp_path / "skillpacks" / "industrial_readiness" / "references" / "curated_standards.json"
    _write_json(corpus, {"entries": [{"title": "One"}, "bad", {"title": "Two"}]})
    assert [entry["title"] for entry in orch._load_curated_standards()] == ["One", "Two"]
    corpus.write_text("[]", encoding="utf-8")
    assert orch._load_curated_standards() == []


def test_agent_orchestrator_provider_selection_and_run_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setattr(orch, "sync_skill_pack", lambda _repo_root: None)
    monkeypatch.setattr(
        orch,
        "_load_curated_standards",
        lambda: [{"title": "Route", "categories": ["geometry", "hidden_zone"], "summary": "Measure", "citation": "local"}],
    )

    assert orch._provider_from_name("openai", repo_root=tmp_path, skill_runner=lambda *_args: None).name == "openai"
    assert orch._provider_from_name("claude", repo_root=tmp_path).name == "claude"
    with pytest.raises(ValueError, match="Unsupported agent provider"):
        orch._provider_from_name("bad", repo_root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported agent review mode"):
        orch.run_agent_review(capture_root=capture_root, provider_name="fake", mode="other")

    monkeypatch.setattr(orch, "_provider_from_name", lambda *_args, **_kwargs: FakeProvider())
    local_bundle = orch.run_agent_review(capture_root=capture_root, provider_name="fake")
    pipeline = capture_root / "pipeline"
    assert local_bundle["provider"] == "fake"
    assert (pipeline / "agent_readiness_memo.md").read_text(encoding="utf-8").startswith("# Agent Review Memo")
    assert any(step["source"] == "local_deterministic" for step in local_bundle["steps"])
    assert "final_operator_summary" in local_bundle["artifacts"]
    assert "humanoid_route_access_review" in local_bundle["specialized_reviews"]

    override_provider = FakeProvider(
        {
            "humanoid_site_readiness_reviewer": {"schema_version": "v1", "summary": "override"},
            "recapture_planner": {"schema_version": "v1", "required": False, "steps": []},
            "readiness_report_writer": {"memo_markdown": "# Provider memo\n"},
        }
    )
    monkeypatch.setattr(orch, "_provider_from_name", lambda *_args, **_kwargs: override_provider)
    override_bundle = orch.run_agent_review(capture_root=capture_root, provider_name="fake")
    assert (pipeline / "agent_readiness_memo.md").read_text(encoding="utf-8") == "# Provider memo\n"
    assert any(step["source"] == "provider_override" for step in override_bundle["steps"])
