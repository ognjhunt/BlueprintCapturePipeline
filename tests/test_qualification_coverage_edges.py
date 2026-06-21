from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.common import StageError
import blueprint_pipeline.qualification as q


def _descriptor(**overrides) -> CaptureDescriptor:
    defaults = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "capture_source": "iphone",
        "capture_tier": "candidate",
        "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
        "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/frames.json",
        "quality": {},
        "raw_video_uri": "gs://bucket/captures/scene-1/capture-1/raw/final_walkthrough.mov",
        "environment_type_hint": "warehouse",
        "capture_modality": "iphone_arkit_lidar",
        "evidence_tier": "pre_screen_video",
        "requested_lanes": ["qualification"],
        "requested_outputs": ["qualification"],
        "coverage_plan": ["dock aisle", "task zone"],
        "metadata": {},
    }
    defaults.update(overrides)
    return CaptureDescriptor(**defaults)


def test_qualification_small_helpers_and_handoff_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert q._default_qa_report_uri("gs://bucket/captures/a/capture_descriptor.json") == (
        "gs://bucket/captures/a/qa_report.json"
    )
    assert q._default_qa_report_uri("gs://bucket/captures/a/descriptor.json") == (
        "gs://bucket/captures/a/descriptor.json/qa_report.json"
    )
    monkeypatch.setattr(q, "resolve_gs_uri_to_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad uri")))
    assert q._safe_path_exists("gs://bucket/missing", tmp_path) is False
    assert q._try_read_optional_json_uri("gs://bucket/nope.json", tmp_path) is None
    assert q._string_list(123) == ["123"]

    assert q._presentation_bundle_status(
        emit_presentation=False,
        primary_asset=None,
        render_inputs={},
    ) == "disabled"
    assert q._presentation_bundle_status(
        emit_presentation=True,
        primary_asset={"path": "asset.ply"},
        render_inputs={},
    ) == "ready"
    assert q._presentation_bundle_status(
        emit_presentation=True,
        primary_asset={"path": "asset.ply"},
        render_inputs={"missing_inputs": ["camera"]},
    ) == "partial"

    build_report = tmp_path / "raw" / "object_index_build_report.json"
    build_report.parent.mkdir()
    build_report.write_text(
        json.dumps(
            {
                "runtime_preflight": {
                    "backends": {
                        "optional_backend": {"support_level": "optional"},
                        "required_backend": {"support_level": "required"},
                    }
                },
                "backend_summary": {
                    "providers": [
                        "bad",
                        {"backend": "optional_backend", "reason": "weights_missing"},
                        {"backend": "required_backend", "reason": "failed_to_launch"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert q._object_index_runtime_blockers(tmp_path) == [
        "object_index_backend:required_backend:failed_to_launch"
    ]

    descriptor = _descriptor(
        intake_packet_uri="gs://bucket/intake.json",
        metadata={
            "task_statement": "Pick tote",
            "task_zone": {"label": "Dock A"},
            "success_criteria": ["Tote staged"],
            "intake_source": "operator_form",
        },
    )
    assert q._has_structured_intake(descriptor)
    assert q._normalize_task_hypothesis_source({}, descriptor) == "operator_form"
    assert q._normalize_task_hypothesis_source({"source": "ai_inferred"}, descriptor) == "ai_inferred"

    industrial_contradiction = q._build_task_hypothesis_report(
        descriptor=_descriptor(environment_type_hint="warehouse"),
        raw_task_hypothesis={
            "source": "ai_inferred",
            "workflow_name": "Bedroom closet laundry reset",
            "task_steps": ["open closet"],
            "confidence": 0.95,
        },
        object_index_entries=[],
        task_targets_payload={},
    )
    assert industrial_contradiction["task_hypothesis_status"] == "contradicted"
    rejected = q._build_task_hypothesis_report(
        descriptor=_descriptor(environment_type_hint="kitchen"),
        raw_task_hypothesis={
            "source": "ai_inferred",
            "workflow_name": "Forklift pallet staging",
            "task_steps": ["move pallet"],
            "status": "rejected",
            "confidence": 0.9,
        },
        object_index_entries=[],
        task_targets_payload={},
    )
    assert "rejected before qualification" in " ".join(rejected["contradictions"])
    warning_report = q._build_task_hypothesis_report(
        descriptor=_descriptor(environment_type_hint="warehouse"),
        raw_task_hypothesis={
            "source": "ai_inferred",
            "workflow_name": "Pick and place",
            "task_steps": ["scan"],
            "confidence": 0.2,
        },
        object_index_entries=[],
        task_targets_payload={},
    )
    assert warning_report["task_hypothesis_status"] == "needs_confirmation"
    accepted_report = q._build_task_hypothesis_report(
        descriptor=_descriptor(environment_type_hint="warehouse"),
        raw_task_hypothesis={
            "source": "ai_inferred",
            "workflow_name": "Move tote to rack",
            "task_steps": ["pick tote"],
            "zone": "Dock",
            "confidence": 0.95,
        },
        object_index_entries=[{"id": "tote-1", "label": "tote"}],
        task_targets_payload={},
    )
    assert accepted_report["task_hypothesis_status"] == "accepted"
    accepted_with_warnings = q._build_task_hypothesis_report(
        descriptor=_descriptor(environment_type_hint="warehouse"),
        raw_task_hypothesis={
            "source": "ai_inferred",
            "workflow_name": "Move unspecified item",
            "task_steps": ["stage item"],
            "zone": "Dock",
            "confidence": 0.65,
        },
        object_index_entries=[],
        task_targets_payload={},
    )
    assert accepted_with_warnings["task_hypothesis_status"] == "accepted_with_warnings"

    effective = q._effective_task_metadata(
        descriptor,
        task_hypothesis_report={
            "task_hypothesis_status": "contradicted",
            "normalized_task_hypothesis": {"confidence": 0.4, "source": "ai_inferred"},
        },
    )
    assert effective["task_hypothesis_status"] == "contradicted"
    assert effective["task_hypothesis_confidence"] == 0.4
    assert q._modality_supports_metric_automation(
        _descriptor(
            evidence_tier="video_with_validated_scaffolding",
            scaffolding_validation={"validated_metric_bundle": True},
        )
    )

    base_dir = tmp_path / "pipeline"
    (base_dir / "advanced_geometry").mkdir(parents=True)
    for name in [
        "advanced_geometry_bundle.json",
        "3dgs_compressed.ply",
        "labels.json",
        "structure.json",
        "holi_spatial_grounding.json",
        "task_targets.synthetic.json",
    ]:
        (base_dir / "advanced_geometry" / name).write_text("{}", encoding="utf-8")
    scene_memory = base_dir / "scene_memory"
    scene_memory.mkdir()
    (scene_memory / "scene_memory_manifest.json").write_text("{}", encoding="utf-8")
    (scene_memory / "scene_memory_readiness.json").write_text("{}", encoding="utf-8")
    (scene_memory / "conditioning_bundle.json").write_text("{}", encoding="utf-8")
    preview = base_dir / "preview_simulation"
    preview.mkdir()
    (preview / "preview_simulation_manifest.json").write_text("{}", encoding="utf-8")
    presentation = base_dir / "presentation_world"
    presentation.mkdir()
    for name in [
        "presentation_bundle.json",
        "presentation_world_manifest.json",
        "runtime_demo_manifest.json",
    ]:
        (presentation / name).write_text("{}", encoding="utf-8")
    handoff = q.attach_handoff_package_paths(
        {"scene_package": {"stale": True}},
        pipeline_dir=base_dir,
        metadata={"scene_package": {"bundle_path": "local_scene_bundle"}},
    )
    assert handoff["geometry_package"]["ply_path"].endswith("3dgs_compressed.ply")
    assert handoff["scene_memory_package"]["runtime_demo_manifest_path"].endswith(
        "runtime_demo_manifest.json"
    )
    assert handoff["scene_package"]["scene_package_path"] == "local_scene_bundle"
    assert q._requested_downstream_lanes(
        descriptor=_descriptor(requested_outputs=["managed_tuning"]),
        requested_lanes=[],
    ) == ["scene_memory"]


def test_qualification_record_brief_and_opportunity_edge_paths(tmp_path: Path) -> None:
    object_entries = [
        {
            "id": "aisle-1",
            "label": "narrow aisle",
            "boundingBox": {"center": [0, 0, 0], "extents": [0.5, 3.0, 1.0]},
        },
        {
            "id": "target-1",
            "label": "valve panel",
            "boundingBox": {"center": [2.0, 0, 0], "extents": [0.2, 0.2, 0.2]},
        },
    ]
    scorecard = {"completeness_status": "sufficient", "qa_status": "passed", "score": 0.9}
    scope = {
        "scope_status": "scoped",
        "target_object_ids": ["target-1"],
        "articulation_required_ids": ["target-1"],
        "task_zone": {"center": [0.0, 0.0, 0.0]},
        "task_hypothesis_status": "needs_confirmation",
        "blockers": [],
        "success_criteria": [],
    }
    record = q._build_qualification_record(
        descriptor=_descriptor(
            evidence_tier="qualified_metric_capture",
            metadata={"privacy_restrictions": ["faces"], "safety_concerns": ["forklift"]},
        ),
        scorecard=scorecard,
        scope_record=scope,
        object_index_entries=object_entries,
        object_index_runtime_blockers=["object_index_backend:required_backend:failed_to_launch"],
    )
    risk_ids = {risk["id"] for risk in record["risks"]}
    assert {
        "task_hypothesis_needs_confirmation",
        "articulation_complexity",
        "route_clearance_risk",
        "reach_risk",
        "privacy_restrictions",
        "object_index_runtime_missing",
    }.issubset(risk_ids)

    scaffolding_record = q._build_qualification_record(
        descriptor=_descriptor(
            capture_modality="glasses_plus_scaffolding",
            evidence_tier="video_with_validated_scaffolding",
            scaffolding_validation={},
        ),
        scorecard=scorecard,
        scope_record={**scope, "task_hypothesis_status": "contradicted"},
        object_index_entries=object_entries,
    )
    scaffolding_risks = {risk["id"] for risk in scaffolding_record["risks"]}
    assert "missing_validated_scaffolding" in scaffolding_risks
    assert "task_hypothesis_contradicted" in scaffolding_risks

    pre_screen_record = q._build_qualification_record(
        descriptor=_descriptor(evidence_tier="pre_screen_video"),
        scorecard=scorecard,
        scope_record={**scope, "task_hypothesis_status": ""},
        object_index_entries=[],
    )
    assert any(risk["id"] == "non_metric_capture" for risk in pre_screen_record["risks"])

    brief = q._build_qualification_brief(
        descriptor=_descriptor(),
        scorecard=scorecard,
        scope_record={"scope_status": "scoped", "task_statement": "Inspect valve"},
        qualification_record={
            "readiness_state": "ready",
            "confidence": 0.9,
            "risks": [],
            "advanced_geometry_recommended": False,
        },
    )
    assert brief["next_steps"] == ["Route the opportunity handoff to deployment, process, and safety reviewers."]

    handoff = q._build_opportunity_handoff(
        descriptor=_descriptor(),
        scorecard=scorecard,
        scope_record={
            "scope_status": "scoped",
            "target_object_ids": ["target-1"],
            "success_criteria": ["Valve reached"],
        },
        qualification_record={"readiness_state": "ready", "confidence": 0.92, "risks": []},
        brief=brief,
        config=SimpleNamespace(robot_type="g1"),
        pipeline_dir=tmp_path,
        metadata_override={
            "site_submission_id": "site-1",
            "buyer_request_id": "buyer-1",
            "capture_job_id": "capture-job-1",
            "operating_hours": "nights",
            "robot_platform": "g1",
        },
    )
    assert handoff["scoped_task_definition"]["in_scope_zone"] == ["target-1"]
    assert handoff["site_constraints"]["operating_constraints"] == ["nights"]
    assert handoff["target_robot_team"]["robot_platform"] == "g1"

    fallback_handoff = q._build_opportunity_handoff(
        descriptor=_descriptor(environment_type_hint="warehouse"),
        scorecard={"completeness_status": "need_more_evidence"},
        scope_record={"scope_status": "needs_clarification"},
        qualification_record={"readiness_state": "not_ready_yet", "confidence": 0.1, "risks": []},
        brief={},
        config=SimpleNamespace(robot_type=None),
        pipeline_dir=tmp_path,
        metadata_override={},
    )
    assert fallback_handoff["scoped_task_definition"]["in_scope_zone"] == "warehouse"
    assert fallback_handoff["site_constraints"]["known_blockers"] == ["No known blockers supplied"]
    scope_defaults = q._build_task_scope_record(
        descriptor=_descriptor(metadata={}),
        task_targets_payload={},
        completeness_status="need_more_evidence",
    )
    assert scope_defaults["success_criteria"] == [
        "Identify the task zone, key objects, and blockers well enough for buyer review."
    ]


def test_scene_memory_bundle_and_geometry_artifact_helpers(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "bucket" / "pipeline"
    advanced_dir = pipeline_dir / "advanced_geometry"
    advanced_dir.mkdir(parents=True)
    for name in [
        "advanced_geometry_bundle.json",
        "3dgs_compressed.ply",
        "labels.json",
        "structure.json",
        "task_targets.synthetic.json",
    ]:
        (advanced_dir / name).write_text("{}", encoding="utf-8")
    descriptor = _descriptor(
        evidence_tier="qualified_metric_capture",
        raw_video_uri="gs://bucket/captures/scene-1/capture-1/privacy/final_walkthrough.mp4",
        metadata={
            "capture_rights": {
                "derived_scene_generation_allowed": True,
                "data_licensing_allowed": True,
            },
            "scene_memory_capture": {"world_model_candidate": True},
        },
    )
    artifacts = q._write_scene_memory_bundle(
        storage_root=tmp_path / "bucket",
        bucket="bucket",
        pipeline_prefix="pipeline",
        pipeline_dir=pipeline_dir,
        descriptor=descriptor,
        scorecard={"completeness_status": "sufficient"},
        qualification_record={"readiness_state": "ready", "metric_ready": True, "confidence": 0.9},
        geometry_artifacts={
            "geometry_summary_uri": "gs://bucket/pipeline/geometry/geometry_summary.json",
            "geometry_manifest_uri": "gs://bucket/pipeline/geometry/geometry_manifest.json",
            "camera_poses_uri": "gs://bucket/pipeline/geometry/camera/poses.jsonl",
            "summary": {
                "status": "completed",
                "ready_for_world_model": True,
                "scale_assessment": {"status": "metric"},
            },
        },
        depth_conditioning={
            "source": "privacy",
            "depth_manifest_uri": "gs://bucket/depth.json",
            "confidence_manifest_uri": "gs://bucket/confidence.json",
        },
    )

    assert artifacts["scene_memory_status"] == "ready"
    assert artifacts["geometry_summary_uri"].endswith("geometry_summary.json")
    assert artifacts["depth_conditioning"]["source"] == "privacy"
    assert (pipeline_dir / "presentation_world" / "authoritative_runtime_render_manifest.json").is_file()

    disabled = q._disabled_task_targets("scene", "capture", "missing")
    assert disabled["inference_mode"] == "disabled"
    assert disabled["video_analysis"]["external_inference"]["reason"] == "missing"


def test_worldlabs_input_video_preparation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "bucket"
    pipeline_dir = storage_root / "pipeline"
    pipeline_dir.mkdir(parents=True)
    no_source = q._prepare_worldlabs_input_video(
        descriptor=_descriptor(raw_video_uri=None),
        privacy_processing={},
        storage_root=storage_root,
        pipeline_dir=pipeline_dir,
        bucket="bucket",
    )
    assert no_source["status"] == "blocked"

    privacy_uri = "gs://bucket/captures/scene-1/capture-1/privacy/final_walkthrough.mp4"
    descriptor = _descriptor(privacy_processed_video_uri=privacy_uri)
    monkeypatch.setattr(
        q.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="ffprobe failed", stdout=""),
    )
    with pytest.raises(StageError, match="ffprobe_failed"):
        q._ffprobe_video_metrics(tmp_path / "bad.mp4")
    monkeypatch.setattr(
        q.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="duration=not-a-number\nsize=not-a-number\n",
        ),
    )
    assert q._ffprobe_video_metrics(tmp_path / "weird.mp4") == {
        "duration_seconds": 0.0,
        "size_bytes": 0,
    }
    assert q._allow_raw_worldlabs_bypass(
        descriptor=_descriptor(metadata={"allow_raw_worldlabs_bypass": "true"}),
        privacy_processing={},
    )
    raw_labeling = q._worldlabs_input_labeling(
        source_id="raw_video_uri",
        privacy_status="not_run",
        bypass_allowed=True,
    )
    assert raw_labeling["review_state"] == "non_production_unredacted_raw_preview"
    monkeypatch.setattr(q, "_resolve_optional_uri_to_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(q, "production_launch_mode", lambda: True)
    with pytest.raises(StageError, match="source_video_missing"):
        q._prepare_worldlabs_input_video(
            descriptor=descriptor,
            privacy_processing={
                "status": "person_removed",
                "privacy_processed_video_uri": privacy_uri,
                "privacy_manifest_uri": "gs://bucket/privacy/manifest.json",
            },
            storage_root=storage_root,
            pipeline_dir=pipeline_dir,
            bucket="bucket",
        )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(q, "production_launch_mode", lambda: False)
    monkeypatch.setattr(q, "_resolve_optional_uri_to_path", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(q.shutil, "which", lambda _name: None)
    with pytest.raises(StageError, match="ffmpeg_not_found"):
        q._prepare_worldlabs_input_video(
            descriptor=descriptor,
            privacy_processing={"status": "person_removed", "privacy_processed_video_uri": privacy_uri},
            storage_root=storage_root,
            pipeline_dir=pipeline_dir,
            bucket="bucket",
        )

    monkeypatch.setattr(q.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(q, "_ffprobe_video_metrics", lambda path: {"duration_seconds": 42.0, "size_bytes": 10})
    monkeypatch.setattr(
        q.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="bad transcode", stdout=""),
    )
    with pytest.raises(StageError, match="ffmpeg_failed"):
        q._prepare_worldlabs_input_video(
            descriptor=descriptor,
            privacy_processing={"status": "person_removed", "privacy_processed_video_uri": privacy_uri},
            storage_root=storage_root,
            pipeline_dir=pipeline_dir,
            bucket="bucket",
        )

    def fake_ffmpeg(command, **_kwargs):
        Path(command[-1]).write_bytes(b"output")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    def fake_ffprobe(path: Path) -> dict[str, object]:
        if path == source:
            return {"duration_seconds": 42.0, "size_bytes": 10}
        return {"duration_seconds": 20.0, "size_bytes": 9}

    monkeypatch.setattr(q.subprocess, "run", fake_ffmpeg)
    monkeypatch.setattr(q, "_ffprobe_video_metrics", fake_ffprobe)
    ready = q._prepare_worldlabs_input_video(
        descriptor=descriptor,
        privacy_processing={
            "status": "person_removed",
            "privacy_processed_video_uri": privacy_uri,
            "privacy_manifest_uri": "gs://bucket/privacy/manifest.json",
        },
        storage_root=storage_root,
        pipeline_dir=pipeline_dir,
        bucket="bucket",
    )
    assert ready["status"] == "ready"
    assert ready["audit_payload"]["privacy_safe_input"] is True

    monkeypatch.setattr(
        q,
        "_worldlabs_source_candidate",
        lambda **_kwargs: {
            "privacy_status": "not_run",
            "raw_video_bypass_allowed": True,
            "selected": {"source_id": "raw_video_uri", "uri": "gs://bucket/raw/final.mov"},
            "candidates": [],
        },
    )
    monkeypatch.setattr(q, "production_launch_mode", lambda: True)
    with pytest.raises(StageError, match="production_worldlabs_input_not_privacy_safe"):
        q._prepare_worldlabs_input_video(
            descriptor=_descriptor(raw_video_uri="gs://bucket/raw/final.mov"),
            privacy_processing={},
            storage_root=storage_root,
            pipeline_dir=pipeline_dir,
            bucket="bucket",
        )


def test_scene_graph_route_readiness_and_llm_payload_edges() -> None:
    descriptor = _descriptor(
        evidence_tier="qualified_metric_capture",
        metadata={"adjacent_systems": ["WMS"], "task_statement": "Move pallet"},
    )
    object_entries = [
        "bad",
        {"label": "missing id"},
        {
            "id": "forklift-lane",
            "label": "guardrail",
            "boundingBox": {"center": [0.5, 0, 0], "extents": [0.6, 2.0, 1.0]},
        },
        {
            "id": "target-1",
            "label": "pallet",
            "boundingBox": {"center": [2.0, 0, 0], "extents": [0.5, 0.5, 0.5]},
        },
    ]
    scope = {
        "task_zone": {"label": "Dock", "center": [0.0, 0.0, 0.0]},
        "target_object_ids": ["target-1"],
    }
    scene_graph = q._build_scene_graph(
        descriptor=descriptor,
        scope_record=scope,
        object_index_entries=object_entries,
    )
    assert any(node["type"] == "system" for node in scene_graph["nodes"])
    assert any(edge["relation"] == "hazard_near_task" for edge in scene_graph["edges"])

    custom_route_graph = q._build_route_graph(
        descriptor=descriptor,
        scene_graph={
            "nodes": [
                {"id": "task_zone", "center_m": [0, 0, 0]},
                {
                    "id": "handoff-1",
                    "category": "handoff_point",
                    "label": "handoff",
                    "center_m": [1, 0, 0],
                },
            ]
        },
    )
    assert any(node["type"] == "handoff" for node in custom_route_graph["nodes"])
    no_task_zone_route = q._build_route_graph(descriptor=descriptor, scene_graph={"nodes": []})
    assert no_task_zone_route["edges"] == []

    capability_checks = q._build_capability_checks(
        descriptor=descriptor,
        geometry_evidence={
            "metric_ready": True,
            "measured_route_width_m": 0.5,
            "target_reach_distance_m": 1.5,
            "workcell_span_m": 3.5,
            "hidden_zone_bound": 0.5,
            "uncertainty_score": 0.4,
        },
        route_graph={"edges": [{"source": "entry", "target": "task_zone"}]},
        scope_record={"target_object_ids": ["target-1"]},
    )
    assert any(check["status"] == "blocked" for check in capability_checks["checks"])

    blocker_register = q._build_blocker_register(
        descriptor=descriptor,
        qualification_record={"risks": ["bad", {"id": "risk", "severity": "high", "detail": "High risk"}]},
        capability_checks={"checks": ["bad", {"id": "ok", "status": "pass"}, {"id": "gap", "status": "blocked", "detail": "Gap"}]},
        geometry_evidence={"uncertainty_score": 0.4},
    )
    assert len(blocker_register["entries"]) == 2
    assert q._measure_width([0.0, 0.0, 1.0]) == 0.0
    assert q._group_objects_by_entity_type(["bad"]) == {}
    decision = q._build_readiness_decision(
        descriptor=descriptor,
        qualification_record={"readiness_state": "ready", "confidence": 0.9},
        blocker_register=blocker_register,
        capability_checks=capability_checks,
        geometry_evidence={"uncertainty_score": 0.1, "hidden_zone_bound": 0.5},
    )
    assert decision["status"] == "not_ready_yet"
    risky = q._build_readiness_decision(
        descriptor=descriptor,
        qualification_record={"readiness_state": "ready", "confidence": 0.9},
        blocker_register={"entries": []},
        capability_checks={"checks": []},
        geometry_evidence={"uncertainty_score": 0.1, "hidden_zone_bound": 0.5},
    )
    assert risky["status"] == "risky"

    report = q._render_readiness_report(
        descriptor=descriptor,
        readiness_decision={"status": "ready", "confidence": 0.9, "human_review_required": True},
        blocker_register={"entries": ["bad"]},
        human_actions_required={"actions": ["bad", {"action": "Review"}]},
    )
    assert "Review" in report
    empty_report = q._render_readiness_report(
        descriptor=descriptor,
        readiness_decision={
            "status": "ready",
            "confidence": 0.9,
            "human_review_required": True,
            "human_review_scope": [],
        },
        blocker_register={"entries": []},
        human_actions_required={"actions": []},
    )
    assert "None recorded" in empty_report
    assert "Final human signoff remains required" in empty_report
    weakness_payload = q._llm_weakness_payload(
        descriptor=descriptor,
        scorecard={"completeness_status": "sufficient"},
        scope_record={"scope_status": "scoped"},
        readiness_decision=decision,
        blocker_register=blocker_register,
        human_actions_required={"actions": []},
    )
    recapture_payload = q._llm_recapture_payload(
        descriptor=descriptor,
        scorecard={"follow_ups": ["rescan"]},
        scope_record={"blockers": ["blocked"]},
        blocker_register=blocker_register,
        human_actions_required={"blocker_details": ["detail"]},
    )
    assert weakness_payload["scene_id"] == descriptor.scene_id
    assert recapture_payload["scorecard_follow_ups"] == ["rescan"]


def test_world_model_payout_and_fidelity_adjustment_edges() -> None:
    descriptor = _descriptor(
        evidence_tier="qualified_metric_capture",
        quoted_payout_cents=5000,
        metadata={
            "capture_rights": {
                "derived_scene_generation_allowed": True,
                "capture_contributor_payout_eligible": True,
                "consent_status": "documented",
            }
        },
    )
    review = {
        "status": "succeeded",
        "confidence": 0.6,
        "scores": {
            "coverage": 0.9,
            "world_model_fitness": 0.8,
            "task_understanding": 0.8,
            "payout_quality": 0.0,
        },
        "assessments": {"blur": {"status": "poor"}},
        "bonus_signals": {"complete_coverage": {"score": "bad"}},
        "findings": {"blur_observations": ["motion blur"]},
    }
    fit = q._build_world_model_fit_summary(
        descriptor=descriptor,
        scorecard={"completeness_status": "sufficient"},
        qualification_record={"readiness_state": "ready", "confidence": 0.9},
        capture_fidelity_review=review,
        privacy_processing={"status": "person_removed", "mode": "local"},
        metadata=descriptor.metadata,
        geometry_summary={
            "status": "completed",
            "ready_for_world_model": True,
            "scale_assessment": {"status": "metric"},
            "deliverables": {"pose_coverage": 0.9},
        },
    )
    assert fit["status"] == "good_candidate"
    assert any("Advisory geometry" in reason for reason in fit["reasons"])
    failed_fit = q._build_world_model_fit_summary(
        descriptor=descriptor,
        scorecard={"completeness_status": "sufficient"},
        qualification_record={"readiness_state": "ready", "confidence": 0.9},
        capture_fidelity_review={"status": "failed", "scores": {}},
        privacy_processing={"status": "person_removed"},
        metadata=descriptor.metadata,
    )
    assert failed_fit["status"] == "review_required"

    payout = q._build_capturer_payout_recommendation(
        descriptor=descriptor,
        capture_fidelity_review=review,
        metadata=descriptor.metadata,
    )
    assert payout["status"] == "discount"
    assert payout["recommended_payout_cents"] == 4000
    assert any("baseline" in reason for reason in payout["reasons"])
    blocked_payout = q._build_capturer_payout_recommendation(
        descriptor=descriptor,
        capture_fidelity_review={"status": "failed"},
        metadata={},
    )
    assert blocked_payout["status"] == "review_required"
    assert blocked_payout["recommended_payout_cents"] is None

    failed_adjusted = q._apply_capture_fidelity_to_qualification(
        qualification_record={"readiness_state": "ready", "confidence": 0.9, "risks": []},
        capture_fidelity_review={"status": "failed"},
        metadata=descriptor.metadata,
    )
    assert failed_adjusted["readiness_state"] == "not_ready_yet"
    assert failed_adjusted["confidence"] == 0.45
    low_coverage = q._apply_capture_fidelity_to_qualification(
        qualification_record={"readiness_state": "ready", "confidence": 0.9, "risks": []},
        capture_fidelity_review={
            "status": "succeeded",
            "confidence": 0.9,
            "scores": {"coverage": 0.4, "world_model_fitness": 0.9, "task_understanding": 0.8},
            "assessments": {},
            "findings": {},
        },
        metadata=descriptor.metadata,
    )
    assert any(risk["id"] == "gemini_missing_views" for risk in low_coverage["risks"])
    review_required = q._apply_capture_fidelity_to_qualification(
        qualification_record={"readiness_state": "ready", "confidence": 0.9, "risks": []},
        capture_fidelity_review={
            "status": "succeeded",
            "confidence": 0.9,
            "scores": {"coverage": 0.9, "world_model_fitness": 0.5, "task_understanding": 0.8},
            "assessments": {
                "task_zone_completeness": {"status": "review_required"},
                "blur": {"status": "review_required"},
            },
            "findings": {"blur_observations": ["blur"]},
        },
        metadata={},
    )
    assert review_required["readiness_state"] == "risky"
    assert review_required["advanced_geometry_recommended"] is False
    assessment_only = q._apply_capture_fidelity_to_qualification(
        qualification_record={"readiness_state": "ready", "confidence": 0.9, "risks": []},
        capture_fidelity_review={
            "status": "succeeded",
            "confidence": 0.9,
            "scores": {"coverage": 0.9, "world_model_fitness": 0.9, "task_understanding": 0.8},
            "assessments": {"blur": {"status": "review_required"}},
            "findings": {},
        },
        metadata=descriptor.metadata,
    )
    assert assessment_only["readiness_state"] == "risky"
    findings_only = q._apply_capture_fidelity_to_qualification(
        qualification_record={"readiness_state": "ready", "confidence": 0.9, "risks": []},
        capture_fidelity_review={
            "status": "succeeded",
            "confidence": 0.9,
            "scores": {"coverage": 0.9, "world_model_fitness": 0.9, "task_understanding": 0.8},
            "assessments": {},
            "findings": {"blur_observations": ["blur"]},
        },
        metadata=descriptor.metadata,
    )
    assert findings_only["readiness_state"] == "risky"


def _write_descriptor(storage_root: Path, descriptor: CaptureDescriptor) -> str:
    descriptor_uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    descriptor_path = (
        storage_root / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    )
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(json.dumps(descriptor.to_dict()), encoding="utf-8")
    return descriptor_uri


def _patch_pipeline_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        q,
        "infer_capture_fidelity_review",
        lambda **_kwargs: {
            "status": "succeeded",
            "confidence": 0.9,
            "scores": {
                "coverage": 0.9,
                "world_model_fitness": 0.9,
                "task_understanding": 0.9,
                "payout_quality": 0.8,
            },
            "assessments": {},
            "findings": {},
            "bonus_signals": {},
        },
    )
    monkeypatch.setattr(
        q,
        "run_privacy_postprocess",
        lambda **_kwargs: {
            "status": "no_people_detected",
            "mode": "test",
            "privacy_manifest_uri": "gs://bucket/pipeline/privacy_processing_manifest.json",
            "privacy_verification_report_uri": "gs://bucket/pipeline/privacy_verification_report.json",
        },
    )

    def enrichment_runner(name, _payload):
        if name == "qualification_weakness_summarizer":
            return {"summary": "weaknesses"}
        if name == "recapture_instruction_writer":
            return {"instructions": ["rescan dock"]}
        return None

    monkeypatch.setattr(q, "build_capture_enrichment_runner", lambda **_kwargs: enrichment_runner)
    monkeypatch.setattr(
        q,
        "write_blueprint_canonical_site_package",
        lambda **_kwargs: {
            "canonical_site_package_uri": "gs://bucket/pipeline/canonical_site_package.json",
            "canonical_site_package": {"status": "ready"},
            "provider_adapter_input_uris": {"world_labs_marble": "gs://bucket/pipeline/adapter.json"},
            "provider_adapter_inputs": {"world_labs_marble": {"status": "ready", "blockers": []}},
        },
    )
    monkeypatch.setattr(q, "sync_webapp_pipeline_attachment", lambda **_kwargs: {"status": "skipped"})
    monkeypatch.setattr(q, "write_pipeline_sync_result", lambda **_kwargs: None)
    monkeypatch.setattr(q, "write_alpha_readiness_summary", lambda **_kwargs: None)


def test_run_qualification_pipeline_disabled_preflight_and_llm_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(
        storage_root,
        _descriptor(
            raw_video_uri=None,
            metadata={
                "task_statement": "Inspect dock",
                "task_zone": {"label": "Dock"},
                "success_criteria": ["Dock inspected"],
            },
        ),
    )
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(q, "load_raw_manifest", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad manifest")))

    result = q.run_qualification_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False),
    )

    pipeline_dir = storage_root / "scenes" / "scene-1" / "captures" / "capture-1" / "pipeline"
    assert result["status"] == "completed"
    assert json.loads((pipeline_dir / "runtime_preflight_report.json").read_text())["status"] == "skipped"
    assert json.loads((pipeline_dir / "task_targets.json").read_text())["inference_mode"] == "disabled"
    assert (pipeline_dir / "qualification_weakness_summary.json").is_file()
    assert (pipeline_dir / "recapture_instructions.json").is_file()


def test_run_qualification_pipeline_object_index_stage_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "gcs"
    descriptor_uri = _write_descriptor(
        storage_root,
        _descriptor(
            raw_video_uri=None,
            object_index_uri="gs://bucket/captures/scene-1/capture-1/object_index.json",
            metadata={"task_statement": "Inspect dock"},
        ),
    )
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(q, "load_raw_manifest", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        q,
        "ensure_object_index_stage",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("object index unavailable")),
    )
    monkeypatch.setattr(q, "load_object_index", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        q,
        "infer_task_targets",
        lambda **_kwargs: q._disabled_task_targets("scene-1", "capture-1", "patched"),
    )

    result = q.run_qualification_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        config=SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False),
    )

    assert result["status"] == "completed"


def test_run_qualification_pipeline_suppresses_failure_writer_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q, "_write_failure", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

    with pytest.raises(q.PipelineError):
        q.run_qualification_pipeline(
            descriptor_gcs_uri="not-a-gs-uri",
            config=SimpleNamespace(gcs_root=tmp_path / "gcs", runtime_preflight_enabled=False),
        )
