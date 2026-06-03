from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.simulation_automation import (
    FakeSimulationAutomationAgentAdapter,
    build_simulation_automation,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1", "site_identity": {"site_id": "site-1"}},
    )
    return capture_root


def _write_worldlabs_and_marble_artifacts(capture_root: Path) -> None:
    pipeline_dir = capture_root / "pipeline"
    _write_json(
        pipeline_dir / "worldlabs_request_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "selected_video_uri": "gs://local-blueprint/privacy/final_walkthrough.mov",
            "privacy_safe_input": True,
        },
    )
    _write_json(
        pipeline_dir / "worldlabs_world_manifest.json",
        {
            "schema_version": "worldlabs_world_manifest.v1",
            "world_id": "world-1",
            "world_marble_url": "https://marble.worldlabs.ai/worlds/world-1",
            "model": "marble-1.1",
            "updated_at": "2026-06-03T00:00:00Z",
            "assets": {
                "mesh": {"collider_mesh_url": "https://cdn.worldlabs.ai/world-1/collider.glb"},
                "splats": {
                    "spz_urls": {"full": "https://cdn.worldlabs.ai/world-1/full.spz"},
                    "semantics_metadata": {
                        "metric_scale_factor": 0.5,
                        "ground_plane_offset": 1.0,
                    },
                },
            },
        },
    )
    _write_json(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json",
        {
            "schema_version": "marble_simready_bridge.v1",
            "status": "review_ready_with_conversion_required",
            "world_id": "world-1",
            "simulator_review_manifests": {
                "isaac_sim": "simulators/isaac_sim_review_manifest.json",
                "mujoco": "simulators/mujoco_review_manifest.json",
                "pybullet": "simulators/pybullet_review_manifest.json",
            },
            "evaluation_prep_summary": {
                "collider_mesh_available": True,
                "metric_alignment_ready": True,
                "robot_readiness_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "marble_sim_assets" / "marble_asset_validation.json",
        {
            "schema_version": "marble_asset_validation.v1",
            "overall_status": "review_ready_with_conversion_required",
            "physics_collision_review_ready": True,
            "isaac_visual_conversion_required": True,
            "robot_readiness_proven": False,
        },
    )
    _write_json(
        pipeline_dir / "simready" / "simready_scene_manifest.json",
        {
            "schema_version": "simready_scene_manifest.v1",
            "status": "prepared_for_review",
            "framework_artifacts": {
                "isaac_sim": {"path": "isaac_sim/site_scene.usda", "load_status": "not_executed"},
                "mujoco": {"path": "mujoco/site_scene.xml", "load_status": "not_executed"},
                "pybullet": {"path": "pybullet/site_scene.urdf", "load_status": "not_executed"},
            },
            "claim_boundary": {
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "simready" / "simready_validation.json",
        {
            "schema_version": "simready_validation.v1",
            "overall_status": "prepared_for_review",
            "claim_boundary": {
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
            },
        },
    )
    _write_json(
        pipeline_dir / "cosmos_training_export" / "manifest.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "source_mode": "capture_grounded_fixture",
            "trainer_config_path": "trainer_config.json",
        },
    )


def test_simulation_automation_default_is_local_only_and_blocked(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)

    result = build_simulation_automation(
        capture_root=capture_root,
        agent_adapter=FakeSimulationAutomationAgentAdapter(),
    )

    automation_root = capture_root / "pipeline" / "simulation_automation"
    plan = _read_json(automation_root / "simulation_automation_plan.json")
    run_manifest = _read_json(automation_root / "simulation_automation_run_manifest.json")
    conversion = _read_json(automation_root / "asset_conversion_plan.json")
    simulator_execution = _read_json(automation_root / "simulator_execution_manifest.json")
    training = _read_json(automation_root / "training_orchestration_manifest.json")
    proof_boundary = _read_json(automation_root / "proof_boundary.json")
    agent_ledger = _read_json(automation_root / "agent_decision_ledger.json")

    assert result["status"] == "blocked"
    assert plan["source_artifacts"]["worldlabs_world_manifest"].endswith(
        "../worldlabs_world_manifest.json"
    )
    assert plan["source_artifacts"]["marble_simready_bridge"].endswith(
        "../marble_sim_assets/marble_simready_bridge.json"
    )
    assert plan["world_model_sources"]["worldlabs"]["world_id"] == "world-1"
    assert conversion["frameworks"]["isaac_sim"]["status"] == "planned_requires_conversion"
    assert conversion["frameworks"]["mujoco"]["status"] == "planned_requires_conversion"
    assert conversion["frameworks"]["pybullet"]["status"] == "planned_requires_conversion"
    assert conversion["frameworks"]["newton"]["status"] == "planned_requires_conversion"
    assert simulator_execution["overall_status"] == "blocked"
    assert {
        record["framework"]: record["status"]
        for record in simulator_execution["simulator_results"]
    } == {
        "isaac_sim": "blocked",
        "mujoco": "blocked",
        "pybullet": "blocked",
        "newton": "blocked",
    }
    assert all(
        record["reason"] == "approval_required"
        for record in simulator_execution["simulator_results"]
    )
    assert training["status"] == "blocked"
    assert training["reason"] == "approval_required"
    assert training["runner"] == "blueprint_pipeline.synthesis.cosmos_lora_training.run_cosmos_lora_training"
    assert proof_boundary["simulator_execution_proven"] is False
    assert proof_boundary["robot_readiness_proven"] is False
    assert proof_boundary["training_proof"]["training_completed"] is False
    assert proof_boundary["public_claim_upgrade_allowed"] is False
    assert run_manifest["live_provider_calls_performed"] is False
    assert run_manifest["remote_asset_downloads_performed"] is False
    assert run_manifest["simulators_run"] is False
    assert run_manifest["gpu_training_run"] is False
    assert agent_ledger["adapter"] == "fake"
    assert agent_ledger["decisions"][0]["decision"] == "plan_next_actions"


def test_missing_simulator_dependency_produces_blocked_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")

    build_simulation_automation(
        capture_root=capture_root,
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": "definitely-missing-blueprint-mujoco"},
    )

    result_path = (
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulators"
        / "mujoco_result.json"
    )
    mujoco_result = _read_json(result_path)

    assert mujoco_result["framework"] == "mujoco"
    assert mujoco_result["status"] == "blocked"
    assert mujoco_result["reason"] == "missing_dependency"
    assert mujoco_result["blocked_manifest"] == str(result_path)
    assert mujoco_result["command"] == ["definitely-missing-blueprint-mujoco"]


def test_fake_agent_adapter_can_plan_and_diagnose_without_network(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_worldlabs_and_marble_artifacts(capture_root)
    adapter = FakeSimulationAutomationAgentAdapter()

    build_simulation_automation(capture_root=capture_root, agent_adapter=adapter)

    ledger = _read_json(
        capture_root / "pipeline" / "simulation_automation" / "agent_decision_ledger.json"
    )
    assert ledger["adapter"] == "fake"
    assert ledger["network_required"] is False
    assert ledger["decisions"][0]["summary"] == (
        "Use deterministic manifests; keep simulator and training execution blocked until explicit approvals and dependencies exist."
    )
    assert ledger["diagnostics"][0]["status"] == "blocked"
    assert "approval_required" in ledger["diagnostics"][0]["blockers"]
