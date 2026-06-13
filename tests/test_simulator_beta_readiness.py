from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline.simulator_beta_readiness import (
    SIMULATOR_BETA_READINESS_SCHEMA_VERSION,
    build_simulator_beta_readiness,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 12), color=(0, 0, 0))
    image.putpixel((0, 0), (255, 255, 255))
    image.save(path)


def _seed_ready_simulator_beta(capture_root: Path) -> None:
    mujoco_dir = capture_root / "pipeline" / "sim_only_beta_rehearsal" / "mujoco_g1_command"
    overview = mujoco_dir / "frames" / "overview_0000.png"
    pov = mujoco_dir / "frames" / "sim_robot_follow_pov_0000.png"
    _write_frame(overview)
    _write_frame(pov)
    scene_trace = mujoco_dir / "scene_load_trace.json"
    spawn_trace = mujoco_dir / "spawn_trace.json"
    policy_trace = mujoco_dir / "policy_execution_trace.json"
    pov_manifest = mujoco_dir / "sim_robot_pov_evidence_manifest.json"
    artifact_manifest = mujoco_dir / "artifact_manifest.json"
    for path in (scene_trace, spawn_trace, policy_trace, pov_manifest, artifact_manifest):
        _write_json(path, {"status": "complete"})
    _write_json(
        mujoco_dir / "mujoco_g1_simulator_output.json",
        {
            "status": "completed",
            "simulator_backend": "mujoco",
            "mujoco_version": "3.9.0",
            "scene_loaded": True,
            "unitree_g1_asset_spawned": True,
            "mujoco_g1_asset_execution_proven": True,
            "default_sim_policy_execution_proven": True,
            "sim_robot_pov_evidence_proven": True,
            "attempts": [
                {
                    "status": "completed",
                    "success": True,
                    "metrics": {"simulated_step_count": 240},
                }
            ],
            "artifact_paths": {
                "scene_trace": str(scene_trace),
                "spawn_trace": str(spawn_trace),
                "policy_trace": str(policy_trace),
                "sim_robot_pov_evidence": str(pov_manifest),
                "artifact_manifest": str(artifact_manifest),
                "frames": [str(overview), str(pov)],
            },
        },
    )

    policy_dir = (
        capture_root
        / "pipeline"
        / "sim_only_beta_rehearsal"
        / "official_unitree_g1_policy_execution"
    )
    trace_path = policy_dir / "policy_execution_trace.jsonl"
    metrics_path = policy_dir / "policy_metrics.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"step": 0}\n', encoding="utf-8")
    _write_json(metrics_path, {"status": "completed"})
    _write_json(
        policy_dir / "official_unitree_g1_policy_execution_manifest.json",
        {
            "status": "completed",
            "policy_id": "unitree_rl_gym_g1_pretrain_motion",
            "source_repository": {"pinned_commit": "abc123"},
            "execution": {
                "trace_path": str(trace_path),
                "metrics_path": str(metrics_path),
            },
            "metrics": {
                "finite_state": True,
                "finite_actions": True,
                "sim_time_s": 4.0,
                "steps": 2000,
                "control_updates": 200,
            },
            "proof_boundary": {
                "non_default_policy_execution_trace_proven": True,
                "policy_metrics_tied_to_scenario_variation": True,
            },
        },
    )

    signed_dir = capture_root / "pipeline" / "g1_controlled_proof_setup" / "signed_runpod_io"
    runtime_manifest = signed_dir / "worker_runtime_manifest.json"
    _write_json(runtime_manifest, {"status": "completed"})
    _write_json(
        signed_dir / "runpod_live_execution_proof.ready.json",
        {
            "status": "runpod_live_proof_collected",
            "production_runpod_worker_execution_proven": True,
            "simulator_execution_proven": True,
            "shutdown_or_termination_proof": True,
            "api_call_performed": True,
            "active_pod_count_before": 0,
            "active_pod_count_after": 0,
            "runtime_manifest_path": str(runtime_manifest),
            "blockers": [],
        },
    )

    webapp_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    _write_json(
        webapp_dir / "webapp_route_forwarding_proof.ready.json",
        {
            "status": "forwarded_to_pipeline_intake",
            "generated_at": "2026-06-13T00:00:00+00:00",
            "webapp_route": {
                "http_status": 202,
                "full_production_webapp_deployment_proven": True,
            },
            "pipeline_forward": {
                "accepted": True,
                "pipeline_status": "staged_for_control_plane",
            },
            "pipeline_intake": {
                "accepted": True,
                "status": "staged_for_control_plane",
                "input_blockers": [],
            },
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": True,
            },
        },
    )


def test_simulator_beta_readiness_marks_physical_gates_out_of_scope(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root)

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["schema_version"] == SIMULATOR_BETA_READINESS_SCHEMA_VERSION
    assert manifest["status"] == "ready_for_simulator_beta"
    assert manifest["ready_for_simulator_beta"] is True
    assert manifest["blocking_gate_ids"] == []
    assert manifest["out_of_scope_gates"]["physical_robot_readiness"] == (
        "out_of_scope_for_simulator_beta"
    )
    gates = manifest["gates"]
    assert gates["site_capture_mujoco_g1_run"]["proven"] is True
    assert gates["official_unitree_g1_policy_execution"]["proven"] is True
    assert gates["production_runpod_worker_execution"]["proven"] is True
    assert gates["customer_website_to_pipeline_request"]["proven"] is True
    assert manifest["claim_boundary"]["physical_robot_readiness_claimed"] is False
    persisted = _read_json(Path(manifest["artifacts"]["manifest"]))
    assert persisted["ready_for_simulator_beta"] is True


def test_simulator_beta_readiness_blocks_missing_mujoco_output(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root)
    (
        capture_root
        / "pipeline"
        / "sim_only_beta_rehearsal"
        / "mujoco_g1_command"
        / "mujoco_g1_simulator_output.json"
    ).unlink()

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["status"] == "blocked_simulator_beta"
    assert manifest["ready_for_simulator_beta"] is False
    assert manifest["blocking_gate_ids"] == ["site_capture_mujoco_g1_run"]
    assert (
        manifest["gates"]["site_capture_mujoco_g1_run"]["status"]
        == "missing_mujoco_g1_simulator_output"
    )


def test_simulator_beta_readiness_cli(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root)

    assert main(["--capture-root", str(capture_root)]) == 0

    manifest = _read_json(
        capture_root
        / "pipeline"
        / "sim_only_beta_rehearsal"
        / "simulator_beta_readiness"
        / "simulator_beta_readiness_manifest.json"
    )
    assert manifest["status"] == "ready_for_simulator_beta"
