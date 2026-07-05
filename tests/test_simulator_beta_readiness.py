from __future__ import annotations

import json
from pathlib import Path

import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.simulator_beta_readiness import (
    SIMULATOR_BETA_READINESS_SCHEMA_VERSION,
    _frame_evidence,
    _handoff_gate,
    _image_nonblank,
    _mujoco_gate,
    _official_policy_gate,
    _runpod_gate,
    _select_runpod_live_execution_proof,
    _webapp_gate,
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


def _seed_handoff(capture_root: Path) -> None:
    handoff_dir = (
        capture_root
        / "pipeline"
        / "sim_only_beta_rehearsal"
        / "official_unitree_g1_policy_execution"
        / "robot_team_handoff"
    )
    artifacts = {
        "robot_team_timeseries": handoff_dir / "robot_team_timeseries.jsonl",
        "policy_execution_trace_enriched": handoff_dir / "policy_execution_trace_enriched.jsonl",
        "sensor_stream_manifest": handoff_dir / "sensor_stream_manifest.json",
        "camera_manifest": handoff_dir / "camera_manifest.json",
        "contact_manifest": handoff_dir / "contact_manifest.json",
        "robot_pov_manifest": handoff_dir / "robot_pov_manifest.json",
        "rendered_motion_manifest": handoff_dir / "rendered_motion_manifest.json",
    }
    artifacts["robot_team_timeseries"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["robot_team_timeseries"].write_text('{"qpos":[0],"qvel":[0]}\n', encoding="utf-8")
    artifacts["policy_execution_trace_enriched"].write_text(
        '{"qpos":[0],"qvel":[0]}\n',
        encoding="utf-8",
    )
    for key, path in artifacts.items():
        if path.suffix == ".json":
            _write_json(path, {"status": "complete", "artifact": key})
    _write_json(
        handoff_dir / "robot_team_handoff_manifest.json",
        {
            "status": "complete",
            "robot_team_handoff_dataset_status": "complete",
            "simulated_robot_pov_status": "complete",
            "high_quality_video_status": "complete",
            "training_grade_policy_rollout_proven": True,
            "walking_motion_proven": True,
            "steps": 2000,
            "control_updates": 200,
            "blockers": [],
            "artifacts": {key: str(path) for key, path in artifacts.items()},
        },
    )


def _seed_ready_simulator_beta(capture_root: Path, *, include_handoff: bool = True) -> None:
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
                "command_xyz": [0.5, 0.0, 0.0],
                "final_base_position_xyz": [1.75, -0.08, 0.77],
            },
            "proof_boundary": {
                "non_default_policy_execution_trace_proven": True,
                "policy_metrics_tied_to_scenario_variation": True,
            },
        },
    )
    if include_handoff:
        _seed_handoff(capture_root)

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


def _seed_ready_isaac_simulator_output(capture_root: Path) -> Path:
    isaac_dir = (
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "isaac_g1_simulator_command"
    )
    video_path = isaac_dir / "realistic_videos" / "episode-1__head_pov.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"mp4-placeholder")
    artifact_paths = {
        "normalized_attempt_trace.json": isaac_dir / "normalized_attempt_trace.json",
        "failure_labels.json": isaac_dir / "failure_labels.json",
        "realistic_video_manifest.json": isaac_dir / "realistic_video_manifest.json",
        "g1_locomotion_trace.jsonl": isaac_dir / "g1_locomotion_trace.jsonl",
        "collision_contact_report.json": isaac_dir / "collision_contact_report.json",
        "batch_closure_manifest": isaac_dir / "isaac_batch_closure_manifest.json",
        "job_run_manifest.json": isaac_dir / "job_run_manifest.json",
        "artifact_manifest": isaac_dir / "artifact_manifest.json",
    }
    for key, path in artifact_paths.items():
        if path.suffix == ".jsonl":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"status":"completed"}\n', encoding="utf-8")
        else:
            _write_json(path, {"status": "complete", "artifact": key})
    video_manifest = {
        "status": "completed",
        "video_count": 1,
        "expected_video_count": 1,
        "videos": [
            {
                "episode_id": "episode-1",
                "camera_id": "head_pov",
                "path": str(video_path),
                "status": "completed",
            }
        ],
    }
    batch_closure = {
        "status": "completed_with_robot_team_grade_blockers",
        "machine_trace_package_complete": True,
        "robot_team_grade_package_complete": False,
    }
    artifact_manifest = {
        "status": "complete",
        "files": {},
    }
    _write_json(artifact_paths["realistic_video_manifest.json"], video_manifest)
    _write_json(artifact_paths["batch_closure_manifest"], batch_closure)
    _write_json(artifact_paths["artifact_manifest"], artifact_manifest)
    output_path = isaac_dir / "isaac_g1_simulator_output.json"
    _write_json(
        output_path,
        {
            "status": "completed",
            "simulator_backend": "isaac_sim",
            "simulator_version": "6.0.0",
            "simulator_execution_proven": True,
            "isaac_sim_execution_proven": True,
            "unitree_g1_asset_spawned": True,
            "scenario_eval_run_count": 1,
            "attempt_count": 1,
            "attempt_count_matches_matrix_count": True,
            "scenario_eval_run_coverage_complete": True,
            "realistic_video_manifest": video_manifest,
            "batch_closure_manifest": batch_closure,
            "artifact_manifest": artifact_manifest,
            "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
        },
    )
    return output_path


def test_simulator_beta_readiness_marks_physical_gates_out_of_scope(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root)

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["schema_version"] == SIMULATOR_BETA_READINESS_SCHEMA_VERSION
    assert manifest["status"] == "ready_for_simulator_beta"
    assert manifest["ready_for_simulator_beta"] is True
    assert manifest["blocking_gate_ids"] == []
    assert manifest["out_of_scope_gates"]["generated_world_rank_fidelity"] == (
        "out_of_scope_for_simulator_beta"
    )
    gates = manifest["gates"]
    assert gates["site_capture_mujoco_g1_run"]["proven"] is True
    assert gates["official_unitree_g1_policy_execution"]["proven"] is True
    assert gates["production_runpod_worker_execution"]["proven"] is True
    assert gates["customer_website_to_pipeline_request"]["proven"] is True
    assert gates["official_policy_robot_team_handoff_dataset"]["proven"] is True
    assert manifest["claim_boundary"]["generated_world_rank_fidelity_claimed"] is False
    assert manifest["claim_boundary"]["walking_motion_proven"] is True
    assert manifest["claim_boundary"]["training_grade_policy_rollout_proven"] is True
    assert manifest["data_gate_ids"] == []
    persisted = _read_json(Path(manifest["artifacts"]["manifest"]))
    assert persisted["ready_for_simulator_beta"] is True


def test_isaac_proof_package_can_satisfy_simulator_beta_without_mujoco(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root)
    _seed_ready_isaac_simulator_output(capture_root)
    (
        capture_root
        / "pipeline"
        / "sim_only_beta_rehearsal"
        / "mujoco_g1_command"
        / "mujoco_g1_simulator_output.json"
    ).unlink()

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["status"] == "ready_for_simulator_beta"
    assert manifest["blocking_gate_ids"] == []
    assert manifest["default_robot"]["simulator_backend"] == "isaac_sim"
    assert manifest["gates"]["site_capture_mujoco_g1_run"]["proven"] is False
    assert manifest["gates"]["site_capture_isaac_g1_run"]["proven"] is True
    assert manifest["gates"]["site_capture_simulator_g1_run"]["proven"] is True
    assert manifest["claim_boundary"]["mujoco_proof_counted_as_isaac_proof"] is False


def test_simulator_beta_readiness_defers_to_release_gate_when_present(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    release_gate_path = (
        capture_root
        / "pipeline"
        / "live_pipeline_control_plane"
        / "sim_only_beta_release_gate_report.json"
    )
    _write_json(
        release_gate_path,
        {
            "schema_version": "blueprint.sim_only_beta_release_gate_report.v1",
            "status": "passed",
            "ready_for_beta_release": True,
            "blockers": [],
        },
    )

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["status"] == "ready_for_simulator_beta"
    assert manifest["ready_for_simulator_beta"] is True
    assert manifest["blocking_gate_ids"] == []
    assert manifest["release_authority"]["ready_for_beta_release"] is True
    assert manifest["release_authority"][
        "legacy_provider_rehearsal_gates_are_advisory"
    ] is True
    assert "production_runpod_worker_execution" in manifest[
        "legacy_provider_rehearsal_blocking_gate_ids"
    ]
    assert manifest["claim_boundary"]["sim_only_release_gate_authoritative"] is True


def test_simulator_beta_readiness_does_not_promote_training_grade_without_handoff(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root, include_handoff=False)

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["status"] == "ready_for_simulator_beta"
    assert manifest["ready_for_simulator_beta"] is True
    assert manifest["blocking_gate_ids"] == []
    assert manifest["data_gate_ids"] == ["official_policy_robot_team_handoff_dataset"]
    assert manifest["claim_boundary"]["walking_motion_proven"] is True
    assert manifest["claim_boundary"]["training_grade_policy_rollout_proven"] is False


def test_smoke_command_cannot_satisfy_walking_motion_without_official_policy(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    _seed_ready_simulator_beta(capture_root, include_handoff=False)
    (
        capture_root
        / "pipeline"
        / "sim_only_beta_rehearsal"
        / "official_unitree_g1_policy_execution"
        / "official_unitree_g1_policy_execution_manifest.json"
    ).unlink()

    manifest = build_simulator_beta_readiness(capture_root=capture_root)

    assert manifest["status"] == "blocked_simulator_beta"
    assert "official_unitree_g1_policy_execution" in manifest["blocking_gate_ids"]
    assert manifest["claim_boundary"]["walking_motion_proven"] is False
    assert manifest["claim_boundary"]["training_grade_policy_rollout_proven"] is False


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


def test_simulator_beta_readiness_frame_and_selector_edges(tmp_path: Path) -> None:
    assert _image_nonblank(tmp_path / "missing.png") is False

    blank = tmp_path / "blank.png"
    blank.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(blank)
    evidence, blockers = _frame_evidence(["", tmp_path / "missing-frame.png", blank])

    assert evidence == [str(blank)]
    assert f"missing_sim_frame:{tmp_path / 'missing-frame.png'}" in blockers
    assert f"blank_or_unreadable_sim_frame:{blank}" in blockers

    capture_root = tmp_path / "capture"
    older = (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-1"
        / "runpod_live_execution_proof.json"
    )
    newer = (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-1"
        / "runpod_live_execution_proof.ready.json"
    )
    _write_json(
        older,
        {
            "production_runpod_worker_execution_proven": False,
            "simulator_execution_proven": True,
            "shutdown_or_termination_proof": True,
            "blockers": ["older"],
        },
    )
    _write_json(
        newer,
        {
            "production_runpod_worker_execution_proven": True,
            "simulator_execution_proven": True,
            "shutdown_or_termination_proof": True,
            "blockers": [],
        },
    )

    selected_path, selected_payload = _select_runpod_live_execution_proof(capture_root)

    assert selected_path == newer
    assert selected_payload is not None
    assert selected_payload["production_runpod_worker_execution_proven"] is True


def test_simulator_beta_readiness_negative_gate_branches(tmp_path: Path) -> None:
    bad_mujoco = _mujoco_gate(
        tmp_path / "mujoco.json",
        {
            "status": "failed",
            "scene_loaded": False,
            "unitree_g1_asset_spawned": False,
            "mujoco_g1_asset_execution_proven": False,
            "default_sim_policy_execution_proven": False,
            "sim_robot_pov_evidence_proven": False,
            "attempts": [{"success": False, "metrics": {}}],
            "artifact_paths": {},
        },
    )
    assert "mujoco_status:failed" in bad_mujoco["blockers"]
    assert "scene_loaded_not_true" in bad_mujoco["blockers"]
    assert "mujoco_attempt_not_successful" in bad_mujoco["blockers"]
    assert "mujoco_simulated_step_count_missing" in bad_mujoco["blockers"]

    missing_trace_policy = _official_policy_gate(
        tmp_path / "policy.json",
        {
            "status": "failed",
            "proof_boundary": {},
            "execution": {"trace_path": str(tmp_path / "missing-trace.jsonl")},
            "metrics": {
                "finite_state": False,
                "finite_actions": False,
                "final_base_position_xyz": ["bad"],
                "command_xyz": [],
            },
        },
    )
    assert "official_policy_status:failed" in missing_trace_policy["blockers"]
    assert "non_default_policy_execution_trace_not_proven" in missing_trace_policy["blockers"]
    assert "policy_metrics_not_tied_to_scenario" in missing_trace_policy["blockers"]
    assert "official_policy_finite_state_not_true" in missing_trace_policy["blockers"]
    assert "official_policy_finite_actions_not_true" in missing_trace_policy["blockers"]
    assert "official_policy_base_displacement_missing_or_too_small" in missing_trace_policy["blockers"]
    assert "official_policy_command_profile_missing" in missing_trace_policy["blockers"]
    assert any(str(blocker).startswith("missing_policy_trace:") for blocker in missing_trace_policy["blockers"])

    empty_trace = tmp_path / "empty-trace.jsonl"
    empty_trace.write_text("", encoding="utf-8")
    empty_trace_policy = _official_policy_gate(
        tmp_path / "policy.json",
        {
            "status": "completed",
            "proof_boundary": {
                "non_default_policy_execution_trace_proven": True,
                "policy_metrics_tied_to_scenario_variation": True,
            },
            "execution": {"trace_path": str(empty_trace)},
            "metrics": {
                "finite_state": True,
                "finite_actions": True,
                "final_base_position_xyz": [1.0, 0.0, 0.7],
                "command_xyz": [0.4, 0.0, 0.0],
            },
        },
    )
    assert "empty_policy_execution_trace" in empty_trace_policy["blockers"]

    handoff = _handoff_gate(
        tmp_path / "handoff.json",
        {
            "status": "blocked",
            "robot_team_handoff_dataset_status": "blocked",
            "simulated_robot_pov_status": "blocked",
            "high_quality_video_status": "blocked",
            "training_grade_policy_rollout_proven": False,
            "steps": 0,
            "control_updates": 0,
            "blockers": ["manual_blocker"],
            "artifacts": {},
        },
    )
    assert "handoff_status:blocked" in handoff["blockers"]
    assert "robot_team_handoff_dataset_not_complete" in handoff["blockers"]
    assert "simulated_robot_pov_not_complete" in handoff["blockers"]
    assert "high_quality_video_not_complete" in handoff["blockers"]
    assert "training_grade_policy_rollout_not_proven_by_handoff_gate" in handoff["blockers"]
    assert "handoff_steps_missing" in handoff["blockers"]
    assert "handoff_control_updates_missing" in handoff["blockers"]

    runpod = _runpod_gate(
        tmp_path / "runpod.json",
        {
            "production_runpod_worker_execution_proven": False,
            "simulator_execution_proven": False,
            "shutdown_or_termination_proof": False,
            "blockers": ["provider_blocked"],
        },
    )
    assert "production_runpod_worker_execution_not_proven" in runpod["blockers"]
    assert "runpod_simulator_execution_not_proven" in runpod["blockers"]
    assert "runpod_shutdown_proof_missing" in runpod["blockers"]

    webapp = _webapp_gate(
        tmp_path / "webapp.json",
        {
            "status": "failed",
            "proof_boundary": {},
            "webapp_route": {},
            "pipeline_forward": {},
            "pipeline_intake": {"input_blockers": ["intake_blocker"]},
        },
    )
    assert "webapp_route_status:failed" in webapp["blockers"]
    assert "production_live_webapp_forwarding_not_proven" in webapp["blockers"]
    assert "production_webapp_deployment_not_proven" in webapp["blockers"]
    assert "pipeline_forward_not_accepted" in webapp["blockers"]
    assert "pipeline_intake_not_accepted" in webapp["blockers"]
    assert "intake_blocker" in webapp["blockers"]
