from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline.agent_operator_runtime import LIVE_AGENTS_SDK_ENV
from blueprint_pipeline.sim_only_provider_execution_planner import (
    LIVE_AGENT_PLANNER_ENV,
    build_sim_only_provider_execution_layer,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    image.putpixel((0, 0), (255, 255, 255))
    image.save(path)


def _seed_mujoco_beta_artifacts(capture_root: Path) -> None:
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
    webapp_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    _write_json(
        webapp_dir / "webapp_route_forwarding_proof.ready.json",
        {
            "status": "forwarded_to_pipeline_intake",
            "webapp_route": {
                "http_status": 202,
                "full_production_webapp_deployment_proven": True,
            },
            "pipeline_forward": {"accepted": True},
            "pipeline_intake": {"accepted": True, "input_blockers": []},
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": True,
            },
        },
    )


def _seed_ready_job(capture_root: Path, *, job_id: str = "job-mujoco-runpod") -> Path:
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / job_id
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "operation": "evaluate_only",
            "simulator_preference": "mujoco",
            "requested_tasks": [{"task_id": "walk_to_target"}],
            "robot_profile": {"robot_profile_id": "unitree_g1"},
            "execution_request": {
                "gpu_allocation": {
                    "max_budget_usd": 3.0,
                    "hard_timeout_seconds": 120,
                    "warm_pool_policy": {"enabled": False},
                }
            },
        },
    )
    _write_json(
        job_dir / "scheduler_decision.json",
        {
            "schema_version": "robot_eval_execution_scheduler_decision.v1",
            "status": "awaiting_explicit_gpu_and_simulator_gates",
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "worker_launch_plan.json",
        {
            "schema_version": "robot_eval_worker_launch_plan.v1",
            "status": "awaiting_explicit_provider_gate",
            "provider": "runpod",
            "simulator": "mujoco",
            "blockers": [],
            "launch_mode": {
                "max_active_workers": 1,
                "hard_timeout_seconds": 120,
                "idle_timeout_seconds": 60,
                "external_watchdog_ttl_seconds": 180,
            },
        },
    )
    _write_json(
        job_dir / "worker_manifest.json",
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "status": "ready_for_worker_upload",
            "worker_manifest_uri": "r2://blueprint-artifacts/jobs/job-mujoco-runpod/worker.json",
            "worker_manifest_uri_fetchable_by_provider": True,
            "capture_root_bundle_uri": "r2://blueprint-artifacts/jobs/job-mujoco-runpod/root.zip",
            "input_bundle": {"capture_root_bundle_uri_fetchable_by_provider": True},
            "artifact_output_uri": "r2://blueprint-artifacts/jobs/job-mujoco-runpod/artifacts",
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "gpu_provider_launch_request.json",
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": job_id,
            "provider": "runpod",
            "status": "request_manifest_ready",
            "provider_request_shape": {
                "image": {
                    "configured_image_ref": (
                        "registry.example/blueprint/mujoco-eval-worker:2026-06-12"
                    ),
                    "configured_image_ref_is_versioned": True,
                    "configured_image_ref_fetchable_by_provider": True,
                },
                "inputs": {
                    "manifest_uri": "r2://blueprint-artifacts/jobs/job-mujoco-runpod/worker.json",
                    "manifest_uri_fetchable_by_provider": True,
                    "capture_root_bundle_uri": (
                        "r2://blueprint-artifacts/jobs/job-mujoco-runpod/root.zip"
                    ),
                    "capture_root_bundle_uri_fetchable_by_provider": True,
                    "artifact_output_uri": (
                        "r2://blueprint-artifacts/jobs/job-mujoco-runpod/artifacts"
                    ),
                },
                "gpu": {"preferred_gpu_class": "cpu_or_low_cost_gpu_when_rendering"},
                "limits": {
                    "max_active_workers": 1,
                    "hard_timeout_seconds": 120,
                    "idle_timeout_seconds": 60,
                    "external_watchdog_ttl_seconds": 180,
                    "requested_budget_usd": 3.0,
                },
            },
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "gpu_cost_control_ledger.json",
        {
            "schema_version": "robot_eval_gpu_cost_control_ledger.v1",
            "status": "ready_for_explicit_provider_launcher",
            "gpu_time": {"actual_gpu_seconds": None},
        },
    )
    return job_dir


def test_mujoco_runpod_plan_uses_low_cost_scale_to_zero_path(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["status"] == "ready_for_provider_launch"
    assert plan["preflight"]["status"] == "passed"  # type: ignore[index]
    assert plan["provider_gpu_priority_fallback_list"][0] == "NVIDIA L4"
    assert plan["warm_pool_policy"]["decision"] == "scale_to_zero_on_demand"  # type: ignore[index]
    assert plan["warm_pool_policy"]["active_worker_target"] == 0  # type: ignore[index]
    assert plan["cheapest_sufficient_path"][  # type: ignore[index]
        "mujoco_non_render_or_light_render_avoids_isaac_class_gpus"
    ] is True
    assert plan["persistent_cache_paths"]["mujoco_assets"].endswith("/mujoco_assets")  # type: ignore[index]
    assert (job_dir / "sim_only_provider_execution_plan.json").is_file()


def test_warm_worker_requires_latency_policy_to_justify_idle_cost(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    request = _read_json(job_dir / "job_request.json")
    request["execution_request"]["gpu_allocation"]["warm_pool_policy"] = {  # type: ignore[index]
        "enabled": True,
        "latency_slo_seconds": 30,
        "estimated_idle_cost_usd_per_hour": 0.2,
        "max_idle_cost_usd_per_hour": 1.0,
    }
    _write_json(job_dir / "job_request.json", request)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["warm_pool_policy"]["decision"] == "warm_active_worker"  # type: ignore[index]
    assert plan["warm_pool_policy"]["active_worker_target"] == 1  # type: ignore[index]


def test_missing_provider_inputs_block_before_spend(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    provider = _read_json(job_dir / "gpu_provider_launch_request.json")
    provider["provider_request_shape"]["inputs"]["manifest_uri"] = ""  # type: ignore[index]
    provider["provider_request_shape"]["inputs"]["capture_root_bundle_uri"] = ""  # type: ignore[index]
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["status"] == "blocked_before_spend"
    assert "missing_worker_manifest_uri" in plan["blockers"]
    assert "missing_capture_root_bundle_uri" in plan["blockers"]
    assert plan["cost_ledger"]["status"] == "blocked-before-allocation"  # type: ignore[index]


def test_missing_image_ref_blocks_before_spend(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    provider = _read_json(job_dir / "gpu_provider_launch_request.json")
    provider["provider_request_shape"]["image"]["configured_image_ref"] = ""  # type: ignore[index]
    provider["provider_request_shape"]["image"]["configured_image_ref_is_versioned"] = False  # type: ignore[index]
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["status"] == "blocked_before_spend"
    assert "missing_prebuilt_worker_image_ref" in plan["blockers"]
    assert "prebuilt_worker_image_ref_not_versioned" in plan["blockers"]


def test_unfetchable_image_ref_blocks_before_spend(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    provider = _read_json(job_dir / "gpu_provider_launch_request.json")
    provider["provider_request_shape"]["image"][  # type: ignore[index]
        "configured_image_ref_fetchable_by_provider"
    ] = False
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["status"] == "blocked_before_spend"
    assert "prebuilt_worker_image_ref_not_provider_fetchable" in plan["blockers"]
    assert plan["preflight"]["image_ref_provider_fetchable"] is False  # type: ignore[index]


def test_budget_timeout_and_watchdog_flow_into_cost_ledger(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    ledger = plan["cost_ledger"]
    assert ledger["max_budget_per_job_usd"] == 3.0  # type: ignore[index]
    assert ledger["hard_timeout_seconds"] == 120  # type: ignore[index]
    assert ledger["watchdog_timeout_seconds"] == 180  # type: ignore[index]
    assert ledger["supported_states"] == [  # type: ignore[index]
        "blocked-before-allocation",
        "running",
        "completed",
        "failed",
        "stopped",
    ]


def test_runpod_shutdown_proof_updates_simulator_beta_readiness(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    _seed_mujoco_beta_artifacts(capture_root)
    runtime_manifest = job_dir / "worker_runtime_manifest.json"
    _write_json(
        runtime_manifest,
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "status": "completed",
        },
    )
    _write_json(
        job_dir / "runpod_live_execution_proof.json",
        {
            "status": "runpod_live_proof_collected",
            "production_runpod_worker_execution_proven": True,
            "simulator_execution_proven": True,
            "shutdown_or_termination_proof": True,
            "active_pod_count_before": 1,
            "active_pod_count_after": 0,
            "runtime_manifest_path": str(runtime_manifest),
            "blockers": [],
        },
    )

    plan = build_sim_only_provider_execution_layer(
        capture_root=capture_root,
        job_dir=job_dir,
        update_simulator_beta_readiness=True,
    )

    assert plan["runtime_manifest"]["status"] == "completed"  # type: ignore[index]
    assert plan["runtime_manifest"]["active_pod_count_after"] == 0  # type: ignore[index]
    assert plan["runtime_manifest"]["shutdown_or_termination_proof"] is True  # type: ignore[index]
    assert plan["cost_ledger"]["status"] == "stopped"  # type: ignore[index]
    readiness = plan["simulator_beta_readiness_update"]
    assert readiness["status"] == "ready_for_simulator_beta"  # type: ignore[index]
    assert readiness["ready_for_simulator_beta"] is True  # type: ignore[index]


def test_agent_planner_is_advisory_and_redacts_secret_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    monkeypatch.setenv(LIVE_AGENTS_SDK_ENV, "true")
    monkeypatch.setenv(LIVE_AGENT_PLANNER_ENV, "true")
    provider = _read_json(job_dir / "gpu_provider_launch_request.json")
    provider["provider_request_shape"]["inputs"]["artifact_output_uri"] = (  # type: ignore[index]
        "r2://blueprint-artifacts/jobs/job-mujoco-runpod/artifacts?token=secret-runpod-key"
    )
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    def fake_executor(prompt: str, context: dict[str, object]) -> dict[str, object]:
        assert "redacted_plan_context" in prompt
        assert "job-mujoco-runpod" in prompt
        assert "secret-runpod-key" not in prompt
        assert "secret-runpod-key" not in json.dumps(context)
        return {
            "final_output": "Use on-demand L4 after preflight passes.",
            "commands_chosen": ["validate_preflight_then_launch_runpod"],
        }

    plan = build_sim_only_provider_execution_layer(
        capture_root=capture_root,
        job_dir=job_dir,
        allow_live_agent_planner=True,
        agent_executor=fake_executor,
    )

    assert plan["agent_planner"]["status"] == "operator_completed"  # type: ignore[index]
    assert plan["proof_booleans_mutable_by_agent"] is False
    persisted = (job_dir / "sim_only_provider_execution_plan.json").read_text(
        encoding="utf-8"
    )
    assert "secret-runpod-key" not in persisted
