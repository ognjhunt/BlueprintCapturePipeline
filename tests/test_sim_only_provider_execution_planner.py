from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.agent_operator_runtime import LIVE_AGENTS_SDK_ENV
from blueprint_pipeline import sim_only_provider_execution_planner as planner
from blueprint_pipeline.provider_closure_audit import (
    PROVIDER_CLOSURE_AUDIT_REPORT_NAME,
    audit_provider_closure,
)
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
                "command_xyz": [0.5, 0.0, 0.0],
                "final_base_position_xyz": [1.75, -0.08, 0.77],
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
        job_dir / "gpu_startup_pipeline_plan.json",
        {
            "schema_version": "robot_eval_gpu_startup_pipeline_plan.v1",
            "job_id": job_id,
            "status": "startup_pipeline_ready",
            "provider_selection_owner": "BlueprintCapturePipeline",
            "selected_provider": "runpod",
            "selected_provider_tier": "managed_secure_cloud_preferred",
            "selected_provider_is_marketplace": False,
            "managed_provider_policy": {
                "managed_provider_priority": [
                    "runpod_secure_cloud",
                    "lambda_cloud",
                    "aws_g6",
                    "coreweave",
                ],
                "provider_api_priority": ["runpod", "gcp", "vast"],
            },
            "marketplace_policy": {
                "customer_job_marketplace_default": (
                    "avoid_unless_explicit_strict_preflight_canary"
                ),
                "selected_provider_is_marketplace": False,
                "marketplace_quarantine_required": False,
            },
            "preflight_canary_policy": {
                "required_before_customer_eval": True,
                "customer_eval_waits_for_canary": True,
                "block_scene_load_until_preflight_passes": True,
            },
            "same_sku_burst_policy": {
                "burst_workers_must_use_same_image_ref": True,
                "burst_workers_must_use_same_gpu_family": True,
                "provider_worker_selection_disallows_random_hosts": True,
            },
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
                "startup_pipeline": {
                    "plan_path": "gpu_startup_pipeline_plan.json",
                    "status": "startup_pipeline_ready",
                    "selected_provider_tier": "managed_secure_cloud_preferred",
                    "selected_provider_is_marketplace": False,
                    "blockers": [],
                },
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
    assert plan["provider_priority"] == ["runpod", "gcp", "vast"]
    assert plan["gpu_startup_pipeline_plan_path"] == "gpu_startup_pipeline_plan.json"
    assert plan["managed_provider_policy"]["managed_provider_priority"] == [  # type: ignore[index]
        "runpod_secure_cloud",
        "lambda_cloud",
        "aws_g6",
        "coreweave",
    ]
    assert plan["marketplace_policy"][  # type: ignore[index]
        "customer_job_marketplace_default"
    ] == "avoid_unless_explicit_strict_preflight_canary"
    assert plan["preflight_canary_policy"][  # type: ignore[index]
        "customer_eval_waits_for_canary"
    ] is True
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


def test_planner_helper_edges_and_policy_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert planner._string_list("one") == ["one"]
    assert planner._string_list("") == []
    assert planner._number("not-a-number", 2.5) == 2.5
    assert planner._redact_uri("https://example.test/path?keep=1&token=secret") == (
        "https://example.test/path?keep=1&token=%3Credacted%3Atoken%3E"
    )
    assert planner._redact_runtime_value("secret", "api_key") == "<redacted:api_key>"
    assert planner._redact_runtime_value("", "api_key") == ""
    with pytest.raises(ValueError, match="job_id or job_dir"):
        planner._job_dir(tmp_path, None, None)
    assert planner._job_dir(tmp_path, "job-1", None) == (
        tmp_path / "pipeline" / "robot_eval_jobs" / "job-1"
    ).resolve()

    monkeypatch.setenv("BLUEPRINT_PROVIDER_CACHE_ROOT", "/cache")
    assert planner._cache_paths("isaac_sim") == {
        "scene_assets": "/cache/scene_assets",
        "policy_files": "/cache/policy_files",
        "worker_deps": "/cache/worker_deps",
    }
    assert planner._provider_gpu_priority(
        "mujoco",
        {"gpu": {"provider_gpu_priority": [" NVIDIA L4 ", "NVIDIA RTX A4000"]}},
    ) == ["NVIDIA L4", "NVIDIA RTX A4000"]
    assert planner._provider_gpu_priority("isaac_sim", {}) == [
        "NVIDIA RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA RTX 6000 Ada Generation",
    ]
    warm_policy = planner._warm_pool_policy(
        request={
            "execution_request": {
                "gpu_allocation": {
                    "warm_pool_policy": {
                        "enabled": True,
                        "latency_slo_seconds": 120,
                        "estimated_idle_cost_usd_per_hour": 2.0,
                        "max_idle_cost_usd_per_hour": 1.0,
                    }
                }
            }
        },
        worker_launch_plan={},
        provider_shape={},
    )
    assert warm_policy["decision"] == "scale_to_zero_on_demand"
    assert warm_policy["decision_reasons"] == [
        "latency_policy_does_not_justify_idle_cost",
        "warm_idle_cost_exceeds_policy",
    ]
    assert planner._artifact_output_writable("") is False


def test_provider_preflight_reports_unready_inputs_timeouts_and_gpu_priority(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_plan["simulator"] = "unknown_sim"
    _write_json(job_dir / "worker_launch_plan.json", worker_plan)
    provider = _read_json(job_dir / "gpu_provider_launch_request.json")
    provider["status"] = "draft"
    shape = provider["provider_request_shape"]  # type: ignore[index]
    shape["inputs"]["manifest_uri_fetchable_by_provider"] = False  # type: ignore[index]
    shape["inputs"]["capture_root_bundle_uri_fetchable_by_provider"] = False  # type: ignore[index]
    shape["inputs"]["artifact_output_uri"] = "https://example.test/output"  # type: ignore[index]
    shape["limits"]["hard_timeout_seconds"] = ""  # type: ignore[index]
    shape["limits"]["external_watchdog_ttl_seconds"] = ""  # type: ignore[index]
    shape["gpu"] = {}  # type: ignore[index]
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["status"] == "blocked_before_spend"
    assert plan["blockers"] == [
        "provider_launch_request_not_ready",
        "worker_manifest_uri_not_fetchable_by_provider",
        "capture_root_bundle_uri_not_fetchable_by_provider",
        "provider_artifact_output_uri_not_writable",
        "missing_hard_timeout_seconds",
        "missing_external_watchdog_ttl_seconds",
        "missing_provider_gpu_priority_fallback_list",
    ]
    missing_output = planner._preflight_result(
        artifacts={
            "gpu_provider_launch_request": {
                "status": "request_manifest_ready",
                "provider_request_shape": {
                    "image": {
                        "configured_image_ref": "registry/worker:tag",
                        "configured_image_ref_is_versioned": True,
                    },
                    "inputs": {
                        "manifest_uri": "r2://bucket/worker.json",
                        "manifest_uri_fetchable_by_provider": True,
                        "capture_root_bundle_uri": "r2://bucket/root.zip",
                        "capture_root_bundle_uri_fetchable_by_provider": True,
                    },
                    "limits": {
                        "hard_timeout_seconds": 30,
                        "external_watchdog_ttl_seconds": 60,
                    },
                },
            }
        },
        provider_gpu_priority=["NVIDIA L4"],
    )
    assert missing_output["blockers"] == ["missing_provider_artifact_output_uri"]


def test_runtime_and_cost_status_cover_pending_blocked_and_failed_paths(tmp_path: Path) -> None:
    pending_capture = tmp_path / "pending"
    pending_job = _seed_ready_job(pending_capture)
    _write_json(
        pending_job / "runpod_provider_adapter_result.json",
        {"api_call_performed": True},
    )

    pending = build_sim_only_provider_execution_layer(
        capture_root=pending_capture,
        job_dir=pending_job,
    )

    assert pending["runtime_manifest"]["status"] == "provider_submitted_runtime_pending"  # type: ignore[index]
    assert pending["cost_ledger"]["status"] == "running"  # type: ignore[index]

    blocked_capture = tmp_path / "blocked"
    blocked_job = _seed_ready_job(blocked_capture)
    _write_json(
        blocked_job / "runpod_live_execution_proof.json",
        {"status": "blocked", "blockers": ["provider_runtime_not_observed"]},
    )

    blocked = build_sim_only_provider_execution_layer(
        capture_root=blocked_capture,
        job_dir=blocked_job,
    )

    assert blocked["runtime_manifest"]["status"] == "blocked"  # type: ignore[index]
    assert blocked["runtime_manifest"]["blockers"] == ["provider_runtime_not_observed"]  # type: ignore[index]

    failed_capture = tmp_path / "failed"
    failed_job = _seed_ready_job(failed_capture)
    _write_json(failed_job / "worker_runtime_manifest.json", {"status": "failed"})

    failed = build_sim_only_provider_execution_layer(
        capture_root=failed_capture,
        job_dir=failed_job,
    )

    assert failed["runtime_manifest"]["status"] == "failed"  # type: ignore[index]
    assert failed["cost_ledger"]["status"] == "failed"  # type: ignore[index]


def test_provider_closure_audit_passes_with_local_runpod_finalizer_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY_FILE", raising=False)
    monkeypatch.setenv("RUNPOD_CONFIG_FILE", str(tmp_path / "missing-runpod-config.toml"))
    build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)
    _write_json(
        job_dir / "runpod_live_execution_proof.json",
        {
            "status": "runpod_live_proof_collected",
            "shutdown_or_termination_proof": True,
            "active_pod_count_before": 1,
            "active_pod_count_after": 0,
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "provider_runtime_finalizer_proof.json",
        {
            "schema_version": "robot_eval_provider_runtime_finalizer_proof.v1",
            "status": "completed",
            "provider": "runpod",
            "artifact_upload_completed_before_shutdown": True,
            "worker_artifacts_finalized_before_shutdown": True,
            "worker_runtime_manifest_upload_completed_before_shutdown": True,
            "provider_shutdown_proven": True,
            "clean_shutdown_proven": True,
            "provider_shutdown_evidence": {
                "zero_active_workers_after_run": True,
                "active_pod_count_after_stop": 0,
            },
            "blockers": [],
        },
    )

    report = audit_provider_closure(job_dir=job_dir)

    assert report["status"] == "passed"
    assert report["live_provider_calls_performed"] is False
    assert report["optional_provider_closure_verified"] is True
    assert report["provider_credentials"]["credential_configured"] is False  # type: ignore[index]
    assert report["checks"]["watchdog"]["status"] == "passed"  # type: ignore[index]
    assert report["checks"]["spend_ledger"]["status"] == "passed"  # type: ignore[index]
    assert report["checks"]["artifact_output_closure"]["status"] == "passed"  # type: ignore[index]
    assert report["checks"]["teardown"]["status"] == "passed"  # type: ignore[index]
    assert report["claim_boundary"]["rank_fidelity_result_proven"] is False  # type: ignore[index]
    assert report["claim_boundary"]["physical_robot_readiness_proven"] is False  # type: ignore[index]
    assert (job_dir / PROVIDER_CLOSURE_AUDIT_REPORT_NAME).is_file()


def test_provider_closure_audit_passes_with_local_vast_teardown_and_upload_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "vast-provider-run"
    job_dir.mkdir(parents=True)
    monkeypatch.setenv("VAST_API_KEY_FILE", str(tmp_path / "missing-vast-api-key"))
    _write_json(
        job_dir / "vast_provider_plan.json",
        {
            "schema_version": "vast_provider_plan.v1",
            "status": "ready_for_explicit_vast_probe",
            "budget": {
                "target_spend_usd": 0.35,
                "hard_cap_usd": 0.75,
                "max_live_minutes": 10,
            },
        },
    )
    _write_json(
        job_dir / "vast_budget_ledger.json",
        {
            "schema_version": "vast_budget_ledger.v1",
            "status": "completed",
            "estimated_cost_usd": 0.03,
            "hard_cap_usd": 0.75,
            "continuing_spend_from_this_run": False,
        },
    )
    _write_json(
        job_dir / "vast_provider_command_result.json",
        {
            "schema_version": "vast_provider_command_result.v1",
            "status": "completed",
            "provider_output_upload_ok": True,
            "provider_runtime_output_zip_produced": True,
            "provider_runtime_output_zip_received": True,
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "vast_teardown_manifest.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "vast_instance_ids": [555],
            "runner_gpu_teardown_completed": True,
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        },
    )
    _write_json(
        job_dir / "vast_final_validation.json",
        {
            "schema_version": "vast_final_validation.v1",
            "status": "passed",
            "vast_instance_ids": [555],
            "estimated_cost_usd": 0.03,
            "spend_hard_cap_usd": 0.75,
            "continuing_spend_from_this_run": False,
            "all_vast_instances_destroyed_by_adapter": True,
            "blockers": [],
        },
    )
    (job_dir / "vast_runtime_phase_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"phase": "vast_instance_teardown_started", "status": "running"}),
                json.dumps({"phase": "vast_instance_teardown_completed", "status": "completed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_provider_closure(job_dir=job_dir, provider="vast")

    assert report["status"] == "passed"
    assert report["provider"] == "vast"
    assert report["provider_credentials"]["credential_configured"] is False  # type: ignore[index]
    assert report["checks"]["watchdog"]["status"] == "passed"  # type: ignore[index]
    assert report["checks"]["artifact_output_closure"]["evidence"][0]["source"] == (  # type: ignore[index]
        "vast_provider_command_result"
    )
    assert report["checks"]["teardown"]["status"] == "passed"  # type: ignore[index]
    assert report["claim_boundary"]["safety_validation_proven"] is False  # type: ignore[index]


def test_provider_closure_audit_blocks_when_provider_artifacts_and_credentials_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root, job_id="job-missing-provider-closure")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY_FILE", raising=False)
    monkeypatch.setenv("RUNPOD_CONFIG_FILE", str(tmp_path / "missing-runpod-config.toml"))

    report = audit_provider_closure(job_dir=job_dir)

    assert report["status"] == "blocked_optional_provider_closure"
    assert report["optional_provider_closure_verified"] is False
    assert "provider_artifact_output_closure_missing" in report["blockers"]
    assert "provider_teardown_evidence_missing" in report["blockers"]
    assert "provider_closure_artifacts_unavailable" in report["blockers"]
    assert "provider_credentials_missing_for_remote_artifact_recovery" in report["blockers"]
    assert report["live_provider_calls_performed"] is False
    assert report["claim_boundary"]["optional_provider_closure_required_for_local_sim_only_beta"] is False  # type: ignore[index]


def test_provider_closure_audit_tracks_remote_execution_and_readiness_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "runpod-dry-run-job"
    job_dir.mkdir(parents=True)
    runpod_key_file = tmp_path / "runpod.key"
    runpod_key_file.write_text("test-runpod-key\n", encoding="utf-8")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(runpod_key_file))
    _write_json(
        job_dir / "remote_cloud_execution_closure_manifest.json",
        {
            "schema_version": "robot_eval_remote_cloud_execution_closure.v1",
            "status": "blocked_before_remote_execution",
            "provider": "runpod",
            "simulator": "mujoco",
            "remote_cloud_execution_proven": False,
            "clean_shutdown_proven": False,
            "live_provider_calls_performed": False,
            "phase": "blocked",
            "provider_job_id": "remote-job-1",
            "pod_id": None,
            "provider_runtime_started_at": None,
            "max_wait_seconds": 180,
            "watchdog_boundary": {
                "hard_timeout_seconds": 120,
                "idle_timeout_seconds": 60,
                "external_watchdog_ttl_seconds": 180,
                "watchdog_owner": "provider_launcher_or_owner_control_plane",
            },
            "max_spend_usd": 0.25,
            "output_uri": "r2://blueprint-artifacts/jobs/remote-job-1/artifacts",
            "output_zip_or_object_size_bytes": None,
            "artifact_manifest": {
                "artifact_output_uri": "r2://blueprint-artifacts/jobs/remote-job-1/artifacts",
                "artifact_upload_evidence_complete": True,
            },
            "teardown_status": "teardown_not_proven",
            "continuing_spend_from_this_run": False,
            "contract_blockers": ["provider_input_setup:provider_inputs_upload_not_proven"],
            "runtime_blockers": ["remote_provider_runtime_not_executed"],
            "checks": {"provider_package_validated_without_spend": True},
        },
    )
    _write_json(
        job_dir / "runpod_provider_readiness_manifest.json",
        {
            "schema_version": "runpod_provider_readiness_manifest.v1",
            "status": "blocked_before_paid_provider_attempt",
            "provider": "runpod",
            "mode": "dry-run",
            "api_call_performed": False,
            "live_provider_call_authorized": False,
            "blockers": ["provider_inputs_upload_not_proven"],
            "spend_limits": {
                "requested_budget_usd": 0.25,
                "bounded_single_worker_attempt": True,
                "hard_timeout_seconds": 120,
                "idle_timeout_seconds": 60,
                "external_watchdog_ttl_seconds": 180,
            },
            "artifact_output": {
                "artifact_output_uri_scheme": "r2",
                "artifact_output_uri_provider_writable": True,
            },
            "watchdog_and_teardown": {
                "upload_before_shutdown_required": True,
                "continuing_spend_from_this_run_must_be_false_after_teardown": True,
            },
            "no_secret_artifact_policy": {
                "raw_api_key_stored": False,
                "secret_env_var_names_stored": False,
            },
            "claim_boundary": {
                "provider_job_submitted": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )

    report = audit_provider_closure(job_dir=job_dir, provider="runpod")

    assert report["status"] == "blocked_optional_provider_closure"
    assert report["optional_provider_closure_verified"] is False
    assert report["provider_credentials"]["credential_configured"] is True  # type: ignore[index]
    remote = report["remote_execution"]
    assert remote["status"] == "blocked_before_remote_execution"  # type: ignore[index]
    assert remote["remote_cloud_execution_proven"] is False  # type: ignore[index]
    assert remote["provider_job_id"] == "remote-job-1"  # type: ignore[index]
    assert remote["max_spend_usd"] == 0.25  # type: ignore[index]
    assert remote["watchdog_boundary"]["hard_timeout_seconds"] == 120  # type: ignore[index]
    assert remote["output_zip_or_object_size_bytes"] is None  # type: ignore[index]
    assert remote["teardown_status"] == "teardown_not_proven"  # type: ignore[index]
    readiness = report["provider_readiness"]
    assert readiness["status"] == "blocked_before_paid_provider_attempt"  # type: ignore[index]
    assert readiness["api_call_performed"] is False  # type: ignore[index]
    assert readiness["live_provider_call_authorized"] is False  # type: ignore[index]
    assert readiness["spend_limits"]["bounded_single_worker_attempt"] is True  # type: ignore[index]
    assert (
        report["observed_artifacts"]["remote_execution_closure"][0]["relative_path"]  # type: ignore[index]
        == "remote_cloud_execution_closure_manifest.json"
    )
    assert (
        report["observed_artifacts"]["provider_readiness"][0]["relative_path"]  # type: ignore[index]
        == "runpod_provider_readiness_manifest.json"
    )


def test_runtime_preflight_simulator_fallback_drives_isaac_priority_and_cache(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_dir = _seed_ready_job(capture_root)
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_plan.pop("simulator")
    _write_json(job_dir / "worker_launch_plan.json", worker_plan)
    scheduler = _read_json(job_dir / "scheduler_decision.json")
    scheduler.pop("simulator", None)
    _write_json(job_dir / "scheduler_decision.json", scheduler)
    provider = _read_json(job_dir / "gpu_provider_launch_request.json")
    provider["provider_request_shape"]["runtime_preflight"] = {"simulator": "isaac_sim"}  # type: ignore[index]
    provider["provider_request_shape"]["gpu"] = {}  # type: ignore[index]
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    plan = build_sim_only_provider_execution_layer(capture_root=capture_root, job_dir=job_dir)

    assert plan["simulator_scope"]["simulator_backend"] == "isaac_sim"  # type: ignore[index]
    assert plan["provider_gpu_priority_fallback_list"][0] == "NVIDIA RTX 4090"
    assert plan["persistent_cache_paths"]["scene_assets"].endswith("/scene_assets")  # type: ignore[index]


def test_agent_planner_blockers_include_missing_sdk_when_module_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planner, "module_available", lambda _names: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(LIVE_AGENTS_SDK_ENV, raising=False)
    monkeypatch.delenv(LIVE_AGENT_PLANNER_ENV, raising=False)

    agent_plan = planner._agent_planner(
        plan_context={"api_key": "secret"},
        allow_live_agent_planner=False,
        executor=None,
        model="test-model",
    )

    assert agent_plan["status"] == "blocked"
    assert agent_plan["blockers"][:3] == [
        "missing_openai_agents_sdk",
        "missing_openai_api_key",
        "missing_cli_allow_sim_only_provider_agent_planner",
    ]


def test_sim_only_provider_execution_planner_main_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_sim_only_provider_execution_layer(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "status": "ready_for_provider_launch" if len(calls) == 1 else "blocked_before_spend",
            "artifacts_manifest_path": "sim_only_provider_artifacts_manifest.json",
        }

    monkeypatch.setattr(
        planner,
        "build_sim_only_provider_execution_layer",
        fake_build_sim_only_provider_execution_layer,
    )

    output_dir = tmp_path / "out"
    assert planner.main(
        [
            "--capture-root",
            str(tmp_path / "capture"),
            "--job-id",
            "job-1",
            "--output-dir",
            str(output_dir),
            "--allow-live-agent-planner",
            "--agent-model",
            "gpt-test",
            "--update-simulator-beta-readiness",
        ]
    ) == 0
    assert planner.main(
        [
            "--capture-root",
            str(tmp_path / "capture"),
            "--job-dir",
            str(tmp_path / "job-dir"),
        ]
    ) == 2

    output = capsys.readouterr().out
    assert "ready_for_provider_launch" in output
    assert str(output_dir) in output
    assert "sim_only_provider_artifacts_manifest.json" in output
    assert calls[0]["allow_live_agent_planner"] is True
    assert calls[0]["agent_model"] == "gpt-test"
    assert calls[0]["update_simulator_beta_readiness"] is True
