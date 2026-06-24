from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline.robot_eval_gpu_startup_pipeline import (
    build_gpu_startup_pipeline_plan,
    build_gpu_startup_pipeline_plan_for_job_dir,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_runpod_secure_cloud_defaults_to_managed_customer_lane() -> None:
    plan = build_gpu_startup_pipeline_plan(
        request={
            "job_id": "job-managed",
            "execution_request": {
                "webapp_role": "queue_and_forward_only",
                "scheduler_owner": "BlueprintCapturePipeline",
                "gpu_allocation": {
                    "allocation_allowed_by_webapp": False,
                    "gpu_spend_approved": False,
                    "max_budget_usd": 3.0,
                },
            },
        },
        job_id="job-managed",
        provisioner="runpod",
        simulator="mujoco",
        scheduler_decision={"status": "awaiting_explicit_gpu_and_simulator_gates"},
        worker_launch_plan={
            "status": "awaiting_explicit_provider_gate",
            "worker_image": {
                "configured_image_ref": (
                    "registry.example/blueprint/mujoco-eval-worker:2026-06-12"
                ),
                "configured_image_ref_present": True,
                "configured_image_ref_is_versioned": True,
                "configured_image_ref_fetchable_by_provider": True,
                "runtime_dependency_install_disallowed": True,
            },
            "launch_mode": {
                "warm_pool_policy": {
                    "decision": "scale_to_zero_on_demand",
                    "warm_worker_recommended": False,
                    "scale_to_zero_default": True,
                }
            },
            "runtime_preflight_contract": {
                "required_before_scene_load": True,
                "required_for_provider": True,
                "worker_blocks_scene_load_on_failed_preflight": True,
                "required_checks": ["python_import_mujoco"],
            },
            "cache_plan": {
                "persistent_cache_recommended": True,
                "targets": ["mujoco_assets"],
                "paths": {"mujoco_assets": "/cache/mujoco_assets"},
            },
        },
        generated_at="2026-06-22T00:00:00Z",
    )

    assert plan["schema_version"] == "robot_eval_gpu_startup_pipeline_plan.v1"
    assert plan["status"] == "startup_pipeline_ready"
    assert plan["selected_provider_tier"] == "managed_secure_cloud_preferred"
    assert plan["selected_provider_is_marketplace"] is False
    assert plan["managed_provider_policy"]["provider_api_priority"] == [  # type: ignore[index]
        "runpod",
        "gcp",
        "vast",
    ]
    assert plan["preflight_canary_policy"][  # type: ignore[index]
        "customer_eval_waits_for_canary"
    ] is True
    session_policy = plan["provider_worker_session_policy"]  # type: ignore[index]
    assert session_policy["allocation_lifecycle"][  # type: ignore[index]
        "provider_allocation_per_inference_allowed"
    ] is False
    assert session_policy["readiness_gate"]["readyz_required_before_first_infer"] is True  # type: ignore[index]
    assert session_policy["http_contract"]["canonical"]["infer"]["path"] == "/infer"  # type: ignore[index]
    assert plan["blockers"] == []


def test_vast_customer_lane_fails_closed_without_explicit_override() -> None:
    plan = build_gpu_startup_pipeline_plan(
        request={
            "job_id": "job-vast",
            "execution_request": {
                "webapp_role": "queue_and_forward_only",
                "scheduler_owner": "BlueprintCapturePipeline",
                "gpu_allocation": {
                    "allocation_allowed_by_webapp": False,
                    "gpu_spend_approved": False,
                },
            },
        },
        job_id="job-vast",
        provisioner="vast",
        simulator="mujoco",
        scheduler_decision={"status": "awaiting_explicit_gpu_and_simulator_gates"},
        worker_launch_plan={"runtime_preflight_contract": {"required_checks": []}},
        generated_at="2026-06-22T00:00:00Z",
    )

    assert plan["status"] == "blocked_before_customer_gpu_allocation"
    assert plan["selected_provider_tier"] == "marketplace_quarantined"
    assert plan["selected_provider_is_marketplace"] is True
    assert plan["provider_worker_session_policy"]["session_scope"] == (  # type: ignore[index]
        "one_ready_worker_per_evaluation_job_or_worker_role"
    )
    assert "marketplace_provider_requires_explicit_customer_job_override" in plan[
        "blockers"
    ]


def test_startup_plan_blocks_one_shot_policy_worker_command(monkeypatch) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE", raising=False)
    plan = build_gpu_startup_pipeline_plan(
        request={"job_id": "job-one-shot"},
        job_id="job-one-shot",
        provisioner="vast",
        simulator="mujoco",
        scheduler_decision={"status": "awaiting_explicit_gpu_and_simulator_gates"},
        worker_launch_plan={
            "provider": "vast",
            "simulator": "mujoco",
            "policy_worker_command": (
                f"{sys.executable} -m "
                "blueprint_pipeline.unitree_groot_n17_sonic_vast_policy_command"
            ),
            "runtime_preflight_contract": {"required_checks": []},
        },
        generated_at="2026-06-22T00:00:00Z",
    )

    session_policy = plan["provider_worker_session_policy"]  # type: ignore[index]
    assert plan["status"] == "blocked_before_customer_gpu_allocation"
    assert session_policy["status"] == "blocked"
    assert session_policy["policy_command_classification"]["invocation_kind"] == (  # type: ignore[index]
        "one_shot_provider_launcher"
    )
    assert (
        "one_shot_provider_launcher_not_allowed_for_repeated_policy_loop"
        in plan["blockers"]
    )
    assert session_policy["allocation_lifecycle"][  # type: ignore[index]
        "provider_allocation_per_inference_allowed"
    ] is False


def test_cli_builder_writes_startup_plan_for_existing_job(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-existing",
            "execution_request": {
                "webapp_role": "queue_and_forward_only",
                "gpu_allocation": {
                    "allocation_allowed_by_webapp": False,
                    "gpu_spend_approved": False,
                },
            },
        },
    )
    _write_json(
        job_dir / "scheduler_decision.json",
        {
            "schema_version": "robot_eval_execution_scheduler_decision.v1",
            "selection": {"provisioner": "runpod", "simulator": "mujoco"},
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "worker_launch_plan.json",
        {
            "schema_version": "robot_eval_worker_launch_plan.v1",
            "provider": "runpod",
            "simulator": "mujoco",
            "runtime_preflight_contract": {"required_checks": ["short_rollout_smoke"]},
        },
    )

    plan = build_gpu_startup_pipeline_plan_for_job_dir(job_dir=job_dir)

    assert plan["job_id"] == "job-existing"
    assert (job_dir / "gpu_startup_pipeline_plan.json").is_file()
