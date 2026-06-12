from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.robot_eval_startup_architecture_audit import (
    build_robot_eval_startup_architecture_audit,
)


EXPECTED_OUTPUTS = [
    "scheduler_decision",
    "worker_launch_plan",
    "worker_manifest",
    "gpu_provider_launch_request",
    "gpu_provider_launcher_result",
    "runpod_provider_adapter_result",
    "gpu_cost_control_ledger",
    "startup_architecture_audit",
    "worker_runtime_manifest",
    "worker_runtime_preflight",
    "job_run_manifest",
    "proof_boundary",
    "metrics",
    "trace",
    "simulator_pov",
    "stdout_log",
    "stderr_log",
]


ISAAC_RUNTIME_PREFLIGHT_CONTRACT = {
    "required_before_scene_load": True,
    "required_for_provider": True,
    "worker_blocks_scene_load_on_failed_preflight": True,
    "executed_by": "blueprint-run-robot-eval-worker",
    "result_artifact": "worker_runtime_preflight.json",
    "run_before": "scene_load_and_policy_execution",
    "simulator": "isaac_sim",
    "renderer_context": "vulkan_rtx",
    "required_checks": [
        "nvidia_smi_gpu_inventory",
        "driver_version",
        "vulkan_device",
        "rtx_renderer_available",
        "isaac_headless_launch",
        "blank_scene_load",
        "test_frame_render",
    ],
    "nvidia_smi_required": True,
    "vulkan_required": True,
    "test_frame_render_required": True,
    "runtime_preflight_is_not_simulator_proof": True,
}

RUNPOD_COST_CONTROL_POLICY = {
    "source": "gpu_provider_launch_request.provider_request_shape.limits",
    "hard_timeout_seconds": 120,
    "idle_timeout_seconds": 60,
    "external_watchdog_ttl_seconds": 180,
    "max_active_workers": 1,
    "serverless_endpoint_controls": {
        "per_request_policy_fields": ["executionTimeout", "ttl", "lowPriority"],
        "endpoint_level_settings_required": [
            "active_workers",
            "max_workers",
            "idle_timeout",
            "execution_timeout",
            "job_ttl",
        ],
        "idle_timeout_set_by_run_request": False,
        "max_workers_set_by_run_request": False,
        "recommended_active_workers": 0,
        "recommended_max_workers": 1,
        "recommended_idle_timeout_seconds": 60,
    },
    "on_demand_pod_controls": {
        "pod_idle_timeout_is_not_provider_native": True,
        "external_watchdog_or_owner_terminator_required": True,
        "external_watchdog_owner": "provider_launcher_or_owner_control_plane",
        "worker_env_shutdown_controls": [
            "BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS",
            "BLUEPRINT_GPU_PROVIDER_IDLE_TIMEOUT_SECONDS",
            "BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS",
        ],
    },
    "proof_boundary": {
        "policy_documents_cost_controls_only": True,
        "provider_idle_shutdown_configured": False,
        "provider_allocation_proven": False,
        "simulator_execution_proven": False,
    },
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _startup_job_dir(tmp_path: Path) -> Path:
    job_dir = tmp_path / "pipeline" / "robot_eval_jobs" / "startup-job-1"
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "startup-job-1",
            "source": {"system": "Blueprint-WebApp"},
        },
    )
    _write_json(
        job_dir / "job_run_manifest.json",
        {
            "schema_version": "robot_eval_job_run_manifest.v1",
            "job_id": "startup-job-1",
            "status": "blocked",
            "claim_boundary": {
                "simulator_execution_proven": False,
                "robot_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )
    _write_json(
        job_dir / "scheduler_decision.json",
        {
            "schema_version": "robot_eval_execution_scheduler_decision.v1",
            "job_id": "startup-job-1",
            "status": "local_fixture_only",
            "webapp_role": "queue_and_forward_only",
            "scheduler_owner": "BlueprintCapturePipeline",
            "queueing": {
                "mode": "async_job",
                "customer_response": "job_id_and_status_only",
                "web_request_must_not_wait_for_simulator": True,
            },
            "selection": {"provisioner": "fixture_local", "simulator": "fixture"},
            "cpu_preflight_gate": {
                "required_before_gpu": True,
                "blocks_gpu_when_missing": True,
                "required_artifact_status": {
                    "scene_asset_inventory": {"present": True},
                    "scene_asset_dependency_audit": {"present": True},
                    "cpu_preflight_scorecard": {"present": True},
                    "episode_spec_manifest": {"present": True},
                    "gpu_handoff_packet": {"present": True},
                },
            },
            "gpu_allocation": {
                "allocation_allowed_by_webapp": False,
                "gpu_spend_approved_by_webapp": False,
                "hard_timeout_seconds": 120,
                "idle_shutdown_required": True,
            },
            "artifact_contract": {
                "expected_outputs": EXPECTED_OUTPUTS,
                "simulator_execution_proven_by_webapp": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )
    _write_json(
        job_dir / "worker_launch_plan.json",
        {
            "schema_version": "robot_eval_worker_launch_plan.v1",
            "job_id": "startup-job-1",
            "status": "not_required_for_fixture_local",
            "worker_image": {
                "entrypoint": "blueprint-run-robot-eval-worker",
                "runtime_dependency_install_disallowed": True,
                "runtime_asset_guessing_disallowed": True,
            },
            "launch_mode": {
                "mode": "on_demand_with_optional_warm_pool",
                "scale_to_zero_default": True,
                "max_active_workers": 1,
                "idle_shutdown_required": True,
                "idle_timeout_seconds": 60,
                "hard_timeout_seconds": 120,
            },
            "cache_plan": {
                "install_simulator_during_customer_job": False,
                "install_python_dependencies_during_customer_job": False,
            },
            "worker_entrypoint_contract": {
                "job_manifest_env": "BLUEPRINT_EVAL_MANIFEST_URI",
                "package_console_script": "blueprint-run-robot-eval-worker",
                "web_request_waits_for_worker": False,
            },
            "artifact_upload_contract": {
                "upload_before_shutdown_required": True,
                "expected_outputs": EXPECTED_OUTPUTS,
            },
        },
    )
    _write_json(
        job_dir / "worker_manifest.json",
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "job_id": "startup-job-1",
            "status": "ready_for_worker_upload",
            "capture_root": str(tmp_path),
            "provisioner": "fixture_local",
            "simulator": "fixture",
            "worker_manifest_uri": None,
            "worker_manifest_uri_required": False,
            "worker_manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "worker_manifest_uri_fetchable_by_provider": False,
            "worker_manifest_uri_scheme": None,
            "artifact_output_uri": None,
            "artifact_output_uri_required": False,
            "artifact_output_uri_env_var": "BLUEPRINT_ARTIFACT_OUTPUT_URI",
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "startup-job-1",
            },
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "gpu_provider_launch_request.json",
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "startup-job-1",
            "status": "not_required_for_fixture_local",
            "live_provider_calls_performed": False,
            "provider_request_shape": {
                "api_payload_is_provider_adapter_template": True,
                "command": "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}",
                "environment": {
                    "secret_env_var_names": ["AWS_SECRET_ACCESS_KEY"],
                    "secret_values_in_artifact": False,
                },
                "inputs": {
                    "manifest_uri_required": True,
                    "manifest_uri_required_for_provider": False,
                    "manifest_uri": None,
                    "manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
                    "manifest_uri_configured": False,
                    "manifest_uri_fetchable_by_provider": False,
                    "manifest_uri_scheme": None,
                    "artifact_output_uri_required": True,
                },
                "limits": {
                    "idle_shutdown_required": True,
                    "hard_timeout_seconds": 120,
                },
            },
        },
    )
    _write_json(
        job_dir / "gpu_cost_control_ledger.json",
        {
            "schema_version": "robot_eval_gpu_cost_control_ledger.v1",
            "job_id": "startup-job-1",
            "status": "blocked_before_allocation",
            "budget": {
                "gpu_spend_approved_by_webapp": False,
                "allocation_allowed_by_webapp": False,
            },
            "worker_limits": {
                "customer_concurrency_limit_required": True,
                "idle_shutdown_required": True,
                "hard_timeout_seconds": 120,
                "max_billable_gpu_seconds": 120,
            },
            "gpu_time": {
                "estimated_gpu_seconds": 0,
                "actual_gpu_seconds": 0,
                "actual_gpu_time_source": "fixture_local_no_gpu",
                "actual_gpu_time_record_required": True,
                "actual_gpu_time_record_present": True,
            },
        },
    )
    return job_dir


def test_robot_eval_startup_architecture_audit_passes_valid_startup_job(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["schema_version"] == "robot_eval_startup_architecture_audit.v1"
    assert result["status"] == "passed"
    assert result["architecture_compliant"] is True
    assert result["blocked_check_count"] == 0
    assert result["proof_boundary"]["read_only_audit"] is True
    assert result["proof_boundary"]["simulator_execution_proven"] is False
    assert Path(str(result["output_path"])).is_file()


def test_robot_eval_startup_architecture_audit_blocks_missing_artifacts(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    (job_dir / "worker_launch_plan.json").unlink()

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert result["architecture_compliant"] is False
    assert "missing_worker_launch_plan" in result["blockers"]
    assert "worker_launch_plan:schema" in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_missing_worker_manifest_job_request(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    worker_manifest = json.loads(
        (job_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    worker_manifest.pop("job_request")
    _write_json(job_dir / "worker_manifest.json", worker_manifest)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker_manifest:strict_manifest_payload" in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_required_worker_artifact_output_uri(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    worker_manifest = json.loads(
        (job_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    worker_manifest["artifact_output_uri_required"] = True
    worker_manifest["artifact_output_uri"] = None
    worker_manifest["blockers"] = ["missing_worker_artifact_output_uri"]
    _write_json(job_dir / "worker_manifest.json", worker_manifest)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker_manifest:artifact_output_when_required" in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_live_provider_without_image_ref(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    worker = json.loads((job_dir / "worker_launch_plan.json").read_text(encoding="utf-8"))
    worker_manifest = json.loads(
        (job_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    provider = json.loads(
        (job_dir / "gpu_provider_launch_request.json").read_text(encoding="utf-8")
    )
    manifest_uri = "r2://blueprint-artifacts/jobs/startup-job-1/worker_manifest.json"
    worker["worker_image"].update(
        {
            "published_image_ref_required": True,
            "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
            "configured_image_ref": None,
            "configured_image_ref_present": False,
            "configured_image_ref_is_versioned": False,
        }
    )
    worker["simulator"] = "isaac_sim"
    worker["runtime_preflight_contract"] = dict(ISAAC_RUNTIME_PREFLIGHT_CONTRACT)
    worker_manifest.update(
        {
            "worker_manifest_uri": manifest_uri,
            "worker_manifest_uri_required": True,
            "worker_manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "worker_manifest_uri_scheme": "r2",
            "worker_manifest_uri_fetchable_by_provider": True,
            "runtime_preflight_contract": dict(ISAAC_RUNTIME_PREFLIGHT_CONTRACT),
        }
    )
    provider["provider_request_shape"]["image"] = {
        "image_family": "isaac-eval-worker",
        "owner_published_image_ref_required": True,
        "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "configured_image_ref": None,
        "configured_image_ref_present": False,
        "configured_image_ref_is_versioned": False,
        "entrypoint": "blueprint-run-robot-eval-worker",
    }
    provider["provider_request_shape"]["inputs"].update(
        {
            "manifest_uri_required_for_provider": True,
            "manifest_uri": manifest_uri,
            "manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "manifest_uri_configured": True,
            "manifest_uri_fetchable_by_provider": True,
            "manifest_uri_scheme": "r2",
        }
    )
    provider["provider_request_shape"]["runtime_preflight"] = dict(
        ISAAC_RUNTIME_PREFLIGHT_CONTRACT
    )
    _write_json(job_dir / "worker_launch_plan.json", worker)
    _write_json(job_dir / "worker_manifest.json", worker_manifest)
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker:published_image_ref_when_live_provider" in result["blockers"]
    assert "provider:published_image_ref_when_live_provider" in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_live_provider_without_manifest_uri(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    worker = json.loads((job_dir / "worker_launch_plan.json").read_text(encoding="utf-8"))
    worker_manifest = json.loads(
        (job_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    provider = json.loads(
        (job_dir / "gpu_provider_launch_request.json").read_text(encoding="utf-8")
    )
    image_ref = "registry.example/blueprint/isaac-eval-worker:2026-06-12"
    provider["provider_request_shape"]["image"] = {
        "image_family": "isaac-eval-worker",
        "owner_published_image_ref_required": True,
        "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "configured_image_ref": image_ref,
        "configured_image_ref_present": True,
        "configured_image_ref_is_versioned": True,
        "entrypoint": "blueprint-run-robot-eval-worker",
    }
    worker["simulator"] = "isaac_sim"
    worker["runtime_preflight_contract"] = dict(ISAAC_RUNTIME_PREFLIGHT_CONTRACT)
    worker_manifest.update(
        {
            "worker_manifest_uri": None,
            "worker_manifest_uri_required": True,
            "worker_manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "worker_manifest_uri_scheme": None,
            "worker_manifest_uri_fetchable_by_provider": False,
            "runtime_preflight_contract": dict(ISAAC_RUNTIME_PREFLIGHT_CONTRACT),
            "blockers": ["missing_worker_manifest_uri"],
        }
    )
    provider["provider_request_shape"]["inputs"].update(
        {
            "manifest_uri_required_for_provider": True,
            "manifest_uri": None,
            "manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "manifest_uri_configured": False,
            "manifest_uri_fetchable_by_provider": False,
            "manifest_uri_scheme": None,
        }
    )
    provider["provider_request_shape"]["runtime_preflight"] = dict(
        ISAAC_RUNTIME_PREFLIGHT_CONTRACT
    )
    _write_json(job_dir / "worker_launch_plan.json", worker)
    _write_json(job_dir / "worker_manifest.json", worker_manifest)
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker_manifest:fetch_uri_when_required" in result["blockers"]
    assert "provider:worker_manifest_uri_when_required" in result["blockers"]
    assert "provider:published_image_ref_when_live_provider" not in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_live_provider_without_runtime_preflight(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    worker = json.loads((job_dir / "worker_launch_plan.json").read_text(encoding="utf-8"))
    worker_manifest = json.loads(
        (job_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    provider = json.loads(
        (job_dir / "gpu_provider_launch_request.json").read_text(encoding="utf-8")
    )
    image_ref = "registry.example/blueprint/isaac-eval-worker:2026-06-12"
    manifest_uri = "r2://blueprint-artifacts/jobs/startup-job-1/worker_manifest.json"
    worker["simulator"] = "isaac_sim"
    worker["worker_image"].update(
        {
            "published_image_ref_required": True,
            "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
            "configured_image_ref": image_ref,
            "configured_image_ref_present": True,
            "configured_image_ref_is_versioned": True,
        }
    )
    worker_manifest.update(
        {
            "worker_manifest_uri": manifest_uri,
            "worker_manifest_uri_required": True,
            "worker_manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "worker_manifest_uri_scheme": "r2",
            "worker_manifest_uri_fetchable_by_provider": True,
        }
    )
    provider["provider"] = "runpod"
    provider["provider_request_shape"]["provider_api"] = "runpod"
    provider["provider_request_shape"]["image"] = {
        "image_family": "isaac-eval-worker",
        "owner_published_image_ref_required": True,
        "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "configured_image_ref": image_ref,
        "configured_image_ref_present": True,
        "configured_image_ref_is_versioned": True,
        "entrypoint": "blueprint-run-robot-eval-worker",
    }
    provider["provider_request_shape"]["inputs"].update(
        {
            "manifest_uri_required_for_provider": True,
            "manifest_uri": manifest_uri,
            "manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "manifest_uri_configured": True,
            "manifest_uri_fetchable_by_provider": True,
            "manifest_uri_scheme": "r2",
        }
    )
    _write_json(job_dir / "worker_launch_plan.json", worker)
    _write_json(job_dir / "worker_manifest.json", worker_manifest)
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker:runtime_preflight_before_scene_load" in result["blockers"]
    assert "worker:isaac_runtime_preflight_checks" in result["blockers"]
    assert "provider:runtime_preflight_before_scene_load" in result["blockers"]
    assert "provider:published_image_ref_when_live_provider" not in result["blockers"]
    assert "provider:worker_manifest_uri_when_required" not in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_missing_worker_runtime_preflight_artifact(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "job_id": "startup-job-1",
            "status": "blocked",
            "runtime_preflight_manifest_path": "worker_runtime_preflight.json",
            "runtime_preflight_required_before_scene_load": True,
            "runtime_preflight_status": "passed",
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker_runtime:preflight_artifact_present" in result["blockers"]


def test_robot_eval_startup_architecture_audit_accepts_worker_runtime_preflight_artifact(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "job_id": "startup-job-1",
            "status": "blocked",
            "runtime_preflight_manifest_path": "worker_runtime_preflight.json",
            "runtime_preflight_required_before_scene_load": True,
            "runtime_preflight_status": "passed",
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )
    _write_json(
        job_dir / "worker_runtime_preflight.json",
        {
            "schema_version": "robot_eval_worker_runtime_preflight.v1",
            "job_id": "startup-job-1",
            "status": "passed",
            "secret_values_in_artifact": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "passed"
    assert "worker_runtime:preflight_artifact_present" not in result["blockers"]
    assert "worker_runtime:preflight_status_consistent" not in result["blockers"]


def test_robot_eval_startup_architecture_audit_accepts_provider_launcher_result(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "gpu_provider_launcher_result.json",
        {
            "schema_version": "robot_eval_gpu_provider_launcher_result.v1",
            "job_id": "startup-job-1",
            "provider": "runpod",
            "status": "completed",
            "execution_performed": True,
            "provider_launcher_command_executed": True,
            "secret_values_in_artifact": False,
            "stdout_stderr_secret_redaction_enabled": True,
            "command": {
                "shell": False,
                "raw_command_stored": False,
                "executable": "provider-launcher",
            },
            "stdout_path": str(job_dir / "gpu_provider_launcher.stdout.log"),
            "stderr_path": str(job_dir / "gpu_provider_launcher.stderr.log"),
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "passed"
    assert "provider_launcher:no_secret_or_proof_upgrade" not in result["blockers"]
    assert "provider_launcher:command_redacted_when_executed" not in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_bad_worker_runtime_preflight(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "job_id": "startup-job-1",
            "status": "blocked",
            "runtime_preflight_manifest_path": "worker_runtime_preflight.json",
            "runtime_preflight_required_before_scene_load": True,
            "runtime_preflight_status": "passed",
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )
    _write_json(
        job_dir / "worker_runtime_preflight.json",
        {
            "schema_version": "robot_eval_worker_runtime_preflight.v1",
            "job_id": "startup-job-1",
            "status": "passed",
            "execution_performed": True,
            "secret_values_in_artifact": True,
            "raw_command": "runtime-preflight --token super-secret",
            "command": {
                "shell": True,
                "raw_command_stored": True,
            },
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "worker_runtime:preflight_command_redacted" in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_bad_provider_launcher_result(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "gpu_provider_launcher_result.json",
        {
            "schema_version": "robot_eval_gpu_provider_launcher_result.v1",
            "job_id": "startup-job-1",
            "provider": "runpod",
            "status": "completed",
            "execution_performed": True,
            "provider_launcher_command_executed": True,
            "secret_values_in_artifact": True,
            "raw_command": "provider-cli --token super-secret",
            "command": {"shell": True, "raw_command_stored": True},
            "simulator_execution_proven": True,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "provider_launcher:no_secret_or_proof_upgrade" in result["blockers"]
    assert "provider_launcher:command_redacted_when_executed" in result["blockers"]


def test_robot_eval_startup_architecture_audit_accepts_runpod_adapter_result(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "runpod_provider_adapter_result.json",
        {
            "schema_version": "runpod_provider_adapter_result.v1",
            "job_id": "startup-job-1",
            "provider": "runpod",
            "mode": "dry-run",
            "status": "dry_run_ready",
            "api_call_performed": False,
            "runpod_side_effects_may_have_occurred": False,
            "secret_values_in_artifact": False,
            "raw_api_key_stored": False,
            "cost_control_policy": RUNPOD_COST_CONTROL_POLICY,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "passed"
    assert "runpod_adapter:no_secret_or_proof_upgrade" not in result["blockers"]
    assert "runpod_adapter:cost_control_policy" not in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_runpod_adapter_missing_cost_policy(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "runpod_provider_adapter_result.json",
        {
            "schema_version": "runpod_provider_adapter_result.v1",
            "job_id": "startup-job-1",
            "provider": "runpod",
            "mode": "dry-run",
            "status": "dry_run_ready",
            "api_call_performed": False,
            "runpod_side_effects_may_have_occurred": False,
            "secret_values_in_artifact": False,
            "raw_api_key_stored": False,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "runpod_adapter:no_secret_or_proof_upgrade" not in result["blockers"]
    assert "runpod_adapter:cost_control_policy" in result["blockers"]


def test_robot_eval_startup_architecture_audit_blocks_bad_runpod_adapter_result(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    _write_json(
        job_dir / "runpod_provider_adapter_result.json",
        {
            "schema_version": "runpod_provider_adapter_result.v1",
            "job_id": "startup-job-1",
            "provider": "runpod",
            "mode": "on-demand-pod",
            "status": "submitted",
            "api_call_performed": True,
            "runpod_side_effects_may_have_occurred": True,
            "secret_values_in_artifact": True,
            "raw_api_key_stored": True,
            "cost_control_policy": {
                **RUNPOD_COST_CONTROL_POLICY,
                "proof_boundary": {
                    **RUNPOD_COST_CONTROL_POLICY["proof_boundary"],
                    "simulator_execution_proven": True,
                },
            },
            "simulator_execution_proven": True,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "blocked"
    assert "runpod_adapter:no_secret_or_proof_upgrade" in result["blockers"]
    assert "runpod_adapter:cost_control_policy" in result["blockers"]


def test_robot_eval_startup_architecture_audit_accepts_versioned_live_provider_image_ref(
    tmp_path: Path,
) -> None:
    job_dir = _startup_job_dir(tmp_path)
    worker = json.loads((job_dir / "worker_launch_plan.json").read_text(encoding="utf-8"))
    worker_manifest = json.loads(
        (job_dir / "worker_manifest.json").read_text(encoding="utf-8")
    )
    provider = json.loads(
        (job_dir / "gpu_provider_launch_request.json").read_text(encoding="utf-8")
    )
    image_ref = "registry.example/blueprint/isaac-eval-worker:2026-06-12"
    manifest_uri = "r2://blueprint-artifacts/jobs/startup-job-1/worker_manifest.json"
    worker["worker_image"].update(
        {
            "published_image_ref_required": True,
            "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
            "configured_image_ref": image_ref,
            "configured_image_ref_present": True,
            "configured_image_ref_is_versioned": True,
        }
    )
    worker["simulator"] = "isaac_sim"
    worker["runtime_preflight_contract"] = dict(ISAAC_RUNTIME_PREFLIGHT_CONTRACT)
    worker_manifest.update(
        {
            "worker_manifest_uri": manifest_uri,
            "worker_manifest_uri_required": True,
            "worker_manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "worker_manifest_uri_scheme": "r2",
            "worker_manifest_uri_fetchable_by_provider": True,
            "runtime_preflight_contract": dict(ISAAC_RUNTIME_PREFLIGHT_CONTRACT),
        }
    )
    provider["provider_request_shape"]["image"] = {
        "image_family": "isaac-eval-worker",
        "owner_published_image_ref_required": True,
        "image_ref_env_var": "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "configured_image_ref": image_ref,
        "configured_image_ref_present": True,
        "configured_image_ref_is_versioned": True,
        "entrypoint": "blueprint-run-robot-eval-worker",
    }
    provider["provider_request_shape"]["inputs"].update(
        {
            "manifest_uri_required_for_provider": True,
            "manifest_uri": manifest_uri,
            "manifest_uri_env_var": "BLUEPRINT_EVAL_MANIFEST_URI",
            "manifest_uri_configured": True,
            "manifest_uri_fetchable_by_provider": True,
            "manifest_uri_scheme": "r2",
        }
    )
    provider["provider_request_shape"]["runtime_preflight"] = dict(
        ISAAC_RUNTIME_PREFLIGHT_CONTRACT
    )
    _write_json(job_dir / "worker_launch_plan.json", worker)
    _write_json(job_dir / "worker_manifest.json", worker_manifest)
    _write_json(job_dir / "gpu_provider_launch_request.json", provider)

    result = build_robot_eval_startup_architecture_audit(job_dir=job_dir)

    assert result["status"] == "passed"
    assert "worker:published_image_ref_when_live_provider" not in result["blockers"]
    assert "provider:published_image_ref_when_live_provider" not in result["blockers"]


def test_robot_eval_startup_architecture_audit_module_cli(tmp_path: Path) -> None:
    job_dir = _startup_job_dir(tmp_path)
    env = os.environ.copy()
    src_root = Path.cwd() / "src"
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.robot_eval_startup_architecture_audit",
            "--job-dir",
            str(job_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=passed" in completed.stdout
    assert "job_id=startup-job-1" in completed.stdout
