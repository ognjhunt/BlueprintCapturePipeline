from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.robot_eval_terminal_artifact_adapter import (
    read_terminal_robot_eval_artifacts,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _terminal_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "canonical-job-1"
    _write(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "canonical-job-1",
            "source": {"execution_admission_digest": "sha256:admitted"},
        },
    )
    _write(
        job_dir / "job_run_manifest.json",
        {
            "schema_version": "robot_eval_job_run_manifest.v1",
            "job_id": "canonical-job-1",
            "status": "simulator_command_completed",
            "simulator_service_status": "completed",
            "blockers": [],
            "claim_boundary": {"simulator_execution_proven": True},
            "request_provenance": {"execution_admission_digest": "sha256:admitted"},
        },
    )
    _write(
        job_dir / "simulator_command_batch_metrics.json",
        {
            "schema_version": "mujoco_g1_batch_metrics.v1",
            "attempt_count": 5,
            "passed_attempt_count": 3,
            "scenario_eval_run_coverage_complete": True,
        },
    )
    _write(
        job_dir / "gpu_cost_control_ledger.json",
        {
            "schema_version": "robot_eval_gpu_cost_control_ledger.v1",
            "job_id": "canonical-job-1",
            "provider": "fixture_local",
            "status": "fixture_local_no_gpu",
            "blockers": [],
            "gpu_time": {
                "actual_gpu_seconds": 0,
                "actual_gpu_time_source": "fixture_local_no_gpu",
                "actual_gpu_time_record_present": True,
            },
        },
    )
    return job_dir


def test_reads_terminal_observed_episode_and_zero_cost(tmp_path: Path) -> None:
    result = read_terminal_robot_eval_artifacts(_terminal_job(tmp_path))

    assert result["status"] == "terminal_observed"
    assert result["job_id"] == "canonical-job-1"
    assert result["execution_admission_digest"] == "sha256:admitted"
    assert result["episode_result"]["episodes_run"] == 5
    assert result["episode_result"]["episodes_succeeded"] == 3
    assert result["cost"]["observed_cost_usd"] == 0.0
    assert result["blockers"] == []


def test_blocks_nonterminal_incomplete_and_unobserved_cost(tmp_path: Path) -> None:
    job_dir = _terminal_job(tmp_path)
    manifest = json.loads((job_dir / "job_run_manifest.json").read_text())
    manifest["status"] = "running"
    manifest["claim_boundary"]["simulator_execution_proven"] = False
    _write(job_dir / "job_run_manifest.json", manifest)
    metrics = json.loads((job_dir / "simulator_command_batch_metrics.json").read_text())
    metrics["scenario_eval_run_coverage_complete"] = False
    _write(job_dir / "simulator_command_batch_metrics.json", metrics)
    ledger = json.loads((job_dir / "gpu_cost_control_ledger.json").read_text())
    ledger["provider"] = "vast"
    ledger["status"] = "provider_runtime_observed"
    ledger["gpu_time"]["actual_gpu_seconds"] = None
    ledger["gpu_time"]["actual_gpu_time_record_present"] = False
    _write(job_dir / "gpu_cost_control_ledger.json", ledger)

    result = read_terminal_robot_eval_artifacts(job_dir)

    assert result["status"] == "blocked"
    assert set(result["blockers"]) >= {
        "job_run_manifest_not_terminal_completed",
        "simulator_execution_not_proven",
        "episode_coverage_not_complete",
        "actual_gpu_time_not_observed",
        "actual_cost_usd_not_observed",
    }


def test_blocks_job_and_admission_provenance_mismatch(tmp_path: Path) -> None:
    job_dir = _terminal_job(tmp_path)
    ledger = json.loads((job_dir / "gpu_cost_control_ledger.json").read_text())
    ledger["job_id"] = "different-job"
    _write(job_dir / "gpu_cost_control_ledger.json", ledger)
    request = json.loads((job_dir / "job_request.json").read_text())
    request["source"]["execution_admission_digest"] = "sha256:different"
    _write(job_dir / "job_request.json", request)

    result = read_terminal_robot_eval_artifacts(job_dir)

    assert result["status"] == "blocked"
    assert "canonical_job_id_mismatch" in result["blockers"]
    assert "execution_admission_digest_mismatch" in result["blockers"]


def test_missing_artifacts_return_blockers_instead_of_raising(tmp_path: Path) -> None:
    result = read_terminal_robot_eval_artifacts(tmp_path / "missing-job")

    assert result["status"] == "blocked"
    assert set(result["blockers"]) >= {
        "job_run_manifest_missing",
        "simulator_command_batch_metrics_missing",
        "gpu_cost_control_ledger_missing",
    }
