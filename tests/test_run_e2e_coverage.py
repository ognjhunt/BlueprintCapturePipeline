from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import run_e2e
from blueprint_pipeline.common import PipelineError


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / "site-1" / "captures" / "cap-1"
    root.mkdir(parents=True)
    return root


def test_run_end_to_end_materializes_raw_and_threads_optional_lanes(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    (raw_root / "capture_upload_complete.json").write_text("{}", encoding="utf-8")

    calls: dict[str, object] = {}
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda root: {"status": "ready", "root": str(root)})
    monkeypatch.setattr(run_e2e, "materialize_capture_bundle", lambda **kwargs: calls.setdefault("materialize", kwargs))
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", lambda **kwargs: {"status": "completed", "lanes": [kwargs["lane"]]})
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **kwargs: {
            "artifacts": {"readiness_report": "ready.md"},
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
            "provider": kwargs["provider_name"],
        },
    )
    monkeypatch.setattr(
        run_e2e,
        "run_evaluation_prep_stage",
        lambda **kwargs: {
            "manifest_path": "eval/manifest.json",
            "webapp_sync_result": {"status": "skipped"},
            "site_package_manifest": {"status": "blocked"},
            "hosted_review_readiness": {"ready": False},
            "proof_pack_manifest": {"proof": False},
            "proof_path_status": {"status": "blocked"},
        },
    )
    monkeypatch.setattr(run_e2e, "run_cosmos_zero_shot_validation_lane", lambda **kwargs: {"status": "completed"})

    result = run_e2e.run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        pipeline_lane="all",
        run_evaluation_prep=True,
        evaluation_prep_provider="manual",
        run_cosmos_validation=True,
    )

    assert result["schema_version"] == "v1"
    assert result["capture_root"] == str(capture_root)
    assert result["provider"] == "openai"
    assert result["preflight_status"] == "ready"
    assert result["pipeline_status"] == "completed"
    assert result["pipeline_lanes"] == ["all"]
    assert result["pipeline_summary"] == "ready.md"
    assert result["final_memo_path"] == "memo.md"
    assert result["webapp_sync_result"] == {"status": "skipped"}
    assert result["site_package_manifest"] == {"status": "blocked"}
    assert result["hosted_review_readiness"] == {"ready": False}
    assert result["proof_pack_manifest"] == {"proof": False}
    assert result["proof_path_status"] == {"status": "blocked"}
    assert result["cosmos_validation"] == {"status": "completed"}
    assert calls["materialize"]["raw_prefix_uri"] == "gs://bucket/scenes/site-1/captures/cap-1/raw"
    stage_ledger = json.loads(
        Path(result["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8")
    )
    assert stage_ledger["schema_version"] == "run_e2e_stage_ledger.v1"
    assert stage_ledger["status"] == "completed"
    assert stage_ledger["last_completed_stage"] == "cosmos_validation"
    assert stage_ledger["stages"]["preflight"]["status"] == "completed"
    assert stage_ledger["stages"]["materialization"]["status"] == "completed"
    assert stage_ledger["stages"]["capture_pipeline"]["detail"] == "completed"
    assert stage_ledger["stages"]["agent_review"]["status"] == "completed"
    assert stage_ledger["stages"]["evaluation_prep"]["status"] == "completed"
    assert stage_ledger["stages"]["cosmos_validation"]["detail"] == "completed"
    assert stage_ledger["stages"]["robot_eval"]["status"] == "skipped"


def test_run_end_to_end_blocks_preflight_and_missing_descriptor(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"missing_required_inputs": ["raw/manifest.json", "raw/video.mov"]})
    with pytest.raises(PipelineError, match="raw/manifest.json,raw/video.mov"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")
    stage_ledger = json.loads(
        (capture_root / "pipeline" / "run_e2e_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert stage_ledger["status"] == "failed"
    assert stage_ledger["failed_stage"] == "preflight"
    assert stage_ledger["stages"]["preflight"]["error_type"] == "PipelineError"

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"})
    with pytest.raises(PipelineError, match="Descriptor is missing"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")
    stage_ledger = json.loads(
        (capture_root / "pipeline" / "run_e2e_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert stage_ledger["status"] == "failed"
    assert stage_ledger["failed_stage"] == "materialization"
    assert stage_ledger["stages"]["preflight"]["status"] == "completed"
    assert stage_ledger["stages"]["materialization"]["error_type"] == "PipelineError"


def test_run_end_to_end_uses_existing_descriptor_without_optional_lanes(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"})
    monkeypatch.setattr(run_e2e, "materialize_capture_bundle", lambda **_kwargs: pytest.fail("materialize should not run"))
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", lambda **_kwargs: {"status": "completed", "lanes": ["current"]})
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **_kwargs: {"artifacts": {}, "final_memo_path": None, "final_bundle_path": None},
    )

    result = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")

    assert result["evaluation_prep"] is None
    assert result["webapp_sync_result"] is None
    assert result["site_package_manifest"] is None
    assert result["hosted_review_readiness"] is None
    assert result["proof_pack_manifest"] is None
    assert result["proof_path_status"] is None
    assert result["cosmos_validation"] is None


def test_run_end_to_end_resumes_completed_stage_snapshots(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    calls = {"preflight": 0, "pipeline": 0, "review": 0}

    def fake_preflight(_root):
        calls["preflight"] += 1
        return {"status": "ready", "probe": calls["preflight"]}

    def fake_pipeline(**kwargs):
        calls["pipeline"] += 1
        return {"status": "completed", "lanes": [kwargs["lane"]]}

    def fake_review(**_kwargs):
        calls["review"] += 1
        return {
            "artifacts": {"readiness_report": "ready.md"},
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
            "provider_token": "must-not-be-written",
        }

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", fake_preflight)
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", fake_pipeline)
    monkeypatch.setattr(run_e2e, "run_agent_review", fake_review)

    first = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")
    assert first["preflight_status"] == "ready"
    assert calls == {"preflight": 1, "pipeline": 1, "review": 1}

    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda _root: pytest.fail("preflight should resume from stage snapshot"),
    )
    monkeypatch.setattr(
        run_e2e,
        "run_capture_pipeline",
        lambda **_kwargs: pytest.fail("pipeline should resume from stage snapshot"),
    )
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **_kwargs: pytest.fail("review should resume from stage snapshot"),
    )

    second = run_e2e.run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        resume_completed_stages=True,
    )

    assert second["preflight_status"] == "ready"
    assert second["pipeline_status"] == "completed"
    assert second["pipeline_lanes"] == ["current"]
    assert second["pipeline_summary"] == "ready.md"
    assert second["final_memo_path"] == "memo.md"
    stage_ledger = json.loads(
        Path(second["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8")
    )
    assert stage_ledger["resume_completed_stages_requested"] is True
    assert stage_ledger["resume_used_count"] == 3
    assert stage_ledger["stages"]["preflight"]["resume_used"] is True
    assert stage_ledger["stages"]["capture_pipeline"]["resume_used"] is True
    assert stage_ledger["stages"]["agent_review"]["resume_used"] is True
    assert (
        stage_ledger["stages"]["agent_review"]["result_snapshot"]["provider_token"]
        == "<redacted>"
    )


def test_run_end_to_end_invalidates_resume_when_upstream_input_changes(
    monkeypatch, tmp_path: Path
) -> None:
    # R078: a resumed stage must validate that upstream inputs are unchanged before
    # replaying its cached snapshot. Unchanged inputs reuse the cache; a changed
    # descriptor invalidates the cache and forces recompute.
    capture_root = _capture_root(tmp_path)
    descriptor = capture_root / "capture_descriptor.json"
    descriptor.write_text('{"rev": 1}', encoding="utf-8")
    calls = {"preflight": 0, "pipeline": 0, "review": 0}

    def fake_preflight(_root):
        calls["preflight"] += 1
        return {"status": "ready"}

    def fake_pipeline(**kwargs):
        calls["pipeline"] += 1
        return {"status": "completed", "lanes": [kwargs["lane"]]}

    def fake_review(**_kwargs):
        calls["review"] += 1
        return {
            "artifacts": {"readiness_report": "ready.md"},
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
        }

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", fake_preflight)
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", fake_pipeline)
    monkeypatch.setattr(run_e2e, "run_agent_review", fake_review)

    # First run populates the ledger (each stage runs once).
    run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")
    assert calls == {"preflight": 1, "pipeline": 1, "review": 1}

    # Resume with UNCHANGED upstream inputs -> cache is reused (no recompute).
    second = run_e2e.run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        resume_completed_stages=True,
    )
    assert calls == {"preflight": 1, "pipeline": 1, "review": 1}
    ledger_second = json.loads(
        Path(second["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8")
    )
    assert ledger_second["resume_used_count"] == 3
    assert "resume_invalidated_stages" not in ledger_second

    # Change the upstream descriptor, then resume again. The stale cache must be
    # invalidated and every stage recomputed rather than serving stale output.
    descriptor.write_text('{"rev": 2}', encoding="utf-8")
    third = run_e2e.run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        resume_completed_stages=True,
    )
    assert calls == {"preflight": 2, "pipeline": 2, "review": 2}
    assert third["preflight_status"] == "ready"
    ledger_third = json.loads(
        Path(third["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8")
    )
    assert ledger_third["resume_invalidated_count"] == 3
    for stage in ("preflight", "capture_pipeline", "agent_review"):
        entry = ledger_third["stages"][stage]
        assert entry["status"] == "completed"
        assert entry["resume_invalidated"] is True
        assert entry["resume_invalidated_reason"] == "upstream_input_fingerprint_changed"


def test_run_end_to_end_threads_robot_eval_job_and_provider_race_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    request_path = tmp_path / "robot_eval_request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda _root: {"status": "ready"},
    )
    monkeypatch.setattr(
        run_e2e,
        "run_capture_pipeline",
        lambda **kwargs: {"status": "completed", "lanes": [kwargs["lane"]]},
    )
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **_kwargs: {
            "artifacts": {},
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
        },
    )
    calls: dict[str, object] = {}

    def fake_build_robot_eval_job(**kwargs):
        calls.update(kwargs)
        job_dir = tmp_path / "robot-job"
        job_dir.mkdir()
        provider_launch = {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "status": "blocked_by_explicit_provider_gate",
            "reason": "explicit_provider_gate_required",
            "provider": "runpod",
            "live_provider_calls_performed": False,
            "prelaunch_spend_guard": {
                "schema_version": "robot_eval_provider_prelaunch_spend_guard.v1",
                "can_launch": False,
                "blockers": ["missing_cli_allow_gpu_provisioning"],
                "provider_race": {
                    "schema_version": "robot_eval_provider_race_contract.v1",
                    "race_module": "blueprint_pipeline.provider_race",
                    "race_required_for_customer_path": True,
                    "customer_path_provider_failover_wired": False,
                    "customer_path_provider_failover_runtime_wired": False,
                    "customer_path_provider_failover_runtime_status": (
                        "blocked_pending_teardown_owned_race_launcher"
                    ),
                    "customer_path_serial_launch_blocked_unless_override": True,
                },
            },
        }
        (job_dir / "gpu_provider_launch_request.json").write_text(
            json.dumps(provider_launch),
            encoding="utf-8",
        )
        (job_dir / "gpu_cost_control_ledger.json").write_text(
            json.dumps(
                {
                    "status": "blocked_before_allocation",
                    "live_provider_calls_performed": False,
                }
            ),
            encoding="utf-8",
        )
        (job_dir / "remote_cloud_execution_closure_manifest.json").write_text(
            json.dumps(
                {
                    "status": "blocked_before_remote_execution",
                    "remote_cloud_execution_proven": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "schema_version": "robot_eval_job_result.v1",
            "job_id": kwargs["job_id"],
            "job_dir": str(job_dir),
            "manifest_path": str(job_dir / "job_run_manifest.json"),
            "status": "blocked",
        }

    monkeypatch.setattr(run_e2e, "build_robot_eval_job", fake_build_robot_eval_job)

    result = run_e2e.run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        robot_eval_job_request=str(request_path),
        robot_eval_job_id="live/provider job",
        robot_eval_provisioner="runpod",
        robot_eval_simulator="mujoco",
        robot_eval_evaluation_substrate="wam",
        robot_eval_budget_usd=12.5,
    )

    assert calls["capture_root"] == capture_root
    assert calls["job_request"] == str(request_path)
    assert calls["job_id"] == "live-provider-job"
    assert calls["provisioner"] == "runpod"
    assert calls["simulator"] == "mujoco"
    assert calls["evaluation_substrate"] == "wam"
    assert calls["budget_usd"] == 12.5
    assert calls["allow_gpu_provisioning"] is False
    assert calls["allow_simulator_execution"] is False
    assert result["robot_eval_job"]["job_id"] == "live-provider-job"
    stage_ledger = json.loads(
        Path(result["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8")
    )
    assert stage_ledger["stages"]["robot_eval"]["status"] == "completed"
    assert stage_ledger["stages"]["robot_eval"]["artifacts"]["mode"] == "job_request"
    assert stage_ledger["stages"]["robot_eval"]["artifacts"]["job_id"] == "live-provider-job"
    provider_runtime = result["robot_eval_provider_runtime"]
    assert provider_runtime["gpu_provider_launch_request_status"] == (
        "blocked_by_explicit_provider_gate"
    )
    assert provider_runtime["prelaunch_spend_guard"]["can_launch"] is False
    assert provider_runtime["provider_race_required_for_customer_path"] is True
    assert provider_runtime["customer_path_provider_failover_wired"] is False
    assert provider_runtime["customer_path_provider_failover_runtime_wired"] is False
    assert provider_runtime["customer_path_provider_failover_runtime_status"] == (
        "blocked_pending_teardown_owned_race_launcher"
    )
    assert provider_runtime["serial_provider_launch_blocked_unless_override"] is True
    assert provider_runtime["live_provider_calls_performed"] is False
    assert provider_runtime["remote_cloud_execution_proven"] is False
    assert provider_runtime["claim_boundary"][
        "run_e2e_robot_eval_handoff_is_not_provider_execution"
    ] is True


def test_robot_eval_provider_runtime_summary_reads_provider_race_launcher_result(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "robot-job"
    job_dir.mkdir()
    (job_dir / "gpu_provider_launch_request.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_gpu_provider_launch_request.v1",
                "status": "request_manifest_ready",
                "provider": "runpod",
                "live_provider_calls_performed": False,
                "prelaunch_spend_guard": {
                    "schema_version": "robot_eval_provider_prelaunch_spend_guard.v1",
                    "can_launch": True,
                    "provider_race": {
                        "schema_version": "robot_eval_provider_race_contract.v1",
                        "race_required_for_customer_path": True,
                        "customer_path_provider_failover_wired": False,
                        "customer_path_provider_failover_runtime_wired": False,
                        "provider_race_handoff_path": (
                            "gpu_provider_race_handoff.json"
                        ),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "gpu_provider_race_handoff.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_gpu_provider_race_handoff.v1",
                "status": "ready_for_customer_provider_race_runtime",
                "provider_race_required_for_customer_path": True,
                "customer_path_provider_failover_handoff_wired": True,
                "customer_path_provider_failover_runtime_wired": True,
                "provider_race_runtime_launcher_available": True,
                "launcher_command": "blueprint-run-robot-eval-provider-race",
                "serial_provider_launch_default_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "gpu_provider_race_launcher_result.json").write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_gpu_provider_race_launcher_result.v1",
                "status": "ready_for_live_provider_race",
                "provider_race_launcher_available": True,
                "provider_race_execution_proven": "true",
                "live_provider_calls_performed": False,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "gpu_cost_control_ledger.json").write_text(
        json.dumps({"status": "blocked_before_allocation"}),
        encoding="utf-8",
    )
    (job_dir / "remote_cloud_execution_closure_manifest.json").write_text(
        json.dumps({"remote_cloud_execution_proven": False}),
        encoding="utf-8",
    )

    summary = run_e2e._robot_eval_provider_runtime_summary(
        {"job_id": "job-1", "job_dir": str(job_dir)}
    )

    assert summary is not None
    assert summary["provider_race_handoff_status"] == (
        "ready_for_customer_provider_race_runtime"
    )
    assert summary["provider_race_launcher_result_status"] == (
        "ready_for_live_provider_race"
    )
    assert summary["customer_path_provider_failover_wired"] is True
    assert summary["customer_path_provider_failover_runtime_wired"] is True
    assert summary["provider_race_launcher_ready"] is True
    assert summary["provider_race_execution_proven"] is False
    assert summary["live_provider_calls_performed"] is False
    assert summary["remote_cloud_execution_proven"] is False
    assert summary["serial_provider_launch_blocked_unless_override"] is True
    assert summary["claim_boundary"][
        "provider_race_launcher_result_is_not_provider_execution"
    ] is True


def test_run_end_to_end_blocks_ambiguous_robot_eval_sources(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    with pytest.raises(PipelineError, match="either robot_eval_job_request"):
        run_e2e.run_end_to_end(
            capture_root=str(capture_root),
            provider="openai",
            robot_eval_job_request="request.json",
            robot_eval_request_inbox="inbox",
        )


def test_run_e2e_main_success_and_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    capture_root = _capture_root(tmp_path)

    def fake_run_end_to_end(**kwargs):
        assert kwargs["provider"] == "openai"
        assert kwargs["openai_phase2_config"].mode == "codex_cli"
        assert kwargs["openai_phase2_config"].model == "gpt-test"
        assert kwargs["openai_phase2_config"].codex_bin == "codex-test"
        assert kwargs["openai_phase2_config"].timeout_seconds == 3
        assert kwargs["openai_phase2_config"].reasoning_effort == "low"
        assert kwargs["run_evaluation_prep"] is True
        assert kwargs["run_cosmos_validation"] is True
        assert kwargs["robot_eval_job_request"] == "robot-request.json"
        assert kwargs["robot_eval_job_id"] == "robot-job"
        assert kwargs["robot_eval_provisioner"] == "runpod"
        assert kwargs["robot_eval_simulator"] == "mujoco"
        assert kwargs["robot_eval_evaluation_substrate"] == "wam"
        assert kwargs["robot_eval_budget_usd"] == 3.5
        assert kwargs["allow_robot_eval_gpu_provisioning"] is True
        assert kwargs["allow_robot_eval_simulator_execution"] is True
        assert kwargs["resume_completed_stages"] is True
        return {
            "preflight_status": "ready",
            "pipeline_status": "completed",
            "pipeline_lanes": ["all"],
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
            "evaluation_prep": {"manifest_path": "eval.json"},
            "cosmos_validation": {"status": "completed"},
            "robot_eval_job": {"manifest_path": "robot-job.json"},
            "robot_eval_request_inbox": None,
        }

    monkeypatch.setattr(run_e2e, "run_end_to_end", fake_run_end_to_end)
    assert run_e2e.main(
        [
            "--capture-root",
            str(capture_root),
            "--provider",
            "openai",
            "--pipeline-lane",
            "all",
            "--openai-phase2-mode",
            "codex_cli",
            "--openai-phase2-model",
            "gpt-test",
            "--openai-phase2-codex-bin",
            "codex-test",
            "--openai-phase2-timeout-seconds",
            "3",
            "--openai-phase2-reasoning-effort",
            "low",
            "--run-evaluation-prep",
            "--run-cosmos-validation",
            "--robot-eval-job-request",
            "robot-request.json",
            "--robot-eval-job-id",
            "robot-job",
            "--robot-eval-provisioner",
            "runpod",
            "--robot-eval-simulator",
            "mujoco",
            "--robot-eval-evaluation-substrate",
            "wam",
            "--robot-eval-budget-usd",
            "3.5",
            "--allow-robot-eval-gpu-provisioning",
            "--allow-robot-eval-simulator-execution",
            "--resume-completed-stages",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "preflight_status=ready" in output
    assert "evaluation_prep=eval.json" in output
    assert "cosmos_validation=completed" in output
    assert "robot_eval_job=robot-job.json" in output

    monkeypatch.setattr(run_e2e, "run_end_to_end", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run_e2e.main(["--capture-root", str(capture_root), "--provider", "claude"]) == 1
    assert "[run-e2e] FAILED: boom" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["run-e2e", "--capture-root", str(capture_root), "--provider", "claude"],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.run_e2e", run_name="__main__")
    assert excinfo.value.code == 1
