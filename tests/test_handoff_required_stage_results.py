"""Required stages cannot be committed or resumed on a non-success result.

Covers (for impacted-test selection):
  src/blueprint_pipeline/run_e2e.py
  src/blueprint_pipeline/pubsub_handoff_listener.py
"""

import json

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import run_e2e
from blueprint_pipeline.common import PipelineError


@pytest.mark.parametrize(
    "status",
    [
        "blocked",
        "failed",
        "disabled",
        "not_started",
        "unknown",
        None,
        "preauthorized_complete_with_failures",
    ],
)
def test_required_supervisor_prevents_success_ack(status):
    result = {
        "pipeline_status": "completed",
        "final_bundle_path": "retained.json",
        "task_evaluation_supervisor": {"status": status},
    }
    disposition, blockers = listener._handoff_result_disposition(result)
    assert disposition == "retryable_blocked"
    assert blockers == [
        f"required_stage_not_complete:task_evaluation_supervisor:{status or 'missing_status'}"
    ]


def _setup(monkeypatch, tmp_path):
    root = tmp_path / "bucket/scenes/site-1/captures/cap-1"
    root.mkdir(parents=True)
    (root / "capture_descriptor.json").write_text("{}")
    calls = {"capture": 0, "supervisor": 0}
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _: {"status": "ready"})

    def capture(**_kwargs):
        calls["capture"] += 1
        return {"status": "completed", "lanes": ["qualification"]}

    monkeypatch.setattr(run_e2e, "run_capture_pipeline", capture)
    return root, calls


@pytest.mark.parametrize("status", ["blocked", "failed", "not_started", None])
def test_required_supervisor_failure_retains_prefix_and_retries_only_failed_stage(
    monkeypatch, tmp_path, status
):
    root, calls = _setup(monkeypatch, tmp_path)

    def supervisor(**_kwargs):
        calls["supervisor"] += 1
        return {"status": status if calls["supervisor"] == 1 else "non_spend_complete"}

    monkeypatch.setattr(run_e2e, "run_capture_build_supervisor", supervisor)
    with pytest.raises(
        PipelineError, match="required_stage_not_complete:task_evaluation_supervisor"
    ):
        run_e2e.run_end_to_end(capture_root=str(root), provider="manual")
    ledger = json.loads((root / "pipeline/run_e2e_stage_ledger.json").read_text())
    assert ledger["status"] == "failed"
    assert ledger["failed_stage"] == "task_evaluation_supervisor"
    assert ledger["stages"]["capture_pipeline"]["status"] == "completed"
    result = run_e2e.run_end_to_end(
        capture_root=str(root), provider="manual", resume_completed_stages=True
    )
    assert result["task_evaluation_supervisor"]["status"] == "non_spend_complete"
    assert calls == {"capture": 1, "supervisor": 2}


def test_old_required_failure_snapshot_is_not_reused():
    ledger = {
        "stages": {
            "capture_pipeline": {
                "status": "completed",
                "resume_result_snapshot_available": True,
                "result_snapshot": {"status": "completed_with_lane_failures"},
            }
        }
    }
    assert run_e2e._completed_stage_resume_snapshot(ledger, stage="capture_pipeline") is None


def test_old_supervisor_completion_reopens_but_preserves_receipt(tmp_path):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline_job_ledger.json").write_text(
        json.dumps({"status": "completed", "attempt_count": 1})
    )
    (tmp_path / "pipeline_job_output_commit.json").write_text(
        json.dumps({"status": "committed", "result_sha256": "retained"})
    )
    (tmp_path / "pipeline/run_e2e_stage_ledger.json").write_text(
        json.dumps(
            {"stages": {"task_evaluation_supervisor": {"result_snapshot": {"status": "blocked"}}}}
        )
    )
    status, ledger = listener._claim_job_lease(
        tmp_path, scene_id="s", capture_id="c", owner="worker", lease_seconds=900
    )
    assert status == "claimed"
    assert ledger["attempt_count"] == 2
    retained = json.loads((tmp_path / "pipeline_job_output_commit.json").read_text())
    assert retained["result_sha256"] == "retained"
    assert retained["status"] != "committed"


@pytest.mark.parametrize(
    "status", ["non_spend_complete", "advise_complete", "shadow_complete", "preauthorized_complete"]
)
def test_completed_supervisor_preserves_optional_readiness_states(status):
    result = {
        "pipeline_status": "completed",
        "task_evaluation_supervisor": {"status": status},
        "evaluation_prep": None,
        "site_package_manifest": {"status": "blocked"},
    }
    assert listener._handoff_result_disposition(result) == ("terminal_success", [])


def test_real_required_supervisor_block_cannot_complete_capture(monkeypatch, tmp_path):
    root, calls = _setup(monkeypatch, tmp_path)
    with pytest.raises(
        PipelineError, match="required_stage_not_complete:task_evaluation_supervisor:blocked"
    ):
        run_e2e.run_end_to_end(capture_root=str(root), provider="manual")
    assert calls["capture"] == 1
    reports = list(
        (root / "pipeline/task_evaluation_supervisor/runs").glob(
            "*/terminal_supervisor_report.json"
        )
    )
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == "blocked"


@pytest.mark.parametrize("status", ["pending", "unknown", "not_started", None])
def test_required_pipeline_unknown_cannot_succeed_from_final_bundle(status):
    disposition, blockers = listener._handoff_result_disposition(
        {"pipeline_status": status, "final_bundle_path": "previous-bundle.json"}
    )
    assert disposition == "retryable_blocked"
    assert blockers == [
        f"required_stage_not_complete:capture_pipeline:{status or 'missing_status'}"
    ]


def test_expired_lease_cannot_recover_commit_with_required_failure(tmp_path):
    listener._write_output_commit(
        tmp_path,
        scene_id="s",
        capture_id="c",
        attempt_count=1,
        result={"pipeline_status": "completed"},
    )
    (tmp_path / "pipeline").mkdir()
    ledger_path = tmp_path / "pipeline/run_e2e_stage_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "stages": {
                    "task_evaluation_supervisor": {
                        "status": "completed",
                        "result_snapshot": {"status": "blocked"},
                    }
                }
            }
        )
    )
    assert listener._output_commit(tmp_path, scene_id="s", capture_id="c") == {}
    assert (
        json.loads((tmp_path / listener.JOB_OUTPUT_COMMIT_FILENAME).read_text())["status"]
        == "committed"
    )
    # Unchanged historical receipts without a retained failure are not guessed
    # invalid; remediation requires evidence and cannot replay arbitrary work.
    ledger_path.unlink()
    assert listener._output_commit(tmp_path, scene_id="s", capture_id="c")["attempt_count"] == 1
