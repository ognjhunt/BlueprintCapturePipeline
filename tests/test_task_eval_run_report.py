"""Tests for the buyer-facing Task Evaluation Run report composer."""

from __future__ import annotations

from blueprint_pipeline.success_claim_contracts import build_success_claim_ledger
from blueprint_pipeline.task_eval_run_report import (
    RECOMMENDED_MIN_TRIALS_PER_CONDITION,
    TASK_EVAL_RUN_REPORT_SCHEMA_VERSION,
    build_task_eval_run_report,
    build_task_eval_scorecard,
)


def _attempts(n_success: int, n_fail: int) -> list[dict]:
    rows = []
    for i in range(n_success):
        rows.append(
            {
                "attempt_id": f"ok_{i}",
                "task_id": "move-tote",
                "scenario_id": "clear-path",
                "success": True,
            }
        )
    for i in range(n_fail):
        rows.append(
            {
                "attempt_id": f"bad_{i}",
                "task_id": "move-tote",
                "scenario_id": "clear-path",
                "success": False,
            }
        )
    return rows


def test_scorecard_never_emits_bare_rate_without_trials_and_interval() -> None:
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(8, 2), evidence_level="review_task_success"
    )
    assert scorecard["status"] == "completed"
    row = scorecard["conditions"][0]
    assert row["trials"] == 10
    assert row["successes"] == 8
    interval = row["success_rate"]
    assert interval["point"] == 0.8
    assert 0.0 <= interval["lower_95"] < 0.8 < interval["upper_95"] <= 1.0
    assert row["below_recommended_trials"] is True
    assert (
        scorecard["recommended_min_trials_per_condition"]
        == RECOMMENDED_MIN_TRIALS_PER_CONDITION
    )


def test_scorecard_rejects_non_boolean_success_labels() -> None:
    attempts = [
        {"attempt_id": "a1", "task_id": "t", "scenario_id": "s", "success": "true"},
    ]
    scorecard = build_task_eval_scorecard(
        attempts=attempts, evidence_level="no_claim"
    )
    assert scorecard["status"] == "blocked"
    assert scorecard["invalid_attempt_ids"] == ["a1"]
    assert any(
        b.startswith("attempts_with_non_boolean_success_label")
        for b in scorecard["blockers"]
    )


def test_report_blocks_without_ledger_and_rights_gate() -> None:
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
    )
    assert report["schema_version"] == TASK_EVAL_RUN_REPORT_SCHEMA_VERSION
    assert report["status"] == "blocked"
    assert "success_claim_ledger_missing" in report["blockers"]
    assert "rights_privacy_gate_missing" in report["blockers"]
    assert report["evidence_level"] == "no_claim"


def test_report_blocks_when_rights_gate_is_present_but_not_cleared() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "review_required", "cleared": False},
    )

    assert report["status"] == "blocked"
    assert "rights_privacy_gate_not_cleared" in report["blockers"]


def test_report_scopes_success_language_to_ledger_claim() -> None:
    ledger = build_success_claim_ledger(
        task_metadata={"task_id": "move-tote"},
        media_validity={"status": "PASS", "blockers": []},
        review_task_success={"status": "PASS", "blockers": []},
    )
    report = build_task_eval_run_report(
        job_id="job-1",
        scene_id="scene-1",
        capture_id="capture-1",
        attempt_trace={"attempts": _attempts(20, 5)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
    )
    assert report["evidence_level"] == "review_task_success"
    assert report["scorecard"]["evidence_level"] == "review_task_success"
    # no top-level bare success boolean is ever exposed
    assert "success" not in report
    assert "task_success" not in report
    assert report["status"] == "ready_review_required"


def test_report_refuses_provider_task_success_claims() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(2, 0)},
        success_claim_ledger=ledger,
        provider_execution={
            "pod_status": "EXITED",
            "task_success": True,
            "success_rate": 1.0,
        },
        rights_privacy_gate={"status": "cleared"},
    )
    provider = report["provider_execution"]
    assert "task_success" not in provider
    assert "success_rate" not in provider
    assert provider["refused_task_success_keys"] == ["success_rate", "task_success"]
    assert any(
        b.startswith("provider_payload_attempted_task_success_claim")
        for b in report["blockers"]
    )
    assert report["status"] == "blocked"
    assert provider["pod_status"] == "EXITED"


def test_report_carries_safety_claim_boundary() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 1)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
    )
    safety = report["claim_boundary"]["safety"]
    assert safety["results_are_evidence_inputs_only"] is True
    assert safety["safety_or_compliance_claimed"] is False
    assert safety["deployment_readiness_claimed"] is False
    assert report["claim_boundary"]["provider_runtime_success_is_not_task_success"] is True


def test_layers_input_composes_ledger_inline() -> None:
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(3, 1)},
        success_claim_layers={
            "media_validity": {"status": "PASS", "blockers": []},
        },
        rights_privacy_gate={"status": "cleared"},
    )
    assert report["evidence_level"] == "media_valid"
    assert report["success_claim_ledger"]["highest_truthful_claim"] == "media_valid"
