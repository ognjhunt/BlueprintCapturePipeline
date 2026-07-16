"""Tests for the buyer-facing Task Evaluation Run report composer."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_scorecard_withholds_numeric_rates_at_no_claim() -> None:
    # A no_claim run has no task-success grounding; publishing success_rate +
    # "completed" lets a buyer anchor on numbers the ladder never earned.
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(3, 1), evidence_level="no_claim"
    )
    assert scorecard["status"] == "rates_withheld_insufficient_evidence"
    assert scorecard["rates_published"] is False
    row = scorecard["conditions"][0]
    assert row["trials"] == 4 and row["successes"] == 3  # factual counts kept
    assert row["success_rate"] is None  # numeric interval withheld


def test_scorecard_withholds_numeric_rates_at_media_valid() -> None:
    # media_valid means the media is decodable — still not a task-success claim.
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(3, 1), evidence_level="media_valid"
    )
    assert scorecard["status"] == "rates_withheld_insufficient_evidence"
    assert scorecard["conditions"][0]["success_rate"] is None


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


def test_report_blocks_when_cleared_rights_gate_carries_blockers_or_takedown() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={
            "status": "cleared",
            "cleared": True,
            "blockers": ["manual_rights_review_required"],
            "revocation_takedown": {"status": "takedown_required"},
        },
    )

    assert report["status"] == "blocked"
    assert (
        "rights_privacy_gate_blocker:manual_rights_review_required"
        in report["blockers"]
    )
    assert (
        "rights_privacy_gate_consent_revoked_takedown_required"
        in report["blockers"]
    )


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


def test_report_refuses_nested_provider_task_success_claims() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(2, 0)},
        success_claim_ledger=ledger,
        provider_execution={
            "pod_status": "EXITED",
            "runtime": {
                "exit_code": 0,
                "taskSuccess": True,
                "deploymentReady": True,
            },
            "summaries": [
                {
                    "provider": "runpod",
                    "successRate": 1.0,
                    "gpu_seconds": 42,
                }
            ],
        },
        rights_privacy_gate={"status": "cleared"},
    )

    provider = report["provider_execution"]
    assert provider["runtime"] == {"exit_code": 0}
    assert provider["summaries"] == [{"provider": "runpod", "gpu_seconds": 42}]
    assert provider["refused_task_success_keys"] == [
        "runtime.deploymentReady",
        "runtime.taskSuccess",
        "summaries[0].successRate",
    ]
    assert (
        "provider_payload_attempted_task_success_claim:runtime.taskSuccess"
        in report["blockers"]
    )
    assert (
        "provider_payload_attempted_task_success_claim:runtime.deploymentReady"
        in report["blockers"]
    )
    assert (
        "provider_payload_attempted_task_success_claim:summaries[0].successRate"
        in report["blockers"]
    )
    assert report["status"] == "blocked"


def test_report_redacts_policy_binding_secrets() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        policy_binding={
            "policy_id": "policy-1",
            "checkpoint_id": "ckpt-1",
            "reference": {
                "endpoint": "https://policy.example",
                "api_key": "sk-live-secret",
                "openai_api_key": "sk-openai-secret",
            },
            "fallbacks": [
                {
                    "provider": "backup",
                    "bearerToken": "token-123",
                    "access_key_id": "AKIA_TEST",
                    "auth": {"mode": "basic", "service_password": "pw"},
                    "aws_secret_access_key": "secret-test",
                    "provider_credentials": {"username": "service", "password": "pw"},
                    "runpod_api_token": "rp-live-token",
                    "secret_access_key": "secret-test",
                }
            ],
        },
        rights_privacy_gate={"status": "cleared"},
    )

    binding = report["policy_binding"]
    assert binding["reference"]["api_key"] == "<redacted>"
    assert binding["reference"]["openai_api_key"] == "<redacted>"
    assert binding["fallbacks"][0]["access_key_id"] == "<redacted>"
    assert binding["fallbacks"][0]["auth"]["mode"] == "basic"
    assert binding["fallbacks"][0]["auth"]["service_password"] == "<redacted>"
    assert binding["fallbacks"][0]["aws_secret_access_key"] == "<redacted>"
    assert binding["fallbacks"][0]["bearerToken"] == "<redacted>"
    assert binding["fallbacks"][0]["provider_credentials"] == "<redacted>"
    assert binding["fallbacks"][0]["runpod_api_token"] == "<redacted>"
    assert binding["fallbacks"][0]["secret_access_key"] == "<redacted>"
    assert binding["reference"]["endpoint"] == "https://policy.example"
    assert binding["checkpoint_id"] == "ckpt-1"
    assert binding["secret_values_redacted"] is True
    assert binding["redacted_secret_paths"] == [
        "fallbacks[0].access_key_id",
        "fallbacks[0].auth.service_password",
        "fallbacks[0].aws_secret_access_key",
        "fallbacks[0].bearerToken",
        "fallbacks[0].provider_credentials",
        "fallbacks[0].runpod_api_token",
        "fallbacks[0].secret_access_key",
        "reference.api_key",
        "reference.openai_api_key",
    ]
    assert (
        "policy_binding_secret_value_redacted:reference.api_key"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:fallbacks[0].bearerToken"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:fallbacks[0].access_key_id"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:fallbacks[0].secret_access_key"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:reference.openai_api_key"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:fallbacks[0].runpod_api_token"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:fallbacks[0].aws_secret_access_key"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:"
        "fallbacks[0].auth.service_password"
        in report["blockers"]
    )
    assert (
        "policy_binding_secret_value_redacted:fallbacks[0].provider_credentials"
        in report["blockers"]
    )
    assert report["status"] == "blocked"


def test_report_preserves_already_redacted_policy_binding_secrets() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        policy_binding={
            "policy_id": "policy-1",
            "reference": {"api_key": "<redacted>"},
        },
        rights_privacy_gate={"status": "cleared"},
    )

    binding = report["policy_binding"]
    assert binding["reference"]["api_key"] == "<redacted>"
    assert binding["secret_values_redacted"] is True
    assert binding["redacted_secret_paths"] == ["reference.api_key"]
    assert not any(
        blocker.startswith("policy_binding_secret_value_redacted")
        for blocker in report["blockers"]
    )
    assert report["status"] == "ready_review_required"


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


def _wam_gate_payload(
    *,
    granted_grade: str,
    consistency_score: float | None,
    anchors_passed: bool,
    anchor_set: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "wam_score_claim_gate.v1",
        "status": "granted",
        "requested_grade": granted_grade,
        "granted_grade": granted_grade,
        "max_allowed_grade": granted_grade,
        "consistency_measured_and_passed": consistency_score is not None,
        "calibration_anchors_present_and_passed": anchors_passed,
        "consistency": {
            "status": "scored" if consistency_score is not None else "missing",
            "consistency_score": consistency_score,
            "passed": consistency_score is not None,
            "blockers": [],
        },
        "calibration_anchors": {
            "anchors_present": anchors_passed,
            "anchors_passed": anchors_passed,
            "anchor_set": anchor_set if anchor_set is not None else [],
            "anchor_validation_status": "recovered" if anchors_passed else None,
            "spearman_rank_correlation_vs_expected": 1.0 if anchors_passed else None,
        },
        "blockers": [],
    }


def test_report_without_wam_evaluation_stays_unchanged() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
    )
    assert report["wam_evaluation"] is None
    assert not any(b.startswith("wam_evaluation") for b in report["blockers"])


def test_report_demotes_above_review_wam_claim_without_evidence() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(2, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
        wam_evaluation=_wam_gate_payload(
            granted_grade="calibrated_evaluator_grade",
            consistency_score=None,
            anchors_passed=False,
        ),
    )
    section = report["wam_evaluation"]
    assert section["wam_score_claim_grade"] == "fixture_evaluator_only"
    assert "wam_score_without_consistency_or_calibration" in section["blockers"]
    assert (
        "wam_evaluation:wam_score_without_consistency_or_calibration"
        in report["blockers"]
    )
    assert report["status"] == "blocked"


def test_report_preserves_calibrated_wam_claim_with_evidence() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    anchor_set = ["policy_clean", "policy_clean_noise_0p1", "policy_clean_noise_0p3"]
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(2, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
        wam_evaluation=_wam_gate_payload(
            granted_grade="calibrated_evaluator_grade",
            consistency_score=0.93,
            anchors_passed=True,
            anchor_set=anchor_set,
        ),
    )
    section = report["wam_evaluation"]
    assert section["wam_score_claim_grade"] == "calibrated_evaluator_grade"
    assert section["consistency_score"] == 0.93
    assert section["calibration_anchor_set"] == anchor_set
    assert section["calibration_anchors_passed"] is True
    assert not any(b.startswith("wam_evaluation") for b in report["blockers"])


def test_report_wam_section_always_shows_anchor_set_and_consistency_number() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
        wam_evaluation=_wam_gate_payload(
            granted_grade="review_grade",
            consistency_score=None,
            anchors_passed=False,
        ),
    )
    section = report["wam_evaluation"]
    # never a bare score: grade always travels with the consistency number
    # and the anchor set, even when unmeasured/absent
    assert "consistency_score" in section
    assert "calibration_anchor_set" in section
    assert "score" not in section
    assert section["bare_score_forbidden"] is True


def test_report_rejects_unrecognized_wam_claim_grade() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
        wam_evaluation=_wam_gate_payload(
            granted_grade="deployment_grade",
            consistency_score=0.99,
            anchors_passed=True,
        ),
    )
    section = report["wam_evaluation"]
    assert section["wam_score_claim_grade"] == "fixture_evaluator_only"
    assert any(b.startswith("wam_evaluation") for b in report["blockers"])


def test_report_recomputes_wam_evidence_from_nested_payload() -> None:
    ledger = build_success_claim_ledger(task_metadata=None)
    tampered = _wam_gate_payload(
        granted_grade="calibrated_evaluator_grade",
        consistency_score=0.99,
        anchors_passed=True,
        anchor_set=["policy_clean", "policy_noisy"],
    )
    tampered["consistency_measured_and_passed"] = True
    tampered["calibration_anchors_present_and_passed"] = True
    tampered["consistency"]["status"] = "blocked"
    tampered["consistency"]["passed"] = False
    tampered["calibration_anchors"]["anchors_passed"] = False

    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
        wam_evaluation=tampered,
    )

    section = report["wam_evaluation"]
    assert section["wam_score_claim_grade"] == "fixture_evaluator_only"
    assert "wam_consistency_claim_flag_without_passing_nested_evidence" in section["blockers"]
    assert "wam_calibration_claim_flag_without_passing_nested_evidence" in section["blockers"]
    assert (
        "wam_evaluation:wam_score_without_consistency_or_calibration"
        in report["blockers"]
    )


def _revoked_capture(tmp_path: Path) -> Path:
    capture_root = tmp_path / "scenes" / "s" / "captures" / "c"
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "raw" / "rights_consent.json").write_text(
        json.dumps(
            {
                "consent_status": "revoked",
                "consent_revoked": True,
                "consent_revoked_at": "2026-07-04T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return capture_root


def test_report_blocks_on_live_consent_revocation_despite_clean_gate(tmp_path) -> None:
    # Rights gate mapping is CLEARED with no revocation — the stale-manifest case.
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared", "cleared": True},
        capture_root=_revoked_capture(tmp_path),
    )
    assert report["status"] == "blocked"
    assert (
        "rights_privacy_gate_consent_revoked_takedown_required" in report["blockers"]
    )


# ---------------------------------------------------------------------------
# R066: industrial-assembly success-metric semantics.
# ---------------------------------------------------------------------------


def _industrial_attempts(family: str = "insertion") -> list[dict]:
    return [
        {
            "attempt_id": "ins_1",
            "task_id": "peg-insert",
            "scenario_id": "clear",
            "success": True,
            "task_family": family,
            "industrial_metrics": {
                "placement_accuracy_m": {
                    "value": 0.0012,
                    "provenance": "simulator_physics",
                },
                "force_torque_within_envelope": True,
            },
        },
        {
            "attempt_id": "ins_2",
            "task_id": "peg-insert",
            "scenario_id": "clear",
            "success": False,
            "task_family": family,
        },
    ]


def test_scorecard_surfaces_industrial_metrics_for_assembly_family() -> None:
    scorecard = build_task_eval_scorecard(
        attempts=_industrial_attempts(),
        evidence_level="review_task_success",
        task_metadata={"industrial_metrics": {"insertion_tolerance_m": 0.002}},
    )
    row = scorecard["conditions"][0]
    block = row["industrial_success_metrics"]
    assert block["surfaced"] is True
    assert block["surfaced_reason"] == "task_family"
    assert block["task_family_matched"] == "insertion"
    metrics = block["metrics"]

    # Measured outcome threaded in from the attempt, with its declared provenance.
    placement = metrics["placement_accuracy_m"]
    assert placement["value"] == 0.0012
    assert placement["unit"] == "meters"
    assert placement["kind"] == "measured_outcome"
    assert placement["provenance"] == "simulator_physics"
    assert placement["status"] == "declared_or_measured_present"

    # Boolean measured outcome stays a real bool, not coerced to 1.0.
    force_torque = metrics["force_torque_within_envelope"]
    assert force_torque["value"] is True
    assert force_torque["unit"] == "boolean"

    # Declared spec tolerance sourced from run-level task_metadata.
    insertion = metrics["insertion_tolerance_m"]
    assert insertion["value"] == 0.002
    assert insertion["kind"] == "declared_tolerance"

    # A metric with no supplied value is surfaced EXPLICITLY as unset, never faked.
    dimensional = metrics["dimensional_tolerance"]
    assert dimensional["value"] is None
    assert dimensional["provenance"] == "unset"
    assert dimensional["status"] == "needs_measurement_or_operator_input"
    assert "not_fabricated_or_verified_success_proof" in dimensional["claim_boundary"]

    assert block["any_metric_declared_or_measured"] is True
    assert block["all_metrics_unset"] is False

    semantics = scorecard["industrial_success_metric_semantics"]
    assert semantics["surfaced_condition_count"] == 1
    assert semantics["any_condition_industrial_assembly"] is True
    assert [f["field"] for f in semantics["fields"]] == [
        "placement_accuracy_m",
        "insertion_tolerance_m",
        "force_torque_within_envelope",
        "dimensional_tolerance",
    ]


def test_scorecard_industrial_metrics_all_unset_when_nothing_declared() -> None:
    scorecard = build_task_eval_scorecard(
        attempts=[
            {
                "attempt_id": "a1",
                "task_id": "assemble",
                "scenario_id": "s",
                "success": True,
                "task_category": "assembly",
            }
        ],
        evidence_level="review_task_success",
    )
    block = scorecard["conditions"][0]["industrial_success_metrics"]
    assert block["all_metrics_unset"] is True
    assert block["any_metric_declared_or_measured"] is False
    for entry in block["metrics"].values():
        assert entry["value"] is None
        assert entry["provenance"] == "unset"
        assert entry["status"] == "needs_measurement_or_operator_input"


def test_scorecard_industrial_metrics_keyed_off_site_taxonomy() -> None:
    # No assembly task family, but the site resolves to an industrial category via
    # site_taxonomy, so the metric semantics are still surfaced (as unset).
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(2, 1),
        evidence_level="review_task_success",
        task_metadata={"site_type": "warehouse aisle"},
    )
    block = scorecard["conditions"][0]["industrial_success_metrics"]
    assert block["surfaced_reason"] == "industrial_site_category"
    assert block["is_industrial_site"] is True
    assert block["site_category"] == "warehouse"
    assert scorecard["industrial_success_metric_semantics"]["is_industrial_site"] is True


def test_scorecard_non_industrial_condition_is_unaffected() -> None:
    # Kitchen / generic move-tote conditions carry no industrial metric block.
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(3, 1),
        evidence_level="review_task_success",
        task_metadata={"site_type": "kitchen"},
    )
    row = scorecard["conditions"][0]
    assert "industrial_success_metrics" not in row
    semantics = scorecard["industrial_success_metric_semantics"]
    assert semantics["surfaced_condition_count"] == 0
    assert semantics["any_condition_industrial_assembly"] is False


def test_scorecard_without_task_metadata_never_surfaces_industrial_block() -> None:
    # Backward-compatible default: no task_metadata, no task_family -> no block.
    scorecard = build_task_eval_scorecard(
        attempts=_attempts(5, 0), evidence_level="review_task_success"
    )
    assert "industrial_success_metrics" not in scorecard["conditions"][0]
    assert (
        scorecard["industrial_success_metric_semantics"]["surfaced_condition_count"]
        == 0
    )


def test_report_surfaces_industrial_metrics_from_task_metadata() -> None:
    ledger = build_success_claim_ledger(
        task_metadata={"task_id": "peg-insert"},
        media_validity={"status": "PASS", "blockers": []},
        review_task_success={"status": "PASS", "blockers": []},
    )
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _industrial_attempts()},
        task_metadata={
            "task_id": "peg-insert",
            "site_type": "factory line-side station",
            "industrial_metrics": {"dimensional_tolerance": 0.0005},
        },
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared"},
    )
    block = report["scorecard"]["conditions"][0]["industrial_success_metrics"]
    assert block["surfaced_reason"] == "task_family_and_industrial_site_category"
    assert block["metrics"]["dimensional_tolerance"]["value"] == 0.0005
    assert block["metrics"]["placement_accuracy_m"]["value"] == 0.0012
    # The industrial metrics are declared/measured inputs — they never become a
    # top-level success claim.
    assert "success" not in report
    assert report["status"] == "ready_review_required"


def test_report_not_blocked_by_live_read_when_consent_documented(tmp_path) -> None:
    capture_root = tmp_path / "scenes" / "s" / "captures" / "c"
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "raw" / "rights_consent.json").write_text(
        json.dumps({"consent_status": "documented"}), encoding="utf-8"
    )
    ledger = build_success_claim_ledger(task_metadata=None)
    report = build_task_eval_run_report(
        job_id="job-1",
        attempt_trace={"attempts": _attempts(1, 0)},
        success_claim_ledger=ledger,
        rights_privacy_gate={"status": "cleared", "cleared": True},
        capture_root=capture_root,
    )
    assert (
        "rights_privacy_gate_consent_revoked_takedown_required"
        not in report["blockers"]
    )
