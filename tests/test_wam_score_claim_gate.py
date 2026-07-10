from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.wam_score_claim_gate import (
    CALIBRATION_ANCHOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    CALIBRATION_ANCHOR_VALIDATION_METHOD,
    SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    WAM_SCORE_CLAIM_GRADES,
    WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER,
    apply_wam_score_claim_gate,
    evaluate_wam_calibration_anchors,
    score_wam_consistency,
    score_wam_rollout_set_consistency,
)


ANCHOR_VALIDATION_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)
EXECUTOR_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x08" * 32)


@pytest.fixture(autouse=True)
def _trusted_anchor_validation_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    public_key = ANCHOR_VALIDATION_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        CALIBRATION_ANCHOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public_key).hexdigest(),
    )
    executor_public_key = EXECUTOR_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(executor_public_key).hexdigest(),
    )


def _trajectory(points: list[list[float]], *, start: float = 0.0, dt: float = 0.1) -> dict:
    return {
        "trajectory": [
            {"timestamp": start + index * dt, "position": point}
            for index, point in enumerate(points)
        ]
    }


def _passing_anchor_validation(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    expected_ranking = [
        "policy_clean",
        "policy_clean_noise_0p1",
        "policy_clean_noise_0p3",
    ]
    ladder = root / "policy_ranking_ladder.json"
    ladder.write_text(
        json.dumps(
            {
                "schema_version": "policy_ranking_ladder.v1",
                "inner_checkpoint_sha256": "a" * 64,
                "inner_command_configured": True,
                "registered_action_bounds_sha256": "b" * 64,
                "expected_ranking": expected_ranking,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    scorecard = root / "policy_ranking_scorecard.json"
    scorecard.write_text(
        json.dumps(
            {
                "schema_version": "policy_ranking_scorecard.v1",
                "status": "completed",
                "policy_rankings": [
                    {
                        "policy_id": policy_id,
                        "predicted_success_rate": 1.0 - index * 0.3,
                    }
                    for index, policy_id in enumerate(expected_ranking)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "policy_ranking_ladder_validation.v1",
        "status": "recovered",
        "ranker_ordering_recovered": True,
        "validation_method": CALIBRATION_ANCHOR_VALIDATION_METHOD,
        "source_validation_recomputed": True,
        "executor_trusted_public_key_sha256": hashlib.sha256(
            EXECUTOR_PRIVATE_KEY.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).hexdigest(),
        "score_field": "predicted_success_rate",
        "minimum_replicate_seed_count": 3,
        "replicate_seed_count_by_policy": dict.fromkeys(expected_ranking, 3),
        "empirical_ground_truth_accepted_by_policy": dict.fromkeys(expected_ranking, True),
        "expected_ranking": expected_ranking,
        "expected_ranking_basis": "signed_matched_runtime_outcomes",
        "spearman_rank_correlation_vs_expected": 1.0,
        "blockers": [],
        "source_artifact_bindings": {
            "ladder": {
                "artifact_id": ladder.name,
                "sha256": hashlib.sha256(ladder.read_bytes()).hexdigest(),
            },
            "scorecard": {
                "artifact_id": scorecard.name,
                "sha256": hashlib.sha256(scorecard.read_bytes()).hexdigest(),
            },
        },
    }
    public_key = ANCHOR_VALIDATION_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    signed_payload_sha256 = hashlib.sha256(message).hexdigest()
    report = root / "policy-ladder-validation-signature-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": signed_payload_sha256,
                "signer_key_id": "policy-ladder-validation-authority",
                "verifier_id": "blueprint-test-verifier",
                "source_artifact_bindings_sha256": hashlib.sha256(
                    json.dumps(
                        payload["source_artifact_bindings"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload["validation_attestation"] = {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "authority_role": "policy_ladder_validation_authority",
        "signer_key_id": "policy-ladder-validation-authority",
        "verifier_id": "blueprint-test-verifier",
        "public_key_base64": base64.b64encode(public_key).decode(),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(ANCHOR_VALIDATION_PRIVATE_KEY.sign(message)).decode(),
        "signed_payload_sha256": signed_payload_sha256,
        "verification_report_artifact": {
            "artifact_id": report.name,
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }
    return payload


def _passing_anchor_check(root: Path) -> dict:
    return evaluate_wam_calibration_anchors(
        _passing_anchor_validation(root),
        allowed_source_root=root,
    )


# --- score_wam_consistency -------------------------------------------------


def test_matching_trajectories_score_high_and_pass() -> None:
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    result = score_wam_consistency(_trajectory(points), _trajectory(points))
    assert result["status"] == "scored"
    assert result["passed"] is True
    assert result["consistency_score"] == pytest.approx(1.0)
    assert result["temporal_consistency"] == pytest.approx(1.0)
    assert result["geometric_consistency"] == pytest.approx(1.0)
    assert result["compared_step_count"] == 4


def test_diverging_trajectory_scores_low_and_fails() -> None:
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
    rollout = _trajectory([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [4.0, 4.0, 0.0], [6.0, 6.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "scored"
    assert result["passed"] is False
    assert result["consistency_score"] is not None
    assert result["consistency_score"] < 0.5


def test_non_monotonic_timestamps_degrade_temporal_consistency() -> None:
    rollout = {
        "trajectory": [
            {"timestamp": 0.0, "position": [0.0, 0.0, 0.0]},
            {"timestamp": 0.2, "position": [0.1, 0.0, 0.0]},
            {"timestamp": 0.1, "position": [0.2, 0.0, 0.0]},
            {"timestamp": 0.3, "position": [0.3, 0.0, 0.0]},
        ]
    }
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "scored"
    assert result["temporal_consistency"] < 1.0
    assert result["passed"] is False


def test_missing_trajectory_fails_closed() -> None:
    result = score_wam_consistency({}, _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    assert result["status"] == "blocked"
    assert result["consistency_score"] is None
    assert result["passed"] is False
    assert any("rollout_trajectory" in blocker for blocker in result["blockers"])


def test_missing_reference_fails_closed() -> None:
    result = score_wam_consistency(_trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), {})
    assert result["status"] == "blocked"
    assert result["passed"] is False
    assert any("reference_trajectory" in blocker for blocker in result["blockers"])


def test_non_finite_values_fail_closed() -> None:
    rollout = _trajectory([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "blocked"
    assert result["passed"] is False
    assert "non_finite_trajectory_values" in result["blockers"]


def test_dimension_mismatch_fails_closed() -> None:
    rollout = {
        "trajectory": [
            {"timestamp": 0.0, "position": [0.0, 0.0]},
            {"timestamp": 0.1, "position": [0.1, 0.0]},
        ]
    }
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "blocked"
    assert "trajectory_dimension_mismatch" in result["blockers"]


def test_reference_accepts_bare_step_sequence_with_waypoints() -> None:
    reference = [
        {"timestamp": 0.0, "waypoint": [0.0, 0.0, 0.0]},
        {"timestamp": 0.1, "waypoint": [0.1, 0.0, 0.0]},
    ]
    rollout = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_consistency(rollout, reference)
    assert result["status"] == "scored"
    assert result["passed"] is True


def test_consistency_claim_boundary_never_upgrades_success() -> None:
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    result = score_wam_consistency(_trajectory(points), _trajectory(points))
    boundary = result["claim_boundary"]
    assert boundary["consistency_score_is_support_signal_not_task_success"] is True
    assert boundary["consistency_score_does_not_prove_rank_fidelity"] is True


# --- score_wam_rollout_set_consistency --------------------------------------


def test_rollout_set_consistency_aggregates_conservatively() -> None:
    good = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    bad = _trajectory([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    result = score_wam_rollout_set_consistency(
        rollouts=[
            {"rollout_id": "r1", **good},
            {"rollout_id": "r2", **bad},
        ],
        reference=reference,
    )
    assert result["status"] == "scored"
    assert result["scored_rollout_count"] == 2
    assert result["consistency_score"] == min(
        row["consistency_score"] for row in result["rollout_scores"]
    )
    assert result["passed"] is False


def test_rollout_set_consistency_blocks_when_no_rollout_scoreable() -> None:
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = score_wam_rollout_set_consistency(
        rollouts=[{"rollout_id": "r1", "generated_video_path": "x.mp4"}],
        reference=reference,
    )
    assert result["status"] == "blocked"
    assert result["consistency_score"] is None
    assert result["passed"] is False


# --- evaluate_wam_calibration_anchors ---------------------------------------


def test_passing_ladder_validation_yields_present_and_passed_anchors(
    tmp_path: Path,
) -> None:
    check = _passing_anchor_check(tmp_path)
    assert check["anchors_present"] is True
    assert check["anchors_passed"] is True
    assert check["anchor_set"] == [
        "policy_clean",
        "policy_clean_noise_0p1",
        "policy_clean_noise_0p3",
    ]
    assert check["blockers"] == []
    assert all("path" not in binding for binding in check["source_artifact_bindings"].values())


def test_not_recovered_ladder_validation_fails_anchor_check(tmp_path: Path) -> None:
    validation = _passing_anchor_validation(tmp_path)
    validation["status"] = "not_recovered"
    validation["ranker_ordering_recovered"] = False
    check = evaluate_wam_calibration_anchors(validation, allowed_source_root=tmp_path)
    assert check["anchors_present"] is True
    assert check["anchors_passed"] is False


def test_missing_anchor_validation_fails_closed() -> None:
    check = evaluate_wam_calibration_anchors(None)
    assert check["anchors_present"] is False
    assert check["anchors_passed"] is False
    assert check["anchor_set"] == []
    assert "calibration_anchor_validation_missing" in check["blockers"]


def test_unrecognized_anchor_schema_fails_closed(tmp_path: Path) -> None:
    validation = _passing_anchor_validation(tmp_path)
    validation["schema_version"] = "something_else.v9"
    check = evaluate_wam_calibration_anchors(validation, allowed_source_root=tmp_path)
    assert check["anchors_present"] is False
    assert check["anchors_passed"] is False
    assert "calibration_anchor_validation_schema_unrecognized" in check["blockers"]


def test_single_anchor_set_is_too_small(tmp_path: Path) -> None:
    validation = _passing_anchor_validation(tmp_path)
    validation["expected_ranking"] = ["policy_clean"]
    check = evaluate_wam_calibration_anchors(validation, allowed_source_root=tmp_path)
    assert check["anchors_passed"] is False
    assert "calibration_anchor_set_too_small" in check["blockers"]


def test_forged_five_field_validation_cannot_unlock_calibrated_grade(
    tmp_path: Path,
) -> None:
    forged = {
        "schema_version": "policy_ranking_ladder_validation.v1",
        "status": "recovered",
        "ranker_ordering_recovered": True,
        "expected_ranking": ["policy-a", "policy-b", "policy-c"],
        "blockers": [],
    }
    check = evaluate_wam_calibration_anchors(
        forged,
        allowed_source_root=tmp_path,
    )
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=_passing_consistency(),
        calibration_anchors=check,
    )

    assert check["anchors_passed"] is False
    assert check["evidence_binding_status"] == "blocked_unverified_or_tampered"
    assert gate["granted_grade"] == "fixture_evaluator_only"


def test_tampered_source_or_unrelated_report_fails_full_binding(
    tmp_path: Path,
) -> None:
    source_tampered = _passing_anchor_validation(tmp_path / "source-tampered")
    ladder_path = (
        tmp_path
        / "source-tampered"
        / source_tampered["source_artifact_bindings"]["ladder"]["artifact_id"]
    )
    ladder_payload = json.loads(ladder_path.read_text(encoding="utf-8"))
    ladder_payload["inner_checkpoint_sha256"] = "f" * 64
    ladder_path.write_text(json.dumps(ladder_payload), encoding="utf-8")
    tampered_check = evaluate_wam_calibration_anchors(
        source_tampered,
        allowed_source_root=tmp_path / "source-tampered",
    )
    assert tampered_check["anchors_passed"] is False
    assert any("sha256_mismatch" in item for item in tampered_check["blockers"])

    wrong_report = _passing_anchor_validation(tmp_path / "wrong-report")
    report_ref = wrong_report["validation_attestation"]["verification_report_artifact"]
    report_path = tmp_path / "wrong-report" / report_ref["artifact_id"]
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["signed_payload_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    report_ref["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    wrong_report_check = evaluate_wam_calibration_anchors(
        wrong_report,
        allowed_source_root=tmp_path / "wrong-report",
    )
    assert wrong_report_check["anchors_passed"] is False
    assert any(
        "verification_report_content_mismatch" in item for item in wrong_report_check["blockers"]
    )

    unsafe_id = _passing_anchor_validation(tmp_path / "unsafe-id")
    unsafe_id["source_artifact_bindings"]["ladder"]["artifact_id"] = str(
        tmp_path / "unsafe-id" / "policy_ranking_ladder.json"
    )
    unsafe_check = evaluate_wam_calibration_anchors(
        unsafe_id,
        allowed_source_root=tmp_path / "unsafe-id",
    )
    assert unsafe_check["anchors_passed"] is False
    assert unsafe_check["source_artifact_bindings"]["ladder"]["artifact_id"] is None
    assert str(tmp_path) not in json.dumps(unsafe_check)

    nul_id = _passing_anchor_validation(tmp_path / "nul-id")
    nul_id["source_artifact_bindings"]["ladder"]["artifact_id"] = "bad\x00name.json"
    nul_check = evaluate_wam_calibration_anchors(
        nul_id,
        allowed_source_root=tmp_path / "nul-id",
    )
    assert nul_check["anchors_passed"] is False
    assert "calibration_anchor_ladder_artifact_id_unsafe" in nul_check["blockers"]
    assert nul_check["source_artifact_bindings"]["ladder"]["artifact_id"] is None

    long_id = _passing_anchor_validation(tmp_path / "long-id")
    long_id["source_artifact_bindings"]["ladder"]["artifact_id"] = "x" * 10_000
    long_check = evaluate_wam_calibration_anchors(
        long_id,
        allowed_source_root=tmp_path / "long-id",
    )
    assert long_check["anchors_passed"] is False
    assert "calibration_anchor_ladder_artifact_id_unsafe" in long_check["blockers"]
    assert long_check["source_artifact_bindings"]["ladder"]["artifact_id"] is None


def test_validation_authority_must_be_distinct_from_runtime_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _passing_anchor_validation(tmp_path)
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        validation["validation_attestation"]["public_key_sha256"],
    )

    check = evaluate_wam_calibration_anchors(
        validation,
        allowed_source_root=tmp_path,
    )

    assert check["anchors_passed"] is False
    assert (
        "calibration_anchor_validation_authority_not_independent_from_executor" in check["blockers"]
    )


def test_validation_binds_the_executor_trust_root_used_by_the_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _passing_anchor_validation(tmp_path)
    different_executor_key = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
    different_executor_public_key = different_executor_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(different_executor_public_key).hexdigest(),
    )

    check = evaluate_wam_calibration_anchors(
        validation,
        allowed_source_root=tmp_path,
    )

    assert check["anchors_passed"] is False
    assert "calibration_anchor_executor_trust_root_binding_mismatch" in check["blockers"]


def test_claim_gate_rejects_forged_normalized_anchor_check() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=_passing_consistency(),
        calibration_anchors={
            "anchors_present": True,
            "anchors_passed": True,
            "evidence_binding_status": "verified_trusted_full_binding",
            "anchor_set": ["policy-a", "policy-b"],
        },
    )

    assert gate["granted_grade"] == "fixture_evaluator_only"


# --- apply_wam_score_claim_gate ----------------------------------------------


def _passing_consistency() -> dict:
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    return score_wam_consistency(_trajectory(points), _trajectory(points))


def test_grade_ladder_orders_fixture_below_review_below_calibrated() -> None:
    assert WAM_SCORE_CLAIM_GRADES == (
        "fixture_evaluator_only",
        "review_grade",
        "calibrated_evaluator_grade",
    )


def test_above_review_claim_without_evidence_fails_closed_to_fixture() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert gate["status"] == "failed_closed"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_above_review_claim_with_only_consistency_fails_closed() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=_passing_consistency(),
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_above_review_claim_with_only_anchors_fails_closed(tmp_path: Path) -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=None,
        calibration_anchors=_passing_anchor_check(tmp_path),
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_review_grade_claim_without_evidence_is_capped_not_demoted() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="review_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "review_grade"
    assert gate["max_allowed_grade"] == "review_grade"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER not in gate["blockers"]
    assert gate["upgrade_requirements"]


def test_calibrated_grade_allowed_with_passing_consistency_and_anchors(
    tmp_path: Path,
) -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=_passing_consistency(),
        calibration_anchors=_passing_anchor_check(tmp_path),
    )
    assert gate["granted_grade"] == "calibrated_evaluator_grade"
    assert gate["max_allowed_grade"] == "calibrated_evaluator_grade"
    assert gate["status"] == "granted"
    assert gate["blockers"] == []
    assert gate["consistency"]["consistency_score"] == pytest.approx(1.0)
    assert gate["calibration_anchors"]["anchor_set"]


def test_failing_consistency_score_blocks_calibrated_grade(tmp_path: Path) -> None:
    reference = _trajectory([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    rollout = _trajectory([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0], [6.0, 6.0, 0.0]])
    gate = apply_wam_score_claim_gate(
        requested_grade="calibrated_evaluator_grade",
        consistency=score_wam_consistency(rollout, reference),
        calibration_anchors=_passing_anchor_check(tmp_path),
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER in gate["blockers"]


def test_fixture_evaluator_only_run_never_exceeds_fixture_grade(
    tmp_path: Path,
) -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="review_grade",
        consistency=_passing_consistency(),
        calibration_anchors=_passing_anchor_check(tmp_path),
        fixture_evaluator_only=True,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert gate["max_allowed_grade"] == "fixture_evaluator_only"


def test_unrecognized_requested_grade_fails_closed() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="deployment_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert gate["granted_grade"] == "fixture_evaluator_only"
    assert "wam_score_claim_grade_unrecognized" in gate["blockers"]


def test_gate_payload_always_carries_anchor_set_and_consistency_number() -> None:
    gate = apply_wam_score_claim_gate(
        requested_grade="review_grade",
        consistency=None,
        calibration_anchors=None,
    )
    assert "consistency_score" in gate["consistency"]
    assert "anchor_set" in gate["calibration_anchors"]
    boundary = gate["claim_boundary"]
    assert boundary["score_above_review_grade_requires_consistency_and_calibration_anchors"] is True
    assert boundary["bare_wam_score_reporting_forbidden"] is True
    assert boundary["rank_fidelity_result_proven"] is False
    assert boundary["public_claim_upgrade_allowed"] is False
