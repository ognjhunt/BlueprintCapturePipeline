"""R026: buyer-facing claim ceiling + blocked-claim launch guard."""

from __future__ import annotations

from blueprint_pipeline import live_robot_eval_closure as closure


def test_default_claim_ceiling_is_sim_review_grade() -> None:
    # The committed CLAIM_BOUNDARY fails closed, so the honest ceiling is a
    # sim/review-grade eval — never executed simulation or executed policy.
    assert (
        closure.highest_truthful_claim()
        == closure.HIGHEST_TRUTHFUL_CLAIM_WHEN_BLOCKED
    )


def test_claim_ceiling_upgrades_only_when_all_proofs_and_upgrade_true() -> None:
    proven = {
        **closure.CLAIM_BOUNDARY,
        "simulator_execution_proven": True,
        "robot_policy_execution_proven": True,
        "public_claim_upgrade_allowed": True,
    }
    assert (
        closure.highest_truthful_claim(proven)
        == closure.HIGHEST_TRUTHFUL_CLAIM_WHEN_PROVEN
    )
    # Missing the public-upgrade flag keeps the ceiling blocked even with proofs.
    still_blocked = {**proven, "public_claim_upgrade_allowed": False}
    assert (
        closure.highest_truthful_claim(still_blocked)
        == closure.HIGHEST_TRUTHFUL_CLAIM_WHEN_BLOCKED
    )


def test_blocked_copy_flags_live_execution_claims_by_default() -> None:
    copy = (
        "Blueprint's beta features live policy execution and executed in the "
        "simulator across every warehouse task."
    )
    violations = closure.blocked_public_claim_violations(copy)
    proof_keys = {v["proof_key"] for v in violations}
    assert "robot_policy_execution_proven" in proof_keys
    assert "simulator_execution_proven" in proof_keys


def test_honest_sim_review_grade_copy_has_no_violations() -> None:
    copy = (
        "Each package is a sim and review-grade Task Evaluation Run with "
        "provenance-tracked capture; no live robot or executed policy is claimed."
    )
    assert closure.blocked_public_claim_violations(copy) == []


def test_claims_allowed_once_proofs_and_upgrade_are_established() -> None:
    proven = {
        **closure.CLAIM_BOUNDARY,
        "simulator_execution_proven": True,
        "robot_policy_execution_proven": True,
        "public_claim_upgrade_allowed": True,
    }
    copy = "We ran the live simulation and executed the policy end to end."
    assert closure.blocked_public_claim_violations(copy, proven) == []


def test_assert_no_blocked_public_claims_aggregates_with_document_labels() -> None:
    documents = [
        ("landing_page", "A sim/review-grade evaluation package."),
        ("press_release", "Our robot executed the task in every trial."),
    ]
    violations = closure.assert_no_blocked_public_claims(documents)
    assert len(violations) == 1
    assert violations[0]["document"] == "press_release"
    assert violations[0]["proof_key"] == "robot_policy_execution_proven"


def test_empty_copy_is_not_a_violation() -> None:
    assert closure.blocked_public_claim_violations("") == []
    assert closure.blocked_public_claim_violations(None) == []
