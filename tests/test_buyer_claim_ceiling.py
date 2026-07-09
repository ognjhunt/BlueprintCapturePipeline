from __future__ import annotations

from blueprint_pipeline.buyer_claim_ceiling import build_buyer_claim_ceiling


def test_buyer_claim_ceiling_blocks_live_execution_copy_without_live_gates() -> None:
    ceiling = build_buyer_claim_ceiling(
        success_claim_ledger={"highest_truthful_claim": "review_task_success"},
        proof_boundary={},
        buyer_copy_inputs={
            "marketing_copy": (
                "Live simulator execution and live policy execution are verified."
            )
        },
    )

    assert ceiling["status"] == "blocked"
    assert ceiling["highest_truthful_claim"] == "review_task_success"
    assert ceiling[
        "buyer_facing_claim_ceiling_pinned_to_highest_truthful_claim"
    ] is True
    assert "buyer_copy_claims_live_simulator_execution_without_live_gate" in ceiling[
        "blockers"
    ]
    assert "buyer_copy_claims_live_policy_execution_without_live_gate" in ceiling[
        "blockers"
    ]
    assert ceiling["live_simulator_execution_claim_allowed"] is False
    assert ceiling["live_policy_execution_claim_allowed"] is False


def test_buyer_claim_ceiling_allows_live_copy_when_live_gates_are_true() -> None:
    ceiling = build_buyer_claim_ceiling(
        success_claim_ledger={"highest_truthful_claim": "policy_task_success"},
        proof_boundary={
            "live_simulator_execution_proven": True,
            "live_policy_execution_proven": True,
        },
        buyer_copy_inputs={
            "report_copy": "Policy executed in simulator with live policy execution."
        },
    )

    assert ceiling["status"] == "passed"
    assert ceiling["blockers"] == []
    assert ceiling["simulator_task_success_claim_allowed"] is True
    assert ceiling["policy_task_success_claim_allowed"] is True
    assert ceiling["live_simulator_execution_claim_allowed"] is True
    assert ceiling["live_policy_execution_claim_allowed"] is True


def test_buyer_claim_ceiling_does_not_block_generic_sim_only_eval_copy() -> None:
    ceiling = build_buyer_claim_ceiling(
        success_claim_ledger={"highest_truthful_claim": "review_task_success"},
        proof_boundary={},
        buyer_copy_inputs={
            "offer_copy": (
                "Sim-only policy comparison with review-grade evaluation artifacts."
            )
        },
    )

    assert ceiling["status"] == "passed"
    assert ceiling["blockers"] == []
