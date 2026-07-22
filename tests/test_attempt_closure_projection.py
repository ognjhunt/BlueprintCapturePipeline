from __future__ import annotations

from blueprint_pipeline.attempt_closure_projection import project_attempt_closure


def test_project_attempt_closure_requires_aggregate_completion() -> None:
    closure = {
        "schema_version": "campaign_attempt_closure.v1",
        "status": "blocked",
        "identity": {"attempt_id": "attempt-1"},
        "proof_rows": [
            {"row_id": "persistent_simulator_transition", "status": "passed"},
        ],
    }

    projection = project_attempt_closure(
        closure,
        expected_schema_version="campaign_attempt_closure.v1",
        incomplete_blocker="campaign_attempt_closure_not_completed",
    )

    assert projection["status"] == "blocked"
    assert projection["task_success_proven"] is False
    assert projection["blockers"] == ["campaign_attempt_closure_not_completed"]


def test_project_attempt_closure_preserves_verified_digest_claim_boundary() -> None:
    closure = {
        "schema_version": "campaign_attempt_closure.v1",
        "status": "completed",
        "identity": {"attempt_id": "attempt-2"},
        "proof_rows": [
            {
                "row_id": "persistent_simulator_transition",
                "status": "passed",
                "evidence": {
                    "verified_leaf_artifacts": [{"sha256": "a" * 64}],
                },
            },
            {"row_id": "teardown", "status": "passed"},
            {"row_id": "final_inventory", "status": "passed"},
        ],
    }

    projection = project_attempt_closure(
        closure,
        expected_schema_version="campaign_attempt_closure.v1",
        incomplete_blocker="campaign_attempt_closure_not_completed",
    )

    assert projection["status"] == "ready"
    assert projection["task_success_proven"] is True
    assert projection["teardown_and_zero_inventory_proven"] is True
    assert projection["verified_leaf_artifact_sha256s"] == {
        "persistent_simulator_transition": ["a" * 64]
    }
