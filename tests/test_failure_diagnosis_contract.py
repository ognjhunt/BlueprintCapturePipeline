from __future__ import annotations

from blueprint_pipeline.failure_diagnosis_contract import build_failure_diagnosis_audit


def test_zero_failure_simulator_batch_labels_are_explicit_review_attestation() -> None:
    audit = build_failure_diagnosis_audit(
        labels_payload={
            "status": "no_failures_labeled",
            "failed_attempt_count": 0,
            "labels": [],
        },
        trace_payload={
            "status": "completed",
            "attempts": [],
        },
    )

    assert audit["failed_attempt_count"] == 0
    assert audit["zero_failures_reviewed"] is True
    assert audit["failure_diagnosis_complete"] is True
    assert audit["blockers"] == []
