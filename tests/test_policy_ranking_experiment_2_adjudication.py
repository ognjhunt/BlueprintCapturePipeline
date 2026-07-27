from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.policy_ranking_evidence import EvidenceStore, utc_now
from blueprint_pipeline.policy_ranking_experiment_2_adjudication import (
    _kendall_permutation,
    _risk_coverage,
    build_prediction_freeze,
)


def test_exact_kendall_permutation_detects_perfect_order() -> None:
    result = _kendall_permutation(
        {"a": 0.1, "b": 0.2, "c": 0.3},
        {"a": 0.0, "b": 1.0, "c": 2.0},
    )
    assert result["observed_tau_b"] == 1.0
    assert result["exact_permutation_count"] == 6
    assert result["one_sided_p_value"] == 1 / 6


def test_risk_curve_is_complete_and_uncertainty_tracks_error() -> None:
    report = _risk_coverage(
        [
            {
                "session_id": "s",
                "left_policy": "a",
                "correctness": 1.0,
                "error": 0.0,
                "confidence": 0.9,
            },
            {
                "session_id": "s",
                "left_policy": "b",
                "correctness": 0.0,
                "error": 1.0,
                "confidence": 0.1,
            },
        ]
    )
    assert len(report["curve"]) == 2
    assert report["curve"][-1]["coverage"] == 1.0
    assert report["uncertainty_error_association_descriptive"] is True


def test_prediction_freeze_requires_exact_complete_identity_set(tmp_path: Path) -> None:
    request = {
        "request_id": "r1",
        "deterministic_input_hash": "h1",
        "session_id": "s1",
        "policy_id": "p1",
        "task_id": "t1",
    }
    root = tmp_path / "evidence"
    store = EvidenceStore(
        root,
        experiment_id="e1",
        inventory_sha256="inventory",
        configuration_sha256="configuration",
    )
    claim = store.claim(
        request,
        arm_id="temporal",
        attempt_type="scientific_request",
        provider="test",
        model_snapshot="m1",
    )
    assert claim
    started = utc_now()
    store.mark_provider_call_started(
        request=request,
        claim_id=claim,
        arm_id="temporal",
        attempt_type="scientific_request",
        provider="test",
        model_snapshot="m1",
        started_at=started,
    )
    store.complete(
        request=request,
        claim_id=claim,
        arm_id="temporal",
        attempt_type="scientific_request",
        provider="test",
        model_snapshot="m1",
        started_at=started,
        elapsed_seconds=1.0,
        structured_response={"request_id": "r1"},
        validation_result="valid",
        usage={"input_tokens": 1, "output_tokens": 1},
        estimated_cost_usd=0.01,
        actual_cost_usd=None,
        response_id="response-1",
        consumed_scientific_response=True,
    )
    result = build_prediction_freeze(
        {"inventory_sha256": "inventory", "requests": [request]},
        evidence_root=root,
    )
    assert result["status"] == "frozen"
    assert result["accepted_request_count"] == 1
    assert result["estimated_cost_usd_recomputed"] == 0.01
