from __future__ import annotations

from blueprint_pipeline.policy_ranking_power_analysis import build_power_analysis


def test_power_analysis_is_cluster_conservative_and_reproducible() -> None:
    result = build_power_analysis()
    assert result["analysis_unit"] == "heldout_session_cluster"
    assert result["within_session_pairs_treated_as_independent"] is False
    assert result["exact_binomial_reference"]["critical_successes"] == 31
    assert 0.678 <= result["exact_binomial_reference"][
        "minimum_accuracy_for_target_power"
    ] <= 0.680
    assert result["interpretation"][
        "wide_registered_confidence_intervals_produce_inconclusive_not_success"
    ] is True
    assert len(result["analysis_sha256"]) == 64
