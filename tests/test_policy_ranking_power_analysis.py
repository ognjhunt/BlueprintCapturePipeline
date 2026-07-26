from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.policy_ranking_power_analysis import (
    build_label_basis_power_sensitivity,
    build_power_analysis,
)


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


def test_label_basis_sensitivity_keeps_sessions_as_clusters(tmp_path: Path) -> None:
    session_ids = [f"session-{index}" for index in range(2)]
    policies = [
        "paligemma_binning_droid",
        "paligemma_diffusion_droid",
        "paligemma_fast_droid",
    ]
    root = tmp_path / "roboarena"
    for index, session_id in enumerate(session_ids):
        session = root / "evaluation_sessions" / session_id
        session.mkdir(parents=True)
        metadata = {
            "policies": {
                "A": {
                    "policy_name": policies[0],
                    "binary_success": index == 0,
                    "partial_success": 1.0,
                },
                "B": {
                    "policy_name": policies[1],
                    "binary_success": False,
                    "partial_success": 0.5,
                },
                "C": {
                    "policy_name": policies[2],
                    "binary_success": False,
                    "partial_success": 0.0,
                },
            },
            "preference": "A",
        }
        (session / "metadata.yaml").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    protocol = {
        "protocol_sha256": "a" * 64,
        "policies": policies,
        "partitions": {"pilot": session_ids},
    }

    result = build_label_basis_power_sensitivity(
        protocol=protocol,
        roboarena_root=root,
        heldout_sessions=10,
    )

    assert result["within_session_pairs_treated_as_independent"] is False
    assert result["basis_results"]["binary_success"][
        "sessions_with_any_informative_pair"
    ] == 1
    assert result["basis_results"]["binary_success"][
        "projected_heldout_informative_session_count"
    ] == 5
    assert result["basis_results"]["binary_then_partial"][
        "sessions_with_any_informative_pair"
    ] == 2
    assert result["basis_results"]["preference_winner_vs_rest"][
        "informative_pair_count"
    ] == 4
