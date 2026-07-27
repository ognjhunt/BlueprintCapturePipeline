from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.policy_ranking_experiment_2 import build_protocol
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def test_experiment_2_protocol_preserves_freeze_and_exact_verdict_values() -> None:
    previous = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experiments/policy_ranking_thesis_20260726/preregistered_protocol.json"
        ).read_text()
    )
    result = build_protocol(previous, source_commit="a" * 40)
    digest = result.pop("protocol_sha256")
    assert canonical_sha256(result) == digest
    assert result["historical_experiment"]["immutable"] is True
    assert result["historical_experiment"]["verdict"] == "inconclusive"
    assert result["arm_freeze"]["new_heldout_ranking_arms_forbidden"] is True
    assert result["label_access"]["heldout_opened"] is False
    assert result["spend"]["physical_robotics_usd_max"] == 0
    assert result["overall_verdict_values"] == [
        "thesis_supported",
        "thesis_not_supported",
        "inconclusive",
    ]
    assert result["claim_boundaries"]["blueprint_physical_robot_operation"] is False
