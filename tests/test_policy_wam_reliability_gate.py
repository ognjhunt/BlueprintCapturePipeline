from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.policy_wam_reliability_gate import (
    FrozenMaximumHorizonTerminalCriterion,
    Tier1VideoReliabilityGate,
)
from blueprint_pipeline.wam_rollout_reliability import (
    TIMING_SCOPE_SESSION,
    ReliabilityThresholds,
    RolloutReliabilityReport,
)


def _report(video: Path, *, reliable: bool) -> RolloutReliabilityReport:
    return RolloutReliabilityReport(
        video_path=str(video),
        n_frames=17,
        n_action_steps=16,
        flags=() if reliable else ("static_under_command",),
        reliable=reliable,
        command_energy_mean=0.4,
        command_energy_std=0.2,
        motion_mean=0.01,
        motion_max=0.02,
        timing_correlation=0.0,
        timing_flag_scope=TIMING_SCOPE_SESSION,
        spatial_std_mean=20.0,
    )


def test_gate_abstains_on_unreliable_video_and_freezes_thresholds(tmp_path: Path) -> None:
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"fixture")
    calls: list[dict[str, Any]] = []

    def assessor(path: Path, actions: np.ndarray, thresholds: ReliabilityThresholds, **kw: Any):
        calls.append({"path": path, "shape": actions.shape, "thresholds": thresholds, **kw})
        return _report(path, reliable=False)

    thresholds = ReliabilityThresholds(static_motion_max=0.04)
    gate = Tier1VideoReliabilityGate(thresholds=thresholds, assessor=assessor)
    result = gate.assess(
        previous_observation={},
        prepared_transition={"reliability_actions_10d": np.zeros((16, 10))},
        wam_prediction={"generated_video_path": str(video)},
        query_index=0,
        output_dir=tmp_path,
    )

    assert result["abstain"] is True
    assert result["reasons"] == ["static_under_command"]
    assert result["thresholds"]["static_motion_max"] == 0.04
    assert len(result["thresholds_sha256"]) == 64
    assert calls[0]["timing_flag_scope"] == TIMING_SCOPE_SESSION


def test_gate_fails_closed_without_video_or_valid_actions(tmp_path: Path) -> None:
    gate = Tier1VideoReliabilityGate(thresholds=ReliabilityThresholds())
    with pytest.raises(ValueError, match="generated_video_path_missing"):
        gate.assess(
            previous_observation={},
            prepared_transition={"reliability_actions_10d": np.zeros((16, 10))},
            wam_prediction={},
            query_index=0,
            output_dir=tmp_path,
        )
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="reliability_actions_10d_invalid"):
        gate.assess(
            previous_observation={},
            prepared_transition={"reliability_actions_10d": np.zeros((16, 8))},
            wam_prediction={"generated_video_path": video},
            query_index=0,
            output_dir=tmp_path,
        )


def test_horizon_terminal_criterion_never_self_grades() -> None:
    result = FrozenMaximumHorizonTerminalCriterion().assess(observation={}, query_index=2)
    assert result["terminal"] is False
    assert result["candidate_policy_graded_itself"] is False
    assert result["wam_graded_itself"] is False
