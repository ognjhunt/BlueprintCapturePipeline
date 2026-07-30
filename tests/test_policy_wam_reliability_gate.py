from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.policy_wam_reliability_gate import (
    FrozenMaximumHorizonTerminalCriterion,
    MultiViewFrameSequenceReliabilityGate,
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


def test_multi_view_gate_scores_current_to_generated_sequence_per_view(tmp_path: Path) -> None:
    view_order = ("exterior_2", "exterior_1", "wrist")
    current_views: dict[str, dict[str, str]] = {}
    sequences: dict[str, list[str]] = {}
    for view_index, view_id in enumerate(view_order):
        current = tmp_path / view_id / "current.png"
        current.parent.mkdir(parents=True)
        current.write_bytes(b"current")
        current_views[view_id] = {"path": str(current)}
        sequences[view_id] = []
        for frame_index in range(5):
            frame = tmp_path / view_id / f"generated_{frame_index}.png"
            frame.write_bytes(b"generated")
            sequences[view_id].append(str(frame))

    calls: list[dict[str, Any]] = []

    def assessor(paths: list[str], actions: np.ndarray, *_args: Any, **kwargs: Any):
        calls.append({"paths": paths, "shape": actions.shape, **kwargs})
        return _report(Path(paths[0]), reliable=True)

    gate = MultiViewFrameSequenceReliabilityGate(
        thresholds=ReliabilityThresholds(),
        view_order=view_order,
        assessor=assessor,
    )
    result = gate.assess(
        previous_observation={},
        prepared_transition={
            "wam_request": {"current_views": current_views},
            "reliability_actions_10d": np.zeros((5, 10)),
        },
        wam_prediction={"generated_view_frame_sequences": sequences},
        query_index=0,
        output_dir=tmp_path,
    )

    assert result["abstain"] is False
    assert result["all_registered_views_scored_separately"] is True
    assert len(calls) == 3
    assert all(len(call["paths"]) == 6 for call in calls)
    assert all(call["paths"][0].endswith("current.png") for call in calls)
    assert all(call["timing_flag_scope"] == TIMING_SCOPE_SESSION for call in calls)


def test_multi_view_gate_abstains_if_any_registered_view_fails(tmp_path: Path) -> None:
    view_order = ("exterior", "wrist")
    current_views = {}
    sequences = {}
    for view_id in view_order:
        current = tmp_path / f"{view_id}_current.png"
        current.write_bytes(b"current")
        current_views[view_id] = {"path": str(current)}
        sequences[view_id] = []
        for index in range(5):
            frame = tmp_path / f"{view_id}_{index}.png"
            frame.write_bytes(b"frame")
            sequences[view_id].append(str(frame))

    def assessor(paths: list[str], *_args: Any, **_kwargs: Any):
        return _report(Path(paths[0]), reliable="wrist" not in paths[0])

    result = MultiViewFrameSequenceReliabilityGate(
        thresholds=ReliabilityThresholds(), view_order=view_order, assessor=assessor
    ).assess(
        previous_observation={},
        prepared_transition={
            "wam_request": {"current_views": current_views},
            "reliability_actions_10d": np.zeros((5, 10)),
        },
        wam_prediction={"generated_view_frame_sequences": sequences},
        query_index=0,
        output_dir=tmp_path,
    )

    assert result["abstain"] is True
    assert result["reasons"] == ["wrist:static_under_command"]
