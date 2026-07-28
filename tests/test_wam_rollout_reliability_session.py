from __future__ import annotations

import pytest
import numpy as np

import blueprint_pipeline.wam_rollout_reliability as reliability
from blueprint_pipeline.wam_rollout_reliability import (
    ACTION_DIM,
    FLAG_STATIC_UNDER_COMMAND,
    FLAG_TIMING_DISAGREEMENT,
    FLAG_TIMING_EVIDENCE_INSUFFICIENT,
    ROT6D_IDENTITY,
    RolloutReliabilityReport,
    SessionReliabilityThresholds,
    assess_session_reliability,
)


FROZEN_THRESHOLDS = SessionReliabilityThresholds(
    timing_correlation_min=0.15,
    minimum_eligible_timing_windows=3,
)


def _window_report(
    correlation: float | None,
    *,
    flags: tuple[str, ...] = (),
    timing_scope: str = "session",
) -> RolloutReliabilityReport:
    return RolloutReliabilityReport(
        video_path="fixture.mp4",
        n_frames=17,
        n_action_steps=16,
        flags=flags,
        reliable=not flags,
        command_energy_mean=0.5,
        command_energy_std=0.2,
        motion_mean=0.2,
        motion_max=0.5,
        timing_correlation=correlation,
        timing_flag_scope=timing_scope,
        spatial_std_mean=20.0,
    )


def test_session_timing_aggregation_does_not_kill_one_noisy_window() -> None:
    report = assess_session_reliability(
        "session-1",
        [_window_report(-0.2), _window_report(0.4), _window_report(0.5)],
        FROZEN_THRESHOLDS,
    )
    assert report.reliable is True
    assert report.timing_correlation_median == 0.4
    assert report.flags == ()


def test_session_timing_aggregation_abstains_on_material_disagreement() -> None:
    report = assess_session_reliability(
        "session-2",
        [_window_report(-0.3), _window_report(-0.2), _window_report(0.4)],
        FROZEN_THRESHOLDS,
    )
    assert report.reliable is False
    assert report.flags == (FLAG_TIMING_DISAGREEMENT,)


def test_session_timing_aggregation_abstains_when_underpowered() -> None:
    report = assess_session_reliability(
        "session-3",
        [_window_report(0.8), _window_report(0.9)],
        FROZEN_THRESHOLDS,
    )
    assert report.reliable is False
    assert report.flags == (FLAG_TIMING_EVIDENCE_INSUFFICIENT,)


def test_session_timing_aggregation_retains_hard_window_failures() -> None:
    report = assess_session_reliability(
        "session-4",
        [
            _window_report(None, flags=(FLAG_STATIC_UNDER_COMMAND,)),
            _window_report(0.8),
            _window_report(0.9),
            _window_report(0.7),
        ],
        FROZEN_THRESHOLDS,
    )
    assert report.reliable is False
    assert report.hard_failure_window_count == 1
    assert FLAG_STATIC_UNDER_COMMAND in report.flags


def test_session_aggregation_rejects_legacy_window_scoped_reports() -> None:
    with pytest.raises(ValueError, match="session_aggregation_requires_session_scoped_windows"):
        assess_session_reliability(
            "session-5",
            [_window_report(0.9, timing_scope="window")],
            FROZEN_THRESHOLDS,
        )


def test_window_assessment_records_but_does_not_kill_timing_by_default(monkeypatch) -> None:
    actions = np.zeros((16, ACTION_DIM))
    actions[:, 3:9] = ROT6D_IDENTITY
    actions[:8, 0] = 0.02
    motion = np.asarray([0.1] * 8 + [1.0] * 8)
    monkeypatch.setattr(
        reliability,
        "video_motion_series",
        lambda *_args, **_kwargs: (motion, 20.0, 17),
    )

    session_scoped = reliability.assess_rollout_reliability("fixture.mp4", actions)
    assert session_scoped.timing_correlation is not None
    assert session_scoped.timing_correlation < 0
    assert session_scoped.timing_flag_scope == "session"
    assert FLAG_TIMING_DISAGREEMENT not in session_scoped.flags

    legacy_window_scoped = reliability.assess_rollout_reliability(
        "fixture.mp4",
        actions,
        timing_flag_scope="window",
    )
    assert FLAG_TIMING_DISAGREEMENT in legacy_window_scoped.flags
