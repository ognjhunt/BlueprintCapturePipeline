from __future__ import annotations

import pytest

from blueprint_pipeline.task_episode_baseline import (
    TASK_EPISODE_BASELINE_SCHEMA_VERSION,
    build_task_episode_baseline,
    canonical_task_contract_sha256,
    evaluate_task_criterion,
    verify_task_episode_baseline,
)


CONTRACT = {
    "registered_criteria": [
        {
            "criterion_id": "microwave_door_open_angle",
            "comparison": "increase_at_least",
            "tolerance": 0.35,
            "unit": "rad",
        }
    ]
}


def _baseline(**overrides):
    fields = {
        "episode_initial_value": 0.05,
        "attempt_id": "run-1-attempt-000001",
        "launch_nonce": "nonce-1",
        "simulator_session_id": "isaac-task-session-abc",
        "stage_id": "f" * 64,
        "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
        "task_contract_sha256": canonical_task_contract_sha256(CONTRACT),
        "criterion_id": "microwave_door_open_angle",
        "unit": "rad",
        "captured_timestamp": "1751971200000000000",
    }
    fields.update(overrides)
    return build_task_episode_baseline(**fields)


def _verify(baseline, **overrides):
    expected = {
        "simulator_session_id": "isaac-task-session-abc",
        "stage_id": "f" * 64,
        "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
        "task_contract_sha256": canonical_task_contract_sha256(CONTRACT),
    }
    expected.update(overrides)
    return verify_task_episode_baseline(baseline, **expected)


def test_baseline_build_and_verify_roundtrip():
    baseline = _baseline()
    assert baseline["schema_version"] == TASK_EPISODE_BASELINE_SCHEMA_VERSION
    assert len(baseline["baseline_digest"]) == 64
    assert _verify(baseline) == []


def test_baseline_digest_binds_exact_task_contract_artifact() -> None:
    baseline = _baseline(task_contract_artifact_sha256="a" * 64)

    assert baseline["task_contract_artifact_sha256"] == "a" * 64
    tampered = {**baseline, "task_contract_artifact_sha256": "b" * 64}
    assert "task_episode_baseline_digest_mismatch" in _verify(tampered)


def test_baseline_build_rejects_missing_binding_field():
    with pytest.raises(ValueError, match="task_episode_baseline_field_missing:attempt_id"):
        _baseline(attempt_id="")
    with pytest.raises(
        ValueError, match="task_episode_baseline_field_missing:episode_initial_value"
    ):
        _baseline(episode_initial_value=float("nan"))


def test_missing_baseline_blocks():
    assert verify_task_episode_baseline(
        None,
        simulator_session_id="s",
        stage_id="g",
        articulation_prim_path="/p",
        task_contract_sha256="a" * 64,
    ) == ["task_episode_baseline_missing"]


def test_tampered_baseline_value_blocks():
    baseline = dict(_baseline())
    baseline["episode_initial_value"] = -5.0
    assert "task_episode_baseline_digest_mismatch" in _verify(baseline)


def test_tampered_baseline_digest_blocks():
    baseline = dict(_baseline())
    baseline["baseline_digest"] = "0" * 64
    assert "task_episode_baseline_digest_mismatch" in _verify(baseline)


def test_session_restart_blocks():
    baseline = _baseline()
    blockers = _verify(baseline, simulator_session_id="isaac-task-session-restarted")
    assert "task_episode_baseline_session_mismatch" in blockers


def test_stage_restart_blocks():
    baseline = _baseline()
    assert "task_episode_baseline_stage_mismatch" in _verify(baseline, stage_id="0" * 64)


def test_changed_target_prim_blocks():
    baseline = _baseline()
    blockers = _verify(baseline, articulation_prim_path="/root/Refrigerator001/Door")
    assert "task_episode_baseline_prim_mismatch" in blockers


def test_changed_task_contract_blocks():
    baseline = _baseline()
    other = canonical_task_contract_sha256({"registered_criteria": []})
    blockers = _verify(baseline, task_contract_sha256=other)
    assert "task_episode_baseline_task_contract_mismatch" in blockers


def test_attempt_and_nonce_binding_blocks_when_checked():
    baseline = _baseline()
    blockers = verify_task_episode_baseline(
        baseline,
        simulator_session_id="isaac-task-session-abc",
        stage_id="f" * 64,
        articulation_prim_path="/root/Microwave017/Microwave017_Door",
        task_contract_sha256=canonical_task_contract_sha256(CONTRACT),
        attempt_id="run-1-attempt-000002",
        launch_nonce="nonce-2",
    )
    assert "task_episode_baseline_attempt_mismatch" in blockers
    assert "task_episode_baseline_nonce_mismatch" in blockers


def test_schema_mismatch_blocks():
    baseline = dict(_baseline())
    baseline["schema_version"] = "task_episode_baseline.v0"
    assert "task_episode_baseline_schema_invalid" in _verify(baseline)


def test_two_small_steps_satisfy_episode_criterion_only_after_step_two():
    criterion = {"comparison": "increase_at_least", "tolerance": 0.35}
    step_one = evaluate_task_criterion(
        criterion, episode_initial_value=0.0, step_before=0.0, step_after=0.20
    )
    step_two = evaluate_task_criterion(
        criterion, episode_initial_value=0.0, step_before=0.20, step_after=0.40
    )
    assert step_one["passed"] is False
    assert step_two["passed"] is True
    assert step_one["step_delta"] == pytest.approx(0.20)
    assert step_two["step_delta"] == pytest.approx(0.20)
    assert step_one["episode_delta"] == pytest.approx(0.20)
    assert step_two["episode_delta"] == pytest.approx(0.40)
    assert step_one["evaluation_basis"] == "episode_relative"


def test_oscillation_uses_current_vs_initial_truth_not_accumulated_motion():
    criterion = {"comparison": "increase_at_least", "tolerance": 0.35}
    opened = evaluate_task_criterion(
        criterion, episode_initial_value=0.0, step_before=0.0, step_after=0.40
    )
    closed_back = evaluate_task_criterion(
        criterion, episode_initial_value=0.0, step_before=0.40, step_after=0.02
    )
    assert opened["passed"] is True
    assert closed_back["passed"] is False
    assert closed_back["episode_delta"] == pytest.approx(0.02)


def test_regression_below_initial_fails_absolute_change_from_episode_initial():
    criterion = {"comparison": "absolute_change_at_least", "tolerance": 0.5}
    result = evaluate_task_criterion(
        criterion, episode_initial_value=1.0, step_before=1.3, step_after=0.9
    )
    assert result["passed"] is False
    assert result["episode_delta"] == pytest.approx(-0.1)


def test_decrease_at_least_uses_episode_initial():
    criterion = {"comparison": "decrease_at_least", "tolerance": 0.3}
    result = evaluate_task_criterion(
        criterion, episode_initial_value=1.0, step_before=0.85, step_after=0.65
    )
    assert result["passed"] is True
    assert result["evaluation_basis"] == "episode_relative"


@pytest.mark.parametrize(
    ("criterion", "step_after", "expected"),
    [
        ({"comparison": "within_tolerance", "target_value": 1.0, "tolerance": 0.1}, 1.09, True),
        ({"comparison": "within_tolerance", "target_value": 1.0, "tolerance": 0.1}, 1.2, False),
        ({"comparison": "at_or_above", "target_value": 1.0, "tolerance": 0.05}, 0.96, True),
        ({"comparison": "at_or_below", "target_value": 1.0, "tolerance": 0.05}, 1.04, True),
    ],
)
def test_absolute_target_criteria_stay_separate_from_episode_baseline(
    criterion, step_after, expected
):
    result = evaluate_task_criterion(
        criterion, episode_initial_value=-9.0, step_before=-9.0, step_after=step_after
    )
    assert result["passed"] is expected
    assert result["evaluation_basis"] == "absolute_target"


def test_unsupported_comparison_raises():
    with pytest.raises(ValueError, match="persistent_isaac_completion_comparison_unsupported"):
        evaluate_task_criterion(
            {"comparison": "wiggles_a_bit"},
            episode_initial_value=0.0,
            step_before=0.0,
            step_after=1.0,
        )


def test_nonfinite_measurement_raises():
    with pytest.raises(ValueError, match="task_episode_measurement_value_nonfinite"):
        evaluate_task_criterion(
            {"comparison": "increase_at_least", "tolerance": 0.1},
            episode_initial_value=0.0,
            step_before=0.0,
            step_after=float("nan"),
        )
