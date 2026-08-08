from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_episode_batch import (
    MAX_EPISODES_PER_CANDIDATE,
    EpisodeBatchError,
    run_episode_batch,
    summarize_candidate_batches,
)
from tests.test_adp009d_policy_episode import (  # reuse the proven fixtures
    _DESTINATION,
    _MEASURED,
    _Environment,
    _Policy,
)


def _batch(environment=None, policy=None, **overrides):
    kwargs = dict(
        environment=environment or _Environment(),
        policy=policy or _Policy(),
        candidate_id="pi05_droid",
        destination_position_world_m=_DESTINATION,
        prompt="pick up the can and place it on the counter",
        gripper=_MEASURED,
        episodes=3,
        max_policy_queries=4,
        settle_window_samples=6,
    )
    kwargs.update(overrides)
    return run_episode_batch(**kwargs)


def test_a_few_episodes_run_independently_and_are_reported_each() -> None:
    batch = _batch()

    assert batch["episodes_requested"] == 3
    assert batch["episodes_scored"] == 3
    assert batch["episodes_failed"] == 0
    assert len(batch["episodes"]) == 3
    assert batch["outcome_counts"] == {"placed": 3}
    # Every episode resets, so identical outcomes mean repeatability.
    assert batch["outcomes_identical_across_episodes"] is True
    assert batch["episodes_policy_outcome_interpretable"] == 3
    row = batch["episodes"][0]
    assert row["joint_position_reset_rad"] == [0.0] * 7
    assert row["joint_position_end_rad"][0] == pytest.approx(1.6)
    assert row["max_abs_joint_delta_from_reset_rad"][0] == pytest.approx(1.6)
    assert row["arm_moved"] is True
    assert row["actions_reached_robot"] is True
    assert row["policy_outcome_interpretable"] is True
    assert row["policy_outcome"] == "placed"
    assert row["harness_finding"] is None
    assert len(row["queries"]) == 4
    assert row["any_joint_limit_clamped_count"] == 0
    assert row["joint_limit_clamped_action_count"] == 0
    assert row["commanded_action_magnitudes"]["policy_action_rows_submitted"] == 32
    assert row["performance_diagnostics"]["timings_seconds"]["total"] >= 0.0


def test_temporal_policy_state_is_reset_before_every_episode() -> None:
    class _TemporalPolicy(_Policy):
        def __init__(self):
            super().__init__()
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    policy = _TemporalPolicy()
    batch = _batch(policy=policy)

    assert batch["episodes_scored"] == 3
    assert policy.reset_count == 3


def test_the_batch_refuses_to_support_a_ranking() -> None:
    """The easiest way to discredit a proof is to let it look like a decision."""

    batch = _batch()

    assert batch["supports_policy_ranking"] is False
    assert batch["sample_purpose"] == "pipeline_proof_not_policy_comparison"
    assert "paired sample size" in batch["ranking_requires"]

    summary = summarize_candidate_batches([batch])
    # It ranks now, but the sample-size caveat travels with the ordering.
    assert summary["supports_policy_ranking"] is False
    assert "paired sample size" in summary["why_not_adjudicated"]
    assert summary["supports_policy_ranking"] is False


def test_one_failed_episode_does_not_erase_the_others() -> None:
    class _Flaky(_Policy):
        def infer(self, observation):
            if len(self.observations) == 1:
                self.observations.append(observation)
                return np.zeros((10, 7))  # malformed on the second query
            return super().infer(observation)

    batch = _batch(policy=_Flaky())

    assert batch["episodes_failed"] >= 1
    assert any(row["status"] == "failed" for row in batch["episodes"])
    assert all("error" in row for row in batch["episodes"] if row["status"] == "failed")


def test_benchmark_scale_is_refused_as_a_scope_error() -> None:
    """Silently obliging would produce numbers that invite a decision."""

    with pytest.raises(EpisodeBatchError) as excinfo:
        _batch(episodes=MAX_EPISODES_PER_CANDIDATE + 1)
    assert any("proof_scale" in e for e in excinfo.value.errors)

    with pytest.raises(EpisodeBatchError):
        _batch(episodes=0)


def test_two_candidates_are_shown_side_by_side_without_a_winner() -> None:
    first = _batch()
    second = dict(_batch())
    second["candidate_id"] = "groot_n17_droid"

    summary = summarize_candidate_batches([second, first])

    assert summary["candidate_count"] == 2
    # Stable ordering, so two runs of the same pair read identically.
    assert [row["candidate_id"] for row in summary["candidates"]] == [
        "groot_n17_droid",
        "pi05_droid",
    ]
    # It ranks now, but the sample-size caveat travels with the ordering.
    assert summary["supports_policy_ranking"] is False
    assert "paired sample size" in summary["why_not_adjudicated"]


def test_harness_fault_is_retained_but_excluded_from_candidate_ranking() -> None:
    class _DroppedActions(_Environment):
        def step(self, isaac_action):
            self.steps.append(list(isaac_action))
            self._t += 1

    batch = _batch(environment=_DroppedActions())
    summary = summarize_candidate_batches([batch])

    assert batch["episodes_scored"] == 3
    assert batch["episodes_policy_outcome_interpretable"] == 0
    assert batch["episodes_policy_outcome_uninterpretable"] == 3
    assert batch["supports_policy_outcome_interpretation"] is False
    assert batch["outcome_counts"] == {"placed": 3}
    assert batch["interpretable_policy_outcome_counts"] == {}
    assert all(
        row["policy_outcome_interpretation"]
        == "nontrivial_actions_not_observed_at_robot_harness_fault"
        for row in batch["episodes"]
    )
    assert all(row["policy_outcome"] is None for row in batch["episodes"])
    assert all(row["harness_finding"] for row in batch["episodes"])
    assert summary["ranking"] == []
    assert summary["leader"] is None
    assert summary["candidates"][0]["mean_outcome_rank"] is None
    assert summary["candidates"][0]["outcome_counts"] == {}
    assert summary["candidates"][0]["raw_object_state_outcome_counts"] == {
        "placed": 3
    }
    assert summary["candidates"][0]["episodes_policy_outcome_uninterpretable"] == 3


def test_receipts_are_digest_bound() -> None:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    batch = _batch()
    assert batch["receipt_digest"] == canonical_digest(
        batch, digest_field="receipt_digest"
    )
    summary = summarize_candidate_batches([batch])
    assert summary["receipt_digest"] == canonical_digest(
        summary, digest_field="receipt_digest"
    )


def test_batch_retains_complete_media_for_every_scored_episode(tmp_path) -> None:
    batch = _batch(episodes=2, media_output_dir=tmp_path)

    assert batch["episodes_media_complete"] == 2
    assert batch["episodes_media_incomplete"] == 0
    assert batch["all_scored_episode_media_complete"] is True
    assert [row["episode_id"] for row in batch["episodes"]] == [
        "pi05_droid-episode-000",
        "pi05_droid-episode-001",
    ]
    assert all(
        (row["visual_evidence"] or {}).get("human_review_available") is True
        for row in batch["episodes"]
    )


def test_batch_rows_retain_step_trace_motion_quality_and_dataset_capture(
    tmp_path,
) -> None:
    """What the loop measures must survive into the persisted batch rows."""

    from blueprint_pipeline.adp009d_dataset_capture import DatasetCaptureRecorder
    from blueprint_pipeline.adp009d_droid_observation import (
        DROID_EXTERIOR_VIEW_1,
        DROID_WRIST_VIEW,
    )

    captured_ids: list[str] = []

    def factory(episode_id: str) -> DatasetCaptureRecorder:
        captured_ids.append(episode_id)
        return DatasetCaptureRecorder(
            output_dir=tmp_path,
            episode_id=episode_id,
            view_keys=(DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW),
        )

    batch = _batch(
        episodes=2,
        media_output_dir=tmp_path,
        dataset_capture_factory=factory,
    )

    assert batch["schema_version"] == "adp009d_episode_batch.v3"
    assert batch["episodes_scored"] == 2
    for row in batch["episodes"]:
        assert row["step_trace"]["total_steps"] == row["environment_steps"]
        assert row["step_trace"]["control_hz"] == 15
        assert row["motion_quality"]["observed_joint_velocity_max_abs_rad_s"] > 0.0
        assert row["dataset_contract"]["control_hz"] == 15
        assert row["object_samples"][0]["step_index"] == 0
        assert row["dataset_capture"]["frame_count"] == row["environment_steps"]
    assert captured_ids == [row["episode_id"] for row in batch["episodes"]]


def test_dataset_capture_factory_requires_media_retention() -> None:
    with pytest.raises(EpisodeBatchError):
        _batch(dataset_capture_factory=lambda episode_id: None)
