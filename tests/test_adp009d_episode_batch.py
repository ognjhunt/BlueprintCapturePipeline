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


def test_the_batch_refuses_to_support_a_ranking() -> None:
    """The easiest way to discredit a proof is to let it look like a decision."""

    batch = _batch()

    assert batch["supports_policy_ranking"] is False
    assert batch["sample_purpose"] == "pipeline_proof_not_policy_comparison"
    assert "paired sample size" in batch["ranking_requires"]

    summary = summarize_candidate_batches([batch])
    assert summary["comparison_verdict"] is None
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
    assert summary["comparison_verdict"] is None


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
