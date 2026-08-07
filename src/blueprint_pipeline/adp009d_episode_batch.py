"""Run a few episodes per candidate and report them comparably.

This is a proof pipeline, not an adjudicated benchmark.  The repository already
carries the machinery for the latter -- eight scenario families, cousin
manifests, a McNemar paired sample size and Wilson intervals -- and none of it
is appropriate here: those exist to decide between two policies with stated
statistical power, which needs dozens of episodes per cell and a suite frozen
before any outcome exists.

What a proof needs instead is narrower and honest about itself: a handful of
episodes per candidate, each scored the same way, aggregated without any claim
the sample supports a decision.  So this refuses to emit a comparison verdict at
all.  It reports what happened per episode and the counts, and records that the
sample is a demonstration -- because the easiest way to discredit a pipeline is
to let it look like it decided something it cannot.

Each episode resets to the sealed start, so episodes are independent rather than
a single long trajectory cut into pieces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_policy_episode import PolicyEpisodeError, run_policy_episode
from .decision_evidence_contracts import canonical_digest

BATCH_SCHEMA_VERSION = "adp009d_episode_batch.v1"

# A proof, deliberately.  Enough to show the path runs repeatably and to surface
# gross nondeterminism; nowhere near enough to rank two policies.
DEFAULT_EPISODES_PER_CANDIDATE = 3
MAX_EPISODES_PER_CANDIDATE = 25

BLOCKER_NO_EPISODES = "episode_batch_no_episodes_requested"
BLOCKER_TOO_MANY = "episode_batch_exceeds_proof_scale"


class EpisodeBatchError(ValueError):
    """Fail-closed batch contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def run_episode_batch(
    *,
    environment: Any,
    policy: Any,
    candidate_id: str,
    destination_position_world_m: Sequence[float],
    prompt: str,
    gripper: Any,
    episodes: int = DEFAULT_EPISODES_PER_CANDIDATE,
    **episode_kwargs: Any,
) -> dict[str, Any]:
    """Run ``episodes`` independent episodes and report them without ranking."""

    if int(episodes) < 1:
        raise EpisodeBatchError([BLOCKER_NO_EPISODES])
    if int(episodes) > MAX_EPISODES_PER_CANDIDATE:
        # A proof pipeline asking for benchmark scale is a scope error, and
        # silently obliging would produce numbers that invite a decision.
        raise EpisodeBatchError([f"{BLOCKER_TOO_MANY}:{episodes}"])

    rows: list[dict[str, Any]] = []
    for index in range(int(episodes)):
        try:
            receipt = run_policy_episode(
                environment=environment,
                policy=policy,
                candidate_id=candidate_id,
                destination_position_world_m=destination_position_world_m,
                prompt=prompt,
                gripper=gripper,
                **episode_kwargs,
            )
            rows.append(
                {
                    "episode_index": index,
                    "status": "scored",
                    "outcome": receipt["score"].get("outcome"),
                    "score_status": receipt["score"].get("status"),
                    "environment_steps": receipt.get("environment_steps"),
                    "policy_queries": receipt.get("policy_queries"),
                    "receipt_digest": receipt.get("receipt_digest"),
                }
            )
        except (PolicyEpisodeError, ValueError, RuntimeError) as exc:
            # A failed episode is evidence, not an excuse to abandon the batch:
            # one bad episode must not erase the ones that ran.
            rows.append(
                {
                    "episode_index": index,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    outcomes: dict[str, int] = {}
    for row in rows:
        if row["status"] == "scored":
            key = str(row.get("outcome"))
            outcomes[key] = outcomes.get(key, 0) + 1

    scored = [row for row in rows if row["status"] == "scored"]
    distinct = {row.get("outcome") for row in scored}
    batch: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "episodes_requested": int(episodes),
        "episodes_scored": len(scored),
        "episodes_failed": len(rows) - len(scored),
        "outcome_counts": dict(sorted(outcomes.items())),
        "episodes": rows,
        # Every episode resets to the sealed start, so a single outcome across
        # all of them is repeatability rather than one trajectory resampled.
        "outcomes_identical_across_episodes": len(distinct) <= 1,
        # Stated rather than implied.  These counts do not support ranking two
        # policies, and the receipt says so where a reader will see it.
        "sample_purpose": "pipeline_proof_not_policy_comparison",
        "supports_policy_ranking": False,
        "ranking_requires": (
            "the frozen scenario suite and paired sample size in "
            "adp009d_franka_evaluation_harness, which this deliberately is not"
        ),
    }
    batch["receipt_digest"] = canonical_digest(batch, digest_field="receipt_digest")
    return batch


def summarize_candidate_batches(
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Put candidates side by side without declaring a winner."""

    rows = [
        {
            "candidate_id": batch.get("candidate_id"),
            "episodes_scored": batch.get("episodes_scored"),
            "episodes_failed": batch.get("episodes_failed"),
            "outcome_counts": batch.get("outcome_counts"),
        }
        for batch in batches
    ]
    summary: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "candidates": sorted(rows, key=lambda row: str(row["candidate_id"])),
        "candidate_count": len(rows),
        "comparison_verdict": None,
        "supports_policy_ranking": False,
        "why_no_verdict": (
            "a proof pipeline runs a handful of episodes per candidate; "
            "ranking needs the frozen suite, its controls and a paired sample "
            "size computed for stated power"
        ),
    }
    summary["receipt_digest"] = canonical_digest(
        summary, digest_field="receipt_digest"
    )
    return summary


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "DEFAULT_EPISODES_PER_CANDIDATE",
    "MAX_EPISODES_PER_CANDIDATE",
    "EpisodeBatchError",
    "run_episode_batch",
    "summarize_candidate_batches",
]
