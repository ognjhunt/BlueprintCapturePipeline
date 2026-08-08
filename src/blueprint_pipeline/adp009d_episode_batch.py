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

try:  # flat provider-bundle layout
    from adp009d_policy_episode import PolicyEpisodeError, run_policy_episode
except ModuleNotFoundError:  # repository package
    from .adp009d_policy_episode import PolicyEpisodeError, run_policy_episode
try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest

BATCH_SCHEMA_VERSION = "adp009d_episode_batch.v2"

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
            motion_evidence = dict(receipt.get("motion_evidence") or {})
            policy_outcome_interpretable = (
                motion_evidence.get("policy_outcome_interpretable") is True
            )
            deterministic_object_outcome = receipt["score"].get("outcome")
            interpretation = motion_evidence.get("interpretation")
            rows.append(
                {
                    "episode_index": index,
                    "status": "scored",
                    # Keep the deterministic object-state score even when the
                    # action-delivery evidence is insufficient, but never
                    # present that score as a policy outcome in that case.
                    "outcome": deterministic_object_outcome,
                    "deterministic_object_outcome": deterministic_object_outcome,
                    "policy_outcome": (
                        deterministic_object_outcome
                        if policy_outcome_interpretable
                        else None
                    ),
                    "outcome_rank": receipt["score"].get("outcome_rank"),
                    "score_status": receipt["score"].get("status"),
                    "environment_steps": receipt.get("environment_steps"),
                    "policy_queries": receipt.get("policy_queries"),
                    "queries": receipt.get("queries"),
                    "joint_position_reset_rad": motion_evidence.get(
                        "joint_position_reset_rad"
                    ),
                    "joint_position_end_rad": motion_evidence.get(
                        "joint_position_end_rad"
                    ),
                    "max_abs_joint_delta_from_reset_rad": motion_evidence.get(
                        "max_abs_joint_delta_from_reset_rad"
                    ),
                    "any_joint_limit_clamped_count": sum(
                        bool(query.get("any_joint_limit_clamped"))
                        for query in (receipt.get("queries") or [])
                    ),
                    "joint_limit_clamped_action_count": sum(
                        int(query.get("joint_limit_clamped_rows") or 0)
                        for query in (receipt.get("queries") or [])
                    ),
                    "commanded_action_magnitudes": receipt.get(
                        "commanded_action_magnitudes"
                    ),
                    "arm_moved": motion_evidence.get("arm_moved"),
                    "actions_reached_robot": motion_evidence.get(
                        "actions_reached_robot"
                    ),
                    "policy_outcome_interpretable": policy_outcome_interpretable,
                    "policy_outcome_interpretation": interpretation,
                    "harness_finding": (
                        None if policy_outcome_interpretable else interpretation
                    ),
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
    policy_outcomes: dict[str, int] = {}
    for row in rows:
        if row["status"] == "scored":
            key = str(row.get("outcome"))
            outcomes[key] = outcomes.get(key, 0) + 1
            if row.get("policy_outcome_interpretable") is True:
                policy_key = str(row.get("policy_outcome"))
                policy_outcomes[policy_key] = policy_outcomes.get(policy_key, 0) + 1

    scored = [row for row in rows if row["status"] == "scored"]
    interpretable = [
        row for row in scored if row.get("policy_outcome_interpretable") is True
    ]
    uninterpretable = [
        row for row in scored if row.get("policy_outcome_interpretable") is not True
    ]
    distinct = {row.get("outcome") for row in scored}
    batch: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "episodes_requested": int(episodes),
        "episodes_scored": len(scored),
        "episodes_failed": len(rows) - len(scored),
        "episodes_policy_outcome_interpretable": len(interpretable),
        "episodes_policy_outcome_uninterpretable": len(uninterpretable),
        # Compatibility name for the deterministic object-state scorer.  It is
        # explicitly typed because these counts can include harness findings.
        "outcome_counts": dict(sorted(outcomes.items())),
        "outcome_counts_kind": (
            "deterministic_object_state_including_uninterpretable_harness_findings"
        ),
        "interpretable_policy_outcome_counts": dict(sorted(policy_outcomes.items())),
        "episodes": rows,
        # Every episode resets to the sealed start, so a single outcome across
        # all of them is repeatability rather than one trajectory resampled.
        "outcomes_identical_across_episodes": len(distinct) <= 1,
        # Stated rather than implied.  These counts do not support ranking two
        # policies, and the receipt says so where a reader will see it.
        "sample_purpose": "pipeline_proof_not_policy_comparison",
        "supports_policy_ranking": False,
        "supports_policy_outcome_interpretation": bool(scored) and not uninterpretable,
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
    """Rank candidates by how far each got, and say what the rank is worth.

    An ordering is reported rather than withheld: a reader asking which policy
    did better on these episodes deserves the answer the data gives.  What the
    ordering must not do is impersonate an adjudicated result, so the sample
    size travels attached to it -- ``supports_policy_ranking`` stays False, and
    ``ranking_basis`` names the statistic.

    The rank is the mean outcome rung on the scoring ladder, which orders by
    how far the task actually progressed rather than by a binary success that
    would throw away the difference between never moving the can and lifting
    it but failing to place it.  Ties are reported as ties.
    """

    rows = []
    for batch in batches:
        scored_episodes = [
            e for e in (batch.get("episodes") or []) if e.get("status") == "scored"
        ]
        episodes = [
            e for e in scored_episodes if e.get("policy_outcome_interpretable") is True
        ]
        policy_outcome_counts: dict[str, int] = {}
        for episode in episodes:
            outcome = episode.get("policy_outcome", episode.get("outcome"))
            key = str(outcome)
            policy_outcome_counts[key] = policy_outcome_counts.get(key, 0) + 1
        ranks = [
            int(e["outcome_rank"]) for e in episodes if e.get("outcome_rank") is not None
        ]
        rows.append(
            {
                "candidate_id": batch.get("candidate_id"),
                "episodes_scored": batch.get("episodes_scored"),
                "episodes_failed": batch.get("episodes_failed"),
                "episodes_policy_outcome_interpretable": len(episodes),
                "episodes_policy_outcome_uninterpretable": (
                    len(scored_episodes) - len(episodes)
                ),
                "outcome_counts": dict(sorted(policy_outcome_counts.items())),
                "raw_object_state_outcome_counts": batch.get("outcome_counts"),
                "mean_outcome_rank": (sum(ranks) / len(ranks)) if ranks else None,
                "best_outcome": max(
                    (e.get("outcome") for e in episodes if e.get("outcome")),
                    key=lambda o: [e["outcome_rank"] for e in episodes if e.get("outcome") == o][0],
                    default=None,
                ) if ranks else None,
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["mean_outcome_rank"] is None,
            -(row["mean_outcome_rank"] or 0.0),
            str(row["candidate_id"]),
        ),
    )
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position if row["mean_outcome_rank"] is not None else None
    scores = [r["mean_outcome_rank"] for r in ranked if r["mean_outcome_rank"] is not None]
    tied = len(scores) > 1 and len(set(scores)) == 1

    summary: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "candidates": ranked,
        "candidate_count": len(rows),
        "ranking": [r["candidate_id"] for r in ranked if r["rank"] is not None],
        "ranking_basis": "mean_outcome_rank_on_the_task_scoring_ladder",
        "tied": tied,
        "leader": (
            None if (tied or not scores) else ranked[0]["candidate_id"]
        ),
        # Stated with the ordering, not instead of it.  These counts show which
        # policy did better on these episodes; they do not establish that it is
        # the better policy.
        "supports_policy_ranking": False,
        "policy_outcomes_excluded_without_action_delivery_evidence": True,
        "why_not_adjudicated": (
            "a handful of episodes per candidate orders what happened here; "
            "a claim about which policy is better needs the frozen scenario "
            "suite, its controls and a paired sample size computed for stated "
            "power"
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
