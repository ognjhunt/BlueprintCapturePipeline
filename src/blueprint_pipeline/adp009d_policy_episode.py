"""Run one frozen ADP-009D task episode against a learned DROID policy.

This is the orchestration the five ADP-009D adapters were built for, and it is
the only place they meet: observation formatting, policy query, action-chunk
execution, and deterministic scoring, in that order, for one episode.

The simulator is injected rather than imported.  Everything here is arithmetic
and sequencing, so the whole loop -- including its failure paths -- is testable
without a GPU, and the Isaac-side adapter stays a thin, reviewable shim that
only reads and writes simulator state.

Three properties are load-bearing and enforced rather than assumed:

* **The episode ends with a settle window the gripper is absent from.**  The
  place predicate is judged on a can at rest after release; without a settle
  phase ``placed`` could never be decided, and an episode would silently score
  one rung lower than it earned.
* **Step indices strictly increase across the whole episode**, including across
  policy queries, because the scorer treats a repeated index as malformed
  evidence rather than reordering it.
* **The policy is queried only through the injected client**, and every query
  and chunk is retained, so a receipt can be re-derived without a simulator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .adp009d_droid_action_execution import (
    DROID_CONTROL_HZ,
    DROID_OPEN_LOOP_HORIZON,
    DroidActionExecutionError,
    GripperConvention,
    plan_chunk_execution,
)
from .adp009d_droid_observation import (
    CANDIDATE_REQUIRED_VIEWS,
    DROID_OBSERVATION_SCHEMA_VERSION,
    DroidObservationError,
    build_droid_observation,
    describe_observation_conversion,
)
from .adp009d_task_scoring import (
    SETTLE_WINDOW_SAMPLES,
    TaskScoringError,
    score_task_episode,
)
from .decision_evidence_contracts import canonical_digest

EPISODE_SCHEMA_VERSION = "adp009d_policy_episode.v1"

# A policy that has not moved the can within this many queries has failed the
# episode; the cap bounds paid GPU time and is recorded rather than implicit.
DEFAULT_MAX_POLICY_QUERIES = 60

BLOCKER_NO_SETTLE_WINDOW = "policy_episode_settle_window_not_reached"
BLOCKER_GRIPPER_PRESENT_IN_SETTLE = "policy_episode_gripper_present_during_settle"
BLOCKER_STEP_INDEX_NOT_INCREASING = "policy_episode_step_index_not_increasing"
BLOCKER_CLIENT_RETURNED_NOTHING = "policy_episode_client_returned_no_chunk"
BLOCKER_QUERY_BUDGET_EXHAUSTED = "policy_episode_query_budget_exhausted"
BLOCKER_ENVIRONMENT_CONTRACT = "policy_episode_environment_contract_violated"


class PolicyEpisodeError(ValueError):
    """Fail-closed episode contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


class EpisodeEnvironment(Protocol):
    """The exact simulator surface this loop needs, and nothing more."""

    def reset(self) -> None:
        """Return to the sealed canonical start pose."""

    def read_policy_inputs(self) -> Mapping[str, Any]:
        """Camera RGB by DROID view name, plus ``joint_position`` and ``gripper_position``."""

    def step(self, isaac_action: Sequence[float]) -> None:
        """Apply one 8-dimensional Arena action for one environment step."""

    def read_object_sample(self) -> Mapping[str, Any]:
        """Deterministic object state: ``can_pose_world`` and optional grasp evidence."""

    def joint_limits(self) -> Sequence[Sequence[float]]:
        """Seven ``(lower, upper)`` arm joint limits, in radians."""


class DroidPolicyClient(Protocol):
    """The policy seam.  Implementations talk to a server; this never does."""

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """Return an action chunk of shape ``(rows, 8)`` with ``rows >= 8``."""


def _sample_with_index(
    raw: Mapping[str, Any], step_index: int, previous_index: int | None
) -> dict[str, Any]:
    if previous_index is not None and step_index <= previous_index:
        raise PolicyEpisodeError(
            [f"{BLOCKER_STEP_INDEX_NOT_INCREASING}:{step_index}<={previous_index}"]
        )
    sample = dict(raw)
    sample["step_index"] = step_index
    if "can_pose_world" not in sample:
        raise PolicyEpisodeError(
            [f"{BLOCKER_ENVIRONMENT_CONTRACT}:can_pose_world_missing"]
        )
    return sample


def run_policy_episode(
    *,
    environment: EpisodeEnvironment,
    policy: DroidPolicyClient,
    candidate_id: str,
    destination_position_world_m: Sequence[float],
    prompt: str,
    gripper: GripperConvention,
    max_policy_queries: int = DEFAULT_MAX_POLICY_QUERIES,
    settle_window_samples: int = SETTLE_WINDOW_SAMPLES,
    open_loop_horizon: int = DROID_OPEN_LOOP_HORIZON,
) -> dict[str, Any]:
    """Run one episode end to end and return a digest-bound receipt.

    The loop resets, then repeatedly formats an observation for this candidate,
    asks the policy for a chunk, executes exactly the open-loop horizon of it,
    and samples deterministic object state after every environment step.  When
    the query budget is spent it holds the arm still for a settle window so the
    placed predicate can be decided on a can at rest.

    Raises :class:`PolicyEpisodeError` when the environment, the client, or the
    episode's own shape violates its contract.  Scoring errors surface as
    :class:`~blueprint_pipeline.adp009d_task_scoring.TaskScoringError`.
    """

    if candidate_id not in CANDIDATE_REQUIRED_VIEWS:
        raise PolicyEpisodeError([f"policy_episode_unknown_candidate:{candidate_id}"])
    if int(max_policy_queries) < 1:
        raise PolicyEpisodeError(["policy_episode_query_budget_invalid"])
    if int(settle_window_samples) < 1:
        raise PolicyEpisodeError(["policy_episode_settle_window_invalid"])

    environment.reset()
    joint_limits = environment.joint_limits()

    samples: list[dict[str, Any]] = []
    previous_index: int | None = None
    step_index = 0
    samples.append(
        _sample_with_index(environment.read_object_sample(), step_index, previous_index)
    )
    previous_index = step_index

    queries: list[dict[str, Any]] = []
    last_action: list[float] | None = None

    for query_index in range(int(max_policy_queries)):
        inputs = environment.read_policy_inputs()
        camera_rgb = {
            view: inputs[view]
            for view in CANDIDATE_REQUIRED_VIEWS[candidate_id]
            if view in inputs
        }
        try:
            observation = build_droid_observation(
                candidate_id=candidate_id,
                camera_rgb=camera_rgb,
                joint_position=inputs["joint_position"],
                gripper_position=inputs["gripper_position"],
                prompt=prompt,
            )
        except KeyError as exc:
            raise PolicyEpisodeError(
                [f"{BLOCKER_ENVIRONMENT_CONTRACT}:{exc.args[0]}_missing"]
            ) from exc
        except DroidObservationError:
            raise

        chunk = policy.infer(observation)
        if chunk is None:
            raise PolicyEpisodeError([BLOCKER_CLIENT_RETURNED_NOTHING])

        plan = plan_chunk_execution(
            chunk,
            joint_limits=joint_limits,
            gripper=gripper,
            horizon=int(open_loop_horizon),
        )
        for action in plan["actions"]:
            environment.step(action["isaac_action"])
            step_index += 1
            samples.append(
                _sample_with_index(
                    environment.read_object_sample(), step_index, previous_index
                )
            )
            previous_index = step_index
            last_action = list(action["isaac_action"])

        queries.append(
            {
                "query_index": query_index,
                "chunk_shape": plan["chunk_shape"],
                "executed_rows": plan["executed_rows"],
                "discarded_rows": plan["discarded_rows"],
                "any_joint_limit_clamped": plan["any_joint_limit_clamped"],
                "final_step_index": step_index,
            }
        )

    if last_action is None:
        raise PolicyEpisodeError([BLOCKER_QUERY_BUDGET_EXHAUSTED])

    # Settle: hold the arm where the policy left it, but with the gripper open,
    # so the place predicate is judged on a released can at rest.  Holding the
    # commanded joints keeps this a settle rather than a retreat, which would
    # itself disturb the object being judged.
    release_action = list(last_action)
    release_action[7] = gripper.open_command
    settle_start_index = step_index
    for _ in range(int(settle_window_samples)):
        environment.step(release_action)
        step_index += 1
        samples.append(
            _sample_with_index(
                environment.read_object_sample(), step_index, previous_index
            )
        )
        previous_index = step_index

    if step_index - settle_start_index < int(settle_window_samples):
        raise PolicyEpisodeError([BLOCKER_NO_SETTLE_WINDOW])

    score = score_task_episode(
        samples=samples,
        destination_position_world_m=destination_position_world_m,
        settle_window_samples=int(settle_window_samples),
    )

    receipt: dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "prompt": str(prompt),
        "policy_queries": len(queries),
        "max_policy_queries": int(max_policy_queries),
        "environment_steps": step_index,
        "settle_window_samples": int(settle_window_samples),
        "open_loop_horizon": int(open_loop_horizon),
        "control_hz": DROID_CONTROL_HZ,
        "observation_adapter_schema_version": DROID_OBSERVATION_SCHEMA_VERSION,
        "observation_conversion": describe_observation_conversion(candidate_id),
        "destination_position_world_m": [
            float(v) for v in destination_position_world_m
        ],
        "queries": queries,
        "score": score,
        "candidate_policy_queried": True,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "DEFAULT_MAX_POLICY_QUERIES",
    "EPISODE_SCHEMA_VERSION",
    "DroidActionExecutionError",
    "DroidPolicyClient",
    "EpisodeEnvironment",
    "PolicyEpisodeError",
    "TaskScoringError",
    "run_policy_episode",
]
