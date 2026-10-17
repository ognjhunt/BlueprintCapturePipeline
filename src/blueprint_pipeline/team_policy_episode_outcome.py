"""Pure shared policy-outcome classification and the first-batch spending gate."""
from collections.abc import Mapping, Sequence
from typing import Any

FIRST_BATCH_CELLS = 25
INTERFACE_FAILURE_REASONS = frozenset({"policy_service_did_not_answer", "policy_action_invalid",
                                      "policy_service_error", "policy_action_beyond_joint_limits"})

_POLICY_ERROR_PREFIXES = ("droid_https_policy_response_", "team_policy_https_response_",
                          "droid_https_policy_identity_mismatch")


def _scored(row: Mapping[str, Any]) -> bool | None:
    """The simulator's verdict for one episode, or None when it was not scored."""

    score = (row.get("episode") or {}).get("score") if isinstance(row.get("episode"), Mapping) else None
    if (row.get("status") != "completed" or row.get("scoring_authority") != "deterministic_simulator_state"
            or not isinstance(score, Mapping) or score.get("status") != "scored"
            or not isinstance(score.get("task_succeeded"), bool)):
        return None
    return score["task_succeeded"]


def episode_outcome(row: Mapping[str, Any]) -> tuple[str, str]:
    """``(outcome, reason)``: success, failure (scored or policy-caused) or void (simulator)."""

    verdict = _scored(row)
    motion = (row.get("episode") or {}).get("motion_evidence") if isinstance(row.get("episode"), Mapping) else None
    if verdict is not None and isinstance(motion, Mapping) and motion.get("robot_fell") is True:
        # A free base that fell failed the job, whatever became of the object.
        return "failure", "robot_fell"
    if verdict is not None:
        return ("success", "task_succeeded") if verdict else ("failure", "task_not_completed")
    if row.get("status") == "completed":
        return "void", "simulator_score_undetermined"
    if row.get("episode_failure_stage") == "action_delivery_rejected":
        return "failure", "policy_action_beyond_joint_limits"
    attempted = row.get("candidate_policy_query_attempted") is True or row.get("candidate_policy_queried") is True
    message = str(row.get("failure_message") or "")
    if attempted:
        if row.get("episode_failure_cause") == "policy":
            return "failure", "policy_service_error"
        if row.get("candidate_action_returned") is True and not (
                row.get("candidate_action_shape_validated") is True
                and row.get("candidate_action_finite_validated") is True):
            return "failure", "policy_action_invalid"
        if message.startswith(_POLICY_ERROR_PREFIXES):
            return "failure", "policy_service_error"
        if (row.get("episode_failure_cause") not in ("infrastructure", "media")
                and row.get("policy_response_status") not in (None, "received")):
            return "failure", "policy_service_did_not_answer"
    visual = row.get("visual_evidence") if isinstance(row.get("visual_evidence"), Mapping) else {}
    gap = visual.get("media_gap") if isinstance(visual.get("media_gap"), Mapping) else {}
    if str(gap.get("type") or "").startswith("after_first_observation_media_"):
        return "void", "episode_media_unavailable"
    if row.get("episode_failure_stage") == "after_first_observation":
        return "void", "simulator_failed_during_episode"
    return "void", "simulator_failed_before_policy"


def first_batch_failure(rows: Sequence[Mapping[str, Any]]) -> str | None:
    reasons: dict[str, int] = {}
    for row in rows:
        _, reason = episode_outcome(row)
        if reason in INTERFACE_FAILURE_REASONS:
            reasons[reason] = reasons.get(reason, 0) + 1
    if rows and sum(reasons.values()) > len(rows) / 2:
        return f"team_policy_interface_failures:{len(rows)}_episodes:" + ",".join(
            f"{name}={count}" for name, count in sorted(reasons.items()))
    return None
