"""Secret-clean terminal projection for internal unqualified policy canaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest


SCHEMA_VERSION = "task_evaluation_policy_canary_result_projection.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "schemas" / f"{SCHEMA_VERSION}.schema.json"
)


class TaskEvaluationPolicyCanaryResultError(ValueError):
    """A canary result would overclaim or omit terminal evidence."""


@lru_cache(maxsize=1)
def policy_canary_result_schema() -> dict[str, Any]:
    import jsonschema

    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(value)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_schema_invalid") from exc
    return deepcopy(value)


def validate_policy_canary_result(value: Mapping[str, Any]) -> dict[str, Any]:
    import jsonschema

    result = deepcopy(dict(value))
    errors = sorted(
        jsonschema.Draft202012Validator(
            policy_canary_result_schema(),
            format_checker=jsonschema.FormatChecker(),
        ).iter_errors(result),
        key=lambda row: list(row.path),
    )
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationPolicyCanaryResultError(f"policy_canary_result_invalid:{path}")
    if result["projection_digest"] != cross_runtime_canonical_digest(
        result, digest_field="projection_digest"
    ):
        raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_digest_mismatch")
    counts = result["counts"]
    if (
        counts["completed_learned_policy_rollout_count"] != len(result["episodes"])
        or counts["completed_diagnostic_control_rollout_count"]
        > counts["diagnostic_control_rollout_count"]
        or [row["candidate_id"] for row in result["candidate_results"]]
        != ["pi05_droid", "groot_n17_droid"]
    ):
        raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_episode_counts_invalid")
    if result["result_status"] == "completed_unqualified":
        if counts["completed_learned_policy_rollout_count"] != 20 or result["blockers"]:
            raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_completion_invalid")
    elif not result["blockers"]:
        raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_terminal_blocker_missing")
    for episode in result["episodes"]:
        if episode["terminal_state"] == "completed" and not (
            episode["candidate_policy_queried"] and episode["actions_reached_robot"]
        ):
            raise TaskEvaluationPolicyCanaryResultError(
                "policy_canary_result_completed_episode_execution_unproven"
            )
        if not episode["actions_reached_robot"] and episode["policy_outcome_interpretable"]:
            raise TaskEvaluationPolicyCanaryResultError(
                "policy_canary_result_action_delivery_interpretability_invalid"
            )
        evidence = episode["evidence"]
        artifact_roles = (
            "reset_state",
            "frame_manifest",
            "review_video",
            "policy_query_receipt",
            "action_sequence",
            "action_delivery_readback",
            "state_trace",
            "contact_force_trace",
            "task_object_trajectory",
            "score_receipt",
        )
        if episode["terminal_state"] == "completed" and (
            evidence["evidence_gaps"] or any(evidence[role] is None for role in artifact_roles)
        ):
            raise TaskEvaluationPolicyCanaryResultError(
                "policy_canary_result_completed_episode_evidence_incomplete"
            )
    notification = result["notification_delivery"]
    if notification["status"] == "pending":
        if (
            notification["attempts"] != 0
            or notification["message_id"] is not None
            or notification["delivered_at"] is not None
        ):
            raise TaskEvaluationPolicyCanaryResultError(
                "policy_canary_result_notification_pending_invalid"
            )
    elif notification["attempts"] < 1:
        raise TaskEvaluationPolicyCanaryResultError(
            "policy_canary_result_notification_attempts_invalid"
        )
    return result


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "TaskEvaluationPolicyCanaryResultError",
    "policy_canary_result_schema",
    "validate_policy_canary_result",
]
