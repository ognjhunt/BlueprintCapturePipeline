"""Secret-clean terminal projection for internal unqualified policy canaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .adp_task_scoring import (
    TaskNeutralScoringError,
    validate_rigid_task_success_contract,
)
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .rigid_task_success_contract_schema import rigid_task_success_contract_schema


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
        value["$defs"]["taskSuccessContract"] = (
            rigid_task_success_contract_schema()
        )
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
    try:
        task_success_contract = validate_rigid_task_success_contract(
            result["task_success_contract"]
        )
    except TaskNeutralScoringError as exc:
        raise TaskEvaluationPolicyCanaryResultError(
            "policy_canary_result_task_success_contract_invalid:" + str(exc)
        ) from exc
    if (
        result["task_success_contract_digest"]
        != task_success_contract["contract_digest"]
    ):
        raise TaskEvaluationPolicyCanaryResultError(
            "policy_canary_result_task_success_contract_digest_mismatch"
        )
    counts = result["counts"]
    if (
        counts["completed_learned_policy_rollout_count"]
        != sum(
            episode["terminal_state"] == "completed"
            for episode in result["episodes"]
        )
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
    from .policy_canary_control_result_delivery import WARNINGS, CONTROL_IDS, control_summary
    if result["warning"] != WARNINGS[result["scene_controls_status"]]:
        raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_controls_warning_invalid")
    controls = result.get("controls")
    if controls is not None:
        summary = control_summary(controls, result["episodes"])
        pairs = {(row["cell_id"], row["control_id"]) for row in controls}
        if (len(pairs) != len(controls) or result.get("controls_summary") != summary
                or counts["completed_diagnostic_control_rollout_count"] != summary["completed_count"]
                or any(row["control_id"] not in CONTROL_IDS for row in controls)):
            raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_controls_counts_invalid")
        for row in controls:
            score = row["score"]
            expected = score.get("task_succeeded") is (row["control_id"] == "deterministic_scripted_positive")
            if row["control_id"] == "zero_action_negative":
                expected = expected and score.get("outcome") == "never_moved"
            if row["control_passed"] and (not expected or score.get("status") != "scored"
                    or row["terminal_state"] != "completed" or row["evidence_gaps"]):
                raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_control_score_invalid")
        gate = result.get("controls_gate") or {}
        verified = (summary["recorded_count"] == summary["passed_count"] == 20 and summary["verified_cell_count"] == 10
                    and gate.get("status") == "passed" and gate.get("required_control_episode_count") == 20
                    and gate.get("candidate_policies_loaded_during_controls") is False)
        expected_status = "controls_verified_development_only" if verified else "controls_failed"
        if result["scene_controls_status"] != expected_status:
            raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_controls_status_invalid")
        if not verified and result["result_status"] == "completed_unqualified":
            raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_required_controls_unverified")
    elif result["scene_controls_status"] != "configured_controls_pending":
        raise TaskEvaluationPolicyCanaryResultError("policy_canary_result_controls_evidence_missing")
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
