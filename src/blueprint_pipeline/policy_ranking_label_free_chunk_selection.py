"""Prospective, outcome-blind action-chunk selection for WAM canaries.

The OSCAR-compatible selector considers only the first 16-step chunk of each
frozen session/policy trace, so the conditioning observation remains the
recorded first RGB frame.  It ranks motion using action geometry only and never
opens metadata, task outcomes, generated media, or evaluator predictions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .policy_ranking_successor_cosmos import validate_droid_action_stream
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "policy_ranking_label_free_chunk_selection.v1"
SELECTION_RULE_ID = "first_frame_compatible_max_translation_path_v1"
FORBIDDEN_FIELDS = frozenset(
    {
        "outcome",
        "outcomes",
        "success",
        "success_rate",
        "score",
        "rank",
        "ranking",
        "evaluator",
        "prediction",
    }
)


def _rot6d_angle_radians(row: Sequence[float]) -> float:
    first = np.asarray(row[3:6], dtype=np.float64)
    second = np.asarray(row[6:9], dtype=np.float64)
    third = np.cross(first, second)
    rotation = np.column_stack((first, second, third))
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def action_chunk_motion_metrics(action_stream: Mapping[str, Any]) -> dict[str, Any]:
    """Measure a valid Cosmos/DROID chunk without task or outcome information."""

    validated = validate_droid_action_stream(action_stream)
    actions = np.asarray(validated["actions"], dtype=np.float64)
    translation = np.linalg.norm(actions[:, :3], axis=1)
    rotation = np.asarray([_rot6d_angle_radians(row) for row in actions], dtype=float)
    gripper = actions[:, 9]
    result: dict[str, Any] = {
        "translation_path_length_m": float(np.sum(translation)),
        "translation_mean_per_step_m": float(np.mean(translation)),
        "translation_max_step_m": float(np.max(translation)),
        "nontrivial_translation_step_count_1mm": int(np.sum(translation >= 0.001)),
        "rotation_path_length_rad": float(np.sum(rotation)),
        "rotation_mean_per_step_rad": float(np.mean(rotation)),
        "rotation_max_step_rad": float(np.max(rotation)),
        "gripper_total_variation": float(np.sum(np.abs(np.diff(gripper)))),
        "gripper_transition_count": int(np.sum(np.abs(np.diff(gripper)) > 1e-9)),
        "action_sha256": str(validated["action_sha256"]),
    }
    result["metrics_sha256"] = canonical_sha256(result)
    return result


def _validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    leaked = sorted(FORBIDDEN_FIELDS.intersection(str(key).lower() for key in candidate))
    if leaked:
        raise ValueError(f"label_or_prediction_field_forbidden:{leaked[0]}")
    session_id = str(candidate.get("session_id") or "")
    policy_id = str(candidate.get("policy_id") or "")
    start_index = candidate.get("start_index")
    if not session_id or not policy_id:
        raise ValueError("candidate_identity_missing")
    if start_index != 0:
        raise ValueError("first_frame_compatible_selector_requires_start_index_zero")
    stream = candidate.get("action_stream")
    if not isinstance(stream, Mapping):
        raise ValueError("candidate_action_stream_missing")
    metrics = action_chunk_motion_metrics(stream)
    return {
        "session_id": session_id,
        "policy_id": policy_id,
        "start_index": 0,
        "action_stream": dict(stream),
        "metrics": metrics,
    }


def _sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    # Translation is the primary label-free signal because the failed canary's
    # dominant defect was a sub-millimetre first chunk.  Rotation and gripper
    # activity are deterministic tie-breakers, followed by stable identity.
    return (
        -float(metrics["translation_path_length_m"]),
        -float(metrics["rotation_path_length_rad"]),
        -float(metrics["gripper_total_variation"]),
        str(candidate["session_id"]),
        str(candidate["policy_id"]),
    )


def select_first_frame_high_motion_pair(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose an active chunk and real policy-swapped control prospectively.

    The swapped chunk must be another real policy from the selected session.
    It is independently ranked by the same action-only rule.
    """

    if len(candidates) < 2:
        raise ValueError("at_least_two_candidates_required")
    validated = [_validate_candidate(candidate) for candidate in candidates]
    recorded = sorted(validated, key=_sort_key)[0]
    swap_options = [
        candidate
        for candidate in validated
        if candidate["session_id"] == recorded["session_id"]
        and candidate["policy_id"] != recorded["policy_id"]
    ]
    if not swap_options:
        raise ValueError("selected_session_has_no_distinct_real_policy_swap")
    swapped = sorted(swap_options, key=_sort_key)[0]
    audit_rows = [
        {
            "session_id_internal_only": candidate["session_id"],
            "policy_id_internal_only": candidate["policy_id"],
            "start_index": 0,
            "metrics": candidate["metrics"],
        }
        for candidate in sorted(validated, key=lambda row: (row["session_id"], row["policy_id"]))
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_rule": (
            "consider only start_index=0; maximize translation path length; then rotation "
            "path; then gripper variation; then lexicographic session and policy identity"
        ),
        "recorded": {
            "session_id_internal_only": recorded["session_id"],
            "policy_id_internal_only": recorded["policy_id"],
            "start_index": 0,
            "action_stream": recorded["action_stream"],
            "metrics": recorded["metrics"],
        },
        "policy_swapped": {
            "session_id_internal_only": swapped["session_id"],
            "policy_id_internal_only": swapped["policy_id"],
            "start_index": 0,
            "action_stream": swapped["action_stream"],
            "metrics": swapped["metrics"],
        },
        "candidate_audit": audit_rows,
        "label_seal": {
            "metadata_opened": False,
            "outcome_labels_accessed": False,
            "generated_media_accessed": False,
            "evaluator_predictions_accessed": False,
        },
        "claim_boundary": "label_free_first_frame_canary_selection_only",
    }
    result["selection_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "SCHEMA_VERSION",
    "SELECTION_RULE_ID",
    "action_chunk_motion_metrics",
    "select_first_frame_high_motion_pair",
]
