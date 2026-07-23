"""Model-neutral RoboWorld-inspired evaluator and admission contracts.

This module deliberately implements the evaluator-side lessons from RoboWorld
without implementing or claiming to reproduce Step Forcing.  It provides:

* a versioned 0--5 task-progress/world-model-failure rubric;
* criterion-scoped authority for fixed, wrist, or future camera views;
* five explicit segment aggregation strategies and an ablation report;
* a blinded multi-judge versus human calibration campaign; and
* a frozen upstream admission/reproduction checklist.

All outputs are support/evaluation artifacts.  They do not prove physical task
success, world-model rank fidelity, or a RoboWorld reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmark_protocol import external_rank_metrics
from .common import read_json_any, write_json


PROFILE_SCHEMA_VERSION = "roboworld_progress_evaluator_profile.v1"
PROGRESS_SCORE_SCHEMA_VERSION = "roboworld_progress_score.v1"
SEGMENT_ABLATION_SCHEMA_VERSION = "segment_aggregation_ablation.v1"
JUDGE_CALIBRATION_REQUEST_SCHEMA_VERSION = "judge_calibration_campaign_request.v1"
JUDGE_CALIBRATION_REPORT_SCHEMA_VERSION = "judge_calibration_campaign_report.v1"
ADMISSION_EVIDENCE_SCHEMA_VERSION = "roboworld_admission_evidence.v1"
ADMISSION_CHECKLIST_SCHEMA_VERSION = "roboworld_admission_reproduction_checklist.v1"

PROGRESS_STAGES = (
    "no_task_directed_behavior",
    "approach",
    "target_contact",
    "target_interaction",
    "near_completion",
    "completed",
)
WORLD_MODEL_FAILURE_STAGES = (
    "none",
    "before_approach",
    "during_approach",
    "upon_contact",
    "during_interaction",
    "after_completion",
)
VIEW_ROLES = (
    "task_progress",
    "task_completion",
    "world_model_failure_detection",
)
AGGREGATION_STRATEGIES = (
    "progress_then_regression_aware",
    "terminal",
    "mean",
    "minimum",
    "maximum_experimental",
    "stable_maintenance",
)
DEFAULT_AGGREGATION_STRATEGY = "terminal"
_SHA256_PATTERN = "sha256:"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(row, Mapping) for row in value):
        return [], False
    return [dict(row) for row in value], True


def _string(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, *, minimum: int = 0, maximum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def _digest(value: Any) -> str:
    text = _string(value).lower()
    bare = text.removeprefix(_SHA256_PATTERN)
    return bare if len(bare) == 64 and all(char in "0123456789abcdef" for char in bare) else ""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_default_progress_profile() -> dict[str, Any]:
    """Build the frozen model-neutral progress rubric and view-authority profile."""

    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": "roboworld_progress_v1",
        "profile_version": "1.0.0",
        "frozen": True,
        "source_method": {
            "name": "RoboWorld-inspired task-progress evaluation",
            "paper": "arXiv:2607.01060v4",
            "paper_metrics_are_not_blueprint_measurements": True,
            "step_forcing_implemented": False,
            "reported_evaluator_ablation": {
                "task_progress_spearman": 0.970,
                "binary_success_spearman": 0.922,
                "wrist_as_success_authority_spearman": 0.862,
                "values_are_external_context_only": True,
            },
        },
        "rubric": [
            {
                "score": 0,
                "policy_progress_stage": "no_task_directed_behavior",
                "allowed_world_model_failure_stages": ["none", "before_approach"],
                "world_model_failure_required": False,
                "meaning": "failure_or_irrelevant_behavior",
            },
            {
                "score": 1,
                "policy_progress_stage": "approach",
                "world_model_failure_stage": "during_approach",
                "allowed_world_model_failure_stages": ["during_approach"],
                "world_model_failure_required": True,
                "meaning": "model_failure_during_task_directed_approach",
            },
            {
                "score": 2,
                "policy_progress_stage": "approach",
                "allowed_world_model_failure_stages": ["none"],
                "world_model_failure_required": False,
                "meaning": "task_directed_approach_without_target_interaction",
            },
            {
                "score": 3,
                "policy_progress_stage": "target_contact",
                "world_model_failure_stage": "upon_contact",
                "allowed_world_model_failure_stages": ["upon_contact"],
                "world_model_failure_required": True,
                "meaning": "model_failure_at_or_immediately_after_target_contact",
            },
            {
                "score": 4,
                "policy_progress_stage": "near_completion",
                "allowed_world_model_failure_stages": ["none", "during_interaction"],
                "world_model_failure_required": False,
                "meaning": "substantial_valid_progress_or_model_failure_during_interaction",
            },
            {
                "score": 5,
                "policy_progress_stage": "completed",
                "allowed_world_model_failure_stages": ["none"],
                "world_model_failure_required": False,
                "meaning": "task_goal_visibly_achieved_and_stably_maintained",
            },
        ],
        "view_authority": {
            "views": [
                {
                    "view_id": "fixed_external_left",
                    "allowed_roles": ["task_progress", "task_completion"],
                },
                {
                    "view_id": "fixed_external_right",
                    "allowed_roles": ["task_progress", "task_completion"],
                },
                {
                    "view_id": "wrist",
                    "allowed_roles": ["world_model_failure_detection"],
                },
            ],
            "criterion_overrides": [],
            "override_requirements": {
                "task_specific": True,
                "independently_accepted": True,
                "calibration_set_sha256_required": True,
                "reason_required": True,
            },
        },
        "segment_aggregation": {
            "sample_rate_hz": 1.0,
            "default_strategy": DEFAULT_AGGREGATION_STRATEGY,
            "maximum_is_experimental": True,
            "stable_maintenance_adjacent_frame_count": 2,
            "strategies": list(AGGREGATION_STRATEGIES),
            "selection_requires_ablation": True,
        },
        "required_score_fields": [
            "task_progress_score",
            "policy_progress_stage",
            "world_model_failure_stage",
            "world_model_failure_detected",
            "criterion_evidence_refs",
            "judge_confidence",
            "judge_abstained",
            "prompt_sha256",
            "judge_model_sha256",
            "calibration_set_sha256",
        ],
        "claim_boundary": {
            "generated_video_score_is_not_physical_robot_success": True,
            "progress_score_is_not_rank_fidelity_measurement": True,
            "view_authority_is_evaluator_configuration_not_sensor_truth": True,
            "step_forcing_or_roboworld_model_reproduced": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    profile["profile_sha256"] = canonical_sha256(profile)
    return profile


def validate_progress_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        blockers.append("progress_profile_schema_missing_or_unsupported")
    if not _string(profile.get("profile_id")):
        blockers.append("progress_profile_id_missing")
    if profile.get("frozen") is not True:
        blockers.append("progress_profile_must_be_frozen")
    rubric, rubric_valid = _strict_rows(profile.get("rubric"))
    scores = [row.get("score") for row in rubric]
    if not rubric_valid or scores != list(range(6)):
        blockers.append("progress_profile_rubric_must_cover_ordered_scores_0_through_5")
    for index, row in enumerate(rubric):
        if row.get("policy_progress_stage") not in PROGRESS_STAGES:
            blockers.append(f"progress_profile_stage_invalid:{index}")
        failure_stage = row.get("world_model_failure_stage")
        if failure_stage is not None and failure_stage not in WORLD_MODEL_FAILURE_STAGES:
            blockers.append(f"progress_profile_failure_stage_invalid:{index}")
        if not _string(row.get("meaning")):
            blockers.append(f"progress_profile_meaning_missing:{index}")
        allowed_failure_stages = row.get("allowed_world_model_failure_stages")
        if (
            not isinstance(allowed_failure_stages, list)
            or not allowed_failure_stages
            or any(stage not in WORLD_MODEL_FAILURE_STAGES for stage in allowed_failure_stages)
        ):
            blockers.append(f"progress_profile_allowed_failure_stages_invalid:{index}")
        if not isinstance(row.get("world_model_failure_required"), bool):
            blockers.append(f"progress_profile_failure_requirement_invalid:{index}")
    authority = _mapping(profile.get("view_authority"))
    views, views_valid = _strict_rows(authority.get("views"))
    if not views_valid or not views:
        blockers.append("progress_profile_views_missing_or_invalid")
    view_ids: list[str] = []
    for index, view in enumerate(views):
        view_id = _string(view.get("view_id"))
        roles = view.get("allowed_roles")
        view_ids.append(view_id)
        if not view_id:
            blockers.append(f"progress_profile_view_id_missing:{index}")
        if (
            not isinstance(roles, list)
            or not roles
            or any(role not in VIEW_ROLES for role in roles)
        ):
            blockers.append(f"progress_profile_view_roles_invalid:{index}")
    if len(view_ids) != len(set(view_ids)):
        blockers.append("progress_profile_view_ids_duplicate")
    overrides, overrides_valid = _strict_rows(authority.get("criterion_overrides"))
    if not overrides_valid:
        blockers.append("progress_profile_criterion_overrides_invalid")
    for index, override in enumerate(overrides):
        if not _string(override.get("criterion_id")):
            blockers.append(f"progress_profile_override_criterion_missing:{index}")
        if _string(override.get("view_id")) not in set(view_ids):
            blockers.append(f"progress_profile_override_view_unknown:{index}")
        roles = override.get("allowed_roles")
        if not isinstance(roles, list) or any(role not in VIEW_ROLES for role in roles):
            blockers.append(f"progress_profile_override_roles_invalid:{index}")
        if override.get("independently_accepted") is not True:
            blockers.append(f"progress_profile_override_not_accepted:{index}")
        if not _digest(override.get("calibration_set_sha256")):
            blockers.append(f"progress_profile_override_calibration_digest_missing:{index}")
        if not _string(override.get("reason")):
            blockers.append(f"progress_profile_override_reason_missing:{index}")
    aggregation = _mapping(profile.get("segment_aggregation"))
    if aggregation.get("default_strategy") == "maximum_experimental":
        blockers.append("maximum_segment_aggregation_cannot_be_default")
    if aggregation.get("default_strategy") not in AGGREGATION_STRATEGIES:
        blockers.append("segment_aggregation_default_strategy_invalid")
    if aggregation.get("maximum_is_experimental") is not True:
        blockers.append("maximum_segment_aggregation_must_remain_experimental")
    if aggregation.get("selection_requires_ablation") is not True:
        blockers.append("segment_aggregation_selection_must_require_ablation")
    expected_digest = _digest(profile.get("profile_sha256"))
    unsigned = dict(profile)
    unsigned.pop("profile_sha256", None)
    if expected_digest != canonical_sha256(unsigned):
        blockers.append("progress_profile_digest_mismatch")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "roboworld_progress_evaluator_profile_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "profile_id": _string(profile.get("profile_id")) or None,
        "profile_sha256": expected_digest or None,
        "blockers": blockers,
    }


def _view_roles_for_criterion(
    profile: Mapping[str, Any], *, criterion_id: str
) -> dict[str, set[str]]:
    authority = _mapping(profile.get("view_authority"))
    roles = {
        _string(view.get("view_id")): {
            _string(role) for role in view.get("allowed_roles", []) if _string(role)
        }
        for view in _rows(authority.get("views"))
    }
    for override in _rows(authority.get("criterion_overrides")):
        if (
            _string(override.get("criterion_id")) == criterion_id
            and override.get("independently_accepted") is True
            and _digest(override.get("calibration_set_sha256"))
        ):
            roles[_string(override.get("view_id"))] = {
                _string(role)
                for role in override.get("allowed_roles", [])
                if _string(role)
            }
    return roles


def validate_progress_score(
    score: Mapping[str, Any], *, profile: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one score artifact against rubric and criterion view authority."""

    selected_profile = dict(profile or build_default_progress_profile())
    blockers = list(validate_progress_profile(selected_profile)["blockers"])
    if score.get("schema_version") != PROGRESS_SCORE_SCHEMA_VERSION:
        blockers.append("progress_score_schema_missing_or_unsupported")
    if _string(score.get("profile_id")) != _string(selected_profile.get("profile_id")):
        blockers.append("progress_score_profile_id_mismatch")
    if _digest(score.get("profile_sha256")) != _digest(selected_profile.get("profile_sha256")):
        blockers.append("progress_score_profile_digest_mismatch")
    criterion_id = _string(score.get("criterion_id"))
    if not criterion_id:
        blockers.append("progress_score_criterion_id_missing")
    task_score = _integer(score.get("task_progress_score"), minimum=0, maximum=5)
    if task_score is None:
        blockers.append("task_progress_score_missing_or_out_of_range")
    policy_stage = _string(score.get("policy_progress_stage"))
    if policy_stage not in PROGRESS_STAGES:
        blockers.append("policy_progress_stage_missing_or_invalid")
    failure_detected = score.get("world_model_failure_detected")
    failure_stage = _string(score.get("world_model_failure_stage"))
    if not isinstance(failure_detected, bool):
        blockers.append("world_model_failure_detected_must_be_boolean")
    if failure_stage not in WORLD_MODEL_FAILURE_STAGES:
        blockers.append("world_model_failure_stage_missing_or_invalid")
    if failure_detected is True and failure_stage == "none":
        blockers.append("world_model_failure_detected_requires_failure_stage")
    if failure_detected is False and failure_stage != "none":
        blockers.append("world_model_failure_stage_requires_detected_failure")
    if task_score is not None:
        rubric_row = next(
            (
                row
                for row in _rows(selected_profile.get("rubric"))
                if row.get("score") == task_score
            ),
            {},
        )
        if policy_stage != _string(rubric_row.get("policy_progress_stage")):
            blockers.append("task_progress_score_policy_stage_mismatch")
        allowed_failure_stages = {
            _string(value)
            for value in rubric_row.get("allowed_world_model_failure_stages", [])
            if _string(value)
        }
        if failure_stage not in allowed_failure_stages:
            blockers.append("task_progress_score_failure_stage_not_allowed")
        if rubric_row.get("world_model_failure_required") is True and failure_detected is not True:
            blockers.append("task_progress_score_failure_stage_mismatch")
    evidence_refs = score.get("criterion_evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs or any(
        not _string(item) for item in evidence_refs
    ):
        blockers.append("criterion_evidence_refs_missing_or_invalid")
    confidence = _number(score.get("judge_confidence"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        blockers.append("judge_confidence_missing_or_out_of_range")
    abstained = score.get("judge_abstained")
    if not isinstance(abstained, bool):
        blockers.append("judge_abstained_must_be_boolean")
    if abstained is True and not _string(score.get("abstention_reason")):
        blockers.append("judge_abstention_reason_missing")
    for field in ("prompt_sha256", "judge_model_sha256", "calibration_set_sha256"):
        if not _digest(score.get(field)):
            blockers.append(f"progress_score_digest_missing_or_invalid:{field}")
    view_evidence, view_payload_valid = _strict_rows(score.get("view_evidence"))
    if not view_payload_valid or not view_evidence:
        blockers.append("progress_score_view_evidence_missing_or_invalid")
    allowed_roles = _view_roles_for_criterion(selected_profile, criterion_id=criterion_id)
    used_progress_authority = False
    used_completion_authority = False
    used_failure_authority = False
    for index, row in enumerate(view_evidence):
        view_id = _string(row.get("view_id"))
        roles = row.get("roles_used")
        if view_id not in allowed_roles:
            blockers.append(f"progress_score_unknown_view:{index}")
            continue
        if not isinstance(roles, list) or not roles:
            blockers.append(f"progress_score_view_roles_missing:{index}")
            continue
        for role in roles:
            if role not in VIEW_ROLES:
                blockers.append(f"progress_score_view_role_invalid:{index}")
            elif role not in allowed_roles[view_id]:
                blockers.append(f"progress_score_view_role_unauthorized:{view_id}:{role}")
            else:
                used_progress_authority |= role == "task_progress"
                used_completion_authority |= role == "task_completion"
                used_failure_authority |= role == "world_model_failure_detection"
        refs = row.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(not _string(ref) for ref in refs):
            blockers.append(f"progress_score_view_evidence_refs_missing:{index}")
    if task_score is not None and task_score > 0 and not used_progress_authority:
        blockers.append("task_progress_score_requires_authorized_progress_view")
    if task_score == 5 and not used_completion_authority:
        blockers.append("task_completion_requires_authorized_completion_view")
    if failure_detected is True and not used_failure_authority:
        blockers.append("world_model_failure_requires_authorized_failure_detection_view")
    frame_scores = score.get("sampled_frame_scores")
    if not isinstance(frame_scores, list) or any(
        _integer(value, minimum=0, maximum=5) is None for value in frame_scores
    ):
        blockers.append("sampled_frame_scores_missing_or_invalid")
    blockers = sorted(set(blockers))
    normalized = {
        "schema_version": PROGRESS_SCORE_SCHEMA_VERSION,
        "profile_id": _string(selected_profile.get("profile_id")),
        "profile_sha256": _digest(selected_profile.get("profile_sha256")),
        "rollout_id": _string(score.get("rollout_id")) or None,
        "segment_index": score.get("segment_index"),
        "criterion_id": criterion_id or None,
        "task_progress_score": task_score,
        "policy_progress_stage": policy_stage or None,
        "world_model_failure_stage": failure_stage or None,
        "world_model_failure_detected": failure_detected
        if isinstance(failure_detected, bool)
        else None,
        "criterion_evidence_refs": list(evidence_refs)
        if isinstance(evidence_refs, list)
        else [],
        "judge_confidence": confidence,
        "judge_abstained": abstained if isinstance(abstained, bool) else None,
        "abstention_reason": _string(score.get("abstention_reason")) or None,
        "prompt_sha256": _digest(score.get("prompt_sha256")) or None,
        "judge_model_sha256": _digest(score.get("judge_model_sha256")) or None,
        "calibration_set_sha256": _digest(score.get("calibration_set_sha256")) or None,
        "view_evidence": view_evidence,
        "sampled_frame_scores": list(frame_scores) if isinstance(frame_scores, list) else [],
        "status": (
            "blocked"
            if blockers
            else "abstained"
            if abstained is True
            else "validated"
        ),
        "blockers": blockers,
        "claim_boundary": {
            "score_is_model_or_human_review_of_generated_media": True,
            "score_is_not_physical_task_success": True,
            "score_does_not_prove_rank_fidelity": True,
        },
    }
    normalized["score_sha256"] = canonical_sha256(normalized)
    return normalized


def aggregate_segment_scores(
    scores: Sequence[Mapping[str, Any]], *, profile: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Compute every declared strategy without silently selecting max(segment)."""

    selected_profile = dict(profile or build_default_progress_profile())
    validations = [validate_progress_score(score, profile=selected_profile) for score in scores]
    blockers = [
        f"segment_{index}:{blocker}"
        for index, validation in enumerate(validations)
        for blocker in validation["blockers"]
    ]
    blockers.extend(
        f"segment_{index}:judge_abstained"
        for index, validation in enumerate(validations)
        if validation.get("status") == "abstained"
    )
    indices = [validation.get("segment_index") for validation in validations]
    if any(_integer(index, minimum=0) is None for index in indices):
        blockers.append("segment_indices_missing_or_invalid")
    if len(indices) != len(set(indices)):
        blockers.append("segment_indices_duplicate")
    ordered = sorted(validations, key=lambda row: int(row.get("segment_index") or 0))
    values = [
        int(row["task_progress_score"])
        for row in ordered
        if isinstance(row.get("task_progress_score"), int)
        and row.get("status") == "validated"
    ]
    if len(values) != len(ordered) or not values:
        blockers.append("segment_scores_unavailable")
    frame_values = [
        int(value)
        for row in ordered
        for value in row.get("sampled_frame_scores", [])
        if isinstance(value, int)
    ]
    stable_count = int(
        _mapping(selected_profile.get("segment_aggregation")).get(
            "stable_maintenance_adjacent_frame_count", 2
        )
    )
    stable_success = bool(
        stable_count >= 2
        and len(frame_values) >= stable_count
        and all(value == 5 for value in frame_values[-stable_count:])
    )
    if not frame_values:
        blockers.append("stable_maintenance_requires_sampled_frame_scores")
    aggregations: dict[str, float | None] = {strategy: None for strategy in AGGREGATION_STRATEGIES}
    if values:
        terminal = float(values[-1])
        cumulative_regression = sum(
            max(0.0, float(previous - current))
            for previous, current in zip(values, values[1:])
        )
        aggregations.update(
            {
                "terminal": terminal,
                "mean": sum(values) / len(values),
                "minimum": float(min(values)),
                "maximum_experimental": float(max(values)),
                "progress_then_regression_aware": max(0.0, terminal - cumulative_regression),
                "stable_maintenance": 5.0 if stable_success else min(4.0, terminal),
            }
        )
    default_strategy = _string(
        _mapping(selected_profile.get("segment_aggregation")).get("default_strategy")
    )
    if default_strategy == "maximum_experimental":
        blockers.append("maximum_segment_aggregation_cannot_be_default")
    blockers = sorted(set(blockers))
    selected_value = aggregations.get(default_strategy)
    result = {
        "schema_version": "segment_aggregation_result.v1",
        "status": "complete" if not blockers else "blocked",
        "profile_id": _string(selected_profile.get("profile_id")),
        "profile_sha256": _digest(selected_profile.get("profile_sha256")),
        "rollout_id": ordered[0].get("rollout_id") if ordered else None,
        "segment_count": len(ordered),
        "ordered_segment_scores": values,
        "sampled_frame_scores": frame_values,
        "stable_maintenance_adjacent_frame_count": stable_count,
        "stable_success": stable_success,
        "aggregations": {
            key: round(value, 6) if value is not None else None
            for key, value in aggregations.items()
        },
        "default_strategy": default_strategy,
        "selected_score": (
            round(float(selected_value), 6) if selected_value is not None else None
        ),
        "maximum_is_experimental": True,
        "maximum_selected_as_default": default_strategy == "maximum_experimental",
        "blockers": blockers,
        "claim_boundary": {
            "aggregation_ablation_required_before_strategy_promotion": True,
            "maximum_segment_score_is_not_default": default_strategy != "maximum_experimental",
            "stable_generated_media_completion_is_not_physical_success": True,
        },
    }
    result["aggregation_result_sha256"] = canonical_sha256(result)
    return result


def build_segment_aggregation_ablation(
    rollout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare segment strategies against independently accepted reference scores."""

    blockers: list[str] = []
    strategy_policy_values: dict[str, dict[str, list[float]]] = {
        strategy: defaultdict(list) for strategy in AGGREGATION_STRATEGIES
    }
    policy_reference_values: dict[str, list[float]] = defaultdict(list)
    rollout_results: list[dict[str, Any]] = []
    for index, row in enumerate(rollout_rows):
        policy_id = _string(row.get("policy_id"))
        reference = _number(row.get("reference_score"))
        if not policy_id:
            blockers.append(f"ablation_policy_id_missing:{index}")
            continue
        if reference is None:
            blockers.append(f"ablation_reference_score_missing:{index}")
            continue
        if row.get("reference_independently_accepted") is not True:
            blockers.append(f"ablation_reference_not_independently_accepted:{index}")
            continue
        aggregation = aggregate_segment_scores(_rows(row.get("segment_scores")))
        rollout_results.append(aggregation)
        if aggregation["status"] != "complete":
            blockers.extend(f"ablation_rollout_invalid:{index}:{item}" for item in aggregation["blockers"])
            continue
        policy_reference_values[policy_id].append(reference)
        for strategy, value in aggregation["aggregations"].items():
            if value is not None:
                strategy_policy_values[strategy][policy_id].append(float(value))
    policy_ids = sorted(policy_reference_values)
    if len(policy_ids) < 3:
        blockers.append("segment_ablation_requires_at_least_three_policies")
    reference_by_policy = {
        policy_id: sum(values) / len(values)
        for policy_id, values in policy_reference_values.items()
    }
    strategies: dict[str, Any] = {}
    for strategy in AGGREGATION_STRATEGIES:
        predicted_by_policy = {
            policy_id: sum(values) / len(values)
            for policy_id, values in strategy_policy_values[strategy].items()
        }
        matched = [policy_id for policy_id in policy_ids if policy_id in predicted_by_policy]
        metrics = (
            external_rank_metrics(
                [predicted_by_policy[policy_id] for policy_id in matched],
                [reference_by_policy[policy_id] for policy_id in matched],
            )
            if len(matched) >= 3
            else {metric: None for metric in external_rank_metrics([], [])}
        )
        strategies[strategy] = {
            "experimental": strategy == "maximum_experimental",
            "policy_count": len(matched),
            "policy_scores": [
                {
                    "policy_id": policy_id,
                    "aggregated_score": round(predicted_by_policy[policy_id], 6),
                    "reference_score": round(reference_by_policy[policy_id], 6),
                }
                for policy_id in matched
            ],
            "metrics": {
                key: round(value, 6) if value is not None else None
                for key, value in metrics.items()
            },
        }
    blockers = sorted(set(blockers))
    report = {
        "schema_version": SEGMENT_ABLATION_SCHEMA_VERSION,
        "status": "measured" if not blockers else "blocked",
        "default_strategy": DEFAULT_AGGREGATION_STRATEGY,
        "maximum_is_experimental": True,
        "rollout_count": len(rollout_results),
        "policy_count": len(policy_ids),
        "strategies": strategies,
        "strategy_promotion_decision": "not_automatically_selected",
        "blockers": blockers,
        "claim_boundary": {
            "ablation_is_dataset_specific": True,
            "maximum_strategy_not_promoted_by_implementation": True,
            "reference_agreement_is_not_physical_robot_readiness": True,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _confidence_bin(confidence: float) -> int:
    return min(9, int(confidence * 10.0))


def _bias_breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in ("task_id", "view_condition", "contact_stage", "artifact_type"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_string(row.get(axis)) or "unspecified"].append(row)
        result[axis] = [
            {
                "value": value,
                "sample_count": len(group),
                "mean_signed_error": round(
                    sum(float(item["error"]) for item in group) / len(group), 6
                ),
                "mean_absolute_error": round(
                    sum(abs(float(item["error"])) for item in group) / len(group), 6
                ),
                "false_success_rate": round(
                    sum(bool(item["false_success"]) for item in group) / len(group), 6
                ),
                "abstention_rate": round(
                    sum(bool(item["abstained"]) for item in group) / len(group), 6
                ),
            }
            for value, group in sorted(grouped.items())
        ]
    return result


def run_judge_calibration_campaign(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run a deterministic blinded VLM-versus-human calibration analysis."""

    blockers: list[str] = []
    if request.get("schema_version") != JUDGE_CALIBRATION_REQUEST_SCHEMA_VERSION:
        blockers.append("judge_campaign_schema_missing_or_unsupported")
    if request.get("frozen") is not True:
        blockers.append("judge_campaign_must_be_frozen")
    for field in ("campaign_id", "campaign_version"):
        if not _string(request.get(field)):
            blockers.append(f"judge_campaign_identity_missing:{field}")
    if not _digest(request.get("sample_manifest_sha256")):
        blockers.append("judge_campaign_sample_manifest_digest_missing")
    samples, samples_valid = _strict_rows(request.get("samples"))
    if not samples_valid or not samples:
        blockers.append("judge_campaign_samples_missing_or_invalid")
    by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    judge_families: dict[str, str] = {}
    human_scores_by_policy: dict[str, list[float]] = defaultdict(list)
    sample_ids: list[str] = []
    for sample_index, sample in enumerate(samples):
        sample_id = _string(sample.get("sample_id"))
        policy_id = _string(sample.get("policy_id"))
        sample_ids.append(sample_id)
        if not sample_id or not policy_id:
            blockers.append(f"judge_campaign_sample_identity_missing:{sample_index}")
        human = _mapping(sample.get("human_reference"))
        human_score = _integer(human.get("score"), minimum=0, maximum=5)
        if human_score is None:
            blockers.append(f"judge_campaign_human_score_invalid:{sample_index}")
            continue
        if _integer(human.get("reviewer_count"), minimum=2) is None:
            blockers.append(f"judge_campaign_human_reviewer_count_insufficient:{sample_index}")
        if human.get("blinded_to_policy_identity") is not True:
            blockers.append(f"judge_campaign_human_review_not_blinded:{sample_index}")
        if human.get("randomized_order") is not True:
            blockers.append(f"judge_campaign_human_review_not_randomized:{sample_index}")
        if not _digest(human.get("label_artifact_sha256")):
            blockers.append(f"judge_campaign_human_label_digest_missing:{sample_index}")
        human_scores_by_policy[policy_id].append(float(human_score))
        outputs, outputs_valid = _strict_rows(sample.get("judge_outputs"))
        if not outputs_valid or not outputs:
            blockers.append(f"judge_campaign_outputs_missing_or_invalid:{sample_index}")
            continue
        seen_judges: set[str] = set()
        for output_index, output in enumerate(outputs):
            judge_id = _string(output.get("judge_id"))
            family = _string(output.get("judge_family"))
            score = _integer(output.get("score"), minimum=0, maximum=5)
            confidence = _number(output.get("confidence"))
            abstained = output.get("abstained")
            if not judge_id or not family or judge_id in seen_judges:
                blockers.append(
                    f"judge_campaign_judge_identity_missing_or_duplicate:{sample_index}:{output_index}"
                )
                continue
            seen_judges.add(judge_id)
            judge_families[judge_id] = family
            if score is None:
                blockers.append(f"judge_campaign_score_invalid:{sample_index}:{output_index}")
                continue
            if confidence is None or not 0.0 <= confidence <= 1.0:
                blockers.append(
                    f"judge_campaign_confidence_invalid:{sample_index}:{output_index}"
                )
                continue
            if not isinstance(abstained, bool):
                blockers.append(f"judge_campaign_abstention_invalid:{sample_index}:{output_index}")
                continue
            if output.get("blinded_to_policy_identity") is not True:
                blockers.append(f"judge_campaign_output_not_blinded:{sample_index}:{output_index}")
            if output.get("randomized_order") is not True:
                blockers.append(f"judge_campaign_output_not_randomized:{sample_index}:{output_index}")
            for field in ("prompt_sha256", "judge_model_sha256", "calibration_set_sha256"):
                if not _digest(output.get(field)):
                    blockers.append(
                        f"judge_campaign_digest_missing:{sample_index}:{output_index}:{field}"
                    )
            error = float(score - human_score)
            by_judge[judge_id].append(
                {
                    "sample_id": sample_id,
                    "policy_id": policy_id,
                    "human_score": human_score,
                    "judge_score": score,
                    "confidence": confidence,
                    "abstained": abstained,
                    "error": error,
                    "exact": score == human_score,
                    "within_one": abs(error) <= 1.0,
                    "false_success": score == 5 and human_score < 5,
                    "task_id": _string(sample.get("task_id")),
                    "view_condition": _string(sample.get("view_condition")),
                    "contact_stage": _string(sample.get("contact_stage")),
                    "artifact_type": _string(sample.get("artifact_type")),
                }
            )
    if len(sample_ids) != len(set(sample_ids)):
        blockers.append("judge_campaign_sample_ids_duplicate")
    families = {family.lower() for family in judge_families.values()}
    if not any("gpt" in family or "openai" in family for family in families):
        blockers.append("judge_campaign_gpt_family_missing")
    if not any("gemini" in family or "google" in family for family in families):
        blockers.append("judge_campaign_independent_gemini_family_missing")
    if len(human_scores_by_policy) < 3:
        blockers.append("judge_campaign_rank_stability_requires_three_policies")
    judge_reports: list[dict[str, Any]] = []
    human_policy_scores = {
        policy_id: sum(values) / len(values)
        for policy_id, values in human_scores_by_policy.items()
    }
    for judge_id, rows in sorted(by_judge.items()):
        non_abstained = [row for row in rows if not row["abstained"]]
        confusion = {
            str(human): {str(predicted): 0 for predicted in range(6)} for human in range(6)
        }
        for row in non_abstained:
            confusion[str(row["human_score"])][str(row["judge_score"])] += 1
        bins: list[dict[str, Any]] = []
        for bin_index in range(10):
            bin_rows = [
                row
                for row in non_abstained
                if _confidence_bin(float(row["confidence"])) == bin_index
            ]
            if not bin_rows:
                continue
            bins.append(
                {
                    "lower": bin_index / 10.0,
                    "upper": (bin_index + 1) / 10.0,
                    "sample_count": len(bin_rows),
                    "mean_confidence": round(
                        sum(float(row["confidence"]) for row in bin_rows) / len(bin_rows), 6
                    ),
                    "exact_match_rate": round(
                        sum(bool(row["exact"]) for row in bin_rows) / len(bin_rows), 6
                    ),
                }
            )
        ece = (
            sum(
                abs(bin_row["mean_confidence"] - bin_row["exact_match_rate"])
                * bin_row["sample_count"]
                for bin_row in bins
            )
            / len(non_abstained)
            if non_abstained
            else None
        )
        predicted_by_policy: dict[str, list[float]] = defaultdict(list)
        for row in non_abstained:
            predicted_by_policy[_string(row["policy_id"])].append(float(row["judge_score"]))
        matched_policies = sorted(set(predicted_by_policy) & set(human_policy_scores))
        metrics = (
            external_rank_metrics(
                [sum(predicted_by_policy[item]) / len(predicted_by_policy[item]) for item in matched_policies],
                [human_policy_scores[item] for item in matched_policies],
            )
            if len(matched_policies) >= 3
            else {metric: None for metric in external_rank_metrics([], [])}
        )
        judge_reports.append(
            {
                "judge_id": judge_id,
                "judge_family": judge_families[judge_id],
                "sample_count": len(rows),
                "non_abstained_count": len(non_abstained),
                "abstention_rate": round(
                    sum(bool(row["abstained"]) for row in rows) / len(rows), 6
                )
                if rows
                else None,
                "mean_signed_error": round(
                    sum(float(row["error"]) for row in non_abstained) / len(non_abstained), 6
                )
                if non_abstained
                else None,
                "mean_absolute_error": round(
                    sum(abs(float(row["error"])) for row in non_abstained)
                    / len(non_abstained),
                    6,
                )
                if non_abstained
                else None,
                "exact_agreement": round(
                    sum(bool(row["exact"]) for row in non_abstained) / len(non_abstained), 6
                )
                if non_abstained
                else None,
                "within_one_agreement": round(
                    sum(bool(row["within_one"]) for row in non_abstained)
                    / len(non_abstained),
                    6,
                )
                if non_abstained
                else None,
                "false_success_rate": round(
                    sum(bool(row["false_success"]) for row in non_abstained)
                    / len(non_abstained),
                    6,
                )
                if non_abstained
                else None,
                "success_brier_score": round(
                    sum(
                        (
                            (
                                float(row["confidence"])
                                if row["judge_score"] == 5
                                else 1.0 - float(row["confidence"])
                            )
                            - float(row["human_score"] == 5)
                        )
                        ** 2
                        for row in non_abstained
                    )
                    / max(1, len(non_abstained)),
                    6,
                ),
                "expected_calibration_error_exact_match": round(ece, 6)
                if ece is not None
                else None,
                "confidence_bins": bins,
                "confusion_matrix": confusion,
                "policy_rank_stability": {
                    key: round(value, 6) if value is not None else None
                    for key, value in metrics.items()
                },
                "bias_breakdowns": _bias_breakdowns(rows),
            }
        )
    blockers = sorted(set(blockers))
    report = {
        "schema_version": JUDGE_CALIBRATION_REPORT_SCHEMA_VERSION,
        "campaign_id": _string(request.get("campaign_id")) or None,
        "campaign_version": _string(request.get("campaign_version")) or None,
        "status": "measured" if not blockers else "blocked",
        "frozen": request.get("frozen") is True,
        "sample_manifest_sha256": _digest(request.get("sample_manifest_sha256")) or None,
        "sample_count": len(samples),
        "policy_count": len(human_scores_by_policy),
        "human_reference": {
            "minimum_reviewers_per_sample": 2,
            "blinded": True,
            "randomized": True,
        },
        "judges": judge_reports,
        "blockers": blockers,
        "claim_boundary": {
            "human_labels_are_campaign_reference_not_physical_ground_truth": True,
            "judge_calibration_is_not_world_model_rank_fidelity": True,
            "judge_provider_is_not_evaluator_truth_authority": True,
            "public_claim_upgrade_allowed": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def build_roboworld_admission_checklist(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate frozen prerequisites for a future licensed RoboWorld integration."""

    blockers: list[str] = []
    if evidence.get("schema_version") != ADMISSION_EVIDENCE_SCHEMA_VERSION:
        blockers.append("roboworld_admission_evidence_schema_missing_or_unsupported")
    expected_paper = "arXiv:2607.01060v4"
    if _string(evidence.get("paper_version")) != expected_paper:
        blockers.append("roboworld_paper_version_not_frozen_to_v4")
    release = _mapping(evidence.get("upstream_release"))
    code_released = release.get("code_released") is True
    weights_released = release.get("weights_released") is True
    required_release_fields = (
        "source_uri",
        "source_revision",
        "software_license",
        "software_license_sha256",
        "checkpoint_uri",
        "checkpoint_sha256",
        "weights_license",
        "weights_license_sha256",
        "container_image_digest",
        "preprocessing_manifest_sha256",
        "data_filter_manifest_sha256",
        "action_normalization_manifest_sha256",
        "training_schedule_manifest_sha256",
        "evaluation_script_sha256",
    )
    if not code_released:
        blockers.append("roboworld_upstream_code_not_released")
    if not weights_released:
        blockers.append("roboworld_upstream_weights_not_released")
    for field in required_release_fields:
        value = release.get(field)
        if field.endswith("sha256"):
            if not _digest(value):
                blockers.append(f"roboworld_upstream_release_digest_missing:{field}")
        elif not _string(value):
            blockers.append(f"roboworld_upstream_release_field_missing:{field}")
    image = _string(release.get("container_image_digest"))
    if image and "@sha256:" not in image:
        blockers.append("roboworld_container_image_not_digest_pinned")
    diagnostic = _mapping(evidence.get("diagnostic_reproduction"))
    diagnostic_required = (
        "bair_protocol_sha256",
        "step_forcing_checkpoint_sha256",
        "metrics_artifact_sha256",
    )
    if diagnostic.get("executed") is not True:
        blockers.append("roboworld_bair_diagnostic_not_reproduced")
    for field in diagnostic_required:
        if not _digest(diagnostic.get(field)):
            blockers.append(f"roboworld_diagnostic_digest_missing:{field}")
    published = _mapping(evidence.get("published_result_reproduction"))
    if published.get("executed") is not True:
        blockers.append("roboworld_roboarena_result_not_reproduced")
    for field in (
        "roboarena_snapshot_sha256",
        "policy_registry_sha256",
        "initial_condition_manifest_sha256",
        "rollout_results_sha256",
        "judge_prompt_sha256",
        "rank_report_sha256",
    ):
        if not _digest(published.get(field)):
            blockers.append(f"roboworld_published_reproduction_digest_missing:{field}")
    if _integer(published.get("policy_count"), minimum=8) is None:
        blockers.append("roboworld_published_reproduction_requires_eight_policies")
    if _integer(published.get("rollout_count"), minimum=4186) is None:
        blockers.append("roboworld_published_reproduction_rollout_count_insufficient")
    comparison = _mapping(evidence.get("blueprint_comparison"))
    if comparison.get("executed") is not True:
        blockers.append("roboworld_blueprint_identical_matrix_comparison_not_executed")
    for field in (
        "frozen_benchmark_spec_sha256",
        "environment_manifest_sha256",
        "policy_registry_sha256",
        "current_wam_results_sha256",
        "physics_sim_results_sha256",
        "roboworld_results_sha256",
        "external_anchor_results_sha256",
        "comparison_report_sha256",
    ):
        if not _digest(comparison.get(field)):
            blockers.append(f"roboworld_blueprint_comparison_digest_missing:{field}")
    blockers = sorted(set(blockers))
    if not code_released or not weights_released:
        status = "awaiting_upstream_release"
    elif blockers:
        status = "blocked_reproduction_incomplete"
    else:
        status = "admitted_for_configured_backend_evaluation"
    checklist = {
        "schema_version": ADMISSION_CHECKLIST_SCHEMA_VERSION,
        "checklist_id": "roboworld_step_forcing_admission_v1",
        "checklist_version": "1.0.0",
        "frozen": True,
        "paper_version": expected_paper,
        "status": status,
        "upstream_release_gate": {
            "passed": code_released
            and weights_released
            and not any("upstream_release" in item or "container_image" in item for item in blockers),
            "code_released": code_released,
            "weights_released": weights_released,
            "required_fields": list(required_release_fields),
        },
        "diagnostic_reproduction_gate": {
            "passed": diagnostic.get("executed") is True
            and not any("diagnostic" in item for item in blockers),
            "target": "BAIR action-conditioned Step Forcing diagnostic",
        },
        "published_result_reproduction_gate": {
            "passed": published.get("executed") is True
            and not any("published_reproduction" in item or "roboarena_result" in item for item in blockers),
            "targets": {
                "policy_count": 8,
                "minimum_rollout_count": 4186,
                "paper_pearson": 0.989,
                "paper_spearman": 0.970,
                "paper_values_are_not_acceptance_substitutes": True,
            },
        },
        "blueprint_identical_matrix_gate": {
            "passed": comparison.get("executed") is True
            and not any("blueprint_comparison" in item for item in blockers),
            "required_backends": [
                "current_configured_wam",
                "physics_simulation",
                "roboworld_candidate",
                "external_real_anchor",
            ],
        },
        "deferred_work": {
            "step_forcing_training_reimplementation_authorized": False,
            "paper_only_model_backend_integration_authorized": False,
            "reason": "wait_for_licensed_code_weights_and_reproducible_runtime_artifacts",
        },
        "blockers": blockers,
        "claim_boundary": {
            "checklist_completion_is_not_model_quality_proof": True,
            "paper_metrics_are_not_blueprint_metrics": True,
            "reproduction_is_not_physical_robot_readiness": True,
            "public_claim_upgrade_allowed": False,
        },
    }
    checklist["checklist_sha256"] = canonical_sha256(checklist)
    return checklist


def normalize_wam_progress_label(
    command_payload: Mapping[str, Any],
    label: Mapping[str, Any],
    success_value: bool | None,
) -> dict[str, Any]:
    """Validate optional progress evidence and return normalized label fields."""

    configured_profile = _mapping(command_payload.get("progress_evaluator_profile")) or None
    progress_source = label.get("progress_evaluation")
    requested = (
        _string(label.get("evaluation_profile_id")) == "roboworld_progress_v1"
        or isinstance(progress_source, Mapping)
    )
    evaluation = (
        validate_progress_score(progress_source, profile=configured_profile)
        if isinstance(progress_source, Mapping)
        else None
    )
    blockers = [
        f"wam_progress_evaluation:{blocker}"
        for blocker in _rows_or_strings(evaluation.get("blockers") if evaluation else [])
    ]
    if evaluation and evaluation.get("status") == "validated":
        progress_score = evaluation.get("task_progress_score")
        if progress_score == 5 and success_value is not True:
            blockers.append("wam_progress_score_success_verdict_mismatch")
        if isinstance(progress_score, int) and progress_score < 5 and success_value is True:
            blockers.append("wam_progress_score_noncompletion_verdict_mismatch")
    elif requested and evaluation is None:
        blockers.append("wam_progress_evaluation_missing_for_requested_profile")
    return {
        "blockers": blockers,
        "label_fields": {
            "evaluation_profile_id": "roboworld_progress_v1" if requested else None,
            "progress_evaluation": evaluation,
            "task_progress_score": evaluation.get("task_progress_score") if evaluation else None,
            "policy_progress_stage": evaluation.get("policy_progress_stage") if evaluation else None,
            "world_model_failure_stage": (
                evaluation.get("world_model_failure_stage") if evaluation else None
            ),
            "world_model_failure_detected": (
                evaluation.get("world_model_failure_detected") if evaluation else None
            ),
        },
    }


def wam_progress_label_counts(labels: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Summarize requested and validated progress-profile label coverage."""

    return {
        "progress_profile_label_count": sum(
            row.get("evaluation_profile_id") == "roboworld_progress_v1" for row in labels
        ),
        "validated_progress_label_count": sum(
            _mapping(row.get("progress_evaluation")).get("status") == "validated"
            for row in labels
        ),
    }


def _rows_or_strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [_string(item) for item in value if _string(item)]


def _load_mapping(path: str | Path) -> dict[str, Any]:
    value = read_json_any(Path(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--output", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument("--output", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", required=True)
    aggregate_parser.add_argument("--output", required=True)
    ablation_parser = subparsers.add_parser("ablate-segments")
    ablation_parser.add_argument("--input", required=True)
    ablation_parser.add_argument("--output", required=True)
    calibration_parser = subparsers.add_parser("calibrate-judges")
    calibration_parser.add_argument("--input", required=True)
    calibration_parser.add_argument("--output", required=True)
    admission_parser = subparsers.add_parser("admission")
    admission_parser.add_argument("--input", required=True)
    admission_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "profile":
        result = build_default_progress_profile()
    else:
        payload = _load_mapping(args.input)
        if args.command == "score":
            result = validate_progress_score(payload)
        elif args.command == "aggregate":
            result = aggregate_segment_scores(_rows(payload.get("segment_scores")))
        elif args.command == "ablate-segments":
            result = build_segment_aggregation_ablation(_rows(payload.get("rollouts")))
        elif args.command == "calibrate-judges":
            result = run_judge_calibration_campaign(payload)
        else:
            result = build_roboworld_admission_checklist(payload)
    write_json(Path(args.output), result)
    return (
        0
        if result.get("status")
        not in {"blocked", "blocked_reproduction_incomplete", "awaiting_upstream_release"}
        else 2
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
