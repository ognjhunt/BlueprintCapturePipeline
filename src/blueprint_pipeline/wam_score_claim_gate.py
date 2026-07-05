"""Fail-closed claim gate for WAM evaluator scores.

Doctrine (WORLD_MODEL_STRATEGY_CONTEXT.md): a learned-WAM eval score is only
trustworthy above review grade when a consistency scorer and calibration
anchors exist and pass. This module provides the three pieces that enforce it:

- ``score_wam_consistency(rollout, reference)`` — deterministic temporal +
  geometric consistency of a generated rollout trajectory against the
  conditioning reference trace. No GPU, no model call; missing or degenerate
  trajectories block instead of scoring.
- ``evaluate_wam_calibration_anchors(...)`` — reads a
  ``policy_ranking_ladder_validation.v1`` payload (the known-ordering ladder
  from ``policy_ranking_ladder``) and reports whether calibration anchors are
  present and the ranker recovered their ordering.
- ``apply_wam_score_claim_gate(...)`` — a WAM score claim above
  ``review_grade`` requires both to be present and passing; an above-review
  claim without them fails closed to ``fixture_evaluator_only`` with the
  ``wam_score_without_consistency_or_calibration`` blocker. A review-grade
  claim without them is capped (not demoted).

The gate payload always carries the anchor set and the consistency number so
downstream surfaces never show a bare score.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from .common import utc_now_iso

WAM_CONSISTENCY_SCORE_SCHEMA_VERSION = "wam_consistency_score.v1"
WAM_ROLLOUT_SET_CONSISTENCY_SCHEMA_VERSION = "wam_rollout_set_consistency.v1"
WAM_CALIBRATION_ANCHOR_CHECK_SCHEMA_VERSION = "wam_calibration_anchor_check.v1"
WAM_SCORE_CLAIM_GATE_SCHEMA_VERSION = "wam_score_claim_gate.v1"

CALIBRATION_ANCHOR_VALIDATION_SCHEMA_VERSION = "policy_ranking_ladder_validation.v1"

# Ordered weakest -> strongest. Anything above review_grade requires a passing
# consistency score AND passing calibration anchors.
WAM_SCORE_CLAIM_GRADES: tuple[str, ...] = (
    "fixture_evaluator_only",
    "review_grade",
    "calibrated_evaluator_grade",
)

WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER = (
    "wam_score_without_consistency_or_calibration"
)

DEFAULT_MIN_CONSISTENCY_SCORE = 0.8

_TRAJECTORY_KEYS = ("trajectory", "steps", "frames", "predicted_states")
_TIMESTAMP_KEYS = ("timestamp", "t", "time", "frame_time")
_POSITION_KEYS = ("position", "xyz", "waypoint", "translation", "state", "qpos")


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _consistency_claim_boundary() -> Dict[str, Any]:
    return {
        "consistency_score_is_support_signal_not_task_success": True,
        "consistency_score_does_not_prove_rank_fidelity": True,
        "consistency_score_does_not_prove_deployment_readiness": True,
        "consistency_input_is_generated_rollout_not_physical_robot": True,
    }


def _steps_from(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for key in _TRAJECTORY_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)
            ):
                return [row for row in candidate if isinstance(row, Mapping)]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [row for row in value if isinstance(row, Mapping)]
    return []


def _step_vector(step: Mapping[str, Any]) -> list[float] | None:
    for key in _POSITION_KEYS:
        candidate = step.get(key)
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            values = [_finite_float(item) for item in candidate]
            if values and all(item is not None for item in values):
                return [float(item) for item in values]  # type: ignore[arg-type]
            if values:
                return None  # present but non-finite: caller must fail closed
    return None


def _step_has_vector_payload(step: Mapping[str, Any]) -> bool:
    return any(
        isinstance(step.get(key), Sequence)
        and not isinstance(step.get(key), (str, bytes, bytearray))
        for key in _POSITION_KEYS
    )


def _step_timestamp(step: Mapping[str, Any]) -> float | None:
    for key in _TIMESTAMP_KEYS:
        value = _finite_float(step.get(key))
        if value is not None:
            return value
    return None


def _extract_trajectory(
    value: Any, *, role: str
) -> tuple[list[list[float]], list[float | None], List[str]]:
    blockers: List[str] = []
    steps = _steps_from(value)
    vectors: list[list[float]] = []
    timestamps: list[float | None] = []
    for step in steps:
        vector = _step_vector(step)
        if vector is None:
            if _step_has_vector_payload(step):
                blockers.append("non_finite_trajectory_values")
            continue
        vectors.append(vector)
        timestamps.append(_step_timestamp(step))
    if len(vectors) < 2:
        blockers.append(f"{role}_trajectory_missing_or_degenerate")
    return vectors, timestamps, sorted(set(blockers))


def _temporal_consistency(timestamps: Sequence[float | None]) -> tuple[float | None, List[str]]:
    finite = [value for value in timestamps if value is not None]
    if len(finite) < 2 or len(finite) != len(timestamps):
        return None, ["rollout_timestamps_missing"]
    gaps = [finite[index + 1] - finite[index] for index in range(len(finite) - 1)]
    positive_gaps = sorted(gap for gap in gaps if gap > 0.0)
    if not positive_gaps:
        return 0.0, []
    median_gap = positive_gaps[len(positive_gaps) // 2]
    violations = sum(1 for gap in gaps if gap <= 0.0 or gap > 3.0 * median_gap)
    return max(0.0, 1.0 - violations / len(gaps)), []


def _geometric_consistency(
    rollout: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]
) -> tuple[float | None, int, List[str]]:
    compared = min(len(rollout), len(reference))
    if compared < 2:
        return None, compared, ["insufficient_comparable_trajectory_steps"]
    dims = {len(vector) for vector in list(rollout)[:compared]} | {
        len(vector) for vector in list(reference)[:compared]
    }
    if len(dims) != 1:
        return None, compared, ["trajectory_dimension_mismatch"]
    squared_errors = [
        sum((r - e) ** 2 for r, e in zip(rollout[index], reference[index]))
        for index in range(compared)
    ]
    rmse = math.sqrt(sum(squared_errors) / compared)
    lows = [min(vector[axis] for vector in reference) for axis in range(len(reference[0]))]
    highs = [max(vector[axis] for vector in reference) for axis in range(len(reference[0]))]
    span = math.sqrt(sum((high - low) ** 2 for low, high in zip(lows, highs)))
    scale = max(span, 1e-6)
    return max(0.0, min(1.0, 1.0 - rmse / scale)), compared, []


def score_wam_consistency(
    rollout: Any,
    reference: Any,
    *,
    min_passing_score: float = DEFAULT_MIN_CONSISTENCY_SCORE,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Score temporal + geometric consistency of a rollout against a reference.

    Fails closed: missing, degenerate, non-finite, or dimension-mismatched
    trajectories produce a blocked result with ``consistency_score=None`` —
    never a passing score.
    """
    rollout_vectors, rollout_timestamps, rollout_blockers = _extract_trajectory(
        rollout, role="rollout"
    )
    reference_vectors, _, reference_blockers = _extract_trajectory(
        reference, role="reference"
    )
    blockers = sorted(set(rollout_blockers) | set(reference_blockers))

    temporal: float | None = None
    geometric: float | None = None
    compared = 0
    if not blockers:
        temporal, temporal_blockers = _temporal_consistency(rollout_timestamps)
        geometric, compared, geometric_blockers = _geometric_consistency(
            rollout_vectors, reference_vectors
        )
        blockers = sorted(set(temporal_blockers) | set(geometric_blockers))

    scored = not blockers and temporal is not None and geometric is not None
    score = round(min(temporal, geometric), 6) if scored else None
    return {
        "schema_version": WAM_CONSISTENCY_SCORE_SCHEMA_VERSION,
        "generated_at": _string(generated_at) or utc_now_iso(),
        "status": "scored" if scored else "blocked",
        "consistency_score": score,
        "temporal_consistency": round(temporal, 6) if scored else None,
        "geometric_consistency": round(geometric, 6) if scored else None,
        "compared_step_count": compared,
        "min_passing_score": float(min_passing_score),
        "passed": bool(scored and score is not None and score >= float(min_passing_score)),
        "blockers": blockers,
        "claim_boundary": _consistency_claim_boundary(),
    }


def score_wam_rollout_set_consistency(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    reference: Any,
    min_passing_score: float = DEFAULT_MIN_CONSISTENCY_SCORE,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Score every rollout against the reference; the set takes the worst score.

    A set with no scoreable rollout is blocked, never silently passing.
    """
    generated = _string(generated_at) or utc_now_iso()
    rollout_scores: List[Dict[str, Any]] = []
    for index, rollout in enumerate(rollouts):
        row = _mapping(rollout)
        scored = score_wam_consistency(
            row,
            reference,
            min_passing_score=min_passing_score,
            generated_at=generated,
        )
        rollout_scores.append(
            {
                "rollout_id": _string(row.get("rollout_id")) or f"rollout_{index + 1:04d}",
                "status": scored["status"],
                "consistency_score": scored["consistency_score"],
                "temporal_consistency": scored["temporal_consistency"],
                "geometric_consistency": scored["geometric_consistency"],
                "compared_step_count": scored["compared_step_count"],
                "passed": scored["passed"],
                "blockers": scored["blockers"],
            }
        )
    scored_rows = [row for row in rollout_scores if row["status"] == "scored"]
    blockers: List[str] = []
    if not rollouts:
        blockers.append("no_rollouts_available_for_consistency_scoring")
    elif not scored_rows:
        blockers.append("no_rollout_carries_scoreable_trajectory")
    if scored_rows and len(scored_rows) < len(rollout_scores):
        blockers.append("some_rollouts_not_scoreable_for_consistency")
    worst = min((row["consistency_score"] for row in scored_rows), default=None)
    status = "scored" if scored_rows else "blocked"
    return {
        "schema_version": WAM_ROLLOUT_SET_CONSISTENCY_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "consistency_score": worst,
        "aggregation": "minimum_over_scored_rollouts",
        "rollout_count": len(rollout_scores),
        "scored_rollout_count": len(scored_rows),
        "min_passing_score": float(min_passing_score),
        "passed": bool(
            scored_rows
            and all(row["passed"] for row in scored_rows)
            and len(scored_rows) == len(rollout_scores)
        ),
        "rollout_scores": rollout_scores,
        "blockers": blockers,
        "claim_boundary": _consistency_claim_boundary(),
    }


def evaluate_wam_calibration_anchors(
    anchor_validation: Mapping[str, Any] | None,
    *,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Normalize a known-ordering ladder validation into an anchor check.

    Anchors count as present only when the payload is a recognized
    ``policy_ranking_ladder_validation.v1`` document with at least two provable
    rungs; they pass only when the ranker recovered the expected ordering with
    no validation blockers.
    """
    payload = _mapping(anchor_validation)
    blockers: List[str] = []
    anchor_set = [
        _string(item) for item in payload.get("expected_ranking", []) or [] if _string(item)
    ]
    if not payload:
        blockers.append("calibration_anchor_validation_missing")
    elif _string(payload.get("schema_version")) != CALIBRATION_ANCHOR_VALIDATION_SCHEMA_VERSION:
        blockers.append("calibration_anchor_validation_schema_unrecognized")
    elif len(anchor_set) < 2:
        blockers.append("calibration_anchor_set_too_small")
    anchors_present = not blockers
    validation_status = _string(payload.get("status")) or None
    validation_blockers = [
        _string(item) for item in payload.get("blockers", []) or [] if _string(item)
    ]
    anchors_passed = bool(
        anchors_present
        and payload.get("ranker_ordering_recovered") is True
        and validation_status in {"recovered", "recovered_with_ties"}
        and not validation_blockers
    )
    if anchors_present and not anchors_passed:
        blockers.append("calibration_anchor_ordering_not_recovered")
    return {
        "schema_version": WAM_CALIBRATION_ANCHOR_CHECK_SCHEMA_VERSION,
        "generated_at": _string(generated_at) or utc_now_iso(),
        "anchors_present": anchors_present,
        "anchors_passed": anchors_passed,
        "anchor_set": anchor_set if anchors_present else [],
        "anchor_validation_status": validation_status,
        "anchor_validation_blockers": validation_blockers,
        "spearman_rank_correlation_vs_expected": payload.get(
            "spearman_rank_correlation_vs_expected"
        ),
        "anchor_basis": _string(payload.get("expected_ranking_basis")) or None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "anchors_validate_evaluator_ordering_sensitivity_only": True,
            "recovered_ordering_is_not_rank_fidelity_vs_real_world": True,
        },
    }


def _grade_rank(grade: str) -> int:
    return WAM_SCORE_CLAIM_GRADES.index(grade)


def summarize_wam_evaluation_for_report(
    gate_payload: Mapping[str, Any] | None,
) -> tuple[Dict[str, Any] | None, List[str]]:
    """Turn a gate payload into a buyer-report section, re-checking fail-closed.

    Returns ``(section, blockers)``. The section never carries a bare score:
    the grade always travels with the consistency number and the anchor set. A
    payload claiming above ``review_grade`` without passing consistency and
    anchors — or an unrecognized grade — is demoted to
    ``fixture_evaluator_only`` here as well, so a hand-edited artifact cannot
    smuggle an overclaim past the report.
    """
    if gate_payload is None:
        return None, []
    payload = _mapping(gate_payload)
    consistency = _mapping(payload.get("consistency"))
    anchors = _mapping(payload.get("calibration_anchors"))
    blockers: List[str] = []

    grade = _string(payload.get("granted_grade"))
    consistency_score = consistency.get("consistency_score")
    consistency_ok = bool(
        consistency.get("status") == "scored"
        and consistency.get("passed") is True
        and consistency_score is not None
    )
    anchors_ok = bool(
        anchors.get("anchors_present") is True and anchors.get("anchors_passed") is True
    )
    if (
        payload.get("consistency_measured_and_passed") is True
        and not consistency_ok
    ):
        blockers.append("wam_consistency_claim_flag_without_passing_nested_evidence")
    if (
        payload.get("calibration_anchors_present_and_passed") is True
        and not anchors_ok
    ):
        blockers.append("wam_calibration_claim_flag_without_passing_nested_evidence")
    if grade not in WAM_SCORE_CLAIM_GRADES:
        blockers.append("wam_score_claim_grade_unrecognized")
        grade = "fixture_evaluator_only"
    elif _grade_rank(grade) > _grade_rank("review_grade") and not (
        consistency_ok and anchors_ok
    ):
        blockers.append(WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER)
        grade = "fixture_evaluator_only"

    section = {
        "wam_score_claim_grade": grade,
        "grade_ladder": list(WAM_SCORE_CLAIM_GRADES),
        "consistency_score": consistency_score,
        "consistency_status": _string(consistency.get("status")) or "missing",
        "consistency_passed": consistency.get("passed") is True,
        "calibration_anchor_set": [
            _string(item) for item in anchors.get("anchor_set", []) or [] if _string(item)
        ],
        "calibration_anchors_passed": anchors.get("anchors_passed") is True,
        "calibration_anchor_validation_status": anchors.get("anchor_validation_status"),
        "spearman_rank_correlation_vs_expected": anchors.get(
            "spearman_rank_correlation_vs_expected"
        ),
        "bare_score_forbidden": True,
        "blockers": sorted(
            set(blockers)
            | {
                _string(item)
                for item in payload.get("blockers", []) or []
                if _string(item)
            }
        ),
        "claim_boundary": {
            "score_above_review_grade_requires_consistency_and_calibration_anchors": True,
            "grade_is_evaluator_bounded_not_rank_fidelity": True,
            "rank_fidelity_result_proven": False,
        },
    }
    return section, [f"wam_evaluation:{blocker}" for blocker in sorted(set(blockers))]


def apply_wam_score_claim_gate(
    *,
    requested_grade: str,
    consistency: Mapping[str, Any] | None,
    calibration_anchors: Mapping[str, Any] | None,
    fixture_evaluator_only: bool = False,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Gate a WAM score claim grade on consistency + calibration anchors.

    Above-review claims without both passing fail closed to
    ``fixture_evaluator_only``; review-or-below claims are capped, never
    penalized. The returned payload always carries the anchor set and the
    consistency number so no caller can surface a bare score.
    """
    consistency_payload = _mapping(consistency)
    anchors_payload = _mapping(calibration_anchors)
    blockers: List[str] = []

    consistency_ok = bool(
        consistency_payload.get("status") == "scored"
        and consistency_payload.get("passed") is True
    )
    anchors_ok = bool(
        anchors_payload.get("anchors_present") is True
        and anchors_payload.get("anchors_passed") is True
    )
    both_ok = consistency_ok and anchors_ok

    upgrade_requirements: List[str] = []
    if not consistency_ok:
        upgrade_requirements.append("passing_wam_consistency_score_required")
    if not anchors_ok:
        upgrade_requirements.append("passing_calibration_anchor_ordering_required")

    if fixture_evaluator_only:
        max_allowed = "fixture_evaluator_only"
    elif both_ok:
        max_allowed = "calibrated_evaluator_grade"
    else:
        max_allowed = "review_grade"

    requested = _string(requested_grade)
    if requested not in WAM_SCORE_CLAIM_GRADES:
        blockers.append("wam_score_claim_grade_unrecognized")
        granted = "fixture_evaluator_only"
        status = "failed_closed"
    elif _grade_rank(requested) <= _grade_rank(max_allowed):
        granted = requested
        status = "granted"
    elif _grade_rank(requested) > _grade_rank("review_grade"):
        # Attempted above-review claim without consistency + anchors.
        blockers.append(WAM_SCORE_WITHOUT_CONSISTENCY_OR_CALIBRATION_BLOCKER)
        granted = "fixture_evaluator_only"
        status = "failed_closed"
    else:
        granted = max_allowed
        status = "capped"
        if requested == "review_grade" and max_allowed == "fixture_evaluator_only":
            blockers.append("fixture_evaluator_only_run_cannot_claim_review_grade")

    return {
        "schema_version": WAM_SCORE_CLAIM_GATE_SCHEMA_VERSION,
        "generated_at": _string(generated_at) or utc_now_iso(),
        "status": status,
        "requested_grade": requested or None,
        "granted_grade": granted,
        "max_allowed_grade": max_allowed,
        "grade_ladder": list(WAM_SCORE_CLAIM_GRADES),
        "fixture_evaluator_only": bool(fixture_evaluator_only),
        "consistency_measured_and_passed": consistency_ok,
        "calibration_anchors_present_and_passed": anchors_ok,
        "upgrade_requirements": upgrade_requirements if not both_ok else [],
        "consistency": {
            "status": _string(consistency_payload.get("status")) or "missing",
            "consistency_score": consistency_payload.get("consistency_score"),
            "temporal_consistency": consistency_payload.get("temporal_consistency"),
            "geometric_consistency": consistency_payload.get("geometric_consistency"),
            "passed": consistency_payload.get("passed") is True,
            "blockers": [
                _string(item)
                for item in consistency_payload.get("blockers", []) or []
                if _string(item)
            ],
        },
        "calibration_anchors": {
            "anchors_present": anchors_payload.get("anchors_present") is True,
            "anchors_passed": anchors_payload.get("anchors_passed") is True,
            "anchor_set": [
                _string(item)
                for item in anchors_payload.get("anchor_set", []) or []
                if _string(item)
            ],
            "anchor_validation_status": anchors_payload.get("anchor_validation_status"),
            "spearman_rank_correlation_vs_expected": anchors_payload.get(
                "spearman_rank_correlation_vs_expected"
            ),
        },
        "blockers": blockers,
        "claim_boundary": {
            "score_above_review_grade_requires_consistency_and_calibration_anchors": True,
            "bare_wam_score_reporting_forbidden": True,
            "granted_grade_is_evaluator_bounded_not_rank_fidelity": True,
            "consistency_and_anchors_do_not_prove_real_world_outcome": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
