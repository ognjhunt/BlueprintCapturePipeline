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

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import utc_now_iso
from .policy_ranking_ladder import POLICY_LADDER_VALIDATION_METHOD
from .sc3_fidelity_contracts import (
    SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    validate_trusted_ed25519_attestation,
)

WAM_CONSISTENCY_SCORE_SCHEMA_VERSION = "wam_consistency_score.v1"
WAM_ROLLOUT_SET_CONSISTENCY_SCHEMA_VERSION = "wam_rollout_set_consistency.v1"
WAM_CALIBRATION_ANCHOR_CHECK_SCHEMA_VERSION = "wam_calibration_anchor_check.v1"
WAM_SCORE_CLAIM_GATE_SCHEMA_VERSION = "wam_score_claim_gate.v1"

CALIBRATION_ANCHOR_VALIDATION_SCHEMA_VERSION = "policy_ranking_ladder_validation.v1"
CALIBRATION_ANCHOR_VALIDATION_METHOD = POLICY_LADDER_VALIDATION_METHOD
CALIBRATION_ANCHOR_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_POLICY_LADDER_VALIDATION_TRUSTED_PUBLIC_KEY_SHA256"
)
CALIBRATION_ANCHOR_EVIDENCE_BINDING_STATUS = "verified_trusted_full_binding"
CALIBRATION_ANCHOR_MAX_SOURCE_BYTES = 8 * 1024 * 1024
CALIBRATION_ANCHOR_MAX_ARTIFACT_ID_BYTES = 512


class VerifiedCalibrationAnchorCheck(dict[str, Any]):
    """In-process marker proving the strict verifier produced this check."""


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
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
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
    reference_vectors, _, reference_blockers = _extract_trajectory(reference, role="reference")
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


def _sha256_text(value: Any) -> bool:
    text = _string(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _safe_relative_artifact_id(value: Any) -> str | None:
    text = _string(value)
    if not text or "\x00" in text:
        return None
    try:
        relative = Path(text)
        encoded_parts = [os.fsencode(part) for part in relative.parts]
    except (OSError, UnicodeError, ValueError):
        return None
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or sum(len(part) for part in encoded_parts) > CALIBRATION_ANCHOR_MAX_ARTIFACT_ID_BYTES
        or any(len(part) > 255 for part in encoded_parts)
    ):
        return None
    return relative.as_posix()


def _bound_json_artifact(
    ref_value: Any,
    *,
    role: str,
    allowed_source_root: str | Path | None,
    expected_schema_version: str,
) -> tuple[dict[str, Any], List[str], Path | None]:
    ref = _mapping(ref_value)
    blockers: List[str] = []
    artifact_id = _string(ref.get("artifact_id"))
    safe_artifact_id = _safe_relative_artifact_id(artifact_id)
    digest = _string(ref.get("sha256")).lower()
    if "path" in ref:
        return {}, [f"calibration_anchor_{role}_artifact_path_forbidden"], None
    if not artifact_id:
        return {}, [f"calibration_anchor_{role}_artifact_id_missing"], None
    if safe_artifact_id is None:
        return {}, [f"calibration_anchor_{role}_artifact_id_unsafe"], None
    if allowed_source_root is None:
        return {}, ["calibration_anchor_allowed_source_root_missing"], None
    try:
        root = Path(allowed_source_root).expanduser().resolve()
        relative = Path(safe_artifact_id)
        path = (root / relative).resolve()
    except (OSError, RuntimeError, ValueError):
        return {}, [f"calibration_anchor_{role}_artifact_id_unsafe"], None
    try:
        path.relative_to(root)
    except ValueError:
        return {}, [f"calibration_anchor_{role}_artifact_outside_allowed_root"], None
    try:
        artifact_is_file = path.is_file()
    except (OSError, ValueError):
        return {}, [f"calibration_anchor_{role}_artifact_id_unsafe"], None
    if not artifact_is_file:
        blockers.append(f"calibration_anchor_{role}_artifact_missing")
    if not _sha256_text(digest):
        blockers.append(f"calibration_anchor_{role}_artifact_sha256_invalid")
    if blockers:
        return {}, blockers, path
    try:
        size = path.stat().st_size
        if size > CALIBRATION_ANCHOR_MAX_SOURCE_BYTES:
            return {}, [f"calibration_anchor_{role}_artifact_too_large"], path
        encoded = path.read_bytes()
    except OSError:
        return {}, [f"calibration_anchor_{role}_artifact_unreadable"], path
    if hashlib.sha256(encoded).hexdigest() != digest:
        return {}, [f"calibration_anchor_{role}_artifact_sha256_mismatch"], path
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, [f"calibration_anchor_{role}_artifact_json_invalid"], path
    if not isinstance(payload, Mapping):
        return {}, [f"calibration_anchor_{role}_artifact_not_object"], path
    result = dict(payload)
    if result.get("schema_version") != expected_schema_version:
        blockers.append(f"calibration_anchor_{role}_artifact_schema_invalid")
    return result, blockers, path


def _full_anchor_validation_shape_valid(
    payload: Mapping[str, Any], anchor_set: Sequence[str]
) -> bool:
    seed_counts = _mapping(payload.get("replicate_seed_count_by_policy"))
    empirical_acceptance = _mapping(payload.get("empirical_ground_truth_accepted_by_policy"))
    minimum_seed_count = payload.get("minimum_replicate_seed_count")
    return bool(
        payload.get("validation_method") == CALIBRATION_ANCHOR_VALIDATION_METHOD
        and payload.get("source_validation_recomputed") is True
        and payload.get("score_field") == "predicted_success_rate"
        and isinstance(minimum_seed_count, int)
        and not isinstance(minimum_seed_count, bool)
        and minimum_seed_count >= 3
        and all(
            isinstance(seed_counts.get(policy_id), int)
            and not isinstance(seed_counts.get(policy_id), bool)
            and int(seed_counts[policy_id]) >= minimum_seed_count
            and empirical_acceptance.get(policy_id) is True
            for policy_id in anchor_set
        )
    )


def evaluate_wam_calibration_anchors(
    anchor_validation: Mapping[str, Any] | None,
    *,
    allowed_source_root: str | Path | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Normalize a known-ordering ladder validation into an anchor check.

    The validation must be a full, trusted-authority-signed binding to actual
    hash-verified ladder and scorecard source files beneath ``allowed_source_root``.
    A loose or caller-authored ``recovered`` JSON is diagnostic input only and
    can never unlock an above-review score grade.
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
    source_bindings = _mapping(payload.get("source_artifact_bindings"))
    source_binding_blockers: List[str] = []
    if set(source_bindings) != {"ladder", "scorecard"}:
        source_binding_blockers.append("calibration_anchor_source_artifact_bindings_invalid")
    ladder_payload, ladder_blockers, ladder_path = _bound_json_artifact(
        source_bindings.get("ladder"),
        role="ladder",
        allowed_source_root=allowed_source_root,
        expected_schema_version="policy_ranking_ladder.v1",
    )
    scorecard_payload, scorecard_blockers, scorecard_path = _bound_json_artifact(
        source_bindings.get("scorecard"),
        role="scorecard",
        allowed_source_root=allowed_source_root,
        expected_schema_version="policy_ranking_scorecard.v1",
    )
    source_binding_blockers.extend(ladder_blockers)
    source_binding_blockers.extend(scorecard_blockers)
    source_paths = [str(path) for path in (ladder_path, scorecard_path) if path]
    if len(source_paths) != len(set(source_paths)):
        source_binding_blockers.append("calibration_anchor_source_artifact_path_reused")
    if ladder_payload and not (
        _sha256_text(ladder_payload.get("inner_checkpoint_sha256"))
        and ladder_payload.get("inner_command_configured") is True
        and _sha256_text(ladder_payload.get("registered_action_bounds_sha256"))
        and ladder_payload.get("expected_ranking") == anchor_set
    ):
        source_binding_blockers.append("calibration_anchor_ladder_immutable_identity_incomplete")
    scorecard_rankings = scorecard_payload.get("policy_rankings")
    if scorecard_payload:
        if not (
            isinstance(scorecard_rankings, Sequence)
            and not isinstance(scorecard_rankings, (str, bytes, bytearray))
        ):
            source_binding_blockers.append("calibration_anchor_scorecard_rankings_missing")
        else:
            scorecard_policy_ids = [
                _string(_mapping(row).get("policy_id"))
                for row in scorecard_rankings
                if isinstance(row, Mapping)
            ]
            if len(scorecard_policy_ids) != len(set(scorecard_policy_ids)) or not set(
                anchor_set
            ).issubset(scorecard_policy_ids):
                source_binding_blockers.append(
                    "calibration_anchor_scorecard_policy_identity_invalid"
                )
    attestation = _mapping(payload.get("validation_attestation"))
    if attestation.get("authority_role") != "policy_ladder_validation_authority":
        source_binding_blockers.append("calibration_anchor_validation_attestation_role_invalid")
    verification_report_ref = _mapping(attestation.get("verification_report_artifact"))
    report_payload, report_containment_blockers, report_path = _bound_json_artifact(
        verification_report_ref,
        role="validation_attestation_report",
        allowed_source_root=allowed_source_root,
        expected_schema_version="sc3_signature_verification_report.v1",
    )
    source_binding_blockers.extend(report_containment_blockers)
    source_binding_sha256 = _canonical_mapping_sha256(source_bindings)
    if (
        report_payload
        and report_payload.get("source_artifact_bindings_sha256") != source_binding_sha256
    ):
        source_binding_blockers.append(
            "calibration_anchor_validation_report_source_binding_mismatch"
        )
    signed_payload = {
        key: value for key, value in payload.items() if key != "validation_attestation"
    }
    attestation_for_verification = dict(attestation)
    attestation_for_verification["verification_report_artifact"] = {
        "path": str(report_path) if report_path else "",
        "sha256": _string(verification_report_ref.get("sha256")).lower(),
    }
    attestation_validation = validate_trusted_ed25519_attestation(
        attestation_for_verification,
        signed_payload=signed_payload,
        prefix="calibration_anchor_validation_attestation",
        trusted_public_key_sha256_env=(CALIBRATION_ANCHOR_TRUSTED_PUBLIC_KEY_SHA256_ENV),
    )
    source_binding_blockers.extend(
        _string(item) for item in attestation_validation.get("blockers", []) or [] if _string(item)
    )
    validation_authority_fingerprint = _string(attestation.get("public_key_sha256")).lower()
    executor_fingerprint = _string(os.getenv(SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV)).lower()
    if not _sha256_text(executor_fingerprint):
        source_binding_blockers.append("calibration_anchor_executor_trust_root_not_configured")
    else:
        if (
            _string(payload.get("executor_trusted_public_key_sha256")).lower()
            != executor_fingerprint
        ):
            source_binding_blockers.append(
                "calibration_anchor_executor_trust_root_binding_mismatch"
            )
        if executor_fingerprint == validation_authority_fingerprint:
            source_binding_blockers.append(
                "calibration_anchor_validation_authority_not_independent_from_executor"
            )
    full_shape_valid = _full_anchor_validation_shape_valid(payload, anchor_set)
    if not full_shape_valid:
        source_binding_blockers.append("calibration_anchor_full_validation_shape_invalid")
    evidence_binding_valid = not source_binding_blockers
    blockers.extend(source_binding_blockers)
    validation_status = _string(payload.get("status")) or None
    validation_blockers = [
        _string(item) for item in payload.get("blockers", []) or [] if _string(item)
    ]
    anchors_passed = bool(
        anchors_present
        and evidence_binding_valid
        and payload.get("ranker_ordering_recovered") is True
        and validation_status == "recovered"
        and not validation_blockers
    )
    if anchors_present and not anchors_passed:
        blockers.append("calibration_anchor_ordering_not_recovered")
    return VerifiedCalibrationAnchorCheck(
        {
            "schema_version": WAM_CALIBRATION_ANCHOR_CHECK_SCHEMA_VERSION,
            "generated_at": _string(generated_at) or utc_now_iso(),
            "anchors_present": anchors_present,
            "anchors_passed": anchors_passed,
            "evidence_binding_status": (
                CALIBRATION_ANCHOR_EVIDENCE_BINDING_STATUS
                if evidence_binding_valid
                else "blocked_unverified_or_tampered"
            ),
            "source_artifact_bindings": {
                role: {
                    "artifact_id": _safe_relative_artifact_id(
                        _mapping(source_bindings.get(role)).get("artifact_id")
                    ),
                    "sha256": _string(_mapping(source_bindings.get(role)).get("sha256")).lower()
                    or None,
                }
                for role in ("ladder", "scorecard")
            },
            "validation_attestation_public_key_sha256": _string(
                attestation.get("public_key_sha256")
            ).lower()
            or None,
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
    )


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
        anchors.get("anchors_present") is True
        and anchors.get("anchors_passed") is True
        and payload.get("calibration_anchor_verifier_executed") is True
        and anchors.get("evidence_binding_status") == CALIBRATION_ANCHOR_EVIDENCE_BINDING_STATUS
    )
    if payload.get("consistency_measured_and_passed") is True and not consistency_ok:
        blockers.append("wam_consistency_claim_flag_without_passing_nested_evidence")
    if payload.get("calibration_anchors_present_and_passed") is True and not anchors_ok:
        blockers.append("wam_calibration_claim_flag_without_passing_nested_evidence")
    if grade not in WAM_SCORE_CLAIM_GRADES:
        blockers.append("wam_score_claim_grade_unrecognized")
        grade = "fixture_evaluator_only"
    elif _grade_rank(grade) > _grade_rank("review_grade") and not (consistency_ok and anchors_ok):
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
            | {_string(item) for item in payload.get("blockers", []) or [] if _string(item)}
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
    anchors_verified_in_process = isinstance(calibration_anchors, VerifiedCalibrationAnchorCheck)
    consistency_payload = _mapping(consistency)
    anchors_payload = _mapping(calibration_anchors)
    blockers: List[str] = []

    consistency_ok = bool(
        consistency_payload.get("status") == "scored" and consistency_payload.get("passed") is True
    )
    anchors_ok = bool(
        anchors_payload.get("anchors_present") is True
        and anchors_payload.get("anchors_passed") is True
        and anchors_verified_in_process
        and anchors_payload.get("evidence_binding_status")
        == CALIBRATION_ANCHOR_EVIDENCE_BINDING_STATUS
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
        "calibration_anchor_verifier_executed": anchors_verified_in_process,
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
            "evidence_binding_status": anchors_payload.get("evidence_binding_status"),
            "validation_attestation_public_key_sha256": anchors_payload.get(
                "validation_attestation_public_key_sha256"
            ),
            "source_artifact_bindings": _mapping(anchors_payload.get("source_artifact_bindings")),
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
