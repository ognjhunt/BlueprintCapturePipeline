"""Strict numeric action-recovery validation for WAM consistency scorers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


def _string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _string(value).lower()
    if text in {"true", "yes", "pass", "passed", "consistent"}:
        return True
    if text in {"false", "no", "fail", "failed", "inconsistent"}:
        return False
    return None


def confidence_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return max(0.0, min(1.0, float(_string(value))))
    except ValueError:
        return None


def strict_action_consistency_blockers(
    check: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[str]:
    """Validate numeric action recovery; boolean-only labels never satisfy the contract."""
    blockers: list[str] = []
    expected_sha = _string(expected.get("commanded_action_sha256")).lower()
    observed_sha = _string(check.get("commanded_action_sha256")).lower()
    if len(expected_sha) != 64 or observed_sha != expected_sha:
        blockers.append("wam_consistency_commanded_action_sha256_mismatch")
    try:
        expected_dim = int(expected.get("action_dimension"))
    except (TypeError, ValueError):
        expected_dim = 0

    def finite_vector(value: Any, length: int) -> list[float] | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return None
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if len(result) != length or not all(math.isfinite(item) for item in result):
            return None
        return result

    commanded_values = finite_vector(expected.get("commanded_action_vector"), expected_dim)
    recovered_values = finite_vector(check.get("recovered_action"), expected_dim)
    error_values = finite_vector(check.get("per_dimension_error"), expected_dim)
    uncertainty_values = finite_vector(check.get("per_dimension_uncertainty"), expected_dim)
    if expected_dim <= 0 or commanded_values is None:
        blockers.append("wam_consistency_expected_action_vector_invalid")
    expected_units = expected.get("action_units")
    observed_units = check.get("action_units")
    observed_units_valid = isinstance(observed_units, Sequence) and not isinstance(
        observed_units, (str, bytes, bytearray)
    )
    if (
        not isinstance(expected_units, Sequence)
        or isinstance(expected_units, (str, bytes, bytearray))
        or len(expected_units) != expected_dim
        or not all(_string(item) for item in expected_units)
        or not observed_units_valid
        or list(observed_units) != list(expected_units)
    ):
        blockers.append("wam_consistency_action_units_missing_or_mismatch")
    if recovered_values is None:
        blockers.append("wam_consistency_recovered_action_missing_wrong_dim_or_nonfinite")
    if error_values is None:
        blockers.append("wam_consistency_per_dimension_error_missing_wrong_dim_or_nonfinite")
    if uncertainty_values is None or any(value < 0 for value in uncertainty_values or []):
        blockers.append("wam_consistency_uncertainty_missing_wrong_dim_or_nonfinite")
    if commanded_values is not None and recovered_values is not None and error_values is not None:
        recomputed = [abs(a - b) for a, b in zip(commanded_values, recovered_values, strict=True)]
        if any(abs(actual - reported) > 1e-6 for actual, reported in zip(recomputed, error_values, strict=True)):
            blockers.append("wam_consistency_per_dimension_error_mismatch")
    recovered_sha = _string(check.get("recovered_action_sha256")).lower()
    if recovered_values is None or recovered_sha != hashlib.sha256(
        json.dumps(recovered_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        blockers.append("wam_consistency_recovered_action_sha256_mismatch")
    threshold = _mapping(check.get("threshold"))
    threshold_value = confidence_or_none(threshold.get("max_abs_error"))
    if threshold_value is None:
        blockers.append("wam_consistency_numeric_threshold_missing_or_invalid")
    elif error_values is not None and max(error_values, default=0.0) > threshold_value:
        blockers.append("wam_consistency_numeric_threshold_exceeded")
    expected_unit = _string(expected.get("action_unit"))
    if not expected_unit or _string(threshold.get("unit")) != expected_unit:
        blockers.append("wam_consistency_action_unit_missing_or_mismatch")
    calibration = _mapping(check.get("calibration_identity"))
    calibration_sha = _string(calibration.get("sha256")).lower()
    if not _string(calibration.get("calibration_id")) or len(calibration_sha) != 64 or any(
        character not in "0123456789abcdef" for character in calibration_sha
    ):
        blockers.append("wam_consistency_calibration_identity_missing_or_invalid")
    for field in (
        "controller_fk_state_sha256",
        "generated_state_sha256",
        "generated_motion_sha256",
    ):
        expected_digest = _string(expected.get(field)).lower()
        observed_digest = _string(check.get(field)).lower()
        if (
            len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
            or observed_digest != expected_digest
        ):
            blockers.append(f"wam_consistency_{field}_missing_or_mismatch")
    if not _string(check.get("scorer_runtime_id")):
        blockers.append("wam_consistency_scorer_runtime_id_missing")
    if check.get("provider_output_replay_used") is not False:
        blockers.append("wam_consistency_replay_or_replay_status_missing")
    for direction in ("forward", "inverse"):
        result = _mapping(check.get(f"{direction}_result"))
        if (
            not isinstance(result.get("passed"), bool)
            or result.get("passed") is not check.get(f"{direction}_consistent")
            or not _string(result.get("method"))
        ):
            blockers.append(f"wam_consistency_{direction}_result_missing_or_invalid")
    timing = _mapping(check.get("action_timing"))
    expected_timing = _mapping(expected.get("action_timing"))
    try:
        observed_sim_time = float(timing.get("sim_time_s"))
        expected_sim_time = float(expected_timing.get("sim_time_s"))
        timing_numeric_match = math.isfinite(observed_sim_time) and math.isfinite(
            expected_sim_time
        ) and abs(observed_sim_time - expected_sim_time) <= 1e-9
    except (TypeError, ValueError):
        timing_numeric_match = False
    if timing.get("step_index") != expected_timing.get("step_index") or not timing_numeric_match or _string(timing.get("unit")) != "s":
        blockers.append("wam_consistency_action_timing_missing_or_invalid")
    for field in ("control_hz", "sample_period_seconds"):
        try:
            observed = float(timing.get(field))
            expected_value = float(expected_timing.get(field))
            matches = math.isfinite(observed) and math.isfinite(expected_value) and abs(
                observed - expected_value
            ) <= 1e-9
        except (TypeError, ValueError):
            matches = False
        if not matches:
            blockers.append("wam_consistency_action_timing_missing_or_invalid")
    termination = _mapping(check.get("termination_chunk"))
    if (
        termination.get("step_index") != expected_timing.get("step_index")
        or _string(termination.get("commanded_action_sha256")).lower() != expected_sha
        or _string(termination.get("generated_motion_sha256")).lower()
        != _string(expected.get("generated_motion_sha256")).lower()
    ):
        blockers.append("wam_consistency_termination_chunk_missing_or_invalid")
    evidence_refs = check.get("evidence_refs")
    if not isinstance(evidence_refs, Sequence) or isinstance(evidence_refs, (str, bytes)) or not evidence_refs:
        blockers.append("wam_consistency_evidence_refs_missing")
    return blockers


def cross_step_action_motion_replay_blockers(
    checks: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject one generated motion or recovered action reused for different commands."""
    motion_to_action: dict[str, str] = {}
    recovered_to_action: dict[str, str] = {}
    blockers: list[str] = []
    for index, check in enumerate(checks):
        action_sha = _string(check.get("commanded_action_sha256")).lower()
        for field, seen, blocker in (
            (
                "generated_motion_sha256",
                motion_to_action,
                "wam_consistency_generated_motion_reused_for_different_action",
            ),
            (
                "recovered_action_sha256",
                recovered_to_action,
                "wam_consistency_recovered_action_reused_for_different_command",
            ),
        ):
            digest = _string(check.get(field)).lower()
            if not digest or not action_sha:
                continue
            prior = seen.get(digest)
            if prior is not None and prior != action_sha:
                blockers.append(f"{blocker}:step_{index}")
            else:
                seen[digest] = action_sha
    return blockers
