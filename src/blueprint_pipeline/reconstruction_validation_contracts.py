"""Independent camera-rig and metric-scale validation contracts."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION = "camera_rig_validation_request.v1"
CAMERA_RIG_VALIDATION_RESULT_SCHEMA_VERSION = "camera_rig_validation_result.v1"
METRIC_SCALE_ANCHOR_SCHEMA_VERSION = "metric_scale_anchor_declaration.v1"
METRIC_SCALE_VALIDATION_REQUEST_SCHEMA_VERSION = "metric_scale_validation_request.v1"
METRIC_SCALE_VALIDATION_RESULT_SCHEMA_VERSION = "metric_scale_validation_result.v1"


class ReconstructionValidationContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationContractError(["validation_artifact_not_json"]) from exc


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _finalize(value: Mapping[str, Any], *, schema: str, digest_field: str) -> dict[str, Any]:
    artifact = _clone(dict(value))
    supplied = artifact.pop(digest_field, None)
    artifact["schema_version"] = schema
    artifact[digest_field] = canonical_digest(artifact, digest_field=digest_field)
    if supplied is not None and supplied != artifact[digest_field]:
        raise ReconstructionValidationContractError([f"{digest_field}_mismatch"])
    return artifact


def build_camera_rig_validation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION:
        errors.append("camera_rig_request_schema_invalid")
    for key in ("source_capture_digest", "native_360_normalization_digest"):
        if not _digest(request.get(key)):
            errors.append(f"camera_rig_request_{key}_invalid")
    rig = request.get("rig_declaration")
    binding = request.get("dual_fisheye_binding")
    if not isinstance(rig, Mapping) or (
        rig.get("schema_version") != "camera_360_rig_declaration.v1"
        or rig.get("rig_declaration_digest")
        != canonical_digest(rig, digest_field="rig_declaration_digest")
    ):
        errors.append("camera_rig_request_declaration_invalid")
    if not isinstance(binding, Mapping) or (
        binding.get("schema_version") != "dual_fisheye_stream_binding.v1"
        or binding.get("dual_fisheye_binding_digest")
        != canonical_digest(binding, digest_field="dual_fisheye_binding_digest")
    ):
        errors.append("camera_rig_request_stream_binding_invalid")
    if isinstance(rig, Mapping) and isinstance(binding, Mapping) and any(
        item != request.get("source_capture_digest")
        for item in (rig.get("capture_digest"), binding.get("capture_digest"))
    ):
        errors.append("camera_rig_request_capture_binding_mismatch")
    if request.get("agent_may_change_calibration") is not False:
        errors.append("camera_rig_request_agent_calibration_change_forbidden")
    if not str(request.get("timestamp") or "").strip():
        errors.append("camera_rig_request_timestamp_missing")
    if errors:
        raise ReconstructionValidationContractError(errors)
    return _finalize(
        request,
        schema=CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION,
        digest_field="camera_rig_validation_request_digest",
    )


def validate_camera_rig(value: Mapping[str, Any]) -> dict[str, Any]:
    request = build_camera_rig_validation_request(value)
    rig = request["rig_declaration"]
    binding = request["dual_fisheye_binding"]
    blockers = sorted(
        set(rig.get("blockers") or [])
        | set(binding.get("blockers") or [])
        | ({"camera_rig_calibration_invalid"} if rig.get("calibration_status") != "valid" else set())
        | ({"camera_rig_not_fixed"} if rig.get("rig_is_fixed") is not True else set())
        | (
            {"camera_rig_lens_streams_unsynchronized"}
            if binding.get("all_segments_synchronized") is not True
            else set()
        )
        | (
            {"camera_rig_capture_timeline_invalid"}
            if binding.get("capture_timeline_valid") is not True
            else set()
        )
    )
    result = {
        "source_capture_digest": request["source_capture_digest"],
        "camera_rig_validation_request_digest": request[
            "camera_rig_validation_request_digest"
        ],
        "native_360_normalization_digest": request["native_360_normalization_digest"],
        "rig_declaration_digest": rig["rig_declaration_digest"],
        "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
        "status": "validated" if not blockers else "rejected",
        "blockers": blockers,
        "fixed_rig_extrinsics_valid": rig.get("rig_is_fixed") is True and not blockers,
        "lens_calibration_valid": rig.get("calibration_status") == "valid" and not blockers,
        "lens_streams_synchronized": binding.get("all_segments_synchronized") is True,
        "capture_timeline_valid": binding.get("capture_timeline_valid") is True,
        "original_distorted_pixels_preserved": binding.get(
            "original_distorted_pixels_preserved"
        )
        is True,
        "agent_altered_calibration": False,
        "metric_scale_proven": False,
        "camera_trajectory_proven": False,
        "proof_effect": "calibrated_camera_rig_only" if not blockers else "none",
        "claim_ceiling": "calibrated_camera_rig" if not blockers else "decoded_native_container",
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "parent_artifact_or_event": {
            "native_360_normalization_digest": request[
                "native_360_normalization_digest"
            ]
        },
        "timestamp": request["timestamp"],
    }
    return _finalize(
        result,
        schema=CAMERA_RIG_VALIDATION_RESULT_SCHEMA_VERSION,
        digest_field="camera_rig_validation_result_digest",
    )


def build_metric_scale_anchor(value: Mapping[str, Any]) -> dict[str, Any]:
    anchor = _clone(dict(value))
    errors: list[str] = []
    if anchor.get("schema_version") != METRIC_SCALE_ANCHOR_SCHEMA_VERSION:
        errors.append("metric_anchor_schema_invalid")
    if not _digest(anchor.get("source_capture_digest")) or not _digest(
        anchor.get("evidence_digest")
    ):
        errors.append("metric_anchor_digest_invalid")
    if anchor.get("anchor_type") not in {
        "calibration_board",
        "measured_site_anchor",
        "known_marker_dimension",
        "independently_verified_metric_reference",
        "independently_checked_provider_metric_result",
    }:
        errors.append("metric_anchor_type_invalid")
    distance = anchor.get("measured_distance_m")
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(
        float(distance)
    ) or float(distance) <= 0:
        errors.append("metric_anchor_distance_invalid")
    if anchor.get("independently_verified") is not True or anchor.get(
        "learned_or_monocular_depth_only"
    ) is not False:
        errors.append("metric_anchor_independent_evidence_required")
    if not isinstance(anchor.get("coordinate_frame_declaration"), Mapping):
        errors.append("metric_anchor_coordinate_frame_invalid")
    if errors:
        raise ReconstructionValidationContractError(errors)
    return _finalize(
        anchor,
        schema=METRIC_SCALE_ANCHOR_SCHEMA_VERSION,
        digest_field="metric_scale_anchor_digest",
    )


def build_metric_scale_validation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != METRIC_SCALE_VALIDATION_REQUEST_SCHEMA_VERSION:
        errors.append("metric_scale_request_schema_invalid")
    anchor_value = request.get("anchor")
    try:
        anchor = build_metric_scale_anchor(anchor_value) if isinstance(anchor_value, Mapping) else None
    except ReconstructionValidationContractError as exc:
        errors.extend(exc.codes)
        anchor = None
    if anchor is None:
        errors.append("metric_scale_request_anchor_invalid")
    elif request.get("source_capture_digest") != anchor.get("source_capture_digest"):
        errors.append("metric_scale_request_capture_binding_mismatch")
    for key in ("source_capture_digest", "reconstruction_result_digest", "frozen_split_digest"):
        if not _digest(request.get(key)):
            errors.append(f"metric_scale_request_{key}_invalid")
    for key in ("estimated_anchor_distance_units", "maximum_relative_error"):
        number = request.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(
            float(number)
        ) or float(number) < 0 or (key == "estimated_anchor_distance_units" and float(number) == 0):
            errors.append(f"metric_scale_request_{key}_invalid")
    if request.get("maximum_relative_error", 1) >= 1:
        errors.append("metric_scale_request_threshold_invalid")
    if request.get("threshold_frozen_before_validation") is not True:
        errors.append("metric_scale_request_threshold_not_frozen")
    if request.get("candidate_may_change_anchor") is not False:
        errors.append("metric_scale_request_candidate_anchor_change_forbidden")
    if not str(request.get("timestamp") or "").strip():
        errors.append("metric_scale_request_timestamp_missing")
    if errors:
        raise ReconstructionValidationContractError(errors)
    request["anchor"] = anchor
    return _finalize(
        request,
        schema=METRIC_SCALE_VALIDATION_REQUEST_SCHEMA_VERSION,
        digest_field="metric_scale_validation_request_digest",
    )


def validate_metric_scale(value: Mapping[str, Any]) -> dict[str, Any]:
    request = build_metric_scale_validation_request(value)
    anchor = request["anchor"]
    measured = float(anchor["measured_distance_m"])
    estimated = float(request["estimated_anchor_distance_units"])
    relative_error = abs(estimated - measured) / measured
    passed = relative_error <= float(request["maximum_relative_error"])
    result = {
        "source_capture_digest": request["source_capture_digest"],
        "metric_scale_validation_request_digest": request[
            "metric_scale_validation_request_digest"
        ],
        "reconstruction_result_digest": request["reconstruction_result_digest"],
        "frozen_split_digest": request["frozen_split_digest"],
        "metric_scale_anchor_digest": anchor["metric_scale_anchor_digest"],
        "scale_factor_to_meters": round(measured / estimated, 12),
        "measured_anchor_distance_m": measured,
        "estimated_anchor_distance_units": estimated,
        "relative_error": round(relative_error, 12),
        "maximum_relative_error": float(request["maximum_relative_error"]),
        "status": "validated" if passed else "rejected",
        "blockers": [] if passed else ["scale_anchor_rejection"],
        "learned_or_monocular_depth_established_scale": False,
        "agent_changed_anchor_or_threshold": False,
        "proof_effect": "metric_scale_validated" if passed else "none",
        "claim_ceiling": "metric_scale" if passed else "appearance_reconstruction",
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "parent_artifact_or_event": {
            "reconstruction_result_digest": request["reconstruction_result_digest"]
        },
        "timestamp": request["timestamp"],
    }
    return _finalize(
        result,
        schema=METRIC_SCALE_VALIDATION_RESULT_SCHEMA_VERSION,
        digest_field="metric_scale_validation_result_digest",
    )


__all__ = [
    "CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION",
    "CAMERA_RIG_VALIDATION_RESULT_SCHEMA_VERSION",
    "METRIC_SCALE_ANCHOR_SCHEMA_VERSION",
    "METRIC_SCALE_VALIDATION_REQUEST_SCHEMA_VERSION",
    "METRIC_SCALE_VALIDATION_RESULT_SCHEMA_VERSION",
    "ReconstructionValidationContractError",
    "build_camera_rig_validation_request",
    "build_metric_scale_anchor",
    "build_metric_scale_validation_request",
    "validate_camera_rig",
    "validate_metric_scale",
]
