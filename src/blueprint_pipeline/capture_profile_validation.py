"""Deterministic capture-profile validation from immutable media observations.

The validator never changes a declared capture profile. It records whether
bounded probe topology and, for native dual-fisheye input, calibrated native
normalization are compatible with that declaration. A conflict is a blocker
that requires corrected deterministic intake, not an agent decision.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


CAPTURE_PROFILE_VALIDATION_SCHEMA_VERSION = "capture_profile_validation.v1"
_SUPPORTED_360_PROFILES = {
    "camera_360_equirectangular",
    "camera_360_native",
}
_OBSERVED_LANES = {
    "camera_360_equirectangular",
    "camera_360_native_candidate_requires_calibration",
    "unsupported_or_ambiguous_360_topology",
}


class CaptureProfileValidationError(ValueError):
    """Stable fail-closed profile-validation error."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    return re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")) is not None


def _timestamp(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureProfileValidationError(["capture_profile_timestamp_invalid"]) from exc
    if parsed.tzinfo is None:
        raise CaptureProfileValidationError(["capture_profile_timestamp_invalid"])
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_probe_receipts(
    values: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(values, Sequence) or not values:
        raise CaptureProfileValidationError(["capture_profile_probe_receipts_missing"])
    receipt_digests: list[str] = []
    source_digests: list[str] = []
    lanes: list[str] = []
    for ordinal, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise CaptureProfileValidationError(
                [f"capture_profile_probe_receipt_invalid:{ordinal}"]
            )
        receipt = dict(value)
        metadata = receipt.get("format_metadata")
        lane = (
            str(metadata.get("compatible_processing_lane") or "")
            if isinstance(metadata, Mapping)
            else ""
        )
        receipt_digest = receipt.get("probe_receipt_digest")
        source_digest = receipt.get("source_file_digest")
        if (
            receipt.get("schema_version") != "native_360_probe_receipt.v1"
            or receipt.get("probe_status") != "decodable"
            or not _is_digest(receipt_digest)
            or receipt_digest
            != canonical_digest(receipt, digest_field="probe_receipt_digest")
            or not _is_digest(source_digest)
            or lane not in _OBSERVED_LANES
            or metadata.get("processing_lane_claim_ceiling")
            != "container_stream_topology_only"
            or metadata.get("capture_profile_fully_validated") is not False
        ):
            raise CaptureProfileValidationError(
                [f"capture_profile_probe_receipt_invalid:{ordinal}"]
            )
        receipt_digests.append(str(receipt_digest))
        source_digests.append(str(source_digest))
        lanes.append(lane)
    if len(set(receipt_digests)) != len(receipt_digests) or len(set(source_digests)) != len(
        source_digests
    ):
        raise CaptureProfileValidationError(["capture_profile_probe_receipt_duplicate"])
    return sorted(receipt_digests), sorted(source_digests), sorted(lanes)


def _validated_native_normalization(
    value: Mapping[str, Any], *, source_capture_digest: str
) -> str:
    normalization = dict(value)
    digest = normalization.get("native_360_normalization_digest")
    if (
        normalization.get("schema_version") != "native_360_capture_normalization.v1"
        or normalization.get("source_capture_digest") != source_capture_digest
        or normalization.get("status") != "normalized"
        or normalization.get("blockers") != []
        or normalization.get("claim_ceiling") != "calibrated_camera_rig"
        or normalization.get("proof_effect") != "calibrated_native_360_rig_only"
        or not _is_digest(digest)
        or digest
        != canonical_digest(normalization, digest_field="native_360_normalization_digest")
    ):
        raise CaptureProfileValidationError(
            ["capture_profile_native_normalization_invalid"]
        )
    return str(digest)


def build_capture_profile_validation(
    *,
    source_capture_digest: str,
    declared_capture_authority_profile: str,
    probe_receipts: Sequence[Mapping[str, Any]],
    source_commit_sha: str,
    implementation_digest: str,
    timestamp: str,
    native_normalization_result: Mapping[str, Any] | None = None,
    parent_artifact_or_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one replayable 360 profile-validation decision."""

    declared_profile = str(declared_capture_authority_profile or "").strip()
    if (
        declared_profile not in _SUPPORTED_360_PROFILES
        or not _is_digest(source_capture_digest)
        or not _is_digest(implementation_digest)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit_sha) is None
    ):
        raise CaptureProfileValidationError(["capture_profile_request_invalid"])
    compiled_at = _timestamp(timestamp)
    receipt_digests, source_file_digests, lanes = _validated_probe_receipts(
        probe_receipts
    )
    unique_lanes = sorted(set(lanes))
    blockers: list[str] = []
    warnings: list[str] = []
    native_normalization_digest: str | None = None
    compatible_profile: str | None
    if unique_lanes == ["camera_360_equirectangular"]:
        compatible_profile = "camera_360_equirectangular"
        if native_normalization_result is not None:
            warnings.append("native_normalization_ignored_for_stitched_topology")
    elif unique_lanes == ["camera_360_native_candidate_requires_calibration"]:
        compatible_profile = "camera_360_native"
        if native_normalization_result is None:
            blockers.append("native_360_calibrated_normalization_required")
        else:
            native_normalization_digest = _validated_native_normalization(
                native_normalization_result,
                source_capture_digest=source_capture_digest,
            )
    else:
        compatible_profile = None
        blockers.append("unsupported_or_ambiguous_360_stream_topology")
    if compatible_profile is not None and compatible_profile != declared_profile:
        blockers.append("declared_profile_conflicts_with_observed_topology")
    blockers = sorted(set(blockers))
    warnings = sorted(set(warnings))
    status = "validated" if not blockers else "blocked"
    configuration = {
        "source_capture_digest": source_capture_digest,
        "declared_capture_authority_profile": declared_profile,
        "probe_receipt_digests": receipt_digests,
        "probe_source_file_digests": source_file_digests,
        "observed_processing_lanes": lanes,
        "native_normalization_digest": native_normalization_digest,
        "source_commit_sha": source_commit_sha,
        "implementation_digest": implementation_digest,
        "parent_artifact_digest": canonical_digest(dict(parent_artifact_or_event or {})),
    }
    configuration_digest = canonical_digest(configuration)
    result = {
        "schema_version": CAPTURE_PROFILE_VALIDATION_SCHEMA_VERSION,
        "stable_validation_identity": f"capture-profile-{configuration_digest[7:31]}",
        "source_capture_digest": source_capture_digest,
        "declared_capture_authority_profile": declared_profile,
        "compatible_capture_authority_profile": compatible_profile,
        "validation_status": status,
        "producing_method": "deterministic_capture_profile_validator.v1",
        "implementation_version": implementation_digest,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": {
            "probe_receipt_set_digest": canonical_digest(
                {"probe_receipt_digests": receipt_digests}
            ),
            "probe_source_file_set_digest": canonical_digest(
                {"probe_source_file_digests": source_file_digests}
            ),
            "native_normalization_digest": native_normalization_digest,
        },
        "probe_receipt_digests": receipt_digests,
        "probe_source_file_digests": source_file_digests,
        "observed_processing_lanes": lanes,
        "native_normalization_digest": native_normalization_digest,
        "blockers": blockers,
        "warnings": warnings,
        "agent_selected_capture_profile": False,
        "agent_may_change_capture_profile": False,
        "proof_effect": "capture_profile_validation_only" if status == "validated" else "none",
        "claim_ceiling": "capture_profile_compatibility",
        "parent_artifact_or_event": dict(parent_artifact_or_event or {}),
        "timestamp": compiled_at,
        "legal_next_actions": (
            ["compile_profile_specific_reconstruction_plan"]
            if status == "validated"
            else ["preserve_evidence_and_stop", "request_corrected_capture_intake"]
        ),
    }
    routing_binding = {
        key: result[key]
        for key in (
            "schema_version",
            "source_capture_digest",
            "declared_capture_authority_profile",
            "compatible_capture_authority_profile",
            "validation_status",
            "probe_receipt_digests",
            "probe_source_file_digests",
            "observed_processing_lanes",
            "native_normalization_digest",
            "blockers",
            "agent_selected_capture_profile",
            "agent_may_change_capture_profile",
            "proof_effect",
            "claim_ceiling",
            "legal_next_actions",
        )
    }
    result["capture_profile_routing_binding_digest"] = canonical_digest(
        routing_binding
    )
    result["capture_profile_validation_digest"] = canonical_digest(
        result,
        digest_field="capture_profile_validation_digest",
    )
    return validate_capture_profile_validation(result)


def validate_capture_profile_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact replayed profile-validation artifact."""

    try:
        result = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise CaptureProfileValidationError(["capture_profile_result_not_json"]) from exc
    required = {
        "schema_version",
        "stable_validation_identity",
        "source_capture_digest",
        "declared_capture_authority_profile",
        "compatible_capture_authority_profile",
        "validation_status",
        "producing_method",
        "implementation_version",
        "source_commit_sha",
        "deterministic_configuration_digest",
        "input_digests",
        "probe_receipt_digests",
        "probe_source_file_digests",
        "observed_processing_lanes",
        "native_normalization_digest",
        "blockers",
        "warnings",
        "agent_selected_capture_profile",
        "agent_may_change_capture_profile",
        "proof_effect",
        "claim_ceiling",
        "parent_artifact_or_event",
        "timestamp",
        "legal_next_actions",
        "capture_profile_routing_binding_digest",
        "capture_profile_validation_digest",
    }
    status = result.get("validation_status")
    compatible = result.get("compatible_capture_authority_profile")
    blockers = result.get("blockers")
    receipt_digests = result.get("probe_receipt_digests")
    source_file_digests = result.get("probe_source_file_digests")
    observed_lanes = result.get("observed_processing_lanes")
    native_normalization_digest = result.get("native_normalization_digest")
    parent = result.get("parent_artifact_or_event")
    input_digests = result.get("input_digests")
    lists_are_valid = all(
        isinstance(items, list) and bool(items)
        for items in (receipt_digests, source_file_digests, observed_lanes)
    )
    lists_are_valid = lists_are_valid and all(
        _is_digest(item)
        for item in (receipt_digests or []) + (source_file_digests or [])
    )
    lists_are_valid = lists_are_valid and receipt_digests == sorted(
        set(receipt_digests or [])
    )
    lists_are_valid = lists_are_valid and source_file_digests == sorted(
        set(source_file_digests or [])
    )
    lists_are_valid = lists_are_valid and observed_lanes == sorted(
        observed_lanes or []
    )
    lists_are_valid = lists_are_valid and all(
        lane in _OBSERVED_LANES for lane in (observed_lanes or [])
    )

    unique_lanes = sorted(set(observed_lanes or []))
    expected_blockers: list[str] = []
    if unique_lanes == ["camera_360_equirectangular"]:
        expected_compatible: str | None = "camera_360_equirectangular"
        native_binding_valid = native_normalization_digest is None
    elif unique_lanes == ["camera_360_native_candidate_requires_calibration"]:
        expected_compatible = "camera_360_native"
        native_binding_valid = native_normalization_digest is None or _is_digest(
            native_normalization_digest
        )
        if native_normalization_digest is None:
            expected_blockers.append("native_360_calibrated_normalization_required")
    else:
        expected_compatible = None
        native_binding_valid = native_normalization_digest is None
        expected_blockers.append("unsupported_or_ambiguous_360_stream_topology")
    if (
        expected_compatible is not None
        and expected_compatible != result.get("declared_capture_authority_profile")
    ):
        expected_blockers.append("declared_profile_conflicts_with_observed_topology")
    expected_blockers = sorted(set(expected_blockers))
    expected_status = "validated" if not expected_blockers else "blocked"
    expected_actions = (
        ["compile_profile_specific_reconstruction_plan"]
        if expected_status == "validated"
        else ["preserve_evidence_and_stop", "request_corrected_capture_intake"]
    )
    expected_input_digests = {
        "probe_receipt_set_digest": canonical_digest(
            {"probe_receipt_digests": receipt_digests}
        ),
        "probe_source_file_set_digest": canonical_digest(
            {"probe_source_file_digests": source_file_digests}
        ),
        "native_normalization_digest": native_normalization_digest,
    }
    expected_configuration = {
        "source_capture_digest": result.get("source_capture_digest"),
        "declared_capture_authority_profile": result.get(
            "declared_capture_authority_profile"
        ),
        "probe_receipt_digests": receipt_digests,
        "probe_source_file_digests": source_file_digests,
        "observed_processing_lanes": observed_lanes,
        "native_normalization_digest": native_normalization_digest,
        "source_commit_sha": result.get("source_commit_sha"),
        "implementation_digest": result.get("implementation_version"),
        "parent_artifact_digest": canonical_digest(parent if isinstance(parent, dict) else {}),
    }
    expected_configuration_digest = canonical_digest(expected_configuration)
    if (
        not isinstance(result, dict)
        or set(result) != required
        or result.get("schema_version") != CAPTURE_PROFILE_VALIDATION_SCHEMA_VERSION
        or result.get("producing_method")
        != "deterministic_capture_profile_validator.v1"
        or not _is_digest(result.get("source_capture_digest"))
        or not _is_digest(result.get("implementation_version"))
        or re.fullmatch(r"[0-9a-f]{40}", str(result.get("source_commit_sha") or ""))
        is None
        or not isinstance(parent, dict)
        or result.get("declared_capture_authority_profile") not in _SUPPORTED_360_PROFILES
        or compatible not in _SUPPORTED_360_PROFILES | {None}
        or compatible != expected_compatible
        or status != expected_status
        or not lists_are_valid
        or not native_binding_valid
        or input_digests != expected_input_digests
        or result.get("deterministic_configuration_digest")
        != expected_configuration_digest
        or result.get("stable_validation_identity")
        != f"capture-profile-{expected_configuration_digest[7:31]}"
        or not isinstance(blockers, list)
        or blockers != expected_blockers
        or not isinstance(result.get("warnings"), list)
        or result.get("warnings")
        != sorted(set(str(item) for item in result.get("warnings", [])))
        or result.get("agent_selected_capture_profile") is not False
        or result.get("agent_may_change_capture_profile") is not False
        or result.get("claim_ceiling") != "capture_profile_compatibility"
        or result.get("legal_next_actions") != expected_actions
        or not _is_digest(result.get("capture_profile_routing_binding_digest"))
        or result.get("capture_profile_routing_binding_digest")
        != canonical_digest(
            {
                key: result[key]
                for key in (
                    "schema_version",
                    "source_capture_digest",
                    "declared_capture_authority_profile",
                    "compatible_capture_authority_profile",
                    "validation_status",
                    "probe_receipt_digests",
                    "probe_source_file_digests",
                    "observed_processing_lanes",
                    "native_normalization_digest",
                    "blockers",
                    "agent_selected_capture_profile",
                    "agent_may_change_capture_profile",
                    "proof_effect",
                    "claim_ceiling",
                    "legal_next_actions",
                )
            }
        )
        or result.get("proof_effect")
        != ("capture_profile_validation_only" if status == "validated" else "none")
        or result.get("capture_profile_validation_digest")
        != canonical_digest(result, digest_field="capture_profile_validation_digest")
    ):
        raise CaptureProfileValidationError(["capture_profile_result_invalid"])
    _timestamp(result.get("timestamp"))
    return result


__all__ = [
    "CAPTURE_PROFILE_VALIDATION_SCHEMA_VERSION",
    "CaptureProfileValidationError",
    "build_capture_profile_validation",
    "validate_capture_profile_validation",
]
