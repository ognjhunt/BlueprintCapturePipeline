"""Fail-closed contracts for SC3-style evaluator fidelity.

These validators never infer proof from artifact presence. They validate the
content required for synchronized multiview, receding-horizon execution,
SC3-trained checkpoint identity, frozen OOD evaluation, and the external study
that is still required before a public rank-fidelity claim.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .external_study_protocols import (
    EXTERNAL_STUDY_PROTOCOL_PROFILES,
    OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC,
    validate_external_study,
)

__all__ = ["EXTERNAL_STUDY_PROTOCOL_PROFILES", "validate_external_study"]


SC3_CAMERA_COUNT = 3
SC3_MAX_CAMERA_SKEW_MS = 10.0
SC3_PROPOSED_ACTION_COUNT = 25
SC3_PREDICTED_ACTION_COUNT = 24
SC3_RETAINED_ACTION_COUNT = 16
SC3_REQUIRED_TRAINING_MODES = {"forward_dynamics", "inverse_dynamics", "cross_view"}
SC3_OOD_AXES = {
    "site",
    "task",
    "policy_family",
    "embodiment",
    "camera",
    "visual",
    "dynamics",
    "contact",
}
SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV = "BLUEPRINT_SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256"
SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256_ENV = "BLUEPRINT_SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256"
SC3_MULTIVIEW_CHECKER_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_MULTIVIEW_CHECKER_TRUSTED_PUBLIC_KEY_SHA256"
)
SC3_TASK_COMPLETION_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_TASK_COMPLETION_TRUSTED_PUBLIC_KEY_SHA256"
)
SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256"
)
SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256"
)
SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV = (
    "BLUEPRINT_SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256"
)
SC3_MIN_ANCHOR_REPLICATES_PER_POLICY_CONDITION = 20
SC3_MIN_POLICY_GROUPS_FOR_DIAGNOSTIC_CORRELATION = 3
SC3_MIN_OOD_REPLICATES_PER_POLICY_CONDITION = 20
SC3_MIN_OOD_POLICY_COUNT = 3
SC3_OOD_BOOTSTRAP_SAMPLE_COUNT = 512
SC3_OOD_UNCERTAINTY_METHOD = "hierarchical_heldout_group_condition_matched_seed_resample.v1"
SC3_OOD_BOOTSTRAP_CLUSTER_LEVELS = (
    "heldout_group_id",
    "condition_id",
    "replicate_seed",
)
def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [_string(item) for item in value if _string(item)]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256(value: Any) -> bool:
    text = _string(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any], *, exclude: Sequence[str] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in set(exclude)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_ed25519_attestation(
    signature: Mapping[str, Any],
    *,
    signed_payload: Mapping[str, Any],
    prefix: str,
    trusted_public_key_sha256_env: str,
) -> list[str]:
    blockers: list[str] = []
    public_key_sha256 = _string(signature.get("public_key_sha256")).lower()
    signed_payload_sha256 = hashlib.sha256(_canonical_bytes(signed_payload)).hexdigest()
    trusted_public_key_sha256 = _string(os.getenv(trusted_public_key_sha256_env)).lower()
    if not _sha256(trusted_public_key_sha256):
        blockers.append(f"{prefix}_trusted_public_key_not_configured")
    elif public_key_sha256 != trusted_public_key_sha256:
        blockers.append(f"{prefix}_public_key_not_authorized")
    if not (
        signature.get("algorithm") == "Ed25519"
        and signature.get("signature_verified") is True
        and _string(signature.get("verifier_id"))
        and _string(signature.get("signer_key_id"))
        and _sha256(public_key_sha256)
        and _string(signature.get("signed_payload_sha256")).lower() == signed_payload_sha256
    ):
        blockers.append(f"{prefix}_metadata_invalid")
    try:
        public_key_raw = base64.b64decode(
            _string(signature.get("public_key_base64")), validate=True
        )
        signature_raw = base64.b64decode(_string(signature.get("signature_base64")), validate=True)
        if hashlib.sha256(public_key_raw).hexdigest() != public_key_sha256:
            raise ValueError("public key fingerprint mismatch")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            signature_raw,
            _canonical_bytes(signed_payload),
        )
    except (ImportError, TypeError, ValueError):
        blockers.append(f"{prefix}_cryptographic_verification_failed")
    except Exception:  # cryptography raises InvalidSignature from a backend module
        blockers.append(f"{prefix}_cryptographic_verification_failed")
    report_ref = _mapping(signature.get("verification_report_artifact"))
    blockers.extend(_validate_artifact_ref(report_ref, prefix=f"{prefix}_verification_report"))
    report_path = Path(_string(report_ref.get("path"))).expanduser()
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = _mapping(json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    if not (
        report.get("schema_version") == "sc3_signature_verification_report.v1"
        and report.get("algorithm") == "Ed25519"
        and report.get("verification_status") == "verified"
        and report.get("public_key_sha256") == public_key_sha256
        and report.get("signed_payload_sha256") == signed_payload_sha256
        and report.get("signer_key_id") == signature.get("signer_key_id")
        and report.get("verifier_id") == signature.get("verifier_id")
    ):
        blockers.append(f"{prefix}_verification_report_content_mismatch")
    return sorted(set(blockers))


def validate_trusted_ed25519_attestation(
    signature: Mapping[str, Any],
    *,
    signed_payload: Mapping[str, Any],
    prefix: str,
    trusted_public_key_sha256_env: str,
) -> dict[str, Any]:
    blockers = _verify_ed25519_attestation(
        signature,
        signed_payload=signed_payload,
        prefix=prefix,
        trusted_public_key_sha256_env=trusted_public_key_sha256_env,
    )
    return {
        "status": "validated" if not blockers else "blocked",
        "trusted_public_key_sha256_env": trusted_public_key_sha256_env,
        "blockers": blockers,
    }


def _validate_artifact_ref(ref: Mapping[str, Any], *, prefix: str) -> list[str]:
    blockers: list[str] = []
    path_text = _string(ref.get("path"))
    digest = _string(ref.get("sha256")).lower()
    if not path_text:
        blockers.append(f"{prefix}_path_missing")
        return blockers
    path = Path(path_text).expanduser()
    if not path.is_file():
        blockers.append(f"{prefix}_file_missing")
    if not _sha256(digest):
        blockers.append(f"{prefix}_sha256_invalid")
    elif path.is_file() and _file_sha256(path) != digest:
        blockers.append(f"{prefix}_sha256_mismatch")
    return blockers


def _valid_intrinsics(value: Any) -> bool:
    intrinsics = _mapping(value)
    values = [_number(intrinsics.get(key)) for key in ("fx", "fy", "cx", "cy", "width", "height")]
    return bool(
        all(item is not None for item in values)
        and values[0] > 0
        and values[1] > 0
        and values[4] > 0
        and values[5] > 0
        and 0 <= values[2] < values[4]
        and 0 <= values[3] < values[5]
    )


def _valid_extrinsics(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if len(value) != 4:
        return False
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            return False
        numbers = [_number(item) for item in row]
        if len(numbers) != 4 or any(item is None for item in numbers):
            return False
        matrix.append([float(item) for item in numbers if item is not None])
    if any(abs(left - right) > 1e-6 for left, right in zip(matrix[3], [0, 0, 0, 1])):
        return False
    rotation = [row[:3] for row in matrix[:3]]
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    orthonormal = all(
        abs(
            sum(rotation[row][axis] * rotation[column][axis] for axis in range(3))
            - (1.0 if row == column else 0.0)
        )
        <= 1e-3
        for row in range(3)
        for column in range(3)
    )
    return abs(determinant - 1.0) <= 1e-3 and orthonormal


def _decoded_grayscale_signature(path: Path) -> tuple[list[float], float] | None:
    try:
        with Image.open(path) as image:
            values = [float(value) for value in image.convert("L").resize((32, 32)).getdata()]
    except (OSError, ValueError):
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return values, variance


def _image_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-9:
        return None
    return (
        sum(
            left_value * right_value
            for left_value, right_value in zip(left_centered, right_centered)
        )
        / denominator
    )


def validate_synchronized_multiview(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(manifest.get("synchronized_multiview") or manifest)
    frame_groups = _rows(payload.get("frame_groups"))
    blockers: list[str] = []
    expected_camera_ids = [_string(item) for item in payload.get("expected_camera_ids", []) or []]
    if (
        len(expected_camera_ids) != SC3_CAMERA_COUNT
        or any(not item for item in expected_camera_ids)
        or len(set(expected_camera_ids)) != len(expected_camera_ids)
    ):
        blockers.append("multiview_expected_camera_registry_invalid")
    if payload.get("joint_generation_proven") is not True:
        blockers.append("multiview_joint_generation_not_proven")
    blockers.extend(
        _validate_artifact_ref(
            _mapping(payload.get("joint_generation_artifact")),
            prefix="multiview_joint_generation_artifact",
        )
    )
    if not frame_groups:
        blockers.append("synchronized_multiview_frame_groups_missing")
    elif len(frame_groups) < 3:
        blockers.append("synchronized_multiview_requires_three_temporal_frame_groups")
    seen_group_ids: set[str] = set()
    observed_group_indices: list[int] = []
    frame_group_input_sha256: list[str] = []
    prior_camera_signatures: dict[str, list[float]] = {}
    for group_index, group in enumerate(frame_groups):
        group_id = _string(group.get("frame_group_id"))
        if not group_id or group_id in seen_group_ids:
            blockers.append(f"multiview_frame_group_id_missing_or_duplicate:{group_index}")
        seen_group_ids.add(group_id)
        frames = _rows(group.get("frames"))
        if len(frames) < SC3_CAMERA_COUNT:
            blockers.append(f"multiview_camera_count_lt_3:{group_index}")
            continue
        camera_ids = [_string(frame.get("camera_id")) for frame in frames]
        if any(not camera_id for camera_id in camera_ids) or len(set(camera_ids)) != len(
            camera_ids
        ):
            blockers.append(f"multiview_camera_ids_missing_or_duplicate:{group_index}")
        if set(camera_ids) != set(expected_camera_ids):
            blockers.append(f"multiview_camera_registry_mismatch:{group_index}")
        content_hashes = [_string(frame.get("image_sha256")) for frame in frames]
        if any(not _sha256(value) for value in content_hashes):
            blockers.append(f"multiview_image_sha256_missing_or_invalid:{group_index}")
        if len(set(content_hashes)) != len(content_hashes):
            blockers.append(f"multiview_duplicate_camera_content:{group_index}")
        camera_signatures: dict[str, list[float]] = {}
        for frame_index, frame in enumerate(frames):
            blockers.extend(
                _validate_artifact_ref(
                    {
                        "path": frame.get("image_path"),
                        "sha256": frame.get("image_sha256"),
                    },
                    prefix=f"multiview_image_artifact:{group_index}:{frame_index}",
                )
            )
            path = Path(_string(frame.get("image_path"))).expanduser()
            if path.is_file():
                try:
                    with Image.open(path) as decoded:
                        decoded.load()
                        decoded_size = decoded.size
                except (OSError, ValueError):
                    blockers.append(f"multiview_image_decode_failed:{group_index}:{frame_index}")
                else:
                    intrinsics = _mapping(frame.get("intrinsics"))
                    if decoded_size != (
                        int(_number(intrinsics.get("width")) or -1),
                        int(_number(intrinsics.get("height")) or -1),
                    ):
                        blockers.append(
                            f"multiview_image_resolution_mismatch:{group_index}:{frame_index}"
                        )
                    signature = _decoded_grayscale_signature(path)
                    if signature is None or signature[1] < 25.0:
                        blockers.append(
                            f"multiview_image_visual_structure_insufficient:{group_index}:{frame_index}"
                        )
                    else:
                        camera_signatures[_string(frame.get("camera_id"))] = signature[0]
        if len(camera_signatures) == len(frames):
            pairwise_correlations = [
                _image_correlation(camera_signatures[left], camera_signatures[right])
                for left_index, left in enumerate(camera_ids)
                for right in camera_ids[left_index + 1 :]
            ]
            if any(
                correlation is None or correlation < 0.1 for correlation in pairwise_correlations
            ):
                blockers.append(f"multiview_cross_view_visual_structure_inconsistent:{group_index}")
            if prior_camera_signatures:
                for camera_id in camera_ids:
                    current = camera_signatures.get(camera_id)
                    previous = prior_camera_signatures.get(camera_id)
                    if current is None or previous is None:
                        continue
                    same_camera_correlation = _image_correlation(current, previous)
                    other_camera_correlations = [
                        correlation
                        for other_id, other_signature in prior_camera_signatures.items()
                        if other_id != camera_id
                        and (correlation := _image_correlation(current, other_signature))
                        is not None
                    ]
                    if same_camera_correlation is None or (
                        other_camera_correlations
                        and same_camera_correlation + 1e-6 < max(other_camera_correlations)
                    ):
                        blockers.append(
                            f"multiview_temporal_camera_assignment_inconsistent:{group_index}:{camera_id}"
                        )
            prior_camera_signatures = camera_signatures
        timestamps = [_number(frame.get("timestamp_sec")) for frame in frames]
        if any(value is None for value in timestamps):
            blockers.append(f"multiview_timestamp_missing_or_nonfinite:{group_index}")
        else:
            skew_ms = (max(timestamps) - min(timestamps)) * 1000.0  # type: ignore[arg-type]
            if skew_ms > SC3_MAX_CAMERA_SKEW_MS:
                blockers.append(f"multiview_timestamp_skew_exceeded:{group_index}")
        simultaneous_indices = [frame.get("simultaneous_frame_index") for frame in frames]
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in simultaneous_indices
        ):
            blockers.append(f"multiview_simultaneous_index_missing:{group_index}")
        elif len(set(simultaneous_indices)) != 1:
            blockers.append(f"multiview_unsynchronized_frame_indices:{group_index}")
        else:
            observed_group_indices.append(simultaneous_indices[0])
        for frame_index, frame in enumerate(frames):
            if not _valid_intrinsics(frame.get("intrinsics")):
                blockers.append(f"multiview_intrinsics_invalid:{group_index}:{frame_index}")
            if not _valid_extrinsics(frame.get("world_from_camera")):
                blockers.append(f"multiview_extrinsics_invalid:{group_index}:{frame_index}")
        camera_positions = []
        for frame in frames:
            matrix = frame.get("world_from_camera")
            if (
                isinstance(matrix, Sequence)
                and not isinstance(matrix, (str, bytes, bytearray))
                and len(matrix) == 4
            ):
                try:
                    camera_positions.append(tuple(float(matrix[axis][3]) for axis in range(3)))
                except (IndexError, TypeError, ValueError):
                    pass
        if len(camera_positions) != len(frames) or len(set(camera_positions)) != len(frames):
            blockers.append(f"multiview_camera_baselines_missing_or_duplicate:{group_index}")
        elif any(
            math.dist(camera_positions[left], camera_positions[right]) <= 1e-3
            for left in range(len(camera_positions))
            for right in range(left + 1, len(camera_positions))
        ):
            blockers.append(f"multiview_camera_baseline_too_small:{group_index}")
        correspondence = _mapping(group.get("correspondence_check"))
        group_input_sha256 = _canonical_sha256(
            {
                "frame_group_id": group_id,
                "frames": frames,
            }
        )
        frame_group_input_sha256.append(group_input_sha256)
        error = _number(correspondence.get("reprojection_error_px"))
        threshold = _number(correspondence.get("threshold_px"))
        if not (
            correspondence.get("status") == "passed"
            and error is not None
            and threshold is not None
            and 0.0 <= error <= threshold
            and isinstance(correspondence.get("matched_point_count"), int)
            and not isinstance(correspondence.get("matched_point_count"), bool)
            and correspondence.get("matched_point_count") >= 8
        ):
            blockers.append(f"multiview_correspondence_check_failed:{group_index}")
        for check_name in ("occlusion_reentry_check", "camera_assignment_check"):
            check = _mapping(group.get(check_name))
            if check.get("status") != "passed":
                blockers.append(f"multiview_{check_name}_failed:{group_index}")
        for check_name in (
            "correspondence_check",
            "occlusion_reentry_check",
            "camera_assignment_check",
        ):
            check = _mapping(group.get(check_name))
            if not (
                _string(check.get("checker_id"))
                and _sha256(check.get("checker_code_sha256"))
                and _string(check.get("input_manifest_sha256")) == group_input_sha256
            ):
                blockers.append(f"multiview_{check_name}_provenance_invalid:{group_index}")
            blockers.extend(
                _validate_artifact_ref(
                    _mapping(check.get("evidence_artifact")),
                    prefix=f"multiview_{check_name}_evidence:{group_index}",
                )
            )
            evidence_ref = _mapping(check.get("evidence_artifact"))
            evidence_path = Path(_string(evidence_ref.get("path"))).expanduser()
            evidence_payload: dict[str, Any] = {}
            if evidence_path.is_file():
                try:
                    evidence_payload = _mapping(
                        json.loads(evidence_path.read_text(encoding="utf-8"))
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            if check_name == "correspondence_check":
                expected_result = {
                    "reprojection_error_px": check.get("reprojection_error_px"),
                    "threshold_px": check.get("threshold_px"),
                    "matched_point_count": check.get("matched_point_count"),
                }
            elif check_name == "occlusion_reentry_check":
                expected_result = {
                    "visible_before_occlusion": check.get("visible_before_occlusion"),
                    "occlusion_observed": check.get("occlusion_observed"),
                    "reentry_correspondence_verified": check.get("reentry_correspondence_verified"),
                }
            else:
                expected_result = {"verified_camera_ids": check.get("verified_camera_ids")}
            if not (
                evidence_payload.get("schema_version") == "sc3_multiview_check_evidence.v1"
                and evidence_payload.get("check_type") == check_name
                and evidence_payload.get("status") == check.get("status")
                and evidence_payload.get("checker_id") == check.get("checker_id")
                and evidence_payload.get("checker_code_sha256") == check.get("checker_code_sha256")
                and evidence_payload.get("input_manifest_sha256") == group_input_sha256
                and _mapping(evidence_payload.get("result")) == expected_result
            ):
                blockers.append(f"multiview_{check_name}_evidence_content_mismatch:{group_index}")
            blockers.extend(
                _verify_ed25519_attestation(
                    _mapping(evidence_payload.get("checker_attestation")),
                    signed_payload={
                        key: value
                        for key, value in evidence_payload.items()
                        if key != "checker_attestation"
                    },
                    prefix=(f"multiview_{check_name}_checker_attestation:{group_index}"),
                    trusted_public_key_sha256_env=(
                        SC3_MULTIVIEW_CHECKER_TRUSTED_PUBLIC_KEY_SHA256_ENV
                    ),
                )
            )
        occlusion = _mapping(group.get("occlusion_reentry_check"))
        if not (
            occlusion.get("visible_before_occlusion") is True
            and occlusion.get("occlusion_observed") is True
            and occlusion.get("reentry_correspondence_verified") is True
        ):
            blockers.append(f"multiview_occlusion_reentry_evidence_invalid:{group_index}")
        assignment = _mapping(group.get("camera_assignment_check"))
        if set(_string_list(assignment.get("verified_camera_ids"))) != set(expected_camera_ids):
            blockers.append(f"multiview_camera_assignment_evidence_invalid:{group_index}")
    if observed_group_indices and observed_group_indices != list(
        range(observed_group_indices[0], observed_group_indices[0] + len(frame_groups))
    ):
        blockers.append("multiview_frame_group_indices_not_contiguous")
    joint_ref = _mapping(payload.get("joint_generation_artifact"))
    joint_path = Path(_string(joint_ref.get("path"))).expanduser()
    joint_payload: dict[str, Any] = {}
    if joint_path.is_file():
        try:
            joint_payload = _mapping(json.loads(joint_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    if not (
        joint_payload.get("schema_version") == "sc3_joint_multiview_generation.v1"
        and joint_payload.get("joint_generation_proven") is True
        and _string_list(joint_payload.get("expected_camera_ids")) == expected_camera_ids
        and _string_list(joint_payload.get("frame_group_input_sha256")) == frame_group_input_sha256
    ):
        blockers.append("multiview_joint_generation_artifact_content_mismatch")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "sc3_synchronized_multiview_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "frame_group_count": len(frame_groups),
        "required_camera_count": SC3_CAMERA_COUNT,
        "max_camera_skew_ms": SC3_MAX_CAMERA_SKEW_MS,
        "blockers": blockers,
    }


def validate_horizon_execution_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    proposed = _rows(trace.get("proposed_actions"))
    predicted = _rows(trace.get("world_model_predictions"))
    retained = _rows(trace.get("retained_actions"))
    executed = _rows(trace.get("executed_actions"))
    discarded = _rows(trace.get("discarded_predictions"))
    blockers: list[str] = []
    for identity_field in ("runtime_session_id", "runtime_executor_id", "controller_id"):
        if not _string(trace.get(identity_field)):
            blockers.append(f"horizon_{identity_field}_missing")
    for digest_field in (
        "runtime_executor_code_sha256",
        "controller_sha256",
        "world_model_checkpoint_sha256",
    ):
        if not _sha256(trace.get(digest_field)):
            blockers.append(f"horizon_{digest_field}_missing_or_invalid")
    trace_artifact = _mapping(trace.get("executor_trace_artifact"))
    blockers.extend(
        _validate_artifact_ref(trace_artifact, prefix="horizon_executor_trace_artifact")
    )
    bound_fields = (
        "trace_producer_id",
        "runtime_session_id",
        "runtime_executor_id",
        "runtime_executor_code_sha256",
        "controller_id",
        "controller_sha256",
        "world_model_checkpoint_sha256",
        "runtime_execution_proven",
        "world_model_prediction_proven",
        "receding_horizon_controller_proven",
        "proposed_actions",
        "world_model_predictions",
        "retained_actions",
        "executed_actions",
        "discarded_predictions",
        "control_rate_hz",
        "chunk_start_timestamp_sec",
        "requery_timestamp_sec",
    )
    artifact_payload: dict[str, Any] = {}
    artifact_path = Path(_string(trace_artifact.get("path"))).expanduser()
    if artifact_path.is_file():
        try:
            artifact_payload = _mapping(json.loads(artifact_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            blockers.append("horizon_executor_trace_artifact_content_invalid")
    if not (
        artifact_payload.get("schema_version") == "sc3_horizon_executor_trace.v1"
        and all(artifact_payload.get(field) == trace.get(field) for field in bound_fields)
    ):
        blockers.append("horizon_executor_trace_artifact_content_mismatch")
    if trace.get("trace_producer_id") != "blueprint_sc3_receding_horizon_executor":
        blockers.append("horizon_trace_not_emitted_by_registered_executor")
    executor_attestation = _mapping(trace.get("executor_attestation"))
    blockers.extend(
        _verify_ed25519_attestation(
            executor_attestation,
            signed_payload={field: trace.get(field) for field in bound_fields},
            prefix="horizon_executor_attestation",
            trusted_public_key_sha256_env=(SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV),
        )
    )
    if trace.get("runtime_execution_proven") is not True:
        blockers.append("horizon_runtime_execution_not_proven")
    if trace.get("world_model_prediction_proven") is not True:
        blockers.append("horizon_world_model_prediction_not_proven")
    if trace.get("receding_horizon_controller_proven") is not True:
        blockers.append("horizon_receding_controller_not_proven")
    for name, rows, expected in (
        ("proposed", proposed, SC3_PROPOSED_ACTION_COUNT),
        ("predicted", predicted, SC3_PREDICTED_ACTION_COUNT),
        ("retained", retained, SC3_RETAINED_ACTION_COUNT),
        ("executed", executed, SC3_RETAINED_ACTION_COUNT),
        ("discarded", discarded, SC3_PREDICTED_ACTION_COUNT - SC3_RETAINED_ACTION_COUNT),
    ):
        if len(rows) != expected:
            blockers.append(f"horizon_{name}_count_must_equal_{expected}")
    proposed_ids = [_string(row.get("action_id")) for row in proposed]
    predicted_ids = [_string(row.get("action_id")) for row in predicted]
    retained_ids = [_string(row.get("action_id")) for row in retained]
    executed_ids = [_string(row.get("action_id")) for row in executed]
    discarded_ids = [_string(row.get("action_id")) for row in discarded]
    if any(not value for value in proposed_ids) or len(set(proposed_ids)) != len(proposed_ids):
        blockers.append("horizon_proposed_action_ids_missing_or_duplicate")
    proposed_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(proposed):
        vector = row.get("action_vector_7d")
        values = (
            [_number(value) for value in vector]
            if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes, bytearray))
            else []
        )
        if len(values) != 7 or any(value is None for value in values):
            blockers.append(f"horizon_action_vector_not_finite_7d:{index}")
        if not _sha256(row.get("action_sha256")):
            blockers.append(f"horizon_action_sha256_missing_or_invalid:{index}")
        elif len(values) == 7 and all(value is not None for value in values):
            computed = hashlib.sha256(
                json.dumps(
                    [float(value) for value in values if value is not None],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if _string(row.get("action_sha256")).lower() != computed:
                blockers.append(f"horizon_action_sha256_mismatch:{index}")
        if proposed_ids[index]:
            proposed_by_id[proposed_ids[index]] = row
    prediction_indices: list[int] = []
    prediction_ids: set[str] = set()
    prediction_runtime_result_ids: set[str] = set()
    prediction_evidence_sha256s: set[str] = set()
    for index, row in enumerate(predicted):
        prediction_index = row.get("prediction_index")
        if isinstance(prediction_index, bool) or not isinstance(prediction_index, int):
            blockers.append(f"horizon_prediction_index_missing_or_invalid:{index}")
        else:
            prediction_indices.append(prediction_index)
        if row.get("prediction_status") != "completed":
            blockers.append(f"horizon_prediction_not_completed:{index}")
        prediction_id = _string(row.get("prediction_id"))
        prediction_runtime_result_id = _string(row.get("prediction_runtime_result_id"))
        if not prediction_id or prediction_id in prediction_ids:
            blockers.append(f"horizon_prediction_id_missing:{index}")
        prediction_ids.add(prediction_id)
        if (
            not prediction_runtime_result_id
            or prediction_runtime_result_id in prediction_runtime_result_ids
        ):
            blockers.append(f"horizon_prediction_runtime_result_id_missing_or_duplicate:{index}")
        prediction_runtime_result_ids.add(prediction_runtime_result_id)
        source = proposed_by_id.get(predicted_ids[index], {})
        if _string(row.get("action_sha256")) != _string(source.get("action_sha256")):
            blockers.append(f"horizon_prediction_action_digest_mismatch:{index}")
        evidence_payload, evidence_blockers = _load_json_artifact(
            row.get("prediction_evidence_artifact"),
            prefix=f"horizon_prediction_evidence:{index}",
        )
        blockers.extend(evidence_blockers)
        evidence_sha256 = _string(
            _mapping(row.get("prediction_evidence_artifact")).get("sha256")
        ).lower()
        if not evidence_sha256 or evidence_sha256 in prediction_evidence_sha256s:
            blockers.append(f"horizon_prediction_evidence_missing_or_duplicate:{index}")
        prediction_evidence_sha256s.add(evidence_sha256)
        if not (
            row.get("prediction_result_schema_version") == "sc3_world_model_prediction_result.v1"
            and evidence_payload.get("schema_version") == "sc3_world_model_prediction_evidence.v1"
            and evidence_payload.get("status") == "completed"
            and evidence_payload.get("runtime_session_id") == trace.get("runtime_session_id")
            and evidence_payload.get("runtime_result_id") == prediction_runtime_result_id
            and evidence_payload.get("prediction_id") == prediction_id
            and evidence_payload.get("action_id") == row.get("action_id")
            and evidence_payload.get("action_sha256") == row.get("action_sha256")
            and evidence_payload.get("world_model_checkpoint_sha256")
            == trace.get("world_model_checkpoint_sha256")
        ):
            blockers.append(f"horizon_prediction_evidence_binding_invalid:{index}")
    if prediction_indices != list(range(SC3_PREDICTED_ACTION_COUNT)):
        blockers.append("horizon_prediction_indices_not_contiguous_0_to_23")
    for index, row in enumerate(retained):
        if row.get("retention_status") != "retained_for_execution":
            blockers.append(f"horizon_retained_status_invalid:{index}")
        source = proposed_by_id.get(retained_ids[index], {})
        if _string(row.get("action_sha256")) != _string(source.get("action_sha256")):
            blockers.append(f"horizon_retained_action_digest_mismatch:{index}")
    execution_timestamps: list[float] = []
    controller_runtime_result_ids: set[str] = set()
    controller_evidence_sha256s: set[str] = set()
    for index, row in enumerate(executed):
        if row.get("execution_status") != "executed":
            blockers.append(f"horizon_execution_status_invalid:{index}")
        timestamp = _number(row.get("execution_timestamp_sec"))
        if timestamp is None:
            blockers.append(f"horizon_execution_timestamp_missing_or_invalid:{index}")
        else:
            execution_timestamps.append(timestamp)
        source = proposed_by_id.get(executed_ids[index], {})
        if _string(row.get("action_sha256")) != _string(source.get("action_sha256")):
            blockers.append(f"horizon_executed_action_digest_mismatch:{index}")
        controller_runtime_result_id = _string(row.get("controller_runtime_result_id"))
        if (
            not controller_runtime_result_id
            or controller_runtime_result_id in controller_runtime_result_ids
        ):
            blockers.append(f"horizon_controller_runtime_result_id_missing_or_duplicate:{index}")
        controller_runtime_result_ids.add(controller_runtime_result_id)
        evidence_payload, evidence_blockers = _load_json_artifact(
            row.get("controller_evidence_artifact"),
            prefix=f"horizon_controller_evidence:{index}",
        )
        blockers.extend(evidence_blockers)
        evidence_sha256 = _string(
            _mapping(row.get("controller_evidence_artifact")).get("sha256")
        ).lower()
        if not evidence_sha256 or evidence_sha256 in controller_evidence_sha256s:
            blockers.append(f"horizon_controller_evidence_missing_or_duplicate:{index}")
        controller_evidence_sha256s.add(evidence_sha256)
        if not (
            row.get("execution_result_schema_version") == "sc3_controller_execution_result.v1"
            and evidence_payload.get("schema_version") == "sc3_controller_execution_evidence.v1"
            and evidence_payload.get("status") == "completed"
            and evidence_payload.get("runtime_session_id") == trace.get("runtime_session_id")
            and evidence_payload.get("runtime_result_id") == controller_runtime_result_id
            and evidence_payload.get("action_id") == row.get("action_id")
            and evidence_payload.get("action_sha256") == row.get("action_sha256")
            and evidence_payload.get("controller_id") == trace.get("controller_id")
            and evidence_payload.get("controller_sha256") == trace.get("controller_sha256")
            and _number(evidence_payload.get("execution_timestamp_sec")) == timestamp
        ):
            blockers.append(f"horizon_controller_evidence_binding_invalid:{index}")
    for index, row in enumerate(discarded):
        if not (
            row.get("retention_status") == "discarded_not_executed" and row.get("executed") is False
        ):
            blockers.append(f"horizon_discarded_status_invalid:{index}")
        source = proposed_by_id.get(discarded_ids[index], {})
        if _string(row.get("action_sha256")) != _string(source.get("action_sha256")):
            blockers.append(f"horizon_discarded_action_digest_mismatch:{index}")
    if predicted_ids != proposed_ids[:SC3_PREDICTED_ACTION_COUNT]:
        blockers.append("horizon_prediction_actions_not_first_24_proposed")
    if retained_ids != predicted_ids[:SC3_RETAINED_ACTION_COUNT]:
        blockers.append("horizon_retained_actions_not_first_16_predicted")
    if executed_ids != retained_ids:
        blockers.append("horizon_executed_actions_do_not_match_retained_16")
    if discarded_ids != predicted_ids[SC3_RETAINED_ACTION_COUNT:]:
        blockers.append("horizon_discarded_predictions_not_final_8")
    control_rate = _number(trace.get("control_rate_hz"))
    start = _number(trace.get("chunk_start_timestamp_sec"))
    requery = _number(trace.get("requery_timestamp_sec"))
    if control_rate is None or control_rate <= 0:
        blockers.append("horizon_control_rate_missing_or_invalid")
    if start is None or requery is None:
        blockers.append("horizon_requery_timestamps_missing_or_invalid")
    elif control_rate and abs(requery - (start + SC3_RETAINED_ACTION_COUNT / control_rate)) > 1e-6:
        blockers.append("horizon_requery_timestamp_mismatch")
    if (
        control_rate
        and start is not None
        and len(execution_timestamps) == SC3_RETAINED_ACTION_COUNT
    ):
        expected_execution_timestamps = [
            start + index / control_rate for index in range(SC3_RETAINED_ACTION_COUNT)
        ]
        if any(
            abs(observed - expected) > 1e-6
            for observed, expected in zip(execution_timestamps, expected_execution_timestamps)
        ):
            blockers.append("horizon_execution_timestamps_do_not_match_control_rate")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "sc3_horizon_execution_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "proposed_action_count": len(proposed),
        "predicted_action_count": len(predicted),
        "retained_action_count": len(retained),
        "executed_action_count": len(executed),
        "discarded_prediction_count": len(discarded),
        "requery_timestamp_sec": requery,
        "blockers": blockers,
    }


def validate_checkpoint_attestation(attestation: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if attestation.get("status") != "attested":
        blockers.append("sc3_checkpoint_not_attested")
    if attestation.get("base_checkpoint_only") is not False:
        blockers.append("base_checkpoint_without_sc3_finetuning")
    for key in (
        "checkpoint_sha256",
        "training_dataset_sha256",
        "training_split_sha256",
        "training_objective_sha256",
        "trainer_code_sha256",
    ):
        if not _sha256(attestation.get(key)):
            blockers.append(f"sc3_checkpoint_{key}_missing_or_invalid")
    for ref_key, digest_key in (
        ("checkpoint_artifact", "checkpoint_sha256"),
        ("training_dataset_manifest_artifact", "training_dataset_sha256"),
        ("training_split_manifest_artifact", "training_split_sha256"),
        ("training_objective_artifact", "training_objective_sha256"),
        ("trainer_code_artifact", "trainer_code_sha256"),
    ):
        ref = _mapping(attestation.get(ref_key))
        blockers.extend(_validate_artifact_ref(ref, prefix=f"sc3_checkpoint_{ref_key}"))
        if (
            ref
            and _string(ref.get("sha256")).lower() != _string(attestation.get(digest_key)).lower()
        ):
            blockers.append(f"sc3_checkpoint_{ref_key}_digest_attestation_mismatch")
    signature = _mapping(attestation.get("attestation_signature"))
    blockers.extend(
        _verify_ed25519_attestation(
            signature,
            signed_payload={
                key: value for key, value in attestation.items() if key != "attestation_signature"
            },
            prefix="sc3_checkpoint_attestation_signature",
            trusted_public_key_sha256_env=(SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256_ENV),
        )
    )
    modes = set(str(item) for item in attestation.get("trained_modes", []) or [])
    if modes != SC3_REQUIRED_TRAINING_MODES:
        blockers.append("sc3_checkpoint_training_modes_incomplete")
    probes = _rows(attestation.get("golden_functional_probes"))
    passed_modes: set[str] = set()
    seen_probe_ids: set[str] = set()
    for index, row in enumerate(probes):
        mode = _string(row.get("mode"))
        probe_id = _string(row.get("probe_id"))
        input_ref = _mapping(row.get("input_artifact"))
        output_ref = _mapping(row.get("output_artifact"))
        probe_blockers = [
            *_validate_artifact_ref(input_ref, prefix=f"sc3_checkpoint_golden_probe_input:{index}"),
            *_validate_artifact_ref(
                output_ref, prefix=f"sc3_checkpoint_golden_probe_output:{index}"
            ),
        ]
        if _string(input_ref.get("sha256")) != _string(row.get("input_sha256")):
            probe_blockers.append(f"sc3_checkpoint_golden_probe_input_digest_mismatch:{index}")
        if _string(output_ref.get("sha256")) != _string(row.get("output_sha256")):
            probe_blockers.append(f"sc3_checkpoint_golden_probe_output_digest_mismatch:{index}")
        if not probe_id or probe_id in seen_probe_ids:
            probe_blockers.append(f"sc3_checkpoint_golden_probe_id_missing_or_duplicate:{index}")
        seen_probe_ids.add(probe_id)
        input_payload: dict[str, Any] = {}
        output_payload: dict[str, Any] = {}
        try:
            input_payload = _mapping(
                json.loads(Path(_string(input_ref.get("path"))).read_text(encoding="utf-8"))
            )
            output_payload = _mapping(
                json.loads(Path(_string(output_ref.get("path"))).read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            probe_blockers.append(f"sc3_checkpoint_golden_probe_payload_invalid:{index}")
        input_values = input_payload.get("input_values")
        input_numbers = (
            [_number(value) for value in input_values]
            if isinstance(input_values, Sequence)
            and not isinstance(input_values, (str, bytes, bytearray))
            else []
        )
        if not (
            input_payload.get("schema_version") == "sc3_golden_probe_input.v1"
            and input_payload.get("mode") == mode
            and input_payload.get("probe_id") == probe_id
            and input_numbers
            and all(value is not None for value in input_numbers)
        ):
            probe_blockers.append(f"sc3_checkpoint_golden_probe_input_content_invalid:{index}")
        output_key = {
            "forward_dynamics": "predicted_next_state",
            "inverse_dynamics": "predicted_action_7d",
            "cross_view": "predicted_cross_view_embedding",
        }.get(mode, "")
        output_values = output_payload.get(output_key) if output_key else None
        output_numbers = (
            [_number(value) for value in output_values]
            if isinstance(output_values, Sequence)
            and not isinstance(output_values, (str, bytes, bytearray))
            else []
        )
        expected_output_length = 7 if mode == "inverse_dynamics" else None
        if not (
            output_payload.get("schema_version") == "sc3_golden_probe_output.v1"
            and output_payload.get("mode") == mode
            and output_payload.get("probe_id") == probe_id
            and output_payload.get("input_sha256") == input_ref.get("sha256")
            and output_payload.get("status") == "completed"
            and output_numbers
            and all(value is not None for value in output_numbers)
            and (expected_output_length is None or len(output_numbers) == expected_output_length)
        ):
            probe_blockers.append(f"sc3_checkpoint_golden_probe_output_content_invalid:{index}")
        blockers.extend(probe_blockers)
        if row.get("status") == "passed" and not probe_blockers and mode:
            passed_modes.add(mode)
    if passed_modes != SC3_REQUIRED_TRAINING_MODES:
        blockers.append("sc3_checkpoint_golden_functional_probes_incomplete")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "sc3_checkpoint_attestation_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "trained_modes": sorted(modes),
        "passed_probe_modes": sorted(passed_modes),
        "blockers": blockers,
    }


def _pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a) * sum(value * value for value in centered_b)
    )
    if denominator <= 0.0:
        return None
    return sum(left * right for left, right in zip(centered_a, centered_b)) / denominator


def _mean_maximum_rank_violation(
    predicted_rates: Sequence[float], actual_rates: Sequence[float]
) -> float:
    maximum_violations: list[float] = []
    for index, (predicted, actual) in enumerate(zip(predicted_rates, actual_rates)):
        maximum = 0.0
        for other_index, (other_predicted, other_actual) in enumerate(
            zip(predicted_rates, actual_rates)
        ):
            if index == other_index:
                continue
            if (predicted > other_predicted) != (actual > other_actual):
                maximum = max(maximum, abs(actual - other_actual))
        maximum_violations.append(maximum)
    return sum(maximum_violations) / len(maximum_violations)


def _load_json_artifact(
    ref_value: Any,
    *,
    prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    ref = _mapping(ref_value)
    blockers = _validate_artifact_ref(ref, prefix=prefix)
    payload: dict[str, Any] = {}
    path = Path(_string(ref.get("path"))).expanduser()
    if path.is_file():
        try:
            payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"{prefix}_json_invalid")
    return payload, blockers


def _validate_ood_split_provenance(
    *,
    axis: str,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    registered_policies = [
        {
            "policy_id": _string(row.get("policy_id")),
            "policy_checkpoint_sha256": _string(row.get("policy_checkpoint_sha256")).lower(),
            "policy_family_id": _string(row.get("policy_family_id")),
        }
        for row in _rows(result.get("registered_policies"))
    ]
    registered_policy_ids = [row["policy_id"] for row in registered_policies]
    registered_checkpoint_sha256s = [row["policy_checkpoint_sha256"] for row in registered_policies]
    if (
        len(registered_policies) < SC3_MIN_OOD_POLICY_COUNT
        or any(
            not row["policy_id"]
            or not _sha256(row["policy_checkpoint_sha256"])
            or not row["policy_family_id"]
            for row in registered_policies
        )
        or len(set(registered_policy_ids)) != len(registered_policy_ids)
        or registered_policy_ids != sorted(registered_policy_ids)
    ):
        blockers.append(f"ood_axis_registered_policy_manifest_invalid:{axis}")
    if len(set(registered_checkpoint_sha256s)) < SC3_MIN_OOD_POLICY_COUNT:
        blockers.append(
            f"ood_axis_distinct_policy_checkpoint_count_lt_{SC3_MIN_OOD_POLICY_COUNT}:{axis}"
        )
    registered_uncertainty_method = _string(result.get("registered_uncertainty_method"))
    if registered_uncertainty_method != SC3_OOD_UNCERTAINTY_METHOD:
        blockers.append(f"ood_axis_registered_uncertainty_method_invalid:{axis}")
    context: dict[str, Any] = {
        "train_group_ids": _string_list(result.get("train_group_ids")),
        "heldout_group_ids": _string_list(result.get("heldout_group_ids")),
        "train_source_ids": _string_list(result.get("train_source_ids")),
        "heldout_source_ids": _string_list(result.get("heldout_source_ids")),
        "train_split_sha256": _string(result.get("train_split_sha256")).lower(),
        "heldout_split_sha256": _string(result.get("heldout_split_sha256")).lower(),
        "source_manifest_sha256": _string(result.get("source_manifest_sha256")).lower(),
        "decision_thresholds_sha256": _string(result.get("decision_thresholds_sha256")).lower(),
        "registered_policies": registered_policies,
        "registered_policy_by_id": {
            row["policy_id"]: row for row in registered_policies if row["policy_id"]
        },
        "registered_uncertainty_method": registered_uncertainty_method,
    }
    for name in (
        "train_group_ids",
        "heldout_group_ids",
        "train_source_ids",
        "heldout_source_ids",
    ):
        values = context[name]
        if not values or len(set(values)) != len(values):
            blockers.append(f"ood_axis_{name}_missing_or_duplicate:{axis}")
    if set(context["train_group_ids"]) & set(context["heldout_group_ids"]):
        blockers.append(f"ood_axis_train_heldout_group_overlap:{axis}")
    if set(context["train_source_ids"]) & set(context["heldout_source_ids"]):
        blockers.append(f"ood_axis_train_heldout_source_overlap:{axis}")
    if not _sha256(context["decision_thresholds_sha256"]):
        blockers.append(f"ood_axis_decision_thresholds_sha256_invalid:{axis}")

    source_payload, source_blockers = _load_json_artifact(
        result.get("source_manifest_artifact"),
        prefix=f"ood_axis_source_manifest_artifact:{axis}",
    )
    blockers.extend(source_blockers)
    source_ref = _mapping(result.get("source_manifest_artifact"))
    if not (
        _sha256(context["source_manifest_sha256"])
        and _string(source_ref.get("sha256")).lower() == context["source_manifest_sha256"]
        and source_payload.get("schema_version") == "sc3_ood_axis_source_manifest.v2"
        and source_payload.get("axis") == axis
        and _string_list(source_payload.get("train_source_ids")) == context["train_source_ids"]
        and _string_list(source_payload.get("heldout_source_ids")) == context["heldout_source_ids"]
        and _rows(source_payload.get("registered_policies")) == context["registered_policies"]
        and source_payload.get("registered_uncertainty_method")
        == context["registered_uncertainty_method"]
    ):
        blockers.append(f"ood_axis_source_manifest_binding_invalid:{axis}")

    for split_name in ("train", "heldout"):
        payload, artifact_blockers = _load_json_artifact(
            result.get(f"{split_name}_split_artifact"),
            prefix=f"ood_axis_{split_name}_split_artifact:{axis}",
        )
        blockers.extend(artifact_blockers)
        ref = _mapping(result.get(f"{split_name}_split_artifact"))
        digest = context[f"{split_name}_split_sha256"]
        if not (
            _sha256(digest)
            and _string(ref.get("sha256")).lower() == digest
            and payload.get("schema_version") == "sc3_ood_axis_split.v1"
            and payload.get("axis") == axis
            and payload.get("split") == split_name
            and _string_list(payload.get("group_ids")) == context[f"{split_name}_group_ids"]
            and _string_list(payload.get("source_ids")) == context[f"{split_name}_source_ids"]
            and _string(payload.get("source_manifest_sha256")).lower()
            == context["source_manifest_sha256"]
        ):
            blockers.append(f"ood_axis_{split_name}_split_binding_invalid:{axis}")
    return context, sorted(set(blockers))


OOD_REPLICATE_BINDING_FIELDS = (
    "axis",
    "policy_id",
    "policy_checkpoint_sha256",
    "policy_family_id",
    "condition_id",
    "heldout_group_id",
    "source_id",
    "replicate_id",
    "replicate_seed",
    "predicted_success",
    "actual_success",
    "abstained",
    "train_split_sha256",
    "heldout_split_sha256",
    "source_manifest_sha256",
    "decision_thresholds_sha256",
)


def _ood_metrics_from_replicates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policy_ids = sorted({_string(row.get("policy_id")) for row in rows})
    accepted_by_policy: dict[str, list[Mapping[str, Any]]] = {
        policy_id: [
            row
            for row in rows
            if _string(row.get("policy_id")) == policy_id and row.get("abstained") is False
        ]
        for policy_id in policy_ids
    }
    predicted_rates: list[float] = []
    actual_rates: list[float] = []
    complete_policy_rates = True
    for policy_id in policy_ids:
        accepted = accepted_by_policy[policy_id]
        if not accepted:
            complete_policy_rates = False
            continue
        predicted_rates.append(
            sum(1 for row in accepted if row.get("predicted_success") is True) / len(accepted)
        )
        actual_rates.append(
            sum(1 for row in accepted if row.get("actual_success") is True) / len(accepted)
        )
    abstention_count = sum(1 for row in rows if row.get("abstained") is True)
    metrics: dict[str, Any] = {
        "sample_count": len(rows),
        "accepted_sample_count": len(rows) - abstention_count,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_count / max(1, len(rows)),
        "policy_count": len(policy_ids),
        "distinct_policy_checkpoint_count": len(
            {
                _string(row.get("policy_checkpoint_sha256")).lower()
                for row in rows
                if _sha256(row.get("policy_checkpoint_sha256"))
            }
        ),
        "policy_family_count": len(
            {
                _string(row.get("policy_family_id"))
                for row in rows
                if _string(row.get("policy_family_id"))
            }
        ),
        "condition_count": len({_string(row.get("condition_id")) for row in rows}),
    }
    if complete_policy_rates and predicted_rates:
        pearson = _pearson(predicted_rates, actual_rates)
        if pearson is not None:
            metrics["pearson_success_rate_correlation"] = pearson
        metrics["mean_absolute_success_rate_error"] = sum(
            abs(predicted - actual) for predicted, actual in zip(predicted_rates, actual_rates)
        ) / len(predicted_rates)
        metrics["mean_maximum_rank_violation"] = _mean_maximum_rank_violation(
            predicted_rates, actual_rates
        )
    return metrics


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    values: Sequence[float],
    *,
    estimate: float,
    minimum: float,
    maximum: float,
) -> list[float]:
    lower = min(estimate, _quantile(values, 0.025))
    upper = max(estimate, _quantile(values, 0.975))
    return [
        round(max(minimum, lower), 6),
        round(min(maximum, upper), 6),
    ]


def _hierarchical_ood_bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    seed_material: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: set[str] = set()
    nested: dict[
        str,
        dict[str, dict[str, dict[int, Mapping[str, Any]]]],
    ] = {}
    for row in rows:
        group_id = _string(row.get("heldout_group_id"))
        condition_id = _string(row.get("condition_id"))
        policy_id = _string(row.get("policy_id"))
        seed = row.get("replicate_seed")
        if (
            group_id
            and condition_id
            and policy_id
            and isinstance(seed, int)
            and not isinstance(seed, bool)
        ):
            nested.setdefault(group_id, {}).setdefault(condition_id, {}).setdefault(policy_id, {})[
                seed
            ] = row
    group_ids = sorted(nested)
    policy_ids = sorted({_string(row.get("policy_id")) for row in rows})
    matched_seeds: dict[tuple[str, str], list[int]] = {}
    for group_id in group_ids:
        for condition_id, policies in sorted(nested[group_id].items()):
            if set(policies) != set(policy_ids):
                blockers.add(
                    f"ood_axis_bootstrap_policy_coverage_mismatch:{axis}:{group_id}:{condition_id}"
                )
                continue
            policy_seed_sets = [set(policies[policy_id]) for policy_id in policy_ids]
            if any(seeds != policy_seed_sets[0] for seeds in policy_seed_sets[1:]):
                blockers.add(
                    f"ood_axis_bootstrap_seed_sets_not_matched:{axis}:{group_id}:{condition_id}"
                )
            common_seeds = sorted(set.intersection(*policy_seed_sets))
            if not common_seeds:
                blockers.add(
                    f"ood_axis_bootstrap_common_seed_set_missing:{axis}:{group_id}:{condition_id}"
                )
                continue
            matched_seeds[(group_id, condition_id)] = common_seeds
    if not group_ids:
        blockers.add(f"ood_axis_bootstrap_heldout_groups_missing:{axis}")
    seed_sha256 = _canonical_sha256(
        {
            **dict(seed_material),
            "registered_uncertainty_method": SC3_OOD_UNCERTAINTY_METHOD,
            "bootstrap_cluster_levels": list(SC3_OOD_BOOTSTRAP_CLUSTER_LEVELS),
        }
    )
    generator = random.Random(int(seed_sha256[:16], 16))
    samples: dict[str, list[float]] = {
        "pearson_success_rate_correlation": [],
        "mean_maximum_rank_violation": [],
        "mean_absolute_success_rate_error": [],
        "abstention_rate": [],
    }
    for _ in range(SC3_OOD_BOOTSTRAP_SAMPLE_COUNT):
        sampled_rows: list[Mapping[str, Any]] = []
        sampled_group_ids = (
            [group_ids[generator.randrange(len(group_ids))] for _draw in group_ids]
            if group_ids
            else []
        )
        for group_id in sampled_group_ids:
            condition_ids = sorted(nested[group_id])
            if not condition_ids:
                continue
            sampled_condition_ids = [
                condition_ids[generator.randrange(len(condition_ids))] for _draw in condition_ids
            ]
            for condition_id in sampled_condition_ids:
                seeds = matched_seeds.get((group_id, condition_id), [])
                if not seeds:
                    continue
                for _draw in seeds:
                    selected_seed = seeds[generator.randrange(len(seeds))]
                    sampled_rows.extend(
                        nested[group_id][condition_id][policy_id][selected_seed]
                        for policy_id in policy_ids
                    )
        metrics = _ood_metrics_from_replicates(sampled_rows)
        for metric_name in samples:
            value = _number(metrics.get(metric_name))
            if value is not None:
                samples[metric_name].append(value)
    if len(samples["pearson_success_rate_correlation"]) < (SC3_OOD_BOOTSTRAP_SAMPLE_COUNT // 2):
        blockers.add(f"ood_axis_bootstrap_pearson_not_stable:{axis}")
    point = _ood_metrics_from_replicates(rows)
    intervals: dict[str, Any] = {
        "bootstrap_method": SC3_OOD_UNCERTAINTY_METHOD,
        "bootstrap_cluster_levels": list(SC3_OOD_BOOTSTRAP_CLUSTER_LEVELS),
        "bootstrap_sample_count": SC3_OOD_BOOTSTRAP_SAMPLE_COUNT,
        "bootstrap_seed_sha256": seed_sha256,
        "abstention_interval_method": SC3_OOD_UNCERTAINTY_METHOD,
    }
    interval_specs = (
        (
            "pearson_95_ci",
            "pearson_success_rate_correlation",
            -1.0,
            1.0,
        ),
        ("mmrv_95_ci", "mean_maximum_rank_violation", 0.0, 1.0),
        (
            "error_95_ci",
            "mean_absolute_success_rate_error",
            0.0,
            1.0,
        ),
        ("abstention_95_ci", "abstention_rate", 0.0, 1.0),
    )
    for interval_name, metric_name, minimum, maximum in interval_specs:
        estimate = _number(point.get(metric_name))
        values = samples[metric_name]
        if estimate is None or not values:
            blockers.add(f"ood_axis_bootstrap_metric_missing:{axis}:{metric_name}")
            continue
        intervals[interval_name] = _bootstrap_interval(
            values,
            estimate=estimate,
            minimum=minimum,
            maximum=maximum,
        )
    return intervals, sorted(blockers)


def _recomputed_ood_axis_metrics(
    *,
    axis: str,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    context, blockers = _validate_ood_split_provenance(axis=axis, result=result)
    raw_ref = _mapping(result.get("raw_rows_artifact"))
    payload, raw_artifact_blockers = _load_json_artifact(
        raw_ref,
        prefix=f"ood_axis_raw_rows_artifact:{axis}",
    )
    blockers.extend(raw_artifact_blockers)
    if not (
        payload.get("schema_version") == "sc3_ood_axis_raw_replicates.v3"
        and payload.get("axis") == axis
        and payload.get("train_split_sha256") == context.get("train_split_sha256")
        and payload.get("heldout_split_sha256") == context.get("heldout_split_sha256")
        and payload.get("source_manifest_sha256") == context.get("source_manifest_sha256")
        and payload.get("decision_thresholds_sha256") == context.get("decision_thresholds_sha256")
        and _string_list(payload.get("heldout_group_ids")) == context.get("heldout_group_ids")
        and _string_list(payload.get("heldout_source_ids")) == context.get("heldout_source_ids")
        and _rows(payload.get("registered_policies")) == context.get("registered_policies")
        and payload.get("registered_uncertainty_method")
        == context.get("registered_uncertainty_method")
    ):
        blockers.append(f"ood_axis_raw_rows_content_binding_invalid:{axis}")
    rows = _rows(payload.get("rows"))
    if not rows:
        blockers.append(f"ood_axis_raw_replicate_rows_missing:{axis}")

    validated_rows: list[dict[str, Any]] = []
    seen_replicate_keys: set[tuple[str, str, int]] = set()
    seen_replicate_ids: set[str] = set()
    seen_artifact_paths: set[str] = set()
    seen_artifact_digests: set[str] = set()
    for index, row in enumerate(rows):
        row_blockers: list[str] = []
        policy_id = _string(row.get("policy_id"))
        policy_checkpoint_sha256 = _string(row.get("policy_checkpoint_sha256")).lower()
        policy_family_id = _string(row.get("policy_family_id"))
        registered_policy = _mapping(
            _mapping(context.get("registered_policy_by_id")).get(policy_id)
        )
        condition_id = _string(row.get("condition_id"))
        heldout_group_id = _string(row.get("heldout_group_id"))
        source_id = _string(row.get("source_id"))
        replicate_id = _string(row.get("replicate_id"))
        seed = row.get("replicate_seed")
        if not (
            registered_policy
            and registered_policy.get("policy_checkpoint_sha256") == policy_checkpoint_sha256
            and registered_policy.get("policy_family_id") == policy_family_id
        ):
            row_blockers.append(f"ood_axis_policy_identity_binding_invalid:{axis}:{index}")
        if not (
            policy_id
            and _sha256(policy_checkpoint_sha256)
            and policy_family_id
            and registered_policy.get("policy_checkpoint_sha256") == policy_checkpoint_sha256
            and registered_policy.get("policy_family_id") == policy_family_id
            and condition_id
            and heldout_group_id in set(context.get("heldout_group_ids") or [])
            and source_id in set(context.get("heldout_source_ids") or [])
            and replicate_id
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(row.get("predicted_success"), bool)
            and isinstance(row.get("actual_success"), bool)
            and isinstance(row.get("abstained"), bool)
            and row.get("axis") == axis
            and row.get("train_split_sha256") == context.get("train_split_sha256")
            and row.get("heldout_split_sha256") == context.get("heldout_split_sha256")
            and row.get("source_manifest_sha256") == context.get("source_manifest_sha256")
            and row.get("decision_thresholds_sha256") == context.get("decision_thresholds_sha256")
        ):
            row_blockers.append(f"ood_axis_raw_replicate_invalid:{axis}:{index}")
        if isinstance(seed, int) and not isinstance(seed, bool):
            replicate_key = (policy_id, condition_id, seed)
            if replicate_key in seen_replicate_keys:
                row_blockers.append(f"ood_axis_raw_replicate_key_duplicate:{axis}:{index}")
            seen_replicate_keys.add(replicate_key)
        if replicate_id in seen_replicate_ids:
            row_blockers.append(f"ood_axis_raw_replicate_id_duplicate:{axis}:{index}")
        seen_replicate_ids.add(replicate_id)

        evidence_ref = _mapping(row.get("evidence_artifact"))
        evidence_path_text = _string(evidence_ref.get("path"))
        evidence_path = str(Path(evidence_path_text).expanduser().resolve())
        evidence_digest = _string(evidence_ref.get("sha256")).lower()
        if evidence_path in seen_artifact_paths:
            row_blockers.append(f"ood_axis_replicate_evidence_path_reused:{axis}:{index}")
        if evidence_digest in seen_artifact_digests:
            row_blockers.append(f"ood_axis_replicate_evidence_digest_reused:{axis}:{index}")
        seen_artifact_paths.add(evidence_path)
        seen_artifact_digests.add(evidence_digest)
        evidence_payload, evidence_blockers = _load_json_artifact(
            evidence_ref,
            prefix=f"ood_axis_replicate_evidence_artifact:{axis}:{index}",
        )
        row_blockers.extend(evidence_blockers)
        expected_payload = {
            "schema_version": "sc3_ood_replicate_evidence.v2",
            **{field: row.get(field) for field in OOD_REPLICATE_BINDING_FIELDS},
        }
        if evidence_payload != expected_payload:
            row_blockers.append(f"ood_axis_replicate_evidence_binding_invalid:{axis}:{index}")
        row_blockers.extend(
            _verify_ed25519_attestation(
                _mapping(row.get("evidence_attestation")),
                signed_payload=evidence_payload,
                prefix=f"ood_axis_replicate_evidence_attestation:{axis}:{index}",
                trusted_public_key_sha256_env=(SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256_ENV),
            )
        )
        blockers.extend(row_blockers)
        if not row_blockers:
            validated_rows.append(evidence_payload)

    policy_ids = sorted({_string(row.get("policy_id")) for row in validated_rows})
    registered_policy_ids = sorted(_mapping(context.get("registered_policy_by_id")))
    if policy_ids != registered_policy_ids:
        blockers.append(f"ood_axis_registered_policy_coverage_mismatch:{axis}")
    if len(policy_ids) < SC3_MIN_OOD_POLICY_COUNT:
        blockers.append(f"ood_axis_policy_count_lt_{SC3_MIN_OOD_POLICY_COUNT}:{axis}")
    checkpoint_sha256s = {
        _string(row.get("policy_checkpoint_sha256")).lower()
        for row in validated_rows
        if _sha256(row.get("policy_checkpoint_sha256"))
    }
    if len(checkpoint_sha256s) < SC3_MIN_OOD_POLICY_COUNT:
        blockers.append(
            f"ood_axis_distinct_policy_checkpoint_count_lt_{SC3_MIN_OOD_POLICY_COUNT}:{axis}"
        )
    policy_checkpoint_sets: dict[str, set[str]] = {}
    policy_family_sets: dict[str, set[str]] = {}
    checkpoint_policy_ids: dict[str, set[str]] = {}
    for row in validated_rows:
        policy_id = _string(row.get("policy_id"))
        checkpoint_sha256 = _string(row.get("policy_checkpoint_sha256")).lower()
        family_id = _string(row.get("policy_family_id"))
        policy_checkpoint_sets.setdefault(policy_id, set()).add(checkpoint_sha256)
        policy_family_sets.setdefault(policy_id, set()).add(family_id)
        checkpoint_policy_ids.setdefault(checkpoint_sha256, set()).add(policy_id)
    for policy_id in policy_ids:
        if len(policy_checkpoint_sets.get(policy_id, set())) != 1:
            blockers.append(f"ood_axis_policy_checkpoint_identity_changed:{axis}:{policy_id}")
        if len(policy_family_sets.get(policy_id, set())) != 1:
            blockers.append(f"ood_axis_policy_family_identity_changed:{axis}:{policy_id}")
    if any(len(ids) != 1 for ids in checkpoint_policy_ids.values()):
        blockers.append(f"ood_axis_policy_checkpoint_alias_detected:{axis}")
    nested: dict[str, dict[str, set[int]]] = {}
    condition_groups: dict[str, set[str]] = {}
    condition_sources: dict[str, set[str]] = {}
    for row in validated_rows:
        condition_id = _string(row.get("condition_id"))
        policy_id = _string(row.get("policy_id"))
        seed = int(row["replicate_seed"])
        nested.setdefault(condition_id, {}).setdefault(policy_id, set()).add(seed)
        condition_groups.setdefault(condition_id, set()).add(_string(row.get("heldout_group_id")))
        condition_sources.setdefault(condition_id, set()).add(_string(row.get("source_id")))
    minimum_matched_seed_count = min(
        (len(seeds) for policies in nested.values() for seeds in policies.values()),
        default=0,
    )
    for condition_id, policies in nested.items():
        if set(policies) != set(policy_ids):
            blockers.append(f"ood_axis_condition_policy_coverage_mismatch:{axis}:{condition_id}")
        seed_sets = list(policies.values())
        if any(len(seeds) < SC3_MIN_OOD_REPLICATES_PER_POLICY_CONDITION for seeds in seed_sets):
            blockers.append(
                "ood_axis_policy_condition_replicates_lt_"
                f"{SC3_MIN_OOD_REPLICATES_PER_POLICY_CONDITION}:{axis}:{condition_id}"
            )
        if seed_sets and any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            blockers.append(f"ood_axis_condition_seed_sets_not_matched:{axis}:{condition_id}")
        if len(condition_groups.get(condition_id, set())) != 1:
            blockers.append(f"ood_axis_condition_heldout_group_ambiguous:{axis}:{condition_id}")
        if len(condition_sources.get(condition_id, set())) != 1:
            blockers.append(f"ood_axis_condition_heldout_source_ambiguous:{axis}:{condition_id}")
    if {_string(row.get("heldout_group_id")) for row in validated_rows} != set(
        context.get("heldout_group_ids") or []
    ):
        blockers.append(f"ood_axis_heldout_group_coverage_mismatch:{axis}")
    if {_string(row.get("source_id")) for row in validated_rows} != set(
        context.get("heldout_source_ids") or []
    ):
        blockers.append(f"ood_axis_heldout_source_coverage_mismatch:{axis}")

    metrics = _ood_metrics_from_replicates(validated_rows)
    metrics["minimum_matched_seed_count"] = minimum_matched_seed_count
    metrics["registered_uncertainty_method"] = context.get("registered_uncertainty_method")
    pearson = _number(metrics.get("pearson_success_rate_correlation"))
    if pearson is None:
        blockers.append(f"ood_axis_raw_pearson_not_computable:{axis}")
    intervals, interval_blockers = _hierarchical_ood_bootstrap_intervals(
        validated_rows,
        axis=axis,
        seed_material={
            "axis": axis,
            "train_split_sha256": context.get("train_split_sha256"),
            "heldout_split_sha256": context.get("heldout_split_sha256"),
            "source_manifest_sha256": context.get("source_manifest_sha256"),
            "raw_rows_sha256": raw_ref.get("sha256"),
            "registered_uncertainty_method": context.get("registered_uncertainty_method"),
        },
    )
    blockers.extend(interval_blockers)
    metrics.update(intervals)
    return metrics, sorted(set(blockers))


def _declared_interval_matches(value: Any, expected: Any) -> bool:
    declared = (
        [_number(item) for item in value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        else []
    )
    computed = (
        [_number(item) for item in expected]
        if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray))
        else []
    )
    return bool(
        len(declared) == len(computed) == 2
        and all(item is not None for item in (*declared, *computed))
        and all(abs(float(left) - float(right)) <= 1e-6 for left, right in zip(declared, computed))
    )


def _finite_interval_bounds(
    value: Any,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, float] | None:
    bounds = (
        [_number(item) for item in value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        else []
    )
    if not (
        len(bounds) == 2
        and bounds[0] is not None
        and bounds[1] is not None
        and minimum <= bounds[0] <= bounds[1] <= maximum
    ):
        return None
    return float(bounds[0]), float(bounds[1])


def validate_ood_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    axes = set(str(item) for item in registry.get("frozen_axes", []) or [])
    if axes != SC3_OOD_AXES:
        blockers.append("frozen_ood_axes_incomplete_or_changed")
    if not _sha256(registry.get("registry_sha256")):
        blockers.append("frozen_ood_registry_sha256_missing_or_invalid")
    elif _string(registry.get("registry_sha256")).lower() != _canonical_sha256(
        registry, exclude=("registry_sha256",)
    ):
        blockers.append("frozen_ood_registry_sha256_mismatch")
    rows = _rows(registry.get("leave_one_group_results"))
    row_axes = [_string(row.get("axis")) for row in rows]
    if len(rows) != len(SC3_OOD_AXES) or set(row_axes) != SC3_OOD_AXES:
        blockers.append("ood_axis_results_must_match_frozen_axes_exactly_once")
    if len(set(row_axes)) != len(row_axes):
        blockers.append("ood_axis_results_duplicate")
    by_axis = {_string(row.get("axis")): row for row in rows}
    thresholds = _mapping(registry.get("decision_thresholds"))
    min_pearson = _number(thresholds.get("min_pearson_success_rate_correlation"))
    max_mmrv = _number(thresholds.get("max_mean_maximum_rank_violation"))
    max_error = _number(thresholds.get("max_mean_absolute_success_rate_error"))
    max_abstention = _number(thresholds.get("max_abstention_rate"))
    registered_uncertainty_method = _string(thresholds.get("registered_uncertainty_method"))
    decision_thresholds_sha256 = _canonical_sha256(thresholds)
    if not (
        min_pearson is not None
        and -1.0 <= min_pearson <= 1.0
        and max_mmrv is not None
        and 0.0 <= max_mmrv <= 1.0
        and max_error is not None
        and 0.0 <= max_error <= 1.0
        and max_abstention is not None
        and 0.0 <= max_abstention <= 1.0
        and registered_uncertainty_method == SC3_OOD_UNCERTAINTY_METHOD
    ):
        blockers.append("ood_decision_thresholds_missing_or_invalid")
    recomputed_by_axis: dict[str, dict[str, Any]] = {}
    for axis in sorted(SC3_OOD_AXES):
        row = _mapping(by_axis.get(axis))
        if not row:
            blockers.append(f"ood_axis_result_missing:{axis}")
            continue
        if row.get("decision_thresholds_sha256") != decision_thresholds_sha256:
            blockers.append(f"ood_axis_decision_thresholds_digest_mismatch:{axis}")
        if row.get("registered_uncertainty_method") != registered_uncertainty_method:
            blockers.append(f"ood_axis_registered_uncertainty_method_mismatch:{axis}")
        recomputed, raw_row_blockers = _recomputed_ood_axis_metrics(
            axis=axis,
            result=row,
        )
        recomputed_by_axis[axis] = recomputed
        blockers.extend(raw_row_blockers)
        for count_name in (
            "sample_count",
            "accepted_sample_count",
            "abstention_count",
            "policy_count",
            "distinct_policy_checkpoint_count",
            "policy_family_count",
            "condition_count",
            "minimum_matched_seed_count",
            "bootstrap_sample_count",
        ):
            declared = row.get(count_name)
            computed = recomputed.get(count_name)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or computed is None
                or declared != computed
            ):
                blockers.append(
                    f"ood_axis_declared_count_does_not_match_raw_rows:{axis}:{count_name}"
                )
        declared_metrics = {
            "pearson_success_rate_correlation": _number(
                row.get("pearson_success_rate_correlation")
            ),
            "mean_maximum_rank_violation": _number(row.get("mean_maximum_rank_violation")),
            "mean_absolute_success_rate_error": _number(
                row.get("mean_absolute_success_rate_error")
            ),
            "abstention_rate": _number(row.get("abstention_rate")),
        }
        for metric_name, declared in declared_metrics.items():
            computed = _number(recomputed.get(metric_name))
            if declared is None or computed is None or abs(declared - computed) > 1e-6:
                blockers.append(
                    f"ood_axis_declared_metric_does_not_match_raw_rows:{axis}:{metric_name}"
                )
        for interval_name in (
            "pearson_95_ci",
            "mmrv_95_ci",
            "error_95_ci",
            "abstention_95_ci",
        ):
            if not _declared_interval_matches(
                row.get(interval_name), recomputed.get(interval_name)
            ):
                blockers.append(
                    f"ood_axis_declared_interval_does_not_match_raw_rows:{axis}:{interval_name}"
                )
        for field_name in (
            "registered_uncertainty_method",
            "bootstrap_method",
            "bootstrap_cluster_levels",
            "bootstrap_seed_sha256",
            "abstention_interval_method",
        ):
            if row.get(field_name) != recomputed.get(field_name):
                blockers.append(
                    f"ood_axis_declared_uncertainty_method_mismatch:{axis}:{field_name}"
                )
        pearson = _number(recomputed.get("pearson_success_rate_correlation"))
        mmrv = _number(recomputed.get("mean_maximum_rank_violation"))
        error = _number(recomputed.get("mean_absolute_success_rate_error"))
        abstention = _number(recomputed.get("abstention_rate"))
        pearson_ci = _finite_interval_bounds(
            recomputed.get("pearson_95_ci"), minimum=-1.0, maximum=1.0
        )
        mmrv_ci = _finite_interval_bounds(recomputed.get("mmrv_95_ci"), minimum=0.0, maximum=1.0)
        error_ci = _finite_interval_bounds(recomputed.get("error_95_ci"), minimum=0.0, maximum=1.0)
        abstention_ci = _finite_interval_bounds(
            recomputed.get("abstention_95_ci"), minimum=0.0, maximum=1.0
        )
        computed_thresholds_passed = bool(
            pearson is not None
            and min_pearson is not None
            and pearson >= min_pearson
            and pearson_ci is not None
            and pearson_ci[0] >= min_pearson
            and mmrv is not None
            and max_mmrv is not None
            and mmrv <= max_mmrv
            and mmrv_ci is not None
            and mmrv_ci[1] <= max_mmrv
            and error is not None
            and max_error is not None
            and error <= max_error
            and error_ci is not None
            and error_ci[1] <= max_error
            and abstention is not None
            and max_abstention is not None
            and abstention <= max_abstention
            and abstention_ci is not None
            and abstention_ci[1] <= max_abstention
        )
        if row.get("thresholds_passed") is not computed_thresholds_passed:
            blockers.append(f"ood_axis_declared_threshold_result_mismatch:{axis}")
        if not computed_thresholds_passed:
            blockers.append(f"ood_axis_threshold_failed:{axis}")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "sc3_frozen_ood_validation.v3",
        "status": "validated" if not blockers else "blocked",
        "axes": sorted(axes),
        "per_axis_result_count": len(rows),
        "minimum_replicates_per_policy_condition": (SC3_MIN_OOD_REPLICATES_PER_POLICY_CONDITION),
        "minimum_policy_count": SC3_MIN_OOD_POLICY_COUNT,
        "registered_uncertainty_method": registered_uncertainty_method,
        "conservative_ci_thresholds_required": True,
        "recomputed_results_by_axis": recomputed_by_axis,
        "pooled_ood_headline_allowed": not blockers,
        "blockers": blockers,
    }


def validate_benchmark_cards(cards: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    sc3 = _mapping(cards.get("sc3_eval"))
    oscar = _mapping(cards.get("oscar"))
    if sc3.get("benchmark_family") != "sc3_eval":
        blockers.append("sc3_benchmark_card_missing_or_mislabeled")
    if oscar.get("benchmark_family") != "oscar":
        blockers.append("oscar_benchmark_card_missing_or_mislabeled")
    sc3_metrics = {_string(item).lower() for item in sc3.get("metric_names", []) or []}
    oscar_metrics = {_string(item).lower() for item in oscar.get("metric_names", []) or []}
    if not {
        "pearson_success_rate_correlation",
        "spearman_rank_correlation",
        "mean_maximum_rank_violation",
    }.issubset(sc3_metrics):
        blockers.append("sc3_correlation_metric_names_incomplete")
    if OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC not in oscar_metrics:
        blockers.append("oscar_success_rate_difference_pp_metric_missing")
    if "sisr_delta" in oscar_metrics:
        blockers.append("oscar_metric_mislabeled_as_sisr_delta")
    if oscar_metrics & {
        "pearson",
        "pearson_success_rate_correlation",
        "spearman",
        "spearman_rank_correlation",
        "srcc",
        "mmrv",
        "mean_maximum_rank_violation",
    }:
        blockers.append("sc3_metric_transferred_to_oscar_card")
    if sc3_metrics & {OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC, "mae"}:
        blockers.append("oscar_metric_transferred_to_sc3_card")
    for card_name, card in (("sc3", sc3), ("oscar", oscar)):
        for field in ("model_id", "protocol_id", "label_unit", "sample_unit"):
            if not _string(card.get(field)):
                blockers.append(f"{card_name}_benchmark_{field}_missing")
    if _string(sc3.get("model_id")) == _string(oscar.get("model_id")):
        blockers.append("sc3_and_oscar_model_ids_must_be_distinct")
    if _string(sc3.get("protocol_id")) == _string(oscar.get("protocol_id")):
        blockers.append("sc3_and_oscar_protocol_ids_must_be_distinct")
    if not (
        _string(sc3.get("label_unit")) == "criterion"
        and _string(sc3.get("sample_unit")) == "checkpoint_criterion"
    ):
        blockers.append("sc3_benchmark_units_must_be_criterion_checkpoint_criterion")
    if not (
        _string(oscar.get("label_unit")) == "episode"
        and _string(oscar.get("sample_unit")) == "rollout"
    ):
        blockers.append("oscar_benchmark_units_must_be_episode_rollout")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "sc3_benchmark_card_separation_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "metric_transfer_allowed": False,
        "blockers": blockers,
    }


def validate_anchor_artifacts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    prediction_authority_sha256 = _string(
        os.getenv(SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV)
    ).lower()
    outcome_authority_sha256 = _string(
        os.getenv(SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV)
    ).lower()
    if (
        _sha256(prediction_authority_sha256)
        and _sha256(outcome_authority_sha256)
        and prediction_authority_sha256 == outcome_authority_sha256
    ):
        blockers.append("accepted_anchor_prediction_outcome_authorities_not_separated")
    valid_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    seen_artifact_paths: set[str] = set()
    seen_artifact_digests: set[str] = set()
    policy_checkpoint_bindings: dict[str, set[tuple[str, str]]] = {}
    checkpoint_policy_bindings: dict[str, set[str]] = {}
    condition_descriptors: dict[str, set[tuple[str, ...]]] = {}
    registered_splits: set[str] = set()
    split_manifest_digests: set[str] = set()
    required_keys = (
        "policy_id",
        "checkpoint_id",
        "policy_checkpoint_sha256",
        "criterion_id",
        "registered_split",
        "split_manifest_id",
        "split_manifest_sha256",
        "task_family",
        "task_id",
        "scenario_eval_run_id",
        "scenario_variation_instance_id",
        "condition_id",
        "condition_source_id",
        "replicate_id",
    )
    artifact_join_fields = (*required_keys, "replicate_seed")
    for index, raw in enumerate(rows):
        row = dict(raw)
        key = tuple(_string(row.get(field)) for field in required_keys)
        if any(not value for value in key):
            blockers.append(f"accepted_anchor_join_key_incomplete:{index}")
            continue
        if key in seen_keys:
            blockers.append(f"accepted_anchor_join_key_duplicate:{index}")
            continue
        seen_keys.add(key)
        registered_split = _string(row.get("registered_split"))
        if registered_split not in {"test", "locked_test"}:
            blockers.append(f"accepted_anchor_registered_split_not_evaluation:{index}")
            continue
        registered_splits.add(registered_split)
        split_manifest_digest = _string(row.get("split_manifest_sha256")).lower()
        split_manifest_digests.add(split_manifest_digest)
        if not isinstance(row.get("predicted_success"), bool) or not isinstance(
            row.get("actual_success"), bool
        ):
            blockers.append(f"accepted_anchor_outcome_not_strict_boolean:{index}")
            continue
        row_blockers = [
            *_validate_artifact_ref(
                _mapping(row.get("split_manifest_artifact")),
                prefix=f"accepted_anchor_split_manifest_artifact:{index}",
            ),
            *_validate_artifact_ref(
                _mapping(row.get("policy_checkpoint_artifact")),
                prefix=f"accepted_anchor_policy_checkpoint_artifact:{index}",
            ),
            *_validate_artifact_ref(
                _mapping(row.get("prediction_artifact")),
                prefix=f"accepted_anchor_prediction_artifact:{index}",
            ),
            *_validate_artifact_ref(
                _mapping(row.get("outcome_artifact")),
                prefix=f"accepted_anchor_outcome_artifact:{index}",
            ),
        ]
        if row_blockers:
            blockers.extend(row_blockers)
            continue
        split_ref = _mapping(row.get("split_manifest_artifact"))
        checkpoint_ref = _mapping(row.get("policy_checkpoint_artifact"))
        split_payload: dict[str, Any] = {}
        try:
            split_payload = _mapping(
                json.loads(Path(_string(split_ref.get("path"))).read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            pass
        if not (
            _string(split_ref.get("sha256")).lower() == split_manifest_digest
            and split_payload.get("schema_version") == "sc3_anchor_split_manifest.v1"
            and split_payload.get("status") == "frozen"
            and split_payload.get("split_manifest_id") == row.get("split_manifest_id")
            and split_payload.get("registered_split") == registered_split
        ):
            blockers.append(f"accepted_anchor_split_manifest_binding_invalid:{index}")
            continue
        if (
            _string(checkpoint_ref.get("sha256")).lower()
            != _string(row.get("policy_checkpoint_sha256")).lower()
        ):
            blockers.append(f"accepted_anchor_policy_checkpoint_artifact_binding_invalid:{index}")
            continue
        artifact_values: list[tuple[str, Any, tuple[str, ...]]] = [
            (
                "prediction",
                row.get("predicted_success"),
                ("predicted_success", "success", "value"),
            ),
            (
                "outcome",
                row.get("actual_success"),
                ("actual_success", "success", "value"),
            ),
        ]
        artifact_content_valid = True
        for artifact_name, expected_value, keys in artifact_values:
            ref = _mapping(row.get(f"{artifact_name}_artifact"))
            artifact_path = str(Path(_string(ref.get("path"))).expanduser().resolve())
            artifact_digest = _string(ref.get("sha256")).lower()
            if artifact_path in seen_artifact_paths:
                blockers.append(f"accepted_anchor_artifact_path_reused:{artifact_name}:{index}")
                artifact_content_valid = False
            if artifact_digest in seen_artifact_digests:
                blockers.append(f"accepted_anchor_artifact_digest_reused:{artifact_name}:{index}")
                artifact_content_valid = False
            seen_artifact_paths.add(artifact_path)
            seen_artifact_digests.add(artifact_digest)
            try:
                payload = json.loads(Path(_string(ref.get("path"))).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                blockers.append(f"accepted_anchor_{artifact_name}_content_invalid:{index}")
                artifact_content_valid = False
                continue
            observed_value = None
            if isinstance(payload, Mapping):
                expected_schema = (
                    "sc3_anchor_prediction.v1"
                    if artifact_name == "prediction"
                    else "sc3_anchor_outcome.v1"
                )
                if payload.get("schema_version") != expected_schema:
                    blockers.append(f"accepted_anchor_{artifact_name}_schema_invalid:{index}")
                    artifact_content_valid = False
                for field in artifact_join_fields:
                    if payload.get(field) != row.get(field):
                        blockers.append(
                            f"accepted_anchor_{artifact_name}_join_key_mismatch:{field}:{index}"
                        )
                        artifact_content_valid = False
                authority_field = "authority_attestation"
                authority_validation = _verify_ed25519_attestation(
                    _mapping(payload.get(authority_field)),
                    signed_payload={
                        field: value for field, value in payload.items() if field != authority_field
                    },
                    prefix=f"accepted_anchor_{artifact_name}_authority:{index}",
                    trusted_public_key_sha256_env=(
                        SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV
                        if artifact_name == "prediction"
                        else SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV
                    ),
                )
                if authority_validation:
                    blockers.extend(authority_validation)
                    artifact_content_valid = False
                observed_value = next(
                    (payload[key] for key in keys if isinstance(payload.get(key), bool)),
                    None,
                )
            if observed_value is not expected_value:
                blockers.append(f"accepted_anchor_{artifact_name}_content_mismatch:{index}")
                artifact_content_valid = False
        seed = row.get("replicate_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            blockers.append(f"accepted_anchor_replicate_seed_invalid:{index}")
            artifact_content_valid = False
        if not artifact_content_valid:
            continue
        policy_id = _string(row.get("policy_id"))
        checkpoint_binding = (
            _string(row.get("checkpoint_id")),
            _string(row.get("policy_checkpoint_sha256")).lower(),
        )
        policy_checkpoint_bindings.setdefault(policy_id, set()).add(checkpoint_binding)
        checkpoint_policy_bindings.setdefault(checkpoint_binding[1], set()).add(policy_id)
        condition_id = _string(row.get("condition_id"))
        condition_descriptors.setdefault(condition_id, set()).add(
            (
                registered_split,
                _string(row.get("task_family")),
                _string(row.get("task_id")),
                _string(row.get("criterion_id")),
                _string(row.get("scenario_variation_instance_id")),
                _string(row.get("condition_source_id")),
                split_manifest_digest,
            )
        )
        valid_rows.append(row)
    decision_blockers: list[str] = []
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in valid_rows:
        cells.setdefault(
            (_string(row.get("policy_id")), _string(row.get("condition_id"))),
            [],
        ).append(row)
    policy_ids = sorted({_string(row.get("policy_id")) for row in valid_rows})
    condition_ids = sorted({_string(row.get("condition_id")) for row in valid_rows})
    seed_sets: dict[tuple[str, str], set[int]] = {}
    cell_summary: list[dict[str, Any]] = []
    if len(policy_ids) < SC3_MIN_POLICY_GROUPS_FOR_DIAGNOSTIC_CORRELATION:
        decision_blockers.append(
            "accepted_anchor_policy_group_count_lt_"
            f"{SC3_MIN_POLICY_GROUPS_FOR_DIAGNOSTIC_CORRELATION}"
        )
    if len(registered_splits) != 1:
        blockers.append("accepted_anchor_registered_split_mixed")
    if len(split_manifest_digests) != 1:
        blockers.append("accepted_anchor_split_manifest_mixed")
    for policy_id, bindings in policy_checkpoint_bindings.items():
        if len(bindings) != 1:
            decision_blockers.append(
                f"accepted_anchor_policy_checkpoint_identity_not_unique:{policy_id}"
            )
    if any(len(policies) != 1 for policies in checkpoint_policy_bindings.values()):
        decision_blockers.append("accepted_anchor_checkpoint_reused_across_policies")
    if len(checkpoint_policy_bindings) < SC3_MIN_POLICY_GROUPS_FOR_DIAGNOSTIC_CORRELATION:
        decision_blockers.append(
            "accepted_anchor_distinct_checkpoint_count_lt_"
            f"{SC3_MIN_POLICY_GROUPS_FOR_DIAGNOSTIC_CORRELATION}"
        )
    for condition_id, descriptors in condition_descriptors.items():
        if len(descriptors) != 1:
            decision_blockers.append(
                f"accepted_anchor_condition_descriptor_mismatch:{condition_id}"
            )
    for policy_id in policy_ids:
        for condition_id in condition_ids:
            cell_rows = cells.get((policy_id, condition_id), [])
            seeds = {
                int(row["replicate_seed"])
                for row in cell_rows
                if isinstance(row.get("replicate_seed"), int)
                and not isinstance(row.get("replicate_seed"), bool)
            }
            seed_sets[(policy_id, condition_id)] = seeds
            if len(cell_rows) < SC3_MIN_ANCHOR_REPLICATES_PER_POLICY_CONDITION:
                decision_blockers.append(
                    "accepted_anchor_cell_replicates_lt_"
                    f"{SC3_MIN_ANCHOR_REPLICATES_PER_POLICY_CONDITION}:"
                    f"{policy_id}:{condition_id}"
                )
            if len(seeds) != len(cell_rows):
                decision_blockers.append(
                    f"accepted_anchor_cell_seeds_missing_or_duplicate:{policy_id}:{condition_id}"
                )
            cell_summary.append(
                {
                    "policy_id": policy_id,
                    "condition_id": condition_id,
                    "replicate_count": len(cell_rows),
                    "unique_seed_count": len(seeds),
                }
            )
    for condition_id in condition_ids:
        expected_seeds: set[int] | None = None
        for policy_id in policy_ids:
            seeds = seed_sets.get((policy_id, condition_id), set())
            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                decision_blockers.append(
                    f"accepted_anchor_matched_seed_set_mismatch:{condition_id}"
                )
                break
    blockers = sorted(set(blockers))
    decision_blockers = sorted(set(decision_blockers))
    return {
        "schema_version": "sc3_anchor_artifact_validation.v1",
        "status": "validated" if valid_rows and not blockers else "blocked",
        "input_row_count": len(rows),
        "valid_row_count": len(valid_rows),
        "valid_rows": valid_rows,
        "decision_grade_status": (
            "decision_grade"
            if valid_rows and not blockers and not decision_blockers
            else "inconclusive_insufficient_n_or_unmatched"
        ),
        "minimum_replicates_per_policy_condition": (SC3_MIN_ANCHOR_REPLICATES_PER_POLICY_CONDITION),
        "minimum_policy_groups": SC3_MIN_POLICY_GROUPS_FOR_DIAGNOSTIC_CORRELATION,
        "cell_summary": cell_summary,
        "decision_grade_blockers": decision_blockers,
        "blockers": blockers or ([] if valid_rows else ["accepted_anchor_rows_missing"]),
    }
