"""Typed ARKit-anchored pose-refinement worker contracts."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .camera_geometry_validation import validate_se3_matrix
from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import FAILURE_CODES


POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION = "pose_refinement_execution_request.v1"
POSE_REFINEMENT_RESULT_SCHEMA_VERSION = "pose_refinement_result.v1"
REFINED_CAMERA_POSE_MANIFEST_SCHEMA_VERSION = "refined_camera_pose_manifest.v1"


class PoseRefinementContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PoseRefinementContractError(["pose_refinement_artifact_not_json"]) from exc


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _positive(value: Any, *, allow_zero: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (float(value) >= 0 if allow_zero else float(value) > 0)
    )


def build_pose_refinement_execution_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION:
        errors.append("pose_refinement_request_schema_invalid")
    for key in (
        "source_capture_digest",
        "reconstruction_dataset_digest",
        "frozen_split_digest",
        "camera_observation_digest",
        "camera_calibration_digest",
        "initial_pose_manifest_digest",
        "implementation_digest",
        "container_image_digest",
    ):
        if not _digest(request.get(key)):
            errors.append(f"pose_refinement_request_{key}_invalid")
    commit = str(request.get("source_commit_sha") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        errors.append("pose_refinement_request_source_commit_invalid")
    if request.get("capture_profile") != "iphone_arkit_lidar":
        errors.append("pose_refinement_request_capture_profile_invalid")
    if request.get("initial_pose_source") != "verified_arkit_raw_contract_3_2":
        errors.append("pose_refinement_request_initial_pose_source_invalid")
    if request.get("method_id") not in {
        "arkit_anchored_bundle_adjustment_v1",
        "arkit_anchored_pose_graph_refinement_v1",
    }:
        errors.append("pose_refinement_request_method_invalid")
    thresholds = request.get("drift_thresholds")
    if not isinstance(thresholds, Mapping) or any(
        not _positive(thresholds.get(key))
        for key in ("maximum_translation_m", "maximum_rotation_degrees")
    ):
        errors.append("pose_refinement_request_drift_thresholds_invalid")
    if request.get("thresholds_frozen_before_execution") is not True:
        errors.append("pose_refinement_request_thresholds_not_frozen")
    if request.get("raw_arkit_poses_may_be_modified") is not False:
        errors.append("pose_refinement_request_raw_pose_mutation_forbidden")
    if request.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("pose_refinement_request_hidden_access_forbidden")
    if not isinstance(request.get("coordinate_frame_declaration"), Mapping):
        errors.append("pose_refinement_request_coordinate_frame_invalid")
    if not isinstance(request.get("resource_request"), Mapping):
        errors.append("pose_refinement_request_resource_invalid")
    if not _positive(request.get("timeout_seconds")) or not _positive(
        request.get("spend_cap_usd"), allow_zero=True
    ):
        errors.append("pose_refinement_request_execution_bounds_invalid")
    if not isinstance(request.get("random_seed"), int) or isinstance(
        request.get("random_seed"), bool
    ):
        errors.append("pose_refinement_request_random_seed_invalid")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("pose_refinement_request_authority_invalid")
    if not str(request.get("stable_run_identity") or "").strip() or not str(
        request.get("timestamp") or ""
    ).strip():
        errors.append("pose_refinement_request_identity_or_timestamp_missing")
    supplied = request.pop("pose_refinement_execution_request_digest", None)
    request["deterministic_configuration_digest"] = canonical_digest(
        {
            "method_id": request.get("method_id"),
            "dataset": request.get("reconstruction_dataset_digest"),
            "split": request.get("frozen_split_digest"),
            "calibration": request.get("camera_calibration_digest"),
            "initial_poses": request.get("initial_pose_manifest_digest"),
            "thresholds": request.get("drift_thresholds"),
            "seed": request.get("random_seed"),
            "implementation": request.get("implementation_digest"),
            "container": request.get("container_image_digest"),
        }
    )
    request["pose_refinement_execution_request_digest"] = canonical_digest(
        request, digest_field="pose_refinement_execution_request_digest"
    )
    if supplied is not None and supplied != request[
        "pose_refinement_execution_request_digest"
    ]:
        errors.append("pose_refinement_request_digest_mismatch")
    if errors:
        raise PoseRefinementContractError(errors)
    return request


def build_pose_refinement_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _clone(dict(value))
    errors: list[str] = []
    if result.get("schema_version") != POSE_REFINEMENT_RESULT_SCHEMA_VERSION:
        errors.append("pose_refinement_result_schema_invalid")
    for key in (
        "source_capture_digest",
        "pose_refinement_execution_request_digest",
        "frozen_split_digest",
        "camera_calibration_digest",
        "initial_pose_manifest_digest",
        "implementation_digest",
        "container_image_digest",
    ):
        if not _digest(result.get(key)):
            errors.append(f"pose_refinement_result_{key}_invalid")
    if result.get("status") not in {"succeeded", "failed", "timed_out", "interrupted"}:
        errors.append("pose_refinement_result_status_invalid")
    failure = result.get("failure_code")
    if result.get("status") == "succeeded":
        if failure is not None or not _digest(result.get("refined_pose_manifest_digest")):
            errors.append("pose_refinement_result_success_artifact_invalid")
        reference = result.get("refined_pose_manifest_reference")
        if reference is not None and (
            not isinstance(reference, Mapping)
            or not str(reference.get("relative_path") or "").strip()
            or not _digest(reference.get("artifact_digest"))
            or reference.get("manifest_digest")
            != result.get("refined_pose_manifest_digest")
        ):
            errors.append("pose_refinement_result_manifest_reference_invalid")
    elif failure not in FAILURE_CODES:
        errors.append("pose_refinement_result_failure_code_invalid")
    metrics = result.get("drift_metrics")
    if not isinstance(metrics, Mapping) or any(
        not _positive(metrics.get(key), allow_zero=True)
        for key in (
            "maximum_translation_m",
            "mean_translation_m",
            "maximum_rotation_degrees",
            "mean_rotation_degrees",
        )
    ):
        errors.append("pose_refinement_result_drift_metrics_invalid")
    for key in ("registered_observation_ids", "rejected_observation_ids", "warnings", "blockers"):
        if not isinstance(result.get(key), list):
            errors.append(f"pose_refinement_result_{key}_invalid")
    if result.get("raw_arkit_poses_modified") is not False:
        errors.append("pose_refinement_result_raw_pose_mutation_forbidden")
    if result.get("heldout_labels_included") is not False or result.get(
        "candidate_self_graded"
    ) is not False:
        errors.append("pose_refinement_result_hidden_or_self_grading_forbidden")
    if result.get("proof_effect") != "bounded_refined_trajectory_candidate_only" or result.get(
        "claim_ceiling"
    ) != "calibrated_camera_trajectory":
        errors.append("pose_refinement_result_claim_boundary_invalid")
    if not _positive(result.get("cost_usd"), allow_zero=True) or not _positive(
        result.get("duration_seconds"), allow_zero=True
    ):
        errors.append("pose_refinement_result_execution_accounting_invalid")
    supplied = result.pop("pose_refinement_result_digest", None)
    result["pose_refinement_result_digest"] = canonical_digest(
        result, digest_field="pose_refinement_result_digest"
    )
    if supplied is not None and supplied != result["pose_refinement_result_digest"]:
        errors.append("pose_refinement_result_digest_mismatch")
    if errors:
        raise PoseRefinementContractError(errors)
    return result


def build_refined_camera_pose_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a derived pose artifact without changing its ARKit parent.

    The manifest contains candidate-only refined poses. Its lineage binds it to
    the immutable raw-pose manifest and frozen refinement request; it cannot
    contain evaluator-held-out observations or claim that raw ARKit data was
    modified.
    """

    manifest = _clone(dict(value))
    errors: list[str] = []
    if manifest.get("schema_version") != REFINED_CAMERA_POSE_MANIFEST_SCHEMA_VERSION:
        errors.append("refined_pose_manifest_schema_invalid")
    for key in (
        "source_capture_digest",
        "frozen_split_digest",
        "camera_calibration_digest",
        "initial_pose_manifest_digest",
        "pose_refinement_execution_request_digest",
        "implementation_digest",
        "container_image_digest",
    ):
        if not _digest(manifest.get(key)):
            errors.append(f"refined_pose_manifest_{key}_invalid")
    if any(
        not str(manifest.get(key) or "").strip()
        for key in (
            "stable_run_identity",
            "source_capture_identity",
            "producing_method",
            "implementation_version",
            "timestamp",
        )
    ):
        errors.append("refined_pose_manifest_identity_or_timestamp_missing")
    commit = str(manifest.get("source_commit_sha") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        errors.append("refined_pose_manifest_source_commit_invalid")
    if manifest.get("method_id") not in {
        "arkit_anchored_bundle_adjustment_v1",
        "arkit_anchored_pose_graph_refinement_v1",
    }:
        errors.append("refined_pose_manifest_method_invalid")
    if not isinstance(manifest.get("coordinate_frame_declaration"), Mapping):
        errors.append("refined_pose_manifest_coordinate_frame_invalid")
    if manifest.get("units") != "meters" or manifest.get("metric_scale_status") not in {
        "sensor_metric_unvalidated",
        "independently_validated_metric",
    }:
        errors.append("refined_pose_manifest_units_or_scale_status_invalid")
    if any(
        not isinstance(manifest.get(key), expected)
        for key, expected in (
            ("provider_runtime_identity", Mapping),
            ("authority_used", Mapping),
            ("parent_artifact_or_event", Mapping),
            ("original_file_references", list),
            ("input_digests", list),
            ("output_digests", list),
            ("warnings", list),
            ("blockers", list),
        )
    ):
        errors.append("refined_pose_manifest_provenance_envelope_invalid")
    original_files = manifest.get("original_file_references")
    if isinstance(original_files, list) and (
        not original_files
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("artifact_id") or row.get("relative_path") or "")
            or not _digest(row.get("digest"))
            for row in original_files
        )
    ):
        errors.append("refined_pose_manifest_original_files_invalid")
    if not _positive(manifest.get("cost_usd"), allow_zero=True) or not _positive(
        manifest.get("duration_seconds"), allow_zero=True
    ):
        errors.append("refined_pose_manifest_execution_accounting_invalid")
    if manifest.get("raw_arkit_poses_modified") is not False:
        errors.append("refined_pose_manifest_raw_pose_mutation_forbidden")
    if manifest.get("hidden_heldout_observations_included") is not False:
        errors.append("refined_pose_manifest_hidden_access_forbidden")
    if manifest.get("proof_effect") != "bounded_refined_trajectory_candidate_only" or manifest.get(
        "claim_ceiling"
    ) != "calibrated_camera_trajectory":
        errors.append("refined_pose_manifest_claim_boundary_invalid")
    observations = manifest.get("observations")
    if not isinstance(observations, list) or not observations:
        errors.append("refined_pose_manifest_observations_missing")
    else:
        identifiers: set[str] = set()
        for row in observations:
            if not isinstance(row, Mapping):
                errors.append("refined_pose_manifest_observation_invalid")
                continue
            observation_id = str(row.get("observation_id") or "")
            if not observation_id or observation_id in identifiers:
                errors.append("refined_pose_manifest_observation_id_invalid_or_duplicate")
            identifiers.add(observation_id)
            validation = validate_se3_matrix(
                row.get("T_world_camera"),
                field="refined_T_world_camera",
            )
            if not validation["valid"]:
                errors.append("refined_pose_manifest_transform_invalid")
    supplied_configuration = manifest.pop("deterministic_configuration_digest", None)
    manifest["deterministic_configuration_digest"] = canonical_digest(
        {
            "pose_refinement_execution_request_digest": manifest.get(
                "pose_refinement_execution_request_digest"
            ),
            "initial_pose_manifest_digest": manifest.get("initial_pose_manifest_digest"),
            "camera_calibration_digest": manifest.get("camera_calibration_digest"),
            "method_id": manifest.get("method_id"),
            "implementation_digest": manifest.get("implementation_digest"),
            "container_image_digest": manifest.get("container_image_digest"),
            "observations": manifest.get("observations"),
        }
    )
    if (
        supplied_configuration is not None
        and supplied_configuration != manifest["deterministic_configuration_digest"]
    ):
        errors.append("refined_pose_manifest_configuration_digest_mismatch")
    supplied = manifest.pop("refined_camera_pose_manifest_digest", None)
    manifest["refined_camera_pose_manifest_digest"] = canonical_digest(
        manifest,
        digest_field="refined_camera_pose_manifest_digest",
    )
    if supplied is not None and supplied != manifest["refined_camera_pose_manifest_digest"]:
        errors.append("refined_pose_manifest_digest_mismatch")
    if errors:
        raise PoseRefinementContractError(errors)
    return manifest


__all__ = [
    "POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION",
    "POSE_REFINEMENT_RESULT_SCHEMA_VERSION",
    "REFINED_CAMERA_POSE_MANIFEST_SCHEMA_VERSION",
    "PoseRefinementContractError",
    "build_refined_camera_pose_manifest",
    "build_pose_refinement_execution_request",
    "build_pose_refinement_result",
]
