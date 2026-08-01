"""Compile native-360 pose requests from accepted deterministic artifacts."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .reconstruction_validation_contracts import validate_camera_rig
from .reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    ReconstructionWorkerContractError,
    build_pose_estimation_request,
    build_worker_build_receipt,
    build_worker_smoke_receipt,
    build_worker_stack_manifest,
)


_METHOD_MODEL_ASSETS = {
    "colmap_sift_bruteforce_v1": (None, None),
    "colmap_sift_lightglue_v1": (None, PINNED_MODEL_ASSETS[2]["digest"]),
    "colmap_aliked_bruteforce_v1": (
        PINNED_MODEL_ASSETS[0]["digest"],
        PINNED_MODEL_ASSETS[3]["digest"],
    ),
    "colmap_aliked_lightglue_v1": (
        PINNED_MODEL_ASSETS[0]["digest"],
        PINNED_MODEL_ASSETS[1]["digest"],
    ),
}


class Native360PoseRequestCompilationError(ValueError):
    """Stable fail-closed error for native pose request compilation."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _mapping(value: Any) -> dict[str, Any]:
    return json.loads(canonical_json(dict(value))) if isinstance(value, Mapping) else {}


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def compile_native_360_pose_estimation_request(
    *,
    stable_run_identity: str,
    reconstruction_dataset: Mapping[str, Any],
    camera_rig_validation_request: Mapping[str, Any],
    camera_rig_validation_result: Mapping[str, Any],
    worker_stack_manifest: Mapping[str, Any],
    worker_build_receipt: Mapping[str, Any],
    worker_smoke_receipt: Mapping[str, Any],
    execution_configuration: Mapping[str, Any],
    execution_authority: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Bind one candidate-only native dataset to a qualified pose worker."""

    dataset = _mapping(reconstruction_dataset)
    rig_request = _mapping(camera_rig_validation_request)
    rig_result = _mapping(camera_rig_validation_result)
    configuration = _mapping(execution_configuration)
    authority = _mapping(execution_authority)
    errors: list[str] = []

    dataset_digest = dataset.get("dataset_manifest_digest")
    if (
        dataset.get("schema_version") != "reconstruction_dataset_manifest.v1"
        or not _digest(dataset_digest)
        or dataset_digest != canonical_digest(dataset, digest_field="dataset_manifest_digest")
        or dataset.get("capture_authority_profile") != "camera_360_native"
    ):
        errors.append("native_pose_dataset_manifest_invalid")
    if (
        dataset.get("candidate_dataset_contains_hidden_heldout_pixels") is not False
        or dataset.get("candidate_can_modify_split") is not False
        or dataset.get("raw_capture_bytes_remain_authoritative") is not True
        or not _digest(dataset.get("train_heldout_split_digest"))
    ):
        errors.append("native_pose_dataset_isolation_invalid")
    original_files = dataset.get("original_file_references")
    if (
        not isinstance(original_files, list)
        or not original_files
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("relative_path") or "")
            or not _digest(row.get("digest"))
            for row in original_files or []
        )
    ):
        errors.append("native_pose_original_file_lineage_invalid")

    try:
        expected_rig_result = validate_camera_rig(rig_request)
    except ValueError:
        expected_rig_result = {}
        errors.append("native_pose_camera_rig_request_invalid")
    if not rig_result or canonical_json(rig_result) != canonical_json(expected_rig_result):
        errors.append("native_pose_camera_rig_result_not_reproducible")
    if (
        rig_result.get("status") != "validated"
        or rig_result.get("capture_timeline_valid") is not True
        or rig_result.get("metric_scale_proven") is not False
        or rig_result.get("camera_trajectory_proven") is not False
    ):
        errors.append("native_pose_camera_rig_not_accepted")
    capture_digest = dataset.get("source_capture_digest")
    if not _digest(capture_digest) or rig_result.get("source_capture_digest") != capture_digest:
        errors.append("native_pose_capture_binding_invalid")
    calibration_binding = dataset.get("camera_calibration_binding")
    stream_metadata = dataset.get("stream_metadata")
    calibration_binding = (
        dict(calibration_binding) if isinstance(calibration_binding, Mapping) else {}
    )
    stream_metadata = dict(stream_metadata) if isinstance(stream_metadata, Mapping) else {}
    calibration_digest = calibration_binding.get("camera_360_rig_declaration_digest")
    if calibration_digest != rig_result.get("rig_declaration_digest") or stream_metadata.get(
        "dual_fisheye_binding_digest"
    ) != rig_result.get("dual_fisheye_binding_digest"):
        errors.append("native_pose_calibration_or_stream_binding_mismatch")

    try:
        stack = build_worker_stack_manifest(worker_stack_manifest)
    except ReconstructionWorkerContractError:
        stack = {}
        errors.append("native_pose_worker_stack_invalid")
    try:
        build = build_worker_build_receipt(worker_build_receipt)
    except ReconstructionWorkerContractError:
        build = {}
        errors.append("native_pose_worker_build_receipt_invalid")
    try:
        smoke = build_worker_smoke_receipt(worker_smoke_receipt)
    except ReconstructionWorkerContractError:
        smoke = {}
        errors.append("native_pose_worker_smoke_receipt_invalid")
    image = build.get("resolved_image_digest")
    if (
        build.get("status") != "built"
        or build.get("blockers") != []
        or build.get("worker_stack_manifest_digest") != stack.get("worker_stack_manifest_digest")
        or build.get("source_commit_sha") != stack.get("source_commit_sha")
    ):
        errors.append("native_pose_worker_build_not_accepted")
    provider = configuration.get("provider_runtime_identity")
    provider = dict(provider) if isinstance(provider, Mapping) else {}
    if (
        smoke.get("status") != "passed"
        or smoke.get("build_receipt_digest") != build.get("build_receipt_digest")
        or smoke.get("resolved_image_digest") != image
        or smoke.get("source_commit_sha") != build.get("source_commit_sha")
        or smoke.get("display_attached") is not False
        or smoke.get("provider_runtime_identity") != provider
    ):
        errors.append("native_pose_worker_smoke_not_accepted")

    pinned_repository_asset_digests = {
        row["digest"] for row in PINNED_MODEL_ASSETS if isinstance(row, Mapping)
    }
    worker_asset_digests = {
        row.get("digest") for row in stack.get("model_assets", []) if isinstance(row, Mapping)
    }
    feature_digest = configuration.get("model_asset_digest")
    matcher_digest = configuration.get("matcher_model_asset_digest")
    if feature_digest is not None and (
        feature_digest not in pinned_repository_asset_digests
        or feature_digest not in worker_asset_digests
    ):
        errors.append("native_pose_feature_model_not_pinned")
    if matcher_digest is not None and (
        matcher_digest not in pinned_repository_asset_digests
        or matcher_digest not in worker_asset_digests
    ):
        errors.append("native_pose_matcher_model_not_pinned")
    expected_assets = _METHOD_MODEL_ASSETS.get(configuration.get("method_profile_id"))
    if expected_assets is not None and (feature_digest, matcher_digest) != expected_assets:
        errors.append("native_pose_method_model_assets_invalid")
    if configuration.get("camera_model") not in {
        "OPENCV_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
    }:
        errors.append("native_pose_fisheye_camera_model_required")

    max_spend = authority.get("max_spend_usd")
    spend_cap = configuration.get("spend_cap_usd")
    hard_ttl = authority.get("hard_ttl_seconds")
    timeout = configuration.get("timeout_seconds")
    retry_cap = authority.get("retry_cap")
    is_remote = provider.get("provider") not in {None, "local"}
    if (
        not str(authority.get("authority_id") or "").strip()
        or not _finite(max_spend)
        or not _finite(spend_cap)
        or float(spend_cap or 0) > float(max_spend or 0)
        or isinstance(hard_ttl, bool)
        or not isinstance(hard_ttl, int)
        or hard_ttl <= 0
        or not _finite(timeout, minimum=1)
        or float(timeout or 0) > hard_ttl
        or isinstance(retry_cap, bool)
        or not isinstance(retry_cap, int)
        or retry_cap < 0
    ):
        errors.append("native_pose_execution_authority_invalid")
    capture_authority = dataset.get("authority_used")
    capture_authority = dict(capture_authority) if isinstance(capture_authority, Mapping) else {}
    if is_remote and (
        authority.get("paid_compute_authorized") is not True
        or authority.get("provider_processing_authorized") is not True
        or authority.get("provider_upload_authorized") is not True
        or capture_authority.get("provider_upload_authorized") is not True
    ):
        errors.append("native_pose_remote_authority_missing")

    if errors:
        raise Native360PoseRequestCompilationError(errors)

    assert isinstance(original_files, list)
    deterministic_configuration_digest = canonical_digest(
        {
            "execution_configuration": configuration,
            "execution_bounds": {
                "max_spend_usd": max_spend,
                "hard_ttl_seconds": hard_ttl,
                "retry_cap": retry_cap,
            },
            "dataset_manifest_digest": dataset_digest,
            "camera_rig_validation_result_digest": rig_result[
                "camera_rig_validation_result_digest"
            ],
            "worker_build_receipt_digest": build["build_receipt_digest"],
            "worker_smoke_receipt_digest": smoke["smoke_test_receipt_digest"],
        }
    )
    request = {
        "stable_run_identity": stable_run_identity,
        "source_capture_identity": dataset.get("source_capture_identity"),
        "source_capture_digest": capture_digest,
        "original_file_references": [
            {"artifact_id": row["relative_path"], "digest": row["digest"]} for row in original_files
        ],
        "producing_method": "blueprint.native_360_pose_request_compiler",
        "implementation_version": "1.0.0",
        "container_image_digest": image,
        "source_commit_sha": build["source_commit_sha"],
        "deterministic_configuration_digest": deterministic_configuration_digest,
        "input_digests": [
            {"artifact_id": "reconstruction_dataset", "digest": dataset_digest},
            {
                "artifact_id": "camera_rig_validation",
                "digest": rig_result["camera_rig_validation_result_digest"],
            },
            {"artifact_id": "worker_build_receipt", "digest": build["build_receipt_digest"]},
            {"artifact_id": "worker_smoke_receipt", "digest": smoke["smoke_test_receipt_digest"]},
        ],
        "output_digests": [],
        "train_heldout_split_digest": dataset["train_heldout_split_digest"],
        "camera_calibration_binding": calibration_binding,
        "coordinate_frame_declaration": dataset["coordinate_frame_declaration"],
        "units": "unknown",
        "metric_scale_status": "anchor_required",
        "provider_runtime_identity": provider,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {**authority, "capture_authority": capture_authority},
        "warnings": ["metric_scale_anchor_required_after_pose_estimation"],
        "blockers": [],
        "proof_effect": "none",
        "claim_ceiling": "request_only",
        "parent_artifact_or_event": {
            "dataset_manifest_digest": dataset_digest,
            "camera_rig_validation_result_digest": rig_result[
                "camera_rig_validation_result_digest"
            ],
        },
        "timestamp": timestamp,
        "method_profile_id": configuration.get("method_profile_id"),
        "feature_extractor": configuration.get("feature_extractor"),
        "feature_matcher": configuration.get("feature_matcher"),
        "camera_model": configuration.get("camera_model"),
        "reconstruction_dataset_digest": dataset_digest,
        "camera_rig_digest": rig_result["camera_rig_validation_result_digest"],
        "calibration_digest": calibration_digest,
        "model_asset_digest": feature_digest,
        "matcher_model_asset_digest": matcher_digest,
        "deterministic_matching": True,
        "random_seed": configuration.get("random_seed"),
        "resource_request": configuration.get("resource_request"),
        "timeout_seconds": timeout,
        "hard_ttl_seconds": hard_ttl,
        "retry_cap": retry_cap,
        "spend_cap_usd": spend_cap,
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_change_split": False,
        "candidate_may_read_hidden_heldout": False,
    }
    try:
        return build_pose_estimation_request(request)
    except ReconstructionWorkerContractError as exc:
        raise Native360PoseRequestCompilationError(
            ["compiled_native_pose_request_invalid", *exc.codes]
        ) from exc


__all__ = [
    "Native360PoseRequestCompilationError",
    "compile_native_360_pose_estimation_request",
]
