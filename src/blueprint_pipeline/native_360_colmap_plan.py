"""Compile a deterministic, non-executing COLMAP plan for native 360 rigs.

The plan is an inert artifact: it contains argv arrays, immutable input
materialization bindings, and expected outputs.  It never invokes COLMAP and it
never exposes hidden held-out observations.  A trusted registered runtime may
execute an accepted plan later under the separately admitted pose request.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_pose_estimation_request,
)


NATIVE_360_COLMAP_PLAN_SCHEMA_VERSION = "native_360_colmap_execution_plan.v1"
NATIVE_360_COLMAP_PLAN_IMPLEMENTATION_VERSION = "1.0.0"
_CAMERA_AXES = "+x right, +y down, +z forward"
_CAMERA_MODELS = {
    "OPENCV_FISHEYE": ("opencv_fisheye", 4),
    "RAD_TAN_THIN_PRISM_FISHEYE": ("radtan_thin_prism_fisheye", 12),
}
_LENSES = ("front", "rear")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_MODEL_PATH_ARGUMENTS = {
    "colmap_sift_bruteforce_v1": ((), ()),
    "colmap_sift_lightglue_v1": (
        (),
        ("--SiftMatching.lightglue_model_path", "/opt/models/colmap/sift-lightglue.onnx"),
    ),
    "colmap_aliked_bruteforce_v1": (
        (
            "--AlikedExtraction.n16rot_model_path",
            "/opt/models/colmap/aliked-n16rot.onnx",
        ),
        (
            "--AlikedMatching.bruteforce_model_path",
            "/opt/models/colmap/bruteforce-matcher.onnx",
        ),
    ),
    "colmap_aliked_lightglue_v1": (
        (
            "--AlikedExtraction.n16rot_model_path",
            "/opt/models/colmap/aliked-n16rot.onnx",
        ),
        (
            "--AlikedMatching.lightglue_model_path",
            "/opt/models/colmap/aliked-lightglue.onnx",
        ),
    ),
}


class Native360ColmapPlanError(ValueError):
    """Stable fail-closed error for native COLMAP plan compilation."""

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


def _safe_relative(value: Any) -> str | None:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validated_transform(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        numbers = [_finite_number(item) for item in row]
        if any(item is None for item in numbers):
            return None
        rows.append([float(item) for item in numbers if item is not None])
    if any(abs(rows[3][index] - expected) > 1e-9 for index, expected in enumerate((0, 0, 0, 1))):
        return None
    rotation = [row[:3] for row in rows[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[index][left] * rotation[index][right] for index in range(3))
            if abs(dot - (1.0 if left == right else 0.0)) > 1e-6:
                return None
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return rows if abs(determinant - 1.0) <= 1e-6 else None


def _inverse_rigid(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [row[:3] for row in matrix[:3]]
    translation = [matrix[index][3] for index in range(3)]
    inverse_rotation = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(inverse_rotation[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [[*inverse_rotation[row], inverse_translation[row]] for row in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]
    ]


def _quaternion_wxyz(rotation: list[list[float]]) -> list[float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        values = [
            0.25 * scale,
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
        ]
    else:
        index = max(range(3), key=lambda item: rotation[item][item])
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2
            values = [
                (rotation[2][1] - rotation[1][2]) / scale,
                0.25 * scale,
                (rotation[0][1] + rotation[1][0]) / scale,
                (rotation[0][2] + rotation[2][0]) / scale,
            ]
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2
            values = [
                (rotation[0][2] - rotation[2][0]) / scale,
                (rotation[0][1] + rotation[1][0]) / scale,
                0.25 * scale,
                (rotation[1][2] + rotation[2][1]) / scale,
            ]
        else:
            scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2
            values = [
                (rotation[1][0] - rotation[0][1]) / scale,
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                0.25 * scale,
            ]
    norm = math.sqrt(sum(value * value for value in values))
    normalized = [value / norm for value in values]
    first_nonzero = next((value for value in normalized if abs(value) > 1e-15), 1.0)
    if first_nonzero < 0:
        normalized = [-value for value in normalized]
    return [0.0 if abs(value) < 1e-15 else value for value in normalized]


def _camera_parameters(
    calibration: Mapping[str, Any], camera_model: str, errors: list[str]
) -> list[float] | None:
    lens = str(calibration.get("lens_id") or "unknown")
    intrinsics = calibration.get("intrinsics")
    distortion = calibration.get("distortion")
    if not isinstance(intrinsics, Mapping) or not isinstance(distortion, Mapping):
        errors.append(f"native_colmap_lens_calibration_invalid:{lens}")
        return None
    expected_model, coefficient_count = _CAMERA_MODELS[camera_model]
    coefficients = distortion.get("coefficients")
    if distortion.get("model") != expected_model:
        errors.append(f"native_colmap_distortion_model_mismatch:{lens}")
        return None
    if not isinstance(coefficients, list) or len(coefficients) != coefficient_count:
        errors.append(f"native_colmap_distortion_coefficient_count_invalid:{lens}")
        return None
    raw_values = [
        intrinsics.get("fx"),
        intrinsics.get("fy"),
        intrinsics.get("cx"),
        intrinsics.get("cy"),
        *coefficients,
    ]
    values = [_finite_number(item) for item in raw_values]
    width = intrinsics.get("width")
    height = intrinsics.get("height")
    if (
        any(item is None for item in values)
        or any(float(item or 0) <= 0 for item in values[:2])
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        errors.append(f"native_colmap_lens_calibration_invalid:{lens}")
        return None
    return [float(item) for item in values if item is not None]


def compile_native_360_colmap_execution_plan(
    *,
    stable_run_identity: str,
    reconstruction_dataset: Mapping[str, Any],
    candidate_dataset_manifest: Mapping[str, Any],
    camera_rig_declaration: Mapping[str, Any],
    camera_rig_validation_result: Mapping[str, Any],
    pose_estimation_request: Mapping[str, Any],
    valid_pixel_mask_references: Mapping[str, Mapping[str, Any]],
    timestamp: str,
) -> dict[str, Any]:
    """Compile one candidate-only, calibrated dual-fisheye COLMAP plan."""

    dataset = _mapping(reconstruction_dataset)
    candidate = _mapping(candidate_dataset_manifest)
    rig = _mapping(camera_rig_declaration)
    rig_result = _mapping(camera_rig_validation_result)
    masks = _mapping(valid_pixel_mask_references)
    dataset_outputs = dataset.get("output_digests")
    dataset_outputs = dict(dataset_outputs) if isinstance(dataset_outputs, Mapping) else {}
    dataset_calibration = dataset.get("camera_calibration_binding")
    dataset_calibration = (
        dict(dataset_calibration) if isinstance(dataset_calibration, Mapping) else {}
    )
    errors: list[str] = []
    if not str(stable_run_identity or "").strip() or not str(timestamp or "").strip():
        errors.append("native_colmap_plan_identity_or_timestamp_missing")

    dataset_digest = dataset.get("dataset_manifest_digest")
    if (
        dataset.get("schema_version") != "reconstruction_dataset_manifest.v1"
        or dataset.get("capture_authority_profile") != "camera_360_native"
        or not _digest(dataset_digest)
        or dataset_digest != canonical_digest(dataset, digest_field="dataset_manifest_digest")
        or dataset.get("candidate_dataset_contains_hidden_heldout_pixels") is not False
        or dataset.get("candidate_can_modify_split") is not False
        or dataset.get("raw_capture_bytes_remain_authoritative") is not True
        or dataset.get("blockers") not in (None, [])
    ):
        errors.append("native_colmap_dataset_invalid")
    original_files = dataset.get("original_file_references")
    if (
        not isinstance(original_files, list)
        or not original_files
        or any(
            not isinstance(row, Mapping)
            or _safe_relative(row.get("relative_path")) is None
            or not _digest(row.get("digest"))
            for row in original_files or []
        )
    ):
        errors.append("native_colmap_original_file_lineage_invalid")

    candidate_digest = candidate.get("candidate_dataset_digest")
    if (
        candidate.get("schema_version") != "candidate_reconstruction_dataset_manifest.v1"
        or not _digest(candidate_digest)
        or candidate_digest != canonical_digest(candidate, digest_field="candidate_dataset_digest")
        or candidate.get("capture_digest") != dataset.get("source_capture_digest")
        or candidate.get("split_digest") != dataset.get("train_heldout_split_digest")
        or candidate.get("training_and_validation_only") is not True
        or candidate.get("heldout_pixels_included") is not False
        or dataset_outputs.get("candidate_dataset_digest") != candidate_digest
    ):
        errors.append("native_colmap_candidate_manifest_invalid")

    rig_digest = rig.get("rig_declaration_digest")
    if (
        rig.get("schema_version") != "camera_360_rig_declaration.v1"
        or not _digest(rig_digest)
        or rig_digest != canonical_digest(rig, digest_field="rig_declaration_digest")
        or rig.get("capture_digest") != dataset.get("source_capture_digest")
        or rig.get("calibration_status") != "valid"
        or rig.get("rig_is_fixed") is not True
        or rig.get("agent_may_alter_calibration") is not False
        or rig.get("blockers") != []
        or dataset_calibration.get("camera_360_rig_declaration_digest") != rig_digest
    ):
        errors.append("native_colmap_rig_declaration_invalid")
    if (
        rig_result.get("schema_version") != "camera_rig_validation_result.v1"
        or rig_result.get("camera_rig_validation_result_digest")
        != canonical_digest(rig_result, digest_field="camera_rig_validation_result_digest")
        or rig_result.get("status") != "validated"
        or rig_result.get("rig_declaration_digest") != rig_digest
        or rig_result.get("source_capture_digest") != dataset.get("source_capture_digest")
        or rig_result.get("capture_timeline_valid") is not True
        or rig_result.get("fixed_rig_extrinsics_valid") is not True
        or rig_result.get("camera_trajectory_proven") is not False
        or rig_result.get("metric_scale_proven") is not False
    ):
        errors.append("native_colmap_rig_validation_invalid")

    try:
        pose = build_pose_estimation_request(pose_estimation_request)
    except ReconstructionWorkerContractError:
        pose = {}
        errors.append("native_colmap_pose_request_invalid")
    if (
        pose.get("reconstruction_dataset_digest") != dataset_digest
        or pose.get("train_heldout_split_digest") != dataset.get("train_heldout_split_digest")
        or pose.get("calibration_digest") != rig_digest
        or pose.get("camera_rig_digest") != rig_result.get("camera_rig_validation_result_digest")
        or pose.get("candidate_may_read_hidden_heldout") is not False
        or pose.get("candidate_can_change_split") is not False
        or pose.get("candidate_dataset_contains_hidden_heldout_pixels") is not False
    ):
        errors.append("native_colmap_pose_binding_invalid")
    camera_model = str(pose.get("camera_model") or "")
    if camera_model not in _CAMERA_MODELS:
        errors.append("native_colmap_camera_model_unsupported")
    method_profile = str(pose.get("method_profile_id") or "")
    if method_profile not in _MODEL_PATH_ARGUMENTS:
        errors.append("native_colmap_method_profile_unsupported")

    coordinate = dataset.get("coordinate_frame_declaration")
    if (
        not isinstance(coordinate, Mapping)
        or coordinate.get("handedness") != "right_handed"
        or coordinate.get("camera_axes") != _CAMERA_AXES
        or coordinate.get("rig_frame") != "front_lens_optical_center"
    ):
        errors.append("native_colmap_coordinate_frame_incompatible")

    calibrations = rig.get("lens_calibrations")
    calibration_by_lens = {
        str(row.get("lens_id")): row for row in calibrations or [] if isinstance(row, Mapping)
    }
    if (
        not isinstance(calibrations, list)
        or len(calibrations) != 2
        or set(calibration_by_lens) != set(_LENSES)
    ):
        errors.append("native_colmap_lens_calibrations_invalid")
    if set(masks) != set(_LENSES):
        errors.append("native_colmap_valid_pixel_mask_reference_set_invalid")
    camera_params: dict[str, list[float]] = {}
    if camera_model in _CAMERA_MODELS:
        for lens in _LENSES:
            calibration = calibration_by_lens.get(lens, {})
            parameters = _camera_parameters(calibration, camera_model, errors)
            if parameters is not None:
                camera_params[lens] = parameters
            reference = masks.get(lens)
            reference = dict(reference) if isinstance(reference, Mapping) else {}
            if (
                _safe_relative(reference.get("relative_path")) is None
                or not _digest(reference.get("digest"))
                or reference.get("digest") != calibration.get("valid_pixel_mask_digest")
            ):
                errors.append(f"native_colmap_valid_pixel_mask_reference_invalid:{lens}")

    extrinsics = rig.get("rig_extrinsics")
    extrinsics = dict(extrinsics) if isinstance(extrinsics, Mapping) else {}
    transform = _validated_transform(extrinsics.get("T_front_rear"))
    semantics = extrinsics.get("transform_semantics")
    if (
        transform is None
        or semantics
        not in {
            "rear_camera_from_front_rig",
            "front_rig_from_rear_camera",
        }
        or extrinsics.get("translation_units") != "meters"
    ):
        errors.append("native_colmap_rig_transform_invalid")
    elif semantics == "front_rig_from_rear_camera":
        transform = _inverse_rigid(transform)

    frames = candidate.get("frames")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    if not isinstance(frames, list) or not frames:
        errors.append("native_colmap_candidate_frames_missing")
    else:
        for ordinal, raw in enumerate(frames):
            row = dict(raw) if isinstance(raw, Mapping) else {}
            group_id = str(row.get("observation_group_id") or "")
            frame_id = str(row.get("frame_id") or "")
            lens = str(row.get("source_camera_identity") or "")
            relative_path = _safe_relative(row.get("candidate_relative_path"))
            if (
                _SAFE_IDENTIFIER.fullmatch(group_id) is None
                or _SAFE_IDENTIFIER.fullmatch(frame_id) is None
                or lens not in _LENSES
                or relative_path is None
                or relative_path.startswith("evaluator_hidden/")
                or "/held_out/" in f"/{relative_path}/"
                or not _digest(row.get("frame_digest"))
                or row.get("split") not in {"training", "validation"}
                or _finite_number(row.get("t_video_sec")) is None
            ):
                errors.append(f"native_colmap_candidate_frame_invalid:{ordinal}")
                continue
            lens_rows = grouped.setdefault(group_id, {})
            if lens in lens_rows:
                errors.append(f"native_colmap_candidate_group_duplicate_lens:{group_id}:{lens}")
            lens_rows[lens] = row
    frame_ids = [
        str(row.get("frame_id")) for lens_rows in grouped.values() for row in lens_rows.values()
    ]
    if len(frame_ids) != len(set(frame_ids)):
        errors.append("native_colmap_candidate_frame_id_duplicate")
    if any(set(lens_rows) != set(_LENSES) for lens_rows in grouped.values()):
        errors.append("native_colmap_candidate_group_incomplete")
    if any(
        len({str(row.get("split")) for row in lens_rows.values()}) != 1
        for lens_rows in grouped.values()
        if set(lens_rows) == set(_LENSES)
    ):
        errors.append("native_colmap_candidate_group_split_mismatch")
    if errors:
        raise Native360ColmapPlanError(errors)

    assert transform is not None
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            min(float(row["t_video_sec"]) for row in item[1].values()),
            item[0],
        ),
    )
    materialization: list[dict[str, Any]] = []
    mask_materialization: list[dict[str, Any]] = []
    for group_ordinal, (group_id, lens_rows) in enumerate(ordered_groups):
        filename = f"frame_{group_ordinal:09d}.png"
        for lens in _LENSES:
            row = lens_rows[lens]
            materialization.append(
                {
                    "frame_id": row["frame_id"],
                    "observation_group_id": group_id,
                    "sensor_id": lens,
                    "source_relative_path": row["candidate_relative_path"],
                    "source_digest": row["frame_digest"],
                    "destination_relative_path": f"workspace/images/{lens}/{filename}",
                    "split": row["split"],
                    "captured_observation": True,
                }
            )
            reference = masks[lens]
            mask_materialization.append(
                {
                    "sensor_id": lens,
                    "source_relative_path": reference["relative_path"],
                    "source_digest": reference["digest"],
                    "destination_relative_path": f"workspace/masks/{lens}/{filename}.png",
                    "generated_or_inferred": False,
                }
            )

    rotation = [row[:3] for row in transform[:3]]
    rear_translation = [row[3] for row in transform[:3]]
    rig_config = [
        {
            "cameras": [
                {
                    "image_prefix": "front/",
                    "ref_sensor": True,
                    "camera_model_name": camera_model,
                    "camera_params": camera_params["front"],
                },
                {
                    "image_prefix": "rear/",
                    "cam_from_rig_rotation": _quaternion_wxyz(rotation),
                    "cam_from_rig_translation": rear_translation,
                    "camera_model_name": camera_model,
                    "camera_params": camera_params["rear"],
                },
            ]
        }
    ]
    seed = str(pose["random_seed"])
    extractor = str(pose["feature_extractor"])
    matcher = str(pose["feature_matcher"])
    extractor_model_arguments, matcher_model_arguments = _MODEL_PATH_ARGUMENTS[method_profile]
    common = ["--default_random_seed", seed, "--log_color", "0"]
    commands = [
        {
            "step_id": "extract_features",
            "argv": [
                "colmap",
                "feature_extractor",
                "--database_path",
                "workspace/database.db",
                "--image_path",
                "workspace/images",
                "--ImageReader.mask_path",
                "workspace/masks",
                "--ImageReader.single_camera_per_folder",
                "1",
                "--ImageReader.camera_model",
                camera_model,
                "--FeatureExtraction.type",
                extractor,
                "--FeatureExtraction.use_gpu",
                "1",
                "--FeatureExtraction.gpu_index",
                "0",
                *extractor_model_arguments,
                *common,
            ],
        },
        {
            "step_id": "configure_fixed_rig",
            "argv": [
                "colmap",
                "rig_configurator",
                "--database_path",
                "workspace/database.db",
                "--rig_config_path",
                "workspace/rig_config.json",
                *common,
            ],
        },
        {
            "step_id": "match_sequential_frames",
            "argv": [
                "colmap",
                "sequential_matcher",
                "--database_path",
                "workspace/database.db",
                "--FeatureMatching.type",
                matcher,
                "--FeatureMatching.use_gpu",
                "1",
                "--FeatureMatching.gpu_index",
                "0",
                "--FeatureMatching.skip_image_pairs_in_same_frame",
                "0",
                "--FeatureMatching.rig_verification",
                "1",
                "--TwoViewGeometry.random_seed",
                seed,
                *matcher_model_arguments,
                *common,
            ],
        },
        {
            "step_id": "map_fixed_calibrated_rig",
            "argv": [
                "colmap",
                "mapper",
                "--database_path",
                "workspace/database.db",
                "--image_path",
                "workspace/images",
                "--output_path",
                "workspace/sparse",
                "--Mapper.random_seed",
                seed,
                "--Mapper.ba_refine_sensor_from_rig",
                "0",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
                "--Mapper.multiple_models",
                "0",
                *common,
            ],
        },
        {
            "step_id": "export_registered_model_text",
            "argv": [
                "colmap",
                "model_converter",
                "--input_path",
                "workspace/sparse/0",
                "--output_path",
                "workspace/sparse_text",
                "--output_type",
                "TXT",
                *common,
            ],
        },
    ]
    plan = {
        "schema_version": NATIVE_360_COLMAP_PLAN_SCHEMA_VERSION,
        "stable_run_identity": stable_run_identity,
        "source_capture_identity": dataset.get("source_capture_identity"),
        "source_capture_digest": dataset.get("source_capture_digest"),
        "original_file_references": dataset.get("original_file_references"),
        "producing_method": "blueprint.native_360_colmap_plan_compiler",
        "implementation_version": NATIVE_360_COLMAP_PLAN_IMPLEMENTATION_VERSION,
        "container_image_digest": pose["container_image_digest"],
        "source_commit_sha": pose["source_commit_sha"],
        "input_digests": [
            {"artifact_id": "reconstruction_dataset", "digest": dataset_digest},
            {"artifact_id": "candidate_dataset", "digest": candidate_digest},
            {"artifact_id": "camera_rig_declaration", "digest": rig_digest},
            {
                "artifact_id": "camera_rig_validation",
                "digest": rig_result["camera_rig_validation_result_digest"],
            },
            {
                "artifact_id": "pose_estimation_request",
                "digest": pose["pose_estimation_request_digest"],
            },
            {"artifact_id": "front_valid_pixel_mask", "digest": masks["front"]["digest"]},
            {"artifact_id": "rear_valid_pixel_mask", "digest": masks["rear"]["digest"]},
        ],
        "output_digests": [],
        "train_heldout_split_digest": dataset["train_heldout_split_digest"],
        "camera_calibration_binding": dataset["camera_calibration_binding"],
        "coordinate_frame_declaration": dict(coordinate),
        "units": "source_pixels_seconds_and_declared_rig_meters",
        "metric_scale_status": "anchor_required",
        "provider_runtime_identity": pose["provider_runtime_identity"],
        "authority_used": pose["authority_used"],
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": [
            "plan_not_executed",
            "camera_trajectory_not_established",
            "metric_scale_anchor_required",
        ],
        "blockers": [],
        "proof_effect": "none",
        "claim_ceiling": "execution_plan_only",
        "parent_artifact_or_event": {
            "pose_estimation_request_digest": pose["pose_estimation_request_digest"]
        },
        "timestamp": timestamp,
        "pose_estimation_request_digest": pose["pose_estimation_request_digest"],
        "candidate_dataset_digest": candidate_digest,
        "camera_rig_validation_result_digest": rig_result["camera_rig_validation_result_digest"],
        "workspace_layout": {
            "root": "workspace",
            "images": "workspace/images",
            "masks": "workspace/masks",
            "database": "workspace/database.db",
            "rig_config": "workspace/rig_config.json",
            "sparse_output": "workspace/sparse",
            "sparse_text_output": "workspace/sparse_text",
        },
        "image_materialization": materialization,
        "mask_materialization": mask_materialization,
        "rig_config": rig_config,
        "rig_config_digest": canonical_digest({"rig_config": rig_config}),
        "commands": commands,
        "execution_bounds": {
            "timeout_seconds": pose["timeout_seconds"],
            "hard_ttl_seconds": pose.get("hard_ttl_seconds"),
            "retry_cap": pose.get("retry_cap"),
            "spend_cap_usd": pose["spend_cap_usd"],
        },
        "expected_outputs": [
            "workspace/database.db",
            "workspace/sparse",
            "workspace/sparse_text",
            "workspace/logs",
        ],
        "shell_invocation_allowed": False,
        "network_access_allowed": False,
        "hidden_heldout_access_allowed": False,
        "candidate_can_change_split": False,
        "agent_can_change_calibration": False,
        "plan_executed": False,
        "camera_trajectory_proven": False,
        "metric_scale_proven": False,
        "appearance_reconstruction_proven": False,
        "collision_geometry_proven": False,
        "isaac_compatibility_proven": False,
        "native_360_colmap_execution_plan_digest": None,
    }
    plan["deterministic_configuration_digest"] = canonical_digest(
        {
            "image_materialization": materialization,
            "mask_materialization": mask_materialization,
            "rig_config": rig_config,
            "commands": commands,
        }
    )
    plan["native_360_colmap_execution_plan_digest"] = canonical_digest(
        plan, digest_field="native_360_colmap_execution_plan_digest"
    )
    return json.loads(canonical_json(plan))


__all__ = [
    "NATIVE_360_COLMAP_PLAN_IMPLEMENTATION_VERSION",
    "NATIVE_360_COLMAP_PLAN_SCHEMA_VERSION",
    "Native360ColmapPlanError",
    "compile_native_360_colmap_execution_plan",
]
