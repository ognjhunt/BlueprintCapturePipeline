"""Robot camera profiles and initial policy-observation source resolution."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .common import ensure_dir, read_json_any, resolve_gs_uri_to_path, write_json
from .launch_proof_policy import production_launch_mode
from .synthesis.depth_splat import depth_splat, load_depth_png


ROBOT_CAMERA_PROFILE_REGISTRY_SCHEMA_VERSION = "robot_camera_profile_registry.v1"
ROBOT_CAMERA_PROFILE_LAUNCH_READINESS_SCHEMA_VERSION = (
    "robot_camera_profile_launch_readiness.v1"
)
ROBOT_POV_OBSERVATION_CANDIDATE_SET_SCHEMA_VERSION = (
    "robot_pov_observation_candidate_set.v1"
)
SELECTED_INITIAL_POLICY_OBSERVATION_SCHEMA_VERSION = "selected_initial_policy_observation.v1"
INITIAL_OBSERVATION_SOURCE_QA_SCHEMA_VERSION = "initial_policy_observation_source_qa.v1"
INITIAL_OBSERVATION_RECAPTURE_GUIDANCE_SCHEMA_VERSION = (
    "initial_policy_observation_recapture_guidance.v1"
)

POLICY_OBSERVATION_SCHEMA_ID = "blueprint.robot_eval.observation.v1"
POLICY_OBSERVATION_SCHEMA_REF = "blueprint://schemas/robot_eval_observation.v1"

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_HORIZONTAL_FOV_DEGREES = 75.0
LAUNCH_REQUIRED_CAMERA_CALIBRATION_FIELDS = (
    "owner_provided_intrinsics",
    "owner_provided_extrinsics",
    "owner_provided_fov",
)
SOURCE_PRIORITY = {
    "direct_capture_frame": 100,
    "capture_derived_depth_splat": 80,
    "capture_derived_3dgs": 70,
}


DEFAULT_PROFILE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "robot_profile_id": "unitree_g1",
        "display_name": "Unitree G1",
        "embodiment_type": "humanoid",
        "source": "blueprint_default_robot_camera_profile",
        "cameras": [
            {
                "camera_id": "head_rgbd",
                "display_name": "Head RGB-D",
                "modalities": ["rgb", "depth"],
                "mount": "head",
                "frame_id": "unitree_g1_head_camera",
                "horizontal_fov_degrees": 75.0,
                "intrinsics": {"width": 960, "height": 540},
                "extrinsics": {
                    "reference_frame": "robot_base",
                    "xyz_m": [0.18, 0.0, 1.42],
                    "rpy_rad": [0.0, -0.12, 0.0],
                },
            },
            {
                "camera_id": "chest_rgbd",
                "display_name": "Chest RGB-D",
                "modalities": ["rgb", "depth"],
                "mount": "torso",
                "frame_id": "unitree_g1_chest_camera",
                "horizontal_fov_degrees": 82.0,
                "intrinsics": {"width": 960, "height": 540},
                "extrinsics": {
                    "reference_frame": "robot_base",
                    "xyz_m": [0.12, 0.0, 1.05],
                    "rpy_rad": [0.0, -0.05, 0.0],
                },
            },
        ],
    },
    {
        "robot_profile_id": "unitree_g1_sonic",
        "display_name": "Unitree G1 Sonic",
        "embodiment_type": "humanoid",
        "source": "blueprint_default_robot_camera_profile",
        "cameras": [
            {
                "camera_id": "head_pov",
                "display_name": "Head POV",
                "modalities": ["rgb"],
                "mount": "head",
                "frame_id": "unitree_g1_sonic_head_pov",
                "horizontal_fov_degrees": 75.0,
                "intrinsics": {"width": 960, "height": 540},
                "extrinsics": {
                    "reference_frame": "robot_base",
                    "xyz_m": [0.2, 0.0, 1.48],
                    "rpy_rad": [0.0, -0.1, 0.0],
                },
            }
        ],
    },
    {
        "robot_profile_id": "mobile_manipulator_rgb_v1",
        "display_name": "Mobile manipulator RGB-D v1",
        "embodiment_type": "mobile_manipulator",
        "source": "blueprint_default_robot_camera_profile",
        "cameras": [
            {
                "camera_id": "front_rgbd",
                "display_name": "Front RGB-D",
                "modalities": ["rgb", "depth"],
                "mount": "front_mast",
                "frame_id": "front_rgbd",
                "horizontal_fov_degrees": 75.0,
                "intrinsics": {"width": 960, "height": 540},
                "extrinsics": {
                    "reference_frame": "robot_base",
                    "xyz_m": [0.35, 0.0, 1.2],
                    "rpy_rad": [0.0, -0.08, 0.0],
                },
            },
            {
                "camera_id": "wrist_rgb",
                "display_name": "Wrist RGB",
                "modalities": ["rgb"],
                "mount": "wrist",
                "frame_id": "wrist_rgb",
                "horizontal_fov_degrees": 68.0,
                "intrinsics": {"width": 640, "height": 480},
                "extrinsics": {
                    "reference_frame": "end_effector",
                    "xyz_m": [0.04, 0.0, 0.02],
                    "rpy_rad": [0.0, 0.0, 0.0],
                },
            },
        ],
    },
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _positive_number(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed > 0


def _nonnegative_number(value: Any) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed >= 0


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _stable_id(*parts: Any, prefix: str) -> str:
    raw = ":".join(_string(part) or "unknown" for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _safe_id(value: Any) -> str:
    text = _string(value).lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_") or "unknown"


def _source_claim_boundary() -> dict[str, Any]:
    return {
        "artifact_purpose": "initial_policy_observation_source_selection",
        "raw_capture_evidence_remains_authoritative": True,
        "capture_derived_synthesis_is_support_artifact": True,
        "synthetic_fallback_allowed": False,
        "synthetic_fallback_is_support_artifact": False,
        "robot_policy_execution_proven": False,
        "simulator_execution_proven": False,
        "physical_robot_sensor_proof": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "non_ranking_operational_claim_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def _raw_intrinsics(camera: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(camera.get("intrinsics") or camera.get("camera_intrinsics"))


def _owner_intrinsics_provided(camera: Mapping[str, Any]) -> bool:
    intrinsics = _raw_intrinsics(camera)
    resolution = _mapping(camera.get("resolution"))
    width = intrinsics.get("width") or intrinsics.get("image_width") or resolution.get("width")
    height = intrinsics.get("height") or intrinsics.get("image_height") or resolution.get("height")
    return (
        _positive_number(width)
        and _positive_number(height)
        and _positive_number(intrinsics.get("fx"))
        and _positive_number(intrinsics.get("fy"))
        and _nonnegative_number(intrinsics.get("cx"))
        and _nonnegative_number(intrinsics.get("cy"))
    )


def _raw_extrinsics(camera: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(
        camera.get("extrinsics")
        or camera.get("camera_extrinsics")
        or camera.get("robot_from_camera")
        or camera.get("camera_from_robot")
    )


def _vector_present(value: Any, *, length: int) -> bool:
    values = _list(value)
    if len(values) < length:
        return False
    return all(_float(values[index]) is not None for index in range(length))


def _owner_extrinsics_provided(camera: Mapping[str, Any]) -> bool:
    extrinsics = _raw_extrinsics(camera)
    if not extrinsics:
        return False
    matrix_value = (
        extrinsics.get("matrix")
        or extrinsics.get("T_robot_camera")
        or extrinsics.get("T_base_camera")
        or extrinsics.get("robot_from_camera")
        or extrinsics.get("camera_from_robot")
    )
    matrix_present = _matrix(matrix_value) is not None
    translation = (
        extrinsics.get("xyz_m")
        or extrinsics.get("xyz")
        or extrinsics.get("translation_m")
        or extrinsics.get("translation")
    )
    rotation = (
        extrinsics.get("rpy_rad")
        or extrinsics.get("rpy")
        or extrinsics.get("rotation_rpy_rad")
        or extrinsics.get("rotation")
        or extrinsics.get("quaternion_xyzw")
        or extrinsics.get("quaternion")
    )
    pose_vectors_present = _vector_present(translation, length=3) and _vector_present(
        rotation,
        length=3,
    )
    reference_frame_present = bool(_string(extrinsics.get("reference_frame")))
    return reference_frame_present and (matrix_present or pose_vectors_present)


def _fov_value(camera: Mapping[str, Any], *, axis: str) -> Any:
    fov = _mapping(camera.get("fov"))
    if axis == "horizontal":
        return (
            camera.get("horizontal_fov_degrees")
            or camera.get("horizontalFovDegrees")
            or fov.get("horizontal_degrees")
            or fov.get("horizontalDegrees")
            or fov.get("horizontal")
        )
    return (
        camera.get("vertical_fov_degrees")
        or camera.get("verticalFovDegrees")
        or fov.get("vertical_degrees")
        or fov.get("verticalDegrees")
        or fov.get("vertical")
    )


def _owner_fov_provided(camera: Mapping[str, Any]) -> bool:
    return _positive_number(_fov_value(camera, axis="horizontal")) and _positive_number(
        _fov_value(camera, axis="vertical")
    )


def _smoke_only_source(source: str) -> bool:
    normalized = source.strip().lower()
    return normalized in {
        "blueprint_default_robot_camera_profile",
        "robot_eval_dataset.scenario_cards",
        "active_profile_default",
    } or normalized.startswith("blueprint_default_")


def _horizontal_fov(camera: Mapping[str, Any], intrinsics: Mapping[str, Any]) -> float:
    fov = camera.get("horizontal_fov_degrees") or camera.get("horizontalFovDegrees")
    fov = fov or camera.get("fov_degrees") or camera.get("fovDegrees")
    fov = fov or _mapping(camera.get("fov")).get("horizontal_degrees")
    fov = fov or _mapping(camera.get("fov")).get("horizontalDegrees")
    parsed = _float(fov)
    if parsed and parsed > 0:
        return round(parsed, 6)
    width = _float(intrinsics.get("width"))
    fx = _float(intrinsics.get("fx"))
    if width and fx and fx > 0:
        return round(math.degrees(2.0 * math.atan(width / (2.0 * fx))), 6)
    return DEFAULT_HORIZONTAL_FOV_DEGREES


def _vertical_fov(camera: Mapping[str, Any], intrinsics: Mapping[str, Any]) -> float:
    fov = camera.get("vertical_fov_degrees") or camera.get("verticalFovDegrees")
    fov = fov or _mapping(camera.get("fov")).get("vertical_degrees")
    fov = fov or _mapping(camera.get("fov")).get("verticalDegrees")
    parsed = _float(fov)
    if parsed and parsed > 0:
        return round(parsed, 6)
    height = _float(intrinsics.get("height"))
    fy = _float(intrinsics.get("fy"))
    if height and fy and fy > 0:
        return round(math.degrees(2.0 * math.atan(height / (2.0 * fy))), 6)
    width = _float(intrinsics.get("width")) or DEFAULT_WIDTH
    return round(
        math.degrees(
            2.0
            * math.atan(
                (height or DEFAULT_HEIGHT)
                / max(width, 1.0)
                * math.tan(math.radians(_horizontal_fov(camera, intrinsics)) / 2.0)
            )
        ),
        6,
    )


def _intrinsics_from_camera(camera: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = _raw_intrinsics(camera)
    resolution = _mapping(camera.get("resolution"))
    width = _int(
        intrinsics.get("width")
        or intrinsics.get("image_width")
        or resolution.get("width")
        or camera.get("width"),
        DEFAULT_WIDTH,
    )
    height = _int(
        intrinsics.get("height")
        or intrinsics.get("image_height")
        or resolution.get("height")
        or camera.get("height"),
        DEFAULT_HEIGHT,
    )
    horizontal = _horizontal_fov(camera, {**intrinsics, "width": width, "height": height})
    fx = _float(intrinsics.get("fx"))
    if not fx:
        fx = (float(width) / 2.0) / math.tan(math.radians(horizontal) / 2.0)
    fy = _float(intrinsics.get("fy"))
    if not fy:
        vertical = _vertical_fov(camera, {**intrinsics, "width": width, "height": height})
        fy = (float(height) / 2.0) / math.tan(math.radians(vertical) / 2.0)
    cx = _float(intrinsics.get("cx"), float(width) / 2.0)
    cy = _float(intrinsics.get("cy"), float(height) / 2.0)
    owner_provided = _owner_intrinsics_provided(camera)
    return {
        "width": width,
        "height": height,
        "fx": round(float(fx), 6),
        "fy": round(float(fy), 6),
        "cx": round(float(cx), 6),
        "cy": round(float(cy), 6),
        "camera_model": _string(intrinsics.get("camera_model") or intrinsics.get("model"))
        or "pinhole",
        "source": _string(intrinsics.get("source"))
        or (
            "owner_provided_robot_camera_profile"
            if owner_provided
            else "camera_profile_or_fov_derived"
        ),
        "owner_provided": owner_provided,
        "smoke_only": not owner_provided,
    }


def _pose_vector(value: Any, *, length: int, default: Sequence[float]) -> list[float]:
    raw = value
    if isinstance(value, Mapping):
        raw = value.get("xyz_m") or value.get("xyz") or value.get("rpy_rad") or value.get("rpy")
    items = _list(raw)
    out: list[float] = []
    for index in range(length):
        out.append(round(_float(items[index] if index < len(items) else None, default[index]) or 0.0, 6))
    return out


def _extrinsics_from_camera(camera: Mapping[str, Any]) -> dict[str, Any]:
    extrinsics = _raw_extrinsics(camera)
    xyz = _pose_vector(
        extrinsics.get("xyz_m") or extrinsics.get("xyz") or extrinsics.get("translation_m"),
        length=3,
        default=[0.0, 0.0, 1.2],
    )
    rpy = _pose_vector(
        extrinsics.get("rpy_rad") or extrinsics.get("rpy") or extrinsics.get("rotation_rpy_rad"),
        length=3,
        default=[0.0, 0.0, 0.0],
    )
    owner_provided = _owner_extrinsics_provided(camera)
    return {
        "reference_frame": _string(extrinsics.get("reference_frame")) or "robot_base",
        "child_frame": _string(
            extrinsics.get("child_frame")
            or camera.get("frame_id")
            or camera.get("camera_id")
            or camera.get("id")
        )
        or "camera",
        "xyz_m": xyz,
        "rpy_rad": rpy,
        "source": _string(extrinsics.get("source"))
        or (
            "owner_provided_robot_camera_profile"
            if owner_provided
            else "camera_profile_or_default_mount"
        ),
        "owner_provided": owner_provided,
        "smoke_only": not owner_provided,
    }


def _camera_calibration_contract(camera: Mapping[str, Any]) -> dict[str, Any]:
    owner_intrinsics = _owner_intrinsics_provided(camera)
    owner_extrinsics = _owner_extrinsics_provided(camera)
    owner_fov = _owner_fov_provided(camera)
    flags = {
        "owner_provided_intrinsics": owner_intrinsics,
        "owner_provided_extrinsics": owner_extrinsics,
        "owner_provided_fov": owner_fov,
    }
    missing = [
        field for field in LAUNCH_REQUIRED_CAMERA_CALIBRATION_FIELDS if not bool(flags[field])
    ]
    return {
        "launch_required_fields": list(LAUNCH_REQUIRED_CAMERA_CALIBRATION_FIELDS),
        **flags,
        "owner_provided_calibration_complete": not missing,
        "missing_launch_fields": missing,
        "smoke_only": bool(missing),
        "default_or_derived_values_allowed_for_smoke_only": bool(missing),
        "launch_mode_blocks_without_owner_calibration": bool(missing),
    }


def _normalize_camera(camera: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    camera_id = _string(
        camera.get("camera_id")
        or camera.get("cameraId")
        or camera.get("id")
        or camera.get("name")
        or f"camera_{index}"
    )
    intrinsics = _intrinsics_from_camera(camera)
    calibration_contract = _camera_calibration_contract(camera)
    return {
        "camera_id": camera_id,
        "display_name": _string(camera.get("display_name") or camera.get("name") or camera_id),
        "modalities": [
            _string(item)
            for item in _list(camera.get("modalities") or camera.get("sensors"))
            if _string(item)
        ]
        or ["rgb"],
        "mount": _string(camera.get("mount") or camera.get("mount_point")) or None,
        "frame_id": _string(camera.get("frame_id") or camera.get("frame") or camera_id),
        "intrinsics": intrinsics,
        "extrinsics": _extrinsics_from_camera(camera),
        "horizontal_fov_degrees": _horizontal_fov(camera, intrinsics),
        "vertical_fov_degrees": _vertical_fov(camera, intrinsics),
        "source": _string(camera.get("source")) or "robot_camera_profile",
        "calibration_contract": calibration_contract,
        "smoke_only": calibration_contract["smoke_only"],
    }


def _profile_id(profile: Mapping[str, Any], *, fallback: str) -> str:
    return (
        _string(profile.get("robot_profile_id"))
        or _string(profile.get("robotProfileId"))
        or _string(profile.get("profile_id"))
        or _string(profile.get("id"))
        or fallback
    )


def _normalize_profile(profile: Mapping[str, Any], *, fallback: str, source: str) -> dict[str, Any]:
    profile_id = _profile_id(profile, fallback=fallback)
    cameras = _list(
        profile.get("cameras")
        or profile.get("camera_profiles")
        or profile.get("cameraProfiles")
        or profile.get("sensors")
    )
    default_cameras_used = False
    normalized_cameras = [
        _normalize_camera(_mapping(camera), index=index)
        for index, camera in enumerate(cameras, start=1)
        if isinstance(camera, Mapping)
    ]
    if not normalized_cameras:
        default_cameras_used = True
        default = next(
            (
                item
                for item in DEFAULT_PROFILE_DEFINITIONS
                if _profile_id(item, fallback="") == profile_id
            ),
            DEFAULT_PROFILE_DEFINITIONS[0],
        )
        normalized_cameras = [
            _normalize_camera(_mapping(camera), index=index)
            for index, camera in enumerate(default["cameras"], start=1)
        ]
    primary_camera_id = _string(
        profile.get("primary_camera_id") or profile.get("primaryCameraId")
    ) or normalized_cameras[0]["camera_id"]
    profile_source = _string(profile.get("source")) or source
    camera_missing: dict[str, list[str]] = {}
    for camera in normalized_cameras:
        missing = [
            _string(item)
            for item in _list(
                _mapping(camera.get("calibration_contract")).get("missing_launch_fields")
            )
            if _string(item)
        ]
        if missing:
            camera_missing[_string(camera.get("camera_id"))] = missing
    smoke_only = bool(default_cameras_used or _smoke_only_source(profile_source) or camera_missing)
    launch_ready = bool(normalized_cameras) and not smoke_only
    return {
        "robot_profile_id": profile_id,
        "display_name": _string(profile.get("display_name") or profile.get("name") or profile_id),
        "embodiment_type": _string(
            profile.get("embodiment_type") or profile.get("embodiment") or profile.get("type")
        )
        or "unknown",
        "primary_camera_id": primary_camera_id,
        "camera_count": len(normalized_cameras),
        "cameras": normalized_cameras,
        "action_space": _mapping(profile.get("action_space") or profile.get("actionSpace")),
        "source": profile_source,
        "smoke_only": smoke_only,
        "calibration_contract": {
            "launch_required_fields": list(LAUNCH_REQUIRED_CAMERA_CALIBRATION_FIELDS),
            "owner_provided_intrinsics_required_for_launch": True,
            "owner_provided_extrinsics_required_for_launch": True,
            "owner_provided_fov_required_for_launch": True,
            "default_profile_used": default_cameras_used or _smoke_only_source(profile_source),
            "default_profile_smoke_only": default_cameras_used or _smoke_only_source(profile_source),
            "smoke_only": smoke_only,
            "launch_ready": launch_ready,
            "camera_count": len(normalized_cameras),
            "launch_ready_camera_count": sum(
                1
                for camera in normalized_cameras
                if not _mapping(camera.get("calibration_contract")).get("missing_launch_fields")
            ),
            "camera_missing_launch_fields": camera_missing,
            "missing_launch_fields": sorted(
                {field for fields in camera_missing.values() for field in fields}
            ),
            "launch_mode_blocks_without_owner_calibration": not launch_ready,
        },
        "claim_boundary": "robot_camera_profile_defines_eval_input_contract_not_rank_fidelity",
    }


def build_robot_camera_profile_launch_readiness(
    *,
    registry: Mapping[str, Any],
    generated_at: str,
    launch_mode: bool | None = None,
) -> dict[str, Any]:
    """Validate owner-provided robot camera calibration for launch-mode use."""

    launch_mode_enabled = production_launch_mode() if launch_mode is None else bool(launch_mode)
    profiles = [dict(item) for item in registry.get("profiles", []) if isinstance(item, Mapping)]
    profile_summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    launch_ready_count = 0
    smoke_only_count = 0
    default_smoke_only_count = 0
    for profile in profiles:
        profile_id = _string(profile.get("robot_profile_id"))
        contract = _mapping(profile.get("calibration_contract"))
        launch_ready = bool(contract.get("launch_ready"))
        smoke_only = bool(contract.get("smoke_only") or profile.get("smoke_only"))
        default_smoke_only = bool(contract.get("default_profile_smoke_only"))
        if launch_ready:
            launch_ready_count += 1
        if smoke_only:
            smoke_only_count += 1
        if default_smoke_only:
            default_smoke_only_count += 1
            blockers.append(f"default_robot_camera_profile_smoke_only:{profile_id}")
        camera_summaries: list[dict[str, Any]] = []
        for camera in profile.get("cameras", []) or []:
            if not isinstance(camera, Mapping):
                continue
            camera_id = _string(camera.get("camera_id"))
            camera_contract = _mapping(camera.get("calibration_contract"))
            missing = [
                _string(item)
                for item in _list(camera_contract.get("missing_launch_fields"))
                if _string(item)
            ]
            for field in missing:
                blockers.append(f"missing_{field}:{profile_id}:{camera_id}")
            camera_summaries.append(
                {
                    "camera_id": camera_id,
                    "owner_provided_intrinsics": bool(
                        camera_contract.get("owner_provided_intrinsics")
                    ),
                    "owner_provided_extrinsics": bool(
                        camera_contract.get("owner_provided_extrinsics")
                    ),
                    "owner_provided_fov": bool(camera_contract.get("owner_provided_fov")),
                    "launch_ready": not missing,
                    "smoke_only": bool(camera_contract.get("smoke_only")),
                    "missing_launch_fields": missing,
                }
            )
        profile_summaries.append(
            {
                "robot_profile_id": profile_id,
                "source": _string(profile.get("source")),
                "camera_count": int(profile.get("camera_count") or len(camera_summaries)),
                "launch_ready": launch_ready,
                "smoke_only": smoke_only,
                "default_profile_smoke_only": default_smoke_only,
                "missing_launch_fields": _list(contract.get("missing_launch_fields")),
                "cameras": camera_summaries,
            }
        )
    unique_blockers = sorted({blocker for blocker in blockers if blocker})
    all_profiles_launch_ready = bool(profiles) and launch_ready_count == len(profiles)
    status = (
        "ready"
        if all_profiles_launch_ready
        else "blocked"
        if launch_mode_enabled
        else "smoke_only"
    )
    return {
        "schema_version": ROBOT_CAMERA_PROFILE_LAUNCH_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "launch_mode": launch_mode_enabled,
        "launch_mode_fail_closed": True,
        "profile_count": len(profiles),
        "launch_ready_profile_count": launch_ready_count,
        "smoke_only_profile_count": smoke_only_count,
        "default_smoke_only_profile_count": default_smoke_only_count,
        "all_profiles_launch_ready": all_profiles_launch_ready,
        "owner_provided_intrinsics_required_for_launch": True,
        "owner_provided_extrinsics_required_for_launch": True,
        "owner_provided_fov_required_for_launch": True,
        "defaults_are_smoke_only": default_smoke_only_count > 0,
        "blockers": unique_blockers if launch_mode_enabled or unique_blockers else [],
        "profiles": profile_summaries,
        "claim_boundary": {
            **_source_claim_boundary(),
            "artifact_purpose": "robot_camera_profile_launch_readiness_gate",
            "owner_provided_camera_calibration_required_for_launch": True,
            "defaults_can_only_support_smoke_artifacts": True,
            "launch_ready_does_not_prove_robot_policy_execution": True,
            "launch_ready_does_not_prove_generated_world_rank_fidelity": True,
        },
    }


def _scenario_profile_ids(scenario_cards: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in scenario_cards.get("cards", []) or []:
        if not isinstance(item, Mapping):
            continue
        profile_id = _string(
            item.get("robot_profile_id") or item.get("robotProfileId") or item.get("robot_profile")
        )
        if profile_id and profile_id not in ids:
            ids.append(profile_id)
    return ids


def build_robot_camera_profile_registry(
    *,
    job_request: Mapping[str, Any],
    scenario_cards: Mapping[str, Any] | None = None,
    generated_at: str,
) -> dict[str, Any]:
    """Build a normalized multi-profile robot camera registry for an eval job."""

    cards = _mapping(scenario_cards)
    raw_profiles: list[Mapping[str, Any]] = []
    single_profile = _mapping(job_request.get("robot_profile") or job_request.get("robotProfile"))
    if single_profile:
        raw_profiles.append(single_profile)
    for key in ("robot_profiles", "robotProfiles"):
        for item in _list(job_request.get(key)):
            if isinstance(item, Mapping):
                raw_profiles.append(item)

    seen: set[str] = set()
    profiles: list[dict[str, Any]] = []
    for index, profile in enumerate(raw_profiles, start=1):
        normalized = _normalize_profile(
            profile,
            fallback=f"request_robot_{index}",
            source="job_request",
        )
        if normalized["robot_profile_id"] in seen:
            continue
        profiles.append(normalized)
        seen.add(normalized["robot_profile_id"])

    scenario_added: list[str] = []
    for profile_id in _scenario_profile_ids(cards):
        if profile_id in seen:
            continue
        default_profile = next(
            (
                item
                for item in DEFAULT_PROFILE_DEFINITIONS
                if _profile_id(item, fallback="") == profile_id
            ),
            {"robot_profile_id": profile_id, "display_name": profile_id},
        )
        profiles.append(
            _normalize_profile(
                default_profile,
                fallback=profile_id,
                source="robot_eval_dataset.scenario_cards",
            )
        )
        seen.add(profile_id)
        scenario_added.append(profile_id)

    if not profiles:
        default = _normalize_profile(
            DEFAULT_PROFILE_DEFINITIONS[0],
            fallback="unitree_g1",
            source="blueprint_default_robot_camera_profile",
        )
        profiles.append(default)
        seen.add(default["robot_profile_id"])

    active_profile_id = (
        _profile_id(single_profile, fallback="")
        or _string(job_request.get("robot_profile_id") or job_request.get("robotProfileId"))
        or profiles[0]["robot_profile_id"]
    )
    if active_profile_id not in seen:
        profiles.insert(
            0,
            _normalize_profile(
                {"robot_profile_id": active_profile_id},
                fallback=active_profile_id,
                source="active_profile_default",
            ),
        )
    return {
        "schema_version": ROBOT_CAMERA_PROFILE_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "active_robot_profile_id": active_profile_id,
        "profile_count": len(profiles),
        "profiles": profiles,
        "scenario_profile_ids_added": scenario_added,
        "claim_boundary": _source_claim_boundary(),
    }


def _read_optional_any(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return read_json_any(path)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in (
        "frames",
        "records",
        "items",
        "references",
        "capture_frames",
        "captureFrames",
        "frame_index",
        "frameIndex",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return [dict(payload)] if payload else []


def _normalize_geometry_intrinsics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    intrinsics = _mapping(payload)
    width = _int(intrinsics.get("width") or intrinsics.get("image_width"), DEFAULT_WIDTH)
    height = _int(intrinsics.get("height") or intrinsics.get("image_height"), DEFAULT_HEIGHT)
    fx = _float(intrinsics.get("fx"))
    fy = _float(intrinsics.get("fy"))
    if not fx or not fy:
        horizontal = _float(intrinsics.get("horizontal_fov_degrees"), DEFAULT_HORIZONTAL_FOV_DEGREES)
        fx = (float(width) / 2.0) / math.tan(math.radians(float(horizontal)) / 2.0)
        fy = fx
    return {
        "width": width,
        "height": height,
        "fx": round(float(fx), 6),
        "fy": round(float(fy), 6),
        "cx": round(_float(intrinsics.get("cx"), float(width) / 2.0) or float(width) / 2.0, 6),
        "cy": round(_float(intrinsics.get("cy"), float(height) / 2.0) or float(height) / 2.0, 6),
        "camera_model": _string(intrinsics.get("camera_model") or intrinsics.get("model"))
        or "pinhole",
        "source": "pipeline.geometry.camera.intrinsics",
    }


def _frame_record_key(record: Mapping[str, Any]) -> str:
    frame_id = _string(record.get("frame_id") or record.get("frameId"))
    if frame_id:
        return frame_id
    frame_index = record.get("frame_index") or record.get("frameIndex")
    if frame_index is None:
        return ""
    return _string(frame_index)


def _load_geometry_frame_records(capture_root: Path) -> list[dict[str, Any]]:
    frame_index_path = capture_root / "pipeline" / "geometry" / "frames" / "frame_index.jsonl"
    if not frame_index_path.is_file():
        return []

    intrinsics = _normalize_geometry_intrinsics(
        _read_optional_any(capture_root / "pipeline" / "geometry" / "camera" / "intrinsics.json")
    )
    poses_by_key: dict[str, dict[str, Any]] = {}
    for pose in _read_jsonl(capture_root / "pipeline" / "geometry" / "camera" / "poses.jsonl"):
        for key in {
            _frame_record_key(pose),
            _string(pose.get("frame_index") or pose.get("frameIndex")),
        }:
            if key:
                poses_by_key[key] = pose

    records: list[dict[str, Any]] = []
    for record in _read_jsonl(frame_index_path):
        enriched = dict(record)
        key = _frame_record_key(enriched)
        pose = poses_by_key.get(key) or poses_by_key.get(_string(enriched.get("frame_index")))
        if pose:
            for pose_key in ("world_from_camera", "camera_from_world", "T_world_camera"):
                if pose_key in pose and pose_key not in enriched:
                    enriched[pose_key] = pose[pose_key]
        if intrinsics and "intrinsics" not in enriched:
            enriched["intrinsics"] = intrinsics
        enriched.setdefault("camera_id", "capture_geometry_rgbd")
        enriched.setdefault("camera_frame", "capture_geometry_rgbd")
        enriched.setdefault("source_kind", "capture_geometry_rgbd_frame")
        enriched.setdefault("geometry_index_path", str(frame_index_path))
        records.append(enriched)
    return records


def _local_reference_path(value: Any, *, capture_root: Path, job_dir: Path) -> Path | None:
    text = _string(value)
    if not text:
        return None
    if text.startswith("file://"):
        return Path(text[7:]).expanduser()
    if text.startswith("gs://"):
        default_gcs_root = capture_root.parents[3] if len(capture_root.parents) > 3 else capture_root
        return resolve_gs_uri_to_path(text, Path(os.getenv("GCS_ROOT", str(default_gcs_root))))
    if "://" in text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    job_candidate = job_dir / path
    if job_candidate.exists():
        return job_candidate
    return capture_root / path


def _path_info(value: Any, *, capture_root: Path, job_dir: Path) -> dict[str, Any]:
    text = _string(value)
    if not text:
        return {}
    local = _local_reference_path(text, capture_root=capture_root, job_dir=job_dir)
    if local is not None and local.exists():
        return {
            "uri": text if "://" in text else None,
            "path": str(local),
            "exists": True,
            "size_bytes": local.stat().st_size if local.is_file() else None,
            "sha256": _sha256(local) if local.is_file() else None,
        }
    return {
        "uri": text if "://" in text else None,
        "path": str(local) if local is not None and "://" not in text else None,
        "exists": False,
        "remote_or_unresolved": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_frame_ref(record: Mapping[str, Any]) -> str:
    for key in (
        "image_path",
        "imagePath",
        "rgb_path",
        "rgbPath",
        "frame_path",
        "framePath",
        "path",
        "uri",
        "image_uri",
        "imageUri",
        "rgb_uri",
        "rgbUri",
        "frame_uri",
        "frameUri",
    ):
        value = _string(record.get(key))
        if value:
            return value
    return ""


def _depth_ref(record: Mapping[str, Any]) -> str:
    for key in (
        "depth_path",
        "depthPath",
        "depth_uri",
        "depthUri",
        "depth_map_path",
        "depthMapPath",
    ):
        value = _string(record.get(key))
        if value:
            return value
    return ""


def _record_intrinsics(record: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = _mapping(record.get("intrinsics") or record.get("camera_intrinsics"))
    width = _int(
        intrinsics.get("width")
        or intrinsics.get("image_width")
        or record.get("width")
        or record.get("image_width"),
        DEFAULT_WIDTH,
    )
    height = _int(
        intrinsics.get("height")
        or intrinsics.get("image_height")
        or record.get("height")
        or record.get("image_height"),
        DEFAULT_HEIGHT,
    )
    fx = _float(intrinsics.get("fx"))
    fy = _float(intrinsics.get("fy"))
    if not fx or not fy:
        h_fov = _float(record.get("horizontal_fov_degrees"), DEFAULT_HORIZONTAL_FOV_DEGREES)
        fx = (float(width) / 2.0) / math.tan(math.radians(float(h_fov)) / 2.0)
        fy = fx
    return {
        "width": width,
        "height": height,
        "fx": round(float(fx), 6),
        "fy": round(float(fy), 6),
        "cx": round(_float(intrinsics.get("cx"), float(width) / 2.0) or float(width) / 2.0, 6),
        "cy": round(_float(intrinsics.get("cy"), float(height) / 2.0) or float(height) / 2.0, 6),
        "camera_model": _string(intrinsics.get("camera_model") or intrinsics.get("model"))
        or "pinhole",
        "source": _string(intrinsics.get("source")) or "capture_frame_record",
    }


def _record_has_intrinsics(record: Mapping[str, Any]) -> bool:
    intrinsics = _mapping(record.get("intrinsics") or record.get("camera_intrinsics"))
    if not intrinsics:
        return False
    required = ("fx", "fy", "cx", "cy")
    has_calibration = all(_float(intrinsics.get(key)) is not None for key in required)
    has_size = bool(
        _int(intrinsics.get("width") or intrinsics.get("image_width"))
        and _int(intrinsics.get("height") or intrinsics.get("image_height"))
    )
    return bool(has_calibration and has_size)


def _matrix(value: Any) -> list[list[float]] | None:
    if isinstance(value, Mapping):
        value = value.get("matrix") or value.get("value")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
        if len(rows) == 16:
            try:
                flat = [float(item) for item in rows]
            except (TypeError, ValueError):
                return None
            return [flat[index : index + 4] for index in range(0, 16, 4)]
        if len(rows) == 4 and all(isinstance(row, Sequence) for row in rows):
            out: list[list[float]] = []
            try:
                for row in rows:
                    items = list(row)  # type: ignore[arg-type]
                    if len(items) != 4:
                        return None
                    out.append([float(item) for item in items])
            except (TypeError, ValueError):
                return None
            return out
    return None


def _record_pose(record: Mapping[str, Any]) -> list[list[float]] | None:
    for key in (
        "T_world_camera",
        "world_from_camera",
        "worldFromCamera",
        "camera_pose",
        "cameraPose",
        "pose",
    ):
        pose = _matrix(record.get(key))
        if pose is not None:
            return pose
    return None


def _pose_from_xyz(xyz: Sequence[float]) -> list[list[float]]:
    x = float(xyz[0]) if len(xyz) > 0 else 0.0
    y = float(xyz[1]) if len(xyz) > 1 else 0.0
    z = float(xyz[2]) if len(xyz) > 2 else 0.0
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def _target_pose(
    *,
    job_request: Mapping[str, Any],
    eval_run: Mapping[str, Any],
    camera: Mapping[str, Any],
    reference_pose: list[list[float]] | None = None,
) -> tuple[list[list[float]], str]:
    initial = _mapping(
        job_request.get("initial_observation")
        or job_request.get("initialObservation")
        or job_request.get("initial_policy_observation")
        or job_request.get("initialPolicyObservation")
    )
    explicit = _matrix(
        initial.get("target_camera_pose")
        or initial.get("targetCameraPose")
        or initial.get("T_world_camera")
        or initial.get("world_from_camera")
    )
    if explicit is not None:
        return explicit, "job_request.initial_observation.target_camera_pose"
    mutation = _mapping(eval_run.get("concrete_mutation"))
    spawn = (
        eval_run.get("spawn_pose")
        or eval_run.get("start_pose")
        or mutation.get("spawn_pose")
        or mutation.get("start_pose")
    )
    if isinstance(spawn, Sequence) and not isinstance(spawn, (str, bytes)) and len(spawn) >= 2:
        xyz = [float(spawn[0]), float(spawn[1]), float(spawn[2]) if len(spawn) > 2 else 0.0]
        camera_xyz = _mapping(camera.get("extrinsics")).get("xyz_m") or [0.0, 0.0, 0.0]
        camera_offset = [float(item) for item in camera_xyz[:3]]
        return (
            _pose_from_xyz(
                [
                    xyz[0] + camera_offset[0],
                    xyz[1] + camera_offset[1],
                    xyz[2] + camera_offset[2],
                ]
            ),
            "scenario_eval_run.spawn_pose_plus_camera_extrinsics",
        )
    if reference_pose is not None:
        return reference_pose, "reference_capture_pose_fallback"
    return _pose_from_xyz(_mapping(camera.get("extrinsics")).get("xyz_m") or [0.0, 0.0, 1.2]), "camera_profile_extrinsics_fallback"


def _load_capture_frame_records(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("capture_frames", "captureFrames", "frame_index", "frameIndex"):
        records.extend(_records_from_payload(job_request.get(key)))
    initial = _mapping(
        job_request.get("initial_observation")
        or job_request.get("initialObservation")
        or job_request.get("initial_policy_observation")
        or job_request.get("initialPolicyObservation")
    )
    for key in ("capture_frames", "captureFrames", "frame_index", "frameIndex"):
        records.extend(_records_from_payload(initial.get(key)))

    refs = [
        initial.get("capture_frame_index_uri"),
        initial.get("captureFrameIndexUri"),
        job_request.get("capture_frame_index_uri"),
        job_request.get("captureFrameIndexUri"),
    ]
    for ref in refs:
        path = _local_reference_path(ref, capture_root=capture_root, job_dir=job_dir)
        if path and path.is_file():
            if path.suffix.lower() == ".jsonl":
                records.extend(_read_jsonl(path))
            else:
                records.extend(_records_from_payload(_read_optional_any(path)))

    candidate_paths = [
        job_dir / "capture_frame_index.json",
        job_dir / "capture_frames.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "capture_frame_index.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "robot_camera_frame_index.json",
        capture_root / "pipeline" / "geometry" / "frames" / "frame_index.jsonl",
        capture_root / "pipeline" / "capture_frames" / "frame_index.json",
        capture_root / "pipeline" / "capture_frames" / "frame_index.jsonl",
        capture_root / "pipeline" / "retrieval_index" / "site_reference_index.jsonl",
        capture_root / "raw" / "frames" / "frame_index.json",
        capture_root / "raw" / "frames" / "frame_index.jsonl",
        capture_root / "raw" / "arkit" / "frame_index.jsonl",
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        if path == capture_root / "pipeline" / "geometry" / "frames" / "frame_index.jsonl":
            records.extend(_load_geometry_frame_records(capture_root))
        elif path.suffix.lower() == ".jsonl":
            records.extend(_read_jsonl(path))
        else:
            records.extend(_records_from_payload(_read_optional_any(path)))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = json.dumps(record, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _select_profile(registry: Mapping[str, Any], profile_id: str | None = None) -> dict[str, Any]:
    wanted = _string(profile_id) or _string(registry.get("active_robot_profile_id"))
    profiles = [dict(item) for item in registry.get("profiles", []) if isinstance(item, Mapping)]
    for profile in profiles:
        if _string(profile.get("robot_profile_id")) == wanted:
            return profile
    return profiles[0] if profiles else {}


def _select_camera(profile: Mapping[str, Any], job_request: Mapping[str, Any]) -> dict[str, Any]:
    initial = _mapping(
        job_request.get("initial_observation")
        or job_request.get("initialObservation")
        or job_request.get("initial_policy_observation")
        or job_request.get("initialPolicyObservation")
    )
    requested = _string(
        initial.get("camera_id")
        or initial.get("cameraId")
        or job_request.get("camera_id")
        or job_request.get("cameraId")
        or profile.get("primary_camera_id")
    )
    cameras = [dict(item) for item in profile.get("cameras", []) if isinstance(item, Mapping)]
    for camera in cameras:
        if _string(camera.get("camera_id")) == requested or _string(camera.get("frame_id")) == requested:
            return camera
    return cameras[0] if cameras else _normalize_camera({}, index=1)


def _record_matches_target(
    record: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    camera: Mapping[str, Any],
) -> tuple[bool, list[str], int]:
    reasons: list[str] = []
    score = SOURCE_PRIORITY["direct_capture_frame"]
    direct_match = False
    record_profile = _string(record.get("robot_profile_id") or record.get("robotProfileId"))
    if record_profile and record_profile == _string(profile.get("robot_profile_id")):
        reasons.append("robot_profile_id_match")
        score += 12
        direct_match = True
    camera_tokens = {
        _string(camera.get("camera_id")),
        _string(camera.get("frame_id")),
        _string(camera.get("mount")),
    }
    camera_tokens = {token for token in camera_tokens if token}
    record_camera = _string(record.get("camera_id") or record.get("cameraId") or record.get("camera"))
    record_frame = _string(record.get("camera_frame") or record.get("frame_id") or record.get("frameId"))
    record_view = _string(record.get("viewpoint") or record.get("source_kind") or record.get("sourceKind")).lower()
    if record_camera in camera_tokens or record_frame in camera_tokens:
        reasons.append("camera_id_or_frame_match")
        score += 20
        direct_match = True
    if "robot_pov" in record_view or "robot_camera" in record_view:
        reasons.append("record_marked_robot_pov")
        score += 18
        direct_match = True
    if _record_pose(record):
        reasons.append("capture_pose_available")
        score += 5
    if _mapping(record.get("intrinsics") or record.get("camera_intrinsics")):
        reasons.append("capture_intrinsics_available")
        score += 5
    return bool(direct_match and _candidate_frame_ref(record)), reasons, score


def _provenance(
    *,
    source_kind: str,
    source_artifacts: Mapping[str, Any] | None = None,
    direct_capture: bool = False,
    capture_derived: bool = False,
    synthetic: bool = False,
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "raw_capture_frame": direct_capture,
        "capture_truth": direct_capture,
        "geometry_truth": False,
        "collision_truth": False,
        "capture_derived": capture_derived,
        "synthetic_fallback": synthetic,
        "raw_capture_evidence_authority_preserved": True,
        "generated_or_synthesized_support_artifact": capture_derived or synthetic,
        "paid_provider_call_performed": False,
        "physical_robot_sensor_proof": False,
        "robot_policy_execution_proven": False,
        "source_artifacts": dict(source_artifacts or {}),
        "claim_boundary": _source_claim_boundary(),
    }


def _direct_capture_candidates(
    *,
    records: Sequence[Mapping[str, Any]],
    capture_root: Path,
    job_dir: Path,
    profile: Mapping[str, Any],
    camera: Mapping[str, Any],
    eval_run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        matches, reasons, score = _record_matches_target(record, profile=profile, camera=camera)
        if not matches:
            continue
        frame_ref = _candidate_frame_ref(record)
        depth_ref = _depth_ref(record)
        frame_info = _path_info(frame_ref, capture_root=capture_root, job_dir=job_dir)
        depth_info = _path_info(depth_ref, capture_root=capture_root, job_dir=job_dir)
        pose = _record_pose(record)
        candidate_id = _stable_id(frame_ref, profile.get("robot_profile_id"), camera.get("camera_id"), prefix="direct")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_kind": "direct_capture_frame",
                "status": "available",
                "selection_score": score,
                "selection_reasons": reasons,
                "robot_profile_id": profile.get("robot_profile_id"),
                "camera_id": camera.get("camera_id"),
                "camera": camera,
                "scenario_eval_run_id": eval_run.get("scenario_eval_run_id"),
                "visual_observation": {
                    "available": bool(frame_ref),
                    "camera_frame_path": frame_info.get("path"),
                    "camera_frame_uri": frame_info.get("uri"),
                    "source_image_path": frame_info.get("path"),
                    "source_image_uri": frame_info.get("uri"),
                    "source_depth_path": depth_info.get("path"),
                    "source_depth_uri": depth_info.get("uri"),
                    "source_pose_available": pose is not None,
                    "source_intrinsics_available": _record_has_intrinsics(record),
                    "direct_capture_frame_match": True,
                    "capture_truth": True,
                    "geometry_truth": bool(pose is not None and _record_has_intrinsics(record)),
                    "collision_truth": False,
                    "capture_derived": False,
                    "synthetic_camera_view": False,
                    "sha256": frame_info.get("sha256"),
                    "size_bytes": frame_info.get("size_bytes"),
                },
                "capture_frame": {
                    "frame_id": _string(record.get("frame_id") or record.get("frameId")) or None,
                    "frame_index": record.get("frame_index") or record.get("frameIndex"),
                    "timestamp": record.get("timestamp") or record.get("timestamp_s"),
                    "camera_id": _string(record.get("camera_id") or record.get("cameraId")) or None,
                    "intrinsics": _record_intrinsics(record),
                    "pose_available": pose is not None,
                    "depth_available": bool(depth_info.get("exists")),
                    "path": frame_info.get("path"),
                    "uri": frame_info.get("uri"),
                },
                "provenance": _provenance(
                    source_kind="direct_capture_frame",
                    direct_capture=True,
                    source_artifacts={"capture_frame_ref": frame_ref},
                ),
            }
        )
    return sorted(candidates, key=lambda item: int(item.get("selection_score") or 0), reverse=True)[:3]


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)
    if data.ndim != 3:
        raise ValueError("image array must be 2D or 3D")
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    if data.shape[2] > 3:
        data = data[:, :, :3]
    if data.dtype == np.uint8:
        return data.astype(np.uint8)
    data = data.astype(np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)
    if data[finite].max() <= 1.0:
        data = data * 255.0
    return np.clip(data, 0, 255).astype(np.uint8)


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return _to_uint8_rgb(np.load(path))
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _write_preview_png(source_path: Path, output_path: Path) -> Path | None:
    try:
        image = _load_image(source_path)
    except Exception:
        return None
    ensure_dir(output_path.parent)
    Image.fromarray(image).save(output_path)
    return output_path


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    return load_depth_png(path)


def _depth_splat_candidate(
    *,
    records: Sequence[Mapping[str, Any]],
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
    profile: Mapping[str, Any],
    camera: Mapping[str, Any],
    eval_run: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any] | None:
    output_dir = job_dir / "initial_policy_observation_sources"
    for record in records:
        frame_ref = _candidate_frame_ref(record)
        depth_ref = _depth_ref(record)
        if not frame_ref or not depth_ref:
            continue
        frame_path = _local_reference_path(frame_ref, capture_root=capture_root, job_dir=job_dir)
        depth_path = _local_reference_path(depth_ref, capture_root=capture_root, job_dir=job_dir)
        source_pose = _record_pose(record)
        if frame_path is None or depth_path is None or not frame_path.is_file() or not depth_path.is_file():
            continue
        if source_pose is None:
            continue
        ref_intrinsics = _record_intrinsics(record)
        target_intrinsics = _mapping(camera.get("intrinsics"))
        target_pose, target_pose_source = _target_pose(
            job_request=job_request,
            eval_run=eval_run,
            camera=camera,
            reference_pose=source_pose,
        )
        try:
            ref_image = _load_image(frame_path)
            ref_depth = _load_depth(depth_path)
            if ref_depth.shape[:2] != ref_image.shape[:2]:
                depth_image = Image.fromarray(ref_depth.astype(np.float32), mode="F")
                depth_image = depth_image.resize(
                    (ref_image.shape[1], ref_image.shape[0]),
                    resample=Image.Resampling.BILINEAR,
                )
                ref_depth = np.array(depth_image, dtype=np.float32)
            height = min(
                _int(target_intrinsics.get("height"), ref_image.shape[0]) or ref_image.shape[0],
                ref_image.shape[0],
            )
            width = min(
                _int(target_intrinsics.get("width"), ref_image.shape[1]) or ref_image.shape[1],
                ref_image.shape[1],
            )
            profile_width = float(target_intrinsics.get("width") or width)
            profile_height = float(target_intrinsics.get("height") or height)
            scale_x = float(width) / profile_width if profile_width else 1.0
            scale_y = float(height) / profile_height if profile_height else 1.0
            target_k = {
                "fx": float(target_intrinsics.get("fx") or ref_intrinsics["fx"]) * scale_x,
                "fy": float(target_intrinsics.get("fy") or ref_intrinsics["fy"]) * scale_y,
                "cx": float(target_intrinsics.get("cx") or profile_width / 2.0) * scale_x,
                "cy": float(target_intrinsics.get("cy") or profile_height / 2.0) * scale_y,
            }
            ref_profile_width = float(ref_intrinsics.get("width") or ref_image.shape[1])
            ref_profile_height = float(ref_intrinsics.get("height") or ref_image.shape[0])
            ref_scale_x = float(ref_image.shape[1]) / ref_profile_width if ref_profile_width else 1.0
            ref_scale_y = float(ref_image.shape[0]) / ref_profile_height if ref_profile_height else 1.0
            ref_k = {
                "fx": float(ref_intrinsics["fx"]) * ref_scale_x,
                "fy": float(ref_intrinsics["fy"]) * ref_scale_y,
                "cx": float(ref_intrinsics["cx"]) * ref_scale_x,
                "cy": float(ref_intrinsics["cy"]) * ref_scale_y,
            }
            warped, mask = depth_splat(
                ref_image=ref_image,
                ref_depth=ref_depth,
                T_world_ref=np.array(source_pose, dtype=np.float64),
                K_ref=ref_k,
                T_world_target=np.array(target_pose, dtype=np.float64),
                K_target=target_k,
                target_h=int(height),
                target_w=int(width),
                fill_holes=True,
            )
        except Exception as exc:
            return {
                "candidate_id": _stable_id(frame_ref, depth_ref, prefix="depth_splat"),
                "source_kind": "capture_derived_depth_splat",
                "status": "blocked_synthesis_failed",
                "selection_score": SOURCE_PRIORITY["capture_derived_depth_splat"] - 20,
                "blockers": [f"depth_splat_failed:{type(exc).__name__}"],
                "robot_profile_id": profile.get("robot_profile_id"),
                "camera_id": camera.get("camera_id"),
                "camera": camera,
                "provenance": _provenance(
                    source_kind="capture_derived_depth_splat",
                    capture_derived=True,
                    source_artifacts={"reference_frame": frame_ref, "reference_depth": depth_ref},
                ),
            }
        ensure_dir(output_dir)
        output_path = output_dir / "capture_derived_depth_splat_initial_policy_observation.png"
        Image.fromarray(warped).save(output_path)
        coverage = round(float(mask.sum()) / float(mask.size), 6) if mask.size else 0.0
        return {
            "candidate_id": _stable_id(frame_ref, depth_ref, prefix="depth_splat"),
            "source_kind": "capture_derived_depth_splat",
            "status": "available",
            "generated_at": generated_at,
            "selection_score": SOURCE_PRIORITY["capture_derived_depth_splat"] + int(coverage * 10),
            "selection_reasons": ["depth_map_and_pose_available", "local_depth_splat_succeeded"],
            "robot_profile_id": profile.get("robot_profile_id"),
            "camera_id": camera.get("camera_id"),
            "camera": camera,
            "scenario_eval_run_id": eval_run.get("scenario_eval_run_id"),
            "visual_observation": {
                "available": True,
                "camera_frame_path": str(output_path),
                "source_image_path": str(frame_path),
                "source_depth_path": str(depth_path),
                "source_pose_available": True,
                "source_intrinsics_available": True,
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
                "capture_derived": True,
                "synthetic_camera_view": False,
                "depth_splat_coverage_ratio": coverage,
                "sha256": _sha256(output_path),
            },
            "synthesis": {
                "method": "depth_splat",
                "target_pose_source": target_pose_source,
                "reference_pose_available": True,
                "paid_provider_call_performed": False,
            },
            "provenance": _provenance(
                source_kind="capture_derived_depth_splat",
                capture_derived=True,
                source_artifacts={
                    "reference_frame": frame_ref,
                    "reference_depth": depth_ref,
                    "output_frame_path": str(output_path),
                },
            ),
        }
    return None


def _find_3dgs_assets(capture_root: Path, job_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        capture_root / "pipeline" / "advanced_geometry" / "3dgs_compressed.ply",
        capture_root / "pipeline" / "advanced_geometry" / "splat.ply",
        capture_root / "pipeline" / "advanced_geometry" / "scene.splat",
        capture_root / "pipeline" / "advanced_geometry" / "advanced_geometry_bundle.json",
        capture_root / "advanced_geometry" / "3dgs_compressed.ply",
        job_dir / "initial_observation_3dgs.png",
        job_dir / "capture_derived_3dgs_initial_policy_observation.png",
    ]
    assets: list[dict[str, Any]] = []
    for path in candidates:
        if path.is_file():
            assets.append(
                {
                    "path": str(path),
                    "kind": "rendered_frame" if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else "asset",
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return assets


def _capture_derived_3dgs_candidate(
    *,
    capture_root: Path,
    job_dir: Path,
    profile: Mapping[str, Any],
    camera: Mapping[str, Any],
    eval_run: Mapping[str, Any],
) -> dict[str, Any] | None:
    assets = _find_3dgs_assets(capture_root, job_dir)
    if not assets:
        return None
    rendered = next((asset for asset in assets if asset["kind"] == "rendered_frame"), None)
    status = "available" if rendered else "renderer_required"
    score = SOURCE_PRIORITY["capture_derived_3dgs"] if rendered else SOURCE_PRIORITY["capture_derived_3dgs"] - 20
    return {
        "candidate_id": _stable_id(*(asset["path"] for asset in assets), prefix="capture_3dgs"),
        "source_kind": "capture_derived_3dgs",
        "status": status,
        "selection_score": score,
        "selection_reasons": ["capture_3dgs_asset_present"]
        + (["local_3dgs_rendered_frame_present"] if rendered else []),
        "robot_profile_id": profile.get("robot_profile_id"),
        "camera_id": camera.get("camera_id"),
        "camera": camera,
        "scenario_eval_run_id": eval_run.get("scenario_eval_run_id"),
        "visual_observation": {
            "available": bool(rendered),
            "camera_frame_path": rendered.get("path") if rendered else None,
            "source_3dgs_assets": assets,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "capture_derived": True,
            "synthetic_camera_view": False,
            "sha256": rendered.get("sha256") if rendered else None,
        },
        "synthesis": {
            "method": "3dgs_render",
            "local_rendered_frame_present": bool(rendered),
            "paid_provider_call_performed": False,
            "renderer_required": not bool(rendered),
        },
        "provenance": _provenance(
            source_kind="capture_derived_3dgs",
            capture_derived=True,
            source_artifacts={"assets": assets},
        ),
    }


def _object_identifier(item: Mapping[str, Any]) -> str:
    return _string(item.get("object_id") or item.get("id") or item.get("uuid") or item.get("name"))


def _object_bbox(item: Mapping[str, Any]) -> Any:
    return (
        item.get("boundingBox")
        or item.get("bounding_box")
        or item.get("placement_bbox")
        or item.get("bbox")
        or item.get("bounds")
    )


def _object_is_capture_grounded(item: Mapping[str, Any]) -> bool:
    provenance = _mapping(item.get("provenance"))
    grounding = _string(provenance.get("grounding_level") or item.get("grounding_level")).lower()
    canonical = provenance.get("canonical_truth")
    return bool(
        _object_identifier(item)
        and _object_bbox(item)
        and grounding in {"observed", "capture_backed", "capture-grounded", "reconstructed"}
        and canonical is not False
    )


def _object_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("grounded_objects", "objects", "items", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    geometry = _mapping(payload.get("geometry"))
    object_index = _mapping(geometry.get("object_index") or payload.get("object_index"))
    value = object_index.get("objects")
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _load_object_grounding(capture_root: Path) -> dict[str, Any]:
    sources = [
        capture_root / "raw" / "object_grounding_hints.json",
        capture_root / "raw" / "object_index.json",
        capture_root / "pipeline" / "robot_eval_dataset" / "site_card.json",
        capture_root / "pipeline" / "object_geometry" / "object_geometry_manifest.json",
    ]
    objects_by_id: dict[str, dict[str, Any]] = {}
    source_paths: list[str] = []
    for path in sources:
        payload = _read_optional_any(path)
        records = _object_records_from_payload(payload)
        if not records:
            continue
        source_paths.append(str(path))
        for item in records:
            object_id = _object_identifier(item)
            if not object_id:
                continue
            existing = objects_by_id.get(object_id)
            if existing and _object_is_capture_grounded(existing):
                continue
            objects_by_id[object_id] = item
    objects = sorted(objects_by_id.values(), key=lambda item: _object_identifier(item))
    grounded = [item for item in objects if _object_is_capture_grounded(item)]
    return {
        "status": "ready" if grounded else "missing",
        "source_paths": source_paths,
        "object_count": len(objects),
        "grounded_object_count": len(grounded),
        "grounded_objects": grounded,
        "grounded_object_ids": [_object_identifier(item) for item in grounded],
    }


def _target_object_ids(
    *,
    job_request: Mapping[str, Any],
    eval_run: Mapping[str, Any],
    scenario_cards: Mapping[str, Any],
    task_cards: Mapping[str, Any],
) -> list[str]:
    ids: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            if _string(value) and _string(value) not in ids:
                ids.append(_string(value))
            return
        for item in _list(value):
            if isinstance(item, Mapping):
                add(item.get("object_id") or item.get("id"))
            elif _string(item) and _string(item) not in ids:
                ids.append(_string(item))

    initial = _mapping(
        job_request.get("initial_observation")
        or job_request.get("initialObservation")
        or job_request.get("initial_policy_observation")
        or job_request.get("initialPolicyObservation")
    )
    for payload in (initial, job_request, eval_run):
        add(payload.get("target_object_id") or payload.get("targetObjectId"))
        add(payload.get("target_object_ids") or payload.get("targetObjectIds"))
        add(payload.get("target_objects") or payload.get("targetObjects"))

    task_id = _string(eval_run.get("task_id"))
    scenario_id = _string(eval_run.get("scenario_id"))
    for card in _list(scenario_cards.get("cards")):
        if not isinstance(card, Mapping):
            continue
        if scenario_id and _string(card.get("scenario_id")) != scenario_id:
            continue
        add(card.get("target_object_ids") or card.get("targetObjectIds"))
        add(card.get("target_objects") or card.get("targetObjects"))
    for card in _list(task_cards.get("cards")):
        if not isinstance(card, Mapping):
            continue
        if task_id and _string(card.get("task_id")) != task_id:
            continue
        add(card.get("target_object_ids") or card.get("targetObjectIds"))
        add(card.get("target_objects") or card.get("targetObjects"))
    return ids


def _initial_observation_source_qa(
    *,
    records: Sequence[Mapping[str, Any]],
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
    eval_run: Mapping[str, Any],
    scenario_cards: Mapping[str, Any],
    task_cards: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    local_frame_count = 0
    local_depth_count = 0
    pose_count = 0
    intrinsics_count = 0
    usable_record_ids: list[str] = []
    for record in records:
        frame_ref = _candidate_frame_ref(record)
        depth_ref = _depth_ref(record)
        frame_path = _local_reference_path(frame_ref, capture_root=capture_root, job_dir=job_dir)
        depth_path = _local_reference_path(depth_ref, capture_root=capture_root, job_dir=job_dir)
        has_frame = bool(frame_path and frame_path.is_file())
        has_depth = bool(depth_path and depth_path.is_file())
        has_pose = _record_pose(record) is not None
        has_intrinsics = _record_has_intrinsics(record)
        local_frame_count += int(has_frame)
        local_depth_count += int(has_depth)
        pose_count += int(has_pose)
        intrinsics_count += int(has_intrinsics)
        if has_frame and has_depth and has_pose and has_intrinsics:
            usable_record_ids.append(_frame_record_key(record) or frame_ref)

    grounding = _load_object_grounding(capture_root)
    target_ids = _target_object_ids(
        job_request=job_request,
        eval_run=eval_run,
        scenario_cards=scenario_cards,
        task_cards=task_cards,
    )
    grounded_ids = set(grounding.get("grounded_object_ids") or [])
    missing_target_ids = [target_id for target_id in target_ids if target_id not in grounded_ids]

    blockers: list[str] = []
    if not records:
        blockers.append("capture_frame_index_missing")
    if local_frame_count <= 0:
        blockers.append("local_rgb_frame_missing")
    if local_depth_count <= 0:
        blockers.append("depth_map_missing")
    if pose_count <= 0:
        blockers.append("camera_pose_missing")
    if intrinsics_count <= 0:
        blockers.append("camera_intrinsics_missing")
    if grounding.get("grounded_object_count", 0) <= 0:
        blockers.append("object_grounding_missing")
    if missing_target_ids:
        blockers.append("target_object_grounding_missing")

    return {
        "schema_version": INITIAL_OBSERVATION_SOURCE_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready" if not blockers else "blocked",
        "capture_root": str(capture_root),
        "job_dir": str(job_dir),
        "reference_record_count": len(records),
        "local_frame_count": local_frame_count,
        "local_depth_count": local_depth_count,
        "pose_count": pose_count,
        "intrinsics_count": intrinsics_count,
        "usable_geometry_record_count": len(usable_record_ids),
        "usable_geometry_record_ids": usable_record_ids[:12],
        "object_grounding": {
            "status": grounding.get("status"),
            "source_paths": grounding.get("source_paths", []),
            "object_count": grounding.get("object_count", 0),
            "grounded_object_count": grounding.get("grounded_object_count", 0),
            "grounded_object_ids": grounding.get("grounded_object_ids", []),
            "target_object_ids": target_ids,
            "missing_target_object_ids": missing_target_ids,
        },
        "required_modalities": {
            "local_rgb_frame": local_frame_count > 0,
            "depth": local_depth_count > 0,
            "camera_pose": pose_count > 0,
            "camera_intrinsics": intrinsics_count > 0,
            "object_grounding": grounding.get("grounded_object_count", 0) > 0,
            "target_object_grounding": not missing_target_ids,
        },
        "blockers": blockers,
        "claim_boundary": _source_claim_boundary(),
    }


def _recapture_guidance(
    *,
    source_qa: Mapping[str, Any],
    selected: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    blockers = [str(item) for item in source_qa.get("blockers", []) or []]
    if not selected and "no_capture_backed_initial_policy_observation_candidate" not in blockers:
        blockers.append("no_capture_backed_initial_policy_observation_candidate")
    actions_by_blocker = {
        "capture_frame_index_missing": "Export a frame index that lists local RGB frames with frame ids and timestamps.",
        "local_rgb_frame_missing": "Stage at least one local RGB frame referenced by the frame index.",
        "depth_map_missing": "Recapture or reprocess with per-frame depth maps aligned to the RGB frames.",
        "camera_pose_missing": "Export world-from-camera poses for the selected frames.",
        "camera_intrinsics_missing": "Export calibrated camera intrinsics including width, height, fx, fy, cx, and cy.",
        "object_grounding_missing": "Run object indexing/grounding and retain capture-backed object ids with bounding boxes.",
        "target_object_grounding_missing": "Ground every requested target object id to an observed object record.",
        "no_capture_backed_initial_policy_observation_candidate": (
            "Provide a direct robot POV capture frame or capture geometry that can render the robot camera POV."
        ),
    }
    actions = [actions_by_blocker.get(blocker, blocker) for blocker in blockers]
    return {
        "schema_version": INITIAL_OBSERVATION_RECAPTURE_GUIDANCE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "not_required" if not blockers else "recapture_required",
        "blockers": blockers,
        "recommended_actions": actions,
        "recapture_guidance_does_not_claim_rank_fidelity": True,
        "claim_boundary": _source_claim_boundary(),
    }


def _contact_sheet_image_for_path(path: Path, *, max_size: tuple[int, int]) -> Image.Image | None:
    try:
        image = Image.fromarray(_load_image(path))
    except Exception:
        return None
    image.thumbnail(max_size)
    canvas = Image.new("RGB", max_size, (18, 20, 24))
    x = (max_size[0] - image.width) // 2
    y = (max_size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _write_initial_observation_contact_sheet(
    *,
    candidates: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    selected_candidate_id: str | None,
    output_path: Path,
    capture_root: Path,
    job_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_path.parent)
    tile_size = (240, 150)
    rows: list[tuple[str, Path | None]] = []
    for candidate in candidates:
        visual = _mapping(candidate.get("visual_observation"))
        path_text = _string(visual.get("camera_frame_path") or visual.get("source_image_path"))
        rows.append((f"{candidate.get('candidate_id')} {candidate.get('source_kind')}", Path(path_text) if path_text else None))
    if not rows:
        for record in records[:8]:
            frame_ref = _candidate_frame_ref(record)
            rows.append(
                (
                    f"source_frame {_frame_record_key(record) or len(rows)}",
                    _local_reference_path(frame_ref, capture_root=capture_root, job_dir=job_dir)
                    if frame_ref
                    else None,
                )
            )
    if not rows:
        rows.append(("no readable capture-backed candidates", None))

    width = tile_size[0] * min(3, max(1, len(rows)))
    height = tile_size[1] * math.ceil(len(rows) / min(3, max(1, len(rows))))
    sheet = Image.new("RGB", (width, height), (30, 32, 36))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(rows):
        col = index % 3
        row = index // 3
        x = col * tile_size[0]
        y = row * tile_size[1]
        image = _contact_sheet_image_for_path(path, max_size=tile_size) if path else None
        if image:
            sheet.paste(image, (x, y))
        else:
            draw.rectangle((x + 8, y + 8, x + tile_size[0] - 8, y + tile_size[1] - 8), outline=(90, 94, 102), width=2)
        is_selected = selected_candidate_id and label.startswith(str(selected_candidate_id))
        outline = (96, 190, 135) if is_selected else (92, 118, 150)
        draw.rectangle((x + 2, y + 2, x + tile_size[0] - 3, y + tile_size[1] - 3), outline=outline, width=3)
        draw.rectangle((x, y + tile_size[1] - 28, x + tile_size[0], y + tile_size[1]), fill=(0, 0, 0))
        draw.text((x + 8, y + tile_size[1] - 22), label[:44], fill=(235, 240, 245))
    sheet.save(output_path, quality=90)
    return {
        "status": "written",
        "contact_sheet_path": str(output_path),
        "candidate_tile_count": len(candidates),
        "source_record_tile_count": len(rows) if not candidates else 0,
        "selected_candidate_id": selected_candidate_id,
    }


def _first_eval_run(
    *,
    scenario_eval_matrix: Mapping[str, Any] | None,
    observations: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    for item in observations or []:
        if isinstance(item, Mapping):
            return dict(item)
    matrix = _mapping(scenario_eval_matrix)
    for item in matrix.get("runs", []) or []:
        if isinstance(item, Mapping):
            return dict(item)
    return {
        "scenario_eval_run_id": "initial_policy_observation",
        "task_id": None,
        "scenario_id": None,
        "variation_name": "initial",
    }


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selectable = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("status") == "available"
        and bool(_mapping(candidate.get("visual_observation")).get("available"))
    ]
    if not selectable:
        return {}
    return sorted(
        selectable,
        key=lambda item: (
            SOURCE_PRIORITY.get(_string(item.get("source_kind")), 0),
            int(item.get("selection_score") or 0),
        ),
        reverse=True,
    )[0]


def build_initial_observation_source_resolution(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    generated_at: str,
    scenario_cards: Mapping[str, Any] | None = None,
    task_cards: Mapping[str, Any] | None = None,
    scenario_eval_matrix: Mapping[str, Any] | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve and write a fail-closed, capture-backed initial policy observation."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    cards = _mapping(scenario_cards)
    tasks = _mapping(task_cards)
    registry = build_robot_camera_profile_registry(
        job_request=job_request,
        scenario_cards=cards,
        generated_at=generated_at,
    )
    launch_readiness = build_robot_camera_profile_launch_readiness(
        registry=registry,
        generated_at=generated_at,
    )
    eval_run = _first_eval_run(scenario_eval_matrix=scenario_eval_matrix, observations=observations)
    profile = _select_profile(registry, _string(eval_run.get("robot_profile_id")) or None)
    camera = _select_camera(profile, job_request)
    records = _load_capture_frame_records(
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
    )
    source_qa = _initial_observation_source_qa(
        records=records,
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
        eval_run=eval_run,
        scenario_cards=cards,
        task_cards=tasks,
        generated_at=generated_at,
    )

    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _direct_capture_candidates(
            records=records,
            capture_root=capture_path,
            job_dir=resolved_job_dir,
            profile=profile,
            camera=camera,
            eval_run=eval_run,
        )
    )
    depth_candidate = _depth_splat_candidate(
        records=records,
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
        profile=profile,
        camera=camera,
        eval_run=eval_run,
        generated_at=generated_at,
    )
    if depth_candidate:
        candidates.append(depth_candidate)
    gs_candidate = _capture_derived_3dgs_candidate(
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        profile=profile,
        camera=camera,
        eval_run=eval_run,
    )
    if gs_candidate:
        candidates.append(gs_candidate)

    selected = _select_candidate(candidates) if source_qa.get("status") == "ready" else {}
    if selected:
        candidate_frame = _string(_mapping(selected.get("visual_observation")).get("camera_frame_path"))
        if candidate_frame and candidate_frame.endswith(".npy"):
            preview_path = (
                resolved_job_dir
                / "initial_policy_observation_sources"
                / f"{selected.get('candidate_id')}_selected_preview.png"
            )
            written = _write_preview_png(Path(candidate_frame), preview_path)
            if written is not None:
                selected = dict(selected)
                selected_visual = _mapping(selected.get("visual_observation"))
                selected_visual["camera_frame_path"] = str(written)
                selected_visual["source_frame_array_path"] = candidate_frame
                selected_visual["sha256"] = _sha256(written)
                selected["visual_observation"] = selected_visual

    blockers = [str(item) for item in source_qa.get("blockers", []) or []]
    if not selected:
        blockers.append("no_capture_backed_initial_policy_observation_candidate")
    source_qa_path = resolved_job_dir / "initial_policy_observation_source_qa.json"
    contact_sheet_path = resolved_job_dir / "initial_policy_observation_contact_sheet.jpg"
    recapture_guidance_path = resolved_job_dir / "initial_policy_observation_recapture_guidance.json"
    contact_sheet = _write_initial_observation_contact_sheet(
        candidates=candidates,
        records=records,
        selected_candidate_id=_string(selected.get("candidate_id")) or None,
        output_path=contact_sheet_path,
        capture_root=capture_path,
        job_dir=resolved_job_dir,
    )
    recapture_guidance = _recapture_guidance(
        source_qa=source_qa,
        selected=selected,
        generated_at=generated_at,
    )

    source_qa_summary = {
        "status": source_qa.get("status"),
        "local_frame_count": source_qa.get("local_frame_count"),
        "local_depth_count": source_qa.get("local_depth_count"),
        "pose_count": source_qa.get("pose_count"),
        "intrinsics_count": source_qa.get("intrinsics_count"),
        "grounded_object_count": _mapping(source_qa.get("object_grounding")).get(
            "grounded_object_count"
        ),
        "blockers": source_qa.get("blockers", []),
    }
    candidate_set = {
        "schema_version": ROBOT_POV_OBSERVATION_CANDIDATE_SET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready" if selected else "blocked",
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "candidate_count": len(candidates),
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_source_kind": selected.get("source_kind"),
        "source_priority_order": [
            "direct_capture_frame",
            "capture_derived_depth_splat",
            "capture_derived_3dgs",
        ],
        "synthetic_fallback_allowed": False,
        "camera_profile_registry_path": "robot_camera_profile_registry.json",
        "camera_profile_launch_readiness_path": "robot_camera_profile_launch_readiness.json",
        "camera_profile_registry": registry,
        "camera_profile_launch_readiness": launch_readiness,
        "target": {
            "robot_profile_id": profile.get("robot_profile_id"),
            "camera_id": camera.get("camera_id"),
            "scenario_eval_run_id": eval_run.get("scenario_eval_run_id"),
            "task_id": eval_run.get("task_id"),
            "scenario_id": eval_run.get("scenario_id"),
        },
        "capture_frame_record_count": len(records),
        "candidates": candidates,
        "source_qa_path": str(source_qa_path),
        "contact_sheet_path": str(contact_sheet_path),
        "recapture_guidance_path": str(recapture_guidance_path),
        "source_qa_summary": source_qa_summary,
        "contact_sheet": contact_sheet,
        "blockers": blockers,
        "paid_provider_calls_performed": False,
        "claim_boundary": _source_claim_boundary(),
    }
    visual = _mapping(selected.get("visual_observation"))
    selected_observation = {
        "schema_version": SELECTED_INITIAL_POLICY_OBSERVATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready" if selected else "blocked",
        "selected_candidate_id": selected.get("candidate_id"),
        "selection_source_kind": selected.get("source_kind"),
        "selection_reasons": list(selected.get("selection_reasons") or []),
        "robot_profile_id": profile.get("robot_profile_id"),
        "camera_id": camera.get("camera_id"),
        "camera": selected.get("camera") or camera,
        "camera_profile_launch_readiness": {
            "path": "robot_camera_profile_launch_readiness.json",
            "status": launch_readiness.get("status"),
            "launch_mode": launch_readiness.get("launch_mode"),
            "all_profiles_launch_ready": launch_readiness.get("all_profiles_launch_ready"),
        },
        "scenario_eval_run_id": eval_run.get("scenario_eval_run_id"),
        "task_id": eval_run.get("task_id"),
        "scenario_id": eval_run.get("scenario_id"),
        "camera_frame_path": visual.get("camera_frame_path"),
        "camera_frame_uri": visual.get("camera_frame_uri"),
        "visual_observation": visual,
        "source_qa_path": str(source_qa_path),
        "contact_sheet_path": str(contact_sheet_path),
        "recapture_guidance_path": str(recapture_guidance_path),
        "source_qa": source_qa,
        "object_grounding": _mapping(source_qa.get("object_grounding")),
        "blockers": blockers,
        "observation_schema": {
            "schema_id": POLICY_OBSERVATION_SCHEMA_ID,
            "schema_ref": POLICY_OBSERVATION_SCHEMA_REF,
            "rgb": "local_capture_backed_image",
            "depth": "required_capture_depth_map",
            "robot_state": ["base_pose", "joint_state_optional", "gripper_state_optional"],
        },
        "source_candidate": selected,
        "provenance": _mapping(selected.get("provenance")),
        "paid_provider_calls_performed": False,
        "claim_boundary": {
            **_source_claim_boundary(),
            "selected_observation_is_policy_input_not_success_proof": True,
            "capture_truth": selected.get("source_kind") == "direct_capture_frame",
            "geometry_truth": selected.get("source_kind") == "capture_derived_depth_splat",
            "collision_truth": False,
            "selected_direct_capture_frame": selected.get("source_kind") == "direct_capture_frame",
            "selected_capture_derived_synthesis": selected.get("source_kind")
            in {"capture_derived_depth_splat", "capture_derived_3dgs"},
            "selected_synthetic_fallback": False,
            "required_depth_pose_intrinsics_object_grounding": True,
            "provider_success_separate_from_visually_useful_rollout": True,
            "visually_useful_rollout": False,
        },
    }
    write_json(resolved_job_dir / "robot_camera_profile_registry.json", registry)
    write_json(
        resolved_job_dir / "robot_camera_profile_launch_readiness.json",
        launch_readiness,
    )
    write_json(source_qa_path, source_qa)
    write_json(recapture_guidance_path, recapture_guidance)
    write_json(resolved_job_dir / "robot_pov_observation_candidate_set.json", candidate_set)
    write_json(resolved_job_dir / "selected_initial_policy_observation.json", selected_observation)
    return {
        "candidate_set": candidate_set,
        "selected_initial_policy_observation": selected_observation,
    }
