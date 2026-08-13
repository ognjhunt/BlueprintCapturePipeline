"""Build a task-neutral native camera/reset request from bound evidence.

The DROID Franka reset, wrist mount, and pinhole intrinsics are copied from a
previously sealed native Arena request.  External and overview poses are
derived only from the registered interaction target and Franka base.  The
result is a requested native rig, never a calibration or observability claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION as ARENA_REQUEST_SCHEMA
from .native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES
from .paired_target_interaction_affordance_candidate import (
    SCHEMA_VERSION as AFFORDANCE_SCHEMA,
)


SCHEMA_VERSION = "paired_target_native_camera_rig_candidate.v1"


class PairedTargetNativeCameraRigError(ValueError):
    """Stable fail-closed native camera request errors."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeCameraRigError(code) from exc
    if source.is_symlink() or not isinstance(value, dict):
        raise PairedTargetNativeCameraRigError(code)
    return source, value


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _vector(value: Any, *, length: int, code: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise PairedTargetNativeCameraRigError(code) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise PairedTargetNativeCameraRigError(code)
    return result


def _normalize(value: Sequence[float], *, code: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if norm <= 1.0e-9 or not math.isfinite(norm):
        raise PairedTargetNativeCameraRigError(code)
    return [float(item) / norm for item in value]


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _look_at(position: Sequence[float], target: Sequence[float]) -> list[float]:
    forward = _normalize(
        [target[index] - position[index] for index in range(3)],
        code="paired_target_camera_aim_degenerate",
    )
    right = _normalize(
        _cross(forward, [0.0, 0.0, 1.0]),
        code="paired_target_camera_aim_degenerate",
    )
    down = _normalize(
        _cross(forward, right), code="paired_target_camera_aim_degenerate"
    )
    return [
        right[0], down[0], forward[0], float(position[0]),
        right[1], down[1], forward[1], float(position[1]),
        right[2], down[2], forward[2], float(position[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def _world_camera(
    *, role: str, position: Sequence[float], target: Sequence[float], intrinsics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "role": role,
        "policy_input": role == "external",
        "scoring_input": False,
        "pose_frame": "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": _look_at(position, target),
        "intrinsics": json.loads(json.dumps(intrinsics)),
    }


def materialize_paired_target_native_camera_rig_candidate(
    *,
    interaction_affordance_candidate_path: str | Path,
    franka_placement_packet_path: str | Path,
    droid_native_profile_request_path: str | Path,
    output_path: str | Path,
    external_height_m: float = 1.35,
    overview_lateral_distance_m: float = 0.9,
    overview_height_m: float = 1.45,
) -> dict[str, Any]:
    """Materialize one requested three-camera DROID rig plus robot reset."""

    affordance_path, affordance = _read(
        interaction_affordance_candidate_path,
        "paired_target_camera_affordance_invalid",
    )
    if (
        affordance.get("schema_version") != AFFORDANCE_SCHEMA
        or affordance.get("receipt_digest")
        != canonical_digest(affordance, digest_field="receipt_digest")
        or affordance.get("native_contact_executed") is not False
        or affordance.get("candidate", {}).get("pinch_span_within_stroke") is not True
    ):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_affordance_invalid"
        )
    placement_path, placement = _read(
        franka_placement_packet_path, "paired_target_camera_placement_invalid"
    )
    if (
        placement.get("schema_version")
        != "registered_sage_franka_placement_packet.v1"
        or placement.get("packet_digest")
        != canonical_digest(placement, digest_field="packet_digest")
    ):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_placement_invalid"
        )
    base_xyzyaw = _vector(
        placement.get("placement", {}).get("robot_pose_xyzyaw_collision_stage"),
        length=4,
        code="paired_target_camera_placement_invalid",
    )
    base = base_xyzyaw[:3]
    if any(
        abs(base[index] - affordance["robot_base_position_world_m"][index]) > 1.0e-9
        for index in range(3)
    ):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_affordance_placement_mismatch"
        )
    profile_path, profile = _read(
        droid_native_profile_request_path, "paired_target_camera_droid_profile_invalid"
    )
    profile_cameras = profile.get("cameras")
    reset = profile.get("robot_joint_reset_positions_rad")
    if (
        profile.get("schema_version") != ARENA_REQUEST_SCHEMA
        or profile.get("request_digest")
        != canonical_digest(profile, digest_field="request_digest")
        or not isinstance(profile_cameras, list)
        or not isinstance(reset, Mapping)
        or set(reset) != set(DROID_FRANKA_RESET_JOINT_NAMES)
    ):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_droid_profile_invalid"
        )
    by_role = {
        str(row.get("role") or ""): dict(row)
        for row in profile_cameras
        if isinstance(row, Mapping)
    }
    if set(by_role) != {"external", "wrist", "overview"}:
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_droid_profile_invalid"
        )
    wrist = by_role["wrist"]
    if (
        wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path")
        != "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
        or wrist.get("policy_input") is not True
        or wrist.get("scoring_input") is not False
    ):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_droid_profile_invalid"
        )
    target = _vector(
        affordance.get("candidate", {}).get("contact_point_registered_stage_m"),
        length=3,
        code="paired_target_camera_target_invalid",
    )
    height = float(external_height_m)
    lateral = float(overview_lateral_distance_m)
    overview_height = float(overview_height_m)
    if not all(math.isfinite(item) and item > 0.0 for item in (height, lateral, overview_height)):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_geometry_parameters_invalid"
        )
    horizontal = _normalize(
        [target[0] - base[0], target[1] - base[1], 0.0],
        code="paired_target_camera_base_target_degenerate",
    )
    lateral_unit = [-horizontal[1], horizontal[0], 0.0]
    external_position = [base[0], base[1], base[2] + height]
    overview_position = [
        0.5 * (base[0] + target[0]) + lateral * lateral_unit[0],
        0.5 * (base[1] + target[1]) + lateral * lateral_unit[1],
        max(base[2], target[2]) + overview_height,
    ]
    intrinsics = by_role["external"].get("intrinsics")
    if not isinstance(intrinsics, Mapping) or intrinsics != by_role["wrist"].get("intrinsics"):
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_droid_profile_intrinsics_invalid"
        )
    cameras = [
        _world_camera(
            role="external",
            position=external_position,
            target=target,
            intrinsics=intrinsics,
        ),
        json.loads(json.dumps(wrist)),
        _world_camera(
            role="overview",
            position=overview_position,
            target=[
                0.5 * (base[0] + target[0]),
                0.5 * (base[1] + target[1]),
                target[2],
            ],
            intrinsics=intrinsics,
        ),
    ]
    cameras[2]["policy_input"] = False
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "native_camera_rig_requested_requires_readback_and_observability",
        "scene_id": affordance["scene_id"],
        "task_id": affordance["task_id"],
        "interaction_affordance_candidate": _record(
            affordance_path, receipt_digest=affordance["receipt_digest"]
        ),
        "franka_placement_packet": _record(
            placement_path, packet_digest=placement["packet_digest"]
        ),
        "droid_native_profile_source": _record(
            profile_path, request_digest=profile["request_digest"]
        ),
        "robot_base_pose_world": {
            "position_world_m": base,
            "orientation_xyzw": [
                0.0,
                0.0,
                math.sin(base_xyzyaw[3] / 2.0),
                math.cos(base_xyzyaw[3] / 2.0),
            ],
        },
        "robot_joint_reset_positions_rad": {
            name: float(reset[name]) for name in DROID_FRANKA_RESET_JOINT_NAMES
        },
        "cameras": cameras,
        "camera_roles": ["external", "wrist", "overview"],
        "policy_input_roles": ["external", "wrist"],
        "overview_review_only": True,
        "requested_camera_geometry_source": {
            "external_and_overview": "analytic_base_target_look_at_candidate",
            "wrist_and_intrinsics": "digest_bound_prior_native_droid_profile",
        },
        "native_camera_readback_qualified": False,
        "native_semantic_observability_qualified": False,
        "blockers": [
            "native_camera_transform_and_intrinsics_readback_missing",
            "native_external_and_wrist_semantic_observability_missing",
        ],
        "claim_boundary": (
            "requested_camera_and_reset_contract_only;not_native_application_"
            "calibration_observability_policy_task_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetNativeCameraRigError(
            "paired_target_camera_destination_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json.loads(json.dumps(result))


__all__ = [
    "PairedTargetNativeCameraRigError",
    "SCHEMA_VERSION",
    "materialize_paired_target_native_camera_rig_candidate",
]
