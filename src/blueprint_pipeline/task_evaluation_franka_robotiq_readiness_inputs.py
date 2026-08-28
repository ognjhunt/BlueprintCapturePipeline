"""Canonical Franka/Robotiq inputs for configured-scene readiness episodes.

The embodiment facts in this module are reusable.  A base pose and camera set
are not: both remain candidate-only until the native construction worker reads
them back in the exact configured scene.  The materializer therefore binds
those candidates to the configured revision's immutable mount and calibration
documents without upgrading either candidate to qualified evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)


ROBOT_IDENTITY = {
    "id": "franka-panda-robotiq-2f85",
    "version": "isaaclab-arena-droid-8b4a3a47",
}
MATERIALIZATION_SCHEMA_VERSION = "task_evaluation_franka_robotiq_readiness_inputs.v1"

_RESET_JOINT_POSITIONS_RAD = {
    "finger_joint": 0.104255385697,
    "left_inner_finger_joint": -0.080966427922,
    "left_inner_finger_knuckle_joint": -0.071244180202,
    "panda_joint1": 0.0,
    "panda_joint2": -0.628318530718,
    "panda_joint3": 0.0,
    "panda_joint4": -2.513274122872,
    "panda_joint5": 0.0,
    "panda_joint6": 1.884955592154,
    "panda_joint7": 0.0,
    "right_inner_finger_joint": -0.128436118364,
    "right_inner_finger_knuckle_joint": 0.125143155456,
    "right_outer_knuckle_joint": 0.104152053595,
}
_ARM_JOINT_BOUNDS_RAD = {
    "panda_joint1": [-2.8973, 2.8973],
    "panda_joint2": [-1.7628, 1.7628],
    "panda_joint3": [-2.8973, 2.8973],
    "panda_joint4": [-3.0718, -0.0698],
    "panda_joint5": [-2.8973, 2.8973],
    "panda_joint6": [-0.0175, 3.7525],
    "panda_joint7": [-2.8973, 2.8973],
}
_ARM_VELOCITY_BOUNDS_RAD_S = {
    "panda_joint1": 2.175,
    "panda_joint2": 2.175,
    "panda_joint3": 2.175,
    "panda_joint4": 2.175,
    "panda_joint5": 2.61,
    "panda_joint6": 2.61,
    "panda_joint7": 2.61,
}
_CONTROLLER_KINDS = frozenset({"zero_action", "deterministic_scripted"})


class TaskEvaluationFrankaRobotiqReadinessInputsError(ValueError):
    """A readiness input would overclaim or disagree with its configured scene."""


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _read_bound_document(
    path: str | Path,
    *,
    reference: Mapping[str, Any],
    blocker: str,
) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(blocker)
    source = source.resolve()
    try:
        identity = _file_identity(source)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(blocker) from exc
    if (
        identity != (reference.get("digest"), reference.get("size_bytes"))
        or not isinstance(value, Mapping)
        or not str(value.get("schema_version") or "")
    ):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(blocker)
    return source, dict(value)


def _scene_document_matches(value: Mapping[str, Any], scene_identity: Mapping[str, Any]) -> bool:
    identity = str(scene_identity.get("id") or "")
    scene_id = str(value.get("scene_id") or "")
    embedded = value.get("scene_identity")
    return embedded == scene_identity or scene_id in {
        identity,
        identity.rsplit("-", 1)[-1],
    }


def _finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def _validated_pose(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != {
        "position_world_m",
        "orientation_xyzw",
    }:
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_base_pose_candidate_invalid"
        )
    position = value.get("position_world_m")
    orientation = value.get("orientation_xyzw")
    if not _finite_vector(position, 3) or not _finite_vector(orientation, 4):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_base_pose_candidate_invalid"
        )
    orientation_values = [float(item) for item in orientation]
    if not math.isclose(
        sum(item * item for item in orientation_values),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_base_pose_candidate_invalid"
        )
    return {
        "position_world_m": [float(item) for item in position],
        "orientation_xyzw": orientation_values,
    }


def _validated_cameras(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_camera_candidates_invalid"
        )
    cameras: list[dict[str, Any]] = []
    roles: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise TaskEvaluationFrankaRobotiqReadinessInputsError(
                "franka_readiness_camera_candidates_invalid"
            )
        camera = dict(row)
        role = str(camera.get("role") or "")
        intrinsics = camera.get("intrinsics")
        if (
            role not in {"external", "wrist", "overview"}
            or role in roles
            or camera.get("optical_convention") != "opencv"
            or camera.get("pose_frame") != ("robot_body" if role == "wrist" else "world")
            or not str(camera.get("parent_prim_path") or "")
            or camera.get("policy_input") is not (role != "overview")
            or camera.get("scoring_input") is not False
            or not _finite_vector(camera.get("frame_from_camera_matrix"), 16)
            or not isinstance(intrinsics, Mapping)
            or set(intrinsics) != {"cx", "cy", "fx", "fy", "height", "width"}
            or not all(
                isinstance(intrinsics.get(name), (int, float))
                and not isinstance(intrinsics.get(name), bool)
                and math.isfinite(float(intrinsics[name]))
                for name in ("cx", "cy", "fx", "fy", "height", "width")
            )
            or float(intrinsics["fx"]) <= 0
            or float(intrinsics["fy"]) <= 0
            or int(intrinsics["height"]) <= 0
            or int(intrinsics["width"]) <= 0
        ):
            raise TaskEvaluationFrankaRobotiqReadinessInputsError(
                "franka_readiness_camera_candidates_invalid"
            )
        roles.add(role)
        cameras.append(json.loads(json.dumps(camera)))
    if roles != {"external", "wrist", "overview"}:
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_camera_candidates_invalid"
        )
    return cameras


def _sealed_document(value: dict[str, Any]) -> dict[str, Any]:
    value["document_digest"] = ""
    value["document_digest"] = canonical_digest(value, digest_field="document_digest")
    return value


def _write_document(root: Path, name: str, value: dict[str, Any]) -> dict[str, Any]:
    path = root / name
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    path.chmod(0o440)
    digest, size = _file_identity(path)
    return {
        "path": str(path),
        "digest": digest,
        "size_bytes": size,
        "mode": "0440",
    }


def materialize_franka_robotiq_readiness_inputs(
    *,
    configured_revision: Mapping[str, Any],
    robot_mount_interface_path: str | Path,
    scene_camera_calibration_path: str | Path,
    base_pose_candidate: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
    controller_identity: Mapping[str, Any],
    controller_kind: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Write one candidate-only, digest-bound readiness input set."""

    revision = validate_configured_scene_revision(configured_revision)
    root = Path(output_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationFrankaRobotiqReadinessInputsError("franka_readiness_output_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve()
    if any(root.iterdir()):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_output_root_not_empty"
        )

    mount_reference = revision["registration"]["robot_mount_interface"]
    calibration_reference = revision["registration"]["camera_calibration"]
    _, mount = _read_bound_document(
        robot_mount_interface_path,
        reference=mount_reference,
        blocker="franka_readiness_robot_mount_interface_binding_invalid",
    )
    _, calibration = _read_bound_document(
        scene_camera_calibration_path,
        reference=calibration_reference,
        blocker="franka_readiness_camera_calibration_binding_invalid",
    )
    scene_identity = revision["scene_identity"]
    if not _scene_document_matches(mount, scene_identity):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_robot_mount_interface_scene_mismatch"
        )
    if not _scene_document_matches(calibration, scene_identity):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_camera_calibration_scene_mismatch"
        )
    if (
        controller_kind not in _CONTROLLER_KINDS
        or not isinstance(controller_identity, Mapping)
        or set(controller_identity) != {"id", "version"}
        or not all(str(controller_identity.get(field) or "") for field in ("id", "version"))
    ):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError("franka_readiness_controller_invalid")
    if (
        base_pose_candidate.get("schema_version")
        != "task_evaluation_planar_push_readiness_candidate.v1"
        or base_pose_candidate.get("status")
        != "candidate_pending_native_construction_readback"
        or base_pose_candidate.get("scene_identity") != scene_identity
        or base_pose_candidate.get("configured_scene_revision_digest")
        != revision["revision_digest"]
        or base_pose_candidate.get("robot_mount_interface_digest") != mount_reference["digest"]
        or base_pose_candidate.get("task_definition_digest")
        != revision["task_template"]["definition"]["digest"]
        or base_pose_candidate.get("workspace_clearance_digest")
        != revision["registration"]["workspace_clearance"]["digest"]
        or base_pose_candidate.get("derivation_method")
        != "reflect_reach_candidate_behind_start_along_frozen_planar_push"
        or base_pose_candidate.get("task_direction_considered") is not True
        or base_pose_candidate.get("robot_base_qualified") is not False
        or base_pose_candidate.get("reachability_qualified") is not False
        or base_pose_candidate.get("collision_clearance_qualified") is not False
        or base_pose_candidate.get("learned_policy_outcomes_consulted") is not False
        or base_pose_candidate.get("native_construction_readback_completed") is not False
        or base_pose_candidate.get("base_pose_candidate_digest")
        != canonical_digest(
            base_pose_candidate,
            digest_field="base_pose_candidate_digest",
        )
    ):
        raise TaskEvaluationFrankaRobotiqReadinessInputsError(
            "franka_readiness_base_pose_candidate_binding_invalid"
        )
    pose = _validated_pose(base_pose_candidate.get("pose_world"))
    camera_rows = _validated_cameras(cameras)

    source = {
        "isaac_lab_asset_symbol": "FRANKA_ROBOTIQ_GRIPPER_CFG",
        "isaac_lab_source_commit": "e57379c634b42db5a0fe9f754341be6e2a7c7c43",
        "isaac_lab_arena_source_commit": "8b4a3a47fc53de23e8205089d71109a2e2348acd",
        "arena_embodiment_id": "droid_abs_joint_pos",
    }
    documents = {
        "robot_configuration": _sealed_document(
            {
                "schema_version": "task_evaluation_native_robot_configuration.v1",
                "identity": dict(ROBOT_IDENTITY),
                "robot": "Franka Panda",
                "gripper": "Robotiq 2F-85",
                "source": source,
                "joint_reset_positions_rad": dict(_RESET_JOINT_POSITIONS_RAD),
                "candidate_policy_queried": False,
            }
        ),
        "robot_kinematics": _sealed_document(
            {
                "schema_version": "task_evaluation_native_robot_kinematics.v1",
                "identity": dict(ROBOT_IDENTITY),
                "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                "command_body": "panda_hand",
                "gripper_drive_joint": "finger_joint",
                "source": source,
            }
        ),
        "robot_joint_bounds": _sealed_document(
            {
                "schema_version": "task_evaluation_native_robot_joint_bounds.v1",
                "identity": dict(ROBOT_IDENTITY),
                "arm_position_bounds_rad": dict(_ARM_JOINT_BOUNDS_RAD),
                "arm_maximum_velocity_rad_s": dict(_ARM_VELOCITY_BOUNDS_RAD_S),
                "runtime_readback_required": True,
                "source": source,
            }
        ),
        "robot_base_registration": _sealed_document(
            {
                "schema_version": "task_evaluation_robot_to_scene_registration.v1",
                "robot_identity": dict(ROBOT_IDENTITY),
                "scene_identity": dict(scene_identity),
                "configured_scene_revision_digest": revision["revision_digest"],
                "robot_mount_interface_digest": mount_reference["digest"],
                "pose_world": pose,
                "pose_status": "candidate_pending_native_construction_readback",
                "robot_base_qualified": False,
                "reachability_qualified": False,
                "collision_clearance_qualified": False,
                "native_construction_readback_required": True,
                "learned_policy_outcomes_consulted": False,
            }
        ),
        "controller_configuration": _sealed_document(
            {
                "schema_version": "task_evaluation_native_controller_configuration.v1",
                "identity": dict(controller_identity),
                "kind": controller_kind,
                "robot_identity": dict(ROBOT_IDENTITY),
                "action_contract": "absolute_seven_arm_joints_plus_gripper",
                "control_frequency_hz": 15,
                "candidate_policy_queried": False,
            }
        ),
        "sensor_configuration": _sealed_document(
            {
                "schema_version": "task_evaluation_native_sensor_configuration.v1",
                "scene_identity": dict(scene_identity),
                "configured_scene_revision_digest": revision["revision_digest"],
                "scene_camera_calibration_digest": calibration_reference["digest"],
                "scene_camera_calibration_size_bytes": calibration_reference["size_bytes"],
                "calibration_status": "candidate_pending_native_construction_readback",
                "native_construction_readback_required": True,
                "cameras": camera_rows,
            }
        ),
    }
    files = {
        role: _write_document(root, f"{role}.v1.json", document)
        for role, document in documents.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "materialized_candidate_pending_native_construction_readback",
        "configured_scene_revision_digest": revision["revision_digest"],
        "robot_identity": dict(ROBOT_IDENTITY),
        "controller_identity": dict(controller_identity),
        "controller_kind": controller_kind,
        "robot_mount_interface_digest": mount_reference["digest"],
        "scene_camera_calibration_digest": calibration_reference["digest"],
        "robot_base_qualified": False,
        "camera_configuration_qualified": False,
        "native_construction_readback_required": True,
        "candidate_policy_queried": False,
        "files": files,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_record = _write_document(
        root, "task_evaluation_franka_robotiq_readiness_inputs.v1.json", manifest
    )
    return {**manifest, "manifest": manifest_record}


__all__ = [
    "MATERIALIZATION_SCHEMA_VERSION",
    "ROBOT_IDENTITY",
    "TaskEvaluationFrankaRobotiqReadinessInputsError",
    "materialize_franka_robotiq_readiness_inputs",
]
