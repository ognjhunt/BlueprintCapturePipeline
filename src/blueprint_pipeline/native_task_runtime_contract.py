"""Seal the task-neutral inputs a native arm evaluation runtime must consume.

The original ADP-009D runtime learned these facts from Python constants: which
USD was the task object, whether it was rigid, where the Franka stood, and what
the cameras observed.  That works for one canned beverage and silently becomes
wrong for the next scene.  This contract makes those choices data, validates
their cross-links before a GPU launch, and gives the runtime one digest-bound
document to read.

This module deliberately does not claim the values were applied by Isaac.
Application and readback remain runtime gates; a valid document is only the
immutable request for those gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .articulated_runtime_composition import plan_articulated_runtime_composition
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .paired_target_native_construction_bindings import (
    PairedTargetNativeConstructionBindingsError,
    SCHEMA_VERSION as PAIRED_TARGET_CONSTRUCTION_SCHEMA_VERSION,
    validate_paired_target_native_construction_bindings,
)
from .replacement_construction_bindings import (
    ReplacementConstructionBindingsError,
    validate_materialized_replacement_construction_bindings,
)


SCHEMA_VERSION = "native_task_runtime_contract.v1"
PROGRAM_ID = "arm-decision-proof-v1"
FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")
SINGULAR_ASSET_ROLES = ("scene_collision", "scene_appearance", "task_object")
REPEATABLE_REPLACEMENT_ROLE = "replacement"
OBJECT_TYPES = frozenset({"RIGID", "ARTICULATION"})
CAMERA_ROLES = ("external", "wrist", "overview")
CAMERA_OPTICAL_CONVENTIONS = ("opencv",)
ENV_ROOT = "{ENV_REGEX_NS}"
TASK_KINDS = ("rigid_pick_place", "articulated_open_close")
SCENARIO_CONTEXT_KINDS = ("construction_canary", "evaluation_cell")
SUPPORTED_SCENARIO_RUNTIME_TARGETS = frozenset(
    {
        "EventManager.reset.object_start_position_m.y",
        "EventManager.reset.object_orientation.yaw",
        "EventManager.reset.external_camera.pose.position.x",
        "EventManager.reset.wrist_camera.pose.position.x",
    }
)
TASK_STATE_BINDING_SCHEMA_VERSION = "native_articulated_task_state_binding.v1"
GRAPH_TASK_STATE_BINDING_SCHEMA_VERSION = (
    "native_articulated_graph_task_state_binding.v1"
)
DROID_FRANKA_RESET_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)


class NativeTaskRuntimeContractError(ValueError):
    """Stable, sorted contract failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _finite_vector(
    value: Any, *, length: int, error: str, errors: list[str]
) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        errors.append(error)
        return []
    if len(result) != length or not all(math.isfinite(item) for item in result):
        errors.append(error)
        return []
    return result


def _pose(value: Any, *, error: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(error)
        return {}
    position = _finite_vector(
        value.get("position_world_m"), length=3, error=error, errors=errors
    )
    orientation = _finite_vector(
        value.get("orientation_xyzw"), length=4, error=error, errors=errors
    )
    if orientation:
        norm = math.sqrt(sum(item * item for item in orientation))
        if abs(norm - 1.0) > 1e-6:
            errors.append(error)
    return {
        "position_world_m": position,
        "orientation_xyzw": orientation,
    }


def _evidence_file_record_valid(record: Any, *, expected_digest: str) -> bool:
    return (
        isinstance(record, Mapping)
        and isinstance(record.get("path"), str)
        and bool(str(record.get("path") or "").strip())
        and isinstance(record.get("size_bytes"), int)
        and not isinstance(record.get("size_bytes"), bool)
        and int(record.get("size_bytes")) > 0
        and _digest(record.get("sha256"))
        and record.get("canonical_digest") == expected_digest
        and isinstance(record.get("schema_version"), str)
        and bool(str(record.get("schema_version") or "").strip())
    )


def _replacement_construction_evidence_bound(
    *,
    binding_rows: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    """Require file-backed evidence receipts before native construction launch."""

    required_receipts = {
        "task_freeze": "task_freeze_digest",
        "mask_set": "mask_set_receipt_digest",
        "gaussian_removal": "source_removal_receipt_digest",
        "replacement_qualification": "replacement_qualification_receipt_digest",
    }
    for index, row in enumerate(binding_rows):
        evidence = row.get("evidence_receipts")
        if not isinstance(evidence, Mapping):
            errors.append(
                f"native_task_runtime_construction_evidence_receipts_missing:{index}"
            )
            continue
        for receipt_id, digest_field in required_receipts.items():
            if not _evidence_file_record_valid(
                evidence.get(receipt_id),
                expected_digest=str(row.get(digest_field) or ""),
            ):
                errors.append(
                    "native_task_runtime_construction_evidence_receipt_invalid:"
                    f"{index}:{receipt_id}"
                )
        collider = evidence.get("source_collider_deletion")
        independent = (
            collider.get("independent") if isinstance(collider, Mapping) else None
        )
        if not _evidence_file_record_valid(
            independent,
            expected_digest=str(row.get("collider_deletion_receipt_digest") or ""),
        ):
            errors.append(
                "native_task_runtime_construction_evidence_receipt_invalid:"
                f"{index}:source_collider_deletion"
            )
        if (
            isinstance(collider, Mapping)
            and collider.get("selected_deletion_id") != row.get("collider_deletion_id")
        ):
            errors.append(
                "native_task_runtime_construction_collider_evidence_mismatch:"
                f"{index}"
            )


def _asset_rows(
    assets: Sequence[Mapping[str, Any]],
    *,
    subject_asset_id: str | None,
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    by_role: dict[str, Mapping[str, Any]] = {}
    replacements: list[Mapping[str, Any]] = []
    for index, row in enumerate(assets):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_runtime_asset_invalid:{index}")
            continue
        role = str(row.get("semantic_role") or "")
        if role == REPEATABLE_REPLACEMENT_ROLE:
            replacements.append(row)
            continue
        if role not in SINGULAR_ASSET_ROLES or role in by_role:
            errors.append(f"native_task_runtime_asset_role_invalid:{role or index}")
            continue
        by_role[role] = row
    if replacements and "task_object" in by_role:
        errors.append("native_task_runtime_legacy_and_replacement_assets_mixed")
    if replacements:
        if len(replacements) > MAX_REPLACEMENT_OBJECTS:
            errors.append("native_task_runtime_replacement_asset_count_out_of_range")
        replacement_ids = [str(row.get("asset_id") or "") for row in replacements]
        if (
            any(
                not asset_id
                or not asset_id.replace("_", "a").isalnum()
                for asset_id in replacement_ids
            )
            or len(replacement_ids) != len(set(replacement_ids))
        ):
            errors.append("native_task_runtime_replacement_asset_ids_invalid")
        selected = str(subject_asset_id or "")
        if replacement_ids.count(selected) != 1:
            errors.append("native_task_runtime_subject_asset_id_invalid")
        else:
            by_role["task_object"] = replacements[replacement_ids.index(selected)]
    elif subject_asset_id not in (None, ""):
        errors.append("native_task_runtime_subject_asset_id_unexpected")
    required = {"scene_collision", "task_object"}
    for role in sorted(required - set(by_role)):
        errors.append(f"native_task_runtime_asset_missing:{role}")

    rows: list[dict[str, Any]] = []
    ordered: list[tuple[str, Mapping[str, Any], bool]] = [
        (role, by_role[role], False)
        for role in ("scene_collision", "scene_appearance")
        if role in by_role
    ]
    if replacements:
        ordered.extend(
            (
                "task_object"
                if str(source.get("asset_id")) == str(subject_asset_id)
                else REPEATABLE_REPLACEMENT_ROLE,
                source,
                str(source.get("asset_id")) == str(subject_asset_id),
            )
            for source in replacements
        )
    elif "task_object" in by_role:
        ordered.append(("task_object", by_role["task_object"], True))
    for role, source, task_subject in ordered:
        asset_id = str(
            source.get("asset_id")
            or ("legacy_task_object" if role == "task_object" else role)
        )
        filename = str(source.get("filename") or "")
        if (
            not filename
            or PurePosixPath(filename).name != filename
            or filename in {".", ".."}
        ):
            errors.append(f"native_task_runtime_asset_filename_invalid:{role}")
        digest = str(source.get("sha256") or "")
        if not _digest(digest):
            errors.append(f"native_task_runtime_asset_digest_invalid:{role}")
        pose = _pose(
            source.get("pose_world"),
            error=f"native_task_runtime_asset_pose_invalid:{role}",
            errors=errors,
        )
        object_type = str(source.get("object_type") or "")
        if replacements and role in {"task_object", REPEATABLE_REPLACEMENT_ROLE}:
            if object_type not in OBJECT_TYPES:
                errors.append(
                    f"native_task_runtime_replacement_object_type_invalid:{asset_id}"
                )
        reset_state = source.get("reset_state")
        reset_joints: dict[str, float] = {}
        if replacements and role in {"task_object", REPEATABLE_REPLACEMENT_ROLE}:
            if not isinstance(reset_state, Mapping):
                errors.append(
                    f"native_task_runtime_replacement_reset_state_missing:{asset_id}"
                )
            else:
                joint_positions = reset_state.get("joint_positions")
                if not isinstance(joint_positions, Mapping):
                    errors.append(
                        f"native_task_runtime_replacement_joint_reset_invalid:{asset_id}"
                    )
                else:
                    for joint_name, raw in joint_positions.items():
                        name = str(joint_name or "")
                        number = _finite_vector(
                            [raw],
                            length=1,
                            error=(
                                "native_task_runtime_replacement_joint_reset_invalid:"
                                + asset_id
                            ),
                            errors=errors,
                        )
                        if not name or PurePosixPath(name).name != name:
                            errors.append(
                                f"native_task_runtime_replacement_joint_reset_invalid:{asset_id}"
                            )
                        elif number:
                            reset_joints[name] = number[0]
            if object_type == "RIGID" and reset_joints:
                errors.append(
                    f"native_task_runtime_rigid_replacement_has_joint_reset:{asset_id}"
                )
            if object_type == "ARTICULATION" and not reset_joints:
                errors.append(
                    f"native_task_runtime_articulated_replacement_joint_reset_missing:{asset_id}"
                )
        rows.append(
            {
                "name": str(source.get("name") or role),
                "semantic_role": role,
                "source_semantic_role": str(source.get("semantic_role") or role),
                "asset_id": asset_id,
                "runtime_name": (
                    "task_object"
                    if task_subject
                    else f"replacement__{asset_id}"
                    if role == REPEATABLE_REPLACEMENT_ROLE
                    else role
                ),
                "task_subject": task_subject,
                "filename": filename,
                "sha256": digest,
                "pose_world": pose,
                "object_type": object_type or None,
                "visible": bool(source.get("visible", True)),
                "reset_state": {
                    "root_pose_world": pose,
                    "joint_positions": reset_joints,
                },
            }
        )
    return rows, by_role


def _camera_rows(
    cameras: Sequence[Mapping[str, Any]], *, errors: list[str]
) -> list[dict[str, Any]]:
    by_role: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(cameras):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_runtime_camera_invalid:{index}")
            continue
        role = str(row.get("role") or "")
        if role not in CAMERA_ROLES or role in by_role:
            errors.append(f"native_task_runtime_camera_role_invalid:{role or index}")
            continue
        by_role[role] = row
    for role in sorted(set(CAMERA_ROLES) - set(by_role)):
        errors.append(f"native_task_runtime_camera_missing:{role}")

    rows: list[dict[str, Any]] = []
    for role in CAMERA_ROLES:
        if role not in by_role:
            continue
        source = by_role[role]
        expected_policy = role in {"external", "wrist"}
        if bool(source.get("policy_input")) is not expected_policy:
            errors.append(f"native_task_runtime_camera_policy_role_invalid:{role}")
        if bool(source.get("scoring_input")):
            errors.append(f"native_task_runtime_camera_scoring_forbidden:{role}")
        frame = str(source.get("pose_frame") or "")
        expected_frame = "robot_body" if role == "wrist" else "world"
        if frame != expected_frame:
            errors.append(f"native_task_runtime_camera_pose_frame_invalid:{role}")
        parent_prim_path = str(source.get("parent_prim_path") or "")
        expected_world_parent = ENV_ROOT
        robot_parent_valid = (
            parent_prim_path.startswith(f"{ENV_ROOT}/Robot/")
            and not parent_prim_path.endswith("/")
            and ".." not in PurePosixPath(parent_prim_path).parts
        )
        if (
            (frame == "world" and parent_prim_path != expected_world_parent)
            or (frame == "robot_body" and not robot_parent_valid)
        ):
            errors.append(f"native_task_runtime_camera_parent_invalid:{role}")
        matrix = _finite_vector(
            source.get("frame_from_camera_matrix"),
            length=16,
            error=f"native_task_runtime_camera_pose_invalid:{role}",
            errors=errors,
        )
        optical_convention = str(source.get("optical_convention") or "")
        if optical_convention not in CAMERA_OPTICAL_CONVENTIONS:
            errors.append(f"native_task_runtime_camera_convention_invalid:{role}")
        if matrix:
            rotation = [matrix[0:3], matrix[4:7], matrix[8:11]]
            bottom = matrix[12:16]
            row_norms = [sum(item * item for item in row) for row in rotation]
            pair_dots = [
                sum(rotation[left][axis] * rotation[right][axis] for axis in range(3))
                for left, right in ((0, 1), (0, 2), (1, 2))
            ]
            determinant = (
                rotation[0][0]
                * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
                - rotation[0][1]
                * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
                + rotation[0][2]
                * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
            )
            if (
                any(abs(norm - 1.0) > 1e-6 for norm in row_norms)
                or any(abs(dot) > 1e-6 for dot in pair_dots)
                or abs(determinant - 1.0) > 1e-6
                or any(
                    abs(actual - expected) > 1e-9
                    for actual, expected in zip(
                        bottom, (0.0, 0.0, 0.0, 1.0), strict=True
                    )
                )
            ):
                errors.append(f"native_task_runtime_camera_pose_invalid:{role}")
        intrinsics = source.get("intrinsics")
        if not isinstance(intrinsics, Mapping):
            errors.append(f"native_task_runtime_camera_intrinsics_invalid:{role}")
            intrinsics = {}
        intrinsic_values: dict[str, float | int] = {}
        try:
            intrinsic_values = {
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "width": int(intrinsics["width"]),
                "height": int(intrinsics["height"]),
            }
        except (KeyError, TypeError, ValueError):
            errors.append(f"native_task_runtime_camera_intrinsics_invalid:{role}")
        else:
            numeric = list(intrinsic_values.values())
            if (
                any(not math.isfinite(float(item)) for item in numeric)
                or intrinsic_values["fx"] <= 0
                or intrinsic_values["fy"] <= 0
                or intrinsic_values["width"] <= 0
                or intrinsic_values["height"] <= 0
            ):
                errors.append(f"native_task_runtime_camera_intrinsics_invalid:{role}")
        rows.append(
            {
                "role": role,
                "policy_input": expected_policy,
                "review_only": role == "overview",
                "scoring_input": False,
                "pose_frame": frame,
                "parent_prim_path": parent_prim_path,
                "optical_convention": optical_convention,
                "frame_from_camera_matrix": matrix,
                "intrinsics": intrinsic_values,
            }
        )
    return rows


def _articulated_task_state_binding(
    value: Any,
    *,
    task_kind: str,
    task_spec: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    if task_kind != "articulated_open_close":
        if value not in (None, {}):
            errors.append("native_task_runtime_state_binding_unexpected")
        return None
    if not isinstance(value, Mapping):
        errors.append("native_task_runtime_state_binding_missing")
        return None
    if task_spec.get("schema_version") == "adp_task_spec.v2":
        if value.get("schema_version") != GRAPH_TASK_STATE_BINDING_SCHEMA_VERSION:
            errors.append("native_task_runtime_graph_state_binding_schema_invalid")
        graph = task_spec.get("articulation_graph")
        affordance = task_spec.get("interaction_affordance")
        if not isinstance(graph, Mapping) or not isinstance(affordance, Mapping):
            errors.append("native_task_runtime_graph_state_contract_missing")
            graph = {"links": []}
            affordance = {}
        graph_digest = canonical_digest(dict(graph))
        affordance_digest = str(affordance.get("affordance_digest") or "")
        if (
            value.get("articulation_graph_digest") != graph_digest
            or value.get("interaction_affordance_digest") != affordance_digest
            or not _digest(graph_digest)
            or not _digest(affordance_digest)
        ):
            errors.append("native_task_runtime_graph_state_digest_mismatch")
        link_ids = {
            str(row.get("link_id") or "")
            for row in graph.get("links") or []
            if isinstance(row, Mapping)
        }
        raw_body_names = value.get("link_native_body_names")
        if not isinstance(raw_body_names, Mapping):
            errors.append("native_task_runtime_graph_link_body_binding_invalid")
            body_names: dict[str, str] = {}
        else:
            body_names = {
                str(link_id): str(body_name)
                for link_id, body_name in raw_body_names.items()
            }
            if (
                set(body_names) != link_ids
                or len(set(body_names.values())) != len(body_names)
                or any(
                    not name or PurePosixPath(name).name != name
                    for name in body_names.values()
                )
            ):
                errors.append("native_task_runtime_graph_link_body_binding_invalid")
        thresholds: dict[str, float] = {}
        for field in (
            "task_contact_minimum_force_n",
            "collision_failure_minimum_force_n",
            "retreat_minimum_separation_m",
            "root_translation_tolerance_m",
            "root_orientation_tolerance_rad",
        ):
            try:
                number = float(value[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"native_task_runtime_state_threshold_invalid:{field}")
                continue
            if not math.isfinite(number) or number <= 0.0:
                errors.append(f"native_task_runtime_state_threshold_invalid:{field}")
            thresholds[field] = number
        return {
            "schema_version": GRAPH_TASK_STATE_BINDING_SCHEMA_VERSION,
            "articulation_graph_digest": graph_digest,
            "interaction_affordance_digest": affordance_digest,
            "link_native_body_names": body_names,
            **thresholds,
            "measurement_authority": {
                "joint_state": "native_complete_graph_coordinate_readback",
                "contact_and_collision": "native_exact_body_filtered_contact_force",
                "containment": "native_task_root_pose_delta",
                "retreat": "native_grasp_frame_to_interaction_point_separation",
                "caller_asserted_booleans_forbidden": True,
            },
        }

    def _source_prim(field: str) -> str:
        path = str(value.get(field) or "")
        pure = PurePosixPath(path)
        if (
            not pure.is_absolute()
            or len(pure.parts) < 3
            or ".." in pure.parts
        ):
            errors.append(f"native_task_runtime_state_prim_invalid:{field}")
        return path

    moving_link = _source_prim("moving_link_prim_path")
    moving_link_native_body_name = str(
        value.get("moving_link_native_body_name") or ""
    )
    if (
        not moving_link_native_body_name
        or PurePosixPath(moving_link_native_body_name).name
        != moving_link_native_body_name
    ):
        errors.append("native_task_runtime_moving_link_body_name_invalid")
    handle_paths_raw = value.get("handle_prim_paths")
    handle_paths: list[str] = []
    if not isinstance(handle_paths_raw, Sequence) or isinstance(
        handle_paths_raw, (str, bytes)
    ):
        errors.append("native_task_runtime_handle_prims_invalid")
    else:
        for index, raw in enumerate(handle_paths_raw):
            path = str(raw or "")
            if (
                not path.startswith(moving_link + "/")
                or ".." in PurePosixPath(path).parts
                or path in handle_paths
            ):
                errors.append(f"native_task_runtime_handle_prim_invalid:{index}")
            handle_paths.append(path)
        if not handle_paths:
            errors.append("native_task_runtime_handle_prims_invalid")

    gripper_pattern = str(value.get("robot_gripper_contact_prim_pattern") or "")
    robot_pattern = str(value.get("robot_collision_prim_pattern") or "")
    for field, pattern in (
        ("robot_gripper_contact_prim_pattern", gripper_pattern),
        ("robot_collision_prim_pattern", robot_pattern),
    ):
        if not pattern.startswith("{ENV_REGEX_NS}/Robot/"):
            errors.append(f"native_task_runtime_robot_contact_pattern_invalid:{field}")

    thresholds: dict[str, float] = {}
    for field in (
        "task_contact_minimum_force_n",
        "collision_failure_minimum_force_n",
        "retreat_minimum_separation_m",
        "root_translation_tolerance_m",
        "root_orientation_tolerance_rad",
    ):
        try:
            number = float(value[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"native_task_runtime_state_threshold_invalid:{field}")
            continue
        if not math.isfinite(number) or number <= 0.0:
            errors.append(f"native_task_runtime_state_threshold_invalid:{field}")
        thresholds[field] = number

    return {
        "schema_version": TASK_STATE_BINDING_SCHEMA_VERSION,
        "moving_link_prim_path": moving_link,
        "moving_link_native_body_name": moving_link_native_body_name,
        "handle_prim_paths": handle_paths,
        "handle_grasp_point_link_m": _finite_vector(
            value.get("handle_grasp_point_link_m"),
            length=3,
            error="native_task_runtime_handle_grasp_point_invalid",
            errors=errors,
        ),
        "robot_gripper_contact_prim_pattern": gripper_pattern,
        "robot_collision_prim_pattern": robot_pattern,
        **thresholds,
        "measurement_authority": {
            "joint_state": "native_articulation_readback",
            "contact_and_collision": "native_filtered_contact_sensor_force",
            "containment": "native_task_root_pose_delta",
            "retreat": "native_grasp_frame_to_handle_distance_after_release",
            "caller_asserted_booleans_forbidden": True,
        },
    }


def _robot_joint_reset_positions(
    value: Any, *, errors: list[str]
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        errors.append("native_task_runtime_robot_reset_joints_missing")
        return {}
    observed = {str(name) for name in value}
    expected = set(DROID_FRANKA_RESET_JOINT_NAMES)
    for name in sorted(expected - observed):
        errors.append(f"native_task_runtime_robot_reset_joint_missing:{name}")
    for name in sorted(observed - expected):
        errors.append(f"native_task_runtime_robot_reset_joint_unexpected:{name}")
    resolved: dict[str, float] = {}
    for name in DROID_FRANKA_RESET_JOINT_NAMES:
        try:
            number = float(value[name])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(number):
            errors.append(f"native_task_runtime_robot_reset_joint_invalid:{name}")
        resolved[name] = number
    return resolved


def _scenario_parameter_rows(
    value: Sequence[Mapping[str, Any]] | None, *, errors: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value or []):
        if not isinstance(raw, Mapping):
            errors.append(f"native_task_runtime_scenario_parameter_invalid:{index}")
            continue
        parameter_id = str(raw.get("parameter_id") or "")
        target = str(raw.get("runtime_target") or "")
        unit = str(raw.get("unit") or "")
        try:
            nominal = float(raw["nominal_value"])
            resolved = float(raw["resolved_value"])
            tolerance = float(raw["application_tolerance"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"native_task_runtime_scenario_parameter_invalid:{index}")
            continue
        if target not in SUPPORTED_SCENARIO_RUNTIME_TARGETS:
            errors.append(f"native_task_runtime_scenario_target_unsupported:{target}")
            continue
        if (
            not parameter_id
            or parameter_id in seen
            or unit not in {"m", "degrees"}
            or not all(math.isfinite(item) for item in (nominal, resolved, tolerance))
            or tolerance <= 0.0
        ):
            errors.append(f"native_task_runtime_scenario_parameter_invalid:{index}")
            continue
        seen.add(parameter_id)
        rows.append(
            {
                "parameter_id": parameter_id,
                "runtime_target": target,
                "unit": unit,
                "nominal_value": nominal,
                "resolved_value": resolved,
                "application_tolerance": tolerance,
            }
        )
    return rows


def materialize_native_task_runtime_contract(
    *,
    scene_id: str,
    task_id: str,
    task_spec: Mapping[str, Any],
    task_joint_bindings: Sequence[Mapping[str, Any]] | None = None,
    task_state_binding: Mapping[str, Any] | None = None,
    assets: Sequence[Mapping[str, Any]],
    robot_base_pose_world: Mapping[str, Any],
    robot_joint_reset_positions_rad: Mapping[str, float],
    cameras: Sequence[Mapping[str, Any]],
    scenario_cell_id: str,
    scenario_instance_digest: str,
    seed: int,
    scenario_context_kind: str = "evaluation_cell",
    scenario_parameter_bindings: Sequence[Mapping[str, Any]] | None = None,
    construction_bindings: Mapping[str, Any] | None = None,
    task_freeze_digest: str | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and freeze one native scene/task request before execution."""

    errors: list[str] = []
    scene = str(scene_id or "").strip()
    task = str(task_id or "").strip()
    if not scene:
        errors.append("native_task_runtime_scene_id_missing")
    if not task:
        errors.append("native_task_runtime_task_id_missing")
    if not isinstance(task_spec, Mapping):
        raise NativeTaskRuntimeContractError(["native_task_runtime_task_spec_invalid"])
    task_kind = str(task_spec.get("task_kind") or "")
    if task_kind not in TASK_KINDS:
        errors.append("native_task_runtime_task_kind_invalid")
    if not str(scenario_cell_id or "").strip():
        errors.append("native_task_runtime_scenario_cell_missing")
    context_kind = str(scenario_context_kind or "").strip()
    if context_kind not in SCENARIO_CONTEXT_KINDS:
        errors.append("native_task_runtime_scenario_context_kind_invalid")
    if not _digest(scenario_instance_digest):
        errors.append("native_task_runtime_scenario_digest_invalid")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("native_task_runtime_seed_invalid")

    subject_asset_id = task_spec.get("subject_asset_id")
    asset_rows, by_asset_role = _asset_rows(
        assets,
        subject_asset_id=(
            str(subject_asset_id) if subject_asset_id is not None else None
        ),
        errors=errors,
    )
    replacement_rows = [
        row
        for row in asset_rows
        if row.get("source_semantic_role") == REPEATABLE_REPLACEMENT_ROLE
    ]
    qualified_construction: dict[str, Any] | None = None
    if replacement_rows:
        if not isinstance(construction_bindings, Mapping):
            errors.append("native_task_runtime_construction_bindings_missing")
        else:
            try:
                if (
                    construction_bindings.get("schema_version")
                    == PAIRED_TARGET_CONSTRUCTION_SCHEMA_VERSION
                ):
                    qualified_construction = (
                        validate_paired_target_native_construction_bindings(
                            construction_bindings
                        )
                    )
                else:
                    qualified_construction = (
                        validate_materialized_replacement_construction_bindings(
                            construction_bindings
                        )
                    )
            except (
                PairedTargetNativeConstructionBindingsError,
                ReplacementConstructionBindingsError,
            ) as exc:
                errors.extend(exc.errors)
        if not _digest(task_freeze_digest):
            errors.append("native_task_runtime_task_freeze_digest_invalid")
        if qualified_construction is not None:
            binding_rows = qualified_construction["bindings"]
            if (
                qualified_construction.get("schema_version")
                != PAIRED_TARGET_CONSTRUCTION_SCHEMA_VERSION
            ):
                _replacement_construction_evidence_bound(
                    binding_rows=binding_rows,
                    errors=errors,
                )
            expected_asset_ids = {row["asset_id"] for row in replacement_rows}
            observed_asset_ids = {row["asset_id"] for row in binding_rows}
            if expected_asset_ids != observed_asset_ids:
                errors.append("native_task_runtime_construction_asset_set_mismatch")
            task_binding = [
                row
                for row in binding_rows
                if row["task_id"] == task and row["asset_id"] == subject_asset_id
            ]
            if len(task_binding) != 1:
                errors.append("native_task_runtime_construction_task_binding_missing")
            elif task_binding[0]["task_freeze_digest"] != task_freeze_digest:
                errors.append("native_task_runtime_task_freeze_binding_mismatch")
            asset_digests = {
                row["asset_id"]: row["sha256"] for row in replacement_rows
            }
            for row in binding_rows:
                if asset_digests.get(row["asset_id"]) != row["replacement_asset_sha256"]:
                    errors.append(
                        "native_task_runtime_replacement_qualification_asset_mismatch:"
                        + row["asset_id"]
                    )
    elif construction_bindings is not None or task_freeze_digest is not None:
        errors.append("native_task_runtime_legacy_construction_bindings_unexpected")
    composition: dict[str, Any] = {}
    if {"scene_collision", "task_object"}.issubset(by_asset_role):
        try:
            composition = plan_articulated_runtime_composition(
                task_spec=task_spec,
                task_joint_bindings=task_joint_bindings,
                twin_usd_filename=str(by_asset_role["task_object"].get("filename") or ""),
                scene_collision_filename=str(
                    by_asset_role["scene_collision"].get("filename") or ""
                ),
                appearance_filename=(
                    str(by_asset_role["scene_appearance"].get("filename") or "")
                    if "scene_appearance" in by_asset_role
                    else None
                ),
                twin_position_world_m=(
                    by_asset_role["task_object"].get("pose_world") or {}
                ).get("position_world_m"),
                twin_object_type=(
                    str(by_asset_role["task_object"].get("object_type") or "")
                    or None
                ),
            )
        except ValueError as exc:
            errors.append(f"native_task_runtime_composition_invalid:{exc}")
    if composition:
        planned = {row["semantic_role"]: row for row in composition["objects"]}
        for row in asset_rows:
            role = row["semantic_role"]
            if role in planned:
                row["object_type"] = planned[role]["object_type"]
                row["visible"] = planned[role]["visible"]

    robot_pose = _pose(
        robot_base_pose_world,
        error="native_task_runtime_robot_base_pose_invalid",
        errors=errors,
    )
    robot_reset_positions = _robot_joint_reset_positions(
        robot_joint_reset_positions_rad, errors=errors
    )
    camera_rows = _camera_rows(cameras, errors=errors)
    scenario_parameters = _scenario_parameter_rows(
        scenario_parameter_bindings, errors=errors
    )
    state_binding = _articulated_task_state_binding(
        task_state_binding,
        task_kind=task_kind,
        task_spec=task_spec,
        errors=errors,
    )
    if errors:
        raise NativeTaskRuntimeContractError(errors)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "scene_id": scene,
        "task_id": task,
        "task_kind": task_kind,
        "task_subject_asset_id": next(
            row["asset_id"] for row in asset_rows if row["task_subject"]
        ),
        "task_freeze_digest": task_freeze_digest,
        "construction_bindings": qualified_construction,
        "task_spec": json.loads(json.dumps(task_spec, sort_keys=True)),
        "task_spec_digest": canonical_digest(dict(task_spec)),
        "scenario": {
            "context_kind": context_kind,
            "cell_id": str(scenario_cell_id),
            "instance_digest": str(scenario_instance_digest),
            "seed": seed,
            "parameter_bindings": scenario_parameters,
        },
        "candidate_ids": list(FROZEN_CANDIDATES),
        "objects": asset_rows,
        "robot": {
            "robot_id": "franka_panda",
            "base_pose_world": robot_pose,
            "joint_reset_positions_rad": robot_reset_positions,
            "grasp_frame": {
                "kind": "body_midpoint",
                "body_names": ["left_inner_finger", "right_inner_finger"],
                "measurement_authority": "native_robot_body_pose_readback",
            },
            "action_seam": {
                "kind": "joint_position_with_gripper",
                "arm_joint_count": 7,
                "action_dimension": 8,
                "gripper_command_source": "native_readback_resolved",
            },
        },
        "cameras": camera_rows,
        "task_sample_binding": composition["task_sample_binding"],
        "task_state_binding": state_binding,
        "reset_contract": {
            "same_scene_bytes": True,
            "same_object_bytes": True,
            "same_cameras": True,
            "same_scorer": True,
            "native_state_readback_required_after_reset": True,
            "per_object_reset_states": {
                row["asset_id"]: row["reset_state"]
                for row in asset_rows
                if row["semantic_role"] in {"task_object", "replacement"}
            },
        },
        "scoring_contract": {
            "source": "deterministic_simulator_state",
            "policy_may_grade_itself": False,
            "overview_camera_may_score": False,
        },
        "runtime_readback_required": {
            "object_types": True,
            "asset_digests": True,
            "world_poses": True,
            "camera_transforms_and_intrinsics": True,
            "scenario_parameters": True,
            "task_joint_indices": task_kind == "articulated_open_close",
        },
        "claim_boundary": {
            "valid_contract_is_not_native_application_proof": True,
            "simulator_execution_is_not_physical_truth": True,
            "public_dataset_rehearsal_only": True,
        },
        "contract_digest": "",
    }
    document["contract_digest"] = canonical_digest(
        document, digest_field="contract_digest"
    )
    if destination is not None:
        write_json(Path(destination), document)
    return json.loads(json.dumps(document))


def load_native_task_runtime_contract(path: str | Path) -> dict[str, Any]:
    """Load a materialized contract and reject schema or digest drift."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskRuntimeContractError(
            ["native_task_runtime_contract_unreadable"]
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise NativeTaskRuntimeContractError(["native_task_runtime_contract_schema_invalid"])
    expected = canonical_digest(document, digest_field="contract_digest")
    if document.get("contract_digest") != expected:
        raise NativeTaskRuntimeContractError(["native_task_runtime_contract_digest_invalid"])
    return document


__all__ = [
    "DROID_FRANKA_RESET_JOINT_NAMES",
    "FROZEN_CANDIDATES",
    "NativeTaskRuntimeContractError",
    "SCENARIO_CONTEXT_KINDS",
    "SCHEMA_VERSION",
    "TASK_STATE_BINDING_SCHEMA_VERSION",
    "load_native_task_runtime_contract",
    "materialize_native_task_runtime_contract",
]
