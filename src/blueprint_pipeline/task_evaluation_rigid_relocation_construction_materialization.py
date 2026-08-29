"""Derive one sealed native rigid-construction authority from configured bytes.

The configured-scene run is robot neutral, so it cannot publish a robot action
plan.  It does publish enough immutable evidence to derive a conservative
planar-push contact candidate: native-qualified replacement geometry, the
registered support, the robot-neutral mount clearance, and preregistered task
centres.  This module binds those exact bytes and emits only inputs for the
existing native construction gates.  Reachability, collision clearance,
contact, support, reset, and task success remain unqualified until those gates
execute in Isaac.
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


SCHEMA_VERSION = "task_evaluation_rigid_relocation_construction_materialization.v1"
STATIC_PATH = "scene.configured_revision.replacement.static_qualification"
NATIVE_PATH = "scene.configured_revision.replacement.native_import_qualification"
SUPPORT_PATH = "scene.configured_revision.registration.support_plane"
MOUNT_PATH = "scene.configured_revision.registration.robot_mount_interface"
WORKSPACE_PATH = "scene.configured_revision.registration.workspace_clearance"
DEFINITION_PATH = "scene.configured_revision.task_template.definition"
SUCCESS_PATH = "scene.configured_revision.task_template.success_criteria"
EXECUTION_PATH = "scene.configured_revision.task_template.execution"

# These are sensor decision thresholds already used by Blueprint's production
# native rigid/articulated controls contracts.  They are explicitly simulator
# gate policy, not inferred force, material, or physical-site truth.
TASK_CONTACT_MINIMUM_FORCE_N = 0.5
COLLISION_FAILURE_MINIMUM_FORCE_N = 1.0
FORCE_THRESHOLD_POLICY = "blueprint_native_contact_sensor_policy.v1"
RELEASE_GRIPPER_WIDTH_MIN_M = 0.06
RELEASE_WIDTH_POLICY = "robotiq_2f85_open_release_policy.v1"


class TaskEvaluationRigidRelocationConstructionMaterializationError(ValueError):
    """Exact configured evidence cannot authorize a rigid construction input."""


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _bound_document(
    references: Mapping[str, Mapping[str, Any]],
    *,
    contract_path: str,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = references.get(contract_path)
    path = Path(str((row or {}).get("materialized_path") or "")).resolve()
    if (
        row is None
        or row.get("contract_path") != contract_path
        or row.get("uri") != expected.get("uri")
        or row.get("digest") != expected.get("digest")
        or row.get("size_bytes") != expected.get("size_bytes")
        or row.get("full_byte_service_account_readback_passed") is not True
        or path.is_symlink()
        or not path.is_file()
        or _file_identity(path) != (row.get("digest"), row.get("size_bytes"))
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            f"rigid_construction_source_invalid:{contract_path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            f"rigid_construction_source_json_invalid:{contract_path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            f"rigid_construction_source_contract_invalid:{contract_path}"
        )
    return json.loads(json.dumps(dict(value))), {
        "contract_path": contract_path,
        "uri": row["uri"],
        "digest": row["digest"],
        "size_bytes": row["size_bytes"],
        "schema_version": value.get("schema_version"),
        "canonical_document_digest": canonical_digest(value),
        "full_byte_service_account_readback_passed": True,
    }


def _vector(value: Any, *, length: int, blocker: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(blocker)
    return result


def _positive(value: Any, *, blocker: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(blocker) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(blocker)
    return result


def _quaternion(value: Any, *, blocker: str) -> list[float]:
    result = _vector(value, length=4, blocker=blocker)
    if not math.isclose(sum(item * item for item in result), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(blocker)
    return result


def _rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _scene_id(value: Mapping[str, Any]) -> str:
    return str(value.get("id") or "").rsplit("-", 1)[-1]


def _sealed(value: Mapping[str, Any], *, field: str, schema: str) -> bool:
    return value.get("schema_version") == schema and value.get(field) == canonical_digest(
        value, digest_field=field
    )


def materialize_rigid_relocation_construction_authority(
    *,
    configured_revision: Mapping[str, Any],
    materialized_references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return digest-bound fields consumed by native rigid construction."""

    revision = validate_configured_scene_revision(configured_revision)
    expected = {
        STATIC_PATH: revision["replacement"]["static_qualification"],
        NATIVE_PATH: revision["replacement"]["native_import_qualification"],
        SUPPORT_PATH: revision["registration"]["support_plane"],
        MOUNT_PATH: revision["registration"]["robot_mount_interface"],
        WORKSPACE_PATH: revision["registration"]["workspace_clearance"],
        DEFINITION_PATH: revision["task_template"]["definition"],
        SUCCESS_PATH: revision["task_template"]["success_criteria"],
        EXECUTION_PATH: revision["task_template"]["execution"],
    }
    documents: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for contract_path, reference in expected.items():
        document, binding = _bound_document(
            materialized_references,
            contract_path=contract_path,
            expected=reference,
        )
        documents[contract_path] = document
        bindings.append(binding)

    static = documents[STATIC_PATH]
    native = documents[NATIVE_PATH]
    support = documents[SUPPORT_PATH]
    mount = documents[MOUNT_PATH]
    workspace = documents[WORKSPACE_PATH]
    definition = documents[DEFINITION_PATH]
    success = documents[SUCCESS_PATH]
    execution = documents[EXECUTION_PATH]
    identity = revision["replacement"]["identity"]
    asset_digest = revision["replacement"]["asset"]["digest"]
    if (
        not _sealed(
            static,
            field="result_digest",
            schema="task_evaluation_rigid_replacement_static_qualification.v1",
        )
        or static.get("status") != "authored_structure_statically_qualified"
        or static.get("replacement_identity") != identity
        or static.get("replacement_usd", {}).get("sha256") != asset_digest
        or static.get("authored_structure_statically_qualified") is not True
        or static.get("structural_findings") != []
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_static_qualification_invalid"
        )
    if (
        not _sealed(
            native,
            field="result_digest",
            schema="task_evaluation_replacement_native_import_result.v1",
        )
        or native.get("status") != "qualified"
        or native.get("replacement_identity") != identity
        or native.get("asset_digest") != asset_digest
        or native.get("static_qualification_digest")
        != revision["replacement"]["static_qualification"]["digest"]
        or native.get("native_isaac_executed") is not True
        or native.get("native_simulator_import_qualified") is not True
        or native.get("support_contact_observed") is not True
        or native.get("blockers") != []
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_native_qualification_invalid"
        )

    observed = static.get("observed_structure")
    bounds = (
        observed.get("collision_bounds_asset_root_m") if isinstance(observed, Mapping) else None
    )
    if not isinstance(bounds, Mapping):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_static_geometry_missing"
        )
    lower = _vector(
        bounds.get("minimum"), length=3, blocker="rigid_construction_static_geometry_invalid"
    )
    upper = _vector(
        bounds.get("maximum"), length=3, blocker="rigid_construction_static_geometry_invalid"
    )
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_static_geometry_invalid"
        )
    dimensions = [high - low for low, high in zip(lower, upper, strict=True)]
    if any(
        not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-7)
        for actual, expected_value in zip(
            dimensions,
            _vector(
                observed.get("collision_dimensions_m"),
                length=3,
                blocker="rigid_construction_static_geometry_invalid",
            ),
            strict=True,
        )
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_static_geometry_invalid"
        )
    rigid_body_paths = observed.get("rigid_body_paths")
    if (
        not isinstance(rigid_body_paths, list)
        or not rigid_body_paths
        or len(set(rigid_body_paths)) != len(rigid_body_paths)
        or any(not str(path).startswith("/") for path in rigid_body_paths)
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_contact_topology_invalid"
        )

    limits = native.get("qualification_limits")
    repeats = native.get("repeats")
    if not isinstance(limits, Mapping) or not isinstance(repeats, list) or not repeats:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_native_stability_authority_missing"
        )
    settle_translation = _positive(
        limits.get("maximum_settle_translation_m"),
        blocker="rigid_construction_native_stability_authority_invalid",
    )
    settle_rotation = _positive(
        limits.get("maximum_settle_rotation_rad"),
        blocker="rigid_construction_native_stability_authority_invalid",
    )
    settle_seconds = _positive(
        limits.get("gravity_settle_seconds"),
        blocker="rigid_construction_native_stability_authority_invalid",
    )
    repeat_count = limits.get("state_digest_repeat_count")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 2
        or repeat_count != len(repeats)
        or repeat_count != native.get("deterministic_reset_state_digest_repeat_count")
        or len({str(row.get("final_state_digest") or "") for row in repeats}) != 1
        or any(row.get("final_state") != repeats[0].get("final_state") for row in repeats)
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_native_stability_authority_invalid"
        )
    stable_state = repeats[0].get("final_state")
    if not isinstance(stable_state, Mapping):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_native_stability_authority_invalid"
        )
    stable_orientation = _quaternion(
        stable_state.get("orientation_xyzw"),
        blocker="rigid_construction_native_stability_authority_invalid",
    )

    scene_id = _scene_id(revision["scene_identity"])
    if (
        support.get("schema_version") != "task_evaluation_support_plane_input.v1"
        or str(support.get("scene_id") or "") != scene_id
        or not str(support.get("sage_prim_path") or "").startswith("/")
        or support.get("required_validation")
        != ["planarity", "finite_bounds", "support_contact", "target_region_inside_bounds"]
        or mount.get("schema_version") != "task_evaluation_robot_mount_interface_plan.v1"
        or str(mount.get("scene_id") or "") != scene_id
        or mount.get("workspace_clearance_envelope_required") is not True
        or mount.get("configuration_run_must_not_claim_any_robot_qualified") is not True
        or workspace.get("schema_version") != "registered_sage_franka_placement_packet.v1"
        or workspace.get("request", {}).get("candidate_may_self_authorize") is not False
        or workspace.get("native_contact_reachability_qualified") is not False
        or workspace.get("policy_execution_authorized") is not False
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_registration_authority_invalid"
        )
    support_minimum = _vector(
        support.get("bounds_min_xyz_m"),
        length=3,
        blocker="rigid_construction_support_bounds_invalid",
    )
    support_maximum = _vector(
        support.get("bounds_max_xyz_m"),
        length=3,
        blocker="rigid_construction_support_bounds_invalid",
    )
    support_top = _positive(
        support.get("top_z_m"), blocker="rigid_construction_support_bounds_invalid"
    )
    clearance = _positive(
        mount.get("minimum_non_target_clearance_m"),
        blocker="rigid_construction_mount_clearance_invalid",
    )

    if (
        definition.get("schema_version") != "task_evaluation_rigid_relocation_template.v1"
        or success.get("schema_version") != "task_evaluation_rigid_relocation_success_criteria.v1"
        or execution.get("schema_version") != "task_evaluation_rigid_relocation_execution_spec.v1"
        or definition.get("task_identity") != revision["task_template"]["identity"]
        or definition.get("object_identity") != identity
        or definition.get("strategy") != "planar_push"
        or execution.get("strategy") != "planar_push"
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_task_authority_invalid"
        )
    start = _vector(
        definition.get("start_center_xyz_m"),
        length=3,
        blocker="rigid_construction_task_pose_invalid",
    )
    target = _vector(
        definition.get("target_center_xyz_m"),
        length=3,
        blocker="rigid_construction_task_pose_invalid",
    )
    if (
        execution.get("start_center_xyz_m") != definition.get("start_center_xyz_m")
        or execution.get("target_center_xyz_m") != definition.get("target_center_xyz_m")
        or success.get("target_center_xyz_m") != definition.get("target_center_xyz_m")
        or not math.isclose(start[2], target[2], rel_tol=0.0, abs_tol=1e-9)
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_task_pose_invalid"
        )
    push = [target[index] - start[index] for index in range(3)]
    push_norm = math.hypot(push[0], push[1])
    if push_norm <= 0.0 or abs(push[2]) > 1e-9:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_planar_push_direction_invalid"
        )
    push_world = [push[0] / push_norm, push[1] / push_norm, 0.0]
    approach_world = [-push_world[0], -push_world[1], 0.0]
    inverse_orientation = [
        -stable_orientation[0],
        -stable_orientation[1],
        -stable_orientation[2],
        stable_orientation[3],
    ]
    approach_local = _rotate_xyzw(inverse_orientation, approach_world)
    half_extents = [value / 2.0 for value in dimensions]
    ray_scales = [
        half_extents[index] / abs(approach_local[index])
        for index in range(3)
        if abs(approach_local[index]) > 1e-9
    ]
    if not ray_scales:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_contact_point_invalid"
        )
    contact_scale = min(ray_scales)
    contact_point = [value * contact_scale for value in approach_local]
    scoring_offset = [(low + high) / 2.0 for low, high in zip(lower, upper, strict=True)]
    rotated_scoring_offset = _rotate_xyzw(stable_orientation, scoring_offset)
    root_position = [start[index] - rotated_scoring_offset[index] for index in range(3)]

    object_height = dimensions[2]
    object_radius = math.hypot(dimensions[0], dimensions[1]) / 2.0
    expected_center_z = support_top + object_height / 2.0
    if abs(start[2] - expected_center_z) > settle_translation:
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_support_height_mismatch"
        )
    inset = object_radius + clearance
    workspace_bounds = {
        "minimum": [
            support_minimum[0] + inset,
            support_minimum[1] + inset,
            start[2] - settle_translation,
        ],
        "maximum": [
            support_maximum[0] - inset,
            support_maximum[1] - inset,
            start[2] + settle_translation,
        ],
    }
    if any(
        not all(
            workspace_bounds["minimum"][index] <= point[index] <= workspace_bounds["maximum"][index]
            for index in range(3)
        )
        for point in (start, target)
    ):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_task_outside_registered_support"
        )
    control_frequency = _positive(
        execution.get("control_frequency_hz"),
        blocker="rigid_construction_control_frequency_invalid",
    )
    settle_window = int(math.ceil(settle_seconds * control_frequency))
    target_tolerance = _positive(
        success.get("maximum_final_planar_target_error_m"),
        blocker="rigid_construction_target_tolerance_invalid",
    )
    minimum_translation = _positive(
        success.get("minimum_planar_displacement_m"),
        blocker="rigid_construction_translation_threshold_invalid",
    )
    action_bounds = execution.get("action_bounds_m_per_step")
    if not isinstance(action_bounds, Mapping):
        raise TaskEvaluationRigidRelocationConstructionMaterializationError(
            "rigid_construction_action_bounds_invalid"
        )
    action_scale = min(
        _positive(
            abs(action_bounds.get("minimum", 0.0)),
            blocker="rigid_construction_action_bounds_invalid",
        ),
        _positive(action_bounds.get("maximum"), blocker="rigid_construction_action_bounds_invalid"),
    )

    affordance: dict[str, Any] = {
        "schema_version": "native_rigid_interaction_affordance.v1",
        "subject_asset_id": identity["id"],
        "scoring_frame_id": "task_scoring_frame",
        "asset_root_from_scoring_frame": {
            "position_m": scoring_offset,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_point_scoring_frame_m": contact_point,
        "approach_unit_scoring_frame": approach_local,
        "lift_unit_world": [0.0, 0.0, 1.0],
        "gripper_orientation_scoring_frame_xyzw": [0.0, 0.0, 0.0, 1.0],
        "pregrasp_clearance_m": clearance,
        "arrival_orientation_tolerance_rad": settle_rotation,
        "allowed_contact_prim_paths": list(rigid_body_paths),
        "intended_support_prim_paths": [str(support["sage_prim_path"])],
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(affordance, digest_field="affordance_digest")
    task_spec_fields = {
        "start_pose_world": [*start, *stable_orientation],
        "destination_position_bounds_world_m": {
            "minimum": [target[index] - target_tolerance for index in range(3)],
            "maximum": [target[index] + target_tolerance for index in range(3)],
        },
        "support_height_interval_m": [
            start[2] - settle_translation,
            start[2] + settle_translation,
        ],
        "destination_orientation_xyzw": stable_orientation,
        "destination_orientation_tolerance_rad": settle_rotation,
        "minimum_lift_m": 0.0,
        "minimum_translation_m": minimum_translation,
        "movement_epsilon_m": min(action_scale, target_tolerance) / 4.0,
        "settle_window_samples": settle_window,
        "release_required": True,
        "release_gripper_width_min_m": RELEASE_GRIPPER_WIDTH_MIN_M,
        "task_contact_minimum_force_n": TASK_CONTACT_MINIMUM_FORCE_N,
        "collision_failure_minimum_force_n": COLLISION_FAILURE_MINIMUM_FORCE_N,
        "reset_translation_tolerance_m": settle_translation,
        "reset_orientation_tolerance_rad": settle_rotation,
        "settle_position_tolerance_m": settle_translation,
        "settle_orientation_tolerance_rad": settle_rotation,
        "relocation_tracking_tolerance_m": min(action_scale, target_tolerance),
        "workspace_position_bounds_world_m": workspace_bounds,
        "interaction_affordance": affordance,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized_pending_native_construction",
        "configured_scene_revision_digest": revision["revision_digest"],
        "task_identity": dict(revision["task_template"]["identity"]),
        "subject_identity": dict(identity),
        "manipulation_strategy": "planar_push",
        "source_bindings": bindings,
        "source_bindings_digest": canonical_digest({"bindings": bindings}),
        "task_spec_fields": task_spec_fields,
        "task_object_pose_world": {
            "position_world_m": root_position,
            "orientation_xyzw": stable_orientation,
        },
        "scenario_parameters": {
            "object_height_m": object_height,
            "object_radius_m": object_radius,
        },
        "threshold_authority": {
            "force_policy": FORCE_THRESHOLD_POLICY,
            "release_width_policy": RELEASE_WIDTH_POLICY,
            "reset_and_settle_limits_from_native_qualification": True,
            "workspace_from_support_radius_and_mount_clearance": True,
        },
        "claim_boundary": {
            "native_construction_qualified": False,
            "robot_reachability_qualified": False,
            "collision_clearance_qualified": False,
            "task_contact_qualified": False,
            "physical_site_truth_claimed": False,
            "learned_policy_outcomes_consulted": False,
        },
        "materialization_digest": "",
    }
    result["materialization_digest"] = canonical_digest(
        result, digest_field="materialization_digest"
    )
    return result


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationRigidRelocationConstructionMaterializationError",
    "materialize_rigid_relocation_construction_authority",
]
