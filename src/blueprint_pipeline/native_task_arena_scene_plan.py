"""Compile a native Arena scene from the task-neutral runtime contract.

This module deliberately imports no Isaac or Arena package.  It settles the
expensive-to-discover facts before a GPU launch: exact staged bytes, spawn
types and prim paths, articulation reset names, contact filters, robot reset
state, camera roles, and the physics/control cadence.  The GPU worker consumes
the resulting plan and must still retain native application/readback evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .articulation_graph_contract import (
    ArticulationGraphContractError,
    validate_articulation_graph,
)
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_articulated_motion_geometry import (
    NativeArticulatedMotionGeometryError,
    derive_native_articulated_motion_geometry,
)
from .native_task_runtime_contract import SCHEMA_VERSION as RUNTIME_CONTRACT_SCHEMA
from .native_task_gpu_collision_qualification import (
    audit_native_task_gpu_collisions,
)
from .native_task_robot_contact_topology import (
    NativeTaskRobotContactTopologyError,
    resolve_native_task_robot_contact_topology,
)


SCHEMA_VERSION = "native_task_arena_scene_plan.v1"
ENV_ROOT = "{ENV_REGEX_NS}"


class NativeTaskArenaScenePlanError(ValueError):
    """Stable, sorted failures raised before the native runtime starts."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_to_spawned_prim(
    source_path: str, *, role: str, source_root_prim_path: str
) -> str:
    if source_path == source_root_prim_path:
        suffix = ""
    elif source_path.startswith(source_root_prim_path + "/"):
        suffix = source_path[len(source_root_prim_path) :]
    else:
        raise NativeTaskArenaScenePlanError(
            [f"native_task_arena_source_prim_invalid:{source_path}"]
        )
    return f"{ENV_ROOT}/{role}{suffix}"


def _verified_articulation_row(
    asset_path: Path, *, role: str, declared: Any
) -> dict[str, Any]:
    """Prove the staged bytes are creatable and agree with the declaration."""

    from .native_task_arena_runtime import (
        NativeTaskArenaRuntimeError,
        verify_grounded_articulation,
    )

    try:
        verified = verify_grounded_articulation(asset_path)
    except NativeTaskArenaRuntimeError as exc:
        raise NativeTaskArenaScenePlanError(list(exc.errors)) from exc
    except Exception as exc:  # pxr raises Tf.ErrorException on unreadable usd
        raise NativeTaskArenaScenePlanError(
            [f"native_task_arena_articulation_unreadable:{role}"]
        ) from exc
    declared_base = (
        declared.get("fixed_base_body_prim_path")
        if isinstance(declared, Mapping)
        else None
    )
    if verified["fixed_base_body_prim_path"] != declared_base:
        raise NativeTaskArenaScenePlanError(
            [f"native_task_arena_articulation_adaptation_mismatch:{role}"]
        )
    if isinstance(declared, Mapping):
        return dict(declared)
    return {
        "fixed_base_body_prim_path": None,
        "adaptation": "candidate_authored_dynamic_articulation",
        "candidate_bytes_modified": False,
    }


def _exact_scene_contact_body_paths(scene_collision_asset_path: Path) -> list[str]:
    """Resolve every static or rigid collision actor to one exact spawned path."""

    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(scene_collision_asset_path))
    except (ImportError, RuntimeError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_scene_collision_topology_unreadable"]
        ) from exc
    if stage is None:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_scene_collision_topology_unreadable"]
        )
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_scene_collision_default_prim_missing"]
        )
    root = str(default_prim.GetPath())
    paths = sorted(
        {
            _source_to_spawned_prim(
                str(prim.GetPath()),
                role="scene_collision",
                source_root_prim_path=root,
            )
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        }
    )
    if not paths:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_scene_collision_bodies_missing"]
        )
    if any(any(token in path for token in ("*", ".*", "[", "]")) for path in paths):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_scene_collision_body_not_exact"]
        )
    return paths


def _validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        contract = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_runtime_contract_invalid"]
        ) from exc
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != RUNTIME_CONTRACT_SCHEMA
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_runtime_contract_invalid"]
        )
    expected = canonical_digest(contract, digest_field="contract_digest")
    if contract.get("contract_digest") != expected:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_runtime_contract_digest_invalid"]
        )
    return contract


def _stage_assets(
    objects: list[dict[str, Any]],
    *,
    provider_asset_directory: Path,
    published_asset_directory: str | None,
) -> list[dict[str, Any]]:
    errors: list[str] = []
    if (
        not provider_asset_directory.is_dir()
        or provider_asset_directory.is_symlink()
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_asset_directory_invalid"]
        )
    rows: list[dict[str, Any]] = []
    for row in objects:
        role = str(row["semantic_role"])
        runtime_name = str(row.get("runtime_name") or role)
        candidate = provider_asset_directory / str(row["filename"])
        path = candidate.resolve()
        if provider_asset_directory != path.parent:
            errors.append(f"native_task_arena_asset_outside_directory:{role}")
            continue
        if candidate.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"native_task_arena_asset_missing:{role}")
            continue
        observed = _sha256(path)
        if observed != row["sha256"]:
            errors.append(f"native_task_arena_asset_digest_mismatch:{role}")
            continue
        rows.append(
            {
                "name": runtime_name,
                "semantic_role": role,
                "source_semantic_role": row.get("source_semantic_role", role),
                "asset_id": row.get("asset_id", role),
                "task_subject": bool(row.get("task_subject")),
                "prim_path": f"{ENV_ROOT}/{runtime_name}",
                "object_type": row["object_type"],
                "usd_path": (
                    f"{published_asset_directory}/{row['filename']}"
                    if published_asset_directory is not None
                    else str(path)
                ),
                "sha256": observed,
                "size_bytes": path.stat().st_size,
                "visible": bool(row["visible"]),
                "pose_world": row["pose_world"],
                "reset_state": row.get("reset_state") or {
                    "root_pose_world": row["pose_world"],
                    "joint_positions": {},
                },
                "activate_contact_sensors": row["object_type"]
                in {"RIGID", "ARTICULATION"},
                # The staged articulated asset must already carry the
                # probe-proven grounding (base link dynamic + world fixed
                # joint authored in the USD): PhysX refuses a kinematic link
                # inside an articulation, and Isaac Lab's fix_root_link is
                # unimplemented for these assets' topology. Verify the bytes
                # and carry the request's authored record through the plan.
                **(
                    {
                        "articulation_adaptation": _verified_articulation_row(
                            path, role=role, declared=row.get("articulation_adaptation")
                        )
                    }
                    if row["object_type"] == "ARTICULATION"
                    else {}
                ),
            }
        )
    if errors:
        raise NativeTaskArenaScenePlanError(errors)
    return rows


def _cadence(contract: Mapping[str, Any], *, physics_frequency_hz: float) -> dict[str, Any]:
    try:
        control_frequency = float(contract["task_spec"]["control_frequency_hz"])
        physics_frequency = float(physics_frequency_hz)
        ratio = physics_frequency / control_frequency
        maximum_action_steps = int(contract["task_spec"]["maximum_action_steps"])
        settle_window_samples = int(
            contract["task_spec"].get("settle_window_samples", 0)
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_control_cadence_invalid"]
        ) from exc
    rounded = round(ratio)
    if (
        not math.isfinite(control_frequency)
        or not math.isfinite(physics_frequency)
        or control_frequency <= 0.0
        or physics_frequency <= 0.0
        or rounded < 1
        or maximum_action_steps <= 0
        or settle_window_samples < 0
        or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_control_cadence_invalid"]
        )
    return {
        "control_frequency_hz": control_frequency,
        "physics_frequency_hz": physics_frequency,
        "physics_dt_seconds": 1.0 / physics_frequency,
        "control_decimation": int(rounded),
        "maximum_action_steps": maximum_action_steps,
        "settle_window_samples": settle_window_samples,
        "episode_length_seconds": (
            maximum_action_steps + settle_window_samples + 1
        )
        / control_frequency,
    }


def _yaw_quaternion_xyzw(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _quaternion_product_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _yaw_degrees_xyzw(value: list[float]) -> float:
    x, y, z, w = value
    return math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def _apply_scenario_parameters(
    *,
    objects: list[dict[str, Any]],
    cameras: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    applications: list[dict[str, Any]] = []
    subject = next(row for row in objects if row.get("task_subject") is True)
    camera_by_role = {row["role"]: row for row in cameras}
    for binding in bindings:
        target = binding["runtime_target"]
        delta = float(binding["resolved_value"]) - float(binding["nominal_value"])
        application = dict(binding)
        application["delta_from_nominal"] = delta
        if target == "EventManager.reset.object_start_position_m.y":
            base = float(subject["pose_world"]["position_world_m"][1])
            if abs(base - float(binding["nominal_value"])) > float(
                binding["application_tolerance"]
            ):
                raise NativeTaskArenaScenePlanError(
                    [f"native_task_arena_scenario_nominal_mismatch:{target}"]
                )
            subject["pose_world"]["position_world_m"][1] += delta
            subject["reset_state"]["root_pose_world"]["position_world_m"][1] += delta
            application.update(
                readback_kind="task_subject_root_position_y_m",
                expected_native_value=subject["pose_world"]["position_world_m"][1],
                runtime_name=subject["name"],
            )
        elif target == "EventManager.reset.object_orientation.yaw":
            base_yaw = _yaw_degrees_xyzw(
                subject["pose_world"]["orientation_xyzw"]
            )
            if abs(base_yaw - float(binding["nominal_value"])) > float(
                binding["application_tolerance"]
            ):
                raise NativeTaskArenaScenePlanError(
                    [f"native_task_arena_scenario_nominal_mismatch:{target}"]
                )
            yaw = _yaw_quaternion_xyzw(delta)
            effective = _quaternion_product_xyzw(
                yaw, subject["pose_world"]["orientation_xyzw"]
            )
            subject["pose_world"]["orientation_xyzw"] = effective
            subject["reset_state"]["root_pose_world"]["orientation_xyzw"] = list(
                effective
            )
            application.update(
                readback_kind="task_subject_root_orientation_xyzw",
                expected_native_value=effective,
                runtime_name=subject["name"],
            )
        elif target in {
            "EventManager.reset.external_camera.pose.position.x",
            "EventManager.reset.wrist_camera.pose.position.x",
        }:
            role = "external" if ".external_camera." in target else "wrist"
            camera = camera_by_role[role]
            base = float(camera["frame_from_camera_matrix"][3])
            if abs(base - float(binding["nominal_value"])) > float(
                binding["application_tolerance"]
            ):
                raise NativeTaskArenaScenePlanError(
                    [f"native_task_arena_scenario_nominal_mismatch:{target}"]
                )
            camera["frame_from_camera_matrix"][3] += delta
            application.update(
                readback_kind="camera_offset_position_x_m",
                expected_native_value=camera["frame_from_camera_matrix"][3],
                camera_role=role,
            )
        else:  # Contract validation owns the supported-target allowlist.
            raise NativeTaskArenaScenePlanError(
                [f"native_task_arena_scenario_target_unsupported:{target}"]
            )
        applications.append(application)
    return applications


def _graph_articulation_plan(
    contract: Mapping[str, Any],
    *,
    task_object_asset_path: Path | None,
    scene_collision_asset_path: Path | None,
) -> dict[str, Any]:
    """Compile a complete graph articulation without handle/one-link assumptions."""

    task_spec = contract["task_spec"]
    try:
        graph = validate_articulation_graph(task_spec["articulation_graph"])
    except (KeyError, ArticulationGraphContractError) as exc:
        errors = (
            list(exc.errors)
            if isinstance(exc, ArticulationGraphContractError)
            else ["native_task_arena_graph_articulation_invalid"]
        )
        raise NativeTaskArenaScenePlanError(errors) from exc
    affordance = task_spec.get("interaction_affordance")
    state_binding = contract.get("task_state_binding")
    sample_binding = contract.get("task_sample_binding")
    subject = next(
        row for row in contract["objects"] if row.get("task_subject") is True
    )
    if (
        not isinstance(affordance, Mapping)
        or affordance.get("affordance_digest")
        != canonical_digest(dict(affordance), digest_field="affordance_digest")
        or not isinstance(state_binding, Mapping)
        or state_binding.get("schema_version")
        != "native_articulated_graph_task_state_binding.v1"
        or not isinstance(sample_binding, Mapping)
        or task_object_asset_path is None
        or scene_collision_asset_path is None
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_articulation_binding_invalid"]
        )
    try:
        from pxr import Usd, UsdPhysics

        task_stage = Usd.Stage.Open(str(task_object_asset_path))
    except (ImportError, RuntimeError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_task_object_topology_unreadable"]
        ) from exc
    task_root = task_stage.GetDefaultPrim() if task_stage is not None else None
    if task_root is None or not task_root.IsValid():
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_task_object_default_prim_missing"]
        )
    gpu_collision_qualification = audit_native_task_gpu_collisions(
        task_object_asset_path
    )
    if gpu_collision_qualification["status"] != "qualified":
        raise NativeTaskArenaScenePlanError(
            list(gpu_collision_qualification["blockers"])
        )
    source_root = str(task_root.GetPath())
    source_task_body_paths = sorted(
        str(prim.GetPath())
        for prim in task_stage.Traverse()
        if prim.IsActive()
        and prim.IsLoaded()
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    contact_body_paths_raw = affordance.get("contact_body_prim_paths")
    if (
        not source_task_body_paths
        or not isinstance(contact_body_paths_raw, list)
        or not contact_body_paths_raw
        or any(str(path) not in source_task_body_paths for path in contact_body_paths_raw)
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_contact_body_paths_invalid"]
        )
    all_task_body_paths = [
        _source_to_spawned_prim(
            path,
            role="task_object",
            source_root_prim_path=source_root,
        )
        for path in source_task_body_paths
    ]
    contact_body_paths = [
        _source_to_spawned_prim(
            str(path),
            role="task_object",
            source_root_prim_path=source_root,
        )
        for path in contact_body_paths_raw
    ]
    joint_by_id = {str(row["joint_id"]): row for row in graph["joints"]}
    graph_joint_ids = set(joint_by_id)
    if set(sample_binding.get("joint_ids") or []) != graph_joint_ids:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_joint_binding_invalid"]
        )
    source_joint_paths = dict(sample_binding.get("joint_prim_paths") or {})
    native_names = dict(sample_binding.get("native_joint_names") or {})
    coordinate_ids = set(sample_binding.get("native_coordinate_joint_ids") or [])
    fixed_ids = set(sample_binding.get("fixed_joint_ids") or [])
    if (
        set(source_joint_paths) != graph_joint_ids
        or coordinate_ids.union(fixed_ids) != graph_joint_ids
        or coordinate_ids.intersection(fixed_ids)
        or set(native_names) != coordinate_ids
        or any(not task_stage.GetPrimAtPath(path).IsValid() for path in source_joint_paths.values())
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_joint_binding_invalid"]
        )
    native_reset = {
        native_names[joint_id]: float(joint_by_id[joint_id]["reset_position"])
        for joint_id in sorted(coordinate_ids)
    }
    subject_reset = dict((subject.get("reset_state") or {}).get("joint_positions") or {})
    if subject_reset != native_reset:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_joint_reset_binding_mismatch"]
        )
    link_native_body_names = dict(state_binding.get("link_native_body_names") or {})
    link_ids = {str(row["link_id"]) for row in graph["links"]}
    contact_link_id = str(affordance.get("contact_link_id") or "")
    if set(link_native_body_names) != link_ids or contact_link_id not in link_ids:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_link_body_binding_invalid"]
        )
    contact_point_link_m = affordance.get("contact_point_link_m")
    if (
        not isinstance(contact_point_link_m, list)
        or len(contact_point_link_m) != 3
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in contact_point_link_m
        )
    ):
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_graph_contact_point_invalid"]
        )
    scene_contact_body_paths = _exact_scene_contact_body_paths(
        scene_collision_asset_path
    )
    try:
        robot_contact_topology = resolve_native_task_robot_contact_topology(
            str(contract["robot"]["robot_id"])
        )
    except (KeyError, NativeTaskRobotContactTopologyError) as exc:
        errors = (
            list(exc.errors)
            if isinstance(exc, NativeTaskRobotContactTopologyError)
            else ["native_task_robot_contact_topology_unavailable"]
        )
        raise NativeTaskArenaScenePlanError(errors) from exc
    forbidden_robot_body_paths = sorted(
        set(robot_contact_topology["protected_collision_body_paths"])
        - set(robot_contact_topology["task_contact_body_paths"])
    )
    contact_sensors: list[dict[str, Any]] = []
    for index, body_path in enumerate(contact_body_paths):
        contact_sensors.append(
            {
                "sensor_instance_id": f"task_robot_contact__graph_{index:02d}",
                "logical_sensor_id": "task_robot_contact",
                "prim_path": body_path,
                "filter_prim_paths_expr": robot_contact_topology[
                    "task_contact_body_paths"
                ],
            }
        )
    for index, body_path in enumerate(all_task_body_paths):
        contact_sensors.extend(
            [
                {
                    "sensor_instance_id": f"task_scene_contact__graph_{index:02d}",
                    "logical_sensor_id": "task_scene_contact",
                    "prim_path": body_path,
                    "filter_prim_paths_expr": scene_contact_body_paths,
                },
                {
                    "sensor_instance_id": (
                        f"robot_task_forbidden_collision__graph_{index:02d}"
                    ),
                    "logical_sensor_id": "robot_task_forbidden_collision",
                    "prim_path": body_path,
                    "filter_prim_paths_expr": forbidden_robot_body_paths,
                },
            ]
        )
    contact_sensors.extend(
        {
            "sensor_instance_id": f"robot_scene_contact__{index:02d}",
            "logical_sensor_id": "robot_scene_contact",
            "prim_path": body_path,
            "filter_prim_paths_expr": scene_contact_body_paths,
        }
        for index, body_path in enumerate(
            robot_contact_topology["protected_collision_body_paths"]
        )
    )
    return {
        "graph_articulation": True,
        "task_joint_reset_positions_rad": native_reset,
        "task_joint_prim_paths": {
            joint_id: _source_to_spawned_prim(
                source_joint_paths[joint_id],
                role="task_object",
                source_root_prim_path=source_root,
            )
            for joint_id in sorted(graph_joint_ids)
        },
        "task_joint_roles": dict(sample_binding["joint_roles"]),
        "link_native_body_names": link_native_body_names,
        "interaction_link_native_body_name": link_native_body_names[contact_link_id],
        "contact_point_link_m": [float(value) for value in contact_point_link_m],
        "contact_sensors": contact_sensors,
        "robot_contact_topology": robot_contact_topology,
        "scene_contact_body_paths": scene_contact_body_paths,
        "task_contact_body_paths": contact_body_paths,
        "task_all_body_paths": all_task_body_paths,
        "forbidden_robot_contact_body_paths": forbidden_robot_body_paths,
        "gpu_collision_qualification": {
            key: value
            for key, value in gpu_collision_qualification.items()
            if key != "usd_path"
        },
        "state_thresholds": {
            field: float(state_binding[field])
            for field in (
                "task_contact_minimum_force_n",
                "collision_failure_minimum_force_n",
                "retreat_minimum_separation_m",
                "root_translation_tolerance_m",
                "root_orientation_tolerance_rad",
            )
        },
    }


def _articulation_plan(
    contract: Mapping[str, Any],
    *,
    task_object_asset_path: Path | None,
    scene_collision_asset_path: Path | None,
) -> dict[str, Any]:
    task_kind = contract["task_kind"]
    if task_kind != "articulated_open_close":
        task_spec = contract["task_spec"]
        subject = next(
            row for row in contract["objects"] if row.get("task_subject") is True
        )
        rigid_joint_resets = (
            dict((subject.get("reset_state") or {}).get("joint_positions") or {})
            if subject.get("object_type") == "ARTICULATION"
            else {}
        )
        required_thresholds = (
            "task_contact_minimum_force_n",
            "collision_failure_minimum_force_n",
            "reset_translation_tolerance_m",
            "reset_orientation_tolerance_rad",
        )
        # Preserve the original rigid packet fixture.  It predates native
        # construction contact gates and remains a compile-only compatibility
        # input; any newly executable rigid construction must bind all four
        # thresholds and receives exact contact topology below.
        if not any(field in task_spec for field in required_thresholds):
            return {
                "task_joint_reset_positions_rad": rigid_joint_resets,
                "task_joint_prim_paths": {},
                "task_joint_roles": {},
                "contact_sensors": [],
                "robot_contact_topology": None,
                "scene_contact_body_paths": [],
            }
        if any(field not in task_spec for field in required_thresholds):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_rigid_state_thresholds_incomplete"]
            )
        if task_object_asset_path is None:
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_task_object_asset_missing"]
            )
        if scene_collision_asset_path is None:
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_scene_collision_asset_missing"]
            )
        scene_contact_body_paths = _exact_scene_contact_body_paths(
            scene_collision_asset_path
        )
        try:
            robot_contact_topology = resolve_native_task_robot_contact_topology(
                str(contract["robot"]["robot_id"])
            )
        except (KeyError, NativeTaskRobotContactTopologyError) as exc:
            errors = (
                list(exc.errors)
                if isinstance(exc, NativeTaskRobotContactTopologyError)
                else ["native_task_robot_contact_topology_unavailable"]
            )
            raise NativeTaskArenaScenePlanError(errors) from exc
        affordance = task_spec.get("interaction_affordance")
        if (
            not isinstance(affordance, Mapping)
            or affordance.get("affordance_digest")
            != canonical_digest(dict(affordance), digest_field="affordance_digest")
            or not isinstance(affordance.get("allowed_contact_prim_paths"), list)
            or not affordance["allowed_contact_prim_paths"]
            or not isinstance(affordance.get("intended_support_prim_paths"), list)
            or not affordance["intended_support_prim_paths"]
        ):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_rigid_interaction_affordance_invalid"]
            )
        try:
            from pxr import Usd, UsdPhysics

            task_stage = Usd.Stage.Open(str(task_object_asset_path))
            task_default = task_stage.GetDefaultPrim() if task_stage is not None else None
        except (ImportError, RuntimeError) as exc:
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_task_object_topology_unreadable"]
            ) from exc
        if task_default is None or not task_default.IsValid():
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_task_object_default_prim_missing"]
            )
        source_root = str(task_default.GetPath())
        source_contact_paths = [
            str(path) for path in affordance["allowed_contact_prim_paths"]
        ]
        source_task_body_paths = sorted(
            str(prim.GetPath())
            for prim in task_stage.Traverse()
            if prim.IsActive()
            and prim.IsLoaded()
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if not source_task_body_paths:
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_rigid_body_topology_missing"]
            )
        # Isaac Lab contact sensors attach to rigid-body prims.  Accepting a
        # collision-only child would silently broaden the measured contact to
        # an inferred ancestor or fail during native construction.
        if any(
            path not in source_task_body_paths
            for path in source_contact_paths
        ):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_rigid_contact_body_paths_invalid"]
            )
        contact_body_paths = [
            _source_to_spawned_prim(
                str(path),
                role="task_object",
                source_root_prim_path=source_root,
            )
            for path in source_contact_paths
        ]
        if len(contact_body_paths) != len(set(contact_body_paths)):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_rigid_contact_body_paths_invalid"]
            )
        all_task_body_paths = [
            _source_to_spawned_prim(
                path,
                role="task_object",
                source_root_prim_path=source_root,
            )
            for path in source_task_body_paths
        ]
        support_stage = Usd.Stage.Open(str(scene_collision_asset_path))
        support_root = (
            support_stage.GetDefaultPrim() if support_stage is not None else None
        )
        if support_root is None or not support_root.IsValid():
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_scene_collision_default_prim_missing"]
            )
        support_body_paths = [
            _source_to_spawned_prim(
                str(path),
                role="scene_collision",
                source_root_prim_path=str(support_root.GetPath()),
            )
            for path in affordance["intended_support_prim_paths"]
        ]
        if (
            len(support_body_paths) != len(set(support_body_paths))
            or any(path not in scene_contact_body_paths for path in support_body_paths)
        ):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_rigid_support_body_paths_invalid"]
            )
        non_support_scene_body_paths = sorted(
            set(scene_contact_body_paths) - set(support_body_paths)
        )
        forbidden_robot_body_paths = sorted(
            set(robot_contact_topology["protected_collision_body_paths"])
            - set(robot_contact_topology["task_contact_body_paths"])
        )
        if not forbidden_robot_body_paths:
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_forbidden_robot_contact_topology_missing"]
            )
        contact_sensors = []
        for index, task_body_path in enumerate(contact_body_paths):
            contact_sensors.append(
                {
                    "sensor_instance_id": f"task_robot_contact__rigid_{index:02d}",
                    "logical_sensor_id": "task_robot_contact",
                    "prim_path": task_body_path,
                    "filter_prim_paths_expr": robot_contact_topology[
                        "task_contact_body_paths"
                    ],
                }
            )
        for index, task_body_path in enumerate(all_task_body_paths):
            contact_sensors.extend(
                [
                    {
                        "sensor_instance_id": f"task_support_contact__rigid_{index:02d}",
                        "logical_sensor_id": "task_support_contact",
                        "prim_path": task_body_path,
                        "filter_prim_paths_expr": support_body_paths,
                    },
                    {
                        "sensor_instance_id": (
                            f"robot_task_forbidden_collision__rigid_{index:02d}"
                        ),
                        "logical_sensor_id": "robot_task_forbidden_collision",
                        "prim_path": task_body_path,
                        "filter_prim_paths_expr": forbidden_robot_body_paths,
                    },
                ]
            )
            if non_support_scene_body_paths:
                contact_sensors.append(
                    {
                        "sensor_instance_id": (
                            f"task_scene_collision__rigid_{index:02d}"
                        ),
                        "logical_sensor_id": "task_scene_collision",
                        "prim_path": task_body_path,
                        "filter_prim_paths_expr": non_support_scene_body_paths,
                    }
                )
        contact_sensors.extend(
            {
                "sensor_instance_id": f"robot_scene_contact__{index:02d}",
                "logical_sensor_id": "robot_scene_contact",
                "prim_path": body_path,
                "filter_prim_paths_expr": scene_contact_body_paths,
            }
            for index, body_path in enumerate(
                robot_contact_topology["protected_collision_body_paths"]
            )
        )
        sample_binding = contract.get("task_sample_binding") or {}
        bound_joint_ids = list(sample_binding.get("joint_ids") or [])
        task_joint_prim_paths: dict[str, str] = {}
        task_joint_roles: dict[str, str] = {}
        if subject.get("object_type") == "ARTICULATION":
            native_names = dict(sample_binding.get("native_joint_names") or {})
            source_joint_paths = dict(sample_binding.get("joint_prim_paths") or {})
            task_joint_roles = dict(sample_binding.get("joint_roles") or {})
            if (
                not bound_joint_ids
                or set(bound_joint_ids) != set(native_names)
                or set(bound_joint_ids) != set(source_joint_paths)
                or set(bound_joint_ids) != set(task_joint_roles)
                or set(task_joint_roles.values()) != {"locked"}
                or set(rigid_joint_resets) != set(native_names.values())
            ):
                raise NativeTaskArenaScenePlanError(
                    ["native_task_arena_rigid_joint_binding_invalid"]
                )
            task_joint_prim_paths = {
                joint_id: _source_to_spawned_prim(
                    source_joint_paths[joint_id],
                    role="task_object",
                    source_root_prim_path=source_root,
                )
                for joint_id in bound_joint_ids
            }
        return {
            "task_joint_reset_positions_rad": rigid_joint_resets,
            "task_joint_prim_paths": task_joint_prim_paths,
            "task_joint_roles": task_joint_roles,
            "contact_sensors": contact_sensors,
            "robot_contact_topology": robot_contact_topology,
            "scene_contact_body_paths": scene_contact_body_paths,
            "support_contact_body_paths": support_body_paths,
            "non_support_scene_contact_body_paths": non_support_scene_body_paths,
            "task_contact_body_paths": contact_body_paths,
            "task_all_body_paths": all_task_body_paths,
            "forbidden_robot_contact_body_paths": forbidden_robot_body_paths,
            "state_thresholds": {
                field: float(task_spec[field]) for field in required_thresholds
            },
        }
    if contract["task_spec"].get("schema_version") == "adp_task_spec.v2":
        return _graph_articulation_plan(
            contract,
            task_object_asset_path=task_object_asset_path,
            scene_collision_asset_path=scene_collision_asset_path,
        )
    sample_binding = contract["task_sample_binding"]
    state_binding = contract["task_state_binding"]
    native_names = dict(sample_binding["native_joint_names"])
    scorer_reset = dict(contract["task_spec"]["joint_reset_positions_rad"])
    try:
        native_reset = {
            native_names[joint_id]: float(value)
            for joint_id, value in sorted(scorer_reset.items())
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_task_joint_reset_invalid"]
        ) from exc
    if task_object_asset_path is None:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_task_object_asset_missing"]
        )
    if scene_collision_asset_path is None:
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_scene_collision_asset_missing"]
        )
    gpu_collision_qualification = audit_native_task_gpu_collisions(
        task_object_asset_path
    )
    if gpu_collision_qualification["status"] != "qualified":
        raise NativeTaskArenaScenePlanError(
            list(gpu_collision_qualification["blockers"])
        )
    target_joint_id = str(contract["task_spec"]["target_joint_id"])
    try:
        motion_geometry = derive_native_articulated_motion_geometry(
            task_object_usd_path=task_object_asset_path,
            task_object_sha256=next(
                row["sha256"]
                for row in contract["objects"]
                if row.get("task_subject") is True
            ),
            target_joint_id=target_joint_id,
            target_joint_prim_path=sample_binding["joint_prim_paths"][
                target_joint_id
            ],
            moving_link_prim_path=state_binding["moving_link_prim_path"],
            handle_grasp_point_moving_link_m=state_binding[
                "handle_grasp_point_link_m"
            ],
            task_object_pose_world=next(
                row["pose_world"]
                for row in contract["objects"]
                if row.get("task_subject") is True
            ),
            reset_angle_rad=contract["task_spec"]["joint_reset_positions_rad"][
                target_joint_id
            ],
            scripted_target_angle_rad=contract["task_spec"][
                "scripted_positive_target_rad"
            ],
        )
    except (KeyError, StopIteration, NativeArticulatedMotionGeometryError) as exc:
        errors = (
            list(exc.errors)
            if isinstance(exc, NativeArticulatedMotionGeometryError)
            else ["native_task_arena_motion_geometry_binding_invalid"]
        )
        raise NativeTaskArenaScenePlanError(errors) from exc
    source_root = motion_geometry["source_asset_root_prim_path"]
    moving_link = _source_to_spawned_prim(
        str(state_binding["moving_link_prim_path"]),
        role="task_object",
        source_root_prim_path=source_root,
    )
    scene_contact_body_paths = _exact_scene_contact_body_paths(
        scene_collision_asset_path
    )
    try:
        robot_contact_topology = resolve_native_task_robot_contact_topology(
            str(contract["robot"]["robot_id"])
        )
    except (KeyError, NativeTaskRobotContactTopologyError) as exc:
        errors = (
            list(exc.errors)
            if isinstance(exc, NativeTaskRobotContactTopologyError)
            else ["native_task_robot_contact_topology_unavailable"]
        )
        raise NativeTaskArenaScenePlanError(errors) from exc
    contact_sensors = [
        {
            "sensor_instance_id": "task_robot_contact__moving_link",
            "logical_sensor_id": "task_robot_contact",
            "prim_path": moving_link,
            "filter_prim_paths_expr": robot_contact_topology[
                "task_contact_body_paths"
            ],
        },
        {
            "sensor_instance_id": "task_scene_contact__moving_link",
            "logical_sensor_id": "task_scene_contact",
            "prim_path": moving_link,
            "filter_prim_paths_expr": scene_contact_body_paths,
        },
    ]
    contact_sensors.extend(
        {
            "sensor_instance_id": f"robot_scene_contact__{index:02d}",
            "logical_sensor_id": "robot_scene_contact",
            "prim_path": body_path,
            "filter_prim_paths_expr": scene_contact_body_paths,
        }
        for index, body_path in enumerate(
            robot_contact_topology["protected_collision_body_paths"]
        )
    )
    return {
        "task_joint_reset_positions_rad": native_reset,
        "task_joint_prim_paths": {
            joint_id: _source_to_spawned_prim(
                path,
                role="task_object",
                source_root_prim_path=source_root,
            )
            for joint_id, path in sorted(
                sample_binding["joint_prim_paths"].items()
            )
        },
        "task_joint_roles": dict(sample_binding["joint_roles"]),
        "moving_link_native_body_name": state_binding[
            "moving_link_native_body_name"
        ],
        "handle_prim_paths": [
            _source_to_spawned_prim(
                path,
                role="task_object",
                source_root_prim_path=source_root,
            )
            for path in state_binding["handle_prim_paths"]
        ],
        "handle_grasp_point_link_m": state_binding["handle_grasp_point_link_m"],
        "motion_geometry": motion_geometry,
        "contact_sensors": contact_sensors,
        "robot_contact_topology": robot_contact_topology,
        "scene_contact_body_paths": scene_contact_body_paths,
        "gpu_collision_qualification": {
            key: value
            for key, value in gpu_collision_qualification.items()
            if key != "usd_path"
        },
        "state_thresholds": {
            key: state_binding[key]
            for key in (
                "task_contact_minimum_force_n",
                "collision_failure_minimum_force_n",
                "retreat_minimum_separation_m",
                "root_translation_tolerance_m",
                "root_orientation_tolerance_rad",
            )
        },
    }


def materialize_native_task_arena_scene_plan(
    *,
    runtime_contract: Mapping[str, Any],
    provider_asset_directory: str | Path,
    physics_frequency_hz: float,
    published_asset_directory: str | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Verify staged bytes and freeze one deterministic native scene plan."""

    contract = _validate_contract(runtime_contract)
    raw_asset_directory = Path(provider_asset_directory).expanduser()
    if raw_asset_directory.is_symlink():
        raise NativeTaskArenaScenePlanError(
            ["native_task_arena_asset_directory_invalid"]
        )
    asset_directory = raw_asset_directory.resolve()
    if published_asset_directory is not None:
        pure = PurePosixPath(published_asset_directory)
        if (
            not published_asset_directory
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.name in {"", ".", ".."}
        ):
            raise NativeTaskArenaScenePlanError(
                ["native_task_arena_published_asset_directory_invalid"]
            )
        published_asset_directory = pure.as_posix().rstrip("/")
    objects = _stage_assets(
        list(contract["objects"]),
        provider_asset_directory=asset_directory,
        published_asset_directory=published_asset_directory,
    )
    cameras = json.loads(json.dumps(contract["cameras"]))
    scenario_parameter_applications = _apply_scenario_parameters(
        objects=objects,
        cameras=cameras,
        bindings=list(contract["scenario"].get("parameter_bindings") or []),
    )
    effective_contract = json.loads(json.dumps(contract))
    staged_by_asset_id = {row["asset_id"]: row for row in objects}
    for row in effective_contract["objects"]:
        staged = staged_by_asset_id[row["asset_id"]]
        row["pose_world"] = staged["pose_world"]
        row["reset_state"] = staged["reset_state"]
    task_object_asset_path = next(
        (
            asset_directory / str(row["filename"])
            for row in contract["objects"]
            if row.get("task_subject") is True
        ),
        None,
    )
    scene_collision_asset_path = next(
        (
            asset_directory / str(row["filename"])
            for row in contract["objects"]
            if row["semantic_role"] == "scene_collision"
        ),
        None,
    )
    articulation = _articulation_plan(
        effective_contract,
        task_object_asset_path=task_object_asset_path,
        scene_collision_asset_path=scene_collision_asset_path,
    )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runtime_contract_digest": contract["contract_digest"],
        "scene_id": contract["scene_id"],
        "task_id": contract["task_id"],
        "task_kind": contract["task_kind"],
        "task_freeze_digest": contract.get("task_freeze_digest"),
        "construction_bindings": contract.get("construction_bindings"),
        "task_spec": contract["task_spec"],
        "task_sample_binding": contract["task_sample_binding"],
        "task_state_binding": contract["task_state_binding"],
        "scenario": {
            **contract["scenario"],
            "parameter_applications": scenario_parameter_applications,
        },
        "asset_directory": published_asset_directory or str(asset_directory),
        "objects": objects,
        "robot": contract["robot"],
        "cameras": cameras,
        "cadence": _cadence(contract, physics_frequency_hz=physics_frequency_hz),
        "articulation": articulation,
        "reset": {
            "robot_joint_positions_rad": contract["robot"][
                "joint_reset_positions_rad"
            ],
            "task_joint_positions_rad": articulation[
                "task_joint_reset_positions_rad"
            ],
            "root_pose_reset_required": True,
            "native_readback_required_after_reset": True,
        },
        "application_readback_required": contract["runtime_readback_required"],
        "claim_boundary": {
            "plan_is_not_native_application_proof": True,
            "plan_is_not_policy_episode_evidence": True,
            "simulator_execution_is_not_physical_truth": True,
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    if destination is not None:
        write_json(Path(destination), plan)
    return json.loads(json.dumps(plan))


__all__ = [
    "NativeTaskArenaScenePlanError",
    "SCHEMA_VERSION",
    "materialize_native_task_arena_scene_plan",
]
