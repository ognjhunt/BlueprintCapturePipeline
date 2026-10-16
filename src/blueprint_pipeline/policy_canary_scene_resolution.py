"""Resolve one policy-canary cell's scene plan from the frozen base plan.

Beta journey step: L5 team policy evaluation (and the DROID canary it shares).
Pure: applies a cell's sealed success contract and scenario parameters (object
start shift and yaw, external camera shift, light, friction) to a copy of the
base plan and records each application the GPU host must read back. Split out
of ``native_task_arena_policy_canary_worker`` so that grandfathered module
stays inside its line budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


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


def resolved_scene_plan(
    base: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    task_success_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = deepcopy(dict(base))
    aim = cell.get("operator_wrist_camera_aim")
    if aim is not None:
        if not isinstance(aim, Mapping) or aim.get("mode") != "point_at_task_object_then_rigidly_follow_wrist" or not str(aim.get("authorized_by") or "").strip() or aim.get("authority_digest") != canonical_digest(aim, digest_field="authority_digest"):
            raise RuntimeError("operator_wrist_camera_aim_authority_invalid")
        plan["operator_wrist_camera_aim"] = dict(aim)
    task_success_contract = task_success_contract or cell.get(
        "task_success_contract"
    )
    if (
        not isinstance(task_success_contract, Mapping)
        or (
            cell.get("task_success_contract_digest")
            or task_success_contract.get("contract_digest")
        )
        != task_success_contract.get("contract_digest")
    ):
        raise RuntimeError("policy_canary_task_success_contract_binding_invalid")
    plan["task_spec"]["task_success_contract"] = deepcopy(
        dict(task_success_contract)
    )
    control = cell.get("control_diagnostic") or {}
    if control.get("mode") == "nonblocking_omitted_by_user":
        from blueprint_pipeline.native_task_arena_policy_canary_session import (
            validate_control_omission_authority,
        )
        from blueprint_pipeline.native_task_camera_start_configuration import (
            validate_camera_start_configuration,
        )
        omission = validate_control_omission_authority(control.get("omission_authority") or {},
            contract_digest=task_success_contract["contract_digest"])
        camera_start = omission.get("policy_canary_camera_start_configuration")
        if camera_start is not None:
            plan["policy_canary_camera_start_configuration"] = validate_camera_start_configuration(plan, camera_start)
    plan["task_spec"]["task_success_contract_digest"] = task_success_contract["contract_digest"]
    scenario = deepcopy(dict(cell["resolved_scenario"]))
    scenario["cell_id"] = cell["cell_id"]
    scenario["seed"] = cell["seed"]
    if cell.get("resolved_scenario_digest") is not None:
        scenario["resolved_scenario_digest"] = cell["resolved_scenario_digest"]
    parameters = dict(scenario.get("parameters") or {})
    applications: list[dict[str, Any]] = []
    coverage_gaps: list[dict[str, Any]] = []
    subject = next(row for row in plan["objects"] if row.get("task_subject") is True)
    # A scored team evaluation shifts the start on both table axes; the
    # published Quick-10 cells shift only y.
    for axis, parameter_id in ((0, "object_start_x_delta_m"), (1, "object_start_y_delta_m")):
        if parameter_id not in parameters:
            continue
        delta = float(parameters[parameter_id])
        subject["pose_world"]["position_world_m"][axis] += delta
        subject["reset_state"]["root_pose_world"]["position_world_m"][axis] += delta
        if plan["task_kind"] == "rigid_pick_place":
            scoring_start_pose = plan["task_spec"].get("start_pose_world")
            if not isinstance(scoring_start_pose, list) or len(scoring_start_pose) != 7:
                raise RuntimeError("policy_canary_task_scoring_start_pose_invalid")
            scoring_start_pose[axis] += delta
        elif "start_pose_world" in plan["task_spec"]:
            raise RuntimeError("policy_canary_articulated_scoring_pose_unexpected")
        applications.append(
            {
                "parameter_id": parameter_id,
                "readback_kind": f"task_subject_root_position_{'xy'[axis]}_m",
                "expected_native_value": subject["pose_world"]["position_world_m"][axis],
                "delta_from_nominal": delta,
            }
        )
    if "object_yaw_delta_degrees" in parameters:
        delta = float(parameters["object_yaw_delta_degrees"])
        orientation = _quaternion_product_xyzw(
            _yaw_quaternion_xyzw(delta),
            list(subject["pose_world"]["orientation_xyzw"]),
        )
        subject["pose_world"]["orientation_xyzw"] = orientation
        subject["reset_state"]["root_pose_world"]["orientation_xyzw"] = list(
            orientation
        )
        if plan["task_kind"] == "rigid_pick_place":
            scoring_start_pose = plan["task_spec"].get("start_pose_world")
            if not isinstance(scoring_start_pose, list) or len(scoring_start_pose) != 7:
                raise RuntimeError("policy_canary_task_scoring_start_pose_invalid")
            scoring_start_pose[3:] = list(orientation)
        elif "start_pose_world" in plan["task_spec"]:
            raise RuntimeError("policy_canary_articulated_scoring_pose_unexpected")
        applications.append(
            {
                "parameter_id": "object_yaw_delta_degrees",
                "readback_kind": "task_subject_root_orientation_xyzw",
                "expected_native_value": orientation,
                "delta_from_nominal": delta,
            }
        )
    if "external_camera_x_delta_m" in parameters:
        delta = float(parameters["external_camera_x_delta_m"])
        camera = next(row for row in plan["cameras"] if row["role"] == "external")
        # The DROID profile below preserves its official sensor even without
        # a custom start binding. Generic scene-camera coordinates are not
        # that sensor's parent-frame coordinates.
        from blueprint_pipeline.native_task_camera_start_configuration import (
            external_camera_offset_position,
        )
        camera["frame_from_camera_matrix"][3] += delta
        expected_camera_x = external_camera_offset_position({**plan, "scenario": scenario})[0]
        applications.append(
            {
                "parameter_id": "external_camera_x_delta_m",
                "readback_kind": "camera_offset_position_x_m",
                "camera_role": "external",
                "expected_native_value": expected_camera_x,
                "delta_from_nominal": delta,
            }
        )
    if "task_light_intensity_scale" in parameters:
        scale = float(parameters["task_light_intensity_scale"])
        applications.append(
            {
                "parameter_id": "task_light_intensity_scale",
                "readback_kind": "task_light_intensity_scale",
                "expected_native_value": scale,
                "nominal_native_intensity": 1500.0,
                "application_tolerance": 1.0e-6,
            }
        )
    from blueprint_pipeline.native_rigid_friction_scenario import (
        add_application as add_rigid_friction,
    )
    add_rigid_friction(parameters, subject, applications, coverage_gaps)
    if "material_cousin" in parameters:
        coverage_gaps.append(
            {
                "family": "admitted_object_material_cousin",
                "reason": "admitted_runtime_material_asset_unavailable",
                "fallback": "canonical_task_material",
            }
        )
    scenario["parameter_applications"] = applications
    for application in applications:
        key = application["parameter_id"]
        application.setdefault("runtime_name", subject["name"])
        application.setdefault("runtime_target", application["readback_kind"])
        application.setdefault("unit", "degrees" if "degrees" in key else "m" if key.endswith("_m") else "ratio")
        application.setdefault("resolved_value", parameters[key])
        # No tolerance is invented for an unqualified diagnostic dimension.
        application.setdefault("application_tolerance", 0.0)
    scenario["runtime_coverage_gaps"] = coverage_gaps
    plan["scenario"] = scenario
    # Both frozen DROID adapters emit actions at 15 Hz. Keep PhysX at 120 Hz
    # and change the exact canary cadence to an integral decimation of eight.
    plan["cadence"]["control_frequency_hz"] = 15.0
    plan["cadence"]["control_decimation"] = 8
    action_steps = int(plan["cadence"]["maximum_action_steps"])
    settle_samples = int(plan["cadence"]["settle_window_samples"])
    plan["task_spec"]["control_frequency_hz"] = 15.0
    plan["task_spec"]["maximum_episode_seconds"] = action_steps / 15.0
    plan["cadence"]["episode_length_seconds"] = (
        action_steps / 15.0
        + settle_samples / 15.0
        + 6.0 * float(plan["cadence"]["physics_dt_seconds"])
    )
    plan["canary_cadence_adjustment"] = {
        "source_control_frequency_hz": float(base["cadence"]["control_frequency_hz"]),
        "resolved_control_frequency_hz": 15.0,
        "physics_frequency_hz": float(plan["cadence"]["physics_frequency_hz"]),
        "control_decimation": 8,
        "reason": "frozen_droid_policy_action_cadence",
    }
    from blueprint_pipeline.droid_policy_canary_embodiment import (
        apply_droid_policy_canary_profile,
    )

    return apply_droid_policy_canary_profile(plan)


__all__ = ["resolved_scene_plan"]
