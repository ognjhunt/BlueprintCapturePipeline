"""One registered non-colliding region on an existing support (ADP-009D).

The same immutable region drives marker placement and whole-object scoring.
It supplies no destination asset or native qualification claim.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import product

from .decision_evidence_contracts import canonical_digest

SCHEMA = "task_evaluation_surface_target.v1"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def validate_surface_target(value):
    fields = {"schema_version", "shape", "non_colliding", "visible_label", "radius_m",
              "surface_position_world_m", "support_prim_path", "support_source_instance_id",
              "maximum_tilt_rad", "stable_seconds", "maximum_linear_speed_m_s",
              "maximum_angular_speed_rad_s", "target_digest"}
    require(isinstance(value, Mapping) and set(value) == fields,
            "surface_target_invalid")
    require(value["schema_version"] == SCHEMA and value["shape"] == "flat_green_disc"
            and value["non_colliding"] is True and bool(value["visible_label"])
            and isinstance(value["support_prim_path"], str)
            and value["support_prim_path"].startswith("/")
            and bool(value["support_source_instance_id"]), "surface_target_invalid")
    position = value["surface_position_world_m"]
    require(isinstance(position, list) and len(position) == 3
            and all(type(x) in (int, float) and math.isfinite(x) for x in position),
            "surface_target_position_invalid")
    for key in ("radius_m", "maximum_tilt_rad", "stable_seconds",
                "maximum_linear_speed_m_s", "maximum_angular_speed_rad_s"):
        require(type(value[key]) in (int, float) and math.isfinite(value[key])
                and value[key] > 0, "surface_target_threshold_invalid")
    require(value["radius_m"] <= 0.5 and value["maximum_tilt_rad"] < math.pi / 2
            and value["target_digest"] == canonical_digest(value, digest_field="target_digest"),
            "surface_target_digest_invalid")
    return dict(value)


def derive_surface_target(*, destination, support, source_min, source_max, support_instance_id):
    require(destination.get("kind") == "green_region" and destination.get("relation") == "on"
            and destination.get("orientation_xyzw") == [0, 0, 0, 1],
            "surface_target_destination_invalid")
    value = {"schema_version": SCHEMA, "shape": "flat_green_disc", "non_colliding": True,
             "visible_label": destination["visible_label"], "radius_m": destination["radius_m"],
             "surface_position_world_m": list(destination["position_world_m"]),
             "support_prim_path": support["sage_prim_path"],
             "support_source_instance_id": str(support_instance_id),
             "maximum_tilt_rad": math.radians(15), "stable_seconds": 1.0,
             "maximum_linear_speed_m_s": 0.02, "maximum_angular_speed_rad_s": 0.1}
    overrides = destination.get("success", {})
    require(isinstance(overrides, Mapping) and set(overrides) <= {
        "maximum_tilt_rad", "stable_seconds", "maximum_linear_speed_m_s", "maximum_angular_speed_rad_s"},
        "surface_target_success_fields_invalid")
    value.update(overrides)
    value["target_digest"] = canonical_digest(value, digest_field="target_digest")
    validate_surface_target(value)
    p, radius = value["surface_position_world_m"], value["radius_m"]
    require(abs(p[2] - support["bounds_max_xyz_m"][2]) <= 0.005,
            "surface_target_support_height_mismatch")
    require(all(support["bounds_min_xyz_m"][i] <= p[i] - radius
                and p[i] + radius <= support["bounds_max_xyz_m"][i] for i in (0, 1)),
            "surface_target_outside_support")
    nearest = [max(source_min[i], min(source_max[i], p[i])) for i in (0, 1)]
    require(math.dist(nearest, p[:2]) > radius, "surface_target_initial_overlap")
    require(math.hypot(*[(source_max[i] - source_min[i]) / 2 for i in (0, 1)]) < radius,
            "surface_target_subject_does_not_fit")
    return value


def marker_for_surface_target(value):
    value = validate_surface_target(value)
    return {"schema_version": "native_task_target_marker.v1",
            **{key: value[key] for key in ("shape", "non_colliding", "radius_m",
                                           "surface_position_world_m")}}


def surface_execution_limits(*, success, support):
    """Freeze a bounded fixed-arm safety profile; values remain proposals.

    These defaults are runtime constraints, not measured physics or reachability.
    Tasks may supply tighter or alternative explicit limits before execution;
    native qualification still has to establish an executable trajectory.
    """
    limits = {
        "retreat_clearance_m": 0.05, "drop_minimum_fall_m": 0.02,
        "maximum_task_contact_force_n": 100.0, "collision_failure_minimum_force_n": 1.0,
        "forbidden_contact_classes": ["robot_background", "robot_object", "object_background"],
        "robot_workspace_position_bounds_world_m": {
            "minimum": [support["bounds_min_xyz_m"][0] - 1.0, support["bounds_min_xyz_m"][1] - 1.0,
                        support["bounds_min_xyz_m"][2] - 1.0],
            "maximum": [support["bounds_max_xyz_m"][0] + 1.0, support["bounds_max_xyz_m"][1] + 1.0,
                        support["top_z_m"] + 1.5]},
    }
    limits.update(success)
    return limits


def bind_native_surface_target(*, task_spec, target, static, support):
    """Bind qualified collider bounds and the retained support before sealing."""
    from .adp_rigid_retreat_scoring import _rotate, _vector
    target = validate_surface_target(target)
    require(target["support_prim_path"] == support["sage_prim_path"]
            and abs(target["surface_position_world_m"][2] - support["top_z_m"]) <= 0.005,
            "surface_target_native_support_mismatch")
    value = dict(task_spec)
    affordance = dict(value["interaction_affordance"])
    transform = affordance["asset_root_from_scoring_frame"]
    offset = _vector(transform.get("position_m"), 3)
    q = _vector(transform.get("orientation_xyzw"), 4)
    bounds = static["observed_structure"]["collision_bounds_body_frame_m"]
    lower, upper = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
    require(offset is not None and q is not None and lower is not None and upper is not None
            and math.isclose(sum(x*x for x in q), 1, abs_tol=1e-6)
            and all(a < b for a, b in zip(lower, upper, strict=True)), "surface_target_native_geometry_invalid")
    corners = [_rotate([c[i] - offset[i] for i in range(3)], [-q[0], -q[1], -q[2], q[3]])
               for c in product(*zip(lower, upper, strict=True))]
    radius = target["radius_m"]
    require(max(math.hypot(*c[:2]) for c in corners) < radius,
            "surface_target_native_subject_does_not_fit")
    p = target["surface_position_world_m"]
    goal = [p[0], p[1], p[2] - min(c[2] for c in corners)]
    tolerance = value["destination_position_tolerance_m"]
    value.update(surface_target=target, visible_target_marker=marker_for_surface_target(target),
        target_position_world_m=goal, destination_pose_world=[*p, 0., 0., 0., 1.],
        subject_collision_bounds_scoring_frame_m={
            "minimum": [min(c[i] for c in corners) for i in range(3)],
            "maximum": [max(c[i] for c in corners) for i in range(3)]},
        destination_position_bounds_world_m={"minimum": [goal[0]-tolerance, goal[1]-tolerance, goal[2]-0.005],
                                             "maximum": [goal[0]+tolerance, goal[1]+tolerance, goal[2]+0.005]},
        support_height_interval_m=[goal[2]-0.005, goal[2]+0.005],
        settle_window_samples=math.ceil(target["stable_seconds"]*value["control_frequency_hz"])+1,
        settle_position_tolerance_m=target["maximum_linear_speed_m_s"]*target["stable_seconds"],
        settle_orientation_tolerance_rad=target["maximum_angular_speed_rad_s"]*target["stable_seconds"],
        destination_orientation_tolerance_rad=target["maximum_tilt_rad"])
    affordance.update(intended_support_prim_paths=[target["support_prim_path"]],
                      insertion_withdrawal_unit_world=[0., 0., 1.])
    affordance["affordance_digest"] = canonical_digest(affordance, digest_field="affordance_digest")
    value["interaction_affordance"] = affordance
    return value


def score_surface_target(*, target, bounds, samples, frequency_hz, minimum_lift_m):
    """Use retained native poses, contact, gripper state and elapsed simulation time.

    Differencing actual poses gives independent velocity evidence, including the
    transition into the stable window. Every rotated collider-bound corner must
    fit the disk. Missing/nonfinite readback is invalid, never a successful pose.
    """
    from .adp_rigid_retreat_scoring import _rotate, _vector

    target = validate_surface_target(target)
    lower, upper = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
    require(lower is not None and upper is not None
            and all(a < b for a, b in zip(lower, upper, strict=True)),
            "surface_target_native_bounds_invalid")
    corners = list(product(*zip(lower, upper, strict=True)))
    count = math.ceil(target["stable_seconds"] * frequency_hz) + 1
    result = {"readback_complete": True, "whole_footprint_contained": False,
              "upright": False, "stable_velocity": False, "lifted_clear": False,
              "initially_outside": False, "supported_and_released": False, "marker_pose_matches": False}
    poses = []
    for row in samples:
        pose = _vector(row.get("task_object_pose_world"), 7)
        if (pose is None or not math.isclose(sum(x*x for x in pose[3:]), 1, abs_tol=1e-5)
                or type(row.get("step_index")) is not int):
            return {**result, "readback_complete": False, "satisfied": False}
        norm = math.sqrt(sum(x*x for x in pose[3:]))
        pose[3:] = [x/norm for x in pose[3:]]
        world = [[pose[i] + v[i] for i in range(3)] for v in (_rotate(c, pose[3:]) for c in corners)]
        poses.append((row, pose, world))
    if len(poses) < count:
        return {**result, "readback_complete": False, "satisfied": False}
    center, radius = target["surface_position_world_m"], target["radius_m"]
    # The native renderer places the two-millimetre-thick disc one millimetre
    # above the support. Its observed transform must still describe this region.
    expected_marker = [center[0], center[1], center[2]+0.001, 0., 0., 0., 1.]
    marker_poses = [_vector(row.get("destination_pose_world"), 7) for row, _, _ in poses]
    if any(p is None or math.dist(p[:3], expected_marker[:3]) > 2e-6
           or min(math.dist(p[3:], expected_marker[3:]), math.dist(p[3:], [-x for x in expected_marker[3:]])) > 2e-6
           for p in marker_poses):
        return {**result, "readback_complete": False, "satisfied": False}
    result["marker_pose_matches"] = True
    # AABB is a conservative native collider envelope, so full footprint
    # containment cannot be manufactured by a center-point-only test.
    initial = poses[0][2]
    nearest = [max(min(p[i] for p in initial), min(max(p[i] for p in initial), center[i])) for i in (0, 1)]
    result["initially_outside"] = math.dist(nearest, center[:2]) > radius
    result["lifted_clear"] = any(min(p[2] for p in world) >= center[2] + minimum_lift_m
                                  and row.get("task_contact_active") is True for row, _, world in poses)
    window = poses[-count:]
    result["whole_footprint_contained"] = all(math.dist(p[:2], center[:2]) <= radius + 1e-12
                                               for _, _, world in window for p in world)
    result["upright"] = all(_rotate([0, 0, 1], pose[3:])[2] >= math.cos(target["maximum_tilt_rad"])
                            for _, pose, _ in window)
    # Native support contact and released gripper are checked by the parent
    # contract; repeat contact requirements here across the exact one-second window.
    result["supported_and_released"] = all(row.get("support_contact_active") is True
        and row.get("task_contact_active") is False for row, _, _ in window)
    result["readback_complete"] = all(type(row.get("support_contact_active")) is bool
        and type(row.get("task_contact_active")) is bool for row, _, _ in window)
    speeds_ok = True
    for (a, p, _), (b, q, _) in zip(window, window[1:]):
        steps = b["step_index"] - a["step_index"]
        if steps != 1:
            result["readback_complete"] = False
            speeds_ok = False
            continue
        dt = steps / frequency_hz
        rotation = 2 * math.acos(min(1.0, abs(sum(x*y for x, y in zip(p[3:], q[3:], strict=True)))))
        speeds_ok &= (math.dist(p[:3], q[:3]) / dt <= target["maximum_linear_speed_m_s"] + 1e-12
                      and rotation / dt <= target["maximum_angular_speed_rad_s"] + 1e-12)
    result["stable_velocity"] = speeds_ok
    result["satisfied"] = all(result.values())
    return result
