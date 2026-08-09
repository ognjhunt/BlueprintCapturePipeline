"""Provider-side blank-stage Isaac diagnostic for a bound articulated USD."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any


def _pose_row(pose: Any) -> dict[str, list[float]]:
    return {
        "position": [float(pose.p.x), float(pose.p.y), float(pose.p.z)],
        "rotation_xyzw": [
            float(pose.r.x),
            float(pose.r.y),
            float(pose.r.z),
            float(pose.r.w),
        ],
    }


def _translation_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    return math.dist(first["position"], second["position"])


def _quaternion_distance_degrees(first: dict[str, Any], second: dict[str, Any]) -> float:
    a = first["rotation_xyzw"]
    b = second["rotation_xyzw"]
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b, strict=True))))
    return math.degrees(2.0 * math.acos(dot))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    asset = Path(args.asset).resolve()
    request_path = Path(args.request).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "adp009d_articulated_native_diagnostic.v1",
        "status": "blocked",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "blockers": [],
    }
    app = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp

        app = SimulationApp(
            {"headless": True, "enable_cameras": False, "renderer": "RayTracedLighting"}
        )
        import carb
        import omni.timeline
        import omni.usd
        from omni.isaac.dynamic_control import _dynamic_control
        from pxr import UsdPhysics

        context = omni.usd.get_context()
        context.open_stage(str(asset))
        for _ in range(30):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("articulated_native_stage_open_failed")
        articulation_spec = request["articulation"]
        runtime = request["runtime"]
        fixed_prim = stage.GetPrimAtPath(
            articulation_spec["fixed_base_body_prim_path"]
        )
        fixed_api = UsdPhysics.RigidBodyAPI(fixed_prim)
        fixed_api.CreateKinematicEnabledAttr(True).Set(True)
        for joint_path in [
            articulation_spec["driven_joint_prim_path"],
            *articulation_spec["locked_joint_prim_paths"],
        ]:
            joint = stage.GetPrimAtPath(joint_path)
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(runtime["drive_stiffness"]))
            drive.CreateDampingAttr(float(runtime["drive_damping"]))
            drive.CreateMaxForceAttr(float(runtime["drive_max_force"]))
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
        scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(60):
            app.update()

        dc = _dynamic_control.acquire_dynamic_control_interface()
        articulation = dc.get_articulation(
            articulation_spec["root_prim_path"]
        )
        if articulation == _dynamic_control.INVALID_HANDLE:
            raise RuntimeError("articulated_native_articulation_handle_invalid")
        dc.wake_up_articulation(articulation)
        dof_count = int(dc.get_articulation_dof_count(articulation))
        body_count = int(dc.get_articulation_body_count(articulation))
        names = []
        handles: dict[str, Any] = {}
        for index in range(dof_count):
            handle = dc.get_articulation_dof(articulation, index)
            name = str(dc.get_dof_name(handle))
            names.append(name)
            handles[name] = handle
        expected_names = {
            Path(articulation_spec["driven_joint_prim_path"]).name,
            *(Path(path).name for path in articulation_spec["locked_joint_prim_paths"]),
        }
        if not expected_names.issubset(handles):
            raise RuntimeError(
                "articulated_native_dof_name_mismatch:"
                + ",".join(sorted(set(expected_names) - set(handles)))
            )
        if dof_count != int(articulation_spec["expected_joint_count"]):
            raise RuntimeError(
                f"articulated_native_dof_count_mismatch:{dof_count}"
            )
        driven_name = Path(articulation_spec["driven_joint_prim_path"]).name
        locked_names = [
            Path(path).name for path in articulation_spec["locked_joint_prim_paths"]
        ]
        driven_handle = handles[driven_name]
        fixed_body = dc.find_articulation_body(
            articulation, Path(articulation_spec["fixed_base_body_prim_path"]).name
        )
        if fixed_body == _dynamic_control.INVALID_HANDLE:
            raise RuntimeError("articulated_native_fixed_body_handle_invalid")
        initial_fixed_pose = _pose_row(dc.get_rigid_body_pose(fixed_body))
        rows = []
        settle_steps = int(runtime["settle_steps_per_command"])
        for angle in articulation_spec["commanded_angles_degrees"]:
            for name in locked_names:
                dc.set_dof_position_target(handles[name], 0.0)
                dc.set_dof_velocity_target(handles[name], 0.0)
            dc.set_dof_position_target(driven_handle, math.radians(float(angle)))
            dc.set_dof_velocity_target(driven_handle, 0.0)
            dc.wake_up_articulation(articulation)
            for _ in range(settle_steps):
                app.update()
            readback = math.degrees(float(dc.get_dof_position(driven_handle)))
            velocity = float(dc.get_dof_velocity(driven_handle))
            locked_readback = {
                name: math.degrees(float(dc.get_dof_position(handles[name])))
                for name in locked_names
            }
            rows.append(
                {
                    "commanded_angle_degrees": float(angle),
                    "readback_angle_degrees": readback,
                    "absolute_error_degrees": abs(readback - float(angle)),
                    "velocity_rad_s_after_settle": velocity,
                    "locked_joint_readback_degrees": locked_readback,
                }
            )
        # Reset replay is a native operation, not a caller assertion.
        dc.set_dof_position_target(driven_handle, 0.0)
        dc.set_dof_velocity_target(driven_handle, 0.0)
        for name in locked_names:
            dc.set_dof_position_target(handles[name], 0.0)
            dc.set_dof_velocity_target(handles[name], 0.0)
        for _ in range(settle_steps):
            app.update()
        reset_readback = math.degrees(float(dc.get_dof_position(driven_handle)))
        final_fixed_pose = _pose_row(dc.get_rigid_body_pose(fixed_body))
        translation_drift = _translation_distance(initial_fixed_pose, final_fixed_pose)
        rotation_drift = _quaternion_distance_degrees(
            initial_fixed_pose, final_fixed_pose
        )
        errors = []
        if any(
            row["absolute_error_degrees"]
            > float(runtime["joint_readback_tolerance_degrees"])
            for row in rows
        ):
            errors.append("articulated_native_joint_readback_out_of_tolerance")
        if any(
            abs(value) > float(runtime["locked_joint_tolerance_degrees"])
            for row in rows
            for value in row["locked_joint_readback_degrees"].values()
        ):
            errors.append("articulated_native_locked_joint_moved")
        if any(
            abs(row["velocity_rad_s_after_settle"])
            > float(runtime["maximum_abs_joint_velocity_rad_s_after_settle"])
            for row in rows
        ):
            errors.append("articulated_native_joint_not_settled")
        if abs(reset_readback) > float(runtime["joint_readback_tolerance_degrees"]):
            errors.append("articulated_native_reset_replay_failed")
        if translation_drift > float(runtime["fixed_base_translation_tolerance_m"]):
            errors.append("articulated_native_fixed_base_translated")
        if rotation_drift > float(runtime["fixed_base_rotation_tolerance_degrees"]):
            errors.append("articulated_native_fixed_base_rotated")
        result.update(
            {
                "status": "completed" if not errors else "blocked",
                "blockers": errors,
                "native_runtime": {
                    "isaac_sim_version": str(carb.settings.get_settings().get("/app/version") or "unknown"),
                    "dynamic_control_api": True,
                },
                "articulation_readback": {
                    "articulation_root_prim_path": articulation_spec["root_prim_path"],
                    "dof_count": dof_count,
                    "body_count": body_count,
                    "dof_names": names,
                    "rows": rows,
                    "reset_readback_degrees": reset_readback,
                },
                "fixed_base_readback": {
                    "initial_pose": initial_fixed_pose,
                    "final_pose": final_fixed_pose,
                    "translation_drift_m": translation_drift,
                    "rotation_drift_degrees": rotation_drift,
                },
                "contact_sensor_status": {
                    "available": False,
                    "gap": "blank_stage_joint_diagnostic_does_not_establish_task_contact_or_initial_penetration",
                },
                "claim_ceiling": "native_isaac_articulation_import_drive_and_reset_only",
            }
        )
        timeline.stop()
    except Exception as exc:
        result["blockers"].append(str(exc))
        result["traceback"] = traceback.format_exc()
    finally:
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if app is not None:
            app.close()
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - provider runtime only
    raise SystemExit(main(sys.argv[1:]))
