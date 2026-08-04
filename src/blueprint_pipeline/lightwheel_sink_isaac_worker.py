"""Isaac 6 worker for the Lightwheel sink articulation and Franka contact canary."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .lightwheel_sink_isaac_bundle import BUNDLE_SCHEMA_VERSION, RUNTIME_RESULT_SCHEMA_VERSION
from .measurement_isaac_runtime_release import ISAAC_VERSION, RUNTIME_IMAGE


def _environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"lightwheel_sink_environment_missing:{name}")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("lightwheel_sink_json_object_required")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_vector(values: Sequence[Any]) -> bool:
    try:
        return bool(values) and all(math.isfinite(float(item)) for item in values)
    except (TypeError, ValueError):
        return False


def _phase(name: str, *, started: float, **detail: Any) -> None:
    """Emit secret-free progress evidence to the provider container log."""

    print(
        "BLUEPRINT_LIGHTWHEEL_SINK_PHASE:"
        + json.dumps(
            {
                "name": name,
                "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
                **detail,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def handle_tangent(handle_center: Sequence[float], pivot: Sequence[float]) -> list[float]:
    """Return the +rotation tangent for a revolute joint whose axis is world X."""

    if len(handle_center) != 3 or len(pivot) != 3:
        raise ValueError("lightwheel_sink_handle_geometry_invalid")
    radial_y = float(handle_center[1]) - float(pivot[1])
    radial_z = float(handle_center[2]) - float(pivot[2])
    tangent = [0.0, -radial_z, radial_y]
    norm = math.sqrt(sum(item * item for item in tangent))
    if norm <= 1e-9 or not math.isfinite(norm):
        raise ValueError("lightwheel_sink_handle_tangent_degenerate")
    return [item / norm for item in tangent]


def damped_least_squares_delta(
    jacobian: Sequence[Sequence[float]],
    position_error: Sequence[float],
    *,
    damping: float = 0.08,
    max_norm: float = 0.035,
) -> list[float]:
    """Compute a bounded differential-IK step using J^T(JJ^T+lambda^2 I)^-1 e."""

    import numpy as np

    matrix = np.asarray(jacobian, dtype=float)
    error = np.asarray(position_error, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 3 or error.shape != (3,):
        raise ValueError("lightwheel_sink_differential_ik_shape_invalid")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(error)):
        raise ValueError("lightwheel_sink_differential_ik_nonfinite")
    regularizer = float(damping) ** 2 * np.eye(3)
    delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + regularizer, error)
    norm = float(np.linalg.norm(delta))
    if norm > float(max_norm):
        delta *= float(max_norm) / norm
    return [float(item) for item in delta]


def _simulation_app_type() -> Any:  # pragma: no cover - Isaac runtime only
    try:
        from isaacsim import SimulationApp

        return SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp

        return SimulationApp


def _single_articulation_type() -> Any:  # pragma: no cover - Isaac runtime only
    try:
        from isaacsim.core.prims import SingleArticulation

        return SingleArticulation
    except Exception:
        from omni.isaac.core.articulations import Articulation as SingleArticulation

        return SingleArticulation


def _articulation_action_type() -> Any:  # pragma: no cover - Isaac runtime only
    try:
        from isaacsim.core.utils.types import ArticulationAction

        return ArticulationAction
    except Exception:
        from omni.isaac.core.utils.types import ArticulationAction

        return ArticulationAction


def _apply_targets(articulation: Any, positions: Any, indices: Any) -> None:  # pragma: no cover
    Action = _articulation_action_type()
    action = Action(joint_positions=positions, joint_indices=indices)
    apply = getattr(articulation, "apply_action", None)
    if callable(apply):
        apply(action)
    else:
        articulation.get_articulation_controller().apply_action(action)


def _joint_values(articulation: Any, index: int) -> tuple[float, float, float | None]:  # pragma: no cover
    import numpy as np

    position = float(np.asarray(articulation.get_joint_positions()).reshape(-1)[index])
    velocity = float(np.asarray(articulation.get_joint_velocities()).reshape(-1)[index])
    effort = None
    for name in ("get_measured_joint_efforts", "get_applied_joint_efforts"):
        getter = getattr(articulation, name, None)
        if callable(getter):
            try:
                effort = float(np.asarray(getter()).reshape(-1)[index])
                break
            except Exception:
                continue
    return position, velocity, effort


def _world_position(stage: Any, path: str, UsdGeom: Any) -> list[float]:  # pragma: no cover
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    value = matrix.ExtractTranslation()
    return [float(value[index]) for index in range(3)]


def _impulse_magnitude(value: Any) -> float:  # pragma: no cover - Isaac runtime only
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        try:
            return math.sqrt(
                sum(float(getattr(value, axis, 0.0)) ** 2 for axis in ("x", "y", "z"))
            )
        except (TypeError, ValueError):
            return 0.0


def _subscribe_contact_reports(contact_log: list) -> Any:  # pragma: no cover - Isaac runtime only
    """Collect PhysX contact headers via the supported subscription callback.

    Polling get_contact_report() outside a simulation callback crashed the app
    natively on the first capsule step (instances 46754204 and 46755516)."""

    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    def _on_contact_report(contact_headers: Any, contact_data: Any) -> None:
        try:
            data = list(contact_data)
            for header in contact_headers:
                paths: list[str] = []
                for name in ("actor0", "actor1", "collider0", "collider1"):
                    encoded = getattr(header, name, 0)
                    try:
                        paths.append(str(PhysicsSchemaTools.intToSdfPath(int(encoded))))
                    except Exception:  # noqa: BLE001 - path decode is best effort
                        paths.append(str(encoded))
                offset = int(getattr(header, "contact_data_offset", 0) or 0)
                count = int(getattr(header, "num_contact_data", 0) or 0)
                samples = [
                    {
                        "impulse": _impulse_magnitude(getattr(sample, "impulse", 0.0)),
                        "separation_m": float(getattr(sample, "separation", 0.0) or 0.0),
                    }
                    for sample in data[offset : offset + min(count, 4)]
                ]
                if len(contact_log) < 4000:
                    contact_log.append({"paths": paths, "samples": samples})
        except Exception:  # noqa: BLE001 - a reporting fault must never kill physics
            pass

    return get_physx_simulation_interface().subscribe_contact_report_events(_on_contact_report)


def _between_contacts(
    records: Sequence[Mapping[str, Any]], first_prefix: str, second_prefix: str
) -> list[dict[str, Any]]:
    result = []
    for row in records:
        paths = [str(item) for item in row.get("paths", [])]
        first = any(
            path == first_prefix or path.startswith(first_prefix + "/") for path in paths
        )
        second = any(
            path == second_prefix or path.startswith(second_prefix + "/") for path in paths
        )
        if first and second:
            result.append(dict(row))
    return result


def _camera_quaternion(forward: Any, up: Any) -> Any:  # pragma: no cover
    import numpy as np

    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f)
    right = np.cross(f, np.asarray(up, dtype=float))
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, f)
    rotation = np.column_stack((right, corrected_up, -f))
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        nxt, final = (index + 1) % 3, (index + 2) % 3
        scale = math.sqrt(1.0 + rotation[index, index] - rotation[nxt, nxt] - rotation[final, final]) * 2.0
        xyz = np.zeros(3)
        xyz[index] = 0.25 * scale
        xyz[nxt] = (rotation[nxt, index] + rotation[index, nxt]) / scale
        xyz[final] = (rotation[final, index] + rotation[index, final]) / scale
        quaternion = np.concatenate(([(rotation[final, nxt] - rotation[nxt, final]) / scale], xyz))
    return quaternion / np.linalg.norm(quaternion)


def _frame_record(camera: Any, label: str) -> dict[str, Any]:  # pragma: no cover
    import io
    import numpy as np
    from PIL import Image

    rgba = np.asarray(camera.get_rgba())
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise RuntimeError(f"lightwheel_sink_camera_frame_invalid:{label}:{rgba.shape}")
    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    payload = output.getvalue()
    return {
        "label": label,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "rgb_mean": float(np.mean(rgb)),
        "rgb_spatial_std": float(np.std(rgb)),
        "nonzero_value_count": int(np.count_nonzero(rgb)),
        "png_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "png_base64": base64.b64encode(payload).decode("ascii"),
    }


def _runtime(
    bundle_root: Path,
    manifest: Mapping[str, Any],
    persist: Any,
) -> dict[str, Any]:  # pragma: no cover - Isaac runtime only
    import numpy as np

    started = time.monotonic()
    deadline_seconds = float(
        dict(manifest["test_configuration"]).get("runtime_deadline_seconds") or 780.0
    )

    def checkpoint(name: str, **detail: Any) -> None:
        _phase(name, started=started, **detail)
        elapsed = time.monotonic() - started
        if elapsed > deadline_seconds:
            raise RuntimeError(
                f"lightwheel_sink_runtime_deadline_exceeded:{name}:{elapsed:.0f}s"
            )

    _phase("simulation_app_launch_start", started=started)
    SimulationApp = _simulation_app_type()
    config = dict(manifest["test_configuration"])
    width, height = map(int, config["render_resolution"])
    simulation_app = SimulationApp(
        {"headless": True, "renderer": config["renderer"], "width": width, "height": height}
    )
    checkpoint("simulation_app_launch_complete")
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        world = World(stage_units_in_meters=1.0, physics_dt=float(config["physics_dt_seconds"]), rendering_dt=float(config["physics_dt_seconds"]))
        checkpoint("world_created")
        stage = world.stage
        add_reference_to_stage(str(bundle_root / manifest["wrapper_path"]), "/World")
        checkpoint("sink_wrapper_referenced")
        sink_path = str(config["sink_prim_path"])
        sink_prim = stage.GetPrimAtPath(sink_path)
        if not sink_prim.IsValid():
            raise RuntimeError("lightwheel_sink_wrapper_reference_not_composed")
        xform = UsdGeom.Xformable(sink_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, config["sink_translation_world_m"])))

        # The generated asset requests convexDecomposition on five dense meshes;
        # VHACD cooking on those stalls PhysX parse for tens of minutes on CPU.
        # This canary only needs a lever test, so the session layer downgrades
        # every sink collider to a convex hull. The source layer stays untouched.
        collider_overrides: list[dict[str, str]] = []
        for prim in Usd.PrimRange(sink_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):
                continue
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            approximation = mesh_collision.GetApproximationAttr()
            collider_overrides.append(
                {
                    "path": str(prim.GetPath()),
                    "source_approximation": str(approximation.Get() or "none"),
                    "session_approximation": "convexHull",
                }
            )
            approximation.Set("convexHull")
        checkpoint("collider_approximation_overridden", collider_count=len(collider_overrides))

        joint_prim = stage.GetPrimAtPath(str(config["handle_joint_path"]))
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(8.0)
        drive.CreateDampingAttr().Set(0.35)
        drive.CreateMaxForceAttr().Set(3.0)

        for root_path in (sink_path, str(config["handle_prim_path"])):
            root = stage.GetPrimAtPath(root_path)
            api = PhysxSchema.PhysxContactReportAPI.Apply(root)
            api.CreateThresholdAttr().Set(0.0)

        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(900.0)
        key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        key.CreateIntensityAttr(2200.0)

        pivot = np.asarray(config["handle_pivot_source_world_m"], dtype=float) + np.asarray(config["sink_translation_world_m"], dtype=float)
        center = np.asarray(config["handle_center_source_world_m"], dtype=float) + np.asarray(config["sink_translation_world_m"], dtype=float)
        tangent = np.asarray(handle_tangent(center, pivot), dtype=float)

        camera_specs = {
            "front": (center + np.asarray([0.48, -0.58, 0.30]), center),
            "side": (center + np.asarray([-0.40, 0.56, 0.22]), center),
            "close": (center + np.asarray([0.30, -0.30, 0.12]), center),
        }
        cameras: dict[str, Any] = {}
        for name, (eye, target) in camera_specs.items():
            path = f"/World/Cameras/{name}"
            prim = UsdGeom.Camera.Define(stage, path)
            prim.CreateVerticalApertureAttr(15.2908)
            prim.CreateHorizontalApertureAttr(15.2908 * width / height)
            prim.CreateFocalLengthAttr(18.0)
            camera_xform = UsdGeom.Xformable(prim.GetPrim())
            camera_xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, eye)))
            q = _camera_quaternion(target - eye, (0.0, 0.0, 1.0))
            camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
            cameras[name] = Camera(prim_path=path, resolution=(width, height))

        # Everything joins the stage before the single physics init: a second
        # world.reset() mid-session made Kit process a quit on the next rendered
        # step (attempt 46754204 exited 0 right after capsule_world_reset_complete).
        pusher_path = "/World/TestTools/KinematicCapsule"
        capsule = UsdGeom.Capsule.Define(stage, pusher_path)
        capsule.CreateRadiusAttr(0.018)
        capsule.CreateHeightAttr(0.10)
        capsule.CreateAxisAttr("X")
        capsule.CreateDisplayColorAttr([(0.9, 0.15, 0.05)])
        capsule_xform = UsdGeom.Xformable(capsule.GetPrim())
        translate_op = capsule_xform.AddTranslateOp()
        capsule_park_m = center + tangent * -0.30
        translate_op.Set(Gf.Vec3d(*map(float, capsule_park_m)))
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
        rigid.CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(capsule.GetPrim()).CreateMassAttr().Set(0.2)
        contact_api = PhysxSchema.PhysxContactReportAPI.Apply(capsule.GetPrim())
        contact_api.CreateThresholdAttr().Set(0.0)

        assets_root = get_assets_root_path() or ""
        if not assets_root:
            raise RuntimeError("lightwheel_sink_isaac_assets_root_missing")
        franka_asset = assets_root.rstrip("/") + str(config["franka_asset_relative_path"])
        checkpoint("franka_asset_reference_start", asset=franka_asset)
        add_reference_to_stage(franka_asset, str(config["franka_prim_path"]))
        checkpoint("franka_asset_referenced")
        # Origin-based placement intersects the sink's counter volume and puts
        # the handle at the edge of reach; stand the base on the +y approach side.
        franka_base_translation = [
            float(item)
            for item in (config.get("franka_base_translation_world_m") or [0.35, 0.30, 0.0])
        ]
        franka_xformable = UsdGeom.Xformable(stage.GetPrimAtPath(str(config["franka_prim_path"])))
        franka_translate_op = next(
            (
                op
                for op in franka_xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        if franka_translate_op is None:
            franka_translate_op = franka_xformable.AddTranslateOp()
        franka_translate_op.Set(Gf.Vec3d(*franka_base_translation))
        checkpoint("franka_base_translated", translation=franka_base_translation)
        hand_path = str(config["franka_prim_path"]) + "/panda_hand"
        if not stage.GetPrimAtPath(hand_path).IsValid():
            children = [
                str(prim.GetPath())
                for prim in stage.GetPrimAtPath(str(config["franka_prim_path"])).GetChildren()
            ]
            raise RuntimeError(f"lightwheel_sink_franka_hand_prim_missing:{children}")
        for root_path in (str(config["franka_prim_path"]), hand_path):
            api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(root_path))
            api.CreateThresholdAttr().Set(0.0)

        Articulation = _single_articulation_type()
        sink = Articulation(prim_path=sink_path, name="lightwheel_sink")
        world.scene.add(sink)
        franka = Articulation(prim_path=str(config["franka_prim_path"]), name="lightwheel_sink_franka")
        world.scene.add(franka)
        checkpoint("world_reset_start")
        world.reset()
        checkpoint("world_reset_complete")
        contact_log: list[dict[str, Any]] = []
        contact_subscription = _subscribe_contact_reports(contact_log)

        def _drain_contacts(prefixes: Sequence[str]) -> list[dict[str, Any]]:
            drained = [
                row
                for row in contact_log
                if any(
                    any(path == prefix or path.startswith(prefix + "/") for path in row["paths"])
                    for prefix in prefixes
                )
            ]
            del contact_log[:]
            return drained

        for camera in cameras.values():
            camera.initialize()
        for _ in range(8):
            world.step(render=True)
        checkpoint("render_warmup_complete")
        initial_joints = np.asarray([0.2897, 0.50732, -0.140016, -2.176, -0.0310497, 2.51592, -0.49251, 0.04, 0.04])
        franka.set_joint_positions(initial_joints)
        for _ in range(int(config["settle_steps"])):
            world.step(render=False)
        checkpoint("franka_initial_pose_settled")
        dof_names = list(getattr(sink, "dof_names", []) or [])
        joint_name = str(config["handle_joint_path"]).rsplit("/", 1)[-1]
        if joint_name not in dof_names:
            raise RuntimeError(f"lightwheel_sink_handle_dof_missing:{dof_names}")
        handle_index = dof_names.index(joint_name)
        root_initial = np.asarray(_world_position(stage, str(config["base_prim_path"]), UsdGeom))

        frames = [_frame_record(cameras["front"], "asset_initial_front"), _frame_record(cameras["side"], "asset_initial_side")]
        checkpoint("initial_frames_captured", frame_count=len(frames))
        sweep_trace: list[dict[str, Any]] = []
        for target_deg in config["handle_joint_targets_degrees"]:
            _apply_targets(sink, np.asarray([math.radians(float(target_deg))]), np.asarray([handle_index]))
            for step in range(int(config["target_steps"])):
                world.step(render=step == int(config["target_steps"]) - 1)
            position, velocity, effort = _joint_values(sink, handle_index)
            root_now = np.asarray(_world_position(stage, str(config["base_prim_path"]), UsdGeom))
            sweep_trace.append(
                {
                    "target_degrees": float(target_deg),
                    "actual_degrees": math.degrees(position),
                    "velocity_rad_s": velocity,
                    "effort_nm": effort,
                    "root_displacement_m": float(np.linalg.norm(root_now - root_initial)),
                }
            )
            checkpoint(
                "sweep_target_measured",
                target_degrees=float(target_deg),
                actual_degrees=round(math.degrees(position), 3),
            )
        frames.extend([_frame_record(cameras["front"], "asset_sweep_120_front"), _frame_record(cameras["close"], "asset_sweep_120_close")])
        checkpoint("joint_target_sweep_complete", target_count=len(sweep_trace))

        # Restore a passive handle, then sweep the pre-built kinematic capsule
        # through it tangentially. No reset: the capsule teleports from its
        # parking spot into the approach position.
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.02)
        sink.set_joint_positions(np.asarray([0.0]), joint_indices=np.asarray([handle_index]))
        for _ in range(30):
            world.step(render=False)
        checkpoint("capsule_push_start")
        pusher_trace: list[dict[str, Any]] = []
        pusher_contacts: list[dict[str, Any]] = []
        for step in range(int(config["pusher_steps"])):
            alpha = step / max(1, int(config["pusher_steps"]) - 1)
            position_m = center + tangent * (-0.11 + 0.22 * alpha)
            translate_op.Set(Gf.Vec3d(*map(float, position_m)))
            world.step(render=step == int(config["pusher_steps"]) - 1)
            if step % 10 == 0 or step == int(config["pusher_steps"]) - 1:
                angle, velocity, effort = _joint_values(sink, handle_index)
                records = _between_contacts(
                    _drain_contacts([pusher_path, str(config["handle_prim_path"])]),
                    pusher_path,
                    str(config["handle_prim_path"]),
                )
                pusher_contacts.extend(records)
                pusher_trace.append(
                    {"step": step, "capsule_position_m": position_m.tolist(), "handle_degrees": math.degrees(angle), "handle_velocity_rad_s": velocity, "handle_effort_nm": effort, "contact_count": len(records)}
                )
        frames.append(_frame_record(cameras["close"], "capsule_push_close"))
        checkpoint("capsule_push_complete", contact_count=len(pusher_contacts))
        # Park the capsule out of the Franka's workspace (teleport, not
        # SetActive: deactivating a body mid-session repopulates the scene).
        translate_op.Set(Gf.Vec3d(*map(float, capsule_park_m)))

        # Scripted differential-IK Franka push against a re-zeroed handle.
        sink.set_joint_positions(np.asarray([0.0]), joint_indices=np.asarray([handle_index]))
        for _ in range(30):
            world.step(render=False)
        checkpoint("franka_push_start")
        franka_initial_handle_angle = math.degrees(_joint_values(sink, handle_index)[0])
        franka_dof_names = list(getattr(franka, "dof_names", []) or [])
        franka_fixed_joint_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(stage.GetPrimAtPath(str(config["franka_prim_path"])))
            if prim.IsA(UsdPhysics.FixedJoint)
        ]
        franka_base_path = str(config["franka_prim_path"]) + "/panda_link0"
        franka_base_initial = np.asarray(_world_position(stage, franka_base_path, UsdGeom))
        arm_indices = np.arange(min(7, len(franka_dof_names)), dtype=int)
        body_names = list(getattr(franka, "body_names", []) or getattr(getattr(franka, "_articulation_view", None), "body_names", []) or [])
        if "panda_hand" not in body_names:
            raise RuntimeError(f"lightwheel_sink_franka_hand_body_missing:{body_names}")
        body_index = body_names.index("panda_hand")
        waypoints = [center - tangent * 0.13, center - tangent * 0.025, center + tangent * 0.09]
        franka_trace: list[dict[str, Any]] = []
        franka_contacts: list[dict[str, Any]] = []
        for waypoint_index, waypoint in enumerate(waypoints):
            for step in range(int(config["ik_steps_per_waypoint"])):
                current = np.asarray(_world_position(stage, hand_path, UsdGeom))
                error = waypoint - current
                jacobians = np.asarray(franka._articulation_view.get_jacobians())
                if jacobians.ndim == 4:
                    jacobians = jacobians[0]
                jacobian_index = body_index - 1 if jacobians.shape[0] == len(body_names) - 1 else body_index
                positional = jacobians[jacobian_index, :3, arm_indices]
                delta = np.asarray(damped_least_squares_delta(positional, error, damping=0.08, max_norm=0.025))
                joints = np.asarray(franka.get_joint_positions()).reshape(-1)
                commanded = joints[arm_indices] + delta
                _apply_targets(franka, commanded, arm_indices)
                world.step(render=step % 30 == 0)
                if step % 15 == 0 or step == int(config["ik_steps_per_waypoint"]) - 1:
                    sink_angle, _, sink_effort = _joint_values(sink, handle_index)
                    contacts = _between_contacts(
                        _drain_contacts(
                            [
                                str(config["franka_prim_path"]),
                                str(config["handle_prim_path"]),
                            ]
                        ),
                        str(config["franka_prim_path"]),
                        str(config["handle_prim_path"]),
                    )
                    franka_contacts.extend(contacts)
                    efforts_getter = getattr(franka, "get_measured_joint_efforts", None)
                    efforts = np.asarray(efforts_getter()).reshape(-1).tolist() if callable(efforts_getter) else []
                    franka_trace.append(
                        {
                            "waypoint_index": waypoint_index,
                            "step": step,
                            "target_m": waypoint.tolist(),
                            "hand_position_m": current.tolist(),
                            "position_error_m": float(np.linalg.norm(error)),
                            "joint_positions_rad": joints.tolist(),
                            "joint_efforts_nm": efforts,
                            "handle_degrees": math.degrees(sink_angle),
                            "handle_effort_nm": sink_effort,
                            "contact_count": len(contacts),
                        }
                    )
            checkpoint(
                "franka_waypoint_complete",
                waypoint_index=waypoint_index,
                handle_degrees=round(math.degrees(_joint_values(sink, handle_index)[0]), 3),
            )
        frames.extend([_frame_record(cameras["front"], "franka_push_front"), _frame_record(cameras["close"], "franka_push_close")])
        checkpoint("franka_push_complete", contact_count=len(franka_contacts))
        try:
            contact_subscription.unsubscribe()
        except Exception:  # noqa: BLE001 - teardown of the reporter is best effort
            pass

        root_final = np.asarray(_world_position(stage, str(config["base_prim_path"]), UsdGeom))
        franka_base_final = np.asarray(_world_position(stage, franka_base_path, UsdGeom))
        final_angle, final_velocity, final_effort = _joint_values(sink, handle_index)
        all_contacts = pusher_contacts + franka_contacts
        separations = [float(sample["separation_m"]) for row in all_contacts for sample in row.get("samples", [])]
        texture_inputs = []
        omni_pbr_shader_count = 0
        for prim in Usd.PrimRange(sink_prim):
            if prim.IsA(UsdShade.Shader):
                source_asset = UsdShade.Shader(prim).GetSourceAsset("mdl")
                if source_asset and "OmniPBR.mdl" in str(source_asset):
                    omni_pbr_shader_count += 1
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    value = attribute.Get()
                    if value:
                        texture_inputs.append(str(value))
        result = {
            "wrapper": {
                "physics_scene_present": stage.GetPrimAtPath("/World/physicsScene").IsValid(),
                "articulation_root_api_present": sink_prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                "fixed_root_joint_present": stage.GetPrimAtPath("/World/Sink/BlueprintFixedRoot").IsValid(),
                "collider_session_overrides": collider_overrides,
                "source_asset_modified": False,
            },
            "asset": {
                "sink_dof_names": dof_names,
                "handle_joint_limits_degrees": [float(joint.GetLowerLimitAttr().Get()), float(joint.GetUpperLimitAttr().Get())],
                "handle_pivot_world_m": pivot.tolist(),
                "handle_center_world_m": center.tolist(),
                "handle_tangent_world": tangent.tolist(),
                "sweep_trace": sweep_trace,
                "capsule_push_trace": pusher_trace,
                "capsule_contact_records": pusher_contacts,
            },
            "franka": {
                "asset_uri": franka_asset,
                "base_translation_world_m": franka_base_translation,
                "fixed_base_requested": True,
                "fixed_joint_paths": franka_fixed_joint_paths,
                "base_displacement_m": float(
                    np.linalg.norm(franka_base_final - franka_base_initial)
                ),
                "controller": "scripted_damped_least_squares_differential_ik",
                "dof_names": franka_dof_names,
                "body_names": body_names,
                "trace": franka_trace,
                "contact_records": franka_contacts,
            },
            "measurements": {
                "final_handle_degrees": math.degrees(final_angle),
                "final_handle_velocity_rad_s": final_velocity,
                "final_handle_effort_nm": final_effort,
                "franka_initial_handle_degrees": franka_initial_handle_angle,
                "franka_handle_motion_degrees": abs(
                    math.degrees(final_angle) - franka_initial_handle_angle
                ),
                "capsule_handle_motion_degrees": (
                    max(float(row["handle_degrees"]) for row in pusher_trace)
                    - min(float(row["handle_degrees"]) for row in pusher_trace)
                    if pusher_trace
                    else 0.0
                ),
                "root_displacement_m": float(np.linalg.norm(root_final - root_initial)),
                "maximum_penetration_m": max([0.0, *[-value for value in separations if value < 0.0]]),
                "numerical_state_finite": _finite_vector([math.degrees(final_angle), final_velocity, *root_final.tolist()]),
                "pusher_contact_count": len(pusher_contacts),
                "franka_handle_contact_count": len(franka_contacts),
            },
            "rendering": {
                "renderer": config["renderer"],
                "omnipbr_shader_count": omni_pbr_shader_count,
                "material_asset_inputs": sorted(set(texture_inputs)),
                "frames": frames,
            },
        }
        _phase("runtime_result_ready", started=started, frame_count=len(frames))
        persist(result, [])
        return result
    except (Exception, SystemExit) as exc:
        # SimulationApp.close() in the finally below terminates the whole
        # process (attempts 46754204/46755181 exited 0 there), so the terminal
        # result must be on disk before this frame unwinds.
        persist({}, [f"lightwheel_sink_runtime_failed:{type(exc).__name__}:{str(exc)[:400]}"])
        raise
    finally:
        simulation_app.close()


def run_lightwheel_sink_canary(bundle_root: Path, output_path: Path) -> int:
    """Validate immutable inputs, execute Isaac, and always emit a terminal JSON result."""

    process_started = time.monotonic()
    _phase("bundle_validation_start", started=process_started)
    manifest = _read_object(bundle_root / "bundle_manifest.json")
    blockers: list[str] = []
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        blockers.append("lightwheel_sink_bundle_manifest_schema_invalid")
    if manifest.get("bundle_manifest_digest") != canonical_digest(manifest, digest_field="bundle_manifest_digest"):
        blockers.append("lightwheel_sink_bundle_manifest_digest_mismatch")
    expected = {
        "source_commit_sha": _environment("BLUEPRINT_LIGHTWHEEL_SINK_SOURCE_COMMIT"),
        "runtime_image_digest": _environment("BLUEPRINT_LIGHTWHEEL_SINK_RUNTIME_IMAGE"),
        "runtime_release_digest": _environment("BLUEPRINT_LIGHTWHEEL_SINK_RUNTIME_RELEASE_DIGEST"),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            blockers.append(f"lightwheel_sink_bundle_{key}_mismatch")
    if manifest.get("runtime_image_digest") != RUNTIME_IMAGE or manifest.get("isaac_sim_version") != ISAAC_VERSION:
        blockers.append("lightwheel_sink_runtime_identity_mismatch")
    asset_before: dict[str, str] = {}
    for record in manifest.get("asset_files", []):
        path = bundle_root / str(record.get("path") or "")
        if not path.is_file() or path.is_symlink():
            blockers.append("lightwheel_sink_asset_file_missing_or_unsafe")
            continue
        digest = _sha256(path)
        asset_before[str(record["path"])] = digest
        if digest != record.get("digest"):
            blockers.append("lightwheel_sink_asset_digest_mismatch")
    wrapper = bundle_root / str(manifest.get("wrapper_path") or "")
    if not wrapper.is_file() or _sha256(wrapper) != manifest.get("wrapper_digest"):
        blockers.append("lightwheel_sink_wrapper_digest_mismatch")
    if not blockers:
        finalized: dict[str, int] = {}

        def _persist(runtime_value: Mapping[str, Any], runtime_blockers: Sequence[str]) -> None:
            finalized["exit_code"] = _finalize_result(
                bundle_root=bundle_root,
                manifest=manifest,
                expected=expected,
                asset_before=asset_before,
                runtime=dict(runtime_value),
                blockers=[*blockers, *runtime_blockers],
                output_path=output_path,
                process_started=process_started,
            )

        try:
            _phase("bundle_validation_complete", started=process_started)
            _runtime(bundle_root, manifest, _persist)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - terminal evidence must survive runtime failure, including Isaac's sys.exit on startup rejection
            if "exit_code" not in finalized:
                blockers.append(
                    f"lightwheel_sink_runtime_failed:{type(exc).__name__}:{str(exc)[:400]}"
                )
        if "exit_code" in finalized:
            return finalized["exit_code"]
    return _finalize_result(
        bundle_root=bundle_root,
        manifest=manifest,
        expected=expected,
        asset_before=asset_before,
        runtime={},
        blockers=blockers,
        output_path=output_path,
        process_started=process_started,
    )


def _finalize_result(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, str],
    asset_before: Mapping[str, str],
    runtime: dict[str, Any],
    blockers: Sequence[str],
    output_path: Path,
    process_started: float,
) -> int:
    """Assemble, gate, and write the terminal result; must run before app close."""

    blockers = list(blockers)
    asset_before = dict(asset_before)
    asset_after = {
        str(record["path"]): _sha256(bundle_root / str(record["path"]))
        for record in manifest.get("asset_files", [])
        if (bundle_root / str(record.get("path") or "")).is_file()
    }
    if asset_before != asset_after:
        blockers.append("lightwheel_sink_source_asset_changed_during_test")
    measurements = runtime.get("measurements") if isinstance(runtime, Mapping) else {}
    rendering = runtime.get("rendering") if isinstance(runtime, Mapping) else {}
    frames = rendering.get("frames") if isinstance(rendering, Mapping) else []
    frame_valid = bool(frames) and all(
        isinstance(row, Mapping)
        and row.get("width") == 640
        and row.get("height") == 480
        and float(row.get("rgb_spatial_std") or 0.0) >= 2.0
        and int(row.get("nonzero_value_count") or 0) > 0
        for row in frames
    )
    gates = {
        "source_digests_preserved": asset_before == asset_after and bool(asset_before),
        "wrapper_composed": runtime.get("wrapper", {}).get("physics_scene_present") is True
        and runtime.get("wrapper", {}).get("articulation_root_api_present") is True
        and runtime.get("wrapper", {}).get("fixed_root_joint_present") is True,
        "asset_sweep_completed": len(runtime.get("asset", {}).get("sweep_trace", [])) == 5,
        "asset_sweep_targets_reached": bool(runtime.get("asset", {}).get("sweep_trace"))
        and all(
            abs(float(row["actual_degrees"]) - float(row["target_degrees"])) <= 5.0
            for row in runtime["asset"]["sweep_trace"]
        ),
        "root_anchor_stable": isinstance(measurements.get("root_displacement_m"), (int, float))
        and float(measurements["root_displacement_m"]) <= 0.003,
        "limit_behavior_stable": bool(runtime.get("asset", {}).get("sweep_trace"))
        and max(float(row["actual_degrees"]) for row in runtime["asset"]["sweep_trace"]) <= 123.0,
        "capsule_contact_observed": int(measurements.get("pusher_contact_count") or 0) > 0,
        "capsule_handle_motion_observed": float(
            measurements.get("capsule_handle_motion_degrees") or 0.0
        )
        >= 1.0,
        "franka_contact_observed": int(measurements.get("franka_handle_contact_count") or 0) > 0,
        "franka_handle_motion_observed": float(
            measurements.get("franka_handle_motion_degrees") or 0.0
        )
        >= 1.0,
        "franka_fixed_base_verified": isinstance(
            runtime.get("franka", {}).get("base_displacement_m"), (int, float)
        )
        and float(runtime["franka"]["base_displacement_m"]) <= 0.003,
        "joint_effort_readback_available": measurements.get("final_handle_effort_nm")
        is not None
        and all(
            row.get("effort_nm") is not None
            for row in runtime.get("asset", {}).get("sweep_trace", [])
        ),
        "numerical_state_finite": measurements.get("numerical_state_finite") is True,
        "omnipbr_materials_present": int(rendering.get("omnipbr_shader_count") or 0) > 0,
        "texture_bindings_present": len(rendering.get("material_asset_inputs") or []) >= 4,
        "rgb_frames_valid": frame_valid,
    }
    for name, passed in gates.items():
        if not passed:
            blockers.append(f"lightwheel_sink_gate_failed:{name}")
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed" if not blockers else "failed",
        "source_commit_sha": expected["source_commit_sha"],
        "runtime_image_digest": expected["runtime_image_digest"],
        "runtime_release_digest": expected["runtime_release_digest"],
        "input_bundle_digest": _environment("BLUEPRINT_LIGHTWHEEL_SINK_INPUT_BUNDLE_DIGEST"),
        "bundle_manifest_digest": manifest.get("bundle_manifest_digest"),
        "source_model_digest": manifest.get("source_model_digest"),
        "texture_manifest_digest": manifest.get("texture_manifest", {}).get("texture_manifest_digest"),
        "wrapper_digest": manifest.get("wrapper_digest"),
        "test_configuration_digest": manifest.get("test_configuration", {}).get("test_configuration_digest"),
        "isaac_sim_version": ISAAC_VERSION,
        "asset_digests_before": asset_before,
        "asset_digests_after": asset_after,
        "runtime": runtime,
        "gates": gates,
        "blockers": sorted(set(blockers)),
        "development_only": True,
        "external_generated_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "physical_success_established": False,
        "production_route_eligible": False,
        "qualification_created": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "development_execution_only" if runtime else "none",
        "claim_ceiling": "isaac_articulation_and_scripted_franka_contact_development" if runtime else "immutable_input_validation_only",
    }
    result["runtime_result_digest"] = canonical_digest(result, digest_field="runtime_result_digest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _phase(
        "terminal_result_written",
        started=process_started,
        status=result["status"],
        blocker_count=len(result["blockers"]),
    )
    return 0 if result["status"] == "passed" else 1


__all__ = ["damped_least_squares_delta", "handle_tangent", "run_lightwheel_sink_canary"]
