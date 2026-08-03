"""Bounded native-Isaac Franka control experiment in the pinned NVIDIA workcell.

The module deliberately has no MuJoCo or synthetic fallback.  Its injected
backend seam exists only for hermetic contract tests; the CLI always selects
the Isaac Sim 6/PhysX backend.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import write_json
from .nvidia_warehouse_native_camera_canary import (
    _apply_runtime_asset_relocations,
    _articulation_link_world_pose_matrices,
    _backend_array_to_numpy,
    _camera_quaternion_wxyz,
    _camera_sensor_annotator_frame,
    _json_safe_evidence,
    _load_materialization_manifest,
    _rigid_wrist_mount_from_initial_task_framing,
    _simulation_app_launch_config,
    _synchronize_camera_to_rigid_link,
    _unified_world_pose_matrix,
    import_simulation_app,
)
from .nvidia_warehouse_workcell import DATASET_REVISION
from .policy_ranking_thesis import canonical_sha256, file_sha256


SPEC_SCHEMA_VERSION = "nvidia_warehouse_native_control_spec.v1"
RESULT_SCHEMA_VERSION = "nvidia_warehouse_native_control_result.v1"
DECISION_SCHEMA_VERSION = "decision_envelope.v1"
CLAIM_LABEL = (
    "NVIDIA-authored SimReady control scene; native Isaac physics; scripted "
    "controllers; single workcell; simulation-only."
)
CONTROLLER_COUNT = 5


def _validated_spec(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError("nvidia_warehouse_native_control_spec_invalid")
    payload = dict(value)
    declared = payload.pop("spec_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("nvidia_warehouse_native_control_spec_sha256_invalid")
    if payload.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("nvidia_warehouse_native_control_revision_invalid")
    controllers = payload.get("controllers")
    if not isinstance(controllers, list) or len(controllers) != CONTROLLER_COUNT:
        raise ValueError("nvidia_warehouse_native_control_controller_count_invalid")
    ids = [row.get("controller_id") for row in controllers if isinstance(row, Mapping)]
    if len(ids) != CONTROLLER_COUNT or len(set(ids)) != CONTROLLER_COUNT:
        raise ValueError("nvidia_warehouse_native_control_controller_ids_invalid")
    payload["spec_sha256"] = declared
    return payload


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.isfinite(array).all():
        return None
    return array


def _quaternion_angle_rad(left: Sequence[float], right: Sequence[float]) -> float:
    left_value = np.asarray(left, dtype=float)
    right_value = np.asarray(right, dtype=float)
    left_value /= max(float(np.linalg.norm(left_value)), 1e-12)
    right_value /= max(float(np.linalg.norm(right_value)), 1e-12)
    return float(2.0 * math.acos(min(1.0, abs(float(np.dot(left_value, right_value))))))


def rank_controller_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically rank frozen controllers without breaking exact ties."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "controller_id": str(row.get("controller_id") or ""),
                "success": row.get("success") is True,
                "partial_progress": round(float(row.get("partial_progress") or 0.0), 9),
                "termination_reason": str(row.get("termination_reason") or "unknown"),
                "steps": int(row.get("steps") or 0),
                "safety_violation": row.get("safety_violation") is True,
            }
        )
    normalized.sort(
        key=lambda row: (
            row["safety_violation"],
            -int(row["success"]),
            -float(row["partial_progress"]),
            int(row["steps"]),
            row["controller_id"],
        )
    )
    rank = 0
    prior_key: tuple[Any, ...] | None = None
    for index, row in enumerate(normalized, start=1):
        key = (
            row["safety_violation"],
            row["success"],
            row["partial_progress"],
            row["steps"],
        )
        if key != prior_key:
            rank = index
            prior_key = key
        row["rank"] = rank
    return normalized


def _decision_envelope(*, status: str, blockers: Sequence[str], result_sha256: str | None) -> dict[str, Any]:
    supported = status == "passed"
    envelope: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": "supported" if supported else "abstain",
        "claim": "blueprint_downstream_native_isaac_controller_evaluation_path_works_in_known_simready_workcell",
        "claim_label": CLAIM_LABEL,
        "evidence_result_sha256": result_sha256,
        "blockers": list(blockers),
        "explicitly_denied_claims": [
            "blueprint_capture_qualification",
            "arkitscenes_collision_readiness",
            "customer_site_validity",
            "learned_policy_quality",
            "sim_to_real_transfer",
            "physical_success",
            "deployment_readiness",
            "safety",
        ],
        "smallest_missing_measurement": None if supported else (blockers[0] if blockers else "native_control_evidence"),
    }
    envelope["envelope_sha256"] = canonical_sha256(envelope)
    return envelope


def assess_native_control_backend_result(
    *, spec: Mapping[str, Any], backend_result: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    physics = backend_result.get("scene_physics")
    physics = physics if isinstance(physics, Mapping) else {}
    if backend_result.get("runtime_backend") != "isaac_sim_6_physx":
        blockers.append("native_isaac_physx_backend_not_proven")
    if backend_result.get("hybrid_or_mujoco_backend_used") is not False:
        blockers.append("hybrid_or_mujoco_backend_not_denied")
    required_physics = (
        "meters_per_unit_valid",
        "up_axis_valid",
        "gravity_valid",
        "collision_inventory_valid",
        "dependency_closure_resolved",
        "settle_stable",
        "support_contact_proven",
        "initial_overlap_clear",
    )
    for field in required_physics:
        if physics.get(field) is not True:
            blockers.append(f"scene_physics_gate_failed:{field}")
    resets = backend_result.get("reset_evidence")
    resets = resets if isinstance(resets, Mapping) else {}
    if resets.get("cycle_count", 0) < int(spec["reset_contract"]["minimum_cycles"]):
        blockers.append("reset_cycle_count_insufficient")
    if resets.get("within_tolerances") is not True:
        blockers.append("reset_reproducibility_failed")
    positive = backend_result.get("positive_control")
    positive = positive if isinstance(positive, Mapping) else {}
    if positive.get("success") is not True:
        blockers.append("native_positive_control_failed")
    controllers = backend_result.get("controller_results")
    controllers = controllers if isinstance(controllers, list) else []
    if positive.get("success") is True and len(controllers) != CONTROLLER_COUNT:
        blockers.append("five_controller_execution_incomplete")
    ids = {str(row.get("controller_id") or "") for row in controllers if isinstance(row, Mapping)}
    expected_ids = {str(row["controller_id"]) for row in spec["controllers"]}
    if controllers and ids != expected_ids:
        blockers.append("five_controller_identity_mismatch")
    if backend_result.get("evidence_complete") is not True:
        blockers.append("native_control_evidence_incomplete")
    return {
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "scene_physics": dict(physics),
        "reset_evidence": dict(resets),
        "positive_control": dict(positive),
        "controller_results": [dict(row) for row in controllers if isinstance(row, Mapping)],
        "ranking": rank_controller_results(
            [row for row in controllers if isinstance(row, Mapping)]
        ),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_json_safe_evidence(dict(row)), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def isaac_sim_6_native_control_backend(
    *, spec: Mapping[str, Any], assets_root: Path, output_dir: Path
) -> dict[str, Any]:  # pragma: no cover - requires pinned Isaac GPU runtime
    """Execute the frozen task with robot, object, contacts, and dynamics in PhysX."""

    SimulationApp = import_simulation_app()
    simulation_app = SimulationApp(_simulation_app_launch_config())
    try:
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import XformPrim
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from isaacsim.storage.native import get_assets_root_path
        from omni.physx import get_physx_simulation_interface
        from pxr import (
            Gf,
            PhysicsSchemaTools,
            PhysxSchema,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
            UsdShade,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = output_dir / "traces"
        frame_dir = output_dir / "frames"
        trace_dir.mkdir()
        frame_dir.mkdir()
        _manifest_path, manifest = _load_materialization_manifest(assets_root)
        relocations = _apply_runtime_asset_relocations(assets_root=assets_root, manifest=manifest)
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(spec["physics"]["physics_dt_seconds"]),
            rendering_dt=float(spec["physics"]["rendering_dt_seconds"]),
            backend="torch",
            device="cuda:0",
        )
        stage = world.stage
        scene = spec["scene"]
        placements = scene["placements"]
        workcell_path = assets_root / str(scene["workcell_usd"])
        spraycan_path = assets_root / str(scene["spraycan_usd"])
        if not workcell_path.is_file() or not spraycan_path.is_file():
            raise FileNotFoundError("native_control_required_usd_missing")
        add_reference_to_stage(str(workcell_path), "/World/WarehouseWorkcell")
        add_reference_to_stage(str(spraycan_path), "/World/Task/SprayCan")

        def set_pose(path: str, position: Sequence[float], orientation: Sequence[float] | None = None) -> None:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError(f"native_control_prim_missing:{path}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
            if orientation is not None:
                w, x, y, z = map(float, orientation)
                xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(w, Gf.Vec3d(x, y, z))
                )

        set_pose("/World/WarehouseWorkcell", placements["workcell_translation_m"])
        set_pose("/World/Task/SprayCan", placements["spraycan_translation_m"])
        spray_prim = stage.GetPrimAtPath("/World/Task/SprayCan")
        rigid = UsdPhysics.RigidBodyAPI.Apply(spray_prim)
        rigid.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(spray_prim).CreateMassAttr(float(scene["spraycan_mass_kg"]))
        PhysxSchema.PhysxContactReportAPI.Apply(spray_prim).CreateThresholdAttr(0.0)

        tray_center = np.asarray(placements["tray_center_translation_m"], dtype=float)
        tray_dims = np.asarray(scene["tray_interior_dimensions_m"], dtype=float)
        wall = float(scene["tray_wall_thickness_m"])

        def static_cube(path: str, center: Sequence[float], scale: Sequence[float], color: Sequence[float]) -> None:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([tuple(map(float, color))])
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
            xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        static_cube("/World/Task/Tray/Floor", tray_center, (tray_dims[0], tray_dims[1], wall), (0.05, 0.1, 0.9))
        for name, offset, dims in (
            ("Left", (-tray_dims[0] / 2, 0, wall * 2), (wall, tray_dims[1], wall * 4)),
            ("Right", (tray_dims[0] / 2, 0, wall * 2), (wall, tray_dims[1], wall * 4)),
            ("Near", (0, -tray_dims[1] / 2, wall * 2), (tray_dims[0], wall, wall * 4)),
            ("Far", (0, tray_dims[1] / 2, wall * 2), (tray_dims[0], wall, wall * 4)),
        ):
            static_cube(
                f"/World/Task/Tray/{name}",
                tray_center + np.asarray(offset), dims, (0.05, 0.1, 0.9),
            )

        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(float(spec["physics"]["gravity_m_s2"]))
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(1000.0)
        key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        key.CreateIntensityAttr(2500.0)

        native_assets = get_assets_root_path() or ""
        franka_asset = native_assets.rstrip("/") + str(scene["franka_asset"])
        robot = world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="native_control_franka",
                position=np.asarray(placements["franka_base_translation_m"], dtype=float),
            )
        )
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath("/World/Franka")
        ).CreateThresholdAttr(0.0)
        spray = world.scene.add(SingleRigidPrim(prim_path="/World/Task/SprayCan", name="native_control_spraycan"))

        external = spec["cameras"]["external"]
        eye = np.asarray(external["position_m"], dtype=float)
        orientation = _camera_quaternion_wxyz(
            np.asarray(external["look_at_m"], dtype=float) - eye, (0.0, 0.0, 1.0)
        )
        camera_path = "/World/Cameras/External"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        camera_prim.CreateVerticalApertureAttr(15.2908)
        camera_prim.CreateHorizontalApertureAttr(20.3877)
        camera_prim.CreateFocalLengthAttr(
            15.2908 / (2.0 * math.tan(math.radians(float(external["vertical_fov_deg"])) / 2.0))
        )
        set_pose(camera_path, eye, orientation)
        wrist_path = "/World/Cameras/Wrist"
        wrist_prim = UsdGeom.Camera.Define(stage, wrist_path)
        wrist_prim.CreateVerticalApertureAttr(15.2908)
        wrist_prim.CreateHorizontalApertureAttr(20.3877)
        wrist_prim.CreateFocalLengthAttr(9.0)
        set_pose(wrist_path, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))
        world.reset()
        camera = CameraSensor(RtxCamera(camera_path), resolution=(480, 640), annotators=["rgba", "distance_to_camera"])
        wrist_pose_view = XformPrim(wrist_path)
        physics_view = world.physics_sim_view
        if physics_view is None:
            raise ValueError("native_control_articulation_physics_view_missing")
        initial_link_matrices = _articulation_link_world_pose_matrices(
            robot=robot, simulation_view=physics_view
        )
        initial_hand = initial_link_matrices.get("panda_hand")
        if initial_hand is None:
            raise ValueError("native_control_panda_hand_link_missing")
        wrist_quaternion, wrist_calibration = _rigid_wrist_mount_from_initial_task_framing(
            parent_to_world=initial_hand,
            mount_translation_parent=(0.0, 0.10, 0.03),
            target_world_points={"spraycan": placements["spraycan_translation_m"], "tray": tray_center},
            world_up=(0.0, 0.0, 1.0),
            camera_eye_world_offset=(0.0, 0.0, 0.08),
        )
        wrist_mount_translation = wrist_calibration["resolved_mount_translation_parent_m"]
        wrist_camera = CameraSensor(
            RtxCamera(wrist_path), resolution=(480, 640), annotators=["rgba", "distance_to_camera"]
        )

        def sync_wrist() -> np.ndarray:
            matrices = _articulation_link_world_pose_matrices(
                robot=robot, simulation_view=physics_view
            )
            hand = matrices.get("panda_hand")
            if hand is None:
                raise ValueError("native_control_panda_hand_link_missing")
            _synchronize_camera_to_rigid_link(
                pose_view=wrist_pose_view,
                parent_to_world=hand,
                mount_translation_parent=wrist_mount_translation,
                mount_orientation_parent_wxyz=wrist_quaternion,
            )
            return _unified_world_pose_matrix(wrist_pose_view)

        initial_joints = np.asarray(spec["reset_contract"]["joint_positions"], dtype=float)
        initial_object = np.asarray(placements["spraycan_translation_m"], dtype=float)
        initial_orientation = np.asarray(placements["spraycan_orientation_wxyz"], dtype=float)
        contacts: list[dict[str, Any]] = []

        def on_contacts(headers: Any, data: Any) -> None:
            step = int(world.current_time_step_index)
            for header in headers:
                try:
                    actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                    actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                except Exception:
                    actor0, actor1 = str(header.actor0), str(header.actor1)
                minimum_separation = None
                maximum_impulse = 0.0
                for index in range(int(header.contact_data_offset), int(header.contact_data_offset + header.num_contact_data)):
                    point = data[index]
                    separation = float(getattr(point, "separation", 0.0))
                    impulse = float(getattr(point, "impulse", 0.0))
                    minimum_separation = separation if minimum_separation is None else min(minimum_separation, separation)
                    maximum_impulse = max(maximum_impulse, abs(impulse))
                contacts.append({
                    "physics_step": step,
                    "actor0": actor0,
                    "actor1": actor1,
                    "event_type": int(getattr(header, "type", -1)),
                    "minimum_separation_m": minimum_separation,
                    "maximum_impulse": maximum_impulse,
                })

        _contact_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        )

        def state() -> dict[str, Any]:
            joints = _backend_array_to_numpy(robot.get_joint_positions()).astype(float)
            joint_vel = _backend_array_to_numpy(robot.get_joint_velocities()).astype(float)
            obj_pos, obj_quat = spray.get_world_pose()
            return {
                "physics_step": int(world.current_time_step_index),
                "joint_positions": joints.tolist(),
                "joint_velocities": joint_vel.tolist(),
                "object_position_m": _backend_array_to_numpy(obj_pos).astype(float).tolist(),
                "object_orientation_wxyz": _backend_array_to_numpy(obj_quat).astype(float).tolist(),
                "object_linear_velocity_m_s": _backend_array_to_numpy(spray.get_linear_velocity()).astype(float).tolist(),
                "object_angular_velocity_rad_s": _backend_array_to_numpy(spray.get_angular_velocity()).astype(float).tolist(),
            }

        def reset_state() -> dict[str, Any]:
            contact_start = len(contacts)
            convert = getattr(getattr(robot, "_backend_utils", None), "convert", None)
            device = getattr(robot, "_device", None)
            joint_values = convert(initial_joints, device) if callable(convert) else initial_joints
            zero_values = (
                convert(np.zeros_like(initial_joints), device)
                if callable(convert)
                else np.zeros_like(initial_joints)
            )
            robot.set_joint_positions(joint_values)
            robot.set_joint_velocities(zero_values)
            spray.set_world_pose(position=initial_object, orientation=initial_orientation)
            spray.set_linear_velocity(np.zeros(3))
            spray.set_angular_velocity(np.zeros(3))
            for _ in range(int(spec["reset_contract"]["settle_steps_after_reset"])):
                world.step(render=False)
            row = state()
            row["wrist_camera_world_matrix"] = sync_wrist().tolist()
            reset_contacts = contacts[contact_start:]
            contact_pairs = sorted(
                {
                    "|".join(sorted((str(item["actor0"]), str(item["actor1"]))))
                    for item in reset_contacts
                }
            )
            contact_blob = json.dumps(reset_contacts)
            row["contact_state"] = {
                "event_count": len(reset_contacts),
                "pair_signature": contact_pairs,
                "pair_signature_sha256": canonical_sha256(contact_pairs),
                "support_contact": (
                    "SprayCan" in contact_blob and "WarehouseWorkcell" in contact_blob
                ),
            }
            object_position = np.asarray(row["object_position_m"], dtype=float)
            row["task_state"] = {
                "inside_tray": bool(
                    np.all(
                        np.abs(object_position[:2] - tray_center[:2])
                        <= tray_dims[:2] / 2.0
                    )
                ),
                "distance_to_tray_xy_m": float(
                    np.linalg.norm(object_position[:2] - tray_center[:2])
                ),
                "success_predicate": False,
            }
            return row

        settle_start = len(contacts)
        settled = reset_state()
        for _ in range(int(spec["physics"]["initial_settle_steps"])):
            world.step(render=False)
        settled = state()
        settle_contacts = contacts[settle_start:]
        object_pos = np.asarray(settled["object_position_m"], dtype=float)
        object_speed = float(np.linalg.norm(settled["object_linear_velocity_m_s"]))
        angular_speed = float(np.linalg.norm(settled["object_angular_velocity_rad_s"]))
        settled_joints = np.asarray(settled["joint_positions"], dtype=float)
        settled_joint_velocity = np.asarray(settled["joint_velocities"], dtype=float)
        contact_text = json.dumps(settle_contacts)
        min_sep = min(
            (float(row["minimum_separation_m"]) for row in settle_contacts if row["minimum_separation_m"] is not None),
            default=0.0,
        )
        collision_inventory = {
            "workcell_collision_prim_count": sum(
                1 for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/WarehouseWorkcell"))
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            ),
            "spraycan_collision_prim_count": sum(
                1 for prim in Usd.PrimRange(spray_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)
            ),
            "tray_collision_prim_count": 5,
            "material_binding_prim_count": sum(
                1
                for prim in stage.TraverseAll()
                if UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
            ),
            "physics_authorship": {
                "nvidia_heavy_duty_table": "source_collision_schema_static_body",
                "nvidia_spraycan": "source_collision_schema_plus_runtime_root_rigid_body_mass",
                "official_franka": "official_articulation_with_link_colliders",
                "blueprint_marked_tray": "runtime_static_collision_target",
            },
        }
        scene_physics = {
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "meters_per_unit_valid": abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) < 1e-12,
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "up_axis_valid": str(UsdGeom.GetStageUpAxis(stage)).upper() == "Z",
            "gravity_m_s2": float(spec["physics"]["gravity_m_s2"]),
            "gravity_valid": abs(float(spec["physics"]["gravity_m_s2"]) - 9.81) < 1e-9,
            "collision_inventory": collision_inventory,
            "collision_inventory_valid": collision_inventory["workcell_collision_prim_count"] >= 2 and collision_inventory["spraycan_collision_prim_count"] >= 1,
            "physics_scene_path": "/World/PhysicsScene",
            "spraycan_rigid_body_authored_at_runtime": spray_prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "dependency_closure_resolved": all(
                (assets_root / str(row.get("relative_path") or "")).is_file()
                for row in manifest.get("files") or []
            ),
            "settle_object_linear_speed_m_s": object_speed,
            "settle_object_angular_speed_rad_s": angular_speed,
            "settle_stable": bool(
                np.isfinite(object_pos).all()
                and np.isfinite(settled_joints).all()
                and np.isfinite(settled_joint_velocity).all()
                and object_pos[2] > float(spec["physics"]["fall_through_z_m"])
                and object_speed <= float(spec["physics"]["settle_linear_speed_max_m_s"])
                and angular_speed <= float(spec["physics"]["settle_angular_speed_max_rad_s"])
                and float(np.max(np.abs(settled_joint_velocity))) <= 5.0
            ),
            "support_contact_proven": "SprayCan" in contact_text and "WarehouseWorkcell" in contact_text,
            "minimum_contact_separation_m": min_sep,
            "initial_overlap_clear": min_sep >= -float(spec["physics"]["maximum_initial_penetration_m"]),
            "initial_contact_support_checks": {
                "robot": {
                    "fixed_base": True,
                    "unexpected_task_object_contact": "Franka" in contact_text and "SprayCan" in contact_text,
                },
                "work_surface": {
                    "collision_authored": collision_inventory["workcell_collision_prim_count"] >= 2,
                    "supports_task_object": "SprayCan" in contact_text and "WarehouseWorkcell" in contact_text,
                },
                "task_object": {
                    "rigid_body": True,
                    "collision_authored": collision_inventory["spraycan_collision_prim_count"] >= 1,
                    "unsupported": not ("SprayCan" in contact_text and "WarehouseWorkcell" in contact_text),
                },
                "tray": {
                    "collision_authored": True,
                    "initial_task_object_contact": "Tray" in contact_text and "SprayCan" in contact_text,
                },
                "relevant_obstacles": {
                    "source": "bounded_heavy_duty_table_only",
                    "unexpected_penetration": min_sep < -float(spec["physics"]["maximum_initial_penetration_m"]),
                },
            },
            "no_nan_or_explosive_motion": bool(
                np.isfinite(object_pos).all()
                and np.isfinite(settled_joints).all()
                and np.isfinite(settled_joint_velocity).all()
                and float(np.max(np.abs(settled_joint_velocity))) <= 5.0
            ),
            "no_falling_through": bool(object_pos[2] > float(spec["physics"]["fall_through_z_m"])),
            "no_unrecoverable_penetration": bool(min_sep >= -float(spec["physics"]["maximum_initial_penetration_m"])),
            "runtime_asset_relocations": relocations,
            "franka_asset": franka_asset,
            "franka_dof_count": int(getattr(robot, "num_dof", getattr(robot, "num_dofs", -1))),
            "franka_dof_names": list(robot.dof_names),
        }
        _write_jsonl(trace_dir / "settle_contacts.jsonl", settle_contacts)

        reset_rows: list[dict[str, Any]] = []
        for cycle in range(int(spec["reset_contract"]["minimum_cycles"])):
            row = reset_state()
            material = {
                key: row[key]
                for key in (
                    "joint_positions", "joint_velocities", "object_position_m",
                    "object_orientation_wxyz", "object_linear_velocity_m_s",
                    "object_angular_velocity_rad_s", "wrist_camera_world_matrix",
                    "contact_state", "task_state",
                )
            }
            row["cycle"] = cycle
            row["state_sha256"] = canonical_sha256(material)
            reset_rows.append(row)
        reference = reset_rows[0]
        joint_dev = max(
            float(np.max(np.abs(np.asarray(row["joint_positions"]) - np.asarray(reference["joint_positions"]))))
            for row in reset_rows
        )
        object_dev = max(
            float(np.max(np.abs(np.asarray(row["object_position_m"]) - np.asarray(reference["object_position_m"]))))
            for row in reset_rows
        )
        orientation_dev = max(
            _quaternion_angle_rad(
                row["object_orientation_wxyz"], reference["object_orientation_wxyz"]
            )
            for row in reset_rows
        )
        joint_velocity_max = max(
            float(np.max(np.abs(np.asarray(row["joint_velocities"]))))
            for row in reset_rows
        )
        object_linear_velocity_max = max(
            float(np.linalg.norm(row["object_linear_velocity_m_s"]))
            for row in reset_rows
        )
        object_angular_velocity_max = max(
            float(np.linalg.norm(row["object_angular_velocity_rad_s"]))
            for row in reset_rows
        )
        camera_dev = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(row["wrist_camera_world_matrix"])
                        - np.asarray(reference["wrist_camera_world_matrix"])
                    )
                )
            )
            for row in reset_rows
        )
        contact_event_dev = max(
            abs(
                int(row["contact_state"]["event_count"])
                - int(reference["contact_state"]["event_count"])
            )
            for row in reset_rows
        )
        contact_signature_equal = all(
            row["contact_state"]["pair_signature_sha256"]
            == reference["contact_state"]["pair_signature_sha256"]
            for row in reset_rows
        )
        support_contact_every_cycle = all(
            row["contact_state"]["support_contact"] is True for row in reset_rows
        )
        task_distance_dev = max(
            abs(
                float(row["task_state"]["distance_to_tray_xy_m"])
                - float(reference["task_state"]["distance_to_tray_xy_m"])
            )
            for row in reset_rows
        )
        task_state_equal = all(
            row["task_state"]["inside_tray"] == reference["task_state"]["inside_tray"]
            and row["task_state"]["success_predicate"]
            == reference["task_state"]["success_predicate"]
            for row in reset_rows
        )
        reset_contract = spec["reset_contract"]
        within_tolerances = bool(
            joint_dev <= float(reset_contract["joint_position_tolerance_rad"])
            and joint_velocity_max
            <= float(reset_contract["joint_velocity_tolerance_rad_s"])
            and object_dev <= float(reset_contract["object_position_tolerance_m"])
            and orientation_dev
            <= float(reset_contract["object_orientation_tolerance_rad"])
            and object_linear_velocity_max
            <= float(reset_contract["object_velocity_tolerance_m_s"])
            and object_angular_velocity_max
            <= float(reset_contract["object_angular_velocity_tolerance_rad_s"])
            and camera_dev <= float(reset_contract["camera_transform_tolerance_m"])
            and contact_event_dev
            <= int(reset_contract["contact_event_count_tolerance"])
            and contact_signature_equal
            and support_contact_every_cycle
            and task_distance_dev <= float(reset_contract["task_state_tolerance_m"])
            and task_state_equal
        )
        reset_evidence = {
            "cycle_count": len(reset_rows),
            "state_hashes": [row["state_sha256"] for row in reset_rows],
            "max_joint_position_deviation_rad": joint_dev,
            "max_joint_velocity_rad_s": joint_velocity_max,
            "max_object_position_deviation_m": object_dev,
            "max_object_orientation_deviation_rad": orientation_dev,
            "max_object_linear_velocity_m_s": object_linear_velocity_max,
            "max_object_angular_velocity_rad_s": object_angular_velocity_max,
            "max_camera_transform_deviation": camera_dev,
            "max_contact_event_count_deviation": contact_event_dev,
            "contact_signature_equal": contact_signature_equal,
            "support_contact_every_cycle": support_contact_every_cycle,
            "max_task_state_distance_deviation_m": task_distance_dev,
            "task_state_equal": task_state_equal,
            "within_tolerances": within_tolerances,
            "cycles": reset_rows,
        }
        write_json(trace_dir / "reset_evidence.json", reset_evidence)

        def save_frame(run_id: str, phase: str) -> list[dict[str, Any]]:
            sync_wrist()
            for _ in range(2):
                world.render()
            saved = []
            for view_id, sensor in (("external", camera), ("wrist", wrist_camera)):
                frame = _camera_sensor_annotator_frame(sensor=sensor, annotator="rgba")
                rgba = np.asarray(frame["data"])
                path = frame_dir / f"{run_id}_{phase}_{view_id}.png"
                Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(path)
                saved.append({"view": view_id, "phase": phase, "relative_path": str(path.relative_to(output_dir)), "sha256": file_sha256(path), "physics_step": int(world.current_time_step_index), "resolution": [640, 480]})
            return saved

        def run_controller(definition: Mapping[str, Any], *, positive: bool) -> dict[str, Any]:
            run_id = str(definition["controller_id"])
            reset_state()
            controller = PickPlaceController(
                name=f"{run_id}_pick_place",
                gripper=robot.gripper,
                robot_articulation=robot,
                events_dt=list(map(float, definition["events_dt"])),
            )
            controller.reset()
            start_state = state()
            start_pos = np.asarray(start_state["object_position_m"], dtype=float)
            initial_distance = float(np.linalg.norm(start_pos[:2] - tray_center[:2]))
            frames = save_frame(run_id, "initial")
            rows: list[dict[str, Any]] = []
            start_contact = len(contacts)
            max_lift = 0.0
            mid_saved = False
            terminated = "timeout"
            max_steps = int(definition["max_steps"])
            actual_steps = 0
            for step in range(max_steps):
                current = state()
                obj = np.asarray(current["object_position_m"], dtype=float)
                pick = obj + np.asarray(definition["pick_offset_m"], dtype=float)
                place = tray_center + np.asarray(definition["place_offset_m"], dtype=float)
                if definition.get("mode") == "stationary":
                    action = None
                else:
                    action = controller.forward(
                        picking_position=pick,
                        placing_position=place,
                        current_joint_positions=np.asarray(current["joint_positions"], dtype=float),
                        end_effector_offset=np.asarray(definition["end_effector_offset_m"], dtype=float),
                    )
                    robot.apply_action(action)
                world.step(render=False)
                actual_steps = step + 1
                after = state()
                max_lift = max(max_lift, float(after["object_position_m"][2]) - float(start_pos[2]))
                if step % int(spec["trace_contract"]["state_stride_steps"]) == 0:
                    rows.append({
                        "run_id": run_id,
                        "step": step,
                        "seed": int(spec["seed"]),
                        "observation": current,
                        "action_joint_positions": None
                        if action is None or getattr(action, "joint_positions", None) is None
                        else _backend_array_to_numpy(action.joint_positions).astype(float).tolist(),
                        "task_state": {"lift_m": max_lift, "distance_to_tray_xy_m": float(np.linalg.norm(np.asarray(after["object_position_m"])[:2] - tray_center[:2]))},
                    })
                if not mid_saved and step >= max_steps // 2:
                    frames.extend(save_frame(run_id, "intermediate"))
                    mid_saved = True
                if definition.get("mode") != "stationary" and controller.is_done():
                    terminated = "controller_done"
                    break
            for _ in range(int(spec["task"]["terminal_stability_steps"])):
                world.step(render=False)
            final = state()
            frames.extend(save_frame(run_id, "terminal"))
            final_pos = np.asarray(final["object_position_m"], dtype=float)
            half = tray_dims[:2] / 2.0
            inside = bool(np.all(np.abs(final_pos[:2] - tray_center[:2]) <= half))
            lifted = max_lift >= float(spec["task"]["minimum_lift_m"])
            stable = float(np.linalg.norm(final["object_linear_velocity_m_s"])) <= float(spec["task"]["terminal_linear_speed_max_m_s"])
            run_contacts = contacts[start_contact:]
            contact_blob = json.dumps(run_contacts)
            gripper_contact = "SprayCan" in contact_blob and ("finger" in contact_blob.lower() or "hand" in contact_blob.lower())
            final_robot_contact = any(
                int(row["physics_step"]) >= int(world.current_time_step_index) - int(spec["task"]["terminal_stability_steps"])
                and "SprayCan" in json.dumps(row)
                and ("finger" in json.dumps(row).lower() or "hand" in json.dumps(row).lower())
                for row in run_contacts
            )
            success = bool(lifted and inside and stable and not final_robot_contact and gripper_contact)
            final_distance = float(np.linalg.norm(final_pos[:2] - tray_center[:2]))
            progress = max(0.0, min(1.0, 0.5 * min(max_lift / float(spec["task"]["minimum_lift_m"]), 1.0) + 0.5 * max(0.0, min(1.0, (initial_distance - final_distance) / max(initial_distance, 1e-9)))))
            trace_path = trace_dir / f"{run_id}.jsonl"
            contact_path = trace_dir / f"{run_id}_contacts.jsonl"
            _write_jsonl(trace_path, rows)
            _write_jsonl(contact_path, run_contacts)
            return {
                "controller_id": run_id,
                "positive_control": positive,
                "seed": int(spec["seed"]),
                "success": success,
                "termination_reason": "success" if success else terminated,
                "steps": actual_steps,
                "partial_progress": progress,
                "max_lift_m": max_lift,
                "initial_distance_to_tray_xy_m": initial_distance,
                "final_distance_to_tray_xy_m": final_distance,
                "predicate_inputs": {"lifted": lifted, "inside_tray": inside, "stable": stable, "gripper_contact": gripper_contact, "not_in_robot_contact_final": not final_robot_contact},
                "contact_event_count": len(run_contacts),
                "safety_violation": any((row.get("minimum_separation_m") or 0.0) < -float(spec["physics"]["maximum_runtime_penetration_m"]) for row in run_contacts),
                "frames": frames,
                "trace": {"relative_path": str(trace_path.relative_to(output_dir)), "sha256": file_sha256(trace_path)},
                "contacts": {"relative_path": str(contact_path.relative_to(output_dir)), "sha256": file_sha256(contact_path)},
            }

        positive = run_controller(spec["positive_control"], positive=True)
        controller_results: list[dict[str, Any]] = []
        if positive["success"]:
            for definition in spec["controllers"]:
                controller_results.append(run_controller(definition, positive=False))
        return {
            "runtime_backend": "isaac_sim_6_physx",
            "hybrid_or_mujoco_backend_used": False,
            "isaac_sim_version": "6.0.1",
            "scene_physics": scene_physics,
            "reset_evidence": reset_evidence,
            "positive_control": positive,
            "controller_results": controller_results,
            "evidence_complete": bool(positive.get("frames") and (not positive["success"] or len(controller_results) == CONTROLLER_COUNT)),
        }
    finally:
        simulation_app.close()


def run_native_control_canary(
    *,
    spec_path: str | Path,
    assets_root: str | Path,
    output_dir: str | Path,
    backend: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    spec = _validated_spec(spec_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("nvidia_warehouse_native_control_output_exists")
    output.mkdir(parents=True)
    backend_result = backend(
        spec=spec,
        assets_root=Path(assets_root).expanduser().resolve(),
        output_dir=output / "runtime",
    )
    if not isinstance(backend_result, Mapping):
        raise ValueError("nvidia_warehouse_native_control_backend_result_invalid")
    assessment = assess_native_control_backend_result(spec=spec, backend_result=backend_result)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": assessment["status"],
        "blockers": assessment["blockers"],
        "claim_label": CLAIM_LABEL,
        "spec_sha256": spec["spec_sha256"],
        "dataset_revision": DATASET_REVISION,
        "runtime_backend": backend_result.get("runtime_backend"),
        "hybrid_or_mujoco_backend_used": backend_result.get("hybrid_or_mujoco_backend_used"),
        "assessment": assessment,
        "backend_result": _json_safe_evidence(backend_result),
        "claim_boundary": {
            "nvidia_authored_simready_control_scene": True,
            "native_isaac_physics": backend_result.get("runtime_backend") == "isaac_sim_6_physx",
            "scripted_controllers": True,
            "single_workcell": True,
            "simulation_only": True,
            "capture_qualification": False,
            "arkitscenes_collision_readiness": False,
            "customer_site_validity": False,
            "learned_policy_quality": False,
            "sim_to_real_transfer": False,
            "physical_success": False,
            "deployment_readiness": False,
            "safety": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json(output / "native_control_result.json", result)
    envelope = _decision_envelope(
        status=result["status"], blockers=result["blockers"], result_sha256=result["result_sha256"]
    )
    write_json(output / "decision_envelope.json", envelope)
    evidence_files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        evidence_files.append({
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        })
    index = {
        "schema_version": "nvidia_warehouse_native_control_evidence_index.v1",
        "files": evidence_files,
        "file_count": len(evidence_files),
    }
    index["index_sha256"] = canonical_sha256(index)
    write_json(output / "evidence_index.json", index)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_native_control_canary(
        spec_path=args.spec,
        assets_root=args.assets_root,
        output_dir=args.output,
        backend=isaac_sim_6_native_control_backend,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
