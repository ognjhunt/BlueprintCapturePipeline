"""Native Isaac camera canary for the pinned NVIDIA Warehouse workcell."""

from __future__ import annotations

import argparse
import json
import math
import posixpath
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import write_json
from .nvidia_warehouse_workcell import CANARY_SPEC_SCHEMA_VERSION
from .policy_ranking_thesis import canonical_sha256, file_sha256


RESULT_SCHEMA_VERSION = "nvidia_warehouse_native_camera_canary_result.v1"
MIN_NONBLANK_SPATIAL_STD = 4.0
MIN_WRIST_WORLD_DISPLACEMENT_M = 0.001
MAX_WRIST_LOCAL_TRANSFORM_DELTA = 1e-9
REQUIRED_VIEWS = ("external", "wrist")


def _simulation_app_launch_config() -> dict[str, Any]:
    """Return a fresh launcher config that lets the evidence wrapper resume."""

    return {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": 640,
        "height": 480,
        # Isaac Sim 6 defaults this to true, which terminates the process from
        # SimulationApp.close() before the outer worker can archive and upload
        # the camera evidence.
        "fast_shutdown": False,
    }


def import_simulation_app() -> Any:
    """Resolve the Isaac launcher while rejecting non-callable API shims."""

    try:
        from isaacsim import SimulationApp

        if callable(SimulationApp):
            return SimulationApp
    except Exception:
        pass
    from omni.isaac.kit import SimulationApp

    if not callable(SimulationApp):
        raise ImportError("isaac_simulation_app_not_callable")
    return SimulationApp


def _camera_quaternion_wxyz(forward: Any, up: Any) -> np.ndarray:
    """Quaternion for a USD camera whose local forward axis is negative Z."""

    forward_array = np.asarray(forward, dtype=float)
    up_array = np.asarray(up, dtype=float)
    forward_array /= np.linalg.norm(forward_array)
    right = np.cross(forward_array, up_array)
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, forward_array)
    corrected_up /= np.linalg.norm(corrected_up)
    rotation = np.column_stack((right, corrected_up, -forward_array))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        final_index = (index + 2) % 3
        scale = math.sqrt(
            1.0 + rotation[index, index] - rotation[next_index, next_index]
            - rotation[final_index, final_index]
        ) * 2.0
        xyz = np.zeros(3)
        xyz[index] = 0.25 * scale
        xyz[next_index] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        xyz[final_index] = (rotation[final_index, index] + rotation[index, final_index]) / scale
        w = (rotation[final_index, next_index] - rotation[next_index, final_index]) / scale
        quaternion = np.concatenate(([w], xyz))
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _project_world_points(
    *, camera_to_world: Any, points: Mapping[str, Any], width: int, height: int, vfov_deg: float
) -> dict[str, bool]:
    """Project named world points with the same negative-Z USD camera convention."""

    matrix = np.asarray(camera_to_world, dtype=float).reshape(4, 4)
    world_to_camera = np.linalg.inv(matrix)
    focal = float(height) / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    result: dict[str, bool] = {}
    for name, value in points.items():
        homogeneous = np.concatenate((np.asarray(value, dtype=float), [1.0]))
        camera = homogeneous @ world_to_camera
        depth = -float(camera[2])
        if depth <= 1e-9:
            result[str(name)] = False
            continue
        u = focal * float(camera[0]) / depth + (float(width) - 1.0) / 2.0
        v = -focal * float(camera[1]) / depth + (float(height) - 1.0) / 2.0
        result[str(name)] = 0.0 <= u < width and 0.0 <= v < height
    return result


def _matrix_array(matrix: Any) -> np.ndarray:
    return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)])


def _relocate_layer_asset_path(
    layer_path: Path, source_asset_uri: str, replacement_authored_path: str
) -> int:  # pragma: no cover - exercised inside the pinned Isaac/OpenUSD image
    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        raise ValueError(f"native_warehouse_asset_relocation_layer_missing:{layer_path}")
    replacements = 0

    def relocate(value: str) -> str:
        nonlocal replacements
        if value == source_asset_uri:
            replacements += 1
            return replacement_authored_path
        return value

    UsdUtils.ModifyAssetPaths(layer, relocate)
    if replacements <= 0:
        raise ValueError("native_warehouse_asset_relocation_source_not_authored")
    layer.Save()
    return replacements


def _apply_runtime_asset_relocations(
    *,
    assets_root: Path,
    manifest: Mapping[str, Any],
    layer_relocator: Callable[[Path, str, str], int] = _relocate_layer_asset_path,
) -> dict[str, Any]:
    """Apply hash-bound local mirrors before the workcell composition is loaded."""

    root = assets_root.expanduser().resolve()
    rows = manifest.get("runtime_asset_relocations") or []
    if not isinstance(rows, list):
        raise ValueError("native_warehouse_asset_relocations_invalid")
    authored_replacement_count = 0
    applied: list[dict[str, Any]] = []
    for value in rows:
        if not isinstance(value, Mapping):
            raise ValueError("native_warehouse_asset_relocation_invalid")
        owner_relative = str(value.get("owner_relative_path") or "")
        replacement_relative = str(value.get("replacement_relative_path") or "")
        source_asset_uri = str(value.get("source_asset_uri") or "")
        replacement_authored = str(value.get("replacement_authored_path") or "")
        owner = (root / owner_relative).resolve()
        replacement = (root / replacement_relative).resolve()
        if (
            not owner.is_relative_to(root)
            or not replacement.is_relative_to(root)
            or not owner.is_file()
            or not replacement.is_file()
            or not source_asset_uri
            or not replacement_authored.startswith(("./", "../"))
        ):
            raise ValueError("native_warehouse_asset_relocation_path_invalid")
        expected_authored = posixpath.relpath(replacement_relative, posixpath.dirname(owner_relative))
        if not expected_authored.startswith(("./", "../")):
            expected_authored = "./" + expected_authored
        if replacement_authored != expected_authored:
            raise ValueError("native_warehouse_asset_relocation_binding_invalid")
        count = int(layer_relocator(owner, source_asset_uri, replacement_authored))
        if count <= 0:
            raise ValueError("native_warehouse_asset_relocation_not_applied")
        authored_replacement_count += count
        applied.append(
            {
                "owner_relative_path": owner_relative,
                "replacement_relative_path": replacement_relative,
                "authored_replacement_count": count,
            }
        )
    return {
        "relocation_count": len(applied),
        "authored_replacement_count": authored_replacement_count,
        "relocations": applied,
    }


def _load_materialization_manifest(assets_root: Path) -> tuple[Path, dict[str, Any]]:
    """Load the manifest from a materialization root or extracted bundle layout."""

    root = assets_root.expanduser().resolve()
    direct = root / "materialization_manifest.json"
    extracted = root.parent / "materialization_manifest.json"
    candidates = [direct]
    if root.name == "assets":
        candidates.append(extracted)
    manifest_path = next(
        (path for path in candidates if path.is_file() and not path.is_symlink()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError("native_warehouse_materialization_manifest_missing")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("native_warehouse_materialization_manifest_not_object")
    return manifest_path, dict(value)


def isaac_sim_6_backend(
    *, spec: Mapping[str, Any], assets_root: Path, output_dir: Path
) -> dict[str, Any]:  # pragma: no cover - requires the pinned Isaac GPU image
    """Load the selected workcell and render synchronized external/wrist frames."""

    SimulationApp = import_simulation_app()
    simulation_app = SimulationApp(_simulation_app_launch_config())
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        output_dir.mkdir(parents=True, exist_ok=True)
        _manifest_path, manifest = _load_materialization_manifest(assets_root)
        relocation_evidence = _apply_runtime_asset_relocations(
            assets_root=assets_root, manifest=manifest
        )
        world = World(stage_units_in_meters=1.0)
        stage = world.stage
        scene = spec["scene"]
        placements = scene["placements"]
        workcell_path = assets_root / str(scene["workcell_usd"])
        spraycan_path = assets_root / str(scene["spraycan_usd"])
        if not workcell_path.is_file() or not spraycan_path.is_file():
            raise FileNotFoundError("native_warehouse_required_usd_missing")

        add_reference_to_stage(str(workcell_path), "/World/WarehouseWorkcell")
        add_reference_to_stage(str(spraycan_path), "/World/Task/SprayCan")
        native_assets = get_assets_root_path() or ""
        franka_path = native_assets.rstrip("/") + str(scene["franka_asset"])
        add_reference_to_stage(franka_path, "/World/Franka")

        def set_pose(prim_path: str, translation: Any, quaternion: Any | None = None) -> None:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"native_warehouse_prim_missing:{prim_path}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
            if quaternion is not None:
                w, x, y, z = map(float, quaternion)
                xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(w, Gf.Vec3d(x, y, z))
                )

        set_pose("/World/WarehouseWorkcell", placements["workcell_translation_m"])
        set_pose("/World/Franka", placements["franka_base_translation_m"])
        set_pose("/World/Task/SprayCan", placements["spraycan_translation_m"])

        spray_prim = stage.GetPrimAtPath("/World/Task/SprayCan")
        rigid = UsdPhysics.RigidBodyAPI.Apply(spray_prim)
        rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(spray_prim).CreateMassAttr(0.25)
        collision_count = sum(
            1 for prim in Usd.PrimRange(spray_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)
        )

        tray_center = placements["tray_center_translation_m"]
        tray = UsdGeom.Cube.Define(stage, "/World/Task/Tray")
        tray.CreateSizeAttr(1.0)
        tray.CreateDisplayColorAttr([(0.05, 0.1, 0.9)])
        tray_xform = UsdGeom.Xformable(tray.GetPrim())
        tray_xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, tray_center)))
        tray_xform.AddScaleOp().Set(Gf.Vec3f(0.36, 0.28, 0.03))
        UsdPhysics.CollisionAPI.Apply(tray.GetPrim())

        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(1000.0)
        distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        distant.CreateIntensityAttr(2500.0)

        camera_objects: dict[str, Camera] = {}
        external = spec["cameras"]["external"]
        external_eye = np.asarray(external["position_m"], dtype=float)
        external_forward = np.asarray(external["look_at_m"], dtype=float) - external_eye
        external_quaternion = _camera_quaternion_wxyz(external_forward, (0.0, 0.0, 1.0))
        external_path = "/World/Cameras/External"
        external_prim = UsdGeom.Camera.Define(stage, external_path)
        external_prim.CreateVerticalApertureAttr(15.2908)
        external_prim.CreateHorizontalApertureAttr(15.2908 * 640.0 / 480.0)
        external_prim.CreateFocalLengthAttr(
            15.2908 / (2.0 * math.tan(math.radians(float(external["vertical_fov_deg"])) / 2.0))
        )
        set_pose(external_path, external_eye, external_quaternion)

        wrist = spec["cameras"]["wrist"]
        wrist_path = "/World/Franka/panda_hand/BlueprintWristCamera"
        wrist_prim = UsdGeom.Camera.Define(stage, wrist_path)
        wrist_prim.CreateVerticalApertureAttr(15.2908)
        wrist_prim.CreateHorizontalApertureAttr(15.2908 * 640.0 / 480.0)
        wrist_prim.CreateFocalLengthAttr(
            15.2908 / (2.0 * math.tan(math.radians(float(wrist["vertical_fov_deg"])) / 2.0))
        )
        wrist_quaternion = _camera_quaternion_wxyz(
            wrist["mount_forward_parent"], wrist["mount_up_parent"]
        )
        set_pose(wrist_path, wrist["mount_translation_m"], wrist_quaternion)

        robot = SingleArticulation(prim_path="/World/Franka", name="native_warehouse_franka")
        world.scene.add(robot)
        world.reset()
        camera_objects["external"] = Camera(prim_path=external_path, resolution=(640, 480))
        camera_objects["wrist"] = Camera(prim_path=wrist_path, resolution=(640, 480))
        for camera in camera_objects.values():
            camera.initialize()

        initial_joints = np.asarray(
            [0.2897, 0.50732, -0.140016, -2.176, -0.0310497, 2.51592, -0.49251, 0.04, 0.04]
        )
        commanded_joints = initial_joints.copy()
        commanded_joints[0] += 0.12
        robot.set_joint_positions(initial_joints)
        for _ in range(20):
            world.step(render=True)

        def camera_matrix(path: str) -> np.ndarray:
            cache = UsdGeom.XformCache()
            return _matrix_array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))

        wrist_local_initial = _matrix_array(
            UsdGeom.Xformable(stage.GetPrimAtPath(wrist_path)).GetLocalTransformation()
        )
        wrist_world_initial = camera_matrix(wrist_path)
        hand_initial = camera_matrix("/World/Franka/panda_hand")[3, :3]
        entity_points = {
            "franka": hand_initial,
            "spraycan": placements["spraycan_translation_m"],
            "tray": tray_center,
        }

        frames: dict[str, dict[str, Any]] = {view: {} for view in REQUIRED_VIEWS}

        def save_frames(phase: str) -> int:
            step = int(world.current_time_step_index)
            for view_id, camera in camera_objects.items():
                rgba = np.asarray(camera.get_rgba())
                if rgba.ndim != 3 or rgba.shape[0:2] != (480, 640):
                    raise ValueError(f"native_camera_frame_shape_invalid:{view_id}:{rgba.shape}")
                path = output_dir / f"{view_id}_{phase}.png"
                Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(path)
                frames[view_id][f"{phase}_frame_path"] = str(path)
                frames[view_id][f"{phase}_physics_step"] = step
            return step

        initial_step = save_frames("initial")
        for view_id, path in (("external", external_path), ("wrist", wrist_path)):
            required_points = (
                entity_points
                if view_id == "external"
                else {key: entity_points[key] for key in ("spraycan", "tray")}
            )
            frames[view_id]["required_entities_projected_in_frame"] = _project_world_points(
                camera_to_world=camera_matrix(path),
                points=required_points,
                width=640,
                height=480,
                vfov_deg=float(spec["cameras"][view_id]["vertical_fov_deg"]),
            )

        robot.set_joint_positions(commanded_joints)
        for _ in range(20):
            world.step(render=True)
        commanded_step = save_frames("commanded")
        wrist_world_commanded = camera_matrix(wrist_path)
        wrist_local_commanded = _matrix_array(
            UsdGeom.Xformable(stage.GetPrimAtPath(wrist_path)).GetLocalTransformation()
        )

        missing = [
            row["relative_path"]
            for row in manifest.get("files") or []
            if not (assets_root / str(row.get("relative_path") or "")).is_file()
        ]
        dof_count = getattr(robot, "num_dof", None)
        if dof_count is None:
            dof_count = getattr(robot, "num_dofs", -1)
        return {
            "isaac_sim_major_version": 6,
            "scene_loaded": stage.GetPrimAtPath("/World/WarehouseWorkcell").IsValid(),
            "missing_dataset_local_dependencies": missing,
            "runtime_asset_relocations": relocation_evidence,
            "franka_dof_count": int(dof_count),
            "franka_dof_names": list(robot.dof_names),
            "spraycan_collision_mesh_count": collision_count,
            "spraycan_runtime_rigid_body": spray_prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "spraycan_kinematic_for_camera_canary": True,
            "views": frames,
            "wrist_camera_world_displacement_m": float(
                np.linalg.norm(wrist_world_commanded[3, :3] - wrist_world_initial[3, :3])
            ),
            "wrist_camera_local_transform_delta": float(
                np.max(np.abs(wrist_local_commanded - wrist_local_initial))
            ),
            "external_wrist_timestamp_pairs_exact": all(
                frames["external"][f"{phase}_physics_step"]
                == frames["wrist"][f"{phase}_physics_step"]
                for phase in ("initial", "commanded")
            ),
            "initial_physics_step": initial_step,
            "commanded_physics_step": commanded_step,
            "rankings_or_policy_outcomes_accessed": False,
        }
    finally:
        simulation_app.close()


def _validated_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != CANARY_SPEC_SCHEMA_VERSION:
        raise ValueError("nvidia_warehouse_native_camera_canary_spec_invalid")
    spec = dict(value)
    declared = spec.pop("spec_sha256", None)
    if declared != canonical_sha256(spec):
        raise ValueError("nvidia_warehouse_native_camera_canary_spec_sha256_invalid")
    spec["spec_sha256"] = declared
    return spec


def _image_evidence(path_value: Any, expected_resolution: list[int]) -> dict[str, Any]:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("nvidia_warehouse_native_camera_frame_missing_or_unsafe")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "resolution": [width, height],
        "spatial_std": float(np.std(rgb)),
        "nonblank": bool(float(np.std(rgb)) >= MIN_NONBLANK_SPATIAL_STD),
        "resolution_matches": [width, height] == expected_resolution,
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_safe_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_evidence(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe_evidence(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def assess_native_camera_backend_result(
    *, spec: Mapping[str, Any], backend_result: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    if backend_result.get("isaac_sim_major_version") != 6:
        blockers.append("isaac_sim_major_version_not_6")
    if backend_result.get("scene_loaded") is not True:
        blockers.append("native_workcell_scene_not_loaded")
    if backend_result.get("missing_dataset_local_dependencies") not in ([], ()):
        blockers.append("native_workcell_dataset_local_dependencies_missing")
    if backend_result.get("franka_dof_count") != 9:
        blockers.append("native_franka_dof_count_invalid")
    if backend_result.get("spraycan_collision_mesh_count", 0) < 1:
        blockers.append("native_spraycan_collision_missing")
    if backend_result.get("spraycan_runtime_rigid_body") is not True:
        blockers.append("native_spraycan_rigid_body_missing")

    views_value = backend_result.get("views")
    views = views_value if isinstance(views_value, Mapping) else {}
    view_evidence: dict[str, Any] = {}
    for view_id in REQUIRED_VIEWS:
        view = views.get(view_id)
        if not isinstance(view, Mapping):
            blockers.append(f"native_camera_view_missing:{view_id}")
            continue
        expected = list(spec["cameras"][view_id]["resolution"])
        frames = {}
        for phase in ("initial", "commanded"):
            try:
                frame = _image_evidence(view.get(f"{phase}_frame_path"), expected)
            except (OSError, ValueError):
                blockers.append(f"native_camera_frame_invalid:{view_id}:{phase}")
                continue
            frames[phase] = frame
            if not frame["nonblank"]:
                blockers.append(f"native_camera_frame_blank:{view_id}:{phase}")
            if not frame["resolution_matches"]:
                blockers.append(f"native_camera_resolution_invalid:{view_id}:{phase}")
        projected = view.get("required_entities_projected_in_frame")
        if not isinstance(projected, Mapping) or not all(projected.values()):
            blockers.append(f"native_camera_required_entity_projection_failed:{view_id}")
        view_evidence[view_id] = {
            "frames": frames,
            "required_entities_projected_in_frame": dict(projected or {}),
        }

    wrist_world_delta = _finite_float(
        backend_result.get("wrist_camera_world_displacement_m")
    )
    wrist_local_delta = _finite_float(
        backend_result.get("wrist_camera_local_transform_delta")
    )
    if wrist_world_delta is None:
        blockers.append("native_wrist_camera_world_displacement_missing_or_invalid")
    elif wrist_world_delta <= MIN_WRIST_WORLD_DISPLACEMENT_M:
        blockers.append("native_wrist_camera_did_not_move_with_hand")
    if wrist_local_delta is None:
        blockers.append("native_wrist_camera_local_transform_missing_or_invalid")
    elif wrist_local_delta > MAX_WRIST_LOCAL_TRANSFORM_DELTA:
        blockers.append("native_wrist_camera_mount_not_rigid")
    if backend_result.get("external_wrist_timestamp_pairs_exact") is not True:
        blockers.append("native_camera_timestamps_not_synchronized")

    return {
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "views": view_evidence,
        "wrist_camera_world_displacement_m": wrist_world_delta,
        "wrist_camera_world_displacement_min_m": MIN_WRIST_WORLD_DISPLACEMENT_M,
        "wrist_camera_local_transform_delta": wrist_local_delta,
        "wrist_camera_local_transform_delta_max": MAX_WRIST_LOCAL_TRANSFORM_DELTA,
        "camera_motion_and_mount_checks_passed": not any(
            blocker.startswith("native_wrist_camera") for blocker in blockers
        ),
    }


def run_native_camera_canary(
    *,
    spec_path: str | Path,
    assets_root: str | Path,
    output_dir: str | Path,
    backend: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one label-free native camera canary through an injected Isaac backend."""

    spec = _validated_spec(spec_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("nvidia_warehouse_native_camera_canary_output_exists")
    output.mkdir(parents=True)
    backend_result = backend(
        spec=spec,
        assets_root=Path(assets_root).expanduser().resolve(),
        output_dir=output / "runtime",
    )
    if not isinstance(backend_result, Mapping):
        raise ValueError("nvidia_warehouse_native_camera_backend_result_invalid")
    assessment = assess_native_camera_backend_result(spec=spec, backend_result=backend_result)
    for view in assessment["views"].values():
        for frame in view["frames"].values():
            try:
                frame["relative_path"] = (
                    Path(str(frame["path"])).resolve().relative_to(output).as_posix()
                )
            except (KeyError, ValueError):
                assessment["blockers"].append(
                    "native_camera_frame_outside_canary_output"
                )
    assessment["blockers"] = sorted(set(assessment["blockers"]))
    assessment["status"] = "passed" if not assessment["blockers"] else "failed"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": assessment["status"],
        "blockers": assessment["blockers"],
        "spec_path": str(Path(spec_path).expanduser().resolve()),
        "spec_sha256": spec["spec_sha256"],
        "assets_root": str(Path(assets_root).expanduser().resolve()),
        "backend_result": _json_safe_evidence(backend_result),
        "assessment": assessment,
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_policy_or_wam_model_invoked": False,
        "claim_boundary": {
            "native_scene_and_camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json(output / "native_camera_canary_result.json", result)
    return result


def main() -> int:  # pragma: no cover - requires the pinned Isaac GPU image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_native_camera_canary(
        spec_path=args.spec,
        assets_root=args.assets_root,
        output_dir=args.output_dir,
        backend=isaac_sim_6_backend,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 1


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "assess_native_camera_backend_result",
    "isaac_sim_6_backend",
    "run_native_camera_canary",
]


if __name__ == "__main__":
    raise SystemExit(main())
