"""Derive the bounded native OVRTX/OVPhysX probe for ADP-009B.

This module only materializes immutable inputs. The paid execution remains in
the canonical Content Agents Vast lane. InteriorGS appearance bytes are not
included: OVRTX renders the approved replacement alone and the returned AOVs
are composited over the separately retained Aura frames after provider return.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .common import ensure_dir, write_json


CAMERAS_RELATIVE_PATH = "inpainting_inputs/840313_ins160_v1/cameras.v1.json"
EXPECTED_CAMERA_IDS = (
    "approach_wide",
    "approach_close",
    "cabinet_context",
    "left_translate",
    "low_approach",
    "raised_left",
    "raised_right",
    "right_translate",
)
HORIZONTAL_APERTURE_MM = 20.955
OVRTX_QUALITY_STEPS = 256
OVRTX_VERSION = "0.4.0.346409"
OVSTAGE_VERSION = "0.1.0.346039"
OVPHYSX_VERSION = "0.4.13"
ISAAC_SIM_VERSION = "6.0.1"
ISAAC_PROBE_PROTOCOL_REVISION = "slide_stimulus_v2"
ISAAC_SLIDE_INITIAL_VELOCITY_MPS = 0.3
ISAAC_COLLIDER_APPROXIMATION = "convexHull"
ISAAC_COLLIDER_CLAIM_CEILING = "native_isaac_convex_hull_approximation_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def opencv_camera_to_usd_row_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Convert column-vector OpenCV camera-to-world into USD/Gf row storage.

    OpenCV camera axes are +X right, +Y down, +Z forward. A USD camera is +X
    right, +Y up and looks down -Z. Gf matrices serialize translation in the
    last row, hence the final transpose.
    """

    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ValueError("simready_native_camera_matrix_not_4x4")
    numeric = [[float(value) for value in row] for row in matrix]
    axis = (1.0, -1.0, -1.0, 1.0)
    usd_column = [
        [numeric[row][column] * axis[column] for column in range(4)]
        for row in range(4)
    ]
    return [[usd_column[column][row] for column in range(4)] for row in range(4)]


def _camera_config(camera: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = camera.get("intrinsics") or {}
    width = int(intrinsics.get("width") or 0)
    height = int(intrinsics.get("height") or 0)
    fx = float(intrinsics.get("fx") or 0.0)
    fy = float(intrinsics.get("fy") or 0.0)
    cx = float(intrinsics.get("cx") or 0.0)
    cy = float(intrinsics.get("cy") or 0.0)
    if width != 2048 or height != 1536 or fx <= 0 or fy <= 0:
        raise ValueError("simready_native_camera_intrinsics_invalid")
    if abs(fx - fy) > 1.0e-9:
        raise ValueError("simready_native_non_square_pixels_unsupported")
    vertical_aperture = HORIZONTAL_APERTURE_MM * height / width
    return {
        "camera_id": str(camera.get("camera_id") or ""),
        "camera_prim_path": "/BlueprintNative/Camera",
        "camera_transform_matrix_usd": opencv_camera_to_usd_row_matrix(
            camera.get("T_world_camera_opencv")
        ),
        "width": width,
        "height": height,
        "focal_length_mm": fx * HORIZONTAL_APERTURE_MM / width,
        "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
        "vertical_aperture_mm": vertical_aperture,
        "horizontal_aperture_offset_mm": (cx - width / 2.0)
        * HORIZONTAL_APERTURE_MM
        / width,
        "vertical_aperture_offset_mm": (height / 2.0 - cy)
        * vertical_aperture
        / height,
        "clipping_range": [0.01, 100.0],
        "render_mode": "PathTracing",
        "warmup_frames": 0,
        "quality_steps": 1,
        "path_tracing_samples_per_pixel": OVRTX_QUALITY_STEPS,
        "delta_time_seconds": 1.0 / 60.0,
        "_blueprint_required_checks": [],
    }


def _render_stage(asset_relative_path: str, placement: list[float]) -> str:
    if len(placement) != 3:
        raise ValueError("simready_native_placement_invalid")
    return f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "BlueprintReplacement" (
        prepend references = @{asset_relative_path}@</canned_beverage>
    )
    {{
        double3 xformOp:translate = ({float(placement[0])}, {float(placement[1])}, {float(placement[2])})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def DistantLight "KeyLight"
    {{
        float angle = 4
        float intensity = 3500
        color3f color = (1, 0.97, 0.93)
        float3 xformOp:rotateXYZ = (35, 0, -35)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

    def DistantLight "FillLight"
    {{
        float angle = 8
        float intensity = 1200
        color3f color = (0.85, 0.92, 1)
        float3 xformOp:rotateXYZ = (65, 0, 145)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}
'''


def _drop_stage(
    composition_relative_path: str,
    placement: list[float],
    support_collider_path: str,
) -> str:
    support_prefix = "/World/Environment/"
    if not support_collider_path.startswith(support_prefix):
        raise ValueError("simready_native_support_collider_path_unsupported")
    support_name = support_collider_path.removeprefix(support_prefix)
    if not support_name or "/" in support_name or '"' in support_name:
        raise ValueError("simready_native_support_collider_path_unsupported")
    drop = [float(placement[0]), float(placement[1]), float(placement[2]) + 0.05]
    return f'''#usda 1.0
(
    subLayers = [@{composition_relative_path}@]
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

over "World"
{{
    def PhysicsScene "physics_scene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    over "BlueprintReplacement" (
        prepend apiSchemas = ["PhysxContactReportAPI"]
    )
    {{
        double3 xformOp:translate = ({drop[0]}, {drop[1]}, {drop[2]})
        uniform token[] xformOpOrder = ["xformOp:translate"]

        over "colliders"
        {{
            over "body_collider"
            {{
                uniform token physics:approximation = "{ISAAC_COLLIDER_APPROXIMATION}"
            }}
        }}
    }}

    over "Environment"
    {{
        over "{support_name}"
        {{
            uniform token physics:approximation = "none"
        }}
    }}
}}
'''


def _isaac_motion_stage(
    composition_relative_path: str,
    placement: list[float],
    support_collider_path: str,
    *,
    linear_velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    support_prefix = "/World/Environment/"
    if not support_collider_path.startswith(support_prefix):
        raise ValueError("simready_native_support_collider_path_unsupported")
    support_name = support_collider_path.removeprefix(support_prefix)
    if not support_name or "/" in support_name or '"' in support_name:
        raise ValueError("simready_native_support_collider_path_unsupported")
    values = [float(value) for value in placement]
    return f'''#usda 1.0
(
    subLayers = [@{composition_relative_path}@]
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

over "World"
{{
    def PhysicsScene "physics_scene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    over "BlueprintReplacement" (
        prepend apiSchemas = ["PhysxContactReportAPI"]
    )
    {{
        float3 physics:angularVelocity = {angular_velocity}
        float3 physics:velocity = {linear_velocity}
        float3 xformOp:rotateXYZ = {rotation_degrees}
        double3 xformOp:translate = ({values[0]}, {values[1]}, {values[2]})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

        over "colliders"
        {{
            over "body_collider"
            {{
                uniform token physics:approximation = "{ISAAC_COLLIDER_APPROXIMATION}"
            }}
        }}
    }}

    over "Environment"
    {{
        over "{support_name}"
        {{
            uniform token physics:approximation = "none"
        }}
    }}
}}
'''


def _isaac_gripper_stage(
    composition_relative_path: str,
    placement: list[float],
    support_collider_path: str,
) -> str:
    support_prefix = "/World/Environment/"
    if not support_collider_path.startswith(support_prefix):
        raise ValueError("simready_native_support_collider_path_unsupported")
    support_name = support_collider_path.removeprefix(support_prefix)
    if not support_name or "/" in support_name or '"' in support_name:
        raise ValueError("simready_native_support_collider_path_unsupported")
    x, y, z = (float(value) for value in placement)
    finger_z = z + 0.084713995
    return f'''#usda 1.0
(
    subLayers = [@{composition_relative_path}@]
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

over "World"
{{
    def PhysicsScene "physics_scene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    over "BlueprintReplacement" (
        prepend apiSchemas = ["PhysxContactReportAPI"]
    )
    {{
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        over "colliders"
        {{
            over "body_collider"
            {{
                uniform token physics:approximation = "{ISAAC_COLLIDER_APPROXIMATION}"
            }}
        }}
    }}

    def Xform "BlueprintProbeGripper"
    {{
        def Cube "left_finger" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI", "PhysxContactReportAPI"]
        )
        {{
            bool physics:kinematicEnabled = 1
            bool physics:rigidBodyEnabled = 1
            double size = 1
            double3 xformOp:scale = (0.01, 0.02, 0.07)
            double3 xformOp:translate = ({x - 0.056}, {y}, {finger_z})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
        def Cube "right_finger" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI", "PhysxContactReportAPI"]
        )
        {{
            bool physics:kinematicEnabled = 1
            bool physics:rigidBodyEnabled = 1
            double size = 1
            double3 xformOp:scale = (0.01, 0.02, 0.07)
            double3 xformOp:translate = ({x + 0.056}, {y}, {finger_z})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}

    over "Environment"
    {{
        over "{support_name}"
        {{
            uniform token physics:approximation = "none"
        }}
    }}
}}
'''


def _usd_scene_inventory(
    path: Path,
    *,
    replacement_path: str,
    support_collider_path: str,
) -> dict[str, Any]:
    """Derive the small physics inventory outside the isolated OVPhysX runtime.

    OVPhysX owns native ingest and dynamics.  The separately pinned
    ``usd-exchange`` environment owns OpenUSD schema inspection so the two USD
    implementations are never imported into the same process.
    """

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError("simready_native_drop_stage_unopenable")
    rigid: list[str] = []
    colliders: list[str] = []
    joints: list[dict[str, Any]] = []
    masses: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    extent_points: list[list[float]] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        in_replacement = prim_path == replacement_path or prim_path.startswith(
            replacement_path + "/"
        )
        is_support = prim_path == support_collider_path
        if in_replacement and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid.append(prim_path)
        if (in_replacement or is_support) and prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(prim_path)
        if in_replacement and prim.IsA(UsdPhysics.Joint):
            joints.append(
                {
                    "path": prim_path,
                    "lower": prim.GetAttribute("physics:lowerLimit").Get(),
                    "upper": prim.GetAttribute("physics:upperLimit").Get(),
                }
            )
        if in_replacement and prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            masses.append(
                {
                    "path": prim_path,
                    "mass": api.GetMassAttr().Get(),
                    "density": api.GetDensityAttr().Get(),
                }
            )
        if in_replacement and prim.HasAPI(UsdPhysics.MaterialAPI):
            api = UsdPhysics.MaterialAPI(prim)
            materials.append(
                {
                    "path": prim_path,
                    "static_friction": api.GetStaticFrictionAttr().Get(),
                    "dynamic_friction": api.GetDynamicFrictionAttr().Get(),
                    "restitution": api.GetRestitutionAttr().Get(),
                }
            )
        if in_replacement:
            extent = prim.GetAttribute("extent").Get()
            if extent is not None and len(extent) == 2:
                extent_points.extend(
                    [[float(value) for value in point] for point in extent]
                )
    if extent_points:
        lower = [min(point[axis] for point in extent_points) for axis in range(3)]
        upper = [max(point[axis] for point in extent_points) for axis in range(3)]
        bounds = {
            "lower_m": lower,
            "upper_m": upper,
            "dimensions_m": [upper[axis] - lower[axis] for axis in range(3)],
        }
    else:
        bounds = {}
    inventory = {
        "authority": "blueprint_usd_exchange_2_3_schema_inspection",
        "source_sha256": _sha256(path),
        "replacement_path": replacement_path,
        "support_collider_path": support_collider_path,
        "rigid_bodies": sorted(rigid),
        "colliders": sorted(colliders),
        "joints": joints,
        "masses": masses,
        "materials": materials,
        "local_bounds": bounds,
    }
    if (
        replacement_path not in inventory["rigid_bodies"]
        or support_collider_path not in inventory["colliders"]
        or not any(path.startswith(replacement_path + "/") for path in colliders)
        or not masses
        or not materials
        or not bounds
    ):
        raise ValueError("simready_native_physics_inventory_incomplete")
    return inventory


def materialize_native_probe(
    *,
    evidence_root: str | Path,
    destination: str | Path,
    replacement_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy exact source bytes and derive sealed native probe configs."""

    evidence = Path(evidence_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    ensure_dir(target)
    cameras_source = (evidence / CAMERAS_RELATIVE_PATH).resolve()
    composition = replacement_receipt.get("composition") or {}
    composition_source = (evidence / str(composition.get("relative_path") or "")).resolve()
    asset_copy = composition.get("replacement_asset_copy") or {}
    asset_source = (evidence / str(asset_copy.get("relative_path") or "")).resolve()
    collision_copy = composition.get("sage_collision_copy") or {}
    collision_source = (evidence / str(collision_copy.get("relative_path") or "")).resolve()
    for path in (cameras_source, composition_source, asset_source, collision_source):
        if path != evidence and evidence not in path.parents:
            raise ValueError("simready_native_source_outside_evidence_root")
        if not path.is_file():
            raise ValueError("simready_native_source_missing")
    if (
        _sha256(composition_source) != composition.get("sha256")
        or _sha256(asset_source) != asset_copy.get("sha256")
        or _sha256(collision_source) != collision_copy.get("sha256")
    ):
        raise ValueError("simready_native_source_identity_mismatch")

    camera_payload = json.loads(cameras_source.read_text(encoding="utf-8"))
    if not isinstance(camera_payload, list):
        raise ValueError("simready_native_camera_manifest_invalid")
    configs = [_camera_config(camera) for camera in camera_payload]
    if tuple(sorted(item["camera_id"] for item in configs)) != tuple(
        sorted(EXPECTED_CAMERA_IDS)
    ):
        raise ValueError("simready_native_camera_set_mismatch")

    scene = target / "scene"
    ensure_dir(scene / "assets")
    shutil.copy2(composition_source, scene / "collision_and_replacement.usda")
    shutil.copy2(asset_source, scene / "assets" / asset_source.name)
    shutil.copy2(collision_source, scene / "assets" / collision_source.name)
    shutil.copy2(cameras_source, target / "cameras.v1.json")
    placement = [
        float(value)
        for value in (replacement_receipt.get("placement") or {}).get(
            "support_aligned_base_placement_m", []
        )
    ]
    render_stage = target / "render_stage.usda"
    render_stage.write_text(
        _render_stage(f"scene/assets/{asset_source.name}", placement), encoding="utf-8"
    )
    replacement_path = str(
        composition.get("composed_replacement_prim_path")
        or "/World/BlueprintReplacement"
    )
    support_collider_path = str(
        composition.get("composed_support_collision_prim_path") or ""
    )
    if not support_collider_path.startswith("/"):
        raise ValueError("simready_native_support_collider_path_missing")
    drop_stage = target / "drop_stage.usda"
    drop_stage.write_text(
        _drop_stage(
            "scene/collision_and_replacement.usda",
            placement,
            support_collider_path,
        ),
        encoding="utf-8",
    )
    usd_scene_inventory = _usd_scene_inventory(
        drop_stage,
        replacement_path=replacement_path,
        support_collider_path=support_collider_path,
    )
    configs_dir = target / "ovrtx_configs"
    ensure_dir(configs_dir)
    config_rows: list[dict[str, Any]] = []
    for config in configs:
        path = configs_dir / f"{config['camera_id']}.json"
        write_json(path, config)
        config_rows.append(
            {
                "camera_id": config["camera_id"],
                "relative_path": path.relative_to(target).as_posix(),
                "sha256": _sha256(path),
            }
        )
    physics_config = {
        "device": "gpu",
        "rigid_body_patterns": ["/World/BlueprintReplacement"],
        "penetration_sensor_patterns": ["/World/BlueprintReplacement"],
        "penetration_filter_patterns": [
            "/World/Environment/_LTFTHJVAZ3VMPTUJU888888"
        ],
        "filters_per_sensor": 1,
        "fixed_step_seconds": 1.0 / 120.0,
        "steps": 360,
        "snapshot_steps": [0, 120, 240, 330, 359],
        "maximum_initial_contact_force": 1.0e-3,
        "contact_force_threshold": 1.0e-3,
        "expected_joint_min_count": 0,
        "mass_bounds_kg": [0.1, 1.0],
        "friction_bounds": [0.0, 1.0],
        "usd_scene_inventory": usd_scene_inventory,
        "drop_contact_settle": {
            "expected_support_z_m": placement[2],
            "initial_drop_height_m": 0.05,
            "minimum_observed_drop_m": 0.025,
            "support_height_tolerance_m": 0.006,
            "settle_window_steps": 30,
            "maximum_settle_motion_m": 0.002,
            "maximum_rotation_from_initial_degrees": 5.0,
            "minimum_contact_steps": 1,
        },
    }
    physics_config_path = target / "ovphysx_config.json"
    write_json(physics_config_path, physics_config)
    isaac_stages = {
        "drop": drop_stage,
        "slide": target / "isaac_slide_stage.usda",
        "tip": target / "isaac_tip_stage.usda",
        "gripper": target / "isaac_gripper_stage.usda",
    }
    isaac_stages["slide"].write_text(
        _isaac_motion_stage(
            "scene/collision_and_replacement.usda",
            placement,
            support_collider_path,
            linear_velocity=(ISAAC_SLIDE_INITIAL_VELOCITY_MPS, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        ),
        encoding="utf-8",
    )
    isaac_stages["tip"].write_text(
        _isaac_motion_stage(
            "scene/collision_and_replacement.usda",
            placement,
            support_collider_path,
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 20.0, 0.0),
            rotation_degrees=(0.0, 6.0, 0.0),
        ),
        encoding="utf-8",
    )
    isaac_stages["gripper"].write_text(
        _isaac_gripper_stage(
            "scene/collision_and_replacement.usda",
            placement,
            support_collider_path,
        ),
        encoding="utf-8",
    )
    isaac_probe_spec = {
        "schema_version": "adp009b_simready_isaac_probe_spec.v1",
        "status": "frozen_before_execution",
        "protocol_revision": ISAAC_PROBE_PROTOCOL_REVISION,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "replacement_prim_path": replacement_path,
        "source_target_collider_path": "/World/Environment/ZHQYGJJVAJYEYPTUKY888888",
        "support_collider_path": support_collider_path,
        "expected_support_z_m": placement[2],
        "replacement_dimensions_m": usd_scene_inventory["local_bounds"]["dimensions_m"],
        "replacement_mass_kg": float(usd_scene_inventory["masses"][0]["mass"]),
        "collider_contract": {
            "runtime_authored_approximation": ISAAC_COLLIDER_APPROXIMATION,
            "source_asset_sdf_consumption_proven": False,
            "exact_source_collider_behavior_proven": False,
            "claim_ceiling": ISAAC_COLLIDER_CLAIM_CEILING,
        },
        "fixed_step_seconds": 1.0 / 120.0,
        "stimuli": {
            "drop": {"initial_linear_velocity_mps": [0.0, 0.0, 0.0]},
            "slide": {
                "initial_linear_velocity_mps": [
                    ISAAC_SLIDE_INITIAL_VELOCITY_MPS,
                    0.0,
                    0.0,
                ]
            },
            "tip": {"initial_linear_velocity_mps": [0.0, 0.0, 0.0]},
            "gripper": {"initial_linear_velocity_mps": [0.0, 0.0, 0.0]},
        },
        "stages": {
            name: {
                "relative_path": path.relative_to(target).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in isaac_stages.items()
        },
        "acceptance": {
            "drop": {
                "minimum_observed_drop_m": 0.025,
                "minimum_contact_events": 1,
                "maximum_support_height_error_m": 0.006,
                "maximum_settle_motion_m": 0.002,
            },
            "slide": {
                "minimum_horizontal_motion_m": 0.002,
                "maximum_horizontal_motion_m": 0.5,
                "maximum_support_height_error_m": 0.008,
            },
            "tip": {
                "minimum_perturbation_degrees": 5.0,
                "maximum_center_drop_m": 0.20,
                "maximum_support_height_error_m": 0.02,
            },
            "gripper": {
                "approach_clearance_m": 0.015,
                "closed_finger_gap_m": 0.060,
                "minimum_finger_contact_events": 1,
                "minimum_lift_m": 0.015,
                "release_required": True,
            },
        },
        "claim_boundaries": {
            "frozen_packet_is_not_execution": True,
            "bounded_two_finger_proxy_is_not_robot_task_success": True,
            "isaac_result_is_not_physical_truth": True,
            "generated_probe_uses_convex_hull_approximation": True,
            "exact_v2_sdf_collider_behavior_proven": False,
            "may_not_be_used_as_adp009d_exact_collider_evidence": True,
        },
    }
    isaac_probe_spec_path = target / "isaac_probe_spec.json"
    write_json(isaac_probe_spec_path, isaac_probe_spec)
    manifest = {
        "schema_version": "adp009b_simready_native_probe.v1",
        "status": "ready",
        "camera_manifest_sha256": _sha256(cameras_source),
        "composition_sha256": _sha256(composition_source),
        "replacement_asset_sha256": _sha256(asset_source),
        "sage_collision_sha256": _sha256(collision_source),
        "render_stage_sha256": _sha256(render_stage),
        "drop_stage_sha256": _sha256(drop_stage),
        "ovrtx": {
            "version": OVRTX_VERSION,
            "ovstage_version": OVSTAGE_VERSION,
            "license": "NVIDIA proprietary SDK license",
            "ovstage_license": "NVIDIA proprietary SDK license",
            "render_mode": "PathTracing",
            "quality_steps": 1,
            "path_tracing_samples_per_pixel": OVRTX_QUALITY_STEPS,
            "modalities": ["rgb", "depth"],
            "optional_modalities_not_required": ["normal"],
            "camera_count": len(config_rows),
            "camera_configs": config_rows,
        },
        "ovphysx": {
            "version": OVPHYSX_VERSION,
            "license": "BSD source plus NVIDIA binary terms",
            "config_sha256": _sha256(physics_config_path),
            "drop_height_m": 0.05,
            "expected_support_z_m": placement[2],
        },
        "isaac": {
            "version": ISAAC_SIM_VERSION,
            "protocol_revision": ISAAC_PROBE_PROTOCOL_REVISION,
            "probe_spec_sha256": _sha256(isaac_probe_spec_path),
            "stage_count": len(isaac_stages),
            "probe_names": sorted(isaac_stages),
            "status": "frozen_not_executed",
            "collider_approximation": ISAAC_COLLIDER_APPROXIMATION,
            "claim_ceiling": ISAAC_COLLIDER_CLAIM_CEILING,
            "exact_v2_sdf_collider_behavior_proven": False,
        },
        "claim_boundaries": {
            "ovrtx_object_only_render_not_3dgs_scene_render": True,
            "local_composite_required_after_provider_return": True,
            "ovphysx_probe_not_isaac_or_physical_truth": True,
            "isaac_probe_packet_is_not_isaac_execution": True,
            "legacy_isaac_probe_uses_convex_hull_approximation": True,
            "legacy_probe_may_not_substitute_for_adp009d_sdf_admission": True,
        },
    }
    write_json(target / "adp009b_simready_native_probe_manifest.json", manifest)
    return manifest


__all__ = [
    "EXPECTED_CAMERA_IDS",
    "OVRTX_QUALITY_STEPS",
    "OVRTX_VERSION",
    "OVSTAGE_VERSION",
    "materialize_native_probe",
    "opencv_camera_to_usd_row_matrix",
]
