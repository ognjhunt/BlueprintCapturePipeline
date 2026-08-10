"""Build the native Isaac Lab Arena environment from a sealed scene plan.

The public compiler in :mod:`native_task_arena_scene_plan` performs all
filesystem and task binding decisions off-GPU.  This module is the deliberately
thin native adapter: it maps that plan onto the pinned Arena APIs and returns
the exact scene handles needed by the episode/readback adapter.  It contains no
scene id, object label, canned-beverage constant, or refrigerator coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
PINHOLE_HORIZONTAL_APERTURE_MM = 20.955
SCENE_PLAN_SCHEMA = "native_task_arena_scene_plan.v1"


class NativeTaskArenaRuntimeError(ValueError):
    """Stable configuration failures at the native adapter boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class NativeTaskArenaEnvironment:
    env: Any
    cfg: Any
    plan: Mapping[str, Any]
    scene_asset_names: Mapping[str, str]
    contact_sensor_names: Mapping[str, tuple[str, ...]]
    camera_scene_names: Mapping[str, str]
    preconstruction_device_binding: Mapping[str, Any] | None = None


def _validated_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        plan = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_plan_invalid"]
        ) from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != SCENE_PLAN_SCHEMA:
        raise NativeTaskArenaRuntimeError(["native_task_arena_runtime_plan_invalid"])
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_plan_digest_invalid"]
        )
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _resolve_portable_assets(
    plan: Mapping[str, Any], *, bundle_root: str | Path | None
) -> list[dict[str, Any]]:
    """Resolve and reverify relative packet assets without changing the seal."""

    objects = json.loads(json.dumps(plan["objects"]))
    relative = [row for row in objects if not Path(str(row["usd_path"])).is_absolute()]
    if not relative:
        return objects
    if bundle_root is None:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_bundle_root_required"]
        )
    raw_root = Path(bundle_root).expanduser()
    if raw_root.is_symlink():
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_bundle_root_invalid"]
        )
    root = raw_root.resolve()
    if not root.is_dir():
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_runtime_bundle_root_invalid"]
        )
    errors: list[str] = []
    for row in relative:
        role = str(row.get("semantic_role") or "")
        relative_path = str(row["usd_path"])
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts or not pure.name:
            errors.append(f"native_task_arena_runtime_asset_path_invalid:{role}")
            continue
        candidate = root.joinpath(*pure.parts)
        resolved = candidate.resolve()
        outside = resolved != root and root not in resolved.parents
        try:
            expected_size = int(row["size_bytes"])
            expected_digest = str(row["sha256"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"native_task_arena_runtime_asset_identity_invalid:{role}")
            continue
        if (
            _has_symlink_component(candidate, root=root)
            or outside
            or not resolved.is_file()
        ):
            errors.append(f"native_task_arena_runtime_asset_missing:{role}")
            continue
        if resolved.stat().st_size != expected_size or _sha256(resolved) != expected_digest:
            errors.append(f"native_task_arena_runtime_asset_identity_mismatch:{role}")
            continue
        row["usd_path"] = str(resolved)
    if errors:
        raise NativeTaskArenaRuntimeError(errors)
    return objects


def _rotation_matrix_to_xyzw(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Convert a proper 3x3 rotation to a canonical XYZW quaternion."""

    m00, m01, m02 = (float(value) for value in matrix[0])
    m10, m11, m12 = (float(value) for value in matrix[1])
    m20, m21, m22 = (float(value) for value in matrix[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 0.0:
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_camera_rotation_invalid"]
        )
    quaternion = [qx / norm, qy / norm, qz / norm, qw / norm]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return quaternion


def camera_runtime_parameters(camera: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one calibrated OpenCV pose/intrinsics row to Isaac CameraCfg data."""

    role = str(camera.get("role") or "")
    matrix = list(camera.get("frame_from_camera_matrix") or [])
    intrinsics = dict(camera.get("intrinsics") or {})
    if len(matrix) != 16 or camera.get("optical_convention") != "opencv":
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_contract_invalid:{role}"]
        )
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_intrinsics_invalid:{role}"]
        ) from exc
    if (
        not math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6)
        or not math.isclose(cx, (width - 1) / 2.0, abs_tol=1e-6)
        or not math.isclose(cy, (height - 1) / 2.0, abs_tol=1e-6)
    ):
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_intrinsics_not_representable:{role}"]
        )
    rotation = [matrix[0:3], matrix[4:7], matrix[8:11]]
    pose_frame = str(camera.get("pose_frame") or "")
    parent = str(camera.get("parent_prim_path") or "")
    expected_frame = "robot_body" if role == "wrist" else "world"
    if (
        pose_frame != expected_frame
        or not parent
        or (pose_frame == "world" and parent != "{ENV_REGEX_NS}")
        or (
            pose_frame == "robot_body"
            and not parent.startswith("{ENV_REGEX_NS}/Robot/")
        )
    ):
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_parent_invalid:{role}"]
        )
    runtime_name = {
        "external": "external_camera",
        "wrist": "wrist_camera",
        "overview": "external_camera_2",
    }.get(role)
    if runtime_name is None:
        raise NativeTaskArenaRuntimeError(
            [f"native_task_arena_camera_role_invalid:{role}"]
        )
    return {
        "role": role,
        "runtime_name": runtime_name,
        "prim_path": f"{parent}/{runtime_name}",
        "pose_frame": pose_frame,
        "parent_prim_path": parent,
        "offset_position_m": [matrix[3], matrix[7], matrix[11]],
        "offset_rotation_xyzw": _rotation_matrix_to_xyzw(rotation),
        # Isaac Lab names the OpenCV optical frame convention "ros": +Z
        # forward, +X right, +Y down.  The plan retains the source name.
        "isaac_offset_convention": "ros",
        "source_optical_convention": "opencv",
        "width": width,
        "height": height,
        "focal_length_mm": fx * PINHOLE_HORIZONTAL_APERTURE_MM / width,
        "horizontal_aperture_mm": PINHOLE_HORIZONTAL_APERTURE_MM,
        "vertical_aperture_mm": PINHOLE_HORIZONTAL_APERTURE_MM * height / width,
        "data_types": (
            ["rgb", "distance_to_camera", "semantic_segmentation"]
            if role in {"external", "wrist"}
            else ["rgb", "semantic_segmentation"]
        ),
        "policy_input": bool(camera["policy_input"]),
        "review_only": bool(camera["review_only"]),
    }


def build_native_task_arena_environment(
    scene_plan: Mapping[str, Any],
    *,
    device: str = "cuda:0",
    bundle_root: str | Path | None = None,
    preconstruction_receipt: Mapping[str, Any] | None = None,
) -> NativeTaskArenaEnvironment:
    """Instantiate the pinned Arena environment from one immutable plan."""

    plan = _validated_plan(scene_plan)
    runtime_objects = _resolve_portable_assets(plan, bundle_root=bundle_root)

    from blueprint_pipeline.native_task_arena_preconstruction import (
        prepare_native_task_arena_preconstruction,
        validate_native_task_arena_preconstruction_receipt,
    )

    if preconstruction_receipt is None:
        preconstruction_receipt = prepare_native_task_arena_preconstruction(
            expected_device=device
        )
    try:
        preconstruction = validate_native_task_arena_preconstruction_receipt(
            preconstruction_receipt, expected_device=device
        )
    except ValueError as exc:
        blockers = list(preconstruction_receipt.get("blockers") or [])
        raise NativeTaskArenaRuntimeError(
            blockers or ["native_task_arena_preconstruction_receipt_invalid"]
        ) from exc

    from blueprint_pipeline.native_task_arena_import_scope import (
        install_scoped_arena_embodiment,
    )

    install_scoped_arena_embodiment(str(plan["robot"]["robot_id"]))

    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.embodiments.droid.droid import (
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    class ConfigAsset(Asset):
        def __init__(
            self,
            *,
            name: str,
            object_cfg: Any,
            event_name: str | None = None,
            event_cfg: Any | None = None,
        ) -> None:
            super().__init__(name=name)
            self.object_cfg = object_cfg
            self.event_name = event_name
            self.event_cfg = event_cfg

        def get_object_cfg(self) -> tuple[str, Any]:
            return self.name, self.object_cfg

        def get_event_cfg(self) -> tuple[str, Any | None]:
            return self.event_name or self.name, self.event_cfg

    class SpawnerObject(Object):
        def __init__(self, *, name: str, prim_path: str, spawner_cfg: Any):
            self.spawner_cfg = spawner_cfg
            super().__init__(
                name=name,
                prim_path=prim_path,
                object_type=ObjectType.SPAWNER,
            )

    robot = plan["robot"]
    robot_pose = robot["base_pose_world"]
    embodiment = DroidAbsoluteJointPositionEmbodiment(
        enable_cameras=True,
        initial_pose=Pose(
            position_xyz=tuple(robot_pose["position_world_m"]),
            rotation_xyzw=tuple(robot_pose["orientation_xyzw"]),
        ),
        initial_joint_pose=list(robot["joint_reset_positions_rad"].values()),
    )
    exact_robot_reset = dict(robot["joint_reset_positions_rad"])
    embodiment.event_config.init_franka_arm_pose.params["default_pose"] = list(
        exact_robot_reset.values()
    )
    embodiment.event_config.randomize_franka_joint_state.params["mean"] = 0.0
    embodiment.event_config.randomize_franka_joint_state.params["std"] = 0.0
    embodiment.get_scene_cfg()
    embodiment.scene_config.stand = None
    embodiment.initial_pose = None
    embodiment.scene_config.robot.init_state = (
        embodiment.scene_config.robot.init_state.replace(joint_pos=exact_robot_reset)
    )
    embodiment.scene_config.robot.spawn.semantic_tags = [("class", "robot")]

    camera_names: dict[str, str] = {}
    for camera in plan["cameras"]:
        parameters = camera_runtime_parameters(camera)
        camera_cfg = getattr(embodiment.camera_config, parameters["runtime_name"])
        camera_cfg.prim_path = parameters["prim_path"]
        camera_cfg.offset.pos = tuple(parameters["offset_position_m"])
        camera_cfg.offset.rot = tuple(parameters["offset_rotation_xyzw"])
        camera_cfg.offset.convention = parameters["isaac_offset_convention"]
        camera_cfg.width = parameters["width"]
        camera_cfg.height = parameters["height"]
        camera_cfg.data_types = list(parameters["data_types"])
        camera_cfg.colorize_semantic_segmentation = False
        camera_cfg.update_period = 0.0
        camera_cfg.update_latest_camera_pose = True
        camera_cfg.spawn.focal_length = parameters["focal_length_mm"]
        camera_cfg.spawn.horizontal_aperture = parameters["horizontal_aperture_mm"]
        camera_cfg.spawn.vertical_aperture = parameters["vertical_aperture_mm"]
        camera_names[parameters["role"]] = parameters["runtime_name"]

    assets: list[Any] = []
    scene_asset_names: dict[str, str] = {}
    task_object: Any | None = None
    for row in runtime_objects:
        role = row["semantic_role"]
        spawn_addon: dict[str, Any] = {"visible": bool(row["visible"])}
        if role == "task_object":
            spawn_addon["semantic_tags"] = [("class", "task_object")]
        obj = Object(
            name=role,
            prim_path=row["prim_path"],
            object_type=ObjectType[row["object_type"]],
            usd_path=row["usd_path"],
            initial_pose=Pose(
                position_xyz=tuple(row["pose_world"]["position_world_m"]),
                rotation_xyzw=tuple(row["pose_world"]["orientation_xyzw"]),
            ),
            spawn_cfg_addon=spawn_addon,
        )
        if role == "task_object" and row["object_type"] == "ARTICULATION":
            obj.object_cfg.init_state = obj.object_cfg.init_state.replace(
                joint_pos=plan["reset"]["task_joint_positions_rad"]
            )
            task_object = obj
        assets.append(obj)
        scene_asset_names[role] = role

    def invalid_exact_contact_path(path: Any) -> bool:
        value = str(path)
        return not value.startswith("{ENV_REGEX_NS}/") or any(
            token in value for token in ("*", ".*", "[", "]")
        )

    contact_sensor_names_mutable: dict[str, list[str]] = {}
    seen_sensor_instances: set[str] = set()
    for index, sensor in enumerate(plan["articulation"]["contact_sensors"]):
        logical_sensor_id = str(sensor.get("logical_sensor_id") or "")
        sensor_instance_id = str(sensor.get("sensor_instance_id") or "")
        prim_path = str(sensor.get("prim_path") or "")
        filter_paths = list(sensor.get("filter_prim_paths_expr") or [])
        if (
            logical_sensor_id
            not in {
                "task_robot_contact",
                "task_scene_contact",
                "robot_scene_contact",
            }
            or not sensor_instance_id
            or sensor_instance_id in seen_sensor_instances
            or invalid_exact_contact_path(prim_path)
            or not filter_paths
            or any(invalid_exact_contact_path(path) for path in filter_paths)
        ):
            raise NativeTaskArenaRuntimeError(
                [f"native_task_arena_contact_sensor_contract_invalid:{index}"]
            )
        seen_sensor_instances.add(sensor_instance_id)
        event_name = None
        event_cfg = None
        if index == 0:
            if task_object is None:
                raise NativeTaskArenaRuntimeError(
                    ["native_task_arena_articulated_object_missing"]
                )
            event_name = "reset_task_object_joints"
            event_cfg = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("task_object"),
                },
            )
        assets.append(
            ConfigAsset(
                name=sensor_instance_id,
                object_cfg=ContactSensorCfg(
                    prim_path=prim_path,
                    filter_prim_paths_expr=filter_paths,
                ),
                event_name=event_name,
                event_cfg=event_cfg,
            )
        )
        contact_sensor_names_mutable.setdefault(logical_sensor_id, []).append(
            sensor_instance_id
        )
    expected_contact_channels = {
        "task_robot_contact",
        "task_scene_contact",
        "robot_scene_contact",
    }
    if plan["task_kind"] == "articulated_open_close" and (
        set(contact_sensor_names_mutable) != expected_contact_channels
    ):
        raise NativeTaskArenaRuntimeError(
            ["native_task_arena_contact_sensor_channels_incomplete"]
        )
    contact_sensor_names = {
        logical_id: tuple(scene_names)
        for logical_id, scene_names in sorted(contact_sensor_names_mutable.items())
    }

    assets.append(
        SpawnerObject(
            name="light",
            prim_path="/World/Light",
            spawner_cfg=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75), intensity=1500.0
            ),
        )
    )
    scene = Scene(assets=assets)
    cadence = plan["cadence"]

    def configure(cfg: Any) -> Any:
        from isaaclab_physx.physics import PhysxCfg

        # Arena applies this callback before parse_env_cfg/gym.make.  Bind the
        # same qualified device here so SimulationContext and PhysxManager
        # cannot silently diverge before the first reset.
        cfg.sim.device = str(preconstruction["expected_device"])
        cfg.sim.dt = cadence["physics_dt_seconds"]
        cfg.seed = int(plan["scenario"]["seed"])
        cfg.sim.render_interval = cadence["control_decimation"]
        cfg.decimation = cadence["control_decimation"]
        cfg.episode_length_s = cadence["episode_length_seconds"]
        cfg.sim.physics = PhysxCfg(
            solver_type=1,
            enable_enhanced_determinism=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**15,
        )
        return cfg

    arena_env = IsaacLabArenaEnvironment(
        name="Blueprint-Native-Task-Evaluation-v1",
        scene=scene,
        embodiment=embodiment,
        task=NoTask(),
        env_cfg_callback=configure,
    )
    builder = ArenaEnvBuilder(
        arena_env,
        argparse.Namespace(
            num_envs=1,
            env_spacing=2.0,
            solve_relations=False,
            placement_seed=int(plan["scenario"]["seed"]),
            mimic=False,
            device=device,
            disable_fabric=False,
            presets=None,
        ),
    )
    env, cfg = builder.make_registered_and_return_cfg(render_mode="rgb_array")
    return NativeTaskArenaEnvironment(
        env=env,
        cfg=cfg,
        plan=plan,
        scene_asset_names=scene_asset_names,
        contact_sensor_names=contact_sensor_names,
        camera_scene_names=camera_names,
        preconstruction_device_binding=preconstruction,
    )


__all__ = [
    "NativeTaskArenaEnvironment",
    "NativeTaskArenaRuntimeError",
    "build_native_task_arena_environment",
    "camera_runtime_parameters",
]
