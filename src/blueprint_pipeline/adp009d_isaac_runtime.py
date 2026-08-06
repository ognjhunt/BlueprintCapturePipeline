"""Native Isaac Lab/Arena runtime for the ADP-009D progressive micro-check.

This module is copied into an immutable provider bundle and executed with
``/isaac-sim/python.sh``.  Isaac imports intentionally happen only after
AppLauncher has started Kit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


RESULT_NAME = "adp009d_native_microcheck.json"
EXPECTED_ASSETS = {
    "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
    "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
}
APPROVED_CAN_ADAPTER_FILENAME = "approved_can_physx_sdf_adapter.usda"
APPROVED_CAN_ADAPTER_SHA256 = (
    "sha256:5db5bc33b72983065bd47e30db0c5945ab3cba8fb3caeb6290bf07edc7337adc"
)
APPROVED_CAN_SOURCE_COLLIDER_PRIM = "/canned_beverage/colliders/body_collider"
APPROVED_CAN_LIVE_COLLIDER_PRIM = "/World/envs/env_0/approved_can/colliders/body_collider"
PHYSX_FALLBACK_MARKER = "falling back to convexHull approximation"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ROBOT_BASE_POSITION_M = (3.4681748, -2.9100837, 0.2766791)
ROBOT_BASE_YAW_RAD = -math.pi / 2
CAN_START_POSITION_M = (3.4681748, -3.3100837, 0.5264650138348479)
RESET_JOINTS = (
    0.0,
    -0.569,
    0.0,
    -2.81,
    0.0,
    3.037,
    0.741,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _to_torch(value: Any) -> Any:
    """Convert simulator-native arrays at the adapter boundary before indexing."""

    if hasattr(value, "detach"):
        return value
    value_module = type(value).__module__
    if value_module == "warp" or value_module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{value_module}.{type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    value_module = type(value).__module__
    if value_module == "warp" or value_module.startswith("warp."):
        value = _to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _phase(name: str, status: str = "started") -> None:
    print(f"BLUEPRINT_WAM_RUNTIME_PHASE:adp009d_native:{name}:{status}", flush=True)


def _fail_on_physx_collision_fallback(messages: list[str]) -> None:
    if messages:
        raise RuntimeError(
            "physx_collision_fallback_detected:" + " | ".join(messages)
        )


def _inspect_physx_sdf_collider(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"physx_sdf_collider_prim_missing:{prim_path}")
    applied_schemas = [str(value) for value in prim.GetAppliedSchemas()]
    if "PhysxSDFMeshCollisionAPI" not in applied_schemas:
        raise RuntimeError(f"physx_sdf_schema_missing:{prim_path}")
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    approximation = mesh_api.GetApproximationAttr().Get() if mesh_api else None
    if str(approximation) != "sdf":
        raise RuntimeError(f"physx_sdf_approximation_invalid:{prim_path}:{approximation}")
    settings = {
        "sdf_margin": prim.GetAttribute("physxSDFMeshCollision:sdfMargin").Get(),
        "sdf_narrow_band_thickness": prim.GetAttribute(
            "physxSDFMeshCollision:sdfNarrowBandThickness"
        ).Get(),
        "sdf_resolution": prim.GetAttribute("physxSDFMeshCollision:sdfResolution").Get(),
        "sdf_subgrid_resolution": prim.GetAttribute(
            "physxSDFMeshCollision:sdfSubgridResolution"
        ).Get(),
    }
    if any(value is None for value in settings.values()):
        raise RuntimeError(f"physx_sdf_cooking_settings_missing:{prim_path}")
    return {
        "prim_path": prim_path,
        "applied_schemas": applied_schemas,
        "approximation": str(approximation),
        **settings,
    }


def _save_camera(output: Path, name: str, camera: Any, *, frame_index: int, sim_time: float) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    camera_output = camera.data.output
    required = {"rgb", "distance_to_camera", "semantic_segmentation"}
    missing = sorted(required - set(camera_output))
    if missing:
        raise RuntimeError(f"camera_outputs_missing:{name}:{','.join(missing)}")
    rgb = _to_torch(camera_output["rgb"])[0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = _to_torch(camera_output["distance_to_camera"])[0].detach().cpu().numpy().astype(np.float32)
    semantic = _to_torch(camera_output["semantic_segmentation"])[0].detach().cpu().numpy()
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    semantic = semantic.astype(np.int32)
    if rgb.shape[:2] != depth.shape[:2] or rgb.shape[:2] != semantic.shape[:2]:
        raise RuntimeError(f"camera_output_shape_mismatch:{name}")
    finite_depth = np.isfinite(depth)
    if not finite_depth.any() or (depth[finite_depth] < 0.0).any():
        raise RuntimeError(f"camera_metric_depth_invalid:{name}")
    camera_dir = output / "camera_frames" / name
    camera_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = camera_dir / f"{frame_index:06d}.png"
    depth_path = camera_dir / f"{frame_index:06d}.distance_to_camera.npy"
    semantic_path = camera_dir / f"{frame_index:06d}.semantic.npy"
    Image.fromarray(rgb, mode="RGB").save(rgb_path, format="PNG", compress_level=9)
    np.save(depth_path, depth, allow_pickle=False)
    np.save(semantic_path, semantic, allow_pickle=False)
    intrinsic = _to_torch(camera.data.intrinsic_matrices)[0]
    pos_w = _to_torch(camera.data.pos_w)[0]
    quat_w_opengl = _to_torch(camera.data.quat_w_opengl)[0]
    return {
        "camera_id": name,
        "frame_index": frame_index,
        "sim_time_seconds": sim_time,
        "timestamp_ns": time.time_ns(),
        "resolution_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "rgb_png": {"path": str(rgb_path.relative_to(output)), "sha256": _sha256(rgb_path)},
        "metric_depth": {
            "aov": "distance_to_camera",
            "units": "meter",
            "path": str(depth_path.relative_to(output)),
            "sha256": _sha256(depth_path),
        },
        "semantic_segmentation": {
            "path": str(semantic_path.relative_to(output)),
            "sha256": _sha256(semantic_path),
            "dtype": str(semantic.dtype),
        },
        "intrinsic_matrix": _jsonable(intrinsic),
        "position_world_m": _jsonable(pos_w),
        "quaternion_world_opengl_xyzw": _jsonable(quat_w_opengl),
        "device": str(camera.data.output["rgb"].device),
        "dlpack_ownership": "isaac_camera_tensor_read_only_copy_retained",
        "synchronization": "environment_step_completed_before_copy",
    }


def _build_environment(runtime: Path, args: argparse.Namespace):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    class SpawnerObject(Object):
        """Use Arena's composition seam without importing its full asset registry."""

        def __init__(self, *, name: str, prim_path: str, spawner_cfg: Any):
            self.spawner_cfg = spawner_cfg
            super().__init__(
                name=name,
                prim_path=prim_path,
                object_type=ObjectType.SPAWNER,
            )

    yaw_half = ROBOT_BASE_YAW_RAD / 2
    robot_pose = Pose(
        position_xyz=ROBOT_BASE_POSITION_M,
        rotation_xyzw=(0.0, 0.0, math.sin(yaw_half), math.cos(yaw_half)),
    )
    embodiment = DroidAbsoluteJointPositionEmbodiment(
        enable_cameras=True,
        initial_pose=robot_pose,
        initial_joint_pose=list(RESET_JOINTS),
    )
    embodiment.scene_config.robot.spawn.semantic_tags = [("class", "robot")]
    # The canonical anchor is immutable: no reset noise is permitted.
    embodiment.event_config.randomize_franka_joint_state = None
    # Apply the official pose helper while its stock stand still exists, then remove
    # that scene-specific stand.  The robot base is supported by sealed SAGE geometry.
    embodiment.get_scene_cfg()
    embodiment.scene_config.stand = None
    embodiment.initial_pose = None
    for camera_name in ("external_camera", "wrist_camera"):
        camera_cfg = getattr(embodiment.camera_config, camera_name)
        camera_cfg.data_types = ["rgb", "distance_to_camera", "semantic_segmentation"]
        camera_cfg.colorize_semantic_segmentation = False
        camera_cfg.update_period = 0.0
    # The second external camera is outside the frozen two-camera policy contract.
    embodiment.camera_config.external_camera_2 = None

    sage = Object(
        name="sage_collision",
        object_type=ObjectType.BASE,
        usd_path=str(runtime / "assets" / "sage_collision_overlay.usda"),
        initial_pose=Pose.identity(),
        spawn_cfg_addon={"visible": False},
    )
    approved_can = Object(
        name="approved_can",
        object_type=ObjectType.RIGID,
        usd_path=str(runtime / "assets" / APPROVED_CAN_ADAPTER_FILENAME),
        initial_pose=Pose(position_xyz=CAN_START_POSITION_M),
        spawn_cfg_addon={
            "semantic_tags": [("class", "approved_can")],
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
        },
    )
    light = SpawnerObject(
        name="light",
        prim_path="/World/Light",
        spawner_cfg=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1500.0,
        )
    )
    scene = Scene(assets=[sage, approved_can, light])

    def configure(cfg):
        from isaaclab_physx.physics import PhysxCfg

        cfg.sim.dt = 1.0 / 120.0
        cfg.seed = 20260806
        cfg.sim.render_interval = 8
        cfg.decimation = 8
        cfg.episode_length_s = 5.0
        cfg.sim.physics = PhysxCfg(
            solver_type=1,
            enable_enhanced_determinism=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**15,
        )
        return cfg

    arena_env = IsaacLabArenaEnvironment(
        name="Blueprint-ADP009D-Franka-Microcheck-v0",
        scene=scene,
        embodiment=embodiment,
        task=NoTask(),
        env_cfg_callback=configure,
    )
    builder_args = argparse.Namespace(
        num_envs=1,
        env_spacing=2.0,
        solve_relations=False,
        placement_seed=20260806,
        mimic=False,
        device=args.device,
        disable_fabric=False,
        presets=None,
    )
    builder = ArenaEnvBuilder(arena_env, builder_args)
    env, cfg = builder.make_registered_and_return_cfg(render_mode="rgb_array")
    return env, cfg, torch


def _run(runtime: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    for name, digest in EXPECTED_ASSETS.items():
        path = runtime / "assets" / name
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"sealed_asset_binding_invalid:{name}")

    adapter_path = runtime / "assets" / APPROVED_CAN_ADAPTER_FILENAME
    if not adapter_path.is_file() or _sha256(adapter_path) != APPROVED_CAN_ADAPTER_SHA256:
        raise RuntimeError("sealed_asset_binding_invalid:approved_can_physx_sdf_adapter.usda")
    from pxr import Usd

    adapter_stage = Usd.Stage.Open(str(adapter_path))
    if adapter_stage is None:
        raise RuntimeError("approved_can_physx_sdf_adapter_unreadable")
    static_collider = _inspect_physx_sdf_collider(
        adapter_stage, APPROVED_CAN_SOURCE_COLLIDER_PRIM
    )
    _phase("static_collider_validation", "completed")

    import omni.log

    fallback_messages: list[str] = []

    def on_log(
        channel, level, module, filename, func, line_no, message, pid, tid, timestamp
    ):
        del channel, level, module, filename, func, line_no, pid, tid, timestamp
        if PHYSX_FALLBACK_MARKER in message:
            fallback_messages.append(str(message))

    log = omni.log.get_log()
    consumer = log.add_message_consumer(on_log)
    env = None
    try:
        _phase("environment_build")
        env, cfg, torch = _build_environment(runtime, args)
        log.flush()
        _phase("environment_build", "completed")
        _fail_on_physx_collision_fallback(fallback_messages)
        import omni.usd

        live_stage = omni.usd.get_context().get_stage()
        live_collider = _inspect_physx_sdf_collider(
            live_stage, APPROVED_CAN_LIVE_COLLIDER_PRIM
        )
        _phase("live_collider_validation", "completed")
        reset_rows: list[dict[str, Any]] = []
        for index in range(2):
            _phase(f"reset_{index}")
            observation, info = env.reset(seed=20260806)
            robot = env.unwrapped.scene["robot"]
            approved_can = env.unwrapped.scene["approved_can"]
            reset_rows.append(
                {
                    "index": index,
                    "joint_pos": _jsonable(_to_torch(robot.data.joint_pos)[0]),
                    "can_root_pose_world": _jsonable(
                        _to_torch(approved_can.data.root_pose_w)[0]
                    ),
                    "observation_keys": sorted(str(key) for key in observation),
                    "info_keys": sorted(str(key) for key in (info or {})),
                }
            )
            log.flush()
            _fail_on_physx_collision_fallback(fallback_messages)
            _phase(f"reset_{index}", "completed")
        joint_a = torch.tensor(reset_rows[0]["joint_pos"])
        joint_b = torch.tensor(reset_rows[1]["joint_pos"])
        if not torch.equal(joint_a, joint_b):
            raise RuntimeError("canonical_reset_not_bitwise_reproducible")

        action = torch.zeros(
            (1, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        log.flush()
        _fail_on_physx_collision_fallback(fallback_messages)
        zero_action_row = {
            "action_dim": env.unwrapped.action_manager.total_action_dim,
            "reward": _jsonable(reward),
            "terminated": _jsonable(terminated),
            "truncated": _jsonable(truncated),
            "observation_keys": sorted(str(key) for key in observation),
            "robot_joint_pos_after_step": _jsonable(
                _to_torch(env.unwrapped.scene["robot"].data.joint_pos)[0]
            ),
            "approved_can_pose_after_step": _jsonable(
                _to_torch(env.unwrapped.scene["approved_can"].data.root_pose_w)[0]
            ),
        }
        env.reset(seed=20260806)
        robot = env.unwrapped.scene["robot"]
        hold_action = torch.zeros_like(action)
        hold_action[:, :7] = _to_torch(robot.data.joint_pos)[:, :7]
        for warmup_index in range(40):
            observation, reward, terminated, truncated, info = env.step(hold_action)
            if (warmup_index + 1) % 10 == 0:
                log.flush()
                _fail_on_physx_collision_fallback(fallback_messages)
                _phase(f"camera_warmup_{warmup_index + 1}", "completed")
        camera_rows = []
        for camera_name in ("external_camera", "wrist_camera"):
            camera_rows.append(
                _save_camera(
                    output,
                    camera_name,
                    env.unwrapped.scene[camera_name],
                    frame_index=40,
                    sim_time=float(env.unwrapped.episode_length_buf[0].item() * cfg.sim.dt * cfg.decimation),
                )
            )
        robot = env.unwrapped.scene["robot"]
        approved_can = env.unwrapped.scene["approved_can"]
        can_pose = _to_torch(approved_can.data.root_pose_w)[0]
        if not torch.isfinite(can_pose).all():
            raise RuntimeError("approved_can_state_nonfinite")
        if float(can_pose[2].item()) < CAN_START_POSITION_M[2] - 0.05:
            raise RuntimeError("approved_can_support_loss_after_zero_action")
        return {
            "schema_version": "adp009d_native_microcheck.v1",
            "status": "completed",
            "arena_revision": ARENA_REVISION,
            "isaac_lab_revision": ISAAC_LAB_REVISION,
            "workflow": "isaac_lab_manager_based_via_arena_composition",
            "embodiment": "official_arena_droid_abs_joint_pos_franka_robotiq_2f_85",
            "physics": {
                "backend": "PhysX",
                "dt_seconds": cfg.sim.dt,
                "decimation": cfg.decimation,
                "solver": "TGS",
                "enhanced_determinism": True,
                "static_collider_validation": static_collider,
                "live_collider_validation": live_collider,
                "fallback_messages": fallback_messages,
            },
            "reset_rows": reset_rows,
            "zero_action_step": {
                **zero_action_row,
            },
            "post_warmup_robot_joint_pos": _jsonable(
                _to_torch(robot.data.joint_pos)[0]
            ),
            "post_warmup_approved_can_root_pose_world": _jsonable(can_pose),
            "camera_frames": camera_rows,
            "camera_warmup_frames": 40,
            "source_target_collider_disabled_by_composed_overlay": True,
            "sealed_source_mutated": False,
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "blockers": [],
        }
    finally:
        log.remove_message_consumer(consumer)
        if env is not None:
            env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    app_launcher = AppLauncher(args)
    output = Path(args.output_dir).resolve()
    runtime = Path(args.runtime_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any]
    try:
        result = _run(runtime, output, args)
    except Exception as exc:
        result = {
            "schema_version": "adp009d_native_microcheck.v1",
            "status": "blocked",
            "blockers": [str(exc)],
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
        }
    result_path = output / RESULT_NAME
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app_launcher.app.close()
    print("BLUEPRINT_ADP009D_NATIVE_MICROCHECK_" + ("OK" if result["status"] == "completed" else "BLOCKED"))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
