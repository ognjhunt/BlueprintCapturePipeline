"""Preserve the retained DROID configuration across the two Arena releases.

The comparison ships the unchanged Isaac 6.0 robot and textures. The original
USD's missing Rubber_Textured.mdl reference remains an explicit evidence gap;
OmniPBR.mdl is supplied by each pinned renderer. Neither is silently replaced.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

ASSET_DIRECTORY = "retained_droid_assets"
ASSET_SOURCE_ROOT = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
ROBOT_PATH = "IsaacLab/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
ASSET_HASHES = {
    ROBOT_PATH: "190810a1810907d6819b2e2b0e009bc0f06fb912be6ac801de4999ae5be10e98",
    "Props/YCB/Axis_Aligned/Materials/Textures/024_bowl_COLOR.png":
        "bc53e093e21cbd6640413a12a548f4d3e3544b9eeca5f08a2cc3f40aada43db4",
    "Props/YCB/Axis_Aligned/Materials/Textures/011_banana_COLOR.png":
        "841dd1e67d23815cfaac3a14019c4dd8712d956664dab853a4d558353eba3795",
}
MATERIAL_GAP = "Materials/2023_1/Base/Plastics/Rubber_Textured.mdl"
ACTUATORS = {
    "panda_shoulder": dict(joint_names_expr=["panda_joint[1-4]"], effort_limit=87.0,
                           velocity_limit=2.175, stiffness=400.0, damping=80.0),
    "panda_forearm": dict(joint_names_expr=["panda_joint[5-7]"], effort_limit=12.0,
                          velocity_limit=2.61, stiffness=400.0, damping=80.0),
    "gripper": dict(joint_names_expr=["finger_joint"], stiffness=None, damping=None,
                    velocity_limit=1.0),
}
FRAMES = (
    ("end_effector", "{ENV_REGEX_NS}/Robot/panda_link0", (0.0, 0.0, 0.1034)),
    ("tool_rightfinger", "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/right_inner_finger", (0.0, 0.0, 0.046)),
    ("tool_leftfinger", "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger", (0.0, 0.0, 0.046)),
)
DROID_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))


def comparison_robot_asset_inputs(request_path: str | Path) -> dict[str, Path]:
    """Bind already retained bytes; bundle construction never fetches assets."""
    root = Path(request_path).resolve().parent / ASSET_DIRECTORY
    result = {}
    for relative, digest in ASSET_HASHES.items():
        path = root / relative
        if (path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError("runtime_comparison_robot_asset_missing_or_changed:" + relative)
        result[f"{ASSET_DIRECTORY}/{relative}"] = path
    if (root / MATERIAL_GAP).exists():
        raise ValueError("runtime_comparison_original_material_gap_changed")
    return result


def droid_constructor_joint_pose(reset: Mapping[str, float]) -> list[float] | None:
    """Apply an arm-only reset through events, not GA's full-articulation setter.

    GA's constructor setter requires thirteen joints. The retained policy input
    supplies seven arm joints; inventing six gripper positions would change it.
    The named spawn state is bound by the arena builder after construction.
    Full reset vectors retain the existing constructor path.
    """
    return None if tuple(reset) == DROID_ARM_JOINT_NAMES else list(reset.values())


def configure_droid_reset(embodiment: Any, reset: Mapping[str, float]) -> None:
    """Keep the legacy reset event, including its ordering and supplied vector."""
    events = embodiment.event_config
    if not hasattr(events, "init_franka_arm_pose"):
        # GA moved reset defaults into init_state. Reintroduce the retained
        # event before randomization, rather than changing reset semantics.
        from isaaclab.managers import EventTermCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_arena.embodiments.franka.franka import franka_stack_events

        if ((len(reset) != 13 and tuple(reset) != DROID_ARM_JOINT_NAMES)
                or not callable(getattr(embodiment, "set_initial_joint_pose", None))):
            raise ValueError("runtime_comparison_droid_reset_api_unrecognized")

        @configclass
        class RetainedDroidEvents:
            init_franka_arm_pose = EventTermCfg(
                func=franka_stack_events.set_default_joint_pose, mode="reset",
                params={"default_pose": list(reset.values())})
            randomize_franka_joint_state = events.randomize_franka_joint_state

        embodiment.event_config = events = RetainedDroidEvents()
    events.init_franka_arm_pose.params["default_pose"] = list(reset.values())
    events.randomize_franka_joint_state.params["mean"] = 0.0
    events.randomize_franka_joint_state.params["std"] = 0.0


def freeze_comparison_droid(embodiment: Any, *, bundle_root: str | Path | None,
                            robot_id: str) -> dict[str, Any] | None:
    if bundle_root is None:
        return None
    request = Path(bundle_root).parent / "runtime_inputs/native_task_runtime_comparison_request.v1.json"
    if not request.is_file():
        return None
    if robot_id != "franka_panda":
        raise ValueError("runtime_comparison_robot_unadmitted")
    from isaaclab.actuators import ImplicitActuatorCfg

    assets = comparison_robot_asset_inputs(request)
    scene = embodiment.scene_config
    scene.robot.spawn.usd_path = str(assets[f"{ASSET_DIRECTORY}/{ROBOT_PATH}"])
    scene.stand = None
    # Both releases accept the legacy aliases. Explicitly constructing fresh
    # configs avoids inheriting GA's different gripper limit or other overrides.
    scene.robot.actuators = {name: ImplicitActuatorCfg(**values) for name, values in ACTUATORS.items()}
    frames = {frame.name: frame for frame in scene.ee_frame.target_frames}
    if set(frames) != {row[0] for row in FRAMES}:
        raise ValueError("runtime_comparison_droid_frames_unrecognized")
    for name, path, offset in FRAMES:
        frames[name].prim_path, frames[name].offset.pos = path, offset
    for name in ("external_camera", "external_camera_2"):
        getattr(embodiment.camera_config, name).prim_path = "{ENV_REGEX_NS}/Robot/" + name
    readback = {
        "asset_source_root": ASSET_SOURCE_ROOT,
        "asset_hashes": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                         for key, path in assets.items()},
        "stand": scene.stand,
        "actuators": {name: {key: getattr(cfg, key) for key in ACTUATORS[name]}
                      for name, cfg in scene.robot.actuators.items()},
        "frames": [{"name": name, "prim_path": frames[name].prim_path,
                    "offset": list(frames[name].offset.pos)} for name, _, _ in FRAMES],
        "camera_parents": {name: getattr(embodiment.camera_config, name).prim_path
                           for name in ("external_camera", "external_camera_2")},
        "reset_pose_vector": list(embodiment.event_config.init_franka_arm_pose.params["default_pose"]),
        "reset_noise": dict(embodiment.event_config.randomize_franka_joint_state.params),
        "retained_asset_gaps": [MATERIAL_GAP],
        "renderer_supplied_assets": ["OmniPBR.mdl"],
    }
    # SceneEntityCfg is runtime-specific; only the unchanged noise numbers are
    # part of this portable configuration receipt.
    readback["reset_noise"] = {key: readback["reset_noise"][key] for key in ("mean", "std")}
    readback["configuration_digest"] = canonical_digest(readback)
    return readback
