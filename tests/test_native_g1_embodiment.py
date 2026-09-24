import hashlib
from copy import deepcopy

import pytest

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_embodiment import ACTION_INTERFACE, validate_g1_robot_plan
from blueprint_pipeline.native_task_arena_runtime import (
    camera_runtime_parameters,
    NativeTaskArenaRuntimeError,
)
from blueprint_pipeline.native_task_robot_registry import (
    NativeRobotAdapter,
    native_robot_adapter,
    register_native_robot_adapter,
)


def robot_plan(tmp_path):
    asset = tmp_path / "robot.usda"
    asset.write_text("#usda 1.0\n")
    return {
        "robot_id": "unitree_g1",
        "hand_id": "unitree_dex3_1",
        "action_interface": ACTION_INTERFACE,
        "usd_path": str(asset),
        "usd_sha256": "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
        "base_pose_world": {"position_world_m": [0, 0, 0.8], "orientation_xyzw": [0, 0, 0, 1]},
        "joint_reset_positions_rad": {name: 0.0 for name in PROTOCOL_V4_FULL_JOINT_ORDER},
        "joint_position_limits_rad": {name: [-1, 1] for name in PROTOCOL_V4_FULL_JOINT_ORDER},
        "actuator_parameters": {
            name: {"stiffness": 100, "damping": 2, "effort_limit": 25, "velocity_limit": 10}
            for name in PROTOCOL_V4_FULL_JOINT_ORDER
        },
    }


def test_reset_binds_all_body_and_hand_joints_and_exact_asset(tmp_path):
    plan = robot_plan(tmp_path)
    admitted = validate_g1_robot_plan(plan)
    assert len(admitted["joint_reset_positions_rad"]) == 43
    assert admitted == plan
    plan["joint_reset_positions_rad"].clear()
    assert len(admitted["joint_reset_positions_rad"]) == 43


@pytest.mark.parametrize(
    "case",
    [
        "missing_hand_joint",
        "wrong_hand",
        "droid_action",
        "bad_gain",
        "nan",
        "out_of_limits",
        "bad_quaternion",
        "changed_asset",
    ],
)
def test_rejects_bad_g1_spawn_before_importing_isaac(tmp_path, case):
    plan = robot_plan(tmp_path)
    name = PROTOCOL_V4_FULL_JOINT_ORDER[-1]
    if case == "missing_hand_joint":
        plan["joint_reset_positions_rad"].pop(name)
    elif case == "wrong_hand":
        plan["hand_id"] = "inspire"
    elif case == "droid_action":
        plan["action_interface"] = "droid_abs_joint"
    elif case == "bad_gain":
        plan["actuator_parameters"][name]["stiffness"] = -1
    elif case == "nan":
        plan["joint_reset_positions_rad"][name] = float("nan")
    elif case == "out_of_limits":
        plan["joint_reset_positions_rad"][name] = 2
    elif case == "bad_quaternion":
        plan["base_pose_world"]["orientation_xyzw"] = [0, 0, 0, 0]
    else:
        plan["usd_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="native_g1"):
        validate_g1_robot_plan(plan)


def test_g1_cameras_follow_robot_links_instead_of_franka_wrist():
    camera = {
        "policy_input": True,
        "review_only": False,
        "role": "head",
        "pose_frame": "robot_body",
        "parent_prim_path": "{ENV_REGEX_NS}/Robot/head_link",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "intrinsics": {"fx": 200, "fy": 200, "cx": 159.5, "cy": 89.5, "width": 320, "height": 180},
    }
    head = camera_runtime_parameters(camera, robot_id="unitree_g1")
    assert head["prim_path"] == "{ENV_REGEX_NS}/Robot/head_link/robot_head_cam"
    wrong = deepcopy(camera)
    wrong["pose_frame"] = "world"
    wrong["parent_prim_path"] = "{ENV_REGEX_NS}"
    with pytest.raises(NativeTaskArenaRuntimeError, match="camera_parent_invalid"):
        camera_runtime_parameters(wrong, robot_id="unitree_g1")
    with pytest.raises(NativeTaskArenaRuntimeError, match="camera_role_invalid"):
        camera_runtime_parameters(camera, robot_id="franka_panda")


def test_registry_cannot_silently_replace_an_existing_robot_adapter():
    g1 = native_robot_adapter("unitree_g1")
    assert g1.factory_name == "build_g1_embodiment"
    with pytest.raises(ValueError, match="already_registered"):
        register_native_robot_adapter(NativeRobotAdapter("unitree_g1", "other", "other", "other"))


def test_spawn_uses_g1_with_43_named_actions_and_no_franka_events(tmp_path, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    from blueprint_pipeline.native_g1_embodiment import build_g1_embodiment

    modules = {}
    for name in [
        "isaaclab",
        "isaaclab.envs",
        "isaaclab.envs.mdp",
        "isaaclab.actuators",
        "isaaclab.managers",
        "isaaclab.utils",
        "isaaclab_arena",
        "isaaclab_arena.embodiments",
        "isaaclab_arena.embodiments.g1",
        "isaaclab_arena.embodiments.g1.g1",
        "isaaclab_arena.utils",
        "isaaclab_arena.utils.cameras",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    mdp = modules["isaaclab.envs.mdp"]
    mdp.JointPositionActionCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    mdp.joint_pos, mdp.joint_vel = object(), object()
    mdp.reset_root_state_uniform, mdp.reset_joints_by_offset = object(), object()
    modules["isaaclab.actuators"].IdealPDActuatorCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    managers = modules["isaaclab.managers"]
    managers.EventTermCfg = managers.ObservationTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    managers.ObservationGroupCfg = type("ObservationGroupCfg", (), {})
    managers.SceneEntityCfg = lambda name: name
    modules["isaaclab.utils"].configclass = lambda cls: cls
    modules["isaaclab_arena.utils.cameras"].ArenaCameraCfg = type("ArenaCameraCfg", (), {})

    class G1:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.scene_config = SimpleNamespace(
                robot=SimpleNamespace(
                    spawn=SimpleNamespace(),
                    init_state=SimpleNamespace(joint_pos={"left_knee_joint": 99.0}),
                    actuators={},
                )
            )
            self.camera_config = SimpleNamespace(robot_head_cam=SimpleNamespace(prim_path="head"))

        def get_scene_cfg(self):
            return self.scene_config

    modules["isaaclab_arena.embodiments.g1.g1"].G1EmbodimentBase = G1
    plan = robot_plan(tmp_path)
    result = build_g1_embodiment(plan, enable_cameras=True, pose_class=lambda **kwargs: kwargs)
    assert isinstance(result, G1)
    assert result.scene_config.robot.spawn.usd_path == plan["usd_path"]
    assert result.scene_config.robot.init_state.joint_pos == plan["joint_reset_positions_rad"]
    assert result.action_config.joint_positions.joint_names == list(PROTOCOL_V4_FULL_JOINT_ORDER)
    assert result.action_config.joint_positions.use_default_offset is False
    assert result.action_config.joint_positions.preserve_order is True
    assert result.event_config.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert not hasattr(result.event_config, "randomize_franka_joint_state")
    for joint in PROTOCOL_V4_FULL_JOINT_ORDER:
        assert (
            result.scene_config.robot.actuators["whole_body_and_hands"].stiffness[joint]
            == plan["actuator_parameters"][joint]["stiffness"]
        )
