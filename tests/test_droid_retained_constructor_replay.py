"""Replay released constructor/reset bodies on the retained seven-arm input.

SDK configuration/assets are stand-ins; upstream constructors and reset functions
execute unchanged. This is CPU compatibility evidence, never native GPU proof.
"""
from __future__ import annotations

from abc import ABC
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from blueprint_pipeline import native_task_droid_runtime_compat as compat
from blueprint_pipeline.native_task_robot_registry import build_droid_embodiment

FIXTURES = Path(__file__).parent / "fixtures/droid_constructor_replay"
RETAINED = json.loads((FIXTURES / "retained_input.json").read_text())


def _module(monkeypatch, name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)


def _source(name):
    records = json.loads((FIXTURES / "upstream.json").read_text())["sources"]
    record = next(row for row in records if row["path"] == name)
    path = FIXTURES / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    return ast.parse(path.read_text(), filename=str(path))


def _released_constructor(monkeypatch, release):
    tree = _source(f"{release}_droid.py")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    # Read the actual release's default joint dictionary, without importing SDK.
    default_node = next(node.value for node in ast.walk(classes["DroidSceneCfg"])
                        if isinstance(node, ast.keyword) and node.arg == "joint_pos")
    defaults = eval(compile(ast.Expression(default_node), "defaults", "eval"), {"torch": NS(pi=math.pi)})

    class State(NS):
        def replace(self, **values):
            return State(**{**vars(self), **values})

    class Base:
        def __init__(self, enable_cameras=False, initial_pose=None, *args, **kwargs):
            self.initial_pose = initial_pose

        def get_scene_cfg(self):
            return self.scene_config

        def add_camera_variations(self, config):
            pass

    def scene():
        return NS(robot=NS(spawn=NS(usd_path="upstream"), init_state=State(joint_pos=deepcopy(defaults))),
                  stand=object())

    def events():
        config = NS(randomize_franka_joint_state=NS(params={"mean": 0.0, "std": .02}))
        if release == "baseline":
            config.init_franka_arm_pose = NS(params={"default_pose": []})
        return config

    namespace = dict(ABC=ABC, EmbodimentBase=Base, ArmMode=NS(SINGLE_ARM="single"),
                     DroidSceneCfg=scene, DroidCameraCfg=NS, DroidObservationsCfg=NS,
                     DroidEventCfg=events, DroidAbsoluteJointPositionActionsCfg=NS,
                     register_asset=lambda cls: cls, RobotiqGripper=NS,
                     _DROID_ROBOT_PRIM=object(), _DROID_STAND_PRIM=NS(
                         stand_default_height=1.35, stand_default_footprint_xy_m=(1.08, .91032)),
                     compose_on_stand_usd=lambda *args, **kwargs: "composed")
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "_DROID_JOINT_NAMES"
                         for target in node.targets))
    nodes.extend(classes[name] for name in ("DroidEmbodimentBase", "DroidAbsoluteJointPositionEmbodiment"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(FIXTURES / f"{release}_droid.py"), "exec"), namespace)
    cls = namespace["DroidAbsoluteJointPositionEmbodiment"]
    _module(monkeypatch, "isaaclab_arena.embodiments.droid.droid", DroidAbsoluteJointPositionEmbodiment=cls)
    _module(monkeypatch, "isaaclab.managers", EventTermCfg=lambda **kw: NS(**kw))
    _module(monkeypatch, "isaaclab.utils.configclass", configclass=lambda cls: cls)
    _module(monkeypatch, "isaaclab_arena.embodiments.franka.franka",
            franka_stack_events=NS(set_default_joint_pose=object()))
    return cls, defaults


def test_exact_ga_constructor_reproduces_retained_refusal(monkeypatch):
    cls, _ = _released_constructor(monkeypatch, "ga")
    with pytest.raises(AssertionError, match="expected 13 joint positions, got 7"):
        cls(initial_joint_pose=list(RETAINED["robot"]["joint_reset_positions_rad"].values()))


@pytest.mark.parametrize("release", ["baseline", "ga"])
@pytest.mark.parametrize("full_reset", [False, True])
def test_real_factory_preserves_seven_arm_and_existing_full_reset(monkeypatch, release, full_reset):
    _, defaults = _released_constructor(monkeypatch, release)
    robot = deepcopy(RETAINED["robot"])
    reset = robot["joint_reset_positions_rad"]
    if full_reset:
        reset.update({name: value for name, value in zip(
            ("finger_joint", "right_outer_knuckle_joint", "right_inner_finger_joint",
             "right_inner_finger_knuckle_joint", "left_inner_finger_knuckle_joint", "left_inner_finger_joint"),
            (.1, .2, .3, .4, .5, .6))})
    built = build_droid_embodiment(robot, enable_cameras=True, pose_class=NS)
    assert built.event_config.init_franka_arm_pose.params["default_pose"] == list(reset.values())
    assert built.event_config.randomize_franka_joint_state.params == {"mean": 0.0, "std": 0.0}
    assert built.initial_pose.position_xyz == tuple(robot["base_pose_world"]["position_world_m"])
    assert built.initial_pose.rotation_xyzw == tuple(robot["base_pose_world"]["orientation_xyzw"])
    assert built.scene_config.stand is None
    # Seven-arm input must not synthesize six gripper values for GA's setter.
    assert built.scene_config.robot.init_state.joint_pos == (reset if full_reset and release == "ga" else defaults)


@pytest.mark.parametrize("name", ["baseline_franka_stack_events.py", "ga_stack_events.py"])
def test_released_reset_updates_only_supplied_arm_prefix(monkeypatch, name):
    function = next(node for node in _source(name).body
                    if isinstance(node, ast.FunctionDef) and node.name == "set_default_joint_pose")

    class Tensor(np.ndarray):
        def repeat(self, *shape):
            return np.tile(np.asarray(self), shape)

    launches = []
    kernel = object()

    def launch(actual_kernel, *, dim, inputs, outputs, device):
        launches.append(dim)
        assert actual_kernel is kernel and device == "cpu"
        # Model the SDK kernel's indexed write; the released event chooses
        # indices, values, repeat count and dimensions, rather than this test.
        outputs[0][:, inputs[1]] = inputs[0].reshape(dim)

    _module(monkeypatch, "isaaclab_physx.assets.articulation.kernels", update_default_joint_values=kernel)
    namespace = dict(torch=NS(tensor=lambda value, **kw: np.array(value).view(Tensor),
                             arange=lambda n, **kw: np.arange(n), int32=np.int32),
                     wp=NS(launch=launch), SceneEntityCfg=lambda name: NS(name=name))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), name, "exec"), namespace)
    initial = np.arange(26, dtype=float).reshape(2, 13) / 10
    data = initial.copy()
    reset = list(RETAINED["robot"]["joint_reset_positions_rad"].values())
    env = NS(scene={"robot": NS(data=NS(default_joint_pos=data))}, num_envs=2, device="cpu")
    namespace["set_default_joint_pose"](env, np.array([0, 1]), reset)
    assert launches == [(2, 7)]
    np.testing.assert_array_equal(data[:, :7], np.tile(reset, (2, 1)))
    np.testing.assert_array_equal(data[:, 7:], initial[:, 7:])


def test_unknown_partial_ga_reset_is_not_admitted(monkeypatch):
    cls, _ = _released_constructor(monkeypatch, "ga")
    reset = {f"wrong_joint{i}": 0.0 for i in range(7)}
    with pytest.raises(ValueError, match="reset_api_unrecognized"):
        compat.configure_droid_reset(cls(), reset)
