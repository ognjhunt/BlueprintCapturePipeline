from copy import deepcopy
import hashlib
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from blueprint_pipeline import native_task_droid_runtime_compat as compat


def install_module(monkeypatch, name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)


def asset_inputs(tmp_path, monkeypatch):
    hashes = {}
    for index, relative in enumerate(compat.ASSET_HASHES):
        path = tmp_path / compat.ASSET_DIRECTORY / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"sealed-asset-{index}".encode()
        path.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(compat, "ASSET_HASHES", hashes)
    request = tmp_path / "native_task_runtime_comparison_request.v1.json"
    request.write_text("{}")
    return request


def embodiment(*, ga):
    events = NS(randomize_franka_joint_state=NS(params={"mean": 0, "std": .02, "asset_cfg": object()}))
    if not ga:
        events.init_franka_arm_pose = NS(params={})
    return NS(event_config=events, set_initial_joint_pose=lambda value: None,
              scene_config=NS(stand=object(), robot=NS(spawn=NS(usd_path="composed-on-stand"),
                  actuators={"gripper": NS(velocity_limit=5.0)}),
                  ee_frame=NS(target_frames=[NS(name=name, prim_path="ga-new-frame", offset=NS(pos=(.131, 0, 0)))
                                            for name, _, _ in compat.FRAMES])),
              camera_config=NS(**{name: NS(prim_path="ga-panda_link0-camera")
                                  for name in ("external_camera", "external_camera_2")}))


def test_both_releases_preserve_reset_order_and_frozen_configuration(tmp_path, monkeypatch):
    request = asset_inputs(tmp_path / "runtime_inputs", monkeypatch)
    install_module(monkeypatch, "isaaclab.actuators", ImplicitActuatorCfg=lambda **kw: NS(**kw))
    install_module(monkeypatch, "isaaclab.managers", EventTermCfg=lambda **kw: NS(**kw))
    install_module(monkeypatch, "isaaclab.utils.configclass", configclass=lambda cls: cls)
    reset_function = object()
    install_module(monkeypatch, "isaaclab_arena.embodiments.franka.franka",
                   franka_stack_events=NS(set_default_joint_pose=reset_function))
    reset = {f"joint-{i}": float(i) / 10 for i in range(13)}
    readbacks = []
    for ga in (False, True):
        robot = embodiment(ga=ga)
        compat.configure_droid_reset(robot, reset)
        assert robot.event_config.init_franka_arm_pose.params["default_pose"] == list(reset.values())
        assert robot.event_config.randomize_franka_joint_state.params["std"] == 0
        if ga:
            terms = [key for key in vars(type(robot.event_config)) if not key.startswith("_")]
            assert terms == ["init_franka_arm_pose", "randomize_franka_joint_state"]
            assert robot.event_config.init_franka_arm_pose.func is reset_function
        value = compat.freeze_comparison_droid(robot, bundle_root=request.parent.parent / "native_task_packet",
                                                robot_id="franka_panda")
        assert robot.scene_config.robot.spawn.usd_path.endswith(compat.ROBOT_PATH)
        assert robot.scene_config.robot.actuators["gripper"].velocity_limit == 1.0
        assert robot.scene_config.stand is None
        assert value["retained_asset_gaps"] == [compat.MATERIAL_GAP]
        readbacks.append(value)
    assert readbacks[0] == readbacks[1]


def test_asset_bytes_and_original_gap_cannot_drift(tmp_path, monkeypatch):
    request = asset_inputs(tmp_path, monkeypatch)
    paths = compat.comparison_robot_asset_inputs(request)
    first = next(iter(paths.values()))
    original = first.read_bytes()
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset_missing_or_changed"):
        compat.comparison_robot_asset_inputs(request)
    first.write_bytes(original)
    missing = tmp_path / compat.ASSET_DIRECTORY / compat.MATERIAL_GAP
    missing.parent.mkdir(parents=True)
    missing.write_text("unadmitted replacement")
    with pytest.raises(ValueError, match="original_material_gap_changed"):
        compat.comparison_robot_asset_inputs(request)


def test_normal_runtime_keeps_its_configuration(tmp_path):
    robot = embodiment(ga=False)
    before = deepcopy(robot.scene_config.robot.actuators)
    assert compat.freeze_comparison_droid(robot, bundle_root=tmp_path / "native_task_packet",
                                          robot_id="franka_panda") is None
    assert vars(robot.scene_config.robot.actuators["gripper"]) == vars(before["gripper"])
