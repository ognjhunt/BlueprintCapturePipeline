"""G1 controller frames must reach the same Arena scene in the declared order."""

from types import SimpleNamespace

import pytest

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_joint_episode_environment import NativeG1JointEpisodeEnvironment


def _adapter():
    names = list(reversed(PROTOCOL_V4_FULL_JOINT_ORDER))
    data = SimpleNamespace(
        root_quat_w=[[0.0, 0.0, 0.0, 1.0]],
        root_pos_w=[[1.0, 2.0, 0.8]],
        joint_pos=[[float(index) / 100 for index in range(len(names))]],
        joint_vel=[[0.0] * len(names)],
    )
    robot = SimpleNamespace(joint_names=names, data=data)
    calls = []
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene={"robot": robot},
            action_manager=SimpleNamespace(total_action_dim=43),
            device="cpu",
        ),
        reset=lambda *, seed: calls.append(("reset", seed)),
        step=lambda action: calls.append(("step", action)),
    )
    plan = {"robot": {
        "robot_id": "unitree_g1",
        "joint_position_limits_rad": {name: [-1.0, 1.0] for name in names},
    }}
    adapter = NativeG1JointEpisodeEnvironment(
        built=SimpleNamespace(plan=plan, env=env),
        to_tensor=lambda value: value,
        make_action_tensor=lambda value, *, device: (value, device),
    )
    return adapter, calls, names


def test_reset_reads_named_state_and_controller_targets_step_same_scene() -> None:
    adapter, calls, live_names = _adapter()
    with pytest.raises(ValueError, match="reset_required"):
        adapter.read_semantic_v3_state()
    reset = adapter.reset(seed=17)
    assert calls == [("reset", 17)]
    assert reset["joint_position_rad"][PROTOCOL_V4_FULL_JOINT_ORDER[0]] == 0.42
    state = adapter.read_semantic_v3_state()
    assert len(state) == 64
    assert state[6] == reset["joint_position_rad"]["left_hip_pitch_joint"]
    target = {name: index / 100 for index, name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)}
    stepped = adapter.step_controller_targets(target)
    assert calls[1] == ("step", ([[target[name] for name in PROTOCOL_V4_FULL_JOINT_ORDER]], "cpu"))
    assert stepped["step_index"] == 1
    assert live_names != list(PROTOCOL_V4_FULL_JOINT_ORDER)


def test_raw_policy_vector_and_invalid_controller_targets_never_step() -> None:
    adapter, calls, _ = _adapter()
    adapter.reset(seed=0)
    with pytest.raises(ValueError, match="joint_inventory_invalid"):
        adapter.step_controller_targets({str(index): 0.0 for index in range(40)})
    target = {name: 0.0 for name in PROTOCOL_V4_FULL_JOINT_ORDER}
    target["right_knee_joint"] = 2.0
    with pytest.raises(ValueError, match="joint_target_invalid"):
        adapter.step_controller_targets(target)
    assert calls == [("reset", 0)]
