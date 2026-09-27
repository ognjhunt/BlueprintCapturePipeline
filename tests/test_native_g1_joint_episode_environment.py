"""G1 controller frames must reach the same Arena scene in the declared order."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_joint_episode_environment import NativeG1JointEpisodeEnvironment


def test_beta2_proxy_camera_buffer_uses_expanded_torch_view() -> None:
    class _Proxy:
        def __init__(self, tensor):
            self.shape = (tensor.shape[0],)
            self.torch = tensor

    measured = torch.zeros((1, 480, 640, 4), dtype=torch.uint8)
    result = NativeG1JointEpisodeEnvironment._array(_Proxy(measured))
    assert result.shape == (1, 480, 640, 4)
    assert result.dtype == np.uint8


def _adapter():
    names = list(reversed(PROTOCOL_V4_FULL_JOINT_ORDER))
    data = SimpleNamespace(
        root_quat_w=[[0.0, 0.0, 0.0, 1.0]],
        root_pos_w=[[1.0, 2.0, 0.8]],
        joint_pos=[[float(index) / 100 for index in range(len(names))]],
        joint_vel=[[0.0] * len(names)],
    )
    robot = SimpleNamespace(joint_names=names, data=data)
    head = SimpleNamespace(
        frame=[0],
        data=SimpleNamespace(
            output={"rgb": np.zeros((1, 480, 640, 4), dtype=np.uint8)},
            intrinsic_matrices=np.eye(3)[None],
            pos_w=np.array([[1.0, 2.0, 3.0]]),
            quat_w_opengl=np.array([[0.0, 0.0, 0.0, 1.0]]),
        ),
        cfg=SimpleNamespace(spawn=SimpleNamespace(clipping_range=(0.01, 20.0))),
    )
    overview = SimpleNamespace(
        frame=[0],
        data=SimpleNamespace(
            output={"rgb": np.zeros((1, 360, 640, 3), dtype=np.uint8)},
            intrinsic_matrices=np.eye(3)[None],
            pos_w=np.array([[0.0, 0.0, 4.0]]),
            quat_w_opengl=np.array([[0.0, 0.0, 0.0, 1.0]]),
        ),
        cfg=SimpleNamespace(spawn=SimpleNamespace(clipping_range=(0.01, 20.0))),
    )
    calls = []
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene={"robot": robot, "head_camera": head, "overview_camera": overview},
            action_manager=SimpleNamespace(total_action_dim=43),
            device="cpu",
        ),
        reset=lambda *, seed: calls.append(("reset", seed)),
        step=lambda action: calls.append(("step", action)),
    )
    plan = {
        "robot": {
            "robot_id": "unitree_g1",
            "joint_position_limits_rad": {name: [-1.0, 1.0] for name in names},
        },
        "cadence": {"control_frequency_hz": 50.0},
    }
    adapter = NativeG1JointEpisodeEnvironment(
        built=SimpleNamespace(
            plan=plan,
            env=env,
            camera_scene_names={"head": "head_camera", "overview": "overview_camera"},
        ),
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


def test_head_policy_frame_and_review_frame_are_exact_and_fresh() -> None:
    adapter, _, _ = _adapter()
    adapter.reset(seed=4)
    policy = adapter.read_policy_inputs()
    assert policy["front_rgb"].shape == (480, 640, 3)
    assert policy["observation_state"] == adapter.read_semantic_v3_state()
    assert policy["sensor_freshness"]["sensor_frame_index"] == 0
    review = adapter.read_review_inputs()
    assert review["overview_rgb"].shape == (360, 640, 3)
    adapter.step_controller_targets({name: 0.0 for name in PROTOCOL_V4_FULL_JOINT_ORDER})
    with pytest.raises(ValueError, match="camera_stale:head"):
        adapter.read_policy_inputs()
    adapter._env.unwrapped.scene["head_camera"].frame = [1]
    assert adapter.read_policy_inputs()["sensor_freshness"]["sensor_frame_index"] == 1


def test_g1_camera_metadata_binds_head_and_overview_to_scene_time() -> None:
    adapter, _, _ = _adapter()
    adapter.reset(seed=4)
    adapter.read_policy_inputs()
    policy = adapter.read_observation_metadata(("head",))
    assert policy["timestamp_ns"] == 0
    assert policy["calibrations"]["head"]["resolution"] == [640, 480]
    assert policy["calibrations"]["head"]["world_from_camera"][0][3] == 1.0
    adapter.step_controller_targets({name: 0.0 for name in PROTOCOL_V4_FULL_JOINT_ORDER})
    adapter._env.unwrapped.scene["head_camera"].frame = [1]
    adapter._env.unwrapped.scene["overview_camera"].frame = [1]
    adapter.read_review_inputs()
    review = adapter.read_observation_metadata(("head", "overview"))
    assert review["timestamp_ns"] == 20_000_000
    assert set(review["calibrations"]) == {"head", "overview"}
    assert review["synchronizations"]["overview"]["host_bytes_ready"] is True


def test_policy_camera_rejects_non_uint8_or_wrong_resolution() -> None:
    adapter, _, _ = _adapter()
    adapter.reset(seed=5)
    camera = adapter._env.unwrapped.scene["head_camera"]
    camera.data.output["rgb"] = np.zeros((1, 480, 640, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="camera_readback_invalid:head"):
        adapter.read_policy_inputs()
    camera.data.output["rgb"] = np.zeros((1, 300, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="policy_camera_shape_invalid"):
        adapter.read_policy_inputs()
