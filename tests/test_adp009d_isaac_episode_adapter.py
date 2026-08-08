from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.adp009d_isaac_episode_adapter import (
    CAMERA_VIEW_BINDING,
    FINGER_BODIES,
    IsaacEpisodeAdapter,
    IsaacEpisodeAdapterError,
    describe_adapter,
    rgb_from_camera_output,
    rotation_row_major_from_quaternion_wxyz,
    validate_adapter_bindings,
)


class _Tensor(list):
    """Enough of a tensor for the adapter's indexing, without torch."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def _to_torch(value):
    return value


class _Data:
    def __init__(self, body_names, joint_pos, body_pose_w, joint_limits):
        self.body_names = body_names
        self.joint_pos = joint_pos
        self.body_pose_w = body_pose_w
        self.joint_limits = joint_limits


class _Robot:
    def __init__(self):
        bodies = [
            "panda_link0", "panda_link7", "base_link",
            "left_inner_finger", "right_inner_finger",
        ]
        # Fingers 0.06 m apart in x, so separation is exactly 0.06.
        poses = np.zeros((1, len(bodies), 7), dtype=float)
        poses[0, bodies.index("base_link"), :7] = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        poses[0, bodies.index("left_inner_finger"), :3] = [0.0, 0.0, 1.0]
        poses[0, bodies.index("right_inner_finger"), :3] = [0.06, 0.0, 1.0]
        self.data = _Data(
            body_names=bodies,
            joint_pos=np.linspace(0.0, 0.6, 13).reshape(1, 13),
            body_pose_w=poses,
            joint_limits=np.tile(np.array([[-2.9, 2.9]]), (1, 13, 1)),
        )


class _Can:
    def __init__(self):
        self.data = _Data(None, None, None, None)
        self.data.root_pose_w = np.array(
            [[3.468, -3.310, 0.526, 0.0, 0.0, 0.0, 1.0]], dtype=float
        )


class _Camera:
    def __init__(self, channels=4):
        frame = np.zeros((1, 720, 1280, channels), dtype=np.uint8)
        frame[..., 0] = 200
        if channels == 4:
            frame[..., 3] = 255
        self.data = type("D", (), {"output": {"rgb": _Tensor(frame)}})()


class _Scene(dict):
    pass


class _Env:
    def __init__(self, channels=4):
        self.reset_calls: list[int] = []
        self.stepped: list[list[float]] = []
        scene = _Scene(
            {"external_camera": _Camera(channels), "wrist_camera": _Camera(channels)}
        )
        self.unwrapped = type("U", (), {"scene": scene, "device": "cpu"})()

    def reset(self, seed=None):
        self.reset_calls.append(seed)

    def step(self, tensor):
        self.stepped.append([float(v) for v in np.asarray(tensor)[0]])


def _adapter(env=None):
    return IsaacEpisodeAdapter(
        env=env or _Env(),
        robot=_Robot(),
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
    )


def test_alpha_is_dropped_at_the_boundary_not_downstream() -> None:
    """A constant alpha channel has already once made a black render look alive."""

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255

    rgb = rgb_from_camera_output(rgba)

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    assert not rgb.any()
    # Already-RGB input passes through unchanged.
    assert rgb_from_camera_output(np.zeros((4, 4, 3), np.uint8)).shape == (4, 4, 3)


def test_malformed_camera_frames_are_refused() -> None:
    for bad in (np.zeros((4, 4)), np.zeros((4, 4, 2)), np.zeros((4, 4, 5))):
        with pytest.raises(IsaacEpisodeAdapterError):
            rgb_from_camera_output(bad)


def test_policy_inputs_carry_both_views_as_uint8_rgb() -> None:
    adapter = _adapter()

    inputs = adapter.read_policy_inputs()

    for view in (DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW):
        assert inputs[view].shape == (720, 1280, 3)
        assert inputs[view].dtype == np.uint8
    assert len(inputs["joint_position"]) == 7
    assert inputs["gripper_position"] == pytest.approx(0.0, abs=1e-9)
    assert inputs["eef_9d"] == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]
    )


def test_isaac_wxyz_quaternion_is_converted_before_nvidia_frame_correction() -> None:
    half_sqrt = 2**-0.5

    rotation = rotation_row_major_from_quaternion_wxyz(
        [half_sqrt, 0.0, 0.0, half_sqrt]
    )

    assert rotation == pytest.approx([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_gripper_width_is_measured_finger_separation() -> None:
    """The same physical quantity the convention probe measures, from the same bodies."""

    adapter = _adapter()

    sample = adapter.read_object_sample()

    assert sample["gripper_width_m"] == pytest.approx(0.06, abs=1e-9)
    # The grasp frame is the midpoint between the fingers.
    assert sample["grasp_frame_position_world_m"] == pytest.approx(
        [0.03, 0.0, 1.0], abs=1e-9
    )


def test_droid_gripper_state_is_closed_fraction_from_the_measured_probe_span() -> None:
    robot = _Robot()
    bodies = robot.data.body_names
    robot.data.body_pose_w[0, bodies.index("right_inner_finger"), :3] = [
        0.03,
        0.0,
        1.0,
    ]
    adapter = IsaacEpisodeAdapter(
        env=_Env(),
        robot=robot,
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
    )

    assert adapter.read_policy_inputs()["gripper_position"] == pytest.approx(0.5)


def test_object_sample_matches_the_scorer_schema() -> None:
    from blueprint_pipeline.adp009d_task_scoring import normalize_object_samples

    adapter = _adapter()
    sample = dict(adapter.read_object_sample())
    sample["step_index"] = 0

    normalized = normalize_object_samples([sample], require_sealed_start_pose=False)
    assert normalized[0]["step_index"] == 0
    assert len(sample["can_pose_world"]) == 7


def test_a_wrong_width_action_never_reaches_the_simulator() -> None:
    env = _Env()
    adapter = _adapter(env)

    with pytest.raises(IsaacEpisodeAdapterError):
        adapter.step([0.0] * 7)
    assert env.stepped == []


def test_reset_uses_the_pinned_seed() -> None:
    env = _Env()
    adapter = _adapter(env)

    adapter.reset()

    assert env.reset_calls == [20260806]


def test_a_missing_finger_body_is_refused_at_construction() -> None:
    """Better to fail building the adapter than to score on a wrong width."""

    class _NoFingers(_Robot):
        def __init__(self):
            super().__init__()
            self.data.body_names = ["panda_link0", "panda_link7"]

    with pytest.raises(IsaacEpisodeAdapterError):
        IsaacEpisodeAdapter(
            env=_Env(),
            robot=_NoFingers(),
            approved_can=_Can(),
            action_dim=8,
            reset_seed=1,
            to_torch=_to_torch,
            gripper_closed_width_m=0.0,
            gripper_open_width_m=0.06,
        )


def test_unmeasured_gripper_width_span_is_refused() -> None:
    with pytest.raises(IsaacEpisodeAdapterError, match="width_calibration_invalid"):
        IsaacEpisodeAdapter(
            env=_Env(),
            robot=_Robot(),
            approved_can=_Can(),
            action_dim=8,
            reset_seed=1,
            to_torch=_to_torch,
            gripper_closed_width_m=0.04,
            gripper_open_width_m=0.04,
        )


def test_a_camera_without_rgb_is_refused() -> None:
    env = _Env()
    env.unwrapped.scene["wrist_camera"].data.output = {}
    adapter = _adapter(env)

    with pytest.raises(IsaacEpisodeAdapterError):
        adapter.read_policy_inputs()


def test_bindings_are_reported_and_drift_is_caught() -> None:
    bindings = describe_adapter()

    assert validate_adapter_bindings(bindings) == []
    assert bindings["camera_view_binding"] == dict(CAMERA_VIEW_BINDING)
    assert bindings["finger_bodies"] == list(FINGER_BODIES)

    drifted = dict(bindings)
    drifted["camera_view_binding"] = {"external_camera": DROID_WRIST_VIEW}
    assert "isaac_episode_adapter_camera_binding_drifted" in validate_adapter_bindings(
        drifted
    )


def test_the_adapter_satisfies_the_episode_loop_seam() -> None:
    """Whatever the loop calls, the adapter must provide."""

    adapter = _adapter()
    for method in (
        "reset",
        "joint_limits",
        "read_policy_inputs",
        "read_arm_joint_positions",
        "step",
        "read_object_sample",
    ):
        assert callable(getattr(adapter, method))
    limits = adapter.joint_limits()
    assert len(limits) == 7
    assert all(len(row) == 2 for row in limits)


def test_module_imports_without_isaac_or_torch() -> None:
    """It must stay testable off-GPU, so simulator imports live inside methods."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_episode_adapter as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    header = source[: source.index("ADAPTER_SCHEMA_VERSION")]
    assert "import torch" not in header
    assert "import isaaclab" not in header
    assert "from pxr" not in header
