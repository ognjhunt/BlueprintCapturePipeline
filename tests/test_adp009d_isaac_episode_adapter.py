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
    GRIPPER_PHYSICAL_FULL_OPENING_M,
    IsaacEpisodeAdapter,
    IsaacEpisodeAdapterError,
    IsaacKinematicReplayWriter,
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
        self.unwrapped = type(
            "U",
            (),
            {
                "scene": scene,
                "device": "cpu",
                # Live environments always expose the sim-time counter; the
                # stub models it so the freshness stamp cannot degrade silently.
                "episode_length_buf": _Tensor([0.0]),
                "cfg": type(
                    "C",
                    (),
                    {"sim": type("S", (), {"dt": 1.0 / 120.0})(), "decimation": 8},
                )(),
            },
        )()

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


def test_gripper_width_is_probe_calibrated_physical_opening() -> None:
    """Link-origin separation is calibrated onto the 2F-85 jaw stroke."""

    adapter = _adapter()

    sample = adapter.read_object_sample()

    assert sample["gripper_width_m"] == pytest.approx(
        GRIPPER_PHYSICAL_FULL_OPENING_M, abs=1e-9
    )
    assert sample["gripper_body_separation_m"] == pytest.approx(0.06, abs=1e-9)
    assert sample["gripper_width_open_fraction_unclamped"] == pytest.approx(1.0)
    assert sample["gripper_width_calibration_clamped"] is False
    # The grasp frame is the midpoint between the fingers.
    assert sample["grasp_frame_position_world_m"] == pytest.approx(
        [0.03, 0.0, 1.0], abs=1e-9
    )


def test_linkage_overtravel_is_bounded_and_raw_measurement_is_retained() -> None:
    """Regression for v74 samples whose link origins exceeded the 85 mm stroke."""

    from blueprint_pipeline.adp009d_task_scoring import normalize_object_samples

    robot = _Robot()
    bodies = robot.data.body_names
    robot.data.body_pose_w[0, bodies.index("right_inner_finger"), :3] = [
        0.10,
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
        gripper_open_width_m=0.0831756591796875,
    )

    sample = dict(adapter.read_object_sample())
    assert sample["gripper_body_separation_m"] == pytest.approx(0.10)
    assert sample["gripper_width_m"] == pytest.approx(
        GRIPPER_PHYSICAL_FULL_OPENING_M
    )
    assert sample["gripper_width_open_fraction_unclamped"] > 1.0
    assert sample["gripper_width_calibration_clamped"] is True
    sample["step_index"] = 0
    normalized = normalize_object_samples([sample], require_sealed_start_pose=False)
    assert normalized[0]["gripper_width_m"] == pytest.approx(
        GRIPPER_PHYSICAL_FULL_OPENING_M
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


def test_reset_callback_can_restore_a_wrist_observable_episode_start() -> None:
    env = _Env()
    calls: list[str] = []
    adapter = IsaacEpisodeAdapter(
        env=env,
        robot=_Robot(),
        approved_can=_Can(),
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        reset_callback=lambda: calls.append("observable_start_restored"),
    )

    adapter.reset()

    assert calls == ["observable_start_restored"]
    assert env.reset_calls == []


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
    assert bindings["gripper_physical_full_opening_m"] == pytest.approx(0.085)
    assert bindings["raw_gripper_body_separation_retained"] is True

    drifted = dict(bindings)
    drifted["camera_view_binding"] = {"external_camera": DROID_WRIST_VIEW}
    assert "isaac_episode_adapter_camera_binding_drifted" in validate_adapter_bindings(
        drifted
    )

    drifted = dict(bindings)
    drifted["gripper_width_source"] = "raw_link_origin_distance"
    assert "isaac_episode_adapter_gripper_width_source_drifted" in (
        validate_adapter_bindings(drifted)
    )


def test_adapter_gripper_stroke_matches_deterministic_scorer() -> None:
    from blueprint_pipeline.adp009d_task_scoring import GRIPPER_FULL_OPENING_M

    assert GRIPPER_PHYSICAL_FULL_OPENING_M == GRIPPER_FULL_OPENING_M


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


def test_policy_inputs_are_stamped_with_rendered_sim_time() -> None:
    env = _Env()
    env.unwrapped.episode_length_buf = _Tensor([3.0])
    env.unwrapped.cfg = type("C", (), {
        "sim": type("S", (), {"dt": 1.0 / 120.0})(),
        "decimation": 8,
    })()
    adapter = _adapter(env)

    inputs = adapter.read_policy_inputs()

    # 3 environment steps x 8 physics steps x 1/120 s = 0.2 s.
    assert inputs["observation_sim_time"] == pytest.approx(0.2)
    assert adapter.sim_time() == pytest.approx(0.2)


def test_sim_time_requires_the_environment_counter() -> None:
    env = _Env()
    env.unwrapped.episode_length_buf = None
    adapter = _adapter(env)

    with pytest.raises(IsaacEpisodeAdapterError, match="sim_time"):
        adapter.sim_time()


def test_full_joint_positions_expose_every_dof() -> None:
    adapter = _adapter()

    full = adapter.read_full_joint_positions()

    assert len(full) == 13
    assert full[0] == pytest.approx(0.0)
    assert full[-1] == pytest.approx(0.6)


def test_kinematic_replay_writer_scrubs_state_and_renders_without_physics() -> None:
    torch = pytest.importorskip("torch")

    class _WritableRobot(_Robot):
        def __init__(self):
            super().__init__()
            self.written: list[tuple] = []

        def write_joint_state_to_sim(self, position, velocity):
            self.written.append((position, velocity))

    class _WritableCan(_Can):
        def __init__(self):
            super().__init__()
            self.poses: list = []
            self.velocities: list = []

        def write_root_pose_to_sim(self, pose):
            self.poses.append(pose)

        def write_root_velocity_to_sim(self, velocity):
            self.velocities.append(velocity)

    class _RecordingSim:
        def __init__(self):
            self.render_calls = 0

        def render(self):
            self.render_calls += 1

    class _RecordingScene(dict):
        def __init__(self, cameras):
            super().__init__(cameras)
            self.update_dts: list[float] = []

        def update(self, dt):
            self.update_dts.append(float(dt))

    env = _Env()
    scene = _RecordingScene(
        {"external_camera": _Camera(4), "wrist_camera": _Camera(4)}
    )
    env.unwrapped.scene = scene
    env.unwrapped.sim = _RecordingSim()
    robot = _WritableRobot()
    can = _WritableCan()
    adapter = IsaacEpisodeAdapter(
        env=env,
        robot=robot,
        approved_can=can,
        action_dim=8,
        reset_seed=20260806,
        to_torch=_to_torch,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
    )
    writer = IsaacKinematicReplayWriter(adapter=adapter)

    state = {
        "kind": "policy-step",
        "frame_index": 0,
        "sim_time_s": 0.0,
        "joint_position_rad": [0.1] * 7,
        "full_joint_position_rad": [0.1] * 13,
        "gripper_width_m": 0.05,
        "object_sample": {
            "step_index": 0,
            "can_pose_world": [1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        },
    }
    writer.write_step_state(state)
    writer.render()
    views = writer.read_views()

    position, velocity = robot.written[0]
    assert list(map(float, torch.as_tensor(position).reshape(-1))) == (
        pytest.approx([0.1] * 13)
    )
    assert float(torch.as_tensor(velocity).abs().sum()) == 0.0
    assert list(map(float, torch.as_tensor(can.poses[0]).reshape(-1))) == (
        pytest.approx([1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    )
    assert float(torch.as_tensor(can.velocities[0]).abs().sum()) == 0.0
    assert env.unwrapped.sim.render_calls == 1
    assert scene.update_dts == [pytest.approx(1.0 / 120.0)]
    assert "observation/exterior_image_1_left" in views


def test_kinematic_replay_writer_requires_the_full_joint_vector() -> None:
    pytest.importorskip("torch")
    writer = IsaacKinematicReplayWriter(adapter=_adapter())

    with pytest.raises(IsaacEpisodeAdapterError, match="full_joint"):
        writer.write_step_state(
            {"joint_position_rad": [0.0] * 7, "object_sample": {}}
        )


class _CudaLikeTensor(_Tensor):
    """Raises on direct numpy conversion until .cpu() runs, like a CUDA tensor."""

    def __init__(self, values, on_device=True):
        super().__init__(values)
        self._on_device = on_device

    def cpu(self):
        return _CudaLikeTensor(list(self), on_device=False)

    def __array__(self, *args, **kwargs):
        if self._on_device:
            raise TypeError(
                "can't convert cuda:0 device type tensor to numpy. "
                "Use Tensor.cpu() to copy the tensor to host memory first."
            )
        return np.asarray(list(self))


def test_sim_time_reads_a_device_resident_counter() -> None:
    """v76 died at the first policy query: episode_length_buf lives on cuda,
    and a bare np.asarray on it raises.  The stamp must take the same
    detach-then-cpu route every other adapter reader takes."""

    env = _Env()
    env.unwrapped.episode_length_buf = _CudaLikeTensor([3.0])
    adapter = _adapter(env)

    assert adapter.sim_time() == pytest.approx(0.2)
    assert adapter.read_policy_inputs()["observation_sim_time"] == pytest.approx(0.2)
