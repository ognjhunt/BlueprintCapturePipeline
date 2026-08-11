from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_deformable_task_arena_readback import (
    CONTACT_CAPABILITY_BLOCKER,
    NativeDeformableTaskArenaReadback,
    NativeDeformableTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_episode_environment import (
    NativeTaskEpisodeEnvironmentError,
    build_native_task_episode_environment,
)


class _Servo:
    def __init__(self):
        self.reset_count = 0
        self.calls = []

    def current_body_pose_world(self):
        return [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]

    def reset_command_state(self):
        self.reset_count += 1

    def action_for_grasp_target(self, **kwargs):
        self.calls.append(kwargs)
        return [0.0] * 7 + [float(kwargs["gripper_command"])], {"ok": True}


class _Readback:
    def __init__(self):
        self.reset_count = 0
        self.capability_check_count = 0

    def read_task_sample(self):
        return {"joint_positions_rad": {"joint": 0.0}}

    def reset_after_native_reset(self):
        self.reset_count += 1

    def ensure_evaluation_capable(self):
        self.capability_check_count += 1


class _Adapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _built(task_kind: str):
    scene = {"robot": object(), "bound_task_asset": object()}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=scene,
            action_manager=SimpleNamespace(total_action_dim=8),
        ),
        reset=lambda *, seed: None,
    )
    return SimpleNamespace(
        env=env,
        plan={
            "task_kind": task_kind,
            "scenario": {"seed": 17},
            "cadence": {"control_frequency_hz": 15.0},
        },
        scene_asset_names={"task_object": "bound_task_asset"},
        camera_scene_names={
            "external": "arena_external_sensor",
            "wrist": "robot_wrist_sensor",
            "overview": "review_sensor",
        },
    )


@pytest.mark.parametrize(
    ("task_kind", "readback", "expected_source"),
    (
        ("rigid_pick_place", None, "native_rigid_body_readback"),
        (
            "articulated_open_close",
            _Readback(),
            "native_articulated_task_readback",
        ),
    ),
)
def test_factory_binds_original_and_articulated_fixtures_without_scene_names(
    monkeypatch: pytest.MonkeyPatch,
    task_kind: str,
    readback,
    expected_source: str,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", _Adapter)
    servo = _Servo()
    adapter, receipt = build_native_task_episode_environment(
        built=_built(task_kind),
        gripper_convention={
            "closed_command": 1.0,
            "open_command": 0.0,
            "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
        },
        servo=servo,
        task_readback=readback,
        to_tensor=lambda value: value,
    )

    assert receipt["task_kind"] == task_kind
    assert receipt["task_state_source"] == expected_source
    assert receipt["camera_scene_names"] == _built(task_kind).camera_scene_names
    assert adapter.kwargs["camera_scene_names"] == receipt["camera_scene_names"]
    assert (adapter.kwargs["rigid_task_object"] is None) == (task_kind != "rigid_pick_place")
    assert (adapter.kwargs["task_sample_callback"] is None) == (task_kind == "rigid_pick_place")
    assert adapter.kwargs["simulation_step_seconds"] == pytest.approx(1.0 / 15.0)

    action = adapter.kwargs["scripted_pose_action_callback"](
        target_position_world_m=[1.0, 2.0, 3.0],
        target_quaternion_world_xyzw=None,
        gripper_command=1.0,
        max_joint_delta_rad=0.03,
        max_joint_setpoint_lead_rad=0.2,
    )
    assert action == [0.0] * 7 + [1.0]
    assert servo.calls[-1]["target_body_quaternion_world_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    adapter.kwargs["reset_callback"]()


def test_articulated_factory_requires_native_task_readback() -> None:
    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_task_readback_missing",
    ):
        build_native_task_episode_environment(
            built=_built("articulated_open_close"),
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=None,
            to_tensor=lambda value: value,
        )


class _DeformableReadbackSubclass(NativeDeformableTaskArenaReadback):
    pass


class _DeformableReadbackProxy:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)


@pytest.mark.parametrize("kind", ("missing", "fake", "subclass", "proxy"))
def test_deformable_factory_rejects_self_attested_readback_identity(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    constructed = []

    def adapter_factory(**kwargs):
        constructed.append(kwargs)
        return _Adapter(**kwargs)

    exact = NativeDeformableTaskArenaReadback.__new__(NativeDeformableTaskArenaReadback)
    readback = {
        "missing": None,
        "fake": _Readback(),
        "subclass": _DeformableReadbackSubclass.__new__(_DeformableReadbackSubclass),
        "proxy": _DeformableReadbackProxy(exact),
    }[kind]
    monkeypatch.setattr(module, "IsaacEpisodeAdapter", adapter_factory)

    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_deformable_readback_identity_untrusted",
    ):
        build_native_task_episode_environment(
            built=_built("deformable_transfer"),
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=readback,
            to_tensor=lambda value: value,
        )

    assert constructed == []


def test_deformable_factory_raises_contact_blocker_before_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    constructed = []

    def adapter_factory(**kwargs):
        constructed.append(kwargs)
        return _Adapter(**kwargs)

    monkeypatch.setattr(module, "IsaacEpisodeAdapter", adapter_factory)
    readback = NativeDeformableTaskArenaReadback.__new__(NativeDeformableTaskArenaReadback)
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match=CONTACT_CAPABILITY_BLOCKER,
    ):
        build_native_task_episode_environment(
            built=_built("deformable_transfer"),
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=readback,
            to_tensor=lambda value: value,
        )

    assert constructed == []


def test_deformable_factory_ignores_instance_shadowed_capability_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_episode_environment as module

    constructed = []

    def adapter_factory(**kwargs):
        constructed.append(kwargs)
        return _Adapter(**kwargs)

    readback = NativeDeformableTaskArenaReadback.__new__(NativeDeformableTaskArenaReadback)
    readback.ensure_evaluation_capable = lambda: None
    readback.reset_after_native_reset = lambda: None
    readback.read_task_sample = lambda: {"self_attested_contact": True}
    monkeypatch.setattr(module, "IsaacEpisodeAdapter", adapter_factory)

    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match=CONTACT_CAPABILITY_BLOCKER,
    ):
        build_native_task_episode_environment(
            built=_built("deformable_transfer"),
            gripper_convention={
                "closed_command": 1.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08, "1.0": 0.01},
            },
            servo=_Servo(),
            task_readback=readback,
            to_tensor=lambda value: value,
        )

    assert constructed == []


def test_factory_rejects_an_ambiguous_gripper_probe() -> None:
    with pytest.raises(
        NativeTaskEpisodeEnvironmentError,
        match="native_task_episode_gripper_convention_invalid",
    ):
        build_native_task_episode_environment(
            built=_built("rigid_pick_place"),
            gripper_convention={
                "closed_command": 0.0,
                "open_command": 0.0,
                "finger_separation_m": {"0.0": 0.08},
            },
            servo=_Servo(),
            task_readback=None,
            to_tensor=lambda value: value,
        )
