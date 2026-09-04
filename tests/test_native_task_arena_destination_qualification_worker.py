from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_arena_destination_qualification_worker import (
    _capture_destination_measurements,
    _maximum_support_penetration_m,
)


POSE = [1.0, 2.0, 0.275, 0.0, 0.0, 0.0, 1.0]


class _Sliceable:
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return [0.0] * 7
        return self


class _Torch:
    float32 = "float32"

    @staticmethod
    def as_tensor(_value):
        return _Sliceable()

    @staticmethod
    def tensor(value, **_kwargs):
        return value


class _Environment:
    def __init__(self) -> None:
        self.reset_seeds: list[int] = []
        self.actions: list[list[list[float]]] = []
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos=[[0.0] * 8]))
        self.unwrapped = SimpleNamespace(scene={"robot": robot}, device="cuda:0")

    def reset(self, *, seed: int) -> None:
        self.reset_seeds.append(seed)

    def step(self, action) -> None:
        self.actions.append(action)


class _Reader:
    def __init__(self) -> None:
        self.count = 0

    def read_task_sample(self):
        self.count += 1
        return {
            "destination_pose_world": POSE,
            "destination_scene_support_contact_peak_force_n": 1.25,
            "destination_scene_forbidden_contact_peak_force_n": 0.0,
        }


def _reset(_built):
    return {
        "passed": True,
        "objects": [
            {
                "asset_id": "blue-document-tray",
                "runtime_name": "task_support",
                "task_subject": False,
                "observed_root_pose_world": {
                    "position_world_m": POSE[:3],
                    "orientation_xyzw": POSE[3:],
                },
            }
        ],
    }


def test_native_worker_rehearses_three_resets_and_settles_without_policy() -> None:
    env = _Environment()
    built = SimpleNamespace(plan={"scenario": {"seed": 841757}})
    reader = _Reader()
    settle, resets, raw = _capture_destination_measurements(
        env=env,
        built=built,
        request={"settle_sample_count": 3, "settle_steps_per_sample": 2},
        collision_bounds={
            "minimum": [-0.165, -0.24, 0.0],
            "maximum": [0.165, 0.24, 0.03],
        },
        support_top_z_m=0.275,
        torch=_Torch,
        reset_reader=_reset,
        sample_reader=reader,
    )

    assert env.reset_seeds == [841757, 841757, 841757]
    assert len(env.actions) == 6
    assert all(action[0][-1] == 0.0 for action in env.actions)
    assert len(settle) == len(resets) == len(raw) == 3
    assert all(row["support_contact_peak_force_n"] == 1.25 for row in settle)
    assert all(row["maximum_penetration_m"] == 0.0 for row in settle)


def test_live_pose_penetration_uses_transformed_destination_bounds() -> None:
    assert _maximum_support_penetration_m(
        pose=[1.0, 2.0, 0.2745, 0.0, 0.0, 0.0, 1.0],
        collision_bounds={
            "minimum": [-0.1, -0.1, 0.0],
            "maximum": [0.1, 0.1, 0.03],
        },
        support_top_z_m=0.275,
    ) == pytest.approx(0.0005)
