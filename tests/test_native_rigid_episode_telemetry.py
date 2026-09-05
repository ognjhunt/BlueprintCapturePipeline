"""Native telemetry uses observed poses, forces and episode execution history."""
import pytest

from blueprint_pipeline.native_rigid_episode_telemetry import NativeRigidEpisodeTelemetry
from blueprint_pipeline.native_task_episode_environment import NativeRigidScoringEnvironment
from tests.test_native_task_episode_environment import (
    _RigidEpisodeEnvironment, _RigidNativeReadback, _rigid_scoring_task_spec,
)


def _spec():
    spec = _rigid_scoring_task_spec()
    spec.update(
        release_gripper_width_min_m=.07,
        robot_workspace_position_bounds_world_m={"minimum": [-1., -1., -1.], "maximum": [3., 3., 3.]},
        subject_collision_bounds_scoring_frame_m={"minimum": [-.1]*3, "maximum": [.1]*3},
        destination_support_asset_id="tray",
    )
    return spec


def _sample():
    return {
        "task_object_pose_world": [1., 2., .85, 0., 0., 0., 1.],
        "grasp_frame_position_world_m": [1., 2., 1.],
        "gripper_width_m": .03, "task_contact_active": True,
        "robot_scene_contact_peak_force_n": 0., "robot_task_forbidden_collision_peak_force_n": 0.,
        "task_scene_collision_peak_force_n": 0., "destination_scene_forbidden_contact_peak_force_n": 0.,
    }


def test_temporal_counts_derive_from_contact_acquisition_and_post_initial_reset_execution():
    telemetry = NativeRigidEpisodeTelemetry(_spec())
    initial = _sample()
    telemetry.observe(initial)
    assert initial["retry_count"] is None
    telemetry.begin_episode()
    telemetry.observe(initial)
    assert initial["retry_count"] == initial["regrasp_count"] == 0
    released = {**_sample(), "task_contact_active": False, "gripper_width_m": .08}
    telemetry.observe(released)
    reacquired = _sample()
    telemetry.observe(reacquired)
    assert reacquired["regrasp_count"] == 1
    telemetry.reset_executed()
    telemetry.observe(_sample())
    row = _sample()
    telemetry.observe(row)
    assert row["retry_count"] == 1
    telemetry.begin_episode()
    telemetry.observe(row)
    assert row["retry_count"] == row["regrasp_count"] == 0


def test_contact_classes_are_filtered_force_measurements_and_missing_channels_remain_unknown():
    telemetry = NativeRigidEpisodeTelemetry(_spec())
    row = _sample()
    row["destination_scene_forbidden_contact_peak_force_n"] = 6.
    telemetry.observe(row)
    assert row["contact_classes_active"] == ["destination_background"]
    del row["robot_scene_contact_peak_force_n"]
    telemetry.observe(row)
    assert row["contact_classes_active"] is None


def test_workspace_checks_oriented_collision_corners_and_measured_gripper():
    telemetry = NativeRigidEpisodeTelemetry(_spec())
    row = _sample()
    telemetry.observe(row)
    assert row["workspace_excursion"] is False
    row["task_object_pose_world"][2] = .75  # center inside, bottom corner outside
    telemetry.observe(row)
    assert row["workspace_excursion"] is True
    row = _sample()
    row["grasp_frame_position_world_m"][0] = 4.
    telemetry.observe(row)
    assert row["workspace_excursion"] is True
    row.pop("grasp_frame_position_world_m")
    telemetry.observe(row)
    assert row["workspace_excursion"] is None


@pytest.mark.parametrize("native_source,expected", [
    ("native_inner_finger_body_origin_midpoint", [1., 2., 1.046]),
    ("native_franka_pose_servo.live_physical_pad_centers", [1., 2., 1.05]),
])
def test_native_overlay_keeps_calibrated_measured_grasp_instead_of_raw_body_origin(native_source, expected):
    base = _RigidEpisodeEnvironment()
    raw_reader = base.read_object_sample
    base.read_object_sample = lambda: {**raw_reader(), "grasp_frame_position_world_m": [1., 2., 1.046]}
    env = NativeRigidScoringEnvironment(
        environment=base, task_spec=_spec(), task_readback=_RigidNativeReadback(
            grasp_frame_position_world_m=[1., 2., 1.05], grasp_frame_position_source=native_source,
            destination_scene_forbidden_contact_peak_force_n=0.),
    )
    env.reset()
    env.begin_episode()
    sample = env.read_object_sample()
    assert sample["grasp_frame_position_world_m"] == expected
    assert sample["native_grasp_frame_position_world_m"] == [1., 2., 1.05]
    assert sample["retry_count"] == 0
    env.reset()
    assert env.read_object_sample()["retry_count"] == 1
