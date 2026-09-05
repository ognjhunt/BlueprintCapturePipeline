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


def test_closed_gripper_contact_flicker_is_retained_as_gap_not_regrasp():
    telemetry = NativeRigidEpisodeTelemetry(_spec())
    telemetry.begin_episode()
    telemetry.observe(_sample())
    missing_contact = {**_sample(), "task_contact_active": False}
    telemetry.observe(missing_contact)
    recovered_contact = _sample()
    telemetry.observe(recovered_contact)
    assert recovered_contact["regrasp_count"] == 0
    assert recovered_contact["closed_grasp_contact_gap_count"] == 1
    # A measured reopen with cleared contact, then a measured closed
    # acquisition, does establish an actual second grasp.
    telemetry.observe({**_sample(), "task_contact_active": False, "gripper_width_m": .08})
    telemetry.observe(recovered_contact)
    assert recovered_contact["regrasp_count"] == 1
    assert recovered_contact["closed_grasp_contact_gap_count"] == 1


def test_dropped_then_recovered_closed_grasp_still_fails_no_drop_without_regrasp():
    import copy

    from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec, seal_rigid_task_success_contract
    from tests.test_adp_task_scoring import _dropped_then_placed_samples, _rigid_v2_sample, _rigid_v2_spec

    spec = _rigid_v2_spec()
    rows = _dropped_then_placed_samples()
    # The hand catches the dropped object without a measured reopen. It later
    # releases onto support and clears contact over a full final settle window.
    rows[3].update(task_contact_active=True, gripper_width_m=.03)
    rows.append(_rigid_v2_sample(6, [1.15, 2., .8]))
    baseline = score_task_episode_from_spec(task_spec=spec, samples=rows)
    criteria = copy.deepcopy(baseline["task_success_contract"]["criteria"])
    criteria["temporal_invariants"]["no_drop"]["mode"] = "required"
    criteria["temporal_invariants"]["maximum_regrasps"] = 0
    spec["task_success_contract"] = seal_rigid_task_success_contract(
        task_spec=spec, site_id="fixture_scene", task_id="fixture_task",
        author_source="task_owner", author_id="fixture_owner", confirmation_status="confirmed",
        confirmed_by_team_id="fixture_owner", criteria=criteria)
    telemetry = NativeRigidEpisodeTelemetry(_spec())
    telemetry.begin_episode()
    for row in rows:
        retained = {**_sample(), **row}
        telemetry.observe(retained)
        row.update(retained)
    report = score_task_episode_from_spec(task_spec=spec, samples=rows)
    assert report["task_succeeded"] is False
    assert report["failed_criteria"] == ["no_drop"]
    assert report["event_ledger"]["drop_events"]
    assert report["event_ledger"]["maximum_regrasps_observed"] == 0
    assert rows[-1]["closed_grasp_contact_gap_count"] == 1
