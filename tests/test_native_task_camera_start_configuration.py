"""Final reset/camera checks reproduce the failed wrist view without Isaac."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.droid_policy_canary_embodiment import (
    apply_droid_policy_canary_profile,
    DROID_NATIVE_RESET_JOINTS_RAD,
)
from blueprint_pipeline.native_task_camera_start_configuration import (
    camera_framing_report,
    external_camera_offset_position,
    resolved_camera_matrices,
    task_framing_points,
    task_external_camera_position,
    validate_camera_start_configuration,
)


def fixture():
    return json.loads(
        (Path(__file__).parent / "fixtures/policy_camera_start_configuration.json").read_text()
    )


def test_source_joint_camera_prediction_preserves_task_aligned_start():
    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    assert validate_camera_start_configuration(plan, binding) == binding
    resolved = apply_droid_policy_canary_profile(plan)
    assert resolved["robot"]["joint_reset_positions_rad"] == binding["joint_reset_positions_rad"]
    assert resolved["robot"]["joint_reset_positions_rad"] != DROID_NATIVE_RESET_JOINTS_RAD
    assert (
        resolved["policy_canary_embodiment_profile"]["preserve_official_policy_camera_calibration"]
        is True
    )
    assert (
        resolved["policy_canary_embodiment_profile"]["preserve_official_reset_joint_positions"]
        is False
    )
    assert (
        camera_framing_report(
            resolved, binding["source_joint_chain"], binding["joint_reset_positions_rad"]
        )["status"]
        == "passed"
    )
    # The predecessor's later default-reset override loses the task at the wrist.
    failed = camera_framing_report(
        plan, binding["source_joint_chain"], DROID_NATIVE_RESET_JOINTS_RAD
    )
    assert failed["status"] == "blocked"
    assert (
        next(row for row in failed["views"] if row["camera_role"] == "wrist")[
            "projected_center_uv"
        ][1]
        > 720
    )


def test_camera_perturbation_preserves_the_custom_start_sealed_camera_plan():
    from blueprint_pipeline.policy_canary_scene_resolution import resolved_scene_plan

    plan = fixture()
    plan.update(task_kind="rigid_pick_place",
        objects=[{"name": "task_object", "task_subject": True}],
        cadence={"control_frequency_hz": 15., "physics_frequency_hz": 120.,
                 "physics_dt_seconds": 1/120, "maximum_action_steps": 30, "settle_window_samples": 4})
    cell = {"cell_id": "sealed-custom-camera", "seed": 74,
        "resolved_scenario": {"family": "camera_sensor", "parameters": {"external_camera_x_delta_m": -.0113}}}
    resolved = resolved_scene_plan(plan, cell, task_success_contract=plan["task_spec"]["task_success_contract"])
    assert resolved["cameras"] == plan["cameras"]
    binding = plan["policy_canary_camera_start_configuration"]
    assert validate_camera_start_configuration(resolved, binding) == binding
    application = resolved["scenario"]["parameter_applications"][0]
    assert application["expected_native_value"] == external_camera_offset_position(resolved)[0]
    assert application["application_tolerance"] == 0


def test_articulated_framing_uses_assembly_extent_and_frozen_stroke():
    plan = fixture()
    plan['task_kind'] = 'articulated_open_close'
    plan['objects'] = [{
        'task_subject': True, 'object_type': 'ARTICULATION',
        'pose_world': {'position_world_m': [-0.6, 0.1, 0.0],
                       'orientation_xyzw': [0.0, 0.0, 0.0, 1.0]},
    }]
    plan['task_object_observability'] = {'task_object_extent_m': [0.54, 0.4, 0.58]}
    plan['task_spec']['interaction_affordance'] = {
        'contact_point_link_m': [0.25, 0.0, 0.08], 'pull_unit_asset_root': [1.0, 0.0, 0.0],
    }
    plan['task_spec']['executable_opening_threshold'] = {
        'joint_type': 'prismatic', 'success_interval': [0.23, 0.38],
    }
    points = task_framing_points(plan)
    assert points['subject_start'] == pytest.approx([-0.35, 0.1, 0.29])
    assert points['subject_destination'] == pytest.approx([-0.12, 0.1, 0.29])
    plan['objects'][0]['object_type'] = 'RIGID'
    with pytest.raises(ValueError, match='policy_camera_articulated_subject_invalid'):
        task_framing_points(plan)


@pytest.mark.parametrize("direction", [1., -1.])
def test_hinged_framing_uses_registered_arc_in_the_scene_frame(direction):
    import math

    plan = fixture()
    plan['task_kind'] = 'articulated_open_close'
    plan['objects'] = [{
        'task_subject': True, 'object_type': 'ARTICULATION',
        'pose_world': {'position_world_m': [2., 3., 1.],
                       'orientation_xyzw': [0., 0., math.sqrt(.5), math.sqrt(.5)]},
    }]
    plan['task_object_observability'] = {'task_object_extent_m': [.5, .6, .8]}
    plan['task_spec']['executable_opening_threshold'] = {
        'joint_type': 'revolute', 'target_joint_id': 'hinge',
        'reset_position': .2, 'success_interval': [.2 + direction*math.pi/2, .2 + direction*math.pi/2],
    }
    affordance = {
        'contact_point_link_m': [0., 0., .8], 'pull_unit_asset_root': [1., 0., 0.],
        'joint_contact_path': [
            {'joint_positions': {'hinge': .2 + direction*angle},
             'contact_pose_asset_root': {'position_m': [math.sin(angle), 0., math.cos(angle)]}}
            for angle in (0., math.pi/4, math.pi/2)
        ],
    }
    affordance['affordance_digest'] = canonical_digest(affordance, digest_field='affordance_digest')
    plan['task_spec']['interaction_affordance'] = affordance
    points = task_framing_points(plan)
    assert points['subject_start'] == pytest.approx([2., 3., 2.])
    assert points['subject_destination'] == pytest.approx([2., 4., 1.])
    assert points['subject_path_01'] == pytest.approx([2., 3.+math.sqrt(.5), 1.+math.sqrt(.5)])

    # A path for a different requested opening cannot silently drive camera selection.
    plan['task_spec']['executable_opening_threshold']['success_interval'][0] += .1
    with pytest.raises(ValueError, match='contact_path_invalid'):
        task_framing_points(plan)
    plan['task_spec']['executable_opening_threshold']['success_interval'][0] -= .1
    affordance['joint_contact_path'][1]['contact_pose_asset_root']['position_m'][0] += .1
    with pytest.raises(ValueError, match='contact_path_invalid'):
        task_framing_points(plan)


def test_fixed_cameras_frame_the_arc_while_reset_wrist_checks_current_contact(monkeypatch):
    import numpy as np
    import blueprint_pipeline.native_task_camera_start_configuration as cameras

    plan = fixture()
    monkeypatch.setattr(cameras, 'task_framing_points', lambda _: {
        'subject_start': [0., 0., -2.], 'subject_destination': [.1, 0., -2.],
        'subject_path_01': [100., 0., -2.],
    })
    monkeypatch.setattr(cameras, 'resolved_camera_matrices', lambda *_: {
        role: (np.eye(4), [[100., 0., 640.], [0., 100., 360.], [0., 0., 1.]])
        for role in ('external', 'wrist', 'overview')
    })
    report = camera_framing_report(plan, [], {})
    assert report['status'] == 'blocked'
    assert [row['subject'] for row in report['views'] if row['camera_role'] == 'wrist'] == ['subject_start']
    assert all(not row['passed'] for row in report['views'] if row['subject'] == 'subject_path_01')


def test_scene_camera_translation_reaches_policy_profile_without_changing_wrist_calibration():
    import numpy as np

    plan = fixture()
    original = deepcopy(plan)
    binding = deepcopy(plan['policy_canary_camera_start_configuration'])
    binding['external_camera_position_parent_m'] = task_external_camera_position(plan)
    binding['external_camera_placement_source'] = 'sealed_task_motion_path'
    binding['configuration_digest'] = canonical_digest(binding, digest_field='configuration_digest')
    plan['policy_canary_camera_start_configuration'] = binding
    assert validate_camera_start_configuration(plan, binding) == binding
    profile = apply_droid_policy_canary_profile(plan)['policy_canary_embodiment_profile']
    assert profile['preserve_official_external_camera_translation'] is False
    assert profile['preserve_official_policy_camera_intrinsics'] is True
    assert original['cameras'] == plan['cameras']
    before = resolved_camera_matrices(plan, binding['source_joint_chain'], binding['joint_reset_positions_rad'])
    assert camera_framing_report(plan, binding['source_joint_chain'], binding['joint_reset_positions_rad'])['status'] == 'passed'
    plan['scenario'] = {'parameters': {'external_camera_x_delta_m': .02}}
    after = resolved_camera_matrices(plan, binding['source_joint_chain'], binding['joint_reset_positions_rad'])
    np.testing.assert_allclose(after['external'][0][:3, 3]-before['external'][0][:3, 3], [.02, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(after['wrist'][0], before['wrist'][0])
    assert after['external'][1] == before['external'][1]


def test_scene_camera_translation_recovers_an_elevated_handle_path(monkeypatch):
    import blueprint_pipeline.native_task_camera_start_configuration as cameras

    plan = fixture()
    binding = plan['policy_canary_camera_start_configuration']
    monkeypatch.setattr(cameras, 'task_framing_points', lambda _: {
        'subject_start': [0., 0., 1.4], 'subject_path_01': [.3, 0., 1.2],
        'subject_destination': [.5, 0., .9],
    })
    before = camera_framing_report(plan, binding['source_joint_chain'], binding['joint_reset_positions_rad'])
    assert any(not row['passed'] for row in before['views'] if row['camera_role'] == 'external')
    binding['external_camera_position_parent_m'] = task_external_camera_position(plan)
    after = camera_framing_report(plan, binding['source_joint_chain'], binding['joint_reset_positions_rad'])
    assert all(row['passed'] for row in after['views'] if row['camera_role'] == 'external')


@pytest.mark.parametrize(
    "fault", ["base", "camera", "joint_limit", "source_chain", "reference", "digest", "task"]
)
def test_start_binding_refuses_configuration_drift(fault):
    plan = fixture()
    binding = deepcopy(plan["policy_canary_camera_start_configuration"])
    if fault == "base":
        plan["robot"]["base_pose_world"]["position_world_m"][0] += 0.1
    if fault == "camera":
        plan["cameras"][-1]["frame_from_camera_matrix"][3] += 0.1
    if fault == "joint_limit":
        binding["joint_reset_positions_rad"]["panda_joint4"] = 0
    if fault == "source_chain":
        binding["source_joint_chain"][0]["local0"][0][3] += 0.1
    if fault == "reference":
        binding["native_reference"]["world_from_wrist_camera_opengl"][2][3] += 0.1
    if fault == "task":
        binding["task_success_contract_digest"] = "sha256:" + "b" * 64
    if fault != "digest":
        binding["configuration_digest"] = canonical_digest(
            binding, digest_field="configuration_digest"
        )
    else:
        binding["configuration_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="policy_camera"):
        validate_camera_start_configuration(plan, binding)


def test_resealed_default_reset_is_refused_before_gpu():
    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    binding["joint_reset_positions_rad"] = dict(DROID_NATIVE_RESET_JOINTS_RAD)
    binding["configuration_digest"] = canonical_digest(binding, digest_field="configuration_digest")
    with pytest.raises(ValueError, match="final_reset_does_not_frame_task"):
        apply_droid_policy_canary_profile(plan)


def test_preregistered_camera_shift_reaches_the_official_parent_frame():
    import numpy as np
    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    nominal = resolved_camera_matrices(plan, binding["source_joint_chain"], binding["joint_reset_positions_rad"])["external"][0]
    plan["scenario"] = {"parameters": {"external_camera_x_delta_m": 0.02}}
    shifted = resolved_camera_matrices(plan, binding["source_joint_chain"], binding["joint_reset_positions_rad"])["external"][0]
    np.testing.assert_allclose(shifted[:3, 3]-nominal[:3, 3], [0.02, 0., 0.], atol=1e-12)
    # The base faces 180 degrees, so a world-positive shift is parent-negative.
    assert external_camera_offset_position(plan)[0] == pytest.approx(0.03)
    assert validate_camera_start_configuration(plan, binding) == binding


def test_the_rendered_external_camera_is_arenas_at_the_binding_offset_not_the_plan_row():
    import numpy as np

    from blueprint_pipeline.native_task_camera_start_configuration import runtime_external_camera_row

    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    binding["external_camera_position_parent_m"] = task_external_camera_position(plan)
    binding["configuration_digest"] = canonical_digest(binding, digest_field="configuration_digest")
    resolved = apply_droid_policy_canary_profile(plan)
    row = runtime_external_camera_row(resolved)
    frame, k = resolved_camera_matrices(resolved, binding["source_joint_chain"],
                                        binding["joint_reset_positions_rad"])["external"]
    opencv = np.asarray(row["frame_from_camera_matrix"]).reshape(4, 4)
    np.testing.assert_allclose(opencv[:3, 3], frame[:3, 3])
    # Every task point lands on the same pixel through the OpenCV row as through Arena's OpenGL frame.
    intrinsics = row["intrinsics"]
    for point in task_framing_points(resolved).values():
        gl = frame[:3, :3].T @ (np.asarray(point) - frame[:3, 3])
        cv = opencv[:3, :3].T @ (np.asarray(point) - opencv[:3, 3])
        assert cv[2] == pytest.approx(-gl[2])
        assert [intrinsics["cx"] + intrinsics["fx"] * cv[0] / cv[2],
                intrinsics["cy"] + intrinsics["fy"] * cv[1] / cv[2]] == pytest.approx(
            [k[0][2] + k[0][0] * gl[0] / -gl[2], k[1][2] - k[1][1] * gl[1] / -gl[2]])
    plan_row = next(camera for camera in resolved["cameras"] if camera["role"] == "external")
    assert not np.allclose(opencv, np.asarray(plan_row["frame_from_camera_matrix"]).reshape(4, 4))
    # Without the DROID profile the runtime renders the plan's own row.
    assert runtime_external_camera_row(fixture()) == plan_row


def test_start_refuses_a_hidden_external_camera_measuring_each_camera_once():
    from blueprint_pipeline.native_task_camera_start_configuration import require_external_camera_scene_view

    nominal = apply_droid_policy_canary_profile(fixture())
    shifted = deepcopy(nominal)
    shifted["scenario"] = {"parameters": {"external_camera_x_delta_m": 0.02}}
    seen = []

    def clear(row, plan):
        seen.append((row["frame_from_camera_matrix"], plan))
        return []

    # Ten cells with two camera positions cost two measurements.
    assert require_external_camera_scene_view([nominal] * 5 + [shifted] * 5, clear) == 2
    assert len(seen) == 2
    hidden = seen[1][0]
    with pytest.raises(ValueError, match="policy_camera_external_scene_view_blocked:"
                                         "scene_camera_contact_occluded_by_scene:external:40%"):
        require_external_camera_scene_view([nominal, shifted], lambda row, plan: (
            ["scene_camera_contact_occluded_by_scene:external:40%"]
            if row["frame_from_camera_matrix"] == hidden else []))
    # A cell that turns the task under the same camera is measured on its own, with its own plan.
    turned = deepcopy(nominal)
    turned["objects"] = [{"task_subject": True, "object_type": "RIGID",
                          "pose_world": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.38, 0.92]}}]
    seen.clear()
    assert require_external_camera_scene_view([nominal, turned, turned], clear) == 2
    assert [plan is turned for _, plan in seen] == [False, True]
