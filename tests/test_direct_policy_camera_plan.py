from copy import deepcopy

import pytest
from blueprint_pipeline import native_task_camera_start_configuration as camera
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_camera_start_construction_materializer import inputs


def plan_inputs():
    value = inputs()
    value.pop('construction')
    value['plan']['robot']['joint_reset_positions_rad'] = deepcopy(value['source_binding']['joint_reset_positions_rad'])
    value['plan']['plan_digest'] = canonical_digest(value['plan'], digest_field='plan_digest')
    return value


def test_direct_camera_binding_needs_no_scripted_construction():
    value = plan_inputs()
    before = deepcopy(value)
    binding = camera.materialize_camera_start_from_plan(**value)
    assert binding['joint_reset_positions_rad'] == value['plan']['robot']['joint_reset_positions_rad']
    assert binding['source_scene_plan_digest'] == value['plan']['plan_digest']
    assert 'construction_result_digest' not in binding
    assert binding['native_qualification_claimed'] is binding['historical_scene_visibility_adopted'] is False
    assert binding['reset_authority'] == 'configured_robot_plan_requires_native_readback'
    assert value == before


@pytest.mark.parametrize('fault', ['plan_digest', 'joint_missing', 'joint_limit', 'robot_asset', 'reference_pose', 'bad_view'])
def test_direct_camera_still_validates_geometry_and_calibration(fault):
    value = plan_inputs()
    if fault == 'plan_digest':
        value['plan']['plan_digest'] = 'sha256:' + '0' * 64
    elif fault == 'joint_missing':
        value['plan']['robot']['joint_reset_positions_rad'].pop('panda_joint1')
    elif fault == 'joint_limit':
        value['plan']['robot']['joint_reset_positions_rad']['panda_joint4'] = 0
    elif fault == 'robot_asset':
        value['robot_asset_sha256'] = 'sha256:' + '0' * 64
    elif fault == 'reference_pose':
        value['native_reference_gate']['snapshot']['cameras'][0]['position_world_m'][0] += 1
    else:
        value['plan']['task_spec']['start_pose_world'][2] += 10
    if fault != 'plan_digest':
        value['plan']['plan_digest'] = canonical_digest(value['plan'], digest_field='plan_digest')
    with pytest.raises((ValueError, KeyError, TypeError)):
        camera.materialize_camera_start_from_plan(**value)


def test_small_wrist_adjustment_frames_task_without_changing_base_or_claims():
    value = plan_inputs()
    plan = value['plan']
    plan['robot']['joint_reset_positions_rad']['panda_joint6'] += .4
    plan['plan_digest'] = canonical_digest(plan, digest_field='plan_digest')
    before = deepcopy(value)
    assert camera.camera_framing_report(plan, value['source_binding']['source_joint_chain'],
        plan['robot']['joint_reset_positions_rad'])['status'] == 'blocked'
    result = camera.materialize_camera_start_from_plan(**value)
    assert result['reset_adjustment'] == {'joint': 'panda_joint6', 'delta_rad': -.1,
        'reason': 'task_outside_wrist_camera_at_selected_reset', 'native_validated': False}
    assert result['robot_base_pose_world'] == plan['robot']['base_pose_world']
    assert result['native_qualification_claimed'] is False
    assert value == before
    assert camera.validate_camera_start_configuration(plan, result) == result
