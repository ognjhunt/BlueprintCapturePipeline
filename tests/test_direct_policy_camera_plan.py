from copy import deepcopy
import hashlib

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
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


def test_fixture_desk_intersection_is_screened_before_policy_rental(tmp_path):
    value = plan_inputs()
    plan = value['plan']
    stage = Usd.Stage.CreateNew(str(tmp_path / 'fixture.usda'))
    root = UsdGeom.Xform.Define(stage, '/Root')
    stage.SetDefaultPrim(root.GetPrim())
    desk = UsdGeom.Mesh.Define(stage, '/Root/desk')
    desk.CreatePointsAttr([Gf.Vec3f(x, y, z) for x in (-1.0, 1.0)
                           for y in (-0.4, 0.4) for z in (0.8, 0.85)])
    desk.CreateFaceVertexCountsAttr([3, 3])
    desk.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
    UsdPhysics.CollisionAPI.Apply(desk.GetPrim())
    floor = UsdGeom.Mesh.Define(stage, '/Root/floor')
    floor.CreatePointsAttr([Gf.Vec3f(-2, -2, 0), Gf.Vec3f(2, -2, 0),
                            Gf.Vec3f(-2, 2, 0), Gf.Vec3f(2, 2, 0)])
    floor.CreateFaceVertexCountsAttr([3, 3])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    stage.GetRootLayer().Save()
    path = tmp_path / 'fixture.usda'
    plan['objects'] = [{'semantic_role': 'scene_collision',
                        'sha256': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest(),
                        'pose_world': {'position_world_m': [0.0, 0.0, 0.0],
                                       'orientation_xyzw': [0.0, 0.0, 0.0, 1.0]}}]
    chain = value['source_binding']['source_joint_chain']
    original = plan['robot']['joint_reset_positions_rad']
    blocked = camera.fixture_reset_clearance(plan, chain, original, path)
    safe = dict(original)
    safe['panda_joint4'] -= 0.5
    passed = camera.fixture_reset_clearance(plan, chain, safe, path)
    assert blocked['status'] == 'blocked'
    assert passed['status'] == 'passed'
    assert passed['minimum_link_centerline_to_obstacle_m'] > 0.13
    assert passed['native_collision_qualified'] is False


def test_fixture_task_asset_intersection_blocks_a_desk_clear_reset(tmp_path):
    value = plan_inputs()
    plan = value['plan']
    chain = value['source_binding']['source_joint_chain']
    joints = dict(plan['robot']['joint_reset_positions_rad'])
    desk_stage = Usd.Stage.CreateNew(str(tmp_path / 'desk.usda'))
    desk_root = UsdGeom.Xform.Define(desk_stage, '/Root')
    desk_stage.SetDefaultPrim(desk_root.GetPrim())
    desk = UsdGeom.Mesh.Define(desk_stage, '/Root/desk')
    desk.CreatePointsAttr([Gf.Vec3f(-0.05, -0.05, 0.8), Gf.Vec3f(0.05, -0.05, 0.8),
                           Gf.Vec3f(-0.05, 0.05, 0.85)])
    desk.CreateFaceVertexCountsAttr([3])
    desk.CreateFaceVertexIndicesAttr([0, 1, 2])
    desk.AddTranslateOp().Set(Gf.Vec3d(10., 0., 0.))
    UsdPhysics.CollisionAPI.Apply(desk.GetPrim())
    desk_stage.GetRootLayer().Save()
    desk_path = tmp_path / 'desk.usda'
    task_stage = Usd.Stage.CreateNew(str(tmp_path / 'task.usda'))
    task_root = UsdGeom.Xform.Define(task_stage, '/Root')
    task_stage.SetDefaultPrim(task_root.GetPrim())
    task = UsdGeom.Cube.Define(task_stage, '/Root/cabinet')
    task.CreateSizeAttr(0.1)
    task_stage.GetRootLayer().Save()
    task_path = tmp_path / 'task.usda'
    body = camera.joint_body_poses(chain, joints, plan['robot']['base_pose_world'])[camera.BODY]
    plan['objects'] = [
        {'semantic_role': 'scene_collision',
         'sha256': 'sha256:' + hashlib.sha256(desk_path.read_bytes()).hexdigest(),
         'pose_world': {'position_world_m': [0., 0., 0.],
                        'orientation_xyzw': [0., 0., 0., 1.]}},
        {'task_subject': True, 'sha256': 'sha256:' + hashlib.sha256(task_path.read_bytes()).hexdigest(),
         'pose_world': {'position_world_m': body[:3, 3].tolist(),
                        'orientation_xyzw': [0., 0., 0., 1.]}},
    ]
    report = camera.fixture_reset_clearance(plan, chain, joints, desk_path, task_path)
    assert report['minimum_link_centerline_to_obstacle_m'] > 0.13
    assert report['minimum_link_centerline_to_task_m'] == 0.0
    assert report['status'] == 'blocked'
    plan['objects'][1]['sha256'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError, match='policy_camera_fixture_task_binding_invalid'):
        camera.fixture_reset_clearance(plan, chain, joints, desk_path, task_path)
