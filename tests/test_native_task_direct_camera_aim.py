"""The camera must face the target and remain rigid on the measured wrist."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from blueprint_pipeline.native_task_direct_camera_aim import (
    install_direct_wrist_camera_aim, pose_matrix, target_facing_attachment,
)


def test_initial_aim_centers_known_object_without_changing_camera_position():
    body = [-1.5, -3.2, .7, *Rotation.from_euler('xyz', [.2, -.7, 1.1]).as_quat()]
    offset = [.011, -.031, -.074]
    target = np.array([-2., -3.44, .286])
    attachment = target_facing_attachment(body, offset, target)
    world = pose_matrix(body) @ attachment
    local_target = np.linalg.inv(world) @ np.r_[target, 1]
    np.testing.assert_allclose(local_target[:2], 0, atol=1e-12)
    assert local_target[2] < 0  # OpenGL optical axis
    np.testing.assert_allclose(attachment[:3, 3], offset, atol=1e-12)


def test_view_reads_uncached_physics_and_follows_wrist_with_fixed_mount():
    measured = np.array([[[0., 0., 1., 0., 0., 0., 1.]]])
    old_usd = np.array([[[10., 10., 10., 0., 0., 0., 1.]]])
    calls = []
    robot = SimpleNamespace(data=SimpleNamespace(body_names=['base_link'], body_pose_w=old_usd,
        _root_view=SimpleNamespace(get_link_transforms=lambda: measured)))
    camera = SimpleNamespace(_view=SimpleNamespace(_use_fabric=True), device='cuda:0',
        cfg=SimpleNamespace(prim_path='/World/Robot/Gripper/base_link/wrist_camera',
            offset=SimpleNamespace(pos=(.011,-.031,-.074)),update_latest_camera_pose=True))
    env = SimpleNamespace()
    env.unwrapped = SimpleNamespace(scene={'robot': robot, 'wrist': camera},
        sim=SimpleNamespace(forward=lambda: calls.append('forward')))
    target = np.array([-.4, -.2, .3])
    receipt = install_direct_wrist_camera_aim(env=env,camera_name='wrist',target=target)
    first = camera._view.world_matrices()[0]
    np.testing.assert_allclose((np.linalg.inv(first) @ np.r_[target,1])[:2],0,atol=1e-12)
    measured[0,0] = [1., .2, 1.1, *Rotation.from_euler('z',.3).as_quat()]
    moved = camera._view.world_matrices()[0]
    np.testing.assert_allclose(np.linalg.inv(pose_matrix(measured[0,0])) @ moved,
        np.asarray(receipt['body_from_camera_opengl']),atol=1e-12)
    assert not np.allclose(moved, first)
    assert not np.allclose((np.linalg.inv(moved) @ np.r_[target,1])[:2],0)
    assert len(calls)==3 and receipt['tracks_object_after_reset'] is False
    assert receipt['official_mount_orientation_preserved'] is False


def test_aim_refuses_camera_on_the_object():
    with pytest.raises(RuntimeError,match='target_degenerate'):
        target_facing_attachment([0,0,0,0,0,0,1],[0,0,0],[0,0,0])


def test_actual_environment_builder_installs_target_aim_and_records_changed_orientation(monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.droid_policy_canary_embodiment import apply_droid_policy_canary_profile
    from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
    from tests.test_native_task_arena_runtime import _ArenaBuilder, _install_fake_native_runtime, _sealed_scene_plan
    _install_fake_native_runtime(monkeypatch)
    plan = _sealed_scene_plan()
    plan.setdefault('task_spec', {})['start_pose_world'] = [.5, .2, .4, 0, 0, 0, 1]
    authority = {'mode': 'point_at_task_object_then_rigidly_follow_wrist', 'authorized_by': 'test-operator'}
    authority['authority_digest'] = canonical_digest(authority, digest_field='authority_digest')
    plan['operator_wrist_camera_aim'] = authority
    plan = apply_droid_policy_canary_profile(plan)
    original = _ArenaBuilder.make_registered_and_return_cfg
    def native_boundary(self, *, render_mode):
        _, cfg = original(self, render_mode=render_mode)
        camera = SimpleNamespace(_view=SimpleNamespace(_use_fabric=True), device='cuda:0',
            cfg=SimpleNamespace(prim_path='/World/Robot/Gripper/base_link/wrist_camera',
                offset=SimpleNamespace(pos=(.011,-.031,-.074)),update_latest_camera_pose=True))
        robot = SimpleNamespace(data=SimpleNamespace(body_names=['base_link'],
            _root_view=SimpleNamespace(get_link_transforms=lambda: np.array([[[0,0,1,0,0,0,1]]],dtype=float))))
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene={'robot':robot,'wrist_camera':camera},
            sim=SimpleNamespace(forward=lambda: None)))
        return env,cfg
    monkeypatch.setattr(_ArenaBuilder,'make_registered_and_return_cfg',native_boundary)
    built=build_native_task_arena_environment(plan)
    assert built.plan['policy_canary_embodiment_profile']['preserve_official_policy_camera_calibration'] is False
    assert built.plan['policy_canary_embodiment_profile']['preserve_official_policy_camera_intrinsics'] is True
    receipt=built.native_configuration_readback['direct_wrist_camera_aim']
    assert receipt['operator_authority']==authority
    camera=built.env.unwrapped.scene['wrist_camera']
    world=camera._view.world_matrices()[0]
    target=np.array([.5,.2,.4,1])
    np.testing.assert_allclose((np.linalg.inv(world)@target)[:2],0,atol=1e-12)
    assert built.native_configuration_readback['cameras']['wrist']['calibration_source']=='operator_requested_target_facing_rigid_mount'
