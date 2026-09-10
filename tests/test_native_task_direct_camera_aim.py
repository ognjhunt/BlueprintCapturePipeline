"""The camera must face the target and remain rigid on the measured wrist."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from blueprint_pipeline.native_task_direct_camera_aim import (
    install_direct_wrist_camera_aim, pose_matrix, target_facing_attachment,
)


def _install_camera_native_edges(monkeypatch):
    class Array:
        def __init__(self, value):
            self.value = np.asarray(value)
        def numpy(self):
            return self.value
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(
        from_numpy=lambda value, **kwargs: Array(value)))
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(Gf=SimpleNamespace(
        Vec3d=lambda *values: values, Quatd=lambda real, imaginary: (real, imaginary))))
    return Array


def _scene_view(calls=None):
    calls = [] if calls is None else calls
    class View:
        _use_fabric = True
        prim_paths = ['/World/Robot/Gripper/base_link/wrist_camera']
        prims = [SimpleNamespace(GetAttribute=lambda name: SimpleNamespace(
            Set=lambda value: (calls.append(('usd', name, value)), True)[1]))]
        def set_world_poses(self, positions, orientations, indices):
            calls.append('scene_write')
            self.positions, self.orientations = positions, orientations
            self.indices = indices
        def get_world_poses(self, indices):
            assert indices is self.indices
            calls.append('scene_read')
            return self.positions, self.orientations
    return View()


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


def test_view_reads_uncached_physics_and_follows_wrist_with_fixed_mount(monkeypatch):
    _install_camera_native_edges(monkeypatch)
    measured = np.array([[[0., 0., 1., 0., 0., 0., 1.]]])
    old_usd = np.array([[[10., 10., 10., 0., 0., 0., 1.]]])
    calls = []
    robot = SimpleNamespace(data=SimpleNamespace(body_names=['base_link'], body_pose_w=old_usd,
        _root_view=SimpleNamespace(get_link_transforms=lambda: measured)))
    camera = SimpleNamespace(_view=_scene_view(), device='cuda:0',
        cfg=SimpleNamespace(prim_path='/World/Robot/Gripper/base_link/wrist_camera',
            offset=SimpleNamespace(pos=(.011,-.031,-.074)),update_latest_camera_pose=True))
    env = SimpleNamespace()
    env.unwrapped = SimpleNamespace(scene={'robot': robot, 'wrist': camera},
        reset=lambda: None,
        sim=SimpleNamespace(forward=lambda: calls.append('forward'), render=lambda: None))
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
    _install_camera_native_edges(monkeypatch)
    plan = _sealed_scene_plan()
    plan.setdefault('task_spec', {})['start_pose_world'] = [.5, .2, .4, 0, 0, 0, 1]
    authority = {'mode': 'point_at_task_object_then_rigidly_follow_wrist', 'authorized_by': 'test-operator'}
    authority['authority_digest'] = canonical_digest(authority, digest_field='authority_digest')
    plan['operator_wrist_camera_aim'] = authority
    plan = apply_droid_policy_canary_profile(plan)
    original = _ArenaBuilder.make_registered_and_return_cfg
    def native_boundary(self, *, render_mode):
        _, cfg = original(self, render_mode=render_mode)
        camera = SimpleNamespace(_view=_scene_view(), device='cuda:0',
            cfg=SimpleNamespace(prim_path='/World/Robot/Gripper/base_link/wrist_camera',
                offset=SimpleNamespace(pos=(.011,-.031,-.074)),update_latest_camera_pose=True))
        robot = SimpleNamespace(data=SimpleNamespace(body_names=['base_link'],
            _root_view=SimpleNamespace(get_link_transforms=lambda: np.array([[[0,0,1,0,0,0,1]]],dtype=float))))
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene={'robot':robot,'wrist_camera':camera},
            reset=lambda: None, sim=SimpleNamespace(forward=lambda: None, render=lambda: None)))
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


def test_aim_uses_completed_reset_and_rtx_scene_write_before_render(monkeypatch):
    Array = _install_camera_native_edges(monkeypatch)
    constructed = np.array([[[0., 0., 1., 0., 0., 0., 1.]]])
    reset_pose = [1., .2, 1.1, *Rotation.from_euler('xyz', [.5, -.2, .3]).as_quat()]
    target = np.array([-.4, -.2, .3])
    calls, rendered = [], []
    scene = _scene_view(calls)
    camera = SimpleNamespace(_view=scene, device='cpu', cfg=SimpleNamespace(
        prim_path='/World/Robot/Gripper/base_link/wrist_camera',
        offset=SimpleNamespace(pos=(.011,-.031,-.074)),update_latest_camera_pose=True))
    robot = SimpleNamespace(data=SimpleNamespace(body_names=['base_link'],
        _root_view=SimpleNamespace(get_link_transforms=lambda: constructed)))
    def reset():
        calls.append('reset')
        constructed[0,0] = reset_pose
    def render():
        # Like IsaacRtxRenderer, the renderer consumes the scene camera. Its
        # update_camera metadata hook is deliberately irrelevant here.
        calls.append('render')
        rendered.append(pose_matrix([*scene.positions.numpy()[0], *scene.orientations.numpy()[0]]))
    env = SimpleNamespace(unwrapped=SimpleNamespace(reset=reset,
        scene={'robot':robot,'wrist':camera}, sim=SimpleNamespace(forward=lambda: None,render=render)))
    receipt=install_direct_wrist_camera_aim(env=env,camera_name='wrist',target=target)
    assert calls and calls[0]=='reset', 'aim must follow the completed native reset'
    indices=Array([0])
    positions, orientations=camera._view.get_world_poses(indices)
    assert calls[0]=='reset' and calls[-3:]==['scene_write','render','scene_read']
    assert receipt['aim_setup_phase']=='after_native_environment_reset'
    np.testing.assert_allclose(receipt['initial_native_body_pose_xyzw'],reset_pose)
    np.testing.assert_allclose((np.linalg.inv(rendered[0])@np.r_[target,1])[:2],0,atol=1e-6)
    assert positions is scene.positions and orientations is scene.orientations
    # A moving arm changes the actual rendered scene pose while keeping the
    # one fixed mount; it must not silently track the task object.
    constructed[0,0,:3] += [.2, -.1, .05]
    camera._view.get_world_poses(indices)
    np.testing.assert_allclose(np.linalg.inv(pose_matrix(constructed[0,0]))@rendered[-1],
        receipt['body_from_camera_opengl'],atol=1e-6)
    assert not np.allclose((np.linalg.inv(rendered[-1])@np.r_[target,1])[:2],0)


def test_scene_readback_copies_cuda_proxy_through_explicit_warp_accessor(monkeypatch):
    from blueprint_pipeline.native_task_direct_camera_aim import NativeAttachedCameraView
    Array = _install_camera_native_edges(monkeypatch)
    class CudaProxy:
        def __init__(self, value):
            self.warp = Array(value)
        def numpy(self):
            # Exact pinned ProxyArray.__getattr__ delegates this call to the
            # CUDA torch tensor, which refuses implicit host conversion.
            raise TypeError("can't convert cuda:0 device type tensor to numpy")
    scene = _scene_view()
    def readback(indices):
        assert indices is scene.indices
        return (CudaProxy(scene.positions.numpy()), CudaProxy(scene.orientations.numpy()))
    scene.get_world_poses = readback
    robot = SimpleNamespace(data=SimpleNamespace(_root_view=SimpleNamespace(
        get_link_transforms=lambda: CudaProxy([[[0.,0.,1.,0.,0.,0.,1.]]]))))
    view = NativeAttachedCameraView(scene, robot, 0, np.eye(4), 'cuda:0', lambda: None, lambda: None)
    positions, quaternions = view.get_world_poses(Array([0]))
    np.testing.assert_allclose(positions.warp.numpy(), [[0.,0.,1.]])
    np.testing.assert_allclose(quaternions.warp.numpy(), [[0.,0.,0.,1.]])


def test_native_tensor_copy_moves_to_host_before_numpy():
    from blueprint_pipeline.native_task_direct_camera_aim import _host_numpy
    calls = []
    expected = np.array([[1.,2.,3.]])
    class Tensor:
        def detach(self):
            calls.append('detach')
            return self
        def cpu(self):
            calls.append('cpu')
            return SimpleNamespace(numpy=lambda: expected)
        def numpy(self):
            raise TypeError('CUDA tensor has no direct NumPy view')
    np.testing.assert_array_equal(_host_numpy(Tensor()), expected)
    assert calls == ['detach', 'cpu']
