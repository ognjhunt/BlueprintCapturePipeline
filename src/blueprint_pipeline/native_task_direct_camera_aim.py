"""Operator-directed initial camera aim, attached to measured native link poses."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(pose[3:7], dtype=float)).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def target_facing_attachment(body_pose, offset_position, target):
    body = pose_matrix(body_pose)
    eye = (body @ np.r_[offset_position, 1.0])[:3]
    forward = np.asarray(target, dtype=float) - eye
    distance = np.linalg.norm(forward)
    if not np.isfinite(distance) or distance < 0.01:
        raise RuntimeError('native_camera_aim_target_degenerate')
    forward /= distance
    up = np.array([0., 0., 1.])
    if abs(float(forward @ up)) > .99:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera = np.eye(4)
    camera[:3, :3] = np.column_stack((right, np.cross(right, forward), -forward))
    camera[:3, 3] = eye
    return np.linalg.inv(body) @ camera


def _host_numpy(value):
    # ProxyArray.numpy() forwards to torch.Tensor.numpy(); on CUDA that
    # refuses instead of copying. Its explicit Warp accessor performs the
    # required host transfer and avoids the deprecated forwarding bridge.
    warp_array = getattr(value, "warp", None)
    if warp_array is not None:
        return np.asarray(warp_array.numpy())
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    numpy = getattr(value, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else value)


def _native_body_poses(robot):
    # Read the exact pinned PhysX view, not a timestamp-cached body_pose_w that
    # a sensor reset may have fetched before the joint forward pass.
    view = getattr(robot.data, "_root_view", None)
    if view is None:
        raise RuntimeError("native_camera_body_transform_view_missing")
    return _host_numpy(view.get_link_transforms())


def _camera_render_identity(camera):
    data = getattr(camera, '_render_data', None)
    spec = getattr(data, 'spec', None)
    rendered = [str(path) for path in getattr(spec, 'camera_prim_paths', ())]
    delegated = [str(path) for path in camera._view.prim_paths]
    products = [str(path) for path in getattr(data, 'render_product_paths', ())]
    if (not delegated or len(set(delegated)) != len(delegated) or rendered != delegated
        or not products or any(not path.startswith('/') for path in products)):
        raise RuntimeError('native_camera_render_product_view_binding_mismatch')
    return {'camera_prim_paths': rendered, 'render_product_paths': products,
        'source': 'native_renderer_spec_and_scene_view_identity'}


class NativeAttachedCameraView:
    """Keep one fixed camera mount on the live PhysX body across every frame."""
    def __init__(self, delegate, robot, body_index, attachment, device, forward, render):
        self._delegate = delegate
        self._robot = robot
        self._body_index = body_index
        self._attachment = np.asarray(attachment)
        self._device = device
        self._forward = forward
        self._render = render

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def world_matrices(self):
        self._forward()
        rows = _native_body_poses(self._robot)
        return np.asarray([pose_matrix(row[self._body_index]) @ self._attachment for row in rows])

    def get_world_poses(self, indices=None):
        import warp as wp
        from pxr import Gf, UsdGeom
        scene_indices = indices
        matrices = self.world_matrices()
        if indices is not None and not isinstance(indices, slice):
            indices = np.asarray(_host_numpy(indices), dtype=int)
        matrices = matrices if indices is None else matrices[indices]
        positions = np.ascontiguousarray(matrices[:, :3, 3], dtype=np.float32)
        quaternions = np.ascontiguousarray(Rotation.from_matrix(matrices[:, :3, :3]).as_quat(), dtype=np.float32)
        prim_indices = np.arange(len(self._delegate.prims))
        if indices is not None:
            prim_indices = prim_indices[indices]
        # The renderer can read the USD camera hierarchy independently of the
        # Fabric pose buffer. Bind both representations to the measured world
        # pose. Resetting this camera's transform stack prevents a stale parent
        # pose from being applied a second time; the rigid mount still follows
        # PhysX through world_matrices() on every observation.
        selected_prims = [self._delegate.prims[int(i)] for i in prim_indices]
        for prim, position, quaternion in zip(selected_prims, positions, quaternions, strict=True):
            UsdGeom.Xformable(prim).SetResetXformStack(True)
            position_set = prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*map(float, position)))
            orientation_set = prim.GetAttribute('xformOp:orient').Set(
                Gf.Quatd(float(quaternion[3]), Gf.Vec3d(*map(float, quaternion[:3]))))
            if not position_set or not orientation_set:
                raise RuntimeError('native_camera_usd_world_pose_write_failed')
        # IsaacRtxRenderer.update_camera is a no-op: changing CameraData only
        # changes metadata. Write the actual Fabric scene camera, then advance
        # the render generation so another camera's same-step pump cannot make
        # this camera consume an image rendered before the write.
        self._delegate.set_world_poses(
            wp.from_numpy(positions, device=self._device),
            wp.from_numpy(quaternions, device=self._device), scene_indices,
        )
        self._render()
        usd_cache = UsdGeom.XformCache()
        usd_matrices = np.asarray([
            np.asarray(usd_cache.GetLocalToWorldTransform(prim), dtype=float).T
            for prim in selected_prims
        ])
        if not np.allclose(usd_matrices, matrices, atol=1e-5, rtol=0):
            raise RuntimeError('native_camera_usd_world_pose_readback_mismatch')
        # Report the scene view readback, rather than our requested pose.
        observed = self._delegate.get_world_poses(scene_indices)
        actual_positions = _host_numpy(observed[0])
        actual_quaternions = _host_numpy(observed[1])
        if (not np.allclose(actual_positions, positions, atol=1e-4, rtol=0)
            or not np.allclose(np.abs(np.sum(actual_quaternions * quaternions, axis=-1)), 1., atol=1e-5)):
            raise RuntimeError('native_camera_aim_scene_pose_write_readback_mismatch')
        return observed


def install_direct_wrist_camera_aim(*, env: Any, camera_name: str, target) -> dict:
    native = env.unwrapped
    # Arena constructs the robot before reset events apply its configured
    # joints. Freeze the mount only after the full native reset has finished.
    # This is setup; the episode still performs its normal, identical reset.
    native.reset()
    native.sim.forward()
    robot = native.scene['robot']
    camera = native.scene[camera_name]
    render_identity = _camera_render_identity(camera)
    parent = camera.cfg.prim_path.rsplit('/', 2)[-2]
    names = list(robot.data.body_names)
    if names.count(parent) != 1:
        raise RuntimeError('native_camera_aim_parent_body_missing:' + parent)
    poses = _native_body_poses(robot)
    if len(poses) != 1:
        raise RuntimeError('native_camera_aim_requires_one_resolved_environment')
    body_index = names.index(parent)
    attachment = target_facing_attachment(poses[0, body_index], camera.cfg.offset.pos, target)
    if getattr(camera._view, '_use_fabric', False) is not True:
        raise RuntimeError('native_camera_aim_fabric_scene_writer_required')
    # Keep authored USD mount metadata consistent with the actual scene write.
    # The Fabric world pose is refreshed from PhysX for every observation.
    from pxr import Gf
    local_quat = Rotation.from_matrix(attachment[:3, :3]).as_quat()
    for prim in camera._view.prims:
        if not prim.GetAttribute('xformOp:orient').Set(
            Gf.Quatd(float(local_quat[3]), Gf.Vec3d(*local_quat[:3]))):
            raise RuntimeError('native_camera_aim_usd_mount_write_failed')
    camera._view = NativeAttachedCameraView(camera._view, robot, body_index, attachment,
        camera.device, native.sim.forward, native.sim.render)
    camera.cfg.offset.rot = tuple(Rotation.from_matrix(attachment[:3, :3]).as_quat())
    camera.cfg.offset.convention = "opengl"
    camera.cfg.update_latest_camera_pose = True
    return {'schema_version': 'native_task_direct_camera_aim.v1',
        'source': 'operator_requested_initial_target_facing_rigid_mount',
        'body_name': parent, 'target_position_world_m': list(target),
        'body_from_camera_opengl': attachment.tolist(),
        'pose_source': 'native_physx_get_link_transforms',
        'aim_setup_phase': 'after_native_environment_reset',
        'initial_native_body_pose_xyzw': poses[0, body_index].tolist(),
        'render_pose_writer': 'native_body_to_usd_and_fabric_world_pose',
        'render_generation_advanced_after_pose_write': True,
        'camera_pose_readback_source': 'usd_world_transform_and_fabric_frame_view',
        'camera_prim_paths': list(camera._view.prim_paths),
        'render_product_binding': render_identity,
        'intrinsics_preserved': True, 'official_mount_orientation_preserved': False,
        'tracks_object_after_reset': False, 'attachment_fixed_during_episode': True}


def install_native_wrist_camera_attachment(*, env: Any, camera_name: str) -> dict:
    """Synchronize the authored rigid mount without changing its calibration."""
    native = env.unwrapped
    robot = native.scene['robot']
    camera = native.scene[camera_name]
    render_identity = _camera_render_identity(camera)
    parent = camera.cfg.prim_path.rsplit('/', 2)[-2]
    names = list(robot.data.body_names)
    if names.count(parent) != 1:
        raise RuntimeError('native_camera_attachment_parent_body_missing:' + parent)
    if getattr(camera._view, '_use_fabric', False) is not True:
        raise RuntimeError('native_camera_attachment_fabric_scene_writer_required')
    # The camera constructor has already converted the configured convention
    # to the authored OpenGL USD local pose. Read that pose; never derive an
    # attachment from a potentially stale world-pose getter.
    translations, orientations = camera._view.get_local_poses()
    positions, quaternions = _host_numpy(translations), _host_numpy(orientations)
    if (positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0
        or quaternions.shape != (len(positions), 4)
        or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
        or not np.allclose(np.linalg.norm(quaternions, axis=-1), 1., atol=1e-5, rtol=0)
        or not np.allclose(positions, positions[:1], atol=1e-6, rtol=0)
        or not np.allclose(quaternions, quaternions[:1], atol=1e-6, rtol=0)):
        raise RuntimeError('native_camera_attachment_local_pose_invalid')
    attachment = pose_matrix([*positions[0], *quaternions[0]])
    camera._view = NativeAttachedCameraView(camera._view, robot, names.index(parent),
        attachment, camera.device, native.sim.forward, native.sim.render)
    camera.cfg.update_latest_camera_pose = True
    return {'schema_version': 'native_task_camera_attachment.v1',
        'source': 'authored_camera_local_pose_on_native_physx_body',
        'body_name': parent, 'body_from_camera_opengl': attachment.tolist(),
        'pose_source': 'native_physx_get_link_transforms',
        'render_pose_writer': 'native_body_to_usd_and_fabric_world_pose',
        'render_generation_advanced_after_pose_write': True,
        'camera_pose_readback_source': 'usd_world_transform_and_fabric_frame_view',
        'camera_prim_paths': list(camera._view.prim_paths),
        'render_product_binding': render_identity,
        'intrinsics_preserved': True, 'configured_mount_orientation_preserved': True,
        'tracks_object_after_reset': False, 'attachment_fixed_during_episode': True}
