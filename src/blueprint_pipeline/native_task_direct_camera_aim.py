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


def _native_body_poses(robot):
    # Read the exact pinned PhysX view, not a timestamp-cached body_pose_w that
    # a sensor reset may have fetched before the joint forward pass.
    view = getattr(robot.data, "_root_view", None)
    if view is None:
        raise RuntimeError("native_camera_body_transform_view_missing")
    value = view.get_link_transforms()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    tensor = getattr(value, "torch", value)
    return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)


class NativeAttachedCameraView:
    """Keep one fixed camera mount on the live PhysX body across every frame."""
    def __init__(self, delegate, robot, body_index, attachment, device, forward):
        self._delegate = delegate
        self._robot = robot
        self._body_index = body_index
        self._attachment = np.asarray(attachment)
        self._device = device
        self._forward = forward

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def world_matrices(self):
        self._forward()
        rows = _native_body_poses(self._robot)
        return np.asarray([pose_matrix(row[self._body_index]) @ self._attachment for row in rows])

    def get_world_poses(self, indices=None):
        import warp as wp
        from isaaclab.utils.warp import ProxyArray
        matrices = self.world_matrices()
        if indices is not None and not isinstance(indices, slice):
            indices = np.asarray(indices.numpy() if hasattr(indices, 'numpy') else indices, dtype=int)
        matrices = matrices if indices is None else matrices[indices]
        positions = np.ascontiguousarray(matrices[:, :3, 3], dtype=np.float32)
        quaternions = np.ascontiguousarray(Rotation.from_matrix(matrices[:, :3, :3]).as_quat(), dtype=np.float32)
        return (ProxyArray(wp.from_numpy(positions, device=self._device)),
                ProxyArray(wp.from_numpy(quaternions, device=self._device)))


def install_direct_wrist_camera_aim(*, env: Any, camera_name: str, target) -> dict:
    native = env.unwrapped
    native.sim.forward()
    robot = native.scene['robot']
    camera = native.scene[camera_name]
    parent = camera.cfg.prim_path.rsplit('/', 2)[-2]
    names = list(robot.data.body_names)
    if names.count(parent) != 1:
        raise RuntimeError('native_camera_aim_parent_body_missing:' + parent)
    poses = _native_body_poses(robot)
    if len(poses) != 1:
        raise RuntimeError('native_camera_aim_requires_one_resolved_environment')
    body_index = names.index(parent)
    attachment = target_facing_attachment(poses[0, body_index], camera.cfg.offset.pos, target)
    camera._view = NativeAttachedCameraView(camera._view, robot, body_index, attachment, camera.device, native.sim.forward)
    camera.cfg.offset.rot = tuple(Rotation.from_matrix(attachment[:3, :3]).as_quat())
    camera.cfg.offset.convention = "opengl"
    camera.cfg.update_latest_camera_pose = True
    return {'schema_version': 'native_task_direct_camera_aim.v1',
        'source': 'operator_requested_initial_target_facing_rigid_mount',
        'body_name': parent, 'target_position_world_m': list(target),
        'body_from_camera_opengl': attachment.tolist(),
        'pose_source': 'native_physx_get_link_transforms',
        'intrinsics_preserved': True, 'official_mount_orientation_preserved': False,
        'tracks_object_after_reset': False, 'attachment_fixed_during_episode': True}
