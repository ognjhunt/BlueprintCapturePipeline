"""Fail-closed DROID action conditioning for the Ctrl-World comparator.

Ctrl-World consumes Cartesian end-effector pose rows shaped ``[T, 7]``.  The
frozen smoke-test policies emit absolute Franka joint positions.  For that
case, deterministic forward kinematics is the least ambiguous conversion: it
does not pretend that absolute positions are the joint velocities expected by
Ctrl-World's separately trained learned adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .droid_policy_bridge import droid_joint_position_action_to_mujoco_targets
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "ctrl_world_droid_action_conditioning.v1"
OFFICIAL_LOCAL_ACTION_ADAPTER_SHA256 = (
    "b1a232a9c0539127ca23e202fd4fbc5c4756d385c890dd4af792ade51dc72f77"
)


def validate_ctrl_world_runtime_assets(
    *,
    world_model_checkpoint: str | Path,
    expected_world_model_sha256: str,
    action_adapter_checkpoint: str | Path | None = None,
    expected_action_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind runtime admission to exact files; never treat the small adapter as the WAM."""

    blockers: list[str] = []
    world = Path(world_model_checkpoint).expanduser().resolve()
    if not world.is_file() or world.is_symlink() or world.stat().st_size <= 0:
        blockers.append("ctrl_world_model_checkpoint_missing")
        world_digest = None
    else:
        world_digest = file_sha256(world)
        if world_digest != str(expected_world_model_sha256):
            blockers.append("ctrl_world_model_checkpoint_sha256_mismatch")
    adapter_digest = None
    adapter_path = None
    if action_adapter_checkpoint is not None:
        adapter_path = Path(action_adapter_checkpoint).expanduser().resolve()
        if not adapter_path.is_file() or adapter_path.is_symlink():
            blockers.append("ctrl_world_action_adapter_checkpoint_missing")
        else:
            adapter_digest = file_sha256(adapter_path)
            if not expected_action_adapter_sha256:
                blockers.append("ctrl_world_action_adapter_expected_sha256_missing")
            elif adapter_digest != str(expected_action_adapter_sha256):
                blockers.append("ctrl_world_action_adapter_checkpoint_sha256_mismatch")
    result: dict[str, Any] = {
        "schema_version": "ctrl_world_runtime_asset_admission.v1",
        "status": "passed" if not blockers else "blocked",
        "world_model_checkpoint": str(world),
        "world_model_sha256": world_digest,
        "action_adapter_checkpoint": str(adapter_path) if adapter_path else None,
        "action_adapter_sha256": adapter_digest,
        "world_model_and_action_adapter_are_distinct_assets": True,
        "blockers": blockers,
    }
    result["admission_sha256"] = canonical_sha256(result)
    return result


def _matrix_to_xyz_euler(rotation: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ Euler angles matching SciPy's ``as_euler('xyz')``."""

    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    singular = sy < 1e-8
    if not singular:
        x = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y = math.atan2(-float(rotation[2, 0]), sy)
        z = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        x = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        y = math.atan2(-float(rotation[2, 0]), sy)
        z = 0.0
    return np.asarray([x, y, z], dtype=np.float64)


@dataclass(frozen=True)
class FrankaCtrlWorldJointPositionAdapter:
    """Convert frozen absolute joint-position chunks with exact MuJoCo FK."""

    runtime: Mapping[str, Any]
    adapter_id: str = "blueprint_franka_joint_position_fk_to_ctrl_world_pose_v1"

    def adapt(
        self,
        *,
        policy_action: Sequence[Sequence[float]],
        history_cartesian_pose_7d: Sequence[Sequence[float]] = (),
    ) -> dict[str, Any]:
        action = np.asarray(policy_action, dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 8 or action.shape[0] not in {10, 15}:
            raise ValueError("ctrl_world_joint_position_action_must_be_10x8_or_15x8")
        if not np.isfinite(action).all():
            raise ValueError("ctrl_world_joint_position_action_nonfinite")
        history = np.asarray(history_cartesian_pose_7d, dtype=np.float64)
        if history.size == 0:
            history = np.empty((0, 7), dtype=np.float64)
        if history.ndim != 2 or history.shape[1] != 7 or not np.isfinite(history).all():
            raise ValueError("ctrl_world_history_cartesian_pose_must_be_finite_nx7")
        model = self.runtime["model"]
        mujoco = self.runtime["mujoco"]
        data = mujoco.MjData(model)
        hand_id = int(self.runtime["ids"]["hand"])
        limits = np.asarray(model.jnt_range[:7], dtype=np.float64)
        pose_rows: list[np.ndarray] = []
        clamped_rows = 0
        for row in action:
            mapped = droid_joint_position_action_to_mujoco_targets(row, joint_limits=limits)
            clamped_rows += int(mapped["joint_limit_clamped"])
            data.qpos[:7] = np.asarray(mapped["joint_position_target_rad"], dtype=np.float64)
            data.qpos[7:9] = float(mapped["gripper_position_target_m"])
            mujoco.mj_forward(model, data)
            position = np.asarray(data.xpos[hand_id], dtype=np.float64)
            rotation = np.asarray(data.xmat[hand_id], dtype=np.float64).reshape(3, 3)
            gripper = float(np.clip(row[7], 0.0, 1.0))
            pose_rows.append(np.concatenate((position, _matrix_to_xyz_euler(rotation), [gripper])))
        pose = np.asarray(pose_rows, dtype=np.float64)
        conditioning = np.vstack((history, pose))
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": self.adapter_id,
            "source_action_space": "absolute_franka_joint_position_plus_gripper",
            "target_action_space": "ctrl_world_cartesian_xyz_euler_xyz_plus_gripper",
            "policy_action_rows": int(action.shape[0]),
            "history_rows": int(history.shape[0]),
            "action_conditioning_7d": conditioning,
            "action_conditioning_shape": list(conditioning.shape),
            "joint_limit_clamped_row_count": clamped_rows,
            "conversion": "deterministic_pinned_franka_forward_kinematics",
            "official_ctrl_world_learned_action_adapter_used": False,
            "reason_official_adapter_not_used": (
                "official learned adapter consumes DROID joint velocity; frozen policies emit "
                "absolute joint position"
            ),
            "physical_future_observation_used": False,
            "task_outcome_accessed": False,
            "claim_boundary": "input-format adaptation only; not Ctrl-World validity or success",
        }
        identity_material = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        result["conditioning_sha256"] = canonical_sha256(identity_material)
        return result


__all__ = [
    "FrankaCtrlWorldJointPositionAdapter",
    "OFFICIAL_LOCAL_ACTION_ADAPTER_SHA256",
    "SCHEMA_VERSION",
    "validate_ctrl_world_runtime_assets",
]
