"""Fail-closed DROID action conditioning for the Ctrl-World comparator.

Ctrl-World consumes Cartesian end-effector pose rows shaped ``[T, 7]``.  The
frozen smoke-test policies emit absolute Franka joint positions.  For that
case, deterministic forward kinematics is the least ambiguous conversion: it
does not pretend that absolute positions are the joint velocities expected by
Ctrl-World's separately trained learned adapter.
"""

from __future__ import annotations

import math
import importlib.util
import sys
import types
from collections.abc import Callable, Mapping, Sequence
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
CTRL_WORLD_RELEASED_ACTION_ADAPTER_SOURCE_SHA256 = (
    "44e5c91d8339512c1a121b72159e77a4aa469dbb134d3f0b373a7447af8a88e2"
)
CTRL_WORLD_RELEASED_FK_SOURCE_SHA256 = (
    "bd9af90afdf379b95c2dfc7c3a5f8f6b8c6f1edc92ef8b8b7b59d08868ecfae3"
)
CTRL_WORLD_RELEASED_ACTION_ROWS = 15
CTRL_WORLD_FUTURE_FRAME_INDICES = (0, 2, 4, 6, 8)
CTRL_WORLD_HISTORY_ROWS = 6
RELIABILITY_ROT6D_IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
CTRL_WORLD_EXECUTED_PREFIX_ROWS = 8
CTRL_WORLD_CAUSAL_SHUFFLE_SEED = 20260730
CTRL_WORLD_CAUSAL_SHUFFLE_ORDER = (3, 5, 6, 0, 1, 4, 7, 2)
CTRL_WORLD_CAUSAL_CONDITIONS = (
    "own_action",
    "no_motion",
    "shuffled",
    "reversed",
    "shifted",
    "policy_swapped",
)


def _native_velocity_action(value: Any, *, reason: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64)
    if action.ndim != 2 or action.shape[0] not in {10, 15} or action.shape[1] != 8:
        raise ValueError(reason)
    if not np.isfinite(action).all():
        raise ValueError(f"{reason}_nonfinite")
    return action.copy()


def _frozen_shuffle_order(seed: int) -> list[int]:
    if seed != CTRL_WORLD_CAUSAL_SHUFFLE_SEED or isinstance(seed, bool):
        raise ValueError("ctrl_world_causal_shuffle_seed_invalid")
    return list(CTRL_WORLD_CAUSAL_SHUFFLE_ORDER)


def build_ctrl_world_current_reference_action_controls(
    *,
    own_policy_action: Any,
    own_source_trace_id: str,
    policy_swapped_action: Any,
    policy_swapped_source_trace_id: str,
    current_gripper_hold: float,
    shuffle_seed: int,
    temporal_shift_steps: int = 1,
) -> dict[str, Any]:
    """Build the frozen six-condition matrix in the released native action space."""

    own_trace = str(own_source_trace_id).strip()
    swapped_trace = str(policy_swapped_source_trace_id).strip()
    if not own_trace or not swapped_trace:
        raise ValueError("ctrl_world_causal_real_trace_identity_missing")
    if own_trace == swapped_trace:
        raise ValueError("ctrl_world_causal_policy_swap_trace_not_distinct")
    own = _native_velocity_action(own_policy_action, reason="ctrl_world_causal_own_action_invalid")
    swapped = _native_velocity_action(
        policy_swapped_action, reason="ctrl_world_causal_policy_swapped_action_invalid"
    )
    hold = float(current_gripper_hold)
    if not math.isfinite(hold) or not 0.0 <= hold <= 1.0:
        raise ValueError("ctrl_world_causal_gripper_hold_invalid")
    if (
        isinstance(temporal_shift_steps, bool)
        or not isinstance(temporal_shift_steps, int)
        or not 0 < temporal_shift_steps < CTRL_WORLD_EXECUTED_PREFIX_ROWS
    ):
        raise ValueError("ctrl_world_causal_temporal_shift_invalid")

    order = _frozen_shuffle_order(shuffle_seed)
    no_motion = np.zeros_like(own)
    no_motion[:, 7] = hold
    shuffled = own.copy()
    shuffled[:CTRL_WORLD_EXECUTED_PREFIX_ROWS] = own[np.asarray(order)]
    reversed_action = own.copy()
    reversed_action[:CTRL_WORLD_EXECUTED_PREFIX_ROWS] = own[:CTRL_WORLD_EXECUTED_PREFIX_ROWS][::-1]
    shifted = own.copy()
    shifted[:CTRL_WORLD_EXECUTED_PREFIX_ROWS] = np.roll(
        own[:CTRL_WORLD_EXECUTED_PREFIX_ROWS],
        -temporal_shift_steps,
        axis=0,
    )
    controls = {
        "own_action": own,
        "no_motion": no_motion,
        "shuffled": shuffled,
        "reversed": reversed_action,
        "shifted": shifted,
        "policy_swapped": swapped,
    }
    prefix_hashes = {
        condition: canonical_sha256(action[:CTRL_WORLD_EXECUTED_PREFIX_ROWS].tolist())
        for condition, action in controls.items()
    }
    if len(set(prefix_hashes.values())) != len(prefix_hashes):
        collisions = sorted(
            condition
            for condition, digest in prefix_hashes.items()
            if list(prefix_hashes.values()).count(digest) > 1
        )
        raise ValueError(
            "ctrl_world_causal_executed_prefixes_not_pairwise_distinct:" + ",".join(collisions)
        )
    complete_hashes = {
        condition: canonical_sha256(action.tolist()) for condition, action in controls.items()
    }
    result: dict[str, Any] = {
        "schema_version": "ctrl_world_current_reference_action_controls.v1",
        "conditions": controls,
        "condition_order": list(CTRL_WORLD_CAUSAL_CONDITIONS),
        "complete_native_action_sha256_by_condition": complete_hashes,
        "executed_prefix_sha256_by_condition": prefix_hashes,
        "native_action_shape_by_condition": {
            condition: list(action.shape) for condition, action in controls.items()
        },
        "executed_prefix_rows": CTRL_WORLD_EXECUTED_PREFIX_ROWS,
        "own_source_trace_id": own_trace,
        "policy_swapped_source_trace_id": swapped_trace,
        "policy_swap_is_distinct_real_trace": True,
        "synthetic_policy_swap_forbidden": True,
        "no_motion_joint_velocity_zero": True,
        "no_motion_gripper_hold": hold,
        "shuffle_seed": shuffle_seed,
        "shuffle_order_first_eight": order,
        "temporal_shift_steps_first_eight": temporal_shift_steps,
        "tail_rows_preserved_for_own_derived_controls": True,
        "physical_outcome_accessed": False,
    }
    identity_material = {
        **result,
        "conditions": {condition: action.tolist() for condition, action in controls.items()},
    }
    result["controls_sha256"] = canonical_sha256(identity_material)
    return result


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


def _pose_from_fk(value: Any) -> np.ndarray:
    """Normalize a registered Franka FK result to XYZ plus intrinsic XYZ Euler."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape == (7,):
        if not np.isfinite(result).all():
            raise ValueError("ctrl_world_fk_pose_nonfinite")
        return result
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("ctrl_world_fk_result_must_be_pose7_or_transform4x4")
    return np.concatenate((result[:3, 3], _matrix_to_xyz_euler(result[:3, :3])))


def cartesian_pose_rows_to_reliability_actions_10d(
    pose_rows: Sequence[Sequence[float]],
) -> np.ndarray:
    """Convert absolute Ctrl-World poses to incremental reliability actions.

    The WAM continues to receive its released seven-dimensional Cartesian
    contract.  This representation is a deterministic measurement adapter for
    Blueprint's existing translation/rotation-6D/gripper reliability gate; it
    is never used as WAM conditioning.  Translation and rotation are expressed
    relative to the preceding generated-frame pose so an unchanged absolute
    pose is a valid no-motion command rather than a false active command.  The
    first row is the zero/identity transition anchored at the current pose.
    """

    poses = np.asarray(pose_rows, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError("ctrl_world_pose_rows_must_be_finite_nx7")
    actions = np.zeros((poses.shape[0], 10), dtype=np.float64)
    actions[:, 9] = poses[:, 6]

    rotations: list[np.ndarray] = []
    for rx, ry, rz in poses[:, 3:6]:
        cx, sx = math.cos(float(rx)), math.sin(float(rx))
        cy, sy = math.cos(float(ry)), math.sin(float(ry))
        cz, sz = math.cos(float(rz)), math.sin(float(rz))
        rotations.append(
            np.asarray(
                [
                    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                    [-sy, cy * sx, cy * cx],
                ],
                dtype=np.float64,
            )
        )
    actions[0, 3:9] = RELIABILITY_ROT6D_IDENTITY
    for index in range(1, poses.shape[0]):
        actions[index, :3] = poses[index, :3] - poses[index - 1, :3]
        relative_rotation = rotations[index - 1].T @ rotations[index]
        actions[index, 3:6] = relative_rotation[0]
        actions[index, 6:9] = relative_rotation[1]
    return actions


@dataclass(frozen=True)
class CtrlWorldReleasedJointVelocityAdapter:
    """Execute Ctrl-World's released DROID joint-velocity state adapter contract.

    ``dynamics_adapter`` is the exact released ``Dynamics`` checkpoint-bound
    inference call and ``forward_kinematics`` is the registered Franka FK used
    by the public Ctrl-World code.  Their concrete runtimes stay injectable so
    the contract can be tested hermetically and run under a pinned GPU image.
    Absolute joint-position policies are intentionally rejected.
    """

    dynamics_adapter: Callable[[np.ndarray, np.ndarray], np.ndarray]
    forward_kinematics: Callable[[np.ndarray], Any]
    gripper_max: float
    adapter_id: str = "ctrl_world_released_joint_velocity_to_cartesian_v1"

    def adapt(
        self,
        *,
        policy_action: Sequence[Sequence[float]],
        current_joint_position: Sequence[float],
        current_gripper_position: Sequence[float],
        history_cartesian_pose_7d: Sequence[Sequence[float]],
        source_action_space: str,
    ) -> dict[str, Any]:
        if source_action_space != "joint_velocity_plus_gripper_position":
            raise ValueError("ctrl_world_released_adapter_requires_joint_velocity_policy")
        action = np.asarray(policy_action, dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 8 or action.shape[0] not in {10, 15}:
            raise ValueError("ctrl_world_joint_velocity_action_must_be_10x8_or_15x8")
        if not np.isfinite(action).all():
            raise ValueError("ctrl_world_joint_velocity_action_nonfinite")
        current_joint = np.asarray(current_joint_position, dtype=np.float64)
        current_gripper = np.asarray(current_gripper_position, dtype=np.float64)
        history = np.asarray(history_cartesian_pose_7d, dtype=np.float64)
        if current_joint.shape != (7,) or not np.isfinite(current_joint).all():
            raise ValueError("ctrl_world_current_joint_position_invalid")
        if current_gripper.shape != (1,) or not np.isfinite(current_gripper).all():
            raise ValueError("ctrl_world_current_gripper_position_invalid")
        if history.shape != (CTRL_WORLD_HISTORY_ROWS, 7) or not np.isfinite(history).all():
            raise ValueError("ctrl_world_history_cartesian_pose_must_be_finite_6x7")
        if not math.isfinite(self.gripper_max) or not 0.0 < self.gripper_max <= 1.0:
            raise ValueError("ctrl_world_gripper_max_invalid")

        if action.shape[0] == 10:
            row_indices = np.asarray([*range(10), 9, 9, 9, 9, 9], dtype=int)
            released_action = action[row_indices]
            padding_rule = "repeat_final_row_to_15"
        else:
            released_action = action.copy()
            padding_rule = "none"
        joint_velocity = released_action[:, :7]
        gripper = np.clip(released_action[:, 7:8], 0.0, self.gripper_max)
        future_joint = np.asarray(
            self.dynamics_adapter(current_joint.copy(), joint_velocity.copy()),
            dtype=np.float64,
        )
        if (
            future_joint.shape != (CTRL_WORLD_RELEASED_ACTION_ROWS, 7)
            or not np.isfinite(future_joint).all()
        ):
            raise ValueError("ctrl_world_dynamics_adapter_output_invalid")

        joint_series = np.concatenate((current_joint[None, :], future_joint), axis=0)[
            :CTRL_WORLD_RELEASED_ACTION_ROWS
        ]
        gripper_series = np.concatenate((current_gripper[None, :], gripper), axis=0)[
            :CTRL_WORLD_RELEASED_ACTION_ROWS
        ]
        pose_series = np.asarray(
            [
                np.concatenate(
                    (
                        _pose_from_fk(self.forward_kinematics(joint_row)),
                        gripper_series[index],
                    )
                )
                for index, joint_row in enumerate(joint_series)
            ],
            dtype=np.float64,
        )
        future_pose = pose_series[np.asarray(CTRL_WORLD_FUTURE_FRAME_INDICES)]
        conditioning = np.concatenate((history, future_pose), axis=0)
        reliability = cartesian_pose_rows_to_reliability_actions_10d(future_pose)
        next_index = CTRL_WORLD_FUTURE_FRAME_INDICES[-1]
        result: dict[str, Any] = {
            "schema_version": "ctrl_world_released_joint_velocity_conditioning.v1",
            "adapter_id": self.adapter_id,
            "source_action_space": source_action_space,
            "target_action_space": "ctrl_world_cartesian_xyz_euler_xyz_plus_gripper",
            "native_policy_action": action,
            "native_policy_action_shape": list(action.shape),
            "released_action_rows": CTRL_WORLD_RELEASED_ACTION_ROWS,
            "ten_row_padding_rule": padding_rule,
            "action_conditioning_7d": conditioning,
            "action_conditioning_shape": list(conditioning.shape),
            "future_cartesian_pose_7d": future_pose,
            "reliability_actions_10d": reliability,
            "next_joint_position": joint_series[next_index],
            "next_gripper_position": gripper_series[next_index],
            "next_cartesian_pose_7d": pose_series[next_index],
            "future_frame_indices": list(CTRL_WORLD_FUTURE_FRAME_INDICES),
            "official_ctrl_world_learned_action_adapter_used": True,
            "official_action_adapter_checkpoint_sha256": (OFFICIAL_LOCAL_ACTION_ADAPTER_SHA256),
            "official_action_adapter_source_sha256": (
                CTRL_WORLD_RELEASED_ACTION_ADAPTER_SOURCE_SHA256
            ),
            "physical_future_observation_used": False,
            "task_outcome_accessed": False,
            "claim_boundary": (
                "released input-format and commanded-state adaptation only; not "
                "Ctrl-World causal qualification, policy rank fidelity, or physical success"
            ),
        }
        identity_material = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        result["conditioning_sha256"] = canonical_sha256(identity_material)
        return result


@dataclass(frozen=True)
class LoadedCtrlWorldReleasedActionRuntime:
    """Exact released Dynamics/FK binding plus immutable asset evidence."""

    adapter: CtrlWorldReleasedJointVelocityAdapter
    evidence: Mapping[str, Any]


def _load_released_dynamics(
    source_path: Path, checkpoint_path: Path, device: str
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - pinned provider runtime owns torch
        raise RuntimeError("ctrl_world_released_dynamics_torch_missing") from exc
    module_name = "blueprint_frozen_ctrl_world_released_dynamics"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ctrl_world_released_dynamics_import_spec_invalid")
    module = importlib.util.module_from_spec(spec)
    previous_decord = sys.modules.get("decord")
    training_only_stubs: list[str] = []
    if importlib.util.find_spec("decord") is None:
        decord_stub = types.ModuleType("decord")

        def _training_only_dependency_used(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("ctrl_world_training_only_decord_path_forbidden")

        decord_stub.VideoReader = _training_only_dependency_used  # type: ignore[attr-defined]
        decord_stub.cpu = _training_only_dependency_used  # type: ignore[attr-defined]
        sys.modules["decord"] = decord_stub
        training_only_stubs.append("decord")
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if previous_decord is None:
            sys.modules.pop("decord", None)
        else:
            sys.modules["decord"] = previous_decord
    model = module.Dynamics(action_dim=7, action_num=15, hidden_size=512).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    def execute(current_joint: np.ndarray, joint_velocity: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            result = model(
                current_joint[None, :],
                joint_velocity,
                None,
                training=False,
            )
        return np.asarray(result, dtype=np.float64)

    execute.blueprint_training_only_import_stubs = tuple(training_only_stubs)  # type: ignore[attr-defined]
    return execute


def _load_released_forward_kinematics(source_path: Path) -> Callable[[np.ndarray], Any]:
    module_name = "blueprint_frozen_ctrl_world_released_fk"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ctrl_world_released_fk_import_spec_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.get_fk_solution


def load_ctrl_world_released_joint_velocity_adapter(
    *,
    ctrl_world_source_dir: str | Path,
    gripper_max: float,
    device: str = "cpu",
    dynamics_loader: Callable[
        [Path, Path, str], Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = _load_released_dynamics,
    forward_kinematics_loader: Callable[[Path], Callable[[np.ndarray], Any]] = (
        _load_released_forward_kinematics
    ),
) -> LoadedCtrlWorldReleasedActionRuntime:
    """Load only exact public Ctrl-World adapter bytes, failing closed on drift."""

    root = Path(ctrl_world_source_dir).expanduser().resolve()
    source = root / "models/action_adapter/train2.py"
    checkpoint = root / "models/action_adapter/model2_15_9.pth"
    fk_source = root / "models/utils.py"
    assets = (
        (source, CTRL_WORLD_RELEASED_ACTION_ADAPTER_SOURCE_SHA256, "source"),
        (checkpoint, OFFICIAL_LOCAL_ACTION_ADAPTER_SHA256, "checkpoint"),
        (fk_source, CTRL_WORLD_RELEASED_FK_SOURCE_SHA256, "fk_source"),
    )
    observed: dict[str, Any] = {}
    for path, expected_digest, name in assets:
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"ctrl_world_released_action_{name}_invalid")
        digest = file_sha256(path)
        if digest != expected_digest:
            raise ValueError(f"ctrl_world_released_action_{name}_hash_mismatch")
        observed[name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    dynamics = dynamics_loader(source, checkpoint, device)
    forward_kinematics = forward_kinematics_loader(fk_source)
    adapter = CtrlWorldReleasedJointVelocityAdapter(
        dynamics_adapter=dynamics,
        forward_kinematics=forward_kinematics,
        gripper_max=gripper_max,
    )
    evidence: dict[str, Any] = {
        "schema_version": "ctrl_world_released_action_runtime.v1",
        "status": "loaded",
        "device": device,
        "assets": observed,
        "native_action_space": "joint_velocity_plus_gripper_position",
        "target_action_space": "ctrl_world_cartesian_xyz_euler_xyz_plus_gripper",
        "official_released_dynamics_and_fk_loaded": True,
        "training_only_import_stubs": list(
            getattr(dynamics, "blueprint_training_only_import_stubs", ())
        ),
        "training_only_stub_paths_forbidden_at_runtime": True,
        "absolute_joint_position_conversion_supported": False,
    }
    evidence["runtime_sha256"] = canonical_sha256(evidence)
    return LoadedCtrlWorldReleasedActionRuntime(adapter=adapter, evidence=evidence)


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
            mapped = droid_joint_position_action_to_mujoco_targets(
                row, joint_limits=limits.tolist()
            )
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
    "CTRL_WORLD_CAUSAL_CONDITIONS",
    "CTRL_WORLD_CAUSAL_SHUFFLE_ORDER",
    "CTRL_WORLD_CAUSAL_SHUFFLE_SEED",
    "CTRL_WORLD_EXECUTED_PREFIX_ROWS",
    "CTRL_WORLD_FUTURE_FRAME_INDICES",
    "CTRL_WORLD_HISTORY_ROWS",
    "CTRL_WORLD_RELEASED_ACTION_ADAPTER_SOURCE_SHA256",
    "CTRL_WORLD_RELEASED_FK_SOURCE_SHA256",
    "CTRL_WORLD_RELEASED_ACTION_ROWS",
    "CtrlWorldReleasedJointVelocityAdapter",
    "FrankaCtrlWorldJointPositionAdapter",
    "LoadedCtrlWorldReleasedActionRuntime",
    "OFFICIAL_LOCAL_ACTION_ADAPTER_SHA256",
    "SCHEMA_VERSION",
    "cartesian_pose_rows_to_reliability_actions_10d",
    "build_ctrl_world_current_reference_action_controls",
    "load_ctrl_world_released_joint_velocity_adapter",
    "validate_ctrl_world_runtime_assets",
]
