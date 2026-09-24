"""One G1 scene episode's native joint readback and controller-output seam.

The owner of this adapter is Blueprint's Arena environment. An upstream SONIC
provider must return named, verified joint targets; policy semantic vectors
cannot be passed here because they have neither this shape nor these names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from .native_g1_humanoidarena_interface import (
    CANONICAL_BODY_JOINT_NAMES_29,
    build_semantic_v3_state,
)


class NativeG1JointEpisodeEnvironment:
    def __init__(self, *, built: Any, to_tensor: Any, make_action_tensor: Any) -> None:
        plan = getattr(built, "plan", None)
        env = getattr(built, "env", None)
        if (
            not isinstance(plan, Mapping)
            or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
            or env is None
        ):
            raise ValueError("native_g1_episode_scene_binding_invalid")
        try:
            action_dim = int(env.unwrapped.action_manager.total_action_dim)
            robot = env.unwrapped.scene["robot"]
            names = [str(name) for name in robot.joint_names]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("native_g1_episode_articulation_unavailable") from exc
        if (
            action_dim != len(PROTOCOL_V4_FULL_JOINT_ORDER)
            or len(names) != len(set(names))
            or not set(PROTOCOL_V4_FULL_JOINT_ORDER).issubset(names)
        ):
            raise ValueError("native_g1_episode_joint_or_action_inventory_invalid")
        limits = plan["robot"].get("joint_position_limits_rad")
        if not isinstance(limits, Mapping) or set(limits) != set(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise ValueError("native_g1_episode_joint_limits_missing")
        self._limits = limits
        self._env = env
        self._robot = robot
        self._to_tensor = to_tensor
        self._make_action_tensor = make_action_tensor
        self._index = {name: index for index, name in enumerate(names)}
        self._initial_root_orientation_xyzw: list[float] | None = None
        self._step_index = 0

    def _row(self, value: Any) -> list[float]:
        try:
            row = [float(item) for item in self._to_tensor(value)[0]]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("native_g1_episode_state_readback_invalid") from exc
        if not all(math.isfinite(item) for item in row):
            raise ValueError("native_g1_episode_state_readback_invalid")
        return row

    def reset(self, *, seed: int) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("native_g1_episode_seed_invalid")
        self._env.reset(seed=seed)
        self._initial_root_orientation_xyzw = self._row(self._robot.data.root_quat_w)
        if len(self._initial_root_orientation_xyzw) != 4:
            raise ValueError("native_g1_episode_root_orientation_invalid")
        self._step_index = 0
        return self.read_state()

    def read_state(self) -> dict[str, Any]:
        positions = self._row(self._robot.data.joint_pos)
        velocities = self._row(self._robot.data.joint_vel)
        root_orientation = self._row(self._robot.data.root_quat_w)
        root_position = self._row(self._robot.data.root_pos_w)
        if (
            len(root_orientation) != 4
            or len(root_position) != 3
            or len(positions) != len(self._index)
            or len(velocities) != len(self._index)
        ):
            raise ValueError("native_g1_episode_state_readback_invalid")
        return {
            "step_index": self._step_index,
            "root_position_world_m": root_position,
            "root_orientation_xyzw": root_orientation,
            "joint_position_rad": {
                name: positions[index] for name, index in self._index.items()
            },
            "joint_velocity_rad_s": {
                name: velocities[index] for name, index in self._index.items()
            },
        }

    def read_semantic_v3_state(self) -> list[float]:
        if self._initial_root_orientation_xyzw is None:
            raise ValueError("native_g1_episode_reset_required")
        state = self.read_state()
        return build_semantic_v3_state(
            initial_root_orientation_xyzw=self._initial_root_orientation_xyzw,
            current_root_orientation_xyzw=state["root_orientation_xyzw"],
            body_joint_positions_rad={name: state["joint_position_rad"][name]
                                      for name in CANONICAL_BODY_JOINT_NAMES_29},
            body_joint_velocities_rad_s={name: state["joint_velocity_rad_s"][name]
                                        for name in CANONICAL_BODY_JOINT_NAMES_29},
        )

    def step_controller_targets(self, targets_by_name: Mapping[str, float]) -> dict[str, Any]:
        if self._initial_root_orientation_xyzw is None:
            raise ValueError("native_g1_episode_reset_required")
        if not isinstance(targets_by_name, Mapping) or set(targets_by_name) != set(PROTOCOL_V4_FULL_JOINT_ORDER):
            raise ValueError("native_g1_controller_joint_inventory_invalid")
        targets: list[float] = []
        for name in PROTOCOL_V4_FULL_JOINT_ORDER:
            try:
                value = float(targets_by_name[name])
                lower, upper = [float(item) for item in self._limits[name]]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("native_g1_controller_joint_target_invalid") from exc
            if not all(math.isfinite(item) for item in (value, lower, upper)) or not lower <= value <= upper:
                raise ValueError("native_g1_controller_joint_target_invalid")
            targets.append(value)
        action = self._make_action_tensor([targets], device=self._env.unwrapped.device)
        self._env.step(action)
        self._step_index += 1
        return self.read_state()
