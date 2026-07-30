"""DROID transition adapter for the Blueprint Ctrl-World current-reference arm."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .ctrl_world_droid_action_adapter import CtrlWorldReleasedJointVelocityAdapter
from .ctrl_world_openpi_preprocessing import (
    preprocess_ctrl_world_frame_for_openpi_policy,
)
from .droid_oscar_closed_loop_adapter import WAM_SOURCE_VIEW_PATHS
from .droid_policy_bridge import (
    DROID_EXTERIOR_VIEW_1,
    DROID_EXTERIOR_VIEW_2,
    DROID_WRIST_VIEW,
    validate_droid_action_chunk,
    validate_droid_observation,
)
from .policy_ranking_thesis import file_sha256


CTRL_WORLD_RELEASED_VIEW_ORDER = (
    DROID_EXTERIOR_VIEW_2,
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
CTRL_WORLD_SELECTED_HISTORY_INDICES = (0, 0, -12, -9, -6, -3)
CTRL_WORLD_INITIAL_HISTORY_LENGTH = 24
CTRL_WORLD_STATE_HISTORY = "blueprint/ctrl_world_cartesian_state_history"
CTRL_WORLD_VIEW_HISTORY_PATHS = "blueprint/ctrl_world_view_history_paths"
CTRL_WORLD_CURRENT_REFERENCE_POLICIES = frozenset(
    {"pi0_droid", "pi0_fast_droid", "pi05_droid"}
)


def _safe_file(value: Any, *, reason: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(reason)
    return path


@dataclass(frozen=True)
class DroidCtrlWorldCurrentReferenceTransitionAdapter:
    """Bind released OpenPI DROID policies to Ctrl-World without contract invention."""

    action_adapter: CtrlWorldReleasedJointVelocityAdapter
    openpi_config_name: str
    action_chunk_rows: int
    policy_image_preprocessor: Callable[[str | Path], np.ndarray] = (
        preprocess_ctrl_world_frame_for_openpi_policy
    )
    adapter_id: str = "droid_ctrl_world_current_reference_joint_velocity_v1"

    def __post_init__(self) -> None:
        if self.openpi_config_name not in CTRL_WORLD_CURRENT_REFERENCE_POLICIES:
            raise ValueError("ctrl_world_current_reference_policy_config_unsupported")
        expected_rows = 15 if self.openpi_config_name == "pi05_droid" else 10
        if self.action_chunk_rows != expected_rows:
            raise ValueError("ctrl_world_openpi_config_action_rows_mismatch")

    def _initial_view_histories(
        self, observation: Mapping[str, Any], output_dir: Path
    ) -> dict[str, list[str]]:
        history_value = observation.get(CTRL_WORLD_VIEW_HISTORY_PATHS)
        if history_value is not None:
            if not isinstance(history_value, Mapping) or set(history_value) != set(
                CTRL_WORLD_RELEASED_VIEW_ORDER
            ):
                raise ValueError("ctrl_world_view_history_mapping_invalid")
            histories: dict[str, list[str]] = {}
            for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
                values = history_value[view_id]
                if not isinstance(values, list) or len(values) < CTRL_WORLD_INITIAL_HISTORY_LENGTH:
                    raise ValueError(f"ctrl_world_view_history_too_short:{view_id}")
                histories[view_id] = [
                    str(_safe_file(path, reason=f"ctrl_world_view_history_file_missing:{view_id}"))
                    for path in values
                ]
            return histories

        initial_dir = output_dir / "initial_views"
        initial_dir.mkdir(parents=True, exist_ok=True)
        histories = {}
        for index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
            path = initial_dir / f"view_{index}.png"
            Image.fromarray(np.asarray(observation[view_id], dtype=np.uint8)).save(path)
            histories[view_id] = [str(path)] * CTRL_WORLD_INITIAL_HISTORY_LENGTH
        return histories

    def _initial_state_history(self, observation: Mapping[str, Any]) -> np.ndarray:
        history_value = observation.get(CTRL_WORLD_STATE_HISTORY)
        if history_value is not None:
            history = np.asarray(history_value, dtype=np.float64)
            if (
                history.ndim != 2
                or history.shape[1] != 7
                or history.shape[0] < CTRL_WORLD_INITIAL_HISTORY_LENGTH
                or not np.isfinite(history).all()
            ):
                raise ValueError("ctrl_world_state_history_invalid")
            return history
        current = np.asarray(
            observation.get("blueprint/ctrl_world_cartesian_pose_7d"), dtype=np.float64
        )
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError("ctrl_world_initial_cartesian_pose_7d_required")
        return np.repeat(current[None, :], CTRL_WORLD_INITIAL_HISTORY_LENGTH, axis=0)

    def prepare_transition(
        self,
        *,
        observation: Mapping[str, Any],
        policy_action: Any,
        task_prompt: str,
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        blockers = validate_droid_observation(
            observation, required_views=CTRL_WORLD_RELEASED_VIEW_ORDER
        )
        if blockers:
            raise ValueError(f"ctrl_world_droid_observation_invalid:{blockers[0]}")
        action_blockers = validate_droid_action_chunk(
            policy_action, expected_rows=self.action_chunk_rows
        )
        if action_blockers:
            raise ValueError(f"ctrl_world_policy_action_invalid:{action_blockers[0]}")
        if executed_prefix_steps != 8:
            raise ValueError("ctrl_world_current_reference_requires_eight_executed_rows")

        view_histories = self._initial_view_histories(observation, output_dir)
        state_history = self._initial_state_history(observation)
        selected_state_history = state_history[
            np.asarray(CTRL_WORLD_SELECTED_HISTORY_INDICES)
        ]
        adapted = self.action_adapter.adapt(
            policy_action=policy_action,
            current_joint_position=observation["observation/joint_position"],
            current_gripper_position=observation["observation/gripper_position"],
            history_cartesian_pose_7d=selected_state_history,
            source_action_space="joint_velocity_plus_gripper_position",
        )
        native_action_path = output_dir / "native_policy_action.npy"
        np.save(native_action_path, np.asarray(policy_action, dtype=np.float64), allow_pickle=False)

        selected_histories: dict[str, list[dict[str, str]]] = {}
        for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
            selected_histories[view_id] = []
            for index in CTRL_WORLD_SELECTED_HISTORY_INDICES:
                path = _safe_file(
                    view_histories[view_id][index],
                    reason=f"ctrl_world_selected_history_file_missing:{view_id}",
                )
                selected_histories[view_id].append(
                    {"path": str(path), "sha256": file_sha256(path)}
                )

        return {
            "wam_request": {
                "schema_version": "blueprint_ctrl_world_current_reference_request.v1",
                "query_index": query_index,
                "task_prompt": task_prompt,
                "view_order": list(CTRL_WORLD_RELEASED_VIEW_ORDER),
                "selected_history_views": selected_histories,
                "selected_history_indices": list(CTRL_WORLD_SELECTED_HISTORY_INDICES),
                "action_conditioning_7d": adapted["action_conditioning_7d"],
                "action_conditioning_shape": [11, 7],
                "predicted_frame_count": 5,
                "executed_prefix_steps": 8,
                "executed_prefix_seconds": 8 / 15,
                "physical_future_observation_used": False,
            },
            "native_policy_action_path": str(native_action_path),
            "native_policy_action_sha256": file_sha256(native_action_path),
            "native_policy_action_shape": list(np.asarray(policy_action).shape),
            "openpi_config_name_internal_only": self.openpi_config_name,
            "source_action_space": "joint_velocity_plus_gripper_position",
            "action_adapter_evidence": adapted,
            "reliability_actions_10d": adapted["reliability_actions_10d"],
            "next_joint_position": adapted["next_joint_position"],
            "next_gripper_position": adapted["next_gripper_position"],
            "next_cartesian_pose_7d": adapted["next_cartesian_pose_7d"],
            "full_view_histories": view_histories,
            "full_state_history": state_history,
            "physical_future_observation_used": False,
        }

    def advance_policy_observation(
        self,
        *,
        previous_observation: Mapping[str, Any],
        prepared_transition: Mapping[str, Any],
        wam_prediction: Mapping[str, Any],
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        del previous_observation, executed_prefix_steps, query_index, output_dir
        sequences = wam_prediction.get("generated_view_frame_sequences")
        if not isinstance(sequences, Mapping) or set(sequences) != set(
            CTRL_WORLD_RELEASED_VIEW_ORDER
        ):
            raise ValueError("ctrl_world_prediction_three_view_sequences_missing")
        generated_paths: dict[str, list[Path]] = {}
        for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
            values = sequences[view_id]
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError(f"ctrl_world_prediction_frame_count_invalid:{view_id}")
            generated_paths[view_id] = [
                _safe_file(path, reason=f"ctrl_world_generated_frame_missing:{view_id}")
                for path in values
            ]

        observation: dict[str, Any] = {
            "observation/joint_position": np.asarray(
                prepared_transition["next_joint_position"], dtype=np.float64
            ),
            "observation/gripper_position": np.asarray(
                prepared_transition["next_gripper_position"], dtype=np.float64
            ),
            "prompt": str(prepared_transition["wam_request"]["task_prompt"]),
            "blueprint/ctrl_world_cartesian_pose_7d": np.asarray(
                prepared_transition["next_cartesian_pose_7d"], dtype=np.float64
            ),
            WAM_SOURCE_VIEW_PATHS: {},
        }
        view_histories = {
            view_id: list(prepared_transition["full_view_histories"][view_id])
            for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER
        }
        generated_hashes: dict[str, str] = {}
        for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
            selected = generated_paths[view_id][-1]
            observation[view_id] = self.policy_image_preprocessor(selected)
            observation[WAM_SOURCE_VIEW_PATHS][view_id] = str(selected)
            generated_hashes[view_id] = file_sha256(selected)
            view_histories[view_id].append(str(selected))
        observation[CTRL_WORLD_VIEW_HISTORY_PATHS] = view_histories
        state_history = np.concatenate(
            (
                np.asarray(prepared_transition["full_state_history"], dtype=np.float64),
                np.asarray(prepared_transition["next_cartesian_pose_7d"], dtype=np.float64)[
                    None, :
                ],
            ),
            axis=0,
        )
        observation[CTRL_WORLD_STATE_HISTORY] = state_history
        blockers = validate_droid_observation(
            observation, required_views=CTRL_WORLD_RELEASED_VIEW_ORDER
        )
        if blockers:
            raise ValueError(f"advanced_ctrl_world_observation_invalid:{blockers[0]}")
        return {
            "observation": observation,
            "provenance": {
                "visual_source": "wam_prediction",
                "state_source": "commanded_prefix_kinematics",
                "physical_future_observation_used": False,
                "generated_view_frame_sha256": generated_hashes,
                "generated_view_count": len(generated_hashes),
                "same_wam_arm_generated_all_views": True,
                "ctrl_world_current_reference_not_exact_paper_reproduction": True,
            },
        }


__all__ = [
    "CTRL_WORLD_CURRENT_REFERENCE_POLICIES",
    "CTRL_WORLD_INITIAL_HISTORY_LENGTH",
    "CTRL_WORLD_RELEASED_VIEW_ORDER",
    "CTRL_WORLD_SELECTED_HISTORY_INDICES",
    "CTRL_WORLD_STATE_HISTORY",
    "CTRL_WORLD_VIEW_HISTORY_PATHS",
    "DroidCtrlWorldCurrentReferenceTransitionAdapter",
]
