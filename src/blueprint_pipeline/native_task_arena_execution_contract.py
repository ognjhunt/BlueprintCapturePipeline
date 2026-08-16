"""Static execution-mode contracts for native Task Arena provider bundles.

This module stays dependency-free so bundle builders and paid-provider preflight
can share one exact mode/result/module contract without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass


CONSTRUCTION_RUNTIME_MODULE_NAMES = (
    "articulation_graph_contract.py",
    "articulated_control_planner.py",
    "decision_evidence_contracts.py",
    "native_articulated_construction_plan.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_task_state.py",
    "native_task_construction_plan.py",
    "native_franka_pose_servo.py",
    "native_franka_action_math.py",
    "native_pose_transforms.py",
    "native_task_arena_readback.py",
    "native_task_arena_device_readback.py",
    "native_task_arena_import_scope.py",
    "native_task_arena_preconstruction.py",
    "native_task_arena_runtime.py",
    "native_task_isaaclab_launch.py",
    "native_task_camera_observability.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
)

CONTROLS_RUNTIME_MODULE_NAMES = (
    "adp009d_control_episode.py",
    "adp009d_contact_envelope.py",
    "adp009d_newton_gripper_drive.py",
    "adp009d_physics_backend_comparison.py",
    "adp009d_droid_observation.py",
    "adp009d_isaac_episode_adapter.py",
    "adp009d_task_scoring.py",
    "adp_task_scoring.py",
    "articulation_graph_contract.py",
    "decision_evidence_contracts.py",
    "episode_visual_evidence.py",
    "groot_n17_droid_policy_runtime.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_task_state.py",
    "native_franka_action_math.py",
    "native_franka_pose_servo.py",
    "native_pose_transforms.py",
    "native_task_arena_construction_worker.py",
    "native_task_arena_import_scope.py",
    "native_task_arena_preconstruction.py",
    "native_task_arena_device_readback.py",
    "native_task_arena_readback.py",
    "native_task_arena_runtime.py",
    "native_task_isaaclab_launch.py",
    "native_task_camera_observability.py",
    "native_task_episode_environment.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
)

POLICY_EXTRA_RUNTIME_MODULE_NAMES = (
    "adp009d_policy_episode.py",
    "adp009d_droid_action_execution.py",
    "droid_policy_bridge.py",
    "openpi_droid_policy_runtime.py",
)


@dataclass(frozen=True)
class NativeTaskArenaExecutionContract:
    expected_output_filename: str
    runtime_module_names: tuple[str, ...]
    policy_candidate_required: bool = False


EXECUTION_MODE_CONTRACTS = {
    "construction_canary": NativeTaskArenaExecutionContract(
        expected_output_filename="native_task_arena_construction_result.v1.json",
        runtime_module_names=CONSTRUCTION_RUNTIME_MODULE_NAMES,
    ),
    "controls": NativeTaskArenaExecutionContract(
        expected_output_filename="native_task_arena_control_result.v1.json",
        runtime_module_names=CONTROLS_RUNTIME_MODULE_NAMES,
    ),
    "policy": NativeTaskArenaExecutionContract(
        expected_output_filename="native_task_arena_policy_result.v1.json",
        runtime_module_names=tuple(
            sorted(
                {
                    *CONTROLS_RUNTIME_MODULE_NAMES,
                    *POLICY_EXTRA_RUNTIME_MODULE_NAMES,
                }
            )
        ),
        policy_candidate_required=True,
    ),
}

NATIVE_TASK_ARENA_RESULT_FILENAMES = frozenset(
    contract.expected_output_filename
    for contract in EXECUTION_MODE_CONTRACTS.values()
)
NATIVE_TASK_ARENA_POLICY_CANDIDATES = frozenset(
    {"groot_n17_droid", "pi05_droid"}
)


def required_archive_entries(execution_mode: str) -> set[str]:
    """Return the exact internal Python module members required by one mode."""

    contract = EXECUTION_MODE_CONTRACTS.get(str(execution_mode))
    if contract is None:
        return set()
    return {
        f"provider_runtime/blueprint_pipeline/{name}"
        for name in contract.runtime_module_names
    }


__all__ = [
    "CONSTRUCTION_RUNTIME_MODULE_NAMES",
    "CONTROLS_RUNTIME_MODULE_NAMES",
    "EXECUTION_MODE_CONTRACTS",
    "NATIVE_TASK_ARENA_RESULT_FILENAMES",
    "NATIVE_TASK_ARENA_POLICY_CANDIDATES",
    "NativeTaskArenaExecutionContract",
    "POLICY_EXTRA_RUNTIME_MODULE_NAMES",
    "required_archive_entries",
]
