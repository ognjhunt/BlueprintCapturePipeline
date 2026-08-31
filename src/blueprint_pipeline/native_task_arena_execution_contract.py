"""Static execution-mode contracts for native Task Arena provider bundles.

This module stays dependency-free so bundle builders and paid-provider preflight
can share one exact mode/result/module contract without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping


CONSTRUCTION_RUNTIME_MODULE_NAMES = (
    "articulation_graph_contract.py",
    "articulated_control_planner.py",
    "decision_evidence_contracts.py",
    "native_articulated_construction_plan.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_task_state.py",
    "native_task_construction_plan.py",
    "native_task_construction_authored_contract.py",
    "native_task_construction_validation.py",
    "native_franka_pose_servo.py",
    "native_franka_grasp_geometry.py",
    "native_franka_action_math.py",
    "native_pose_transforms.py",
    "rigid_frame_transforms.py",
    "native_task_arena_readback.py",
    "native_task_arena_device_readback.py",
    "native_task_arena_dependency_contract.py",
    "native_task_arena_import_scope.py",
    "native_task_arena_preconstruction.py",
    "native_task_arena_runtime.py",
    "native_task_isaaclab_launch.py",
    "native_task_camera_observability.py",
    # measure_native_task_camera_observability imports the framing
    # expectation module lazily, so the closure import probe cannot see
    # this edge; ship it beside observability everywhere observability
    # ships or the pod fails at snapshot time, mid paid run.
    "native_task_camera_framing_expectation.py",
    "native_task_nurec_render_setup.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
    "native_task_torch_runtime_lock.py",
    # The warm construction-repair loop invokes cuRobo as a separate process
    # before the next native attempt.  Ship the exact typed boundary and lazy
    # service entry point; cuRobo itself is independently source-pinned and
    # must pass its GPU runtime probe before use.
    "task_evaluation_collision_aware_candidate_generation.py",
    "task_evaluation_curobo_candidate_generator.py",
    "task_evaluation_curobo_candidate_service.py",
    "task_evaluation_control_search_funnel.py",
    "task_evaluation_isaaclab_control_sweep.py",
    "native_task_isaaclab_control_sweep_runtime.py",
    "native_task_curobo_path_execution.py",
    "native_task_servo_command_limits.py",
    "native_construction_terminal_feedback_contract.py",
    "native_task_arena_feedback_bootstrap_runtime.py",
)

RUNTIME_PREFLIGHT_MODULE_NAMES = tuple(
    sorted(
        {
            *CONSTRUCTION_RUNTIME_MODULE_NAMES,
            "native_task_arena_construction_worker.py",
            "rigid_frame_transforms.py",
        }
    )
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
    "task_control_diagnostic_boundary.py",
    "articulation_graph_contract.py",
    "articulated_control_planner.py",
    "decision_evidence_contracts.py",
    "episode_visual_evidence.py",
    "groot_n17_droid_policy_runtime.py",
    "native_articulated_motion_geometry.py",
    "native_articulated_construction_plan.py",
    "native_articulated_task_state.py",
    "native_franka_action_math.py",
    "native_franka_pose_servo.py",
    "native_franka_grasp_geometry.py",
    "native_pose_transforms.py",
    "rigid_frame_transforms.py",
    "native_task_arena_actuator_sweep.py",
    "native_task_arena_bounded_orientation.py",
    "native_task_arena_branch_continuity.py",
    "native_franka_global_seed_search.py",
    "native_task_arena_grasp_roll.py",
    "native_task_arena_construction_worker.py",
    "native_construction_terminal_feedback_contract.py",
    "native_task_arena_feedback_bootstrap_runtime.py",
    "native_task_curobo_path_execution.py",
    "native_task_arena_dependency_contract.py",
    "native_task_arena_import_scope.py",
    "native_task_arena_preconstruction.py",
    "native_task_arena_device_readback.py",
    "native_task_arena_readback.py",
    "native_task_arena_runtime.py",
    "native_task_isaaclab_launch.py",
    "native_task_camera_observability.py",
    # measure_native_task_camera_observability imports the framing
    # expectation module lazily, so the closure import probe cannot see
    # this edge; ship it beside observability everywhere observability
    # ships or the pod fails at snapshot time, mid paid run.
    "native_task_camera_framing_expectation.py",
    "native_task_construction_plan.py",
    "native_task_construction_authored_contract.py",
    "native_task_construction_validation.py",
    "native_task_nurec_render_setup.py",
    "native_task_rigid_controls.py",
    "native_task_episode_environment.py",
    "native_task_runtime_source_packet.py",
    "native_task_runtime_source_provision.py",
    "native_task_torch_runtime_lock.py",
    "native_task_servo_command_limits.py",
    "task_evaluation_collision_aware_candidate_generation.py",
    "task_evaluation_curobo_candidate_generator.py",
    "task_evaluation_curobo_candidate_service.py",
    "task_evaluation_control_search_funnel.py",
    "task_evaluation_isaaclab_control_sweep.py",
    "native_task_isaaclab_control_sweep_runtime.py",
)

POLICY_EXTRA_RUNTIME_MODULE_NAMES = (
    "adp009d_policy_episode.py",
    "adp009d_policy_episode_evidence.py",
    "adp009d_droid_action_execution.py",
    "adp009d_policy_rights.py",
    "droid_policy_bridge.py",
    "groot_n17_wire_client.py",
    "openpi_droid_policy_runtime.py",
    "policy_episode_lifecycle.py",
)

POLICY_RUNTIME_MODULE_NAMES = tuple(
    sorted(
        {
            *CONTROLS_RUNTIME_MODULE_NAMES,
            *POLICY_EXTRA_RUNTIME_MODULE_NAMES,
        }
    )
)


@dataclass(frozen=True)
class NativeTaskArenaExecutionContract:
    expected_output_filename: str
    runtime_module_names: tuple[str, ...]
    policy_candidate_required: bool = False


EXECUTION_MODE_CONTRACTS = {
    "runtime_preflight": NativeTaskArenaExecutionContract(
        expected_output_filename="native_task_arena_runtime_preflight.v1.json",
        runtime_module_names=RUNTIME_PREFLIGHT_MODULE_NAMES,
    ),
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
        runtime_module_names=POLICY_RUNTIME_MODULE_NAMES,
        policy_candidate_required=True,
    ),
    "policy_diagnostic": NativeTaskArenaExecutionContract(
        expected_output_filename=(
            "native_task_arena_policy_diagnostic_result.v1.json"
        ),
        runtime_module_names=POLICY_RUNTIME_MODULE_NAMES,
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

CONTROLS_RESULT_FILENAME = "native_task_arena_control_result.v1.json"
POLICY_CANARY_RESULT_FILENAME = (
    "native_task_arena_policy_canary_session_result.v1.json"
)
CONTROLS_RESULT_SCHEMA_VERSION = "native_task_arena_control_result.v1"
DOWNSTREAM_DIAGNOSTIC_RESULT_SCHEMA_VERSION = (
    "adp_task_synthetic_post_phase5_downstream_diagnostic.v1"
)


def _canonical_digest(value: Mapping[str, object], *, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def native_task_arena_execution_transport_completed(
    result: Mapping[str, object],
    *,
    expected_output_filename: str,
) -> bool:
    """Accept a completed transport without expanding scientific claims.

    Ordinary worker completion is unchanged.  The only additional terminal
    form is the opt-in, development-only controls diagnostic.  Its exact
    non-qualification fields and digest-bound request/receipt are required so
    a policy or ordinary controls result cannot borrow this transport status.
    """

    if result.get("status") == "completed":
        return True
    if expected_output_filename == POLICY_CANARY_RESULT_FILENAME:
        return (
            result.get("status")
            == "runtime_completed_unqualified_pending_closeout"
            and result.get("schema_version")
            == "native_task_arena_policy_canary_session_result.v1"
            and result.get("run_kind") == "internal_policy_canary"
            and result.get("claim_ceiling") == "diagnostic_policy_execution"
            and result.get("learned_policy_rollout_count") == 20
            and result.get("candidate_policy_queried") is True
            and result.get("scene_promotion_performed") is False
            and result.get("official_ranking_performed") is False
        )
    if (
        expected_output_filename != CONTROLS_RESULT_FILENAME
        or result.get("status") != "diagnostic_completed"
        or result.get("schema_version") != CONTROLS_RESULT_SCHEMA_VERSION
        or result.get("controls_qualified") is not False
        or result.get("qualification_effect") != "none"
        or result.get("development_only") is not True
        or result.get("diagnostic_only") is not True
        or result.get("phase5_qualified") is not False
        or result.get("candidate_policy_queried") is not False
        or result.get("candidate_outcomes_accessed") is not False
        or result.get("blockers") != []
        or result.get("phase_reached")
        != "synthetic_post_phase5_downstream_diagnostic_complete"
        or any(
            key in result
            for key in (
                "control_pair",
                "contact_posture_actuator_sweep",
                "contact_target_reachability_probe",
                "contact_close_posture_sweep",
                "contact_acquisition_sweep",
            )
        )
    ):
        return False
    request = result.get(
        "synthetic_post_phase5_downstream_diagnostic_request"
    )
    diagnostic = result.get(
        "synthetic_post_phase5_downstream_diagnostic"
    )
    matrix = result.get("downstream_phase_posture_matrix")
    if not all(
        isinstance(value, Mapping)
        for value in (request, diagnostic, matrix)
    ):
        return False
    assert isinstance(request, Mapping)
    assert isinstance(diagnostic, Mapping)
    assert isinstance(matrix, Mapping)
    try:
        request_payload = dict(request)
        request_payload.pop("status", None)
        request_digest_valid = request.get("request_digest") == _canonical_digest(
            request_payload, field="request_digest"
        )
        diagnostic_digest_valid = diagnostic.get(
            "receipt_digest"
        ) == _canonical_digest(diagnostic, field="receipt_digest")
    except (TypeError, ValueError):
        return False
    return bool(
        request.get("schema_version")
        == "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1"
        and request.get("status") == "requested"
        and request.get("enabled") is True
        and request.get("development_only") is True
        and request.get("qualification_effect") == "none"
        and request_digest_valid
        and diagnostic.get("schema_version")
        == DOWNSTREAM_DIAGNOSTIC_RESULT_SCHEMA_VERSION
        and diagnostic.get("status") == "measured"
        and diagnostic.get("phase5_qualified") is False
        and diagnostic.get("qualification_effect") == "none"
        and diagnostic.get("control_passed") is False
        and diagnostic_digest_valid
        and matrix.get("status") == "not_run"
        and matrix.get("executed_cell_count") == 0
        and matrix.get("represented_configuration_count") == 0
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
    "CONTROLS_RESULT_FILENAME",
    "CONTROLS_RESULT_SCHEMA_VERSION",
    "DOWNSTREAM_DIAGNOSTIC_RESULT_SCHEMA_VERSION",
    "EXECUTION_MODE_CONTRACTS",
    "NATIVE_TASK_ARENA_RESULT_FILENAMES",
    "NATIVE_TASK_ARENA_POLICY_CANDIDATES",
    "NativeTaskArenaExecutionContract",
    "POLICY_EXTRA_RUNTIME_MODULE_NAMES",
    "POLICY_RUNTIME_MODULE_NAMES",
    "RUNTIME_PREFLIGHT_MODULE_NAMES",
    "native_task_arena_execution_transport_completed",
    "required_archive_entries",
]
