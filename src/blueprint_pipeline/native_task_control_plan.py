"""Dispatch native controls plans without scene- or object-specific branches.

The articulated branch preserves the existing compatibility adapter.  The
rigid branch does not synthesize grasp geometry: it replays only the exact,
digest-bound phases that already passed the native rigid construction gate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_articulated_control_plan import (
    materialize_native_articulated_control_plan,
)
from .native_task_construction_plan import (
    GRAPH_ARTICULATED_SCHEMA_VERSION,
    NativeTaskConstructionPlanError,
    evaluate_graph_articulated_construction_gates,
    materialize_native_task_construction_phase_plan,
)


SCHEMA_VERSION = "adp_task_control_plan.v1"
RIGID_CONSTRUCTION_PLAN_SCHEMA_VERSION = (
    "native_rigid_construction_phase_plan.v1"
)
RIGID_CONSTRUCTION_GATE_SCHEMA_VERSION = (
    "native_rigid_construction_gate_evaluation.v1"
)
CONSTRUCTION_RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"
RIGID_TASK_SPEC_SCHEMA_VERSION = "adp_task_spec.v2"
GRAPH_ARTICULATED_GATE_SCHEMA_VERSION = (
    "native_articulated_graph_construction_gate_evaluation.v1"
)
SUPPORTED_TASK_KINDS = frozenset({"articulated_open_close", "rigid_pick_place"})
MAX_JOINT_DELTA_RAD = 0.03
MAX_JOINT_SETPOINT_LEAD_RAD = 0.20


class NativeTaskControlPlanError(ValueError):
    """Stable task-neutral controls-plan admission failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _copy_mapping(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskControlPlanError([error]) from exc
    if not isinstance(copied, dict):
        raise NativeTaskControlPlanError([error])
    return copied


def _positive_number(value: Any, *, error: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeTaskControlPlanError([error]) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise NativeTaskControlPlanError([error])
    return result


def _positive_integer(value: Any, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeTaskControlPlanError([error])
    return value


def _vector(value: Any, *, length: int, error: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeTaskControlPlanError([error]) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise NativeTaskControlPlanError([error])
    return result


def _quaternion(value: Any, *, error: str) -> list[float]:
    result = _vector(value, length=4, error=error)
    norm = math.sqrt(sum(item * item for item in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise NativeTaskControlPlanError([error])
    return result


def _valid_digest_bound_mapping(
    value: Any, *, digest_field: str, schema_version: str | None = None
) -> bool:
    return (
        isinstance(value, Mapping)
        and (schema_version is None or value.get("schema_version") == schema_version)
        and value.get(digest_field)
        == canonical_digest(dict(value), digest_field=digest_field)
    )


def materialize_native_rigid_control_plan(
    *, scene_plan: Mapping[str, Any], construction_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse one qualified rigid construction trajectory as the positive control."""

    scene = _copy_mapping(scene_plan, error="native_rigid_control_scene_plan_invalid")
    construction = _copy_mapping(
        construction_result,
        error="native_rigid_control_construction_result_invalid",
    )
    errors: list[str] = []
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene.get("task_kind") != "rigid_pick_place"
        or scene.get("plan_digest")
        != canonical_digest(scene, digest_field="plan_digest")
    ):
        errors.append("native_rigid_control_scene_plan_invalid")
    scenario = scene.get("scenario")
    if (
        not isinstance(scenario, Mapping)
        or not isinstance(scenario.get("cell_id"), str)
        or not scenario["cell_id"].strip()
    ):
        errors.append("native_rigid_control_scenario_invalid")
        scenario = {}
    task_spec = scene.get("task_spec")
    if (
        not isinstance(task_spec, Mapping)
        or task_spec.get("schema_version") != RIGID_TASK_SPEC_SCHEMA_VERSION
        or task_spec.get("task_kind") != "rigid_pick_place"
    ):
        errors.append("native_rigid_control_task_spec_invalid")
        task_spec = {}
    if task_spec.get("release_required") is not True:
        errors.append("native_rigid_control_release_contract_invalid")
    try:
        _positive_number(
            task_spec.get("release_gripper_width_min_m"),
            error="native_rigid_control_release_contract_invalid",
        )
        _positive_number(
            task_spec.get("movement_epsilon_m"),
            error="native_rigid_control_movement_contract_invalid",
        )
    except NativeTaskControlPlanError as exc:
        errors.extend(exc.errors)
    affordance = task_spec.get("interaction_affordance")
    if not _valid_digest_bound_mapping(
        affordance, digest_field="affordance_digest"
    ):
        errors.append("native_rigid_control_interaction_affordance_invalid")

    if not _valid_digest_bound_mapping(
        construction,
        digest_field="result_digest",
        schema_version=CONSTRUCTION_RESULT_SCHEMA_VERSION,
    ):
        errors.append("native_rigid_control_construction_result_invalid")
    if (
        construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("blockers") != []
    ):
        errors.append("native_rigid_control_construction_not_qualified")
    if construction.get("scene_plan_digest") != scene.get("plan_digest"):
        errors.append("native_rigid_control_construction_binding_mismatch")

    phase_plan = construction.get("construction_phase_plan")
    if (
        not _valid_digest_bound_mapping(
            phase_plan,
            digest_field="plan_digest",
            schema_version=RIGID_CONSTRUCTION_PLAN_SCHEMA_VERSION,
        )
        or phase_plan.get("task_kind") != "rigid_pick_place"
        or phase_plan.get("scene_plan_digest") != scene.get("plan_digest")
    ):
        errors.append("native_rigid_control_construction_phase_plan_invalid")
        phase_plan = {}
    phase_affordance = phase_plan.get("interaction_affordance")
    if (
        not _valid_digest_bound_mapping(
            phase_affordance, digest_field="affordance_digest"
        )
        or phase_affordance != affordance
    ):
        errors.append("native_rigid_control_interaction_affordance_mismatch")

    gate_evaluation = construction.get("rigid_construction_gates")
    if not _valid_digest_bound_mapping(
        gate_evaluation,
        digest_field="evaluation_digest",
        schema_version=RIGID_CONSTRUCTION_GATE_SCHEMA_VERSION,
    ):
        errors.append("native_rigid_control_gate_evaluation_invalid")
        gate_evaluation = {}
    gate_rows = gate_evaluation.get("gates")
    required_gate_ids = phase_plan.get("required_gate_ids")
    if (
        gate_evaluation.get("phase_plan_digest") != phase_plan.get("plan_digest")
        or gate_evaluation.get("passed") is not True
        or gate_evaluation.get("all_phase_targets_reached") is not True
        or gate_evaluation.get("blockers") != []
        or not isinstance(gate_rows, list)
        or not isinstance(required_gate_ids, list)
        or sorted(
            str(row.get("gate_id") or "")
            for row in gate_rows
            if isinstance(row, Mapping)
        )
        != sorted(str(value) for value in required_gate_ids)
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in gate_rows or []
        )
    ):
        errors.append("native_rigid_control_gate_evaluation_not_qualified")

    camera_gates = construction.get("camera_gates")
    if (
        not isinstance(camera_gates, Mapping)
        or set(camera_gates) != {"external", "wrist", "overview"}
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in camera_gates.values()
        )
    ):
        errors.append("native_rigid_control_camera_preflight_incomplete")
    reset_replay = construction.get("reset_replay")
    if not isinstance(reset_replay, Mapping) or reset_replay.get("passed") is not True:
        errors.append("native_rigid_control_reset_preflight_incomplete")

    phases = phase_plan.get("phases")
    phase_results = construction.get("phase_results")
    if not isinstance(phases, list) or not phases:
        errors.append("native_rigid_control_construction_phases_invalid")
        phases = []
    if not isinstance(phase_results, list) or not phase_results:
        errors.append("native_rigid_control_construction_phase_results_invalid")
        phase_results = []
    expected_ids = [
        str(row.get("phase_id") or "") for row in phases if isinstance(row, Mapping)
    ]
    observed_ids = [
        str(row.get("phase_id") or "")
        for row in phase_results
        if isinstance(row, Mapping)
    ]
    if (
        len(expected_ids) != len(phases)
        or not all(expected_ids)
        or len(set(expected_ids)) != len(expected_ids)
        or observed_ids != expected_ids
        or len(observed_ids) != len(phase_results)
        or any(
            not isinstance(row, Mapping) or row.get("target_reached") is not True
            for row in phase_results
        )
    ):
        errors.append("native_rigid_control_construction_phase_results_invalid")

    execution = phase_plan.get("execution_parameters")
    if not isinstance(execution, Mapping):
        errors.append("native_rigid_control_execution_parameters_invalid")
        execution = {}
    try:
        arrival_tolerance = _positive_number(
            execution.get("arrival_tolerance_m"),
            error="native_rigid_control_execution_parameters_invalid",
        )
        arrival_orientation_tolerance = _positive_number(
            execution.get("arrival_orientation_tolerance_rad"),
            error="native_rigid_control_execution_parameters_invalid",
        )
        stable_samples = _positive_integer(
            execution.get("stable_samples"),
            error="native_rigid_control_execution_parameters_invalid",
        )
        maximum_steps_per_phase = _positive_integer(
            execution.get("maximum_steps_per_phase"),
            error="native_rigid_control_execution_parameters_invalid",
        )
    except NativeTaskControlPlanError as exc:
        errors.extend(exc.errors)
        arrival_tolerance = 0.0
        arrival_orientation_tolerance = 0.0
        stable_samples = 0
        maximum_steps_per_phase = 0

    actions: list[dict[str, Any]] = []
    if len(phases) == len(phase_results):
        for index, (phase, observed) in enumerate(
            zip(phases, phase_results, strict=True)
        ):
            if not isinstance(phase, Mapping) or not isinstance(observed, Mapping):
                errors.append(f"native_rigid_control_phase_invalid:{index}")
                continue
            try:
                position = _vector(
                    phase.get("position_world_m"),
                    length=3,
                    error=f"native_rigid_control_phase_invalid:{index}",
                )
                orientation = _quaternion(
                    phase.get("orientation_world_xyzw"),
                    error=f"native_rigid_control_phase_invalid:{index}",
                )
                observed_steps = _positive_integer(
                    observed.get("steps"),
                    error=f"native_rigid_control_phase_steps_invalid:{index}",
                )
            except NativeTaskControlPlanError as exc:
                errors.extend(exc.errors)
                continue
            gripper_state = str(phase.get("gripper_state") or "")
            if (
                gripper_state not in {"open", "closed"}
                or observed_steps < stable_samples
                or observed_steps > maximum_steps_per_phase
            ):
                errors.append(f"native_rigid_control_phase_invalid:{index}")
                continue
            actions.append(
                {
                    "phase_id": str(phase["phase_id"]),
                    "mode": "ik_pose",
                    "target_position_world_m": position,
                    "target_quaternion_world_xyzw": orientation,
                    "gripper_state": gripper_state,
                    # Reuse the exact deterministic duration that qualified in
                    # construction; controls do not silently tune a new motion.
                    "minimum_steps": observed_steps,
                    "maximum_steps": observed_steps,
                    "arrival_tolerance_m": arrival_tolerance,
                    "arrival_orientation_tolerance_rad": (
                        arrival_orientation_tolerance
                    ),
                    "arrival_stability_steps": stable_samples,
                    "max_joint_delta_rad": MAX_JOINT_DELTA_RAD,
                    "max_joint_setpoint_lead_rad": MAX_JOINT_SETPOINT_LEAD_RAD,
                }
            )

    settle_steps = task_spec.get("settle_window_samples")
    maximum_action_steps = task_spec.get("maximum_action_steps")
    try:
        settle_steps = _positive_integer(
            settle_steps, error="native_rigid_control_settle_window_invalid"
        )
        maximum_action_steps = _positive_integer(
            maximum_action_steps,
            error="native_rigid_control_action_budget_invalid",
        )
    except NativeTaskControlPlanError as exc:
        errors.extend(exc.errors)
        settle_steps = 0
        maximum_action_steps = 0
    maximum_steps = sum(int(row["maximum_steps"]) for row in actions) + int(
        settle_steps
    )
    if maximum_steps > maximum_action_steps or settle_steps > maximum_action_steps:
        errors.append("native_rigid_control_action_budget_exceeded")
    if errors:
        raise NativeTaskControlPlanError(errors)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_kind": "rigid_pick_place",
        "cell_id": scenario["cell_id"],
        "task_spec_digest": canonical_digest(scene["task_spec"]),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": construction["result_digest"],
        "zero_action_steps": settle_steps,
        "scripted_positive_actions": actions,
        "maximum_scripted_and_settle_steps": maximum_steps,
        "construction_scene_plan_digest": scene["plan_digest"],
        "construction_clearance_plan_digest": phase_plan["plan_digest"],
        "construction_gate_evaluation_digest": gate_evaluation[
            "evaluation_digest"
        ],
        "interaction_affordance_digest": affordance["affordance_digest"],
        "positive_trajectory_reexecutes_exact_qualified_phase_targets_and_budgets": True,
        "candidate_policy_queried": False,
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def materialize_native_graph_articulated_control_plan(
    *, scene_plan: Mapping[str, Any], construction_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay exact graph-bound contact phases after complete clearance evidence."""

    scene = _copy_mapping(
        scene_plan, error="native_articulated_graph_control_scene_plan_invalid"
    )
    construction = _copy_mapping(
        construction_result,
        error="native_articulated_graph_control_construction_result_invalid",
    )
    errors: list[str] = []
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene.get("task_kind") != "articulated_open_close"
        or scene.get("plan_digest")
        != canonical_digest(scene, digest_field="plan_digest")
    ):
        errors.append("native_articulated_graph_control_scene_plan_invalid")
    scenario = scene.get("scenario")
    if (
        not isinstance(scenario, Mapping)
        or not isinstance(scenario.get("cell_id"), str)
        or not scenario["cell_id"].strip()
    ):
        errors.append("native_articulated_graph_control_scenario_invalid")
        scenario = {}
    task_spec = scene.get("task_spec")
    if (
        not isinstance(task_spec, Mapping)
        or task_spec.get("schema_version") != "adp_task_spec.v2"
        or task_spec.get("task_kind") != "articulated_open_close"
    ):
        errors.append("native_articulated_graph_control_task_spec_invalid")
        task_spec = {}
    if not _valid_digest_bound_mapping(
        construction,
        digest_field="result_digest",
        schema_version=CONSTRUCTION_RESULT_SCHEMA_VERSION,
    ):
        errors.append("native_articulated_graph_control_construction_result_invalid")
    if (
        construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("blockers") != []
    ):
        errors.append("native_articulated_graph_control_construction_not_qualified")
    if construction.get("scene_plan_digest") != scene.get("plan_digest"):
        errors.append("native_articulated_graph_control_construction_binding_mismatch")

    phase_plan = construction.get("construction_phase_plan")
    if (
        not _valid_digest_bound_mapping(
            phase_plan,
            digest_field="plan_digest",
            schema_version=GRAPH_ARTICULATED_SCHEMA_VERSION,
        )
        or phase_plan.get("task_kind") != "articulated_open_close"
        or phase_plan.get("scene_plan_digest") != scene.get("plan_digest")
    ):
        errors.append("native_articulated_graph_control_phase_plan_invalid")
        phase_plan = {}
    else:
        try:
            recomputed = materialize_native_task_construction_phase_plan(scene)
        except NativeTaskConstructionPlanError as exc:
            errors.extend(
                f"native_articulated_graph_control_recompile_failed:{error}"
                for error in exc.errors
            )
        else:
            if recomputed != phase_plan:
                errors.append("native_articulated_graph_control_phase_plan_recompile_mismatch")

    phase_results = construction.get("phase_results")
    reset_replay = construction.get("reset_replay")
    graph_gate_evaluation: dict[str, Any] = {}
    if phase_plan:
        try:
            graph_gate_evaluation = evaluate_graph_articulated_construction_gates(
                phase_plan=phase_plan,
                phase_results=phase_results,
                reset_replay=reset_replay,
            )
        except NativeTaskConstructionPlanError as exc:
            errors.extend(
                f"native_articulated_graph_control_gate_evaluation_failed:{error}"
                for error in exc.errors
            )
        else:
            if (
                graph_gate_evaluation.get("schema_version")
                != GRAPH_ARTICULATED_GATE_SCHEMA_VERSION
                or graph_gate_evaluation.get("passed") is not True
                or graph_gate_evaluation.get("blockers") != []
            ):
                errors.append("native_articulated_graph_control_gate_evaluation_not_qualified")
    camera_gates = construction.get("camera_gates")
    if (
        not isinstance(camera_gates, Mapping)
        or set(camera_gates) != {"external", "wrist", "overview"}
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in camera_gates.values()
        )
    ):
        errors.append("native_articulated_graph_control_camera_preflight_incomplete")
    if not isinstance(reset_replay, Mapping) or reset_replay.get("passed") is not True:
        errors.append("native_articulated_graph_control_reset_preflight_incomplete")

    exact_phases = phase_plan.get("exact_contact_phases") if phase_plan else None
    if (
        not isinstance(exact_phases, list)
        or not exact_phases
        or any(not isinstance(row, Mapping) for row in exact_phases)
    ):
        errors.append("native_articulated_graph_control_exact_contact_phases_invalid")
        exact_phases = []
    affordance = phase_plan.get("interaction_affordance") if phase_plan else {}
    if not _valid_digest_bound_mapping(
        affordance, digest_field="affordance_digest"
    ):
        errors.append("native_articulated_graph_control_interaction_affordance_invalid")
        affordance = {}
    actions: list[dict[str, Any]] = []
    if affordance:
        for index, phase in enumerate(exact_phases):
            try:
                position = _vector(
                    phase.get("position_world_m"),
                    length=3,
                    error=f"native_articulated_graph_control_phase_invalid:{index}",
                )
                orientation = _quaternion(
                    phase.get("orientation_world_xyzw"),
                    error=f"native_articulated_graph_control_phase_invalid:{index}",
                )
            except NativeTaskControlPlanError as exc:
                errors.extend(exc.errors)
                continue
            gripper_state = str(phase.get("gripper_state") or "")
            if gripper_state not in {"open", "closed"}:
                errors.append(f"native_articulated_graph_control_phase_invalid:{index}")
                continue
            dwell = phase.get("phase_id") in {"contact_close", "release"}
            actions.append(
                {
                    "phase_id": str(phase.get("phase_id") or ""),
                    "mode": "ik_pose",
                    "target_position_world_m": position,
                    "target_quaternion_world_xyzw": orientation,
                    "gripper_state": gripper_state,
                    "minimum_steps": int(
                        affordance[
                            "gripper_dwell_minimum_steps"
                            if dwell
                            else "motion_minimum_steps"
                        ]
                    ),
                    "maximum_steps": int(
                        affordance[
                            "gripper_dwell_maximum_steps"
                            if dwell
                            else "motion_maximum_steps"
                        ]
                    ),
                    "arrival_tolerance_m": float(affordance["arrival_tolerance_m"]),
                    "arrival_orientation_tolerance_rad": float(
                        phase["arrival_orientation_tolerance_rad"]
                    ),
                    "arrival_stability_steps": int(
                        affordance["arrival_stability_steps"]
                    ),
                    "max_joint_delta_rad": float(affordance["max_joint_delta_rad"]),
                    "max_joint_setpoint_lead_rad": float(
                        affordance["max_joint_setpoint_lead_rad"]
                    ),
                    "gate_ids": list(phase.get("gate_ids") or []),
                    **(
                        {
                            "expected_joint_positions": dict(
                                phase["expected_joint_positions"]
                            )
                        }
                        if isinstance(phase.get("expected_joint_positions"), Mapping)
                        else {}
                    ),
                }
            )
    if any(not row["phase_id"] for row in actions) or len(
        {row["phase_id"] for row in actions}
    ) != len(actions):
        errors.append("native_articulated_graph_control_phase_ids_invalid")

    try:
        settle_steps = _positive_integer(
            task_spec.get("settle_window_samples"),
            error="native_articulated_graph_control_settle_window_invalid",
        )
        maximum_action_steps = _positive_integer(
            task_spec.get("maximum_action_steps"),
            error="native_articulated_graph_control_action_budget_invalid",
        )
    except NativeTaskControlPlanError as exc:
        errors.extend(exc.errors)
        settle_steps = 0
        maximum_action_steps = 0
    maximum_steps = sum(int(row["maximum_steps"]) for row in actions) + settle_steps
    if maximum_steps > maximum_action_steps:
        errors.append("native_articulated_graph_control_action_budget_exceeded")
    if errors:
        raise NativeTaskControlPlanError(errors)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_kind": "articulated_open_close",
        "cell_id": scenario["cell_id"],
        "task_spec_digest": canonical_digest(scene["task_spec"]),
        "trajectory_source": "native_ik_preflight",
        "trajectory_derivation": (
            "digest_bound_graph_affordance_after_native_clearance"
        ),
        "planner_receipt_digest": construction["result_digest"],
        "zero_action_steps": settle_steps,
        "scripted_positive_actions": actions,
        "maximum_scripted_and_settle_steps": maximum_steps,
        "construction_scene_plan_digest": scene["plan_digest"],
        "construction_clearance_plan_digest": phase_plan["plan_digest"],
        "construction_gate_evaluation_digest": graph_gate_evaluation[
            "evaluation_digest"
        ],
        "articulation_graph_digest": phase_plan["articulation_graph_digest"],
        "interaction_affordance_digest": affordance["affordance_digest"],
        "target_joint_ids": phase_plan["joint_ids_by_role"]["target"],
        "dependent_joint_ids": phase_plan["joint_ids_by_role"]["dependent"],
        "passive_joint_ids": phase_plan["joint_ids_by_role"]["passive"],
        "locked_joint_ids": phase_plan["joint_ids_by_role"]["locked"],
        "positive_trajectory_reexecutes_exact_qualified_phase_targets_and_budgets": True,
        "candidate_policy_queried": False,
        "plan_digest": "",
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def materialize_native_task_control_plan(
    *, scene_plan: Mapping[str, Any], construction_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Dispatch one sealed task to its task-neutral native controls adapter."""

    task_kind = str(scene_plan.get("task_kind") or "")
    if task_kind == "articulated_open_close":
        if (scene_plan.get("task_spec") or {}).get("schema_version") == "adp_task_spec.v2":
            return materialize_native_graph_articulated_control_plan(
                scene_plan=scene_plan,
                construction_result=construction_result,
            )
        return materialize_native_articulated_control_plan(
            scene_plan=scene_plan,
            construction_result=construction_result,
        )
    if task_kind == "rigid_pick_place":
        return materialize_native_rigid_control_plan(
            scene_plan=scene_plan,
            construction_result=construction_result,
        )
    raise NativeTaskControlPlanError(
        [f"native_task_control_task_kind_unsupported:{task_kind or 'missing'}"]
    )


__all__ = [
    "NativeTaskControlPlanError",
    "SCHEMA_VERSION",
    "SUPPORTED_TASK_KINDS",
    "materialize_native_graph_articulated_control_plan",
    "materialize_native_rigid_control_plan",
    "materialize_native_task_control_plan",
]
