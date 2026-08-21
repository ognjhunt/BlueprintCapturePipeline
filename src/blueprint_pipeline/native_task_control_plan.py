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
from .native_franka_action_math import is_unauthored_identity_quaternion_xyzw
from .native_articulated_control_plan import (
    materialize_native_articulated_control_plan,
)
from .native_task_construction_plan import (
    GRAPH_ARTICULATED_SCHEMA_VERSION,
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD,
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
# NVIDIA GraspDataGen's Robotiq 2F-85 definition uses an 18.5 mm minimum bite:
# half the 37 mm finger-pad length. Native c10 then placed the measured distal
# pad center at that target and measured 0 N on both allowed inner fingers;
# c7's prior aggregate-frame bias placed the physical pads about 9.7 mm deeper,
# produced strong allowed contact, and over-seated into a forbidden collision.
# Use the rounded midpoint of that measured bracket for the next explicit,
# receipt-visible contact-surface qualification instead of hiding it in TCP.
ROBOTOIQ_2F85_MINIMUM_BITE_DEPTH_M = 0.0185
ROBOTOIQ_2F85_BITE_BRACKET_OFFSET_M = 0.005
ROBOTOIQ_2F85_BITE_DEPTH_M = (
    ROBOTOIQ_2F85_MINIMUM_BITE_DEPTH_M + ROBOTOIQ_2F85_BITE_BRACKET_OFFSET_M
)
ROBOTOIQ_2F85_BITE_SOURCE = (
    "NVlabs/GraspDataGen:robotiq_2f_85:minimum_bite=0.0185;"
    "Blueprint:c7_c10_measured_depth_bracket_midpoint=0.005"
)
# c10/c11 attributed a 718-1423 N prealign contact to panda_link5 against the
# same retained-room collider while the grasp TCP sat at the 0.30 m clearance
# target. Move only the controls prealign target 5 cm toward the already
# construction-qualified approach segment; contact and sweep targets remain
# unchanged.
ROBOTOIQ_2F85_PREALIGN_RETRACTION_M = 0.05
ROBOTOIQ_2F85_PREALIGN_RETRACTION_SOURCE = (
    "Blueprint:c10_c11:panda_link5_vs_Z4P5JBBVAJJWSPTUK4888888"
)
# c4 measured the live pad midpoint 12.9 mm outside a right-rim patch only
# 1.23 mm thick in the radial direction.  The generic 20 mm motion tolerance
# therefore admitted an empty-space grasp.  Require the initial open/close
# contact phases to place the pad center within 5 mm of the source-derived
# patch target; subsequent articulated path phases retain the task's authored
# motion tolerance and the deterministic scorer remains the success authority.
ROBOTOIQ_2F85_EXACT_CONTACT_ARRIVAL_TOLERANCE_M = 0.005
ROBOTOIQ_2F85_EXACT_CONTACT_ARRIVAL_TOLERANCE_SOURCE = (
    "Blueprint:c4_live_pad_midpoint_vs_source_rim_patch_radial_overlap"
)


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
                    "position_only_arrival": bool(
                        phase.get("position_only_arrival") is True
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
    if affordance and is_unauthored_identity_quaternion_xyzw(
        affordance.get("gripper_orientation_contact_xyzw")
    ):
        # These phases close the gripper on the handle, so the orientation has to
        # be authored from the real handle geometry.  Identity is the placeholder
        # a quaternion field holds when nobody set it, and it is 120 degrees from
        # this arm's natural hand pose.  Refuse rather than replay it.
        errors.append(
            "native_articulated_graph_control_gripper_orientation_contact_xyzw"
            "_unauthored_identity"
        )
        affordance = {}
    observed_phases = {
        str(row.get("phase_id") or ""): row
        for row in phase_results or []
        if isinstance(row, Mapping)
    }
    qualified_clearance_targets = {
        str(row.get("phase_id") or ""): row
        for row in phase_plan.get("phases") or []
        if isinstance(row, Mapping)
    }
    exact_phase_targets = {
        str(row.get("phase_id") or ""): row for row in exact_phases
    }
    def measured_construction_phase_id(phase_id: str) -> str | None:
        if phase_id in {"prealign", "approach", "retreat"}:
            return phase_id
        if phase_id in {"contact_open", "contact_close"}:
            return "contact_sweep_clearance_00"
        if phase_id.startswith("joint_path_"):
            return "contact_sweep_clearance_" + phase_id.removeprefix(
                "joint_path_"
            )
        if phase_id == "release":
            return "release_clearance"
        return None

    def motion_step_budget(phase_id: str) -> tuple[int, str | None, int | None]:
        authored = int(affordance["motion_maximum_steps"])
        construction_phase_id = measured_construction_phase_id(phase_id)
        observed = observed_phases.get(construction_phase_id or "")
        observed_steps = (
            int(observed.get("steps"))
            if isinstance(observed, Mapping)
            and observed.get("target_reached") is True
            and isinstance(observed.get("steps"), int)
            and not isinstance(observed.get("steps"), bool)
            and int(observed.get("steps")) > 0
            else None
        )
        if observed_steps is None:
            return authored, construction_phase_id, None
        # Construction measured the same TCP targets without contact. Controls
        # may need longer under door load, so retain 3x the observed duration,
        # five setup ticks, and a 25-tick floor, capped by the already-authored
        # fail-safe maximum. r40's 93-step path then receives a 326-step total
        # controls budget rather than the impossible 576-step restatement.
        derived = min(
            authored,
            max(
                int(affordance["motion_minimum_steps"]),
                25,
                observed_steps * 3 + 5,
            ),
        )
        return derived, construction_phase_id, observed_steps

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
            phase_id = str(phase.get("phase_id") or "")
            prealign_retraction = 0.0
            prealign_retraction_toward_phase_id = None
            if phase_id == "prealign":
                approach = exact_phase_targets.get("approach")
                try:
                    approach_position = _vector(
                        approach.get("position_world_m")
                        if isinstance(approach, Mapping)
                        else None,
                        length=3,
                        error="native_articulated_graph_control_approach_target_missing",
                    )
                except NativeTaskControlPlanError as exc:
                    errors.extend(exc.errors)
                    continue
                toward = [
                    approach_position[axis] - position[axis]
                    for axis in range(3)
                ]
                toward_norm = math.sqrt(sum(value * value for value in toward))
                if (
                    not math.isfinite(toward_norm)
                    or toward_norm <= ROBOTOIQ_2F85_PREALIGN_RETRACTION_M
                ):
                    errors.append(
                        "native_articulated_graph_control_prealign_retraction_invalid"
                    )
                    continue
                position = [
                    position[axis]
                    + toward[axis]
                    / toward_norm
                    * ROBOTOIQ_2F85_PREALIGN_RETRACTION_M
                    for axis in range(3)
                ]
                prealign_retraction = ROBOTOIQ_2F85_PREALIGN_RETRACTION_M
                prealign_retraction_toward_phase_id = "approach"
            bite_depth = 0.0
            bite_direction_source_phase_id = None
            contact_standoff = 0.0
            contact_standoff_source = None
            clearance_phase_id = measured_construction_phase_id(phase_id)
            if phase_id in {"contact_open", "contact_close", "release"} or phase_id.startswith(
                "joint_path_"
            ):
                clearance = qualified_clearance_targets.get(clearance_phase_id or "")
                try:
                    clearance_position = _vector(
                        clearance.get("position_world_m")
                        if isinstance(clearance, Mapping)
                        else None,
                        length=3,
                        error=(
                            "native_articulated_graph_control_"
                            f"qualified_clearance_target_missing:{phase_id}"
                        ),
                    )
                except NativeTaskControlPlanError as exc:
                    errors.extend(exc.errors)
                    continue
                outward = [
                    clearance_position[axis] - position[axis]
                    for axis in range(3)
                ]
                outward_norm = math.sqrt(sum(value * value for value in outward))
                if (
                    not math.isfinite(outward_norm)
                    or abs(outward_norm - float(affordance["sweep_clearance_m"]))
                    > 1.0e-6
                ):
                    errors.append(
                        "native_articulated_graph_control_"
                        f"qualified_clearance_direction_invalid:{phase_id}"
                    )
                    continue
                outward_unit = [value / outward_norm for value in outward]
                explicit_standoff = float(
                    affordance.get("contact_outward_standoff_m", 0.0)
                )
                if explicit_standoff > 0.0:
                    # Construction already qualified the receipt-bound
                    # outward-shifted target. Re-applying the generic inward
                    # bite here would undo that clearance and reproduce the
                    # forbidden palm/knuckle contact it measured.
                    contact_standoff = explicit_standoff
                    contact_standoff_source = (
                        "native_droid_grasp_swept_volume"
                    )
                else:
                    bite_depth = ROBOTOIQ_2F85_BITE_DEPTH_M
                    position = [
                        position[axis] - outward_unit[axis] * bite_depth
                        for axis in range(3)
                    ]
                    contact_standoff = -bite_depth
                    contact_standoff_source = ROBOTOIQ_2F85_BITE_SOURCE
                bite_direction_source_phase_id = clearance_phase_id
            derived_maximum, construction_phase_id, observed_steps = (
                motion_step_budget(phase_id)
                if not dwell
                else (
                    int(affordance["gripper_dwell_maximum_steps"]),
                    None,
                    None,
                )
            )
            exact_contact_arrival = phase_id in {
                "contact_open",
                "contact_close",
            }
            arrival_tolerance_m = min(
                float(affordance["arrival_tolerance_m"]),
                ROBOTOIQ_2F85_EXACT_CONTACT_ARRIVAL_TOLERANCE_M,
            ) if exact_contact_arrival else float(
                affordance["arrival_tolerance_m"]
            )
            actions.append(
                {
                    "phase_id": phase_id,
                    "mode": "ik_pose",
                    "target_position_world_m": position,
                    "target_quaternion_world_xyzw": orientation,
                    "gripper_state": gripper_state,
                    "minimum_steps": int(
                        affordance[
                            "gripper_dwell_maximum_steps"
                            if dwell
                            else "motion_minimum_steps"
                        ]
                    ),
                    "maximum_steps": derived_maximum,
                    "hold_arm_joint_positions_during_gripper_transition": dwell,
                    "authored_gripper_dwell_minimum_steps": (
                        int(affordance["gripper_dwell_minimum_steps"])
                        if dwell
                        else None
                    ),
                    "construction_phase_id": construction_phase_id,
                    "construction_observed_steps": observed_steps,
                    "target_position_source_phase_id": phase_id,
                    "contact_standoff_m": contact_standoff,
                    "contact_standoff_source": contact_standoff_source,
                    "contact_bite_depth_m": bite_depth,
                    "contact_bite_source": (
                        ROBOTOIQ_2F85_BITE_SOURCE if bite_depth else None
                    ),
                    "bite_direction_source_phase_id": (
                        bite_direction_source_phase_id
                    ),
                    "prealign_retraction_m": prealign_retraction,
                    "prealign_retraction_source": (
                        ROBOTOIQ_2F85_PREALIGN_RETRACTION_SOURCE
                        if prealign_retraction
                        else None
                    ),
                    "prealign_retraction_toward_phase_id": (
                        prealign_retraction_toward_phase_id
                    ),
                    "step_budget_derivation": (
                        "fixed_gripper_dwell_authored"
                        if dwell
                        else "three_x_measured_plus_five_with_25_floor"
                        if observed_steps is not None
                        else "authored_compatibility_fallback"
                    ),
                    "arrival_tolerance_m": arrival_tolerance_m,
                    "arrival_tolerance_source": (
                        ROBOTOIQ_2F85_EXACT_CONTACT_ARRIVAL_TOLERANCE_SOURCE
                        if exact_contact_arrival
                        else "interaction_affordance.arrival_tolerance_m"
                    ),
                    "arrival_orientation_tolerance_rad": (
                        None
                        if phase.get("arrival_orientation_tolerance_rad")
                        is None
                        else float(
                            phase["arrival_orientation_tolerance_rad"]
                        )
                    ),
                    "position_only_arrival": bool(
                        phase.get("position_only_arrival") is True
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
        "positive_trajectory_reexecutes_exact_qualified_phase_targets": False,
        "positive_trajectory_reexecutes_qualified_clearance_targets": False,
        "positive_trajectory_executes_exact_contact_targets_after_clearance_qualification": False,
        "positive_trajectory_executes_bite_adjusted_contact_targets": (
            float(affordance.get("contact_outward_standoff_m", 0.0)) <= 0.0
        ),
        "contact_standoff_source": (
            "native_droid_grasp_swept_volume"
            if float(affordance.get("contact_outward_standoff_m", 0.0)) > 0.0
            else "nvlabs_graspdatagen_robotiq_2f85_bite_depth"
        ),
        "positive_trajectory_budgets_derived_from_measured_construction": True,
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
    "ROBOTOIQ_2F85_EXACT_CONTACT_ARRIVAL_TOLERANCE_M",
    "ROBOTOIQ_2F85_EXACT_CONTACT_ARRIVAL_TOLERANCE_SOURCE",
    "SCHEMA_VERSION",
    "SUPPORTED_TASK_KINDS",
    "materialize_native_graph_articulated_control_plan",
    "materialize_native_rigid_control_plan",
    "materialize_native_task_control_plan",
]
