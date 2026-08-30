"""Reprove sealed native construction evidence before a paid successor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_camera_observability import (
    NativeTaskCameraObservabilityError,
    validate_native_task_policy_start_camera_observability,
)
from .native_task_construction_plan import (
    NativeTaskConstructionPlanError,
    evaluate_rigid_construction_gates,
    materialize_native_task_construction_phase_plan,
)


class NativeTaskConstructionResultError(ValueError):
    """Stable failures for a stored construction result and its raw readbacks."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def validate_qualified_rigid_construction_result(
    *, scene_plan: Mapping[str, Any], construction_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute every CPU-verifiable rigid construction claim.

    A ``result_digest`` protects bytes from accidental drift; it does not make a
    stored ``passed`` boolean measurement authority.  Controls admission can
    reuse a construction only after replaying the phase-plan compiler, raw
    readback gate evaluator, step budget, and exact policy-start camera evidence.
    """

    try:
        scene = json.loads(json.dumps(dict(scene_plan), allow_nan=False))
        construction = json.loads(
            json.dumps(dict(construction_result), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise NativeTaskConstructionResultError(
            ["native_rigid_construction_result_invalid"]
        ) from exc

    errors: list[str] = []
    scene_digest = scene.get("plan_digest")
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene.get("task_kind") != "rigid_pick_place"
        or scene_digest != canonical_digest(scene, digest_field="plan_digest")
    ):
        errors.append("native_rigid_construction_scene_plan_invalid")
    if (
        construction.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("blockers") != []
        or construction.get("candidate_policy_queried") is not False
        or construction.get("scene_plan_digest") != scene_digest
        or construction.get("result_digest")
        != canonical_digest(construction, digest_field="result_digest")
    ):
        errors.append("native_rigid_construction_result_invalid")

    phase_plan = construction.get("construction_phase_plan")
    try:
        recomputed_phase_plan = materialize_native_task_construction_phase_plan(scene)
    except NativeTaskConstructionPlanError as exc:
        errors.extend(
            f"native_rigid_construction_phase_plan_recompile_failed:{error}"
            for error in exc.errors
        )
        recomputed_phase_plan = None
    if (
        not isinstance(phase_plan, Mapping)
        or recomputed_phase_plan is None
        or dict(phase_plan) != recomputed_phase_plan
    ):
        errors.append("native_rigid_construction_phase_plan_recompile_mismatch")

    phase_results = construction.get("phase_results")
    if not isinstance(phase_results, list) or not phase_results:
        errors.append("native_rigid_construction_phase_results_invalid")
        phase_results = []
    step_sum = 0
    for row in phase_results:
        if (
            not isinstance(row, Mapping)
            or isinstance(row.get("steps"), bool)
            or not isinstance(row.get("steps"), int)
            or int(row["steps"]) <= 0
            or row.get("target_reached") is not True
        ):
            errors.append("native_rigid_construction_phase_results_invalid")
            continue
        step_sum += int(row["steps"])
    if isinstance(phase_plan, Mapping):
        execution = phase_plan.get("execution_parameters")
        maximum = (
            execution.get("maximum_construction_total_steps")
            if isinstance(execution, Mapping)
            else None
        )
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
            or construction.get("total_action_steps") != step_sum
            or step_sum > maximum
        ):
            errors.append("native_rigid_construction_total_action_steps_invalid")

    reset_replay = construction.get("reset_replay")
    stored_gates = construction.get("rigid_construction_gates")
    recomputed_gates: dict[str, Any] | None = None
    if isinstance(phase_plan, Mapping) and phase_results:
        try:
            recomputed_gates = evaluate_rigid_construction_gates(
                phase_plan=phase_plan,
                phase_results=phase_results,
                reset_replay=(
                    reset_replay if isinstance(reset_replay, Mapping) else {}
                ),
            )
        except NativeTaskConstructionPlanError as exc:
            errors.extend(
                f"native_rigid_construction_gate_replay_failed:{error}"
                for error in exc.errors
            )
    if (
        recomputed_gates is None
        or not isinstance(stored_gates, Mapping)
        or dict(stored_gates) != recomputed_gates
        or recomputed_gates.get("passed") is not True
        or recomputed_gates.get("blockers") != []
    ):
        errors.append("native_rigid_construction_gate_replay_mismatch")

    camera_gates = construction.get("camera_gates")
    if (
        not isinstance(camera_gates, Mapping)
        or set(camera_gates) != {"external", "wrist", "overview"}
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in camera_gates.values()
        )
    ):
        errors.append("native_rigid_construction_camera_gates_invalid")
    camera_summary: dict[str, Any] | None = None
    try:
        camera_summary = validate_native_task_policy_start_camera_observability(
            construction
        )
    except NativeTaskCameraObservabilityError as exc:
        errors.extend(exc.errors)

    if not isinstance(reset_replay, Mapping) or reset_replay.get("passed") is not True:
        errors.append("native_rigid_construction_reset_replay_invalid")
    if errors:
        raise NativeTaskConstructionResultError(errors)
    return {
        "scene_plan_digest": scene_digest,
        "construction_result_digest": construction["result_digest"],
        "construction_phase_plan_digest": phase_plan["plan_digest"],
        "gate_evaluation_digest": recomputed_gates["evaluation_digest"],
        "total_action_steps": step_sum,
        "policy_start_camera_observability": camera_summary,
    }


__all__ = [
    "NativeTaskConstructionResultError",
    "validate_qualified_rigid_construction_result",
]
