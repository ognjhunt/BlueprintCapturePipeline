"""Apply exact-workcell cells through the existing native scene/readback adapters.

This port prepares one supported native cell and records its independently
observed parameter application. It does not allocate a provider, query a policy,
change the paired-canary protocol, or qualify a full evaluation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .exact_workcell_variation_matrix import (
    ExactWorkcellVariationError,
    compile_variation_matrix,
)
from .exact_workcell_variation_runtime import compile_isaac_lab_event_plan
from .native_task_runtime_contract import _scenario_parameter_rows


def native_runtime_embodiment_digest(scene_plan: Mapping[str, Any]) -> str:
    """Identity of the native robot, sensor placement and controller cadence."""
    return canonical_digest({name: scene_plan[name] for name in ("robot", "cameras", "cadence")})


def _cell(matrix: Mapping[str, Any], request: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    if dict(matrix) != compile_variation_matrix(request):
        raise ExactWorkcellVariationError(["matrix_does_not_match_request"])
    rows = [row for row in matrix["cells"] if row["cell_id"] == cell_id]
    if len(rows) != 1:
        raise ExactWorkcellVariationError(["runtime_cell_not_in_frozen_matrix"])
    return rows[0]


def native_cell_parameter_bindings(
    *, matrix: Mapping[str, Any], request: Mapping[str, Any], cell_id: str
) -> list[dict[str, Any]]:
    """Translate frozen terms to the native runtime's admitted target contract.

    Values retain their declared native units and absolute nominal values.
    No parameter-name alias, coordinate transform or missing link selector is
    inferred. A task that needs an unsupported target remains resumable input.
    """
    cell = _cell(matrix, request, cell_id)
    rows = [{
        "parameter_id": row["dimension_id"],
        "runtime_target": row["application_target"],
        "unit": row["unit"],
        "nominal_value": row["nominal_value"],
        "resolved_value": row["resolved_value"],
        "application_tolerance": row["application_tolerance"],
    } for row in cell["application_records"]]
    errors: list[str] = []
    dimensions = {row["dimension_id"]: row for row in request["variation_dimensions"]}
    for row in rows:
        if dimensions[row["parameter_id"]]["value_type"] == "categorical":
            errors.append("native_exact_workcell_categorical_target_unavailable:" + row["parameter_id"])
        if row["runtime_target"] == "EventManager.reset.task_subject_link_material.dynamic_friction":
            errors.append("native_exact_workcell_link_selector_required:" + row["parameter_id"])
    if len({row["runtime_target"] for row in rows}) != len(rows):
        errors.append("native_exact_workcell_target_ambiguous")
    bindings = _scenario_parameter_rows(rows, errors=errors)
    if errors:
        raise ExactWorkcellVariationError(errors)
    return bindings


def prepare_native_runtime_cell(
    *, matrix: Mapping[str, Any], request: Mapping[str, Any], cell_id: str,
    source_scene_plan: Mapping[str, Any], source_plan_digest: str,
) -> dict[str, Any]:
    """Use native scene setters before construction, preserving the frozen source.

    The caller supplies the immutable plan digest already admitted by its
    runtime. Scene/task/object identities must also match the matrix. Native
    readback after construction/reset remains mandatory; this result is a plan.
    """
    from .native_task_arena_scene_plan import _apply_scenario_parameters

    cell = _cell(matrix, request, cell_id)
    bindings = native_cell_parameter_bindings(matrix=matrix, request=request, cell_id=cell_id)
    plan = copy.deepcopy(dict(source_scene_plan))
    if (plan.get("schema_version") != "native_task_arena_scene_plan.v1"
            or plan.get("plan_digest") != source_plan_digest
            or source_plan_digest != canonical_digest(plan, digest_field="plan_digest")):
        raise ExactWorkcellVariationError(["native_exact_workcell_source_plan_digest_mismatch"])
    if (plan.get("scene_id") != request["scene_binding"]["scene_id"]
            or plan.get("task_id") != request["task_binding"]["task_id"]):
        raise ExactWorkcellVariationError(["native_exact_workcell_source_identity_mismatch"])
    if (plan.get("robot", {}).get("robot_id") != request["embodiment_binding"]["embodiment_id"]
            or native_runtime_embodiment_digest(plan) != request["embodiment_binding"]["embodiment_digest"]):
        raise ExactWorkcellVariationError(["native_exact_workcell_embodiment_mismatch"])
    if (canonical_digest(plan.get("task_spec") or {}) != request["task_binding"]["task_digest"]
            or (plan.get("task_spec", {}).get("task_success_contract") or {}).get("contract_digest")
            != request["task_binding"]["success_contract_digest"]):
        raise ExactWorkcellVariationError(["native_exact_workcell_task_mismatch"])
    subjects = [row for row in plan.get("objects", []) if row.get("task_subject") is True]
    if (len(subjects) != 1
            or subjects[0].get("asset_id") != request["scene_binding"]["canonical_object_asset_id"]
            or subjects[0].get("sha256") != request["scene_binding"]["canonical_object_asset_digest"]):
        raise ExactWorkcellVariationError(["native_exact_workcell_canonical_object_mismatch"])
    scenario = plan.get("scenario") or {}
    if scenario.get("parameter_applications") or scenario.get("exact_workcell_binding"):
        raise ExactWorkcellVariationError(["native_exact_workcell_source_already_varied"])
    subject = subjects[0]
    before_pose = copy.deepcopy(subject["pose_world"])
    applications = _apply_scenario_parameters(
        objects=plan["objects"], cameras=plan["cameras"], bindings=bindings,
        task_object_asset_path=Path(subject["usd_path"]) if subject.get("usd_path") else None,
    )
    # Independent scoring must use the varied reset, never the nominal start.
    if plan.get("task_kind") == "rigid_pick_place" and "start_pose_world" in plan.get("task_spec", {}):
        before_start = [*before_pose["position_world_m"], *before_pose["orientation_xyzw"]]
        if plan["task_spec"]["start_pose_world"] != before_start:
            raise ExactWorkcellVariationError(["native_exact_workcell_scoring_start_mismatch"])
        plan["task_spec"]["start_pose_world"] = [
            *subject["pose_world"]["position_world_m"], *subject["pose_world"]["orientation_xyzw"],
        ]
    plan["scenario"] = {
        **scenario, "context_kind": "evaluation_cell", "cell_id": cell_id,
        "instance_digest": cell["cell_digest"], "seed": cell["seed"],
        "parameter_bindings": bindings, "parameter_applications": applications,
        "exact_workcell_binding": {
            "request_digest": request["request_digest"], "matrix_digest": matrix["matrix_digest"],
            "cell_digest": cell["cell_digest"], "reset_digest": cell["reset_digest"],
            "source_plan_digest": source_plan_digest,
        },
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def read_native_runtime_cell(
    *, built: Any, matrix: Mapping[str, Any], request: Mapping[str, Any], cell_id: str,
    prepared_plan_digest: str,
) -> dict[str, Any]:
    """Read native values after reset into the existing exact-cell receipt.

    The GPU worker must bind ``prepared_plan_digest`` before construction. This
    function never trusts an apply callback's return value or the requested
    parameter value as the observed value.
    """
    from .native_task_arena_readback import read_native_task_arena_scenario_parameters
    from .native_task_arena_scene_plan import _yaw_degrees_xyzw

    cell = _cell(matrix, request, cell_id)
    bindings = native_cell_parameter_bindings(matrix=matrix, request=request, cell_id=cell_id)
    plan = built.plan
    scenario = plan.get("scenario") or {}
    identity = scenario.get("exact_workcell_binding") or {}
    if (plan.get("plan_digest") != prepared_plan_digest
            or prepared_plan_digest != canonical_digest(plan, digest_field="plan_digest")
            or scenario.get("cell_id") != cell_id or scenario.get("seed") != cell["seed"]
            or scenario.get("instance_digest") != cell["cell_digest"]
            or identity.get("request_digest") != request["request_digest"]
            or identity.get("matrix_digest") != matrix["matrix_digest"]
            or identity.get("cell_digest") != cell["cell_digest"]
            or identity.get("reset_digest") != cell["reset_digest"]
            or scenario.get("parameter_bindings") != bindings):
        raise ExactWorkcellVariationError(["native_exact_workcell_runtime_binding_mismatch"])
    native = read_native_task_arena_scenario_parameters(built)
    measured = {row["parameter_id"]: row for row in native["parameters"]}
    if len(measured) != len(bindings) or set(measured) != {row["parameter_id"] for row in bindings}:
        raise ExactWorkcellVariationError(["native_exact_workcell_readback_terms_mismatch"])
    event = next(row for row in compile_isaac_lab_event_plan(matrix, request=request)["cells"] if row["cell_id"] == cell_id)
    terms = {row["manager_target"]: row for row in event["terms"]}
    observations, blockers = [], []
    for binding in bindings:
        row = measured[binding["parameter_id"]]
        term = terms[binding["runtime_target"]]
        observed = row["observed_native_value"]
        if binding["runtime_target"] == "EventManager.reset.object_orientation.yaw":
            yaw = _yaw_degrees_xyzw(observed)
            # A quaternion's yaw wraps at 180 degrees; keep the same equivalent
            # angle nearest the frozen scalar without changing its tolerance.
            observed = binding["resolved_value"] + ((yaw - binding["resolved_value"] + 180.0) % 360.0 - 180.0)
        passed = (row.get("passed") is True and isinstance(observed, (int, float))
                  and not isinstance(observed, bool) and math.isfinite(observed)
                  and abs(observed - binding["resolved_value"]) <= binding["application_tolerance"])
        observations.append({"term_digest": term["term_digest"], "observed": {
            "unit": binding["unit"], "value": observed, "source": "native_readback",
        }, "passed": passed})
        if not passed:
            blockers.append("runtime_cell_native_value_mismatch:" + term["term_id"])
    result = {
        "schema_version": "exact_workcell_reset_application.v1", "cell_id": cell_id,
        "cell_digest": cell["cell_digest"], "reset_digest": cell["reset_digest"], "seed": cell["seed"],
        "cell_plan_digest": event["cell_plan_digest"], "observations": observations,
        "status": "blocked" if blockers else "applied_and_readback_verified", "blockers": blockers,
        "policy_query_performed": False, "scope": "parameter_application_not_full_reset_or_qualification",
        "prepared_plan_digest": prepared_plan_digest, "native_readback": native,
    }
    result["application_digest"] = canonical_digest(result, digest_field="application_digest")
    return result
