"""Validate one policy-neutral frozen scenario matrix for one task."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_contract import FROZEN_CANDIDATES


SCHEMA_VERSION = "third_scene_task_scenario_suite.v1"
REQUIRED_FAMILIES = frozenset(
    {
        "canonical",
        "placement_approach",
        "illumination",
        "camera_sensor",
        "bounded_physics",
        "admitted_object_cousin",
        "held_out_composed",
    }
)


class DualTaskScenarioSuiteError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:")


def validate_dual_task_scenario_suite(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("dual_task_scenario_suite_schema_invalid")
    if not str(payload.get("scene_id") or "") or not _digest(
        payload.get("shared_scene_freeze_digest")
    ):
        errors.append("dual_task_scenario_suite_scene_binding_invalid")
    if not str(payload.get("task_id") or "") or not _digest(
        payload.get("task_freeze_digest")
    ):
        errors.append("dual_task_scenario_suite_task_binding_invalid")
    if payload.get("candidate_ids") != list(FROZEN_CANDIDATES):
        errors.append("dual_task_scenario_suite_candidates_invalid")
    if payload.get("required_controls") != [
        "zero_action_negative",
        "scripted_positive",
    ]:
        errors.append("dual_task_scenario_suite_controls_invalid")
    if (
        payload.get("overview_camera_policy_input") is not False
        or payload.get("overview_camera_deterministic_scoring_input") is not False
        or payload.get("learned_policy_outcomes_consulted") is not False
    ):
        errors.append("dual_task_scenario_suite_policy_neutrality_invalid")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("dual_task_scenario_suite_cells_missing")
        cells = []
    ids: list[str] = []
    families: list[str] = []
    diagnostic: list[str] = []
    canonical: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            errors.append(f"dual_task_scenario_cell_invalid:{index}")
            continue
        cell_id = str(cell.get("cell_id") or "")
        family = str(cell.get("family") or "")
        seed = cell.get("seed")
        ids.append(cell_id)
        families.append(family)
        if (
            not cell_id
            or family not in REQUIRED_FAMILIES
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(cell.get("resolved_parameters"), Mapping)
            or not isinstance(cell.get("factor_records"), list)
            or cell.get("candidate_ids") != list(FROZEN_CANDIDATES)
            or cell.get("controls_required_before_candidates") is not True
        ):
            errors.append(f"dual_task_scenario_cell_invalid:{index}")
        if family == "canonical":
            canonical.append(cell_id)
            if cell.get("factor_records") != []:
                errors.append("dual_task_scenario_canonical_factors_invalid")
        if cell.get("powered_diagnostic") is True:
            diagnostic.append(cell_id)
            if family == "canonical" or not cell.get("factor_records"):
                errors.append("dual_task_scenario_diagnostic_invalid")
        if cell.get("scheduled_initially") is True and not (
            family == "canonical" or cell.get("powered_diagnostic") is True
        ):
            errors.append("dual_task_scenario_initial_scope_expanded")
    if len(ids) != len(set(ids)) or "" in ids:
        errors.append("dual_task_scenario_cell_ids_invalid")
    if set(families) != REQUIRED_FAMILIES:
        errors.append("dual_task_scenario_families_incomplete")
    if len(canonical) != 1 or len(diagnostic) != 1:
        errors.append("dual_task_scenario_initial_slice_invalid")
    if payload.get("initial_execution_order") != [*canonical, *diagnostic]:
        errors.append("dual_task_scenario_execution_order_invalid")
    if payload.get("suite_digest") != canonical_digest(
        payload, digest_field="suite_digest"
    ):
        errors.append("dual_task_scenario_suite_digest_invalid")
    if errors:
        raise DualTaskScenarioSuiteError(errors)
    return payload


__all__ = [
    "DualTaskScenarioSuiteError",
    "REQUIRED_FAMILIES",
    "SCHEMA_VERSION",
    "validate_dual_task_scenario_suite",
]
