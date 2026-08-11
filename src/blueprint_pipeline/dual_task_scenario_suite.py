"""Validate one policy-neutral frozen scenario matrix for one task."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_contract import (
    FROZEN_CANDIDATES,
    SUPPORTED_SCENARIO_RUNTIME_TARGETS,
)


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
COUSIN_FAMILY = "admitted_object_cousin"
COUSIN_BLOCKER = "admitted_object_cousin_identity_rights_and_bytes_unresolved"
COUSIN_RUNTIME_TARGET = "AssetResolver.task_subject_asset_id"


class DualTaskScenarioSuiteError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


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
        if family == COUSIN_FAMILY:
            admission = cell.get("execution_admission")
            if not isinstance(admission, Mapping):
                errors.append("dual_task_scenario_cousin_admission_missing")
            elif admission.get("status") == "blocked":
                if (
                    admission.get("blocker_code") != COUSIN_BLOCKER
                    or admission.get("cousin_asset") is not None
                    or cell.get("resolved_parameters") != {}
                    or cell.get("factor_records") != []
                    or cell.get("applied_parameter_readback_required") is not False
                    or cell.get("scheduled_initially") is not False
                    or cell.get("powered_diagnostic") is not False
                ):
                    errors.append("dual_task_scenario_cousin_blocker_invalid")
            elif admission.get("status") == "admitted":
                asset = admission.get("cousin_asset")
                factor_records = cell.get("factor_records")
                factor = (
                    factor_records[0]
                    if isinstance(factor_records, list)
                    and len(factor_records) == 1
                    and isinstance(factor_records[0], Mapping)
                    else {}
                )
                if (
                    not isinstance(asset, Mapping)
                    or not str(asset.get("asset_id") or "")
                    or not _digest(asset.get("sha256"))
                    or isinstance(asset.get("size_bytes"), bool)
                    or not isinstance(asset.get("size_bytes"), int)
                    or asset.get("size_bytes", 0) <= 0
                    or not _digest(asset.get("rights_receipt_digest"))
                    or not _digest(asset.get("resolver_catalog_digest"))
                    or asset.get("runtime_asset_role") != "task_subject_cousin"
                    or not factor
                    or factor.get("runtime_target") != COUSIN_RUNTIME_TARGET
                    or not str(factor.get("parameter_id") or "")
                    or factor.get("unit") != "asset_id"
                    or factor.get("resolved_value") != asset.get("asset_id")
                    or cell.get("resolved_parameters")
                    != {factor.get("parameter_id"): asset.get("asset_id")}
                    or COUSIN_RUNTIME_TARGET not in SUPPORTED_SCENARIO_RUNTIME_TARGETS
                ):
                    errors.append("dual_task_scenario_cousin_asset_invalid")
            else:
                errors.append("dual_task_scenario_cousin_admission_invalid")
        elif "execution_admission" in cell:
            errors.append("dual_task_scenario_admission_unexpected")
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
    "COUSIN_BLOCKER",
    "COUSIN_FAMILY",
    "COUSIN_RUNTIME_TARGET",
    "DualTaskScenarioSuiteError",
    "REQUIRED_FAMILIES",
    "SCHEMA_VERSION",
    "validate_dual_task_scenario_suite",
]
