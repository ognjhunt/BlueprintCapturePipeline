"""Family plans retain pending native qualification and honest scenario scope."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from blueprint_pipeline.beta_policy_screening_plan import (
    build_screening_plan, validate_screening_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_family import family_planning_fixture


def _plan(tmp_path, monkeypatch):
    planning, catalog, *_ = family_planning_fixture(tmp_path, monkeypatch)
    return build_screening_plan(catalog, planning["scene"], campaign_id="family-plan",
        capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"]), planning, catalog


def test_family_plan_schema_describes_pending_native_controls_and_matched_reset_seeds(tmp_path, monkeypatch):
    plan, planning, _catalog = _plan(tmp_path, monkeypatch)
    schema = Path(__file__).resolve().parents[1] / "docs/schemas/beta_policy_screening_plan.v1.schema.json"
    Draft202012Validator(json.loads(schema.read_text())).validate(plan)
    assert "robot_free_probe_digest" not in plan
    assert plan["pre_policy_site_qualification_required"] is True
    assert plan["scenario_scope"] == "native_frozen_reset_seeds"
    assert len({cell["seed"] for cell in plan["scenario_cells"]}) == 10
    assert all(cell["family"] == "canonical_anchor" and not cell["resolved_scenario"]["parameters"]
               for cell in plan["scenario_cells"])
    assert plan["family_static_readiness_digest"] == planning["scene"]["family_static_readiness_digest"]


@pytest.mark.parametrize("field,value", [
    ("pre_policy_site_qualification_required", False),
    ("scenario_scope", "placement_robustness"),
    ("robot_free_probe_digest", "sha256:" + "a" * 64),
])
def test_resealing_cannot_claim_unmeasured_native_qualification(tmp_path, monkeypatch, field, value):
    plan, _planning, _catalog = _plan(tmp_path, monkeypatch)
    plan[field] = value
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    with pytest.raises(ValueError):
        validate_screening_plan(plan)


def test_resealing_cannot_claim_unimplemented_family_perturbations(tmp_path, monkeypatch):
    plan, _planning, _catalog = _plan(tmp_path, monkeypatch)
    cell = plan["scenario_cells"][0]
    cell["resolved_scenario"]["parameters"] = {"object_start_x_delta_m": .1}
    cell["cell_spec_digest"] = cell["resolved_scenario_digest"] = canonical_digest(cell["resolved_scenario"])
    plan["scenario_plan_digest"] = canonical_digest({"cells": plan["scenario_cells"]})
    for candidate in plan["candidates"]:
        if candidate["status"] == "selected":
            candidate["scenario_cells"] = deepcopy(plan["scenario_cells"])
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    with pytest.raises(ValueError, match="family_scenario_not_qualified"):
        validate_screening_plan(plan)


def test_family_plan_rejects_a_different_captured_task_context(tmp_path, monkeypatch):
    _plan_value, planning, catalog = _plan(tmp_path, monkeypatch)
    planning["scene"]["task_context_digest"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="native_family_scene_plan_invalid"):
        build_screening_plan(catalog, planning["scene"], campaign_id="changed-context",
            capabilities=planning["capabilities"], cost_quotes=planning["cost_quotes"])


def test_family_task_cannot_enter_the_frozen_internal_adp_pair(tmp_path):
    from blueprint_pipeline.native_task_arena_policy_canary_session import validate_runtime_input_manifest
    from tests.test_native_task_arena_policy_canary_session import _activation, _runtime_inputs
    from tests.test_task_family_policy_execution import _scene

    inputs = _runtime_inputs(tmp_path, _activation())
    task = _scene()["task_spec"]["task_success_contract"]
    inputs.update(task_success_contract=task, task_success_contract_digest=task["contract_digest"])
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    with pytest.raises(ValueError, match="family_requires_beta_admission"):
        validate_runtime_input_manifest(inputs)
