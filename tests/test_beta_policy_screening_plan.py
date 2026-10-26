"""Matched diagnostic cells, independent admission and one aggregate site cap."""

from __future__ import annotations

import datetime as dt
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blueprint_pipeline.beta_policy_candidate_catalog import catalog_candidate, load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import (
    EXECUTION_SCHEMA_VERSION, QUOTE_SCHEMA_VERSION, BetaPolicyScreeningPlanError,
    build_screening_plan, screening_cells, validate_cost_quote, validate_execution_descriptor, validate_screening_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_evaluation_plan import scored_cells

NOW = dt.datetime(2026, 10, 26, tzinfo=dt.timezone.utc)
SHA = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def scene():
    return {"site_id": "site-request1", "scene_id": "site-request1", "capture_id": "capture1", "task_id": "task1",
            "scene_intent_id": "intent1", "scene_intent_digest": SHA, "scene_revision_digest": SHA,
            "task_context_digest": OTHER, "task_success_contract_digest": SHA,
            "robot_free_probe_digest": SHA, "robot_free_probe_status": "passed", "task_kind": "rigid_pick_place",
            "scene_inputs": {"path": "/operator/scene.json", "sha256": SHA, "size_bytes": 128},
            "task_spec": {"task_kind": "rigid_pick_place", "start_pose_world": [0.5, 0.0, 0.8, 0, 0, 0, 1],
                          "destination_position_bounds_world_m": {"minimum": [0.5, 0.25, 0.79], "maximum": [0.6, 0.35, 0.81]},
                          "destination_orientation_tolerance_rad": 0.5}}


def admitted(candidate, *, maximum="2.00"):
    spec = {"schema_version": EXECUTION_SCHEMA_VERSION, "candidate_id": candidate["candidate_id"],
            "configuration_digest": candidate["configuration_digest"], "runtime_digest": SHA,
            "backend": "robolab.v1", "policy_execution_spec": {"path": "/operator/policy.json", "sha256": SHA, "size_bytes": 128},
            "checkpoint_inventory": {"path": "/operator/checkpoint.json", "sha256": SHA, "size_bytes": 256},
            "max_hourly_rate_usd": "1.00", "hard_ttl_seconds": 600, "implementation_commit": "a" * 40,
            "task_success_contract_digest": SHA}
    binding = {"configuration_digest": candidate["configuration_digest"], "runtime_digest": SHA,
               "backend": "robolab.v1", "admission_digest": canonical_digest(spec), "execution_spec": spec}
    quote = {"schema_version": QUOTE_SCHEMA_VERSION, "quote_id": "quote-" + candidate["candidate_id"],
             "price_source": "hermetic price fixture; no allocation", "valid_until": "2026-10-27T00:00:00Z",
             "configuration_digest": candidate["configuration_digest"], "scene_revision_digest": SHA,
             "episode_count": 10, "maximum_cost_usd": maximum}
    quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
    return binding, quote


def inputs(count=2, maximum="2.00"):
    catalog = load_default_catalog()
    if count != 2:
        template = catalog_candidate(catalog, "pi05_droid")
        catalog = seal_catalog({**catalog, "candidates": [{**deepcopy(template), "candidate_id": f"policy_{n:02d}"} for n in range(count)]})
    candidates = [row for row in catalog["candidates"] if row["eligibility"]["enabled"]]
    bindings, quotes = {}, {}
    for candidate in candidates:
        bindings[candidate["candidate_id"]], quotes[candidate["candidate_id"]] = admitted(candidate, maximum=maximum)
    return catalog, {"candidate_bindings": bindings}, quotes


def plan(count=2, maximum="2.00", **kwargs):
    catalog, capabilities, quotes = inputs(count, maximum)
    return build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                cost_quotes=quotes, now=NOW, **kwargs)


def reseal(value):
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def test_forty_candidates_share_ten_cells_and_remain_diagnostic():
    value = plan(40, "0.20")
    assert len(value["candidates"]) == 40
    assert {row["status"] for row in value["candidates"]} == {"selected"}
    assert value["maximum_planned_cost_usd"] == "8.00"
    assert all(row["scenario_cells"] == value["scenario_cells"] for row in value["candidates"])
    assert value["recommendation_eligible"] is False and value["purpose"] == "screening"
    assert validate_screening_plan(value) == value


def test_machine_readable_plan_schema_accepts_mixed_admission_outcomes():
    path = Path(__file__).resolve().parents[1] / "docs/schemas/beta_policy_screening_plan.v1.schema.json"
    schema = json.loads(path.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(plan())
    validator.validate(plan(40))
    validator.validate(plan(maximum="0.01"))


def test_aggregate_budget_defers_candidates_instead_of_multiplying_allowance():
    value = plan(40, "2.00")
    assert sum(row["status"] == "selected" for row in value["candidates"]) == 7
    assert value["maximum_planned_cost_usd"] == "14.00"
    assert all(row["blockers"] == ["site_screening_budget_insufficient"] for row in value["candidates"][7:])
    with pytest.raises(BetaPolicyScreeningPlanError, match="site_budget_exceeded"):
        plan(budget_usd="15.01")


def test_full_length_native_candidate_id_can_have_a_bound_quote_identifier():
    catalog = load_default_catalog()
    candidate = {**catalog_candidate(catalog, "pi05_droid"), "candidate_id": "p" + "a" * 127}
    catalog = seal_catalog({**catalog, "candidates": [candidate]})
    candidate = catalog["candidates"][0]
    binding, quote = admitted(candidate)
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1",
        capabilities={"candidate_bindings": {candidate["candidate_id"]: binding}},
        cost_quotes={candidate["candidate_id"]: quote}, now=NOW)
    assert value["candidates"][0]["status"] == "selected"


def test_catalog_alone_and_label_changes_never_admit_execution():
    value = build_screening_plan(load_default_catalog(), scene(), campaign_id="campaign1", now=NOW)
    assert all(row["status"] == "unavailable" for row in value["candidates"])
    assert all("runtime_candidate_binding_missing" in row["blockers"] for row in value["candidates"])
    assert value["maximum_planned_cost_usd"] == "0.00"


def test_unknown_candidates_and_bad_selection_are_rejected():
    with pytest.raises(BetaPolicyScreeningPlanError, match="candidate_not_in_catalog"):
        plan(candidate_ids=["unknown-policy", "pi05_droid"])
    for selection in (["pi05_droid", "pi05_droid"], "pi05_droid", [None]):
        with pytest.raises(BetaPolicyScreeningPlanError, match="candidate_selection_invalid"):
            plan(candidate_ids=selection)


def test_cost_unknown_stale_or_underpriced_is_deferred_without_losing_others():
    catalog, capabilities, quotes = inputs()
    del quotes["pi05_droid"]
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities, cost_quotes=quotes, now=NOW)
    assert {row["candidate_id"] for row in value["candidates"] if row["status"] == "selected"} == {"groot_n17_droid"}
    quotes["groot_n17_droid"]["valid_until"] = "2026-10-25T00:00:00Z"
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities, cost_quotes=quotes, now=NOW)
    assert not any(row["status"] == "selected" for row in value["candidates"])
    assert any("cost_quote_expired" in row["blockers"][0] for row in value["candidates"] if row["candidate_id"] == "groot_n17_droid")
    low = plan(maximum="0.01")
    assert any(row["blockers"] == ["cost_quote_below_resource_bound"] for row in low["candidates"])


def test_historical_plans_survive_expiry_but_launch_quote_check_does_not():
    value = plan()
    row = next(row for row in value["candidates"] if row["status"] == "selected")
    assert validate_screening_plan(value)
    with pytest.raises(BetaPolicyScreeningPlanError, match="expired"):
        validate_cost_quote(row["cost_quote"], configuration_digest=row["configuration_digest"],
                            scene_revision_digest=SHA, episode_count=10, now=NOW + dt.timedelta(days=2))


def test_hosted_policy_reserves_gpu_plus_per_request_api_cost_and_keeps_secret_out_of_catalog():
    catalog, capabilities, quotes = inputs()
    binding = capabilities["candidate_bindings"]["pi05_droid"]
    spec = binding["execution_spec"]
    spec.update(credential_file="/private/operator.env", credential_origin="https://policy.example.com", provider_hard_cap_usd="1.00",
                hosted_inference_payment="blueprint", hosted_inference_maximum_usd="2.00",
                hosted_max_cost_per_request_usd="0.001", hosted_http_request_limit=2000)
    binding["admission_digest"] = canonical_digest(spec)
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                cost_quotes=quotes, now=NOW)
    row = next(row for row in value["candidates"] if row["candidate_id"] == "pi05_droid")
    assert row["status"] == "deferred" and row["blockers"] == ["cost_quote_below_resource_bound"]
    quotes["pi05_droid"]["maximum_cost_usd"] = "3.00"
    quotes["pi05_droid"]["quote_digest"] = canonical_digest(quotes["pi05_droid"], digest_field="quote_digest")
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                cost_quotes=quotes, now=NOW)
    assert value["maximum_planned_cost_usd"] == "5.00"
    assert "/private/operator.env" not in str(catalog)


@pytest.mark.parametrize("payer", ["caller_paid", "free"])
def test_zero_api_reservation_requires_explicit_payer(payer):
    candidate = catalog_candidate(load_default_catalog(), "pi05_droid")
    binding, _ = admitted(candidate)
    spec = binding["execution_spec"]
    spec.update(credential_file="/private/operator.env", credential_origin="https://policy.example.com", provider_hard_cap_usd="1.00",
                hosted_inference_payment=payer, hosted_inference_maximum_usd="0.00",
                hosted_max_cost_per_request_usd="0", hosted_http_request_limit=2000)
    assert validate_execution_descriptor(spec, candidate_id=candidate["candidate_id"],
                                         configuration_digest=candidate["configuration_digest"]) == spec


@pytest.mark.parametrize("mutation", ["missing_price", "free_with_cost", "zero_paid_price", "underpriced_api", "underpriced_gpu", "request_bool", "origin_credentials", "origin_path"])
def test_hosted_cost_binding_cannot_hide_payment_or_request_cost(mutation):
    candidate = catalog_candidate(load_default_catalog(), "pi05_droid")
    binding, _ = admitted(candidate)
    spec = binding["execution_spec"]
    spec.update(credential_file="/private/operator.env", credential_origin="https://policy.example.com", provider_hard_cap_usd="1.00",
                hosted_inference_payment="blueprint", hosted_inference_maximum_usd="2.00",
                hosted_max_cost_per_request_usd="0.001", hosted_http_request_limit=2000)
    if mutation == "missing_price":
        del spec["hosted_max_cost_per_request_usd"]
    elif mutation == "free_with_cost":
        spec["hosted_inference_payment"] = "free"
    elif mutation == "zero_paid_price":
        spec["hosted_max_cost_per_request_usd"] = "0"
    elif mutation == "underpriced_api":
        spec["hosted_inference_maximum_usd"] = "1.99"
    elif mutation == "underpriced_gpu":
        spec["provider_hard_cap_usd"] = "0.01"
    elif mutation == "request_bool":
        spec["hosted_http_request_limit"] = True
    elif mutation == "origin_credentials":
        spec["credential_origin"] = "https://credential@policy.example.com"
    elif mutation == "origin_path":
        spec["credential_origin"] = "https://policy.example.com/private-path"
    with pytest.raises(BetaPolicyScreeningPlanError, match="hosted"):
        validate_execution_descriptor(spec, candidate_id=candidate["candidate_id"],
                                      configuration_digest=candidate["configuration_digest"])


def test_screening_draws_are_deterministic_and_do_not_reuse_finalist_ids_or_seeds():
    catalog, capabilities, quotes = inputs(40, "0.20")
    first = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities, cost_quotes=quotes, now=NOW)
    catalog["candidates"].reverse()
    catalog = seal_catalog(catalog)
    second = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities, cost_quotes=quotes, now=NOW)
    assert first["scenario_cells"] == second["scenario_cells"]
    finalist = scored_cells(scene_revision_digest=SHA, scene_id="site-request1", salt="private-finalist-salt" * 3,
                            task_spec=scene()["task_spec"], task_kind="rigid_pick_place")
    assert not {cell["seed"] for cell in first["scenario_cells"]} & {cell["seed"] for cell in finalist}
    assert {cell["partition"] for cell in first["scenario_cells"]} == {"screening"}
    with pytest.raises(BetaPolicyScreeningPlanError, match="episode_count_invalid"):
        screening_cells(scene(), 11)


@pytest.mark.parametrize("mutation", ["purpose", "source", "cost", "candidate_cells", "command", "task", "duplicate", "credential_file"])
def test_resealing_cannot_hide_contradictory_plan_bindings(mutation):
    value = plan()
    row = next(row for row in value["candidates"] if row["status"] == "selected")
    if mutation == "purpose":
        value["purpose"] = "finalist"
    elif mutation == "source":
        value["source"]["capture_id"] = "different"
    elif mutation == "cost":
        row["max_cost_usd"] = "0.01"
    elif mutation == "candidate_cells":
        row["scenario_cells"][0]["seed"] += 1
    elif mutation == "command":
        row["execution_spec"]["command"] = "arbitrary"
        row["admission_digest"] = canonical_digest(row["execution_spec"])
    elif mutation == "task":
        row["execution_spec"]["task_success_contract_digest"] = OTHER
        row["admission_digest"] = canonical_digest(row["execution_spec"])
    elif mutation == "duplicate":
        value["candidates"].append(deepcopy(row))
    elif mutation == "credential_file":
        row["execution_spec"]["credential_file"] = "customer-relative-secret"
        row["admission_digest"] = canonical_digest(row["execution_spec"])
    with pytest.raises(BetaPolicyScreeningPlanError):
        validate_screening_plan(reseal(value))


def test_failed_robot_free_probe_blocks_planning_and_keeps_site_budget_identity():
    bad = scene()
    bad["robot_free_probe_status"] = "failed"
    with pytest.raises(BetaPolicyScreeningPlanError, match="robot_free_probe_not_passed"):
        build_screening_plan(load_default_catalog(), bad, campaign_id="campaign1", now=NOW)
    value = plan()
    assert value["site_id"] == "site-request1"
    assert value["source"]["capture_id"] == "capture1"
