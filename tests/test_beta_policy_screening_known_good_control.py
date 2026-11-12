"""The scripted reference check runs first on every campaign; a result must be about the robot, not our set-up.

Owner, 2026-11-12: the reference row is planned first on every campaign, a re-plan too, funded before the AI
robot rows (not on leftover money), with a body Blueprint chooses from the registered scripted bodies of the
tested robots' arm family (never the site's fit table). It passes at 2 of 3, is ``partly`` at 1 of 3, and its
count is recorded. The AI robot rows run whatever its outcome; what the site is told changes with it. Its
$3.00 cap, 3 tries and time limits are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_known_good_control as control
from blueprint_pipeline.beta_policy_candidate_catalog import seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import (
    BetaPolicyScreeningPlanError, build_job_ceiling_screening_plan, job_ceiling_record, validate_screening_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import Backend, plan as queue_plan
from tests.test_beta_policy_screening_job_ceiling import READ_AT, ceiling_summary
from tests.test_beta_policy_screening_plan import NOW, admitted, candidate_rows, inputs, scene

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = ROOT / "docs/schemas/beta_policy_screening_plan.v1.schema.json"
PUBLICATION_SCHEMA = ROOT / "docs/schemas/beta_policy_screening_publication.v3.schema.json"
CONTROL = "blueprint_scripted_test_arm"
UR5E = "universal_robots_ur5e_robotiq_2f85_sim"
FRANKA_BODY = "franka_research_3_robotiq_2f85_sim"


def catalog_with_scripted(count=3, *, robot=None, maximum="2.00", scripted_maximum="1.50"):
    """``count`` learned (DROID Franka) rows and one scripted pick/place profile for the same robot (or ``robot``)."""
    catalog, _, _ = inputs(count, maximum)
    rows = deepcopy(catalog["candidates"])
    script = deepcopy(rows[0])
    script.update(candidate_id=CONTROL, display_name="Blueprint scripted pick/place - test arm")
    script["policy"]["kind"] = "scripted"
    script["interface"]["policy_interface_id"] = control.SCRIPTED_INTERFACE
    if robot is not None:
        script["interface"]["robot_profile_id"] = robot
    return _bound(seal_catalog({**catalog, "candidates": [*rows, script]}), maximum, scripted_maximum)


def catalog_with_bodies(bodies, count=3, *, maximum="2.00", scripted_maximum="1.50"):
    """``count`` learned (DROID Franka) rows and one registered scripted row per body; none for the DROID arm."""
    catalog, _, _ = inputs(count, maximum)
    rows = deepcopy(catalog["candidates"])
    for body in bodies:
        script = deepcopy(rows[0])
        script.update(candidate_id="blueprint_scripted_" + body, display_name="Blueprint scripted pick/place - " + body)
        script["policy"]["kind"] = "scripted"
        script["interface"].update(policy_interface_id=control.SCRIPTED_INTERFACE, robot_profile_id=body)
        rows.append(script)
    return _bound(seal_catalog({**catalog, "candidates": rows}), maximum, scripted_maximum)


def _bound(catalog, maximum, scripted_maximum):
    bindings, quotes = {}, {}
    for row in catalog["candidates"]:
        price = scripted_maximum if row["policy"]["kind"] == "scripted" else maximum
        bindings[row["candidate_id"]], quotes[row["candidate_id"]] = admitted(row, maximum=price)
    return catalog, {"candidate_bindings": bindings}, quotes


def rerated(capabilities, candidate_id, *, rate, ttl):
    """The binding of ``candidate_id`` at another hourly rate and hard TTL, re-sealed as planning seals it."""
    capabilities = deepcopy(capabilities)
    binding = capabilities["candidate_bindings"][candidate_id]
    binding["execution_spec"].update(max_hourly_rate_usd=rate, hard_ttl_seconds=ttl)
    binding["admission_digest"] = canonical_digest(binding["execution_spec"])
    return capabilities


def website_plan(catalog, capabilities, quotes, *, remaining=40, value=None, **options):
    record = job_ceiling_record(ceiling_summary(remaining), local_holds_usd=Decimal(0), read_at=READ_AT)
    return build_job_ceiling_screening_plan(catalog, value or scene(), job_ceiling=record, campaign_id="campaign1",
                                            capabilities=capabilities, cost_quotes=quotes, now=NOW, **options)


def _valid(value):
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)
    return value


# ---- the plan: first, funded first, every campaign ----------------------------------------------------------

def test_a_new_campaign_plans_the_reference_row_first():
    catalog, capabilities, quotes = catalog_with_scripted()
    value = _valid(website_plan(catalog, capabilities, quotes))
    row, *candidates = value["candidates"]
    assert row["candidate_id"] == CONTROL and row["status"] == "selected" and control.first_in_plan(row)
    assert [item["status"] for item in candidates] == ["selected"] * 3 and not any(map(control.is_control, candidates))
    assert row["known_good_control"] == {"schema_version": control.SCHEMA, "role": "reference_only", "episodes": 3,
                                         "pass_at_successes": 2, "partly_at_successes": 1,
                                         "for_candidate_id": candidates[0]["candidate_id"], "chosen_by": "candidate_robot",
                                         "flags": [control.PACKET_STATION_FLAG]}
    # Three nominal (canonical anchor) tries under the scripted profile's own hard cap.
    assert row["episode_count"] == 3 and {cell["family"] for cell in row["scenario_cells"]} == {"canonical_anchor"}
    assert row["max_cost_usd"] == row["cost_quote"]["maximum_cost_usd"] == "1.50"
    assert value["maximum_planned_cost_usd"] == "7.50"


def test_the_reference_row_never_costs_the_job_a_robot():
    """Review 36: funded from what the AI robot rows leave, never ahead of them (at $4 to $20 left)."""
    catalog, capabilities, quotes = catalog_with_scripted(7, maximum="4.00", scripted_maximum="4.00")
    capabilities = rerated(capabilities, CONTROL, rate="0.80", ttl=14_400)
    learned_only = seal_catalog({**catalog, "candidates": [row for row in catalog["candidates"]
                                                           if row["policy"]["kind"] != "scripted"]})
    for remaining in range(4, 21):
        value = _valid(website_plan(catalog, capabilities, quotes, remaining=remaining))
        without = website_plan(learned_only, capabilities, quotes, remaining=remaining)
        row, *learned = value["candidates"]
        assert [item["status"] for item in learned] == [item["status"] for item in candidate_rows(without)], remaining
        left = Decimal(value["budget_usd"]) - sum(Decimal(item["max_cost_usd"]) for item in learned
                                                   if item["status"] == "selected")
        # Selected only when what is left covers its $3.00; else waitlisted behind every AI robot row.
        assert row["status"] == ("selected" if left >= Decimal("3.00") else "waitlisted"), remaining
    # With no AI robot funded it is never paid for.
    assert website_plan(catalog, capabilities, quotes, remaining=3)["candidates"][0]["known_good_control"][
        "unavailable"] == "budget"


def test_a_waitlisted_reference_row_is_funded_only_after_every_ai_row_and_launches_once_funded(tmp_path, summaries):
    # $15: two $4 robots selected, a third waitlisted, the reference waitlisted behind it.
    value = deepcopy(with_control(queue_plan(count=3, cost="4.00")))
    reference, *learned = value["candidates"]
    reference.update(status="waitlisted", blockers=["site_screening_budget_insufficient"])
    learned[2].update(status="waitlisted", blockers=["site_screening_budget_insufficient"])
    value["maximum_planned_cost_usd"] = "8.00"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    coordinator.enqueue_screening_campaign(validate_screening_plan(value), queue_root=tmp_path)
    coordinator._refill_waitlist(tmp_path)
    rows = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]
    # The waiting robot is funded first; the reference still waits behind it in the same pass.
    assert rows["candidate-002"]["status"] == "pending" and rows[CONTROL]["status"] == "waitlisted"
    backend = PerRow(done("candidate-000", "candidate-001", "candidate-002",
                          control_result=closed(_summary(3, 2, scripted=True, planned=3))))
    run(tmp_path, backend, passes=10)
    assert set(backend.order) == {"candidate-000", "candidate-001", "candidate-002", CONTROL}
    assert backend.order[0] != CONTROL  # never ahead of a robot: funded only once one had its money


def test_a_reference_body_that_cannot_reach_is_not_a_reference():
    """Review 36 (2): it would score 0 of 3 and say the simulation may be at fault."""
    catalog, capabilities, quotes = catalog_with_scripted()
    from blueprint_pipeline.website_candidate_feasibility import reach_rank

    rank = reach_rank({"status": "infeasible", "blockers": ["robot_reach_insufficient:place:1.086m_vs_0.990m"]})
    value = website_plan(catalog, {**capabilities, "candidate_reach_rank": {CONTROL: rank}}, quotes)
    row = value["candidates"][0]
    assert row["status"] == "unavailable" and row["known_good_control"]["unavailable"] == "reach"
    assert control.control_flags(row, {"status": "unavailable"}) == {
        "setup_quality_flags": ["known_good_control_unavailable:reach"]}


def _reference_ledger(plan, *, earlier, campaign_two=None):
    """``campaign-0`` holds an earlier attempt of the same check; ``campaign-1`` the plan's pending row."""
    return {"campaigns": {
        "campaign-0": {"site_id": plan["site_id"], "plan_digest": "x",
                       "candidates": {CONTROL: {"known_good_control": True, "known_good_control_key": control.reuse_key(plan),
                                                **earlier}}},
        "campaign-1": {"site_id": plan["site_id"], "plan_digest": plan["plan_digest"],
                       "candidates": {CONTROL: {"status": "pending", "known_good_control": True},
                                      "candidate-000": {"status": "pending"}}},
        **(campaign_two or {})}}


INTERRUPTED = {"status": "blocked", "blockers": ["beta_screening_native_result_missing_or_invalid"]}


@pytest.fixture
def settled(monkeypatch):
    from blueprint_pipeline import beta_policy_screening_settlement as settlement

    monkeypatch.setattr(settlement, "settled_cost", lambda attempt: (
        Decimal(attempt["settled_cost_usd"]) if "settled_cost_usd" in attempt else None))


def test_the_reference_checks_three_dollars_is_a_total_across_attempts(settled):
    """Owner 2026-11-12: $3.00 per scene revision and task contract in total, across every attempt."""
    cheap = {**INTERRUPTED, "attempt": {"reservation_status": "committed", "maximum_cost_usd": "3.00",
                                        "settled_cost_usd": "0.40"}}
    # An interrupted first attempt that cost $0.40 leaves $2.60: a $3.00 second attempt is refused...
    plan = with_control(queue_plan(count=1), cost="3.00")
    ledger = _reference_ledger(plan, earlier=cheap)
    assert control.control_gate(ledger, "campaign-1", plan, plans=lambda *_: plan) == "closed"
    assert control.control_outcome(plan["candidates"][0], ledger["campaigns"]["campaign-1"]["candidates"][CONTROL]) == (
        "known_good_control_unavailable:budget")
    # ...and one at $2.60 or less runs.
    plan = with_control(queue_plan(count=1), cost="2.60")
    assert control.control_gate(_reference_ledger(plan, earlier=cheap), "campaign-1", plan, plans=lambda *_: plan) == "proceed"
    # At planning the row is quoted within what is left.
    catalog, capabilities, quotes = catalog_with_scripted(maximum="4.00", scripted_maximum="4.00")
    capabilities = rerated(capabilities, CONTROL, rate="0.80", ttl=14_400)
    row = website_plan(catalog, {**capabilities, control.REMAINING_CAPABILITY: "2.60"}, quotes)["candidates"][0]
    assert row["status"] == "selected" and Decimal(row["max_cost_usd"]) <= Decimal("2.60")


def test_a_first_attempt_that_held_the_whole_three_dollars_unsettled_allows_none(settled):
    held = {**INTERRUPTED, "attempt": {"reservation_status": "committed", "maximum_cost_usd": "3.00"}}
    plan = with_control(queue_plan(count=1), cost="0.50")
    assert control.control_gate(_reference_ledger(plan, earlier=held), "campaign-1", plan, plans=lambda *_: plan) == "closed"


def test_a_classed_result_is_final_and_a_new_scene_revision_starts_fresh(settled):
    plan = with_control(queue_plan(count=1))
    failed = {"status": "completed", "summary": _summary(3, 0, scripted=True, planned=3),
              "attempt": {"reservation_status": "committed", "maximum_cost_usd": "3.00", "settled_cost_usd": "1.20"}}
    ledger = _reference_ledger(plan, earlier=failed)
    assert control.control_gate(ledger, "campaign-1", plan, plans=lambda *_: plan) == "reused"
    fresh = with_control(queue_plan(count=1), scene_digest="sha256:" + "e" * 64)
    assert control.reuse_key(fresh) != control.reuse_key(plan)
    assert control.reference_spent(ledger, control.reuse_key(fresh), site_id=plan["site_id"]) == 0


def test_two_workers_racing_cannot_both_start_an_attempt(settled):
    plan = with_control(queue_plan(count=1), cost="1.00")
    other = {"campaign-2": {"site_id": plan["site_id"], "plan_digest": plan["plan_digest"],
                            "candidates": {CONTROL: {"status": "pending", "known_good_control": True},
                                           "candidate-000": {"status": "pending"}}}}
    ledger = _reference_ledger(plan, earlier={"status": "unavailable", "blockers": ["candidate_disabled"]},
                               campaign_two=other)
    # The gate runs under the ledger lock: the first claims, the second then waits for its result.
    assert control.control_gate(ledger, "campaign-1", plan, plans=lambda *_: plan) == "proceed"
    assert control.control_gate(ledger, "campaign-2", plan, plans=lambda *_: plan) == "wait"


def test_a_replan_carries_the_reference_row_first():
    catalog, capabilities, quotes = catalog_with_scripted()
    learned = [row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] != "scripted"]
    replan = _valid(website_plan(catalog, capabilities, quotes, candidate_ids=learned[1:]))
    row = replan["candidates"][0]
    # Before: ``append_control_row`` returned a re-plan unchanged; it now stands for the re-plan's own robots.
    assert control.is_control(row) and row["status"] == "selected"
    assert row["known_good_control"]["for_candidate_id"] == learned[1]
    assert [item["candidate_id"] for item in replan["candidates"][1:]] == learned[1:]


def test_the_arm_familys_body_is_chosen_without_any_fit_table():
    # The DROID Franka has no scripted body; its arm family's is the FR3 with the same gripper.
    catalog, capabilities, quotes = catalog_with_bodies([UR5E, FRANKA_BODY])
    value = _valid(website_plan(catalog, capabilities, quotes))
    row = value["candidates"][0]
    assert row["candidate_id"] == "blueprint_scripted_" + FRANKA_BODY
    assert row["known_good_control"]["chosen_by"] == "arm_family"
    # A fit table naming another body changes nothing (``_fallback_body`` read it before).
    table = [{"model_id": "universal_robots_ur5e", "recipe_id": "universal_robots_ur5e", "verdict": "fits"}]
    again = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: table}, quotes)
    assert again["candidates"][0]["candidate_id"] == row["candidate_id"]


def test_the_arm_family_map_names_a_registered_scripted_body():
    from blueprint_pipeline.droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID

    catalog = json.loads((ROOT / "config/beta_policy_candidates.v1.json").read_text())
    scripted = {row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] == "scripted"}
    assert set(control.ARM_FAMILY_BODIES) == {DROID_POLICY_CANARY_PRESET_ID}
    assert "blueprint_scripted_" + FRANKA_BODY in scripted
    assert FRANKA_BODY in control.ARM_FAMILY_BODIES[DROID_POLICY_CANARY_PRESET_ID]


@pytest.mark.parametrize("case, reason", [("no_body", "embodiment"), ("other_family_only", "embodiment"),
                                          ("refused", "embodiment"), ("no_money", "budget"), ("task_kind", "task_kind")])
def test_when_no_reference_can_run_the_job_records_why_and_the_robots_still_plan(case, reason):
    catalog, capabilities, quotes = (catalog_with_bodies([UR5E]) if case == "other_family_only"
                                     else catalog_with_scripted())
    value, remaining = scene(), 40
    if case == "no_body":
        catalog = seal_catalog({**catalog, "candidates": [row for row in catalog["candidates"]
                                                          if row["policy"]["kind"] != "scripted"]})
    elif case == "refused":
        capabilities = {**capabilities, "candidate_infeasible": {CONTROL: ["robot_reach_insufficient"]}}
    elif case == "no_money":
        remaining = 1
    elif case == "task_kind":
        value["task_kind"] = value["task_spec"]["task_kind"] = value["native_scene_plan"]["task_kind"] = "articulated_open"
        value["native_scene_plan"] = {**value["native_scene_plan"], "plan_digest": ""}
        value["native_scene_plan"]["plan_digest"] = canonical_digest(value["native_scene_plan"], digest_field="plan_digest")
        row = control.plan_control_row(catalog, value, tested=["pi05_droid"], capabilities=capabilities,
                                       cost_quotes=quotes, budget_usd=Decimal(40), now=NOW)
        assert row["known_good_control"]["unavailable"] == reason
        return
    planned = _valid(website_plan(catalog, capabilities, quotes, remaining=remaining))
    row = planned["candidates"][0]
    assert control.is_control(row) and row["status"] == "unavailable" and "max_cost_usd" not in row
    assert row["known_good_control"]["unavailable"] == reason
    assert control.control_flags(row, {"status": "unavailable"}) == {
        "setup_quality_flags": [f"known_good_control_unavailable:{reason}"]}
    # Nothing is held back from the robots for a row that cannot run.
    selected = sum(Decimal(item["max_cost_usd"]) for item in planned["candidates"] if item["status"] == "selected")
    assert Decimal(planned["maximum_planned_cost_usd"]) == selected


def test_the_three_dollar_cap_and_its_time_limit_are_unchanged():
    catalog, capabilities, quotes = catalog_with_scripted(7, maximum="4.00", scripted_maximum="4.00")
    # The FR3 body's registered price (host registry, 2026-11-12): $0.80/h, 4 h TTL, $4.00 quote.
    capabilities = rerated(capabilities, CONTROL, rate="0.80", ttl=14_400)
    row = website_plan(catalog, capabilities, quotes, remaining=15)["candidates"][0]
    assert control.REFERENCE_CAP_USD == Decimal("3.00") and row["max_cost_usd"] == "3.00"
    # The $4.00 quote kept $0.80 above $0.80/h x 4 h; the $3.00 row keeps half, so 3 h 15 min of GPU.
    assert row["execution_spec"]["hard_ttl_seconds"] == 11_700 and control.EPISODES == 3
    # At $10/h the cap holds 18 minutes, under the 30-minute shortest TTL: not run, for budget.
    expensive = rerated(capabilities, CONTROL, rate="10.00", ttl=14_400)
    assert website_plan(catalog, expensive, quotes)["candidates"][0]["known_good_control"]["unavailable"] == "budget"


@pytest.mark.parametrize("mutation", ["last", "second", "cells", "money", "for_unknown_candidate", "rule"])
def test_a_retained_reference_row_cannot_be_altered(mutation):
    catalog, capabilities, quotes = catalog_with_scripted()
    value = website_plan(catalog, capabilities, quotes)
    row = value["candidates"][0]
    if mutation == "last":
        value["candidates"].append(value["candidates"].pop(0))
    elif mutation == "second":
        value["candidates"].append(deepcopy(row))
    elif mutation == "cells":
        row["scenario_cells"] = row["scenario_cells"][:2]
    elif mutation == "money":
        row["max_cost_usd"] = "0.50"
    elif mutation == "for_unknown_candidate":
        row["known_good_control"]["for_candidate_id"] = "not_in_this_plan"
    else:
        row["known_good_control"]["chosen_by"] = "fit_table"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    with pytest.raises(BetaPolicyScreeningPlanError):
        validate_screening_plan(value)


# ---- the outcome and its count ------------------------------------------------------------------------------

def _summary(scored, succeeded, *, scripted, planned):
    kind = {"execution_evidence_kind": control.SCRIPTED, "action_source": "scripted_simulator_state",
            "scripted_rollout_count": planned, "learned_policy_rollout_count": 0, "scripted_controller_invoked": True,
            "candidate_policy_queried": False} if scripted else {}
    return {"episodes_scored": scored, "episodes_succeeded": succeeded, "episodes_failed": scored - succeeded,
            "episodes_void": planned - scored, "success_rate": succeeded / scored if scored else None,
            "session_result_digest": "sha256:" + "1" * 64, **kind}


V2 = {"known_good_control": {"schema_version": control.SCHEMA, "pass_at_successes": 2, "partly_at_successes": 1},
      "episode_count": 3}
V1 = {"known_good_control": {"schema_version": control.SCHEMA_V1, "pass_at_successes": 1}, "episode_count": 3}


@pytest.mark.parametrize("configured, scored, succeeded, outcome", [
    (V2, 3, 3, "known_good_control_passed:3_of_3"),
    (V2, 3, 2, "known_good_control_passed:2_of_3"),
    (V2, 3, 1, "known_good_control_partly:1_of_3"),  # origin/main: known_good_control_passed
    (V2, 3, 0, "known_good_control_failed:0_of_3"),
    (V2, 2, 2, "known_good_control_passed:2_of_3"),
    (V2, 2, 1, "known_good_control_interrupted"),
    (V1, 3, 1, "known_good_control_passed:1_of_3"),
    (V1, 3, 0, "known_good_control_failed:0_of_3"),
])
def test_the_outcome_records_its_count_and_one_of_three_is_partly(configured, scored, succeeded, outcome):
    row = {"status": "completed", "summary": _summary(scored, succeeded, scripted=True, planned=3)}
    assert control.control_outcome(configured, row) == outcome


# ---- the queue: first, and the robots run whatever its outcome ---------------------------------------------

def with_control(value, *, cost="3.00", campaign_id=None, scene_digest=None):
    """A coordinator test plan with a selected reference row first, as the website planner makes it."""
    value = deepcopy(value)
    if campaign_id is not None:
        value["campaign_id"] = campaign_id
    if scene_digest is not None:
        value["scene_revision_digest"] = value["source"]["scene_revision_digest"] = scene_digest
        for row in value["candidates"]:
            row["cost_quote"]["scene_revision_digest"] = scene_digest
            row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
    template = value["candidates"][0]
    spec = {**deepcopy(template["execution_spec"]), "candidate_id": CONTROL,
            "configuration_digest": canonical_digest({"value": "script"})}
    quote = {**deepcopy(template["cost_quote"]), "quote_id": "script", "episode_count": 3, "maximum_cost_usd": cost,
             "configuration_digest": spec["configuration_digest"], "quote_digest": ""}
    quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
    cells = []
    for ordinal in range(3):
        scenario = {"family": "canonical_anchor", "ordinal": ordinal, "parameters": {}}
        cells.append({"cell_id": f"screening.{value['scene_id']}.control.{ordinal:03d}", "seed": 7000 + ordinal,
                      "family": "canonical_anchor", "partition": "screening", "resolved_scenario": scenario,
                      "cell_spec_digest": canonical_digest(scenario), "resolved_scenario_digest": canonical_digest(scenario)})
    value["candidates"].insert(0, {
        "candidate_id": CONTROL, "display_name": control.DISPLAY_NAME,
        "configuration_digest": spec["configuration_digest"], "status": "selected", "blockers": [],
        "admission_digest": canonical_digest(spec), "execution_spec": spec, "cost_quote": quote, "max_cost_usd": cost,
        "episode_count": 3, "scenario_cells": cells, "execution_evidence_kind": control.SCRIPTED,
        "known_good_control": {"schema_version": control.SCHEMA, "role": "reference_only", "episodes": 3,
                               "pass_at_successes": 2, "partly_at_successes": 1,
                               "for_candidate_id": template["candidate_id"], "chosen_by": "candidate_robot"}})
    value["maximum_planned_cost_usd"] = f"{sum(Decimal(row['max_cost_usd']) for row in value['candidates']):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


class PerRow(Backend):
    """A backend whose observation is chosen per candidate; records the launch order."""

    def __init__(self, observations):
        super().__init__()
        self.observations, self.order, self.launched = observations, [], []

    def launch_candidate(self, **kwargs):
        self.launches += 1
        self.order.append(kwargs["candidate"]["candidate_id"])
        self.launched.append((kwargs["plan"]["campaign_id"], kwargs["candidate"]["candidate_id"]))
        return {"status": "running", "provider_allocation_performed": True}

    def reconcile_candidate(self, **kwargs):
        self.reconciliations += 1
        return self.observations.get(kwargs["candidate"]["candidate_id"], {"status": "running"})


def closed(result):
    return {"status": "completed", "result": result, "provider_allocation_performed": True, "provider_zero_verified": True}


@pytest.fixture
def summaries(monkeypatch):
    monkeypatch.setattr(coordinator, "candidate_result_summary", lambda result, **_kwargs: dict(result))
    monkeypatch.setattr(coordinator, "_settle", lambda *_args, **_kwargs: False)


def run(root, backend, passes=1, **options):
    for _ in range(passes):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend, **options)
    return coordinator.pending_publications(queue_root=root)


def _rows(publication):
    return {row["candidate_id"]: row for row in publication["candidates"]}


LEARNED_DONE = closed(_summary(10, 4, scripted=False, planned=10))


def done(*keys, control_result=None):
    observations = {key: LEARNED_DONE for key in keys}
    if control_result is not None:
        observations[CONTROL] = control_result
    return observations


def test_the_reference_row_launches_first(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=tmp_path)
    backend = PerRow({})
    [publication] = run(tmp_path, backend, passes=3)
    # Before: the learned rows ran and the reference waited for them to end, then for the campaign to be final.
    assert backend.order == [CONTROL, "candidate-000", "candidate-001"]
    assert _rows(publication)[CONTROL]["setup_quality_flags"] == ["known_good_control_pending"]
    Draft202012Validator(json.loads(PUBLICATION_SCHEMA.read_text())).validate(publication)


@pytest.mark.parametrize("scored, succeeded, outcome", [(3, 2, "known_good_control_passed:2_of_3"),
                                                        (3, 1, "known_good_control_partly:1_of_3"),
                                                        (3, 0, "known_good_control_failed:0_of_3"),
                                                        (2, 0, "known_good_control_interrupted")])
def test_the_robots_run_whatever_the_reference_outcome(tmp_path, summaries, scored, succeeded, outcome):
    reference = tmp_path / "reference"
    coordinator.enqueue_screening_campaign(queue_plan(count=2), queue_root=reference)
    [without] = run(reference, PerRow(done("candidate-000", "candidate-001")), passes=4)
    root = tmp_path / "with-reference"
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=root)
    backend = PerRow(done("candidate-000", "candidate-001",
                          control_result=closed(_summary(scored, succeeded, scripted=True, planned=3))))
    [publication] = run(root, backend, passes=8)
    rows = _rows(publication)
    assert rows.pop(CONTROL)["setup_quality_flags"] == [outcome]
    assert publication["status"] == "completed" and rows == _rows(without)
    Draft202012Validator(json.loads(PUBLICATION_SCHEMA.read_text())).validate(publication)


def test_a_reference_harness_failure_holds_no_robot(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=tmp_path)
    backend = PerRow(done("candidate-000", "candidate-001", control_result={
        "status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
        "blockers": ["beta_screening_native_result_missing_or_invalid"]}))
    [publication] = run(tmp_path, backend, passes=8)
    rows = _rows(publication)
    assert rows[CONTROL]["setup_quality_flags"] == ["known_good_control_interrupted"]
    assert {rows[key]["status"] for key in ("candidate-000", "candidate-001")} == {"completed"}


def test_a_finished_check_is_reused_and_one_running_elsewhere_is_waited_for(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1), campaign_id="campaign-2"),
                                           queue_root=tmp_path)
    backend = PerRow(done("candidate-000"))  # the first check is still running
    run(tmp_path, backend, passes=6, max_launches=3)
    assert backend.order.count(CONTROL) == 1
    states = json.loads((tmp_path / coordinator.LEDGER_FILENAME).read_text())["campaigns"]
    assert sorted(state["candidates"][CONTROL]["status"] for state in states.values()) == ["pending", "running"]
    backend.observations[CONTROL] = closed(_summary(3, 0, scripted=True, planned=3))
    publications = {row["campaign_id"]: row for row in run(tmp_path, backend, passes=6, max_launches=3)}
    assert backend.order.count(CONTROL) == 1  # never paid twice
    flags = sorted(_rows(value)[CONTROL]["setup_quality_flags"][0] for value in publications.values())
    assert flags == ["known_good_control_failed:0_of_3", "known_good_control_failed:reused"]


def test_a_pending_reference_row_is_a_hold_on_the_jobs_ceiling(tmp_path):
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    plain = tmp_path / "plain"
    coordinator.enqueue_screening_campaign(queue_plan(count=1), queue_root=plain)
    with_row = job_ceiling.local_screening_holds(tmp_path, site_id="site-job-1", published_usd=Decimal(0))
    without = job_ceiling.local_screening_holds(plain, site_id="site-job-1", published_usd=Decimal(0))
    assert with_row - without == Decimal("3.00")  # before: equal, it ran only on leftover money


def test_a_fresh_failure_pages_once_and_a_reused_one_never(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    run(tmp_path, PerRow(done("candidate-000", control_result=closed(_summary(3, 0, scripted=True, planned=3)))),
        passes=6)
    ledger = coordinator._ledger(tmp_path)
    campaigns = [(key, state, coordinator._plan(tmp_path, key, state["plan_digest"]))
                 for key, state in ledger["campaigns"].items()]
    assert control.failed_controls(campaigns) == ["campaign-1"]
    posts = []
    first = control.alert_failed_controls(tmp_path, ["campaign-1"], webhook_url="https://alerts.invalid/hook",
                                          poster=lambda url, payload: posts.append(payload))
    again = control.alert_failed_controls(tmp_path, ["campaign-1"], webhook_url="https://alerts.invalid/hook",
                                          poster=lambda url, payload: posts.append(payload))
    assert first["posted_now"] == ["campaign-1"] and again["posted_now"] == [] and len(posts) == 1


def test_a_version_one_plan_with_its_row_last_still_validates():
    value = deepcopy(queue_plan(count=2))
    v2 = with_control(value)
    row = deepcopy(v2["candidates"][0])
    row["known_good_control"] = {"schema_version": control.SCHEMA_V1, "role": "reference_only", "episodes": 3,
                                 "pass_at_successes": 1, "for_candidate_id": value["candidates"][0]["candidate_id"]}
    value["candidates"].append(row)
    value["maximum_planned_cost_usd"] = f"{sum(Decimal(item['max_cost_usd']) for item in value['candidates']):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    assert validate_screening_plan(value) == value and not control.first_in_plan(row)


def test_at_planning_the_ai_rows_are_exactly_as_without_the_reference_over_the_whole_grid():
    """Review 8c (1a): $0 to $30 in 50-cent steps, 1 to 5 AI rows; never selected while an AI row waits."""
    for count in range(1, 6):
        catalog, capabilities, quotes = catalog_with_scripted(count, maximum="4.00", scripted_maximum="4.00")
        capabilities = rerated(capabilities, CONTROL, rate="0.80", ttl=14_400)
        learned_only = seal_catalog({**catalog, "candidates": [row for row in catalog["candidates"]
                                                               if row["policy"]["kind"] != "scripted"]})
        for cents in range(0, 3001, 50):
            remaining = cents / 100
            value = website_plan(catalog, capabilities, quotes, remaining=remaining)
            without = website_plan(learned_only, capabilities, quotes, remaining=remaining)
            row, *learned = value["candidates"]
            assert [(item["candidate_id"], item["status"]) for item in learned] == [
                (item["candidate_id"], item["status"]) for item in candidate_rows(without)], (count, remaining)
            if row["status"] == "selected":
                assert not any(item["status"] == "waitlisted" for item in learned), (count, remaining)
                assert any(item["status"] == "selected" for item in learned), (count, remaining)


def _grid_plan(count, budget, reference):
    """The planner's shape for ``count`` $4 robots in ``budget``, with the reference row as the planner funds it."""
    from blueprint_pipeline.beta_policy_screening_plan import JOB_CEILING_DEFERRED  # noqa: F401

    value = deepcopy(queue_plan(count=count, cost="1.00"))  # validated at $15; re-sealed at $4 below
    for item in value["candidates"]:
        item["max_cost_usd"] = "4.00"
        item["cost_quote"].update(maximum_cost_usd="4.00", quote_digest="")
        item["cost_quote"]["quote_digest"] = canonical_digest(item["cost_quote"], digest_field="quote_digest")
    left = Decimal(budget)
    for item in value["candidates"]:
        if Decimal("4.00") <= left:
            left -= Decimal("4.00")
        elif Decimal("4.00") <= Decimal(budget):
            item.update(status="waitlisted", blockers=["site_screening_budget_insufficient"])
        else:
            for key in ("admission_digest", "execution_spec", "cost_quote", "max_cost_usd", "episode_count",
                        "scenario_cells"):
                item.pop(key)
            item.update(status="deferred", blockers=["site_screening_budget_insufficient"])
    value["budget_usd"] = f"{Decimal(budget):.2f}"
    if reference:
        row = deepcopy(with_control(queue_plan(count=1))["candidates"][0])  # for candidate-000, $3.00
        for field in ("scene_revision_digest",):
            row["cost_quote"][field] = value[field]
        row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
        value["candidates"].insert(0, control.fund_from_leftover(row, value,
                                                                 waitlist_blockers=["site_screening_budget_insufficient"]))
    value["maximum_planned_cost_usd"] = f"{sum(Decimal(r['max_cost_usd']) for r in value['candidates'] if r['status'] == 'selected'):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


@pytest.mark.parametrize("fraction", ["0", "0.25", "1"])
def test_with_settlements_the_ai_rows_launched_are_exactly_as_without_the_reference(tmp_path, summaries, monkeypatch,
                                                                                      fraction):
    """Review 8c (1): funded and launched AI rows identical with and without; no reference without a robot."""
    monkeypatch.setattr(coordinator, "settled_cost", lambda attempt: (
        Decimal(attempt["maximum_cost_usd"]) * Decimal(fraction) if attempt.get("reservation_status") == "committed"
        else None))
    for count in range(1, 6):
        for budget in ("3", "4", "6.5", "7", "8", "10.5", "11", "12", "15"):
            launched = {}
            for reference in (False, True):
                root = tmp_path / f"{fraction}-{count}-{budget}-{reference}"
                coordinator.enqueue_screening_campaign(_grid_plan(count, budget, reference), queue_root=root)
                backend = PerRow(done(*(f"candidate-{index:03}" for index in range(count)),
                                      control_result=closed(_summary(3, 2, scripted=True, planned=3))))
                run(root, backend, passes=3 * count + 6, max_launches=2)
                launched[reference] = backend.order
            ai = [key for key in launched[True] if key != CONTROL]
            assert sorted(ai) == sorted(launched[False]), (fraction, count, budget, launched)
            if CONTROL in launched[True]:
                assert ai and launched[True].index(CONTROL) > 0, (fraction, count, budget, launched)
