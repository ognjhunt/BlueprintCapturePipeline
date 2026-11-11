"""One scripted known-good control row per new website screening campaign; it never gates.

Owner approval 2026-11-10: a paid known-good control before the candidates, so "every candidate
failed" can be told apart from "this simulated set-up cannot register a success". It is reference
only, never a candidate or part of the estimate, never displaces a candidate, never makes one wait
and never stops the job: its outcome is a flag on its own row.
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
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline.beta_policy_candidate_catalog import seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import (
    JOB_CEILING_DEFERRED, WAITLIST_BLOCKER, BetaPolicyScreeningPlanError, build_job_ceiling_screening_plan,
    job_ceiling_record, validate_screening_plan,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import Backend, plan as queue_plan
from tests.test_beta_policy_screening_job_ceiling import READ_AT, ceiling_summary
from tests.test_beta_policy_screening_plan import NOW, admitted, inputs, scene

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = ROOT / "docs/schemas/beta_policy_screening_plan.v1.schema.json"
PUBLICATION_SCHEMA = ROOT / "docs/schemas/beta_policy_screening_publication.v3.schema.json"


def catalog_with_scripted(count=3, *, robot=None, maximum="2.00", scripted_maximum="1.50"):
    """``count`` learned rows and one scripted pick/place profile for the same robot (or ``robot``)."""
    catalog, _, _ = inputs(count, maximum)
    rows = deepcopy(catalog["candidates"])
    script = deepcopy(rows[0])
    script.update(candidate_id="blueprint_scripted_test_arm", display_name="Blueprint scripted pick/place - test arm")
    script["policy"]["kind"] = "scripted"
    script["interface"]["policy_interface_id"] = control.SCRIPTED_INTERFACE
    if robot is not None:
        script["interface"]["robot_profile_id"] = robot
    catalog = seal_catalog({**catalog, "candidates": [*rows, script]})
    bindings, quotes = {}, {}
    for row in catalog["candidates"]:
        price = scripted_maximum if row["policy"]["kind"] == "scripted" else maximum
        bindings[row["candidate_id"]], quotes[row["candidate_id"]] = admitted(row, maximum=price)
    return catalog, {"candidate_bindings": bindings}, quotes


def website_plan(catalog, capabilities, quotes, *, remaining=40, value=None, **options):
    record = job_ceiling_record(ceiling_summary(remaining), local_holds_usd=Decimal(0), read_at=READ_AT)
    return build_job_ceiling_screening_plan(catalog, value or scene(), job_ceiling=record, campaign_id="campaign1",
                                            capabilities=capabilities, cost_quotes=quotes, now=NOW, **options)


def test_a_new_website_campaign_plans_one_control_after_every_candidate():
    catalog, capabilities, quotes = catalog_with_scripted()
    value = website_plan(catalog, capabilities, quotes)
    *candidates, row = value["candidates"]
    assert [item["status"] for item in candidates] == ["selected"] * 3 and not any(map(control.is_control, candidates))
    assert row["candidate_id"] == "blueprint_scripted_test_arm" and row["status"] == "selected"
    assert row["display_name"] == control.DISPLAY_NAME and row["execution_evidence_kind"] == control.SCRIPTED
    assert row["known_good_control"] == {"schema_version": control.SCHEMA, "role": "reference_only", "episodes": 3,
                                         "pass_at_successes": 1, "for_candidate_id": candidates[0]["candidate_id"]}
    # Exactly what runs: three nominal (canonical anchor) episodes under the scripted profile's own hard cap.
    assert row["episode_count"] == 3 and len(row["scenario_cells"]) == 3
    assert {cell["family"] for cell in row["scenario_cells"]} == {"canonical_anchor"}
    assert all(cell["resolved_scenario"]["parameters"] == {} for cell in row["scenario_cells"])
    assert row["max_cost_usd"] == row["cost_quote"]["maximum_cost_usd"] == "1.50"
    assert row["cost_quote"]["episode_count"] == 3
    assert value["maximum_planned_cost_usd"] == "7.50"
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_a_replan_and_a_plan_without_the_field_carry_no_control():
    catalog, capabilities, quotes = catalog_with_scripted()
    learned = [row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] != "scripted"]
    replan = website_plan(catalog, capabilities, quotes, candidate_ids=learned)
    assert not any(map(control.is_control, replan["candidates"]))
    assert replan["maximum_planned_cost_usd"] == "6.00"
    # A plan made before the change has no control row, and stays valid as it is.
    older = queue_plan(count=2)
    assert not any(map(control.is_control, older["candidates"])) and validate_screening_plan(older) == older


@pytest.mark.parametrize("case, reason", [("other_robot", "embodiment"), ("no_profile", "embodiment"),
                                          ("full_budget", "budget"), ("task_kind", "task_kind")])
def test_a_control_that_cannot_run_here_is_listed_with_its_reason_and_spends_nothing(case, reason):
    catalog, capabilities, quotes = catalog_with_scripted(robot="another_robot" if case == "other_robot" else None)
    value = scene()
    if case == "no_profile":
        catalog = seal_catalog({**catalog, "candidates": [row for row in catalog["candidates"]
                                                          if row["policy"]["kind"] != "scripted"]})
    if case == "full_budget":
        # Seven $2 rows fill the $15 plan and waitlist the eighth: the control would come before it.
        catalog, capabilities, quotes = catalog_with_scripted(8)
    if case == "task_kind":
        value["task_kind"] = value["task_spec"]["task_kind"] = value["native_scene_plan"]["task_kind"] = "articulated_open"
        value["native_scene_plan"] = {**value["native_scene_plan"], "plan_digest": ""}
        value["native_scene_plan"]["plan_digest"] = canonical_digest(value["native_scene_plan"], digest_field="plan_digest")
        plan_value = {"candidates": [], "budget_usd": "15.00", "site_id": "site-request1"}
        row = control.plan_control_row(plan_value, catalog, value, capabilities=capabilities, cost_quotes=quotes,
                                       site_remaining_usd=None, now=NOW)
    else:
        planned = website_plan(catalog, capabilities, quotes)
        row = planned["candidates"][-1]
        assert validate_screening_plan(planned) == planned
        Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(planned)
        # Every candidate is planned exactly as it would be with no control at all.
        before = sum(Decimal(item["max_cost_usd"]) for item in planned["candidates"] if item["status"] == "selected")
        assert Decimal(planned["maximum_planned_cost_usd"]) == before
    assert control.is_control(row) and row["status"] == "unavailable" and row["blockers"] == ["candidate_disabled"]
    assert row["known_good_control"]["unavailable"] == reason and "max_cost_usd" not in row
    assert control.control_flags(row, {"status": "unavailable"}) == {
        "setup_quality_flags": [f"known_good_control_unavailable:{reason}"]}


def test_a_family_backend_scene_has_no_control_for_its_backend():
    catalog, capabilities, quotes = catalog_with_scripted()
    value = {key: item for key, item in scene().items() if key not in {"robot_free_probe_digest", "robot_free_probe_status"}}
    value.update(evaluation_backend="task_family_native.v1", family_static_readiness_digest="sha256:" + "c" * 64,
                 pre_policy_site_qualification_required=True, scenario_scope="native_frozen_reset_seeds")
    row = control.plan_control_row({"candidates": [], "budget_usd": "15.00"}, catalog, value, capabilities=capabilities,
                                   cost_quotes=quotes, site_remaining_usd=None, now=NOW)
    assert row["known_good_control"]["unavailable"] == "backend"


def test_a_site_allowance_too_small_for_the_control_lists_it_for_budget():
    catalog, capabilities, quotes = catalog_with_scripted()
    value = website_plan(catalog, capabilities, quotes, site_remaining_usd="7.00")
    assert [row["status"] for row in value["candidates"]] == ["selected", "selected", "selected", "unavailable"]
    assert value["candidates"][-1]["known_good_control"]["unavailable"] == "budget"


@pytest.mark.parametrize("mutation", ["middle", "second", "cells", "money"])
def test_a_retained_control_row_cannot_be_altered(mutation):
    catalog, capabilities, quotes = catalog_with_scripted()
    value = website_plan(catalog, capabilities, quotes)
    row = value["candidates"][-1]
    if mutation == "middle":
        value["candidates"].insert(0, value["candidates"].pop())
    elif mutation == "second":
        value["candidates"].append(deepcopy(row))
    elif mutation == "cells":
        row["scenario_cells"] = row["scenario_cells"][:2]
    else:
        row["max_cost_usd"] = "0.50"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    with pytest.raises(BetaPolicyScreeningPlanError):
        validate_screening_plan(value)


# ---- the queue ----------------------------------------------------------------------------------

def with_control(value, *, cost="3.00", campaign_id=None, scene_digest=None):
    """A coordinator test plan with a selected control row appended, as the website planner makes it."""
    value = deepcopy(value)
    if campaign_id is not None:
        value["campaign_id"] = campaign_id
    if scene_digest is not None:
        value["scene_revision_digest"] = value["source"]["scene_revision_digest"] = scene_digest
        for row in value["candidates"]:
            row["cost_quote"]["scene_revision_digest"] = scene_digest
            row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
    template = value["candidates"][0]
    spec = {**deepcopy(template["execution_spec"]), "candidate_id": "blueprint_scripted_test_arm",
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
    value["candidates"].append({
        "candidate_id": "blueprint_scripted_test_arm", "display_name": control.DISPLAY_NAME,
        "configuration_digest": spec["configuration_digest"], "status": "selected", "blockers": [],
        "admission_digest": canonical_digest(spec), "execution_spec": spec, "cost_quote": quote, "max_cost_usd": cost,
        "episode_count": 3, "scenario_cells": cells, "execution_evidence_kind": control.SCRIPTED,
        "known_good_control": {"schema_version": control.SCHEMA, "role": "reference_only", "episodes": 3,
                               "pass_at_successes": 1, "for_candidate_id": template["candidate_id"]}})
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


def _summary(scored, succeeded, *, scripted, planned):
    kind = {"execution_evidence_kind": control.SCRIPTED, "action_source": "scripted_simulator_state",
            "scripted_rollout_count": planned, "learned_policy_rollout_count": 0, "scripted_controller_invoked": True,
            "candidate_policy_queried": False} if scripted else {}
    return {"episodes_scored": scored, "episodes_succeeded": succeeded, "episodes_failed": scored - succeeded,
            "episodes_void": planned - scored, "success_rate": succeeded / scored if scored else None,
            "session_result_digest": "sha256:" + "1" * 64, **kind}


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


def test_the_control_launches_first_and_no_candidate_waits_for_its_result(tmp_path, summaries):
    value = with_control(queue_plan(count=2))
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = PerRow({})
    [publication] = run(tmp_path, backend)
    assert backend.order == ["blueprint_scripted_test_arm"]
    rows = _rows(publication)
    assert rows["blueprint_scripted_test_arm"]["setup_quality_flags"] == ["known_good_control_pending"]
    # The control is still running: the next slots go to the candidates all the same.
    [publication] = run(tmp_path, backend, passes=2)
    assert backend.order == ["blueprint_scripted_test_arm", "candidate-000", "candidate-001"]
    Draft202012Validator(json.loads(PUBLICATION_SCHEMA.read_text())).validate(publication)
    # When the candidates finish first, the running control keeps the campaign running.
    backend.observations = {key: closed(_summary(10, 4, scripted=False, planned=10)) for key in ("candidate-000", "candidate-001")}
    [publication] = run(tmp_path, backend)
    assert publication["status"] == "running"
    assert {key: row["status"] for key, row in _rows(publication).items()} == {
        "candidate-000": "completed", "candidate-001": "completed", "blueprint_scripted_test_arm": "running"}


@pytest.mark.parametrize("scored, succeeded, outcome", [(3, 1, "known_good_control_passed"),
                                                        (3, 0, "known_good_control_failed"),
                                                        (2, 0, "known_good_control_interrupted")])
def test_every_control_outcome_is_a_flag_and_the_candidates_are_unchanged(tmp_path, summaries, scored, succeeded,
                                                                         outcome):
    reference = tmp_path / "reference"
    coordinator.enqueue_screening_campaign(queue_plan(count=2), queue_root=reference)
    plain = PerRow({key: closed(_summary(10, 4, scripted=False, planned=10)) for key in ("candidate-000", "candidate-001")})
    [without] = run(reference, plain, passes=4)

    root = tmp_path / "with-control"
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=root)
    backend = PerRow({**plain.observations,
                      "blueprint_scripted_test_arm": closed(_summary(scored, succeeded, scripted=True, planned=3))})
    [publication] = run(root, backend, passes=5)
    rows = _rows(publication)
    assert rows.pop("blueprint_scripted_test_arm")["setup_quality_flags"] == [outcome]
    # A failed control holds nothing, blocks nothing and takes no candidate's reservation or run.
    assert publication["status"] == "completed" and rows == _rows(without)
    ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
    for key in ("candidate-000", "candidate-001"):
        assert ledger["campaigns"]["campaign-1"]["candidates"][key]["attempt"]["reservation_status"] == "committed"
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=root) is None


def test_a_control_harness_failure_holds_no_candidate(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=tmp_path)
    backend = PerRow({"blueprint_scripted_test_arm": {"status": "blocked", "provider_allocation_performed": True,
                                                       "provider_zero_verified": True,
                                                       "blockers": ["beta_screening_native_result_missing_or_invalid"]}})
    run(tmp_path, backend, passes=1)
    [publication] = run(tmp_path, backend, passes=3)
    rows = _rows(publication)
    assert rows["blueprint_scripted_test_arm"]["setup_quality_flags"] == ["known_good_control_interrupted"]
    assert backend.order == ["blueprint_scripted_test_arm", "candidate-000", "candidate-001"]
    assert {rows[key]["status"] for key in ("candidate-000", "candidate-001")} == {"running"}
    # A re-plan never takes the control up again.
    assert coordinator.replan_scope("campaign-1", implementation_commit="b" * 40, queue_root=tmp_path) is None


def test_the_control_never_reserves_ahead_of_a_queued_candidate_the_money_cannot_also_hold(tmp_path, summaries,
                                                                                        monkeypatch):
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal("25"))
    # Another campaign at the site holds $15 of $25: $10 is left, enough for both $5 candidates but
    # not for them and the $5 control as well. The candidates go first; the control waits.
    other = queue_plan(campaign_id="campaign-0", cost="5.00", count=3)
    coordinator.enqueue_screening_campaign(other, queue_root=tmp_path)
    for row in other["candidates"]:
        assert coordinator._reserve(tmp_path, other, row) is not None
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2, cost="5.00"), cost="5.00"),
                                           queue_root=tmp_path)
    backend = PerRow({})
    run(tmp_path, backend, passes=6, max_launches=2)
    assert [key for campaign, key in backend.launched if campaign == "campaign-1"] == ["candidate-000", "candidate-001"]
    publication = next(row for row in coordinator.pending_publications(queue_root=tmp_path)
                       if row["campaign_id"] == "campaign-1")
    rows = _rows(publication)
    assert {rows[key]["status"] for key in ("candidate-000", "candidate-001")} == {"running"}
    assert rows["blueprint_scripted_test_arm"]["status"] in {"pending", "waitlisted"}


def test_a_pass_or_fail_is_reused_for_the_same_check_and_never_paid_twice(tmp_path, summaries):
    first = with_control(queue_plan(count=1))
    coordinator.enqueue_screening_campaign(first, queue_root=tmp_path)
    backend = PerRow({"blueprint_scripted_test_arm": closed(_summary(3, 0, scripted=True, planned=3)),
                      "candidate-000": closed(_summary(10, 0, scripted=False, planned=10))})
    run(tmp_path, backend, passes=4)
    second = with_control(queue_plan(count=1), campaign_id="campaign-2")
    coordinator.enqueue_screening_campaign(second, queue_root=tmp_path)
    publications = {row["campaign_id"]: row for row in run(tmp_path, backend, passes=3)}
    assert backend.order.count("blueprint_scripted_test_arm") == 1
    row = _rows(publications["campaign-2"])["blueprint_scripted_test_arm"]
    assert row["status"] == "unavailable" and row["setup_quality_flags"] == ["known_good_control_failed:reused"]
    # Another scene revision is another check.
    third = with_control(queue_plan(count=1), campaign_id="campaign-3", scene_digest="sha256:" + "d" * 64)
    coordinator.enqueue_screening_campaign(third, queue_root=tmp_path)
    run(tmp_path, backend, passes=1)
    assert backend.order.count("blueprint_scripted_test_arm") == 2


def test_a_failed_control_alerts_the_owner_once(tmp_path, summaries, monkeypatch):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    backend = PerRow({"blueprint_scripted_test_arm": closed(_summary(3, 0, scripted=True, planned=3))})
    run(tmp_path, backend, passes=2)
    posts = []
    monkeypatch.setenv("BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL", "https://alerts.invalid/hook")
    from blueprint_pipeline import control_plane_capacity_controller as capacity

    monkeypatch.setattr(capacity, "post_alert", lambda url, report: posts.append(report))
    first = worker.alert_known_good_controls(tmp_path)
    again = worker.alert_known_good_controls(tmp_path)
    assert first["posted_now"] == ["campaign-1"] and again["posted_now"] == []
    assert [alert["code"] for report in posts for alert in report["alerts"]] == [control.ALERT_CODE]


def test_a_passed_control_alerts_nobody(tmp_path, summaries, monkeypatch):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    run(tmp_path, PerRow({"blueprint_scripted_test_arm": closed(_summary(3, 2, scripted=True, planned=3))}), passes=2)
    monkeypatch.setenv("BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL", "https://alerts.invalid/hook")
    assert worker.alert_known_good_controls(tmp_path)["posted_now"] == []


# ---- the body when no selected candidate's robot has a scripted row, and the funding rule -----------------

UR5E, UR10E, KUKA = ("universal_robots_ur5e_robotiq_2f85_sim", "universal_robots_ur10e_robotiq_2f85_sim",
                     "kuka_lbr_iiwa_14_robotiq_2f85_sim")
FRANKA_BODY = "franka_research_3_robotiq_2f85_sim"


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
    catalog = seal_catalog({**catalog, "candidates": rows})
    bindings, quotes = {}, {}
    for row in catalog["candidates"]:
        price = scripted_maximum if row["policy"]["kind"] == "scripted" else maximum
        bindings[row["candidate_id"]], quotes[row["candidate_id"]] = admitted(row, maximum=price)
    return catalog, {"candidate_bindings": bindings}, quotes


def fit(*rows):
    """A robot fit capability: (model_id, recipe_id, verdict) in the table's order."""
    return [{"model_id": model, "recipe_id": recipe, "verdict": verdict} for model, recipe, verdict in rows]


@pytest.mark.parametrize("table, expected", [
    # ``fits`` before ``fits_spec`` before ``unknown``, whatever the table's order.
    (fit(("universal_robots_ur5e", "universal_robots_ur5e", "unknown"),
         ("universal_robots_ur10e", "universal_robots_ur10e", "fits_spec"),
         ("kuka_lbr_iiwa_14", "kuka_lbr_iiwa_14", "fits")), ("kuka_lbr_iiwa_14", KUKA)),
    # A tie goes to the table's own order.
    (fit(("universal_robots_ur10e", "universal_robots_ur10e", "fits_spec"),
         ("universal_robots_ur5e", "universal_robots_ur5e", "fits_spec")), ("universal_robots_ur10e", UR10E)),
    # ``doesnt_fit`` never; ``unknown`` only when nothing better has a body.
    (fit(("universal_robots_ur5e", "universal_robots_ur5e", "doesnt_fit"),
         ("universal_robots_ur10e", "universal_robots_ur10e", "unknown")), ("universal_robots_ur10e", UR10E)),
    # A fitting model with no registered body is passed over.
    (fit(("fanuc_crx_10ia", "fanuc_crx_10ia", "fits"),
         ("universal_robots_ur5e", "universal_robots_ur5e", "fits_spec")), ("universal_robots_ur5e", UR5E)),
    # A recipe that is itself a body.
    (fit(("franka_research_3", FRANKA_BODY, "fits")), ("franka_research_3", FRANKA_BODY)),
])
def test_with_no_scripted_row_for_a_selected_robot_the_fit_table_chooses_one_body(table, expected):
    catalog, capabilities, quotes = catalog_with_bodies([UR5E, UR10E, KUKA, FRANKA_BODY])
    value = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: table}, quotes)
    *candidates, row = value["candidates"]
    model, body = expected
    assert [item["status"] for item in candidates] == ["selected"] * 3
    assert row["status"] == "selected" and row["candidate_id"] == "blueprint_scripted_" + body
    # It stands for the spec-catalog model, not for any candidate; no own station recorded: the packet's.
    assert row["known_good_control"] == {"schema_version": control.SCHEMA, "role": "reference_only", "episodes": 3,
                                         "pass_at_successes": 1, "for_model_id": model,
                                         "flags": ["scene_packet_station"]}
    assert row["episode_count"] == control.EPISODES and row["max_cost_usd"] == "1.50"
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_a_body_bound_to_its_own_station_carries_no_station_flag():
    catalog, capabilities, quotes = catalog_with_bodies([UR10E])
    capabilities = {**capabilities, control.FIT_CAPABILITY: fit(("universal_robots_ur10e", "universal_robots_ur10e", "fits")),
                    control.STATION_CAPABILITY: {"blueprint_scripted_" + UR10E: "embodiment_own_station"}}
    value = website_plan(catalog, capabilities, quotes)
    assert value["candidates"][-1]["known_good_control"] == {
        "schema_version": control.SCHEMA, "role": "reference_only", "episodes": 3, "pass_at_successes": 1,
        "for_model_id": "universal_robots_ur10e"}
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_a_selected_candidates_own_robot_still_comes_first():
    catalog, capabilities, quotes = catalog_with_scripted()
    table = fit(("universal_robots_ur10e", "universal_robots_ur10e", "fits"))
    value = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: table}, quotes)
    marker = value["candidates"][-1]["known_good_control"]
    assert set(marker) - {"schema_version", "role", "episodes", "pass_at_successes"} == {"for_candidate_id"}


@pytest.mark.parametrize("case", ["no_table", "unregistered", "doesnt_fit", "no_recipe"])
def test_no_fallback_without_a_registered_body_for_a_fitting_model(case):
    catalog, capabilities, quotes = catalog_with_bodies([UR5E])
    table = {"no_table": None,
             "unregistered": fit(("universal_robots_ur10e", "universal_robots_ur10e", "fits")),
             "doesnt_fit": fit(("universal_robots_ur5e", "universal_robots_ur5e", "doesnt_fit")),
             "no_recipe": fit(("universal_robots_ur20", None, "fits"))}[case]
    if table is not None:
        capabilities = {**capabilities, control.FIT_CAPABILITY: table}
    value = website_plan(catalog, capabilities, quotes)
    row = value["candidates"][-1]
    assert row["status"] == "unavailable" and row["known_good_control"]["unavailable"] == "embodiment"
    assert validate_screening_plan(value) == value


def test_a_measured_refusal_of_the_fallback_body_lists_it_for_its_embodiment():
    catalog, capabilities, quotes = catalog_with_bodies([UR10E])
    capabilities = {**capabilities, control.FIT_CAPABILITY: fit(("universal_robots_ur10e", "universal_robots_ur10e", "fits")),
                    "candidate_infeasible": {"blueprint_scripted_" + UR10E: ["robot_reach_insufficient:lift:1.2m_vs_1.0m"]}}
    value = website_plan(catalog, capabilities, quotes)
    row = value["candidates"][-1]
    assert row["candidate_id"] == "blueprint_scripted_" + UR10E and row["known_good_control"]["unavailable"] == "embodiment"


def test_a_leftover_too_small_for_any_waiting_row_funds_the_control():
    # Seven $2 rows fill $14 of the $15 plan and waitlist the eighth: $1 is left, which can never fund
    # the $2 row, but holds the $0.50 control. Before this rule any waitlisted row refused the control.
    catalog, capabilities, quotes = catalog_with_scripted(8, scripted_maximum="0.50")
    value = website_plan(catalog, capabilities, quotes)
    statuses = [row["status"] for row in value["candidates"]]
    assert statuses == ["selected"] * 7 + ["waitlisted", "selected"]
    assert value["candidates"][-1]["max_cost_usd"] == "0.50" and value["maximum_planned_cost_usd"] == "14.50"
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


@pytest.mark.parametrize("status, blocker, cost, quoted, planned", [
    ("waitlisted", WAITLIST_BLOCKER, "14.50", None, True),
    ("waitlisted", WAITLIST_BLOCKER, "1.00", None, False),
    ("waitlisted", JOB_CEILING_DEFERRED, "1.00", None, False),
    # A money-deferred row keeps no cost of its own: its quote says what it would cost.
    ("deferred", WAITLIST_BLOCKER, None, "14.50", True),
    ("deferred", WAITLIST_BLOCKER, None, "1.00", False),
    ("deferred", JOB_CEILING_DEFERRED, None, "1.00", False),
    # Nothing states its cost: counted as fundable, so the control never runs ahead of it.
    ("deferred", WAITLIST_BLOCKER, None, None, False),
])
def test_the_control_is_funded_only_when_the_leftover_could_fund_no_waiting_row(status, blocker, cost, quoted,
                                                                                   planned):
    catalog, capabilities, quotes = catalog_with_scripted(2)
    learned = [row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] != "scripted"]
    waiting = {"candidate_id": learned[1], "status": status, "blockers": [blocker],
               **({"max_cost_usd": cost} if cost is not None else {})}
    quotes = {key: value for key, value in quotes.items() if key != learned[1]}
    if quoted is not None:
        quotes[learned[1]] = {"maximum_cost_usd": quoted}
    # $1 selected of $15: $14 is left, which holds the $1.50 control.
    plan_value = {"budget_usd": "15.00", "candidates": [
        {"candidate_id": learned[0], "status": "selected", "blockers": [], "max_cost_usd": "1.00"}, waiting]}
    row = control.plan_control_row(plan_value, catalog, scene(), capabilities=capabilities, cost_quotes=quotes,
                                   site_remaining_usd=None, now=NOW)
    assert row["status"] == ("selected" if planned else "unavailable")
    if not planned:
        assert row["known_good_control"]["unavailable"] == "budget"


def test_the_site_allowance_bounds_the_leftover_too():
    catalog, capabilities, quotes = catalog_with_scripted(2)
    learned = [row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] != "scripted"]
    plan_value = {"budget_usd": "15.00", "candidates": [
        {"candidate_id": learned[0], "status": "selected", "blockers": [], "max_cost_usd": "1.00"},
        {"candidate_id": learned[1], "status": "waitlisted", "blockers": [WAITLIST_BLOCKER], "max_cost_usd": "3.00"}]}
    # The site has $3.50 left: $2.50 after the selected row holds the control and cannot fund the $3 row.
    row = control.plan_control_row(plan_value, catalog, scene(), capabilities=capabilities, cost_quotes=quotes,
                                   site_remaining_usd="3.50", now=NOW)
    assert row["status"] == "selected"
    # With $5 left, the $4 leftover could fund the waiting row: the control waits its turn.
    row = control.plan_control_row(plan_value, catalog, scene(), capabilities=capabilities, cost_quotes=quotes,
                                   site_remaining_usd="5.00", now=NOW)
    assert row["known_good_control"]["unavailable"] == "budget"


def test_a_plan_written_before_the_fallback_validates_unchanged():
    older = with_control(queue_plan(count=2))
    assert set(older["candidates"][-1]["known_good_control"]) == {"schema_version", "role", "episodes",
                                                                    "pass_at_successes", "for_candidate_id"}
    before = json.dumps(older, sort_keys=True)
    assert validate_screening_plan(deepcopy(older)) == older and json.dumps(older, sort_keys=True) == before
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(older)


@pytest.mark.parametrize("change", ["both", "neither", "flag_with_candidate", "other_flag", "model_pattern"])
def test_a_marker_that_stands_for_both_neither_or_anything_else_is_refused(change):
    value = with_control(queue_plan(count=2))
    marker = value["candidates"][-1]["known_good_control"]
    if change == "both":
        marker["for_model_id"] = "universal_robots_ur10e"
    elif change == "neither":
        del marker["for_candidate_id"]
    elif change == "flag_with_candidate":
        marker["flags"] = ["scene_packet_station"]
    elif change == "other_flag":
        del marker["for_candidate_id"]
        marker.update(for_model_id="universal_robots_ur10e", flags=["embodiment_own_station"])
    else:
        del marker["for_candidate_id"]
        marker["for_model_id"] = "../ur10e"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    with pytest.raises(BetaPolicyScreeningPlanError):
        validate_screening_plan(value)
    assert not Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).is_valid(value)


# ---- the website's own pick, when it fits -------------------------------------------------------------------

UR5E_FIT = ("universal_robots_ur5e", "universal_robots_ur5e")
UR10E_FIT = ("universal_robots_ur10e", "universal_robots_ur10e")
KUKA_FIT = ("kuka_lbr_iiwa_14", "kuka_lbr_iiwa_14")


@pytest.mark.parametrize("verdicts, preferred, expected", [
    # Same tier: the website's pick beats the table's order.
    (("fits_spec", "fits_spec", None), ["universal_robots_ur10e"], ("universal_robots_ur10e", True)),
    # fits and fits_spec are one tier for the preference: a preferred fits_spec beats a non-preferred fits.
    (("fits", "fits_spec", None), ["universal_robots_ur10e"], ("universal_robots_ur10e", True)),
    # The preference order holds: the pick, then its runner-up.
    (("fits_spec", "fits_spec", "fits"), ["kuka_lbr_iiwa_14", "universal_robots_ur10e"], ("kuka_lbr_iiwa_14", True)),
    (("fits_spec", "doesnt_fit", "fits"), ["universal_robots_ur10e", "kuka_lbr_iiwa_14"], ("kuka_lbr_iiwa_14", True)),
    # Never an unknown over a fitting model, preferred or not.
    (("fits_spec", "unknown", None), ["universal_robots_ur10e"], ("universal_robots_ur5e", False)),
    # Never doesnt_fit, preferred or not.
    (("fits_spec", "doesnt_fit", None), ["universal_robots_ur10e"], ("universal_robots_ur5e", False)),
    # Among unknowns only, the preference orders them too.
    (("unknown", "unknown", None), ["universal_robots_ur10e"], ("universal_robots_ur10e", True)),
    # A preferred model with no registered body is passed over.
    (("fits", "fits", None), ["fanuc_crx_10ia", "universal_robots_ur10e"], ("universal_robots_ur10e", True)),
])
def test_the_websites_pick_comes_first_among_models_of_its_tier(verdicts, preferred, expected):
    catalog, capabilities, quotes = catalog_with_bodies([UR5E, UR10E, KUKA])
    table = [(*model, verdict) for model, verdict in zip((UR5E_FIT, UR10E_FIT, KUKA_FIT), verdicts) if verdict]
    value = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: fit(*table),
                                   control.PREFERENCE_CAPABILITY: preferred}, quotes)
    marker = value["candidates"][-1]["known_good_control"]
    assert (marker["for_model_id"], marker["preferred"]) == expected
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_no_preference_is_todays_choice_and_marker():
    catalog, capabilities, quotes = catalog_with_bodies([UR5E, UR10E])
    table = fit((*UR5E_FIT, "fits_spec"), (*UR10E_FIT, "fits_spec"))
    value = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: table}, quotes)
    marker = value["candidates"][-1]["known_good_control"]
    assert marker["for_model_id"] == "universal_robots_ur5e" and "preferred" not in marker


@pytest.mark.parametrize("value", [["unknown_model"], "universal_robots_ur10e", [], ["universal_robots_ur10e"] * 2,
                                   ["universal_robots_ur5e", "universal_robots_ur10e", "kuka_lbr_iiwa_14",
                                    "fanuc_crx_10ia"], [7]])
def test_an_unusable_preference_is_ignored_with_a_typed_code(value):
    known = ["universal_robots_ur5e", "universal_robots_ur10e", "kuka_lbr_iiwa_14", "fanuc_crx_10ia"]
    assert control.preferred_model_ids(value, known) == ([], control.PREFERENCE_INVALID)
    assert control.preferred_model_ids(None, known) == ([], None)
    assert control.preferred_model_ids(["universal_robots_ur10e", "kuka_lbr_iiwa_14"], known) == (
        ["universal_robots_ur10e", "kuka_lbr_iiwa_14"], None)


def test_the_planner_reads_the_preference_beside_the_ceiling_and_ignores_a_bad_one(monkeypatch, caplog):
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    seen = []

    def build(catalog, scene_value, **options):
        seen.append(options.get("capabilities"))
        return {}

    import blueprint_pipeline.beta_policy_screening_plan as plan_module
    monkeypatch.setattr(plan_module, "build_job_ceiling_screening_plan", build)
    monkeypatch.setattr(job_ceiling, "local_screening_holds", lambda *_a, **_k: Decimal(0))
    monkeypatch.setattr(job_ceiling, "site_screening_remaining", lambda *_a, **_k: None)
    for sent, expected in ((["universal_robots_ur10e"], ["universal_robots_ur10e"]), (["nope"], None), (None, None)):
        def read(*, context, capture_root, extras=None, sent=sent):
            if sent is not None and extras is not None:
                extras["preferred_robot_model_ids"] = sent
            return ceiling_summary(40)
        monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", read)
        job_ceiling.plan_within_job_ceiling({}, {"site_id": "s"}, queue_root="/tmp", context={"scene_id": "s"},
                                            capture_root="/tmp", capabilities={
            "candidate_bindings": {}, control.FIT_CAPABILITY: fit((*UR10E_FIT, "fits"))})
        assert (seen[-1] or {}).get("preferred_robot_model_ids") == expected
    assert control.PREFERENCE_INVALID in caplog.text


@pytest.mark.parametrize("change", ["preferred_with_candidate", "preferred_not_bool"])
def test_a_preferred_flag_is_validated(change):
    value = with_control(queue_plan(count=2))
    marker = value["candidates"][-1]["known_good_control"]
    if change == "preferred_with_candidate":
        marker["preferred"] = True
    else:
        del marker["for_candidate_id"]
        marker.update(for_model_id="universal_robots_ur10e", preferred="yes")
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    with pytest.raises(BetaPolicyScreeningPlanError):
        validate_screening_plan(value)
    assert not Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).is_valid(value)


# ---- the scripted reference row's own $3.00 cap (owner approval 2026-11-11) --------------------------------

def rerated(capabilities, candidate_id, *, rate, ttl):
    """The binding of ``candidate_id`` at another hourly rate and hard TTL, re-sealed as planning seals it."""
    capabilities = deepcopy(capabilities)
    binding = capabilities["candidate_bindings"][candidate_id]
    binding["execution_spec"].update(max_hourly_rate_usd=rate, hard_ttl_seconds=ttl)
    binding["admission_digest"] = canonical_digest(binding["execution_spec"])
    return capabilities


def test_a_typical_job_now_plans_the_reference_row_at_three_dollars():
    # Seven learned rows at $4.00: three fill $12 of the $15 plan, the rest are waitlisted, $3.00 is left.
    catalog, capabilities, quotes = catalog_with_scripted(7, maximum="4.00", scripted_maximum="4.00")
    capabilities = rerated(capabilities, "blueprint_scripted_test_arm", rate="0.80", ttl=14_400)
    value = website_plan(catalog, capabilities, quotes)
    *candidates, row = value["candidates"]
    assert [item["status"] for item in candidates] == ["selected"] * 3 + ["waitlisted"] * 4
    assert all(item["max_cost_usd"] == "4.00" for item in candidates)
    assert row["status"] == "selected" and row["max_cost_usd"] == row["cost_quote"]["maximum_cost_usd"] == "3.00"
    assert control.REFERENCE_CAP_USD == Decimal("3.00")
    # $0.80/h x 4 h = $3.20 would not fit: the row's TTL is 3.75 h, so $0.80 x 3.75 h = $3.00.
    assert row["execution_spec"]["hard_ttl_seconds"] == 13_500
    assert row["admission_digest"] == canonical_digest(row["execution_spec"])
    assert row["cost_quote"]["quote_digest"] == canonical_digest(row["cost_quote"], digest_field="quote_digest")
    assert value["maximum_planned_cost_usd"] == "15.00"
    # The learned rows keep their own $4.00 cap and TTL.
    assert capabilities["candidate_bindings"][candidates[0]["candidate_id"]]["execution_spec"]["hard_ttl_seconds"] == 600
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_the_launch_check_accepts_the_capped_row(monkeypatch):
    from blueprint_pipeline import beta_policy_screening_launch as launch

    catalog, capabilities, quotes = catalog_with_scripted(7, maximum="4.00", scripted_maximum="4.00")
    capabilities = rerated(capabilities, "blueprint_scripted_test_arm", rate="0.80", ttl=14_400)
    value = website_plan(catalog, capabilities, quotes)
    row = value["candidates"][-1]
    row = deepcopy(row)
    row["cost_quote"]["valid_until"] = "2999-01-01T00:00:00Z"
    row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
    launch._check_quote(value, row)


@pytest.mark.parametrize("case", ["leftover_short", "waiting_row_fundable", "ttl_too_short"])
def test_the_capped_row_is_still_refused_when_the_money_or_the_ttl_does_not_hold(case):
    catalog, capabilities, quotes = catalog_with_scripted(7, maximum="4.00", scripted_maximum="4.00")
    capabilities = rerated(capabilities, "blueprint_scripted_test_arm", rate="0.80", ttl=14_400)
    learned = [row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] != "scripted"]
    if case == "leftover_short":
        # The site has $14.50 left: $2.50 after the three selected rows, under the $3.00 cap.
        value = website_plan(catalog, capabilities, quotes, site_remaining_usd="14.50")
        row = value["candidates"][-1]
    elif case == "waiting_row_fundable":
        plan_value = {"budget_usd": "15.00", "candidates": [
            {"candidate_id": learned[0], "status": "selected", "blockers": [], "max_cost_usd": "4.00"},
            {"candidate_id": learned[1], "status": "waitlisted", "blockers": [WAITLIST_BLOCKER], "max_cost_usd": "2.00"}]}
        row = control.plan_control_row(plan_value, catalog, scene(), capabilities=capabilities, cost_quotes=quotes,
                                       site_remaining_usd=None, now=NOW)
    else:
        # At $10/h the $3.00 cap holds only 18 minutes, under the 30-minute shortest TTL.
        capabilities = rerated(capabilities, "blueprint_scripted_test_arm", rate="10.00", ttl=14_400)
        row = website_plan(catalog, capabilities, quotes)["candidates"][-1]
    assert row["status"] == "unavailable" and row["known_good_control"]["unavailable"] == "budget"


def test_a_four_dollar_row_planned_before_the_cap_still_validates():
    older = with_control(queue_plan(count=2), cost="4.00")
    assert older["candidates"][-1]["max_cost_usd"] == "4.00" and validate_screening_plan(deepcopy(older)) == older
