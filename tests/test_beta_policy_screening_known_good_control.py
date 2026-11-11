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
    WAITLIST_BLOCKER, BetaPolicyScreeningPlanError, build_job_ceiling_screening_plan,
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
                                         "pass_at_successes": 1, "for_candidate_id": candidates[0]["candidate_id"],
                                         "chosen_by": "candidate_robot"}
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

def another_policy(value):
    """The same scene's plan for another learned policy: a policy finished or in flight in one campaign is
    never planned in a second (``beta_policy_screening_tested_pairs.require_new_pairs``)."""
    value = deepcopy(value)
    value["candidates"] = [row for row in value["candidates"] if row["candidate_id"] != "candidate-000"]
    value["maximum_planned_cost_usd"] = f"{sum(Decimal(row['max_cost_usd']) for row in value['candidates']):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


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
    # The job's source stays current (no owner closure, consent granted) unless a test says otherwise.
    monkeypatch.setattr(coordinator, "_source_still_current", lambda _plan: True)
    monkeypatch.setattr(coordinator, "_settle", lambda *_args, **_kwargs: False)


def run(root, backend, passes=1, **options):
    for _ in range(passes):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend, **options)
    return coordinator.pending_publications(queue_root=root)


def _rows(publication):
    return {row["candidate_id"]: row for row in publication["candidates"]}


LEARNED_DONE = closed(_summary(10, 4, scripted=False, planned=10))
CONTROL = "blueprint_scripted_test_arm"


def done(*keys, control_result=None):
    """Observations: each learned key completes; the control completes with ``control_result`` if given."""
    observations = {key: LEARNED_DONE for key in keys}
    if control_result is not None:
        observations[CONTROL] = control_result
    return observations


def test_the_control_runs_last_once_every_learned_row_has_ended(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=tmp_path)
    backend = PerRow({})
    [publication] = run(tmp_path, backend, passes=4)
    # The learned rows run; the control waits for them, never for money.
    assert backend.order == ["candidate-000", "candidate-001"]
    assert _rows(publication)[CONTROL]["setup_quality_flags"] == ["known_good_control_pending"]
    Draft202012Validator(json.loads(PUBLICATION_SCHEMA.read_text())).validate(publication)
    backend.observations = done("candidate-000", "candidate-001",
                                control_result=closed(_summary(3, 1, scripted=True, planned=3)))
    [publication] = run(tmp_path, backend, passes=4)
    assert backend.order == ["candidate-000", "candidate-001", CONTROL]
    assert publication["status"] == "completed"
    assert _rows(publication)[CONTROL]["setup_quality_flags"] == ["known_good_control_passed"]


@pytest.mark.parametrize("scored, succeeded, outcome", [(3, 1, "known_good_control_passed"),
                                                        (3, 0, "known_good_control_failed"),
                                                        (2, 0, "known_good_control_interrupted")])
def test_every_control_outcome_is_a_flag_and_the_candidates_are_unchanged(tmp_path, summaries, scored, succeeded,
                                                                         outcome):
    reference = tmp_path / "reference"
    coordinator.enqueue_screening_campaign(queue_plan(count=2), queue_root=reference)
    plain = PerRow(done("candidate-000", "candidate-001"))
    [without] = run(reference, plain, passes=4)

    root = tmp_path / "with-control"
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=root)
    backend = PerRow(done("candidate-000", "candidate-001",
                          control_result=closed(_summary(scored, succeeded, scripted=True, planned=3))))
    [publication] = run(root, backend, passes=8)
    rows = _rows(publication)
    assert rows.pop(CONTROL)["setup_quality_flags"] == [outcome]
    # A failed control holds nothing, blocks nothing and takes no candidate's reservation or run.
    assert publication["status"] == "completed" and rows == _rows(without)
    ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
    for key in ("candidate-000", "candidate-001"):
        assert ledger["campaigns"]["campaign-1"]["candidates"][key]["attempt"]["reservation_status"] == "committed"
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=root) is None


def test_a_control_harness_failure_holds_no_candidate(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2)), queue_root=tmp_path)
    backend = PerRow(done("candidate-000", "candidate-001", control_result={
        "status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
        "blockers": ["beta_screening_native_result_missing_or_invalid"]}))
    [publication] = run(tmp_path, backend, passes=8)
    rows = _rows(publication)
    assert rows[CONTROL]["setup_quality_flags"] == ["known_good_control_interrupted"]
    assert backend.order == ["candidate-000", "candidate-001", CONTROL]
    assert {rows[key]["status"] for key in ("candidate-000", "candidate-001")} == {"completed"}
    # A re-plan never takes the control up again.
    assert coordinator.replan_scope("campaign-1", implementation_commit="b" * 40, queue_root=tmp_path) is None


def test_a_control_whose_cap_no_longer_fits_ends_at_once_for_budget(tmp_path, summaries, monkeypatch):
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal("25"))
    # Another campaign at the site holds $15 of $25. Both $5 candidates run ($25 held); the $5 control
    # then cannot fit and ends at once: it never waits on money, so the campaign completes.
    other = queue_plan(campaign_id="campaign-0", cost="5.00", count=3)
    coordinator.enqueue_screening_campaign(other, queue_root=tmp_path)
    for row in other["candidates"]:
        assert coordinator._reserve(tmp_path, other, row) is not None
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=2, cost="5.00"), cost="5.00"),
                                           queue_root=tmp_path)
    backend = PerRow(done("candidate-000", "candidate-001"))
    run(tmp_path, backend, passes=8, max_launches=2)
    assert [key for campaign, key in backend.launched if campaign == "campaign-1"] == ["candidate-000", "candidate-001"]
    publication = next(row for row in coordinator.pending_publications(queue_root=tmp_path)
                       if row["campaign_id"] == "campaign-1")
    rows = _rows(publication)
    assert {rows[key]["status"] for key in ("candidate-000", "candidate-001")} == {"completed"}
    assert rows[CONTROL]["status"] == "deferred"
    assert rows[CONTROL]["setup_quality_flags"] == ["known_good_control_unavailable:budget"]
    assert publication["status"] in {"completed", "blocked"}


def test_a_pass_or_fail_is_reused_for_the_same_check_and_never_paid_twice(tmp_path, summaries):
    first = with_control(queue_plan(count=1))
    coordinator.enqueue_screening_campaign(first, queue_root=tmp_path)
    backend = PerRow(done("candidate-000", "candidate-001",
                          control_result=closed(_summary(3, 0, scripted=True, planned=3))))
    run(tmp_path, backend, passes=6)
    second = with_control(another_policy(queue_plan(count=2)), campaign_id="campaign-2")
    coordinator.enqueue_screening_campaign(second, queue_root=tmp_path)
    publications = {row["campaign_id"]: row for row in run(tmp_path, backend, passes=6)}
    assert backend.order.count(CONTROL) == 1
    row = _rows(publications["campaign-2"])[CONTROL]
    assert row["status"] == "unavailable" and row["setup_quality_flags"] == ["known_good_control_failed:reused"]
    # Another scene revision is another check.
    third = with_control(queue_plan(count=1), campaign_id="campaign-3", scene_digest="sha256:" + "d" * 64)
    coordinator.enqueue_screening_campaign(third, queue_root=tmp_path)
    run(tmp_path, backend, passes=6)
    assert backend.order.count(CONTROL) == 2


def test_the_same_check_active_elsewhere_closes_this_one_without_waiting(tmp_path, summaries):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    coordinator.enqueue_screening_campaign(with_control(another_policy(queue_plan(count=2)), campaign_id="campaign-2"),
                                           queue_root=tmp_path)
    backend = PerRow(done("candidate-000", "candidate-001"))  # the first control keeps running (or never reconciles)
    run(tmp_path, backend, passes=8, max_launches=3)
    assert backend.order.count(CONTROL) == 1
    ledger = json.loads((tmp_path / coordinator.LEDGER_FILENAME).read_text())
    # The second is closed as not run at once: it never waits on another campaign's run.
    assert sorted(state["candidates"][CONTROL]["status"] for state in ledger["campaigns"].values()) == [
        "running", "unavailable"]
    publications = {row["campaign_id"]: row for row in coordinator.pending_publications(queue_root=tmp_path)}
    flags = {key: _rows(value)[CONTROL]["setup_quality_flags"] for key, value in publications.items()}
    assert sorted(flag for values in flags.values() for flag in values) == [
        "known_good_control_interrupted", "known_good_control_pending"]


# ---- the control never changes how a learned row is funded, and the campaign always completes -------------

def shaped(*, control_row, waiting=None, count=3):
    """Learned rows at $4.00 (the last one waitlisted at ``waiting`` when given), with or without the control."""
    value = deepcopy(queue_plan(count=count, cost="4.00"))
    if control_row:
        value = deepcopy(with_control(value, cost="3.00"))
    if waiting is not None:
        row = value["candidates"][count - 1]
        row.update(status="waitlisted", blockers=[WAITLIST_BLOCKER], max_cost_usd=waiting)
        row["cost_quote"]["maximum_cost_usd"] = waiting
        row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
    value["maximum_planned_cost_usd"] = f"{sum(Decimal(item['max_cost_usd']) for item in value['candidates'] if item['status'] == 'selected'):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


class FirstPreparationFails(PerRow):
    """``candidate-000``'s preparation fails before any spend, so its reservation is released."""

    def prepare_candidate(self, **kwargs):
        if kwargs["candidate"]["candidate_id"] == "candidate-000":
            raise RuntimeError("preparation failed")
        return super().prepare_candidate(**kwargs)
SHAPES = {
    "all_selected": dict(count=2, waiting=None, refuse=False),
    "waitlisted_never_fundable": dict(count=3, waiting="7.50", refuse=False),
    "waitlisted_funded_by_a_release": dict(count=3, waiting="8.50", refuse=True),
    "waitlisted_funded_with_room_left_after": dict(count=3, waiting="4.00", refuse=True),
}


def drive(root, value, *, refuse, passes=20, max_launches=2):
    coordinator.enqueue_screening_campaign(value, queue_root=root)
    learned = [row["candidate_id"] for row in value["candidates"] if not control.is_control(row)]
    observations = done(*learned, control_result=closed(_summary(3, 1, scripted=True, planned=3)))
    backend = (FirstPreparationFails if refuse else PerRow)(observations)
    run(root, backend, passes=passes, max_launches=max_launches)
    return json.loads((root / coordinator.LEDGER_FILENAME).read_text())["campaigns"]["campaign-1"], backend


def _learned_money(state):
    """Each learned row's final state and money: status, blockers and every attempt's reservation and cap."""
    return {key: (row["status"], sorted(row.get("blockers") or []),
                  [(item["reservation_status"], item["maximum_cost_usd"])
                   for item in [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]])
            for key, row in state["candidates"].items() if not row.get("known_good_control")}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_learned_row_ends_exactly_as_without_the_control(tmp_path, summaries, shape):
    settings = dict(SHAPES[shape])
    refuse = settings.pop("refuse")
    without, plain = drive(tmp_path / "without", shaped(control_row=False, **settings), refuse=refuse)
    with_row, backend = drive(tmp_path / "with", shaped(control_row=True, **settings), refuse=refuse)
    assert _learned_money(with_row) == _learned_money(without)
    assert [key for key in backend.order if key != CONTROL] == plain.order
    # The control ran last, if at all, after every learned launch.
    if CONTROL in backend.order:
        assert backend.order[-1] == CONTROL and backend.order.count(CONTROL) == 1


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_no_state_leaves_the_control_pending_while_nothing_else_can_move(tmp_path, summaries, shape):
    settings = dict(SHAPES[shape])
    refuse = settings.pop("refuse")
    state, _backend = drive(tmp_path, shaped(control_row=True, **settings), refuse=refuse)
    # Quiescent: every row has ended, the control included, and the campaign is no longer queued.
    assert all(row["status"] not in {"pending", "waitlisted", "preparing", "launching", "running",
                                     "reconciliation_required"} for row in state["candidates"].values())
    assert coordinator.campaign_status(state["candidates"]) not in {"queued", "running"}
    configured = next(row for row in shaped(control_row=True, **settings)["candidates"] if control.is_control(row))
    outcome = control.control_outcome(configured, state["candidates"][CONTROL])
    assert outcome in {control.PASSED, control.INTERRUPTED, f"{control.UNAVAILABLE}:budget"}
    # It ran only when the campaign without it ended completed for good, rows closed for money included.
    assert (outcome == control.PASSED) is (shape in {"all_selected", "waitlisted_never_fundable"})


def test_the_reviewers_deadlock_shape_now_completes_and_funds_the_waiting_row(tmp_path, summaries):
    # 2026-11-11 review: three $4 rows, the third waitlisted at $8.50, the $3 control planned. A release
    # funds the waiting row ($4 + $8.50 of $15) exactly as without the control. The released row would be
    # planned again after a deploy, so the control closes without running: the campaign completes.
    state, backend = drive(tmp_path, shaped(control_row=True, count=3, waiting="8.50"), refuse=True)
    assert state["candidates"]["candidate-002"]["status"] == "completed"
    assert state["candidates"][CONTROL]["status"] == "unavailable" and CONTROL not in backend.order


def test_an_unreserved_control_is_not_a_hold_on_the_jobs_ceiling(tmp_path):
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    plain = tmp_path / "plain"
    coordinator.enqueue_screening_campaign(queue_plan(count=1), queue_root=plain)
    assert (job_ceiling.local_screening_holds(tmp_path, site_id="site-job-1", published_usd=Decimal(0))
            == job_ceiling.local_screening_holds(plain, site_id="site-job-1", published_usd=Decimal(0)))


def test_a_failed_control_alerts_the_owner_once(tmp_path, summaries, monkeypatch):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    backend = PerRow(done("candidate-000", control_result=closed(_summary(3, 0, scripted=True, planned=3))))
    run(tmp_path, backend, passes=6)
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
    run(tmp_path, PerRow(done("candidate-000", control_result=closed(_summary(3, 2, scripted=True, planned=3)))),
        passes=6)
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
                                         "flags": ["scene_packet_station"],
                                         # The FR3 is the body nearest the candidates' DROID Franka.
                                         "chosen_by": "nearest_body" if model == "franka_research_3" else "fit_table"}
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
        "for_model_id": "universal_robots_ur10e", "chosen_by": "fit_table"}
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_a_selected_candidates_own_robot_still_comes_first():
    catalog, capabilities, quotes = catalog_with_scripted()
    table = fit(("universal_robots_ur10e", "universal_robots_ur10e", "fits"))
    value = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: table}, quotes)
    marker = value["candidates"][-1]["known_good_control"]
    assert set(marker) - {"schema_version", "role", "episodes", "pass_at_successes"} == {"for_candidate_id", "chosen_by"}


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


def test_at_planning_the_control_needs_only_its_cap_to_fit_what_the_selected_rows_leave():
    catalog, capabilities, quotes = catalog_with_scripted(2)
    learned = [row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] != "scripted"]
    plan_value = {"budget_usd": "15.00", "candidates": [
        {"candidate_id": learned[0], "status": "selected", "blockers": [], "max_cost_usd": "1.00"},
        {"candidate_id": learned[1], "status": "waitlisted", "blockers": [WAITLIST_BLOCKER], "max_cost_usd": "1.00"}]}
    # A waiting row does not refuse it: at run time it is reserved only after every learned row has ended.
    row = control.plan_control_row(plan_value, catalog, scene(), capabilities=capabilities, cost_quotes=quotes,
                                   site_remaining_usd=None, now=NOW)
    assert row["status"] == "selected"
    # The site's allowance bounds it too: $2.40 left, $1.40 after the selected row, under its $1.50 cap.
    row = control.plan_control_row(plan_value, catalog, scene(), capabilities=capabilities, cost_quotes=quotes,
                                   site_remaining_usd="2.40", now=NOW)
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
    # Fitted like a site-fitted quote: the $4.00 quote kept $0.80 above $0.80/h x 4 h; the $3.00 row keeps
    # half of that, $0.40, so its bound is $2.60: 3.25 h at $0.80/h.
    assert row["execution_spec"]["hard_ttl_seconds"] == 11_700
    assert Decimal(row["max_cost_usd"]) - Decimal("2.60") == Decimal("0.40")
    assert "fitted_to_scripted_reference_cap" in row["cost_quote"]["price_source"]
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


@pytest.mark.parametrize("case", ["leftover_short", "ttl_too_short"])
def test_the_capped_row_is_still_refused_when_the_money_or_the_ttl_does_not_hold(case):
    catalog, capabilities, quotes = catalog_with_scripted(7, maximum="4.00", scripted_maximum="4.00")
    capabilities = rerated(capabilities, "blueprint_scripted_test_arm", rate="0.80", ttl=14_400)
    if case == "leftover_short":
        # The site has $14.50 left: $2.50 after the three selected rows, under the $3.00 cap.
        value = website_plan(catalog, capabilities, quotes, site_remaining_usd="14.50")
        row = value["candidates"][-1]
    else:
        # At $10/h the $3.00 cap holds only 18 minutes, under the 30-minute shortest TTL.
        capabilities = rerated(capabilities, "blueprint_scripted_test_arm", rate="10.00", ttl=14_400)
        row = website_plan(catalog, capabilities, quotes)["candidates"][-1]
    assert row["status"] == "unavailable" and row["known_good_control"]["unavailable"] == "budget"


def test_a_four_dollar_row_planned_before_the_cap_still_validates():
    older = with_control(queue_plan(count=2), cost="4.00")
    assert older["candidates"][-1]["max_cost_usd"] == "4.00" and validate_screening_plan(deepcopy(older)) == older


# ---- the default body: the one nearest the learned candidates' own robot --------------------------------------

FR3_FIT = ("franka_research_3", FRANKA_BODY)


@pytest.mark.parametrize("verdicts, preferred, expected", [
    # Nearest body first within the fitting tier, ahead of the website's pick and the table's order.
    (("fits", "fits_spec"), ["universal_robots_ur5e"], ("franka_research_3", "nearest_body")),
    (("fits", "fits"), None, ("franka_research_3", "nearest_body")),
    # Never an unknown nearest body over a fitting model: the website's pick, then the table.
    (("fits", "unknown"), ["universal_robots_ur5e"], ("universal_robots_ur5e", "preferred")),
    (("fits", "unknown"), None, ("universal_robots_ur5e", "fit_table")),
    # Never doesnt_fit, nearest or not.
    (("fits_spec", "doesnt_fit"), None, ("universal_robots_ur5e", "fit_table")),
    # Among unknowns only, the nearest body still goes first.
    (("unknown", "unknown"), None, ("franka_research_3", "nearest_body")),
])
def test_the_body_nearest_the_candidates_robot_is_the_default(verdicts, preferred, expected):
    catalog, capabilities, quotes = catalog_with_bodies([UR5E, FRANKA_BODY])
    table = fit((*UR5E_FIT, verdicts[0]), (*FR3_FIT, verdicts[1]))
    extra = {control.PREFERENCE_CAPABILITY: preferred} if preferred else {}
    value = website_plan(catalog, {**capabilities, control.FIT_CAPABILITY: table, **extra}, quotes)
    marker = value["candidates"][-1]["known_good_control"]
    assert (marker["for_model_id"], marker["chosen_by"]) == expected
    assert validate_screening_plan(value) == value
    Draft202012Validator(json.loads(PLAN_SCHEMA.read_text())).validate(value)


def test_the_nearest_body_map_names_real_catalog_models_and_a_registered_body():
    from blueprint_pipeline.droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID

    spec = json.loads((ROOT / "src/blueprint_pipeline/_catalog_data/robot_spec_catalog.v1.json").read_text())
    recipes = {model["model_id"]: model.get("recipe_id") for model in spec["models"]}
    assert set(control.NEAREST_BODY_MODELS) == {DROID_POLICY_CANARY_PRESET_ID}
    models = control.NEAREST_BODY_MODELS[DROID_POLICY_CANARY_PRESET_ID]
    assert all(model in recipes for model in models)
    # The FR3's recipe is itself a registered scripted body; the Panda's (the DROID asset) is not one.
    catalog = json.loads((ROOT / "config/beta_policy_candidates.v1.json").read_text())
    scripted = {row["candidate_id"] for row in catalog["candidates"] if row["policy"]["kind"] == "scripted"}
    assert "blueprint_scripted_" + recipes["franka_research_3"] in scripted
    assert "blueprint_scripted_" + recipes["franka_emika_panda"] + "_robotiq_2f85_sim" not in scripted


@pytest.mark.parametrize("change", ["candidate_with_other_rule", "model_with_candidate_rule", "unknown_rule"])
def test_a_chosen_by_rule_is_validated(change):
    value = with_control(queue_plan(count=2))
    marker = value["candidates"][-1]["known_good_control"]
    if change == "candidate_with_other_rule":
        marker["chosen_by"] = "fit_table"
    else:
        del marker["for_candidate_id"]
        marker.update(for_model_id="universal_robots_ur10e",
                      chosen_by="candidate_robot" if change == "model_with_candidate_rule" else "guess")
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    with pytest.raises(BetaPolicyScreeningPlanError):
        validate_screening_plan(value)


# ---- round 4: the row runs only when its campaign, without it, ended completed for good --------------------
#
# Real settlement and failure paths: the summaries are passed through, but committed attempts settle through
# the coordinator's own ``_settle`` (at $1 each), harness failures hold siblings, a deploy re-plans and a
# successor campaign runs at the same site. Every shape is driven with and without the row and compared.

HARNESS_FAIL = {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
                "blockers": ["beta_screening_native_result_missing_or_invalid"]}
REVOKED = {"status": "blocked", "provider_allocation_performed": False, "provider_zero_verified": True,
           "blockers": ["beta_screening_current_source_revoked"]}
LATER = "b" * 40


@pytest.fixture
def settles(monkeypatch):
    from blueprint_pipeline import beta_policy_screening_settlement as settlement

    monkeypatch.setattr(coordinator, "candidate_result_summary", lambda result, **_kwargs: dict(result))
    monkeypatch.setattr(coordinator, "_source_still_current", lambda _plan: True)

    def settle(*, attempt_dir, attempt, candidate, settled_at):
        receipt = {"schema_version": settlement.SETTLEMENT_SCHEMA, "settlement_basis": settlement.SETTLEMENT_BASIS,
                   "settled_cost_usd": "1", "settled_at": settled_at, "attempt_id": attempt["attempt_id"],
                   "maximum_cost_usd": attempt["maximum_cost_usd"], "receipt_digest": ""}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        return {"settled_cost_usd": "1", "settlement_basis": settlement.SETTLEMENT_BASIS, "settled_at": settled_at,
                "settlement_receipt": receipt, "settlement_receipt_digest": receipt["receipt_digest"]}

    monkeypatch.setattr(settlement, "attempt_settlement", settle)


def _successor(value, scope):
    """The re-plan a deploy enqueues: the same source, only the candidates ``_replan_scope`` named."""
    value = deepcopy(value)
    value["campaign_id"] = "campaign-2"
    value["candidates"] = [row for row in value["candidates"] if row["candidate_id"] in scope
                           and not control.is_control(row)]
    for row in value["candidates"]:
        row.update(status="selected", blockers=[])
    value["maximum_planned_cost_usd"] = f"{sum(Decimal(row['max_cost_usd']) for row in value['candidates']):.2f}"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


class OtherCampaignKeepsRunning(PerRow):
    """Campaign ``campaign-0`` (another job at the site) keeps its reservations: its runs never end here."""

    def reconcile_candidate(self, **kwargs):
        if kwargs["plan"]["campaign_id"] == "campaign-0":
            self.reconciliations += 1
            return {"status": "running"}
        return super().reconcile_candidate(**kwargs)


def _observe(root, value, observations, *, passes=12, max_launches=2, setup=None):
    coordinator.enqueue_screening_campaign(value, queue_root=root)
    if setup:
        setup(root)
    backend = (OtherCampaignKeepsRunning if setup else PerRow)(observations)
    run(root, backend, passes=passes, max_launches=max_launches)
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
    state = ledger["campaigns"]["campaign-1"]
    learned = {key: row for key, row in state["candidates"].items() if not row.get("known_good_control")}
    facts = {"learned": _learned_money(state), "status": coordinator.campaign_status(learned),
             "scope": coordinator.replan_scope("campaign-1", implementation_commit=LATER, queue_root=root),
             "self_scope": coordinator.replan_scope("campaign-1", implementation_commit="a" * 40, queue_root=root),
             "holds": job_ceiling.local_screening_holds(root, site_id="site-job-1", published_usd=Decimal(0))}
    if facts["scope"]:
        # A deploy: the successor plans what the scope names, then runs; its own rows and the site's money
        # left for it are compared too.
        facts["site_left_for_successor"] = job_ceiling.site_screening_remaining(root, site_id="site-job-1")
        coordinator.supersede_campaign("campaign-1", _successor(value, facts["scope"]), implementation_commit=LATER,
                                       queue_root=root)
        backend.observations = {**backend.observations, **done(*facts["scope"])}
        run(root, backend, passes=passes, max_launches=max_launches)
        ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
        facts["successor"] = _learned_money(ledger["campaigns"]["campaign-2"])
        facts["site_left_after_successor"] = job_ceiling.site_screening_remaining(root, site_id="site-job-1")
    else:
        facts["site_left"] = job_ceiling.site_screening_remaining(root, site_id="site-job-1")
    state = json.loads((root / coordinator.LEDGER_FILENAME).read_text())["campaigns"]["campaign-1"]
    return facts, backend, state


def _other_campaign_holds(amount):
    def setup(root):
        other = queue_plan(campaign_id="campaign-0", cost=amount, count=3)
        coordinator.enqueue_screening_campaign(other, queue_root=root)
        for row in other["candidates"]:
            assert coordinator._reserve(root, other, row) is not None
    return setup


ROUND4 = {
    # The reviewer's shape: both learned rows fail on the harness, retried after the next deploy.
    "harness_failure": dict(count=2, observations={"candidate-000": HARNESS_FAIL, "candidate-001": HARNESS_FAIL}),
    # Row A fails on the harness and settles at $1; B and C are held for the deploy.
    "harness_failure_settled_siblings_held": dict(count=3, observations={"candidate-000": HARNESS_FAIL},
                                                  max_launches=1),
    # A deploy lands while the learned rows are still queued (nothing launched yet).
    "deploy_mid_run": dict(count=3, observations={}, passes=1, max_launches=0),
    # The owner closed the run (the source was revoked) after one learned row and while the other ran.
    "owner_closed_run": dict(count=2, observations={"candidate-000": LEARNED_DONE, "candidate-001": REVOKED},
                             source_current=False),
    # Another campaign at the site keeps $15 of a $21 limit: one $6 learned row fits, the other waits on it.
    "site_limit_binding": dict(count=2, observations=done("candidate-000", "candidate-001"),
                               setup=_other_campaign_holds("5.00"), cost="6.00", site_limit="21"),
    # Everything completes and settles: the row runs.
    "all_completed_and_settled": dict(count=2, observations=done("candidate-000", "candidate-001")),
    # The typical rigid job: three $4 rows run and settle, two more close for the plan budget. A money close is
    # final for the row (owner, 2026-11-11), so it runs on the $3 left; a deploy still plans the closed two again.
    "typical_budget_closed": dict(count=3, waitlisted=2,
                                  observations=done(*(f"candidate-{index:03}" for index in range(5)))),
    # A budget close beside a harness failure: the failure is retried after a deploy, so the row never runs.
    "budget_closed_and_harness_failure": dict(count=3, waitlisted=2, observations={
        **done("candidate-000", "candidate-001", "candidate-003", "candidate-004"), "candidate-002": HARNESS_FAIL}),
}


def _round4_plan(settings, *, control_row):
    count, cost = settings["count"], settings.get("cost", "4.00")
    value = queue_plan(count=count, cost=cost)
    value = with_control(value, cost="3.00") if control_row else value
    return _with_waitlisted(value, settings.get("waitlisted", 0))


def _with_waitlisted(value, count):
    """``count`` more $4.00 rows the plan budget waitlisted, before the control row (the typical rigid job)."""
    if not count:
        return value
    value = deepcopy(value)
    learned = sum(not control.is_control(row) for row in value["candidates"])
    extra = deepcopy(queue_plan(count=learned + count, cost="1.00"))["candidates"][learned:]
    for row in extra:
        row.update(status="waitlisted", blockers=[WAITLIST_BLOCKER], max_cost_usd="4.00")
        row["cost_quote"]["maximum_cost_usd"] = "4.00"
        row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
    rows = value["candidates"]
    position = next((index for index, row in enumerate(rows) if control.is_control(row)), len(rows))
    value["candidates"] = rows[:position] + extra + rows[position:]
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


@pytest.mark.parametrize("shape", sorted(ROUND4))
def test_with_and_without_the_row_every_learned_row_replan_and_successor_are_identical(tmp_path, settles, shape,
                                                                                         monkeypatch):
    settings = ROUND4[shape]
    if settings.get("source_current") is False:
        monkeypatch.setattr(coordinator, "_source_still_current", lambda _plan: False)
    if "site_limit" in settings:
        monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal(settings["site_limit"]))
    options = {key: settings[key] for key in ("passes", "max_launches", "setup") if key in settings}
    control_result = closed(_summary(3, 1, scripted=True, planned=3))
    without, plain, _ = _observe(tmp_path / "without", _round4_plan(settings, control_row=False),
                                 dict(settings["observations"]), **options)
    with_row, backend, state = _observe(tmp_path / "with", _round4_plan(settings, control_row=True),
                                        {**settings["observations"], CONTROL: control_result}, **options)
    ran = CONTROL in backend.order
    # The row runs only when the campaign without it ended completed, with nothing planned again but rows
    # closed for money.
    assert ran is (shape in {"all_completed_and_settled", "typical_budget_closed"})
    if ran:
        assert without["status"] == "completed" and without["self_scope"] is None
    # The one difference: what the row really spent, in the site's money (and the successor's, when there is one).
    money_keys = {"site_left", "holds", "site_left_for_successor", "site_left_after_successor"} if ran else set()
    assert {key: value for key, value in with_row.items() if key not in money_keys} == {
        key: value for key, value in without.items() if key not in money_keys}
    assert [key for key in backend.order if key != CONTROL] == plain.order
    if ran:
        # Settled at $1 of its $3.00 cap: only that is less for the site afterwards, for a successor too.
        for key in money_keys - {"holds"}:
            if key in without:
                assert Decimal(without[key]) - Decimal(with_row[key]) == Decimal("1"), key
    else:
        assert state["candidates"][CONTROL]["status"] != "pending"
        # Never held a reservation, so the re-plan and the successor saw exactly what they would without it.
        assert "attempt" not in state["candidates"][CONTROL]


def test_a_withdrawn_site_never_reserves_the_row(tmp_path, summaries, monkeypatch):
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    backend = PerRow(done("candidate-000", control_result=closed(_summary(3, 1, scripted=True, planned=3))))
    run(tmp_path, backend, passes=1)  # the learned row launches before the withdrawal
    monkeypatch.setattr(coordinator, "site_withdrawal_state", lambda _site: "withdrawn")
    run(tmp_path, backend, passes=6)
    ledger = json.loads((tmp_path / coordinator.LEDGER_FILENAME).read_text())
    row = ledger["campaigns"]["campaign-1"]["candidates"][CONTROL]
    assert CONTROL not in backend.order and "attempt" not in row and row["status"] in {"blocked", "unavailable"}


def test_a_learned_row_active_again_keeps_the_row_from_deciding():
    value = with_control(queue_plan(count=1))
    ledger = {"campaigns": {"campaign-1": {"site_id": "site-job-1", "candidates": {
        "candidate-000": {"status": "running"}, CONTROL: {"status": "pending", "known_good_control": True}}}}}
    decision = control.control_gate(ledger, "campaign-1", value, plans=lambda *_: value,
                                    committed=lambda *_a, **_k: Decimal(0), site_limit=Decimal(30),
                                    final=lambda: pytest.fail("decided before the learned row ended"))
    assert decision == "wait" and ledger["campaigns"]["campaign-1"]["candidates"][CONTROL]["status"] == "pending"
    # Once it ends but the campaign is not final, the row closes as not run; it never waits on anything else.
    ledger["campaigns"]["campaign-1"]["candidates"]["candidate-000"]["status"] = "blocked"
    decision = control.control_gate(ledger, "campaign-1", value, plans=lambda *_: value,
                                    committed=lambda *_a, **_k: Decimal(0), site_limit=Decimal(30),
                                    final=lambda: False)
    row = ledger["campaigns"]["campaign-1"]["candidates"][CONTROL]
    assert decision == "closed" and row["status"] == "unavailable"
    assert control.control_outcome(value["candidates"][-1], row) == control.INTERRUPTED


def test_the_row_takes_the_sibling_harness_hold(tmp_path, summaries):
    value = with_control(queue_plan(count=1))
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    ledger = coordinator._ledger(tmp_path)
    state = ledger["campaigns"]["campaign-1"]
    state["candidates"]["candidate-000"].update(status="blocked", blockers=["beta_screening_native_result_missing_or_invalid"])
    coordinator._save(tmp_path, ledger)
    if not coordinator._holds_siblings(state["candidates"]["candidate-000"]):
        pytest.skip("this harness code does not hold siblings")
    assert coordinator._reserve(tmp_path, value, value["candidates"][-1]) is None
    row = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"][CONTROL]
    assert row["status"] == "blocked" and row["blockers"] == [coordinator.HARNESS_HOLD_BLOCKER]


def test_the_typical_job_closes_the_row_for_budget_when_three_dollars_no_longer_fit(tmp_path, summaries, monkeypatch):
    # No settlement here: another job's finished $5 run and the three $4 rows hold $17 of a $19 site limit,
    # so the $3.00 row cannot fit and ends at once for budget.
    monkeypatch.setattr(coordinator, "SITE_LIMIT", Decimal("19"))
    other = queue_plan(campaign_id="campaign-0", cost="5.00", count=1)
    coordinator.enqueue_screening_campaign(other, queue_root=tmp_path)
    assert coordinator._reserve(tmp_path, other, other["candidates"][0]) is not None
    value = _with_waitlisted(with_control(queue_plan(count=3, cost="4.00"), cost="3.00"), 2)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = PerRow(done(*(f"candidate-{index:03}" for index in range(5)),
                          control_result=closed(_summary(3, 1, scripted=True, planned=3))))
    run(tmp_path, backend, passes=12, max_launches=2)
    publication = next(value for value in coordinator.pending_publications(queue_root=tmp_path)
                       if value["campaign_id"] == "campaign-1")
    row = _rows(publication)[CONTROL]
    assert [key for campaign, key in backend.launched if campaign == "campaign-1"] == [
        "candidate-000", "candidate-001", "candidate-002"]
    assert CONTROL not in backend.order and row["setup_quality_flags"] == ["known_good_control_unavailable:budget"]
