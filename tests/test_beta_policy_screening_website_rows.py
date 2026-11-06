"""A new website campaign plans only the catalog rows that can run on a website job.

Owner decision 2026-11-05 (P-DT-12): the 24 scripted baselines (never funded on a website job
since 2026-11-04) and the 58 robot × route templates (placeholders until a partner supplies an
artifact) are left out of a new website campaign's plan. The catalog keeps every row and id.
A re-plan names its predecessor's rows, so a campaign made before the change, such as one
running during the deploy, keeps its rows, reservations and settlement. No money changes:
a left-out row is never quoted, reserved or funded.
"""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from decimal import Decimal

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
from blueprint_pipeline import beta_policy_screening_plan as planner
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import (
    DISABLED_BLOCKER, ROUTE_TEMPLATE_BLOCKER, build_job_ceiling_screening_plan, build_screening_plan,
    job_ceiling_record, website_never_planned, website_route_template, website_switched_off,
)
from tests.test_beta_policy_screening_job_ceiling import READ_AT, ceiling_summary
from tests.test_beta_policy_screening_plan import NOW, admitted, scene

SITE = "site-request1"
#: A row a plan lists without any money field: never quoted, reserved or funded.
UNFUNDED_FIELDS = {"candidate_id", "display_name", "configuration_digest", "status", "blockers",
                   "execution_evidence_kind", "reach_rank"}


def host_like(catalog):
    """The host enables the scripted baselines through door registrations; give them runtimes too."""
    rows = deepcopy(catalog["candidates"])
    template = next(row for row in rows if row["candidate_id"] == "pi05_droid")
    for row in rows:
        if row["policy"]["kind"] == "scripted":
            row.update(interface=deepcopy(template["interface"]), runtime=deepcopy(template["runtime"]),
                       supported_task_kinds=["rigid_pick_place"])
            row["eligibility"] = {**row["eligibility"], "enabled": True, "blockers": []}
    return seal_catalog({**catalog, "candidates": rows})


def bound(catalog, maximum="2.00"):
    """Synthetic runtime bindings and quotes for every enabled row (scripted ones included)."""
    bindings, quotes = {}, {}
    for row in catalog["candidates"]:
        if row["eligibility"]["enabled"]:
            bindings[row["candidate_id"]], quotes[row["candidate_id"]] = admitted(row, maximum=maximum)
    return {"candidate_bindings": bindings}, quotes


def every_id(catalog):
    return sorted(row["candidate_id"] for row in catalog["candidates"])


def website_plans(catalog, *, remaining=40, holds="0", **options):
    """A new campaign's plan, and the plan a re-plan naming every row makes (the pre-change shape)."""
    capabilities, quotes = bound(catalog)
    capabilities.update(options.pop("capabilities", {}))
    record = job_ceiling_record(ceiling_summary(remaining), local_holds_usd=Decimal(holds), read_at=READ_AT)
    kwargs = dict(job_ceiling=record, campaign_id="campaign1", capabilities=capabilities, cost_quotes=quotes,
                  now=NOW, **options)
    return (build_job_ceiling_screening_plan(catalog, scene(), **kwargs),
            build_job_ceiling_screening_plan(catalog, scene(), candidate_ids=every_id(catalog), **kwargs))


def test_the_rows_a_website_campaign_leaves_out_are_the_templates_and_scripted_baselines():
    catalog = load_default_catalog()
    left_out = [row for row in catalog["candidates"] if website_never_planned(row)]
    templates = [row for row in left_out if website_route_template(row)]
    scripted = [row for row in left_out if website_switched_off(row)]
    assert (len(catalog["candidates"]), len(left_out), len(templates), len(scripted)) == (136, 82, 58, 24)
    assert not {row["candidate_id"] for row in templates} & {row["candidate_id"] for row in scripted}
    # Every template is a robot × route placeholder: no runtime, interface or task, waiting on a partner.
    assert all(row["deployment"]["relationship"] == "planning_candidate" and row["deployment"]["policy_route_id"]
               and row["runtime"] is None and row["interface"] is None and row["supported_task_kinds"] == []
               and not row["eligibility"]["enabled"] and ROUTE_TEMPLATE_BLOCKER in row["eligibility"]["blockers"]
               for row in templates)
    assert Counter(row["deployment"]["policy_route_id"] for row in templates) == {
        "pi05_custom_checkpoint": 16, "skill_trace": 16, "planning_endpoint": 15, "customer_controller": 8,
        "groot_custom_checkpoint": 1, "act_custom_checkpoint": 1, "runway_praxis1": 1}
    # Every other row, licence-blocked, vendor and hosted ones included, is still planned.
    fresh, everything = website_plans(catalog)
    planned = {row["candidate_id"] for row in fresh["candidates"]}
    assert planned == set(every_id(catalog)) - {row["candidate_id"] for row in left_out} and len(planned) == 54
    assert {"dreamzero_droid", "flux3_action_droid", "trlc_dk1_lerobot_controller", "pi05_trlc_rabc_10466a13",
            "openwam_alpha_franka", "molmoact2_bimanual_yam", "dream1_i2rt_yam_bimanual"} <= planned
    # The catalog is untouched: same digest in both plans, every id still readable.
    assert fresh["catalog_digest"] == everything["catalog_digest"] == catalog["catalog_digest"]
    assert len(everything["candidates"]) == 136


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("remaining,site_remaining", [(40, None), (40, "5.00"), (40, "1.50"), (9, None), (3, "3.00"),
                                                      (0, None)])
def test_leaving_the_rows_out_changes_no_money(host, remaining, site_remaining):
    catalog = load_default_catalog()
    catalog = host_like(catalog) if host else catalog
    left_out = {row["candidate_id"] for row in catalog["candidates"] if website_never_planned(row)}
    # Rank some left-out rows first: the order of the rest is unchanged.
    priority = [*sorted(left_out)[:5], "molmoact2_droid", "groot_n17_droid"]
    fresh, everything = website_plans(catalog, remaining=remaining, site_remaining_usd=site_remaining,
                                      priority=priority, capabilities={"candidate_reach_rank": {}})
    for field in ("budget_usd", "maximum_planned_cost_usd", "website_job_ceiling", "budget_basis", "catalog_digest",
                  "scenario_cells", "screening_harness_revision"):
        assert fresh[field] == everything[field], field
    # Every planned row is the pre-change plan's row, byte for byte, in the same order.
    assert fresh["candidates"] == [row for row in everything["candidates"] if row["candidate_id"] not in left_out]
    # Every left-out row was listed there without a quote, a reservation or a cap.
    dropped = [row for row in everything["candidates"] if row["candidate_id"] in left_out]
    assert len(dropped) == 82
    assert all(row["status"] == "unavailable" and row["blockers"] and set(row) <= UNFUNDED_FIELDS for row in dropped)
    if host:
        # Enabled and bound on the host, the scripted rows were still refused as switched off.
        scripted = [row for row in dropped if row["candidate_id"].startswith("blueprint_scripted_")]
        assert len(scripted) == 24 and all(DISABLED_BLOCKER in row["blockers"] for row in scripted)


def test_the_coordinator_reserves_spends_and_publishes_the_same_money(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "_quote_current", lambda candidate: True)
    catalog = host_like(load_default_catalog())
    fresh, everything = website_plans(catalog, remaining=9)
    selected = [row["candidate_id"] for row in fresh["candidates"] if row["status"] == "selected"]

    class FirstRefusedRestAllocated:
        def __init__(self):
            self.launched = []

        def prepare_candidate(self, *, candidate, **_):
            if candidate["candidate_id"] == selected[0]:
                raise ValueError("beta_screening_static_preflight_blocked:no_offer")
            return {"sealed": "local"}

        def launch_candidate(self, *, candidate, **_):
            self.launched.append(candidate["candidate_id"])
            return {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True}

    results = {}
    for name, value in (("new", fresh), ("old", everything)):
        root = tmp_path / name
        coordinator.enqueue_screening_campaign(value, queue_root=root)
        backend = FirstRefusedRestAllocated()
        holds = []
        for _ in range(4):
            coordinator.process_screening_queue(queue_root=root, executor=backend, require_publication_ack=False,
                                                max_launches=10)
            holds.append(job_ceiling.local_screening_holds(root, site_id=SITE, published_usd=Decimal(0)))
        [publication] = coordinator.pending_publications(queue_root=root)
        ledger = coordinator._ledger(root)
        results[name] = {"launched": backend.launched, "holds": holds, "budget": publication["budget"],
                         "site": coordinator._budget(ledger, SITE),
                         "remaining": job_ceiling.site_screening_remaining(root, site_id=SITE),
                         "status": publication["status"],
                         "rows": {row["candidate_id"]: row for row in publication["candidates"]}}
    new, old = results["new"], results["old"]
    assert new["launched"] == old["launched"] and len(new["launched"]) == 4
    assert new["holds"] == old["holds"] and new["budget"] == old["budget"] and new["site"] == old["site"]
    assert new["remaining"] == old["remaining"] and new["budget"]["spent_upper_bound_usd"] == "8"
    assert new["status"] == old["status"]
    assert set(old["rows"]) - set(new["rows"]) == {row["candidate_id"] for row in catalog["candidates"]
                                                   if website_never_planned(row)}
    assert all(new["rows"][key] == old["rows"][key] for key in new["rows"])


def test_a_template_is_planned_again_once_a_partner_artifact_is_registered():
    catalog = load_default_catalog()
    template, other = [row for row in catalog["candidates"] if website_route_template(row)][:2]
    reference = next(row for row in catalog["candidates"] if row["candidate_id"] == "pi05_droid")
    # Registered: the effective catalog row carries the partner's runtime and interface.
    registered = {**deepcopy(template), "runtime": deepcopy(reference["runtime"]),
                  "interface": deepcopy(reference["interface"]),
                  "eligibility": {**template["eligibility"], "enabled": True, "blockers": []}}
    # A registration that failed: the row keeps the registration's own reason instead.
    failed = {**deepcopy(other), "eligibility": {**other["eligibility"],
                                                 "blockers": ["beta_screening_team_binding_invalid"]}}
    assert website_route_template(template) and website_never_planned(template)
    assert not website_route_template(registered) and not website_never_planned(registered)
    assert not website_route_template(failed) and not website_never_planned(failed)
    # A row waiting on a registration that is not a planning candidate is not a template.
    trlc = next(row for row in catalog["candidates"] if row["candidate_id"] == "trlc_dk1_lerobot_controller")
    assert ROUTE_TEMPLATE_BLOCKER in trlc["eligibility"]["blockers"] and not website_never_planned(trlc)
    rows = [registered, failed, *[deepcopy(row) for row in catalog["candidates"]
                                  if row["candidate_id"] not in {template["candidate_id"], other["candidate_id"]}]]
    resealed = seal_catalog({**catalog, "candidates": rows})
    fresh, _ = website_plans(resealed)
    planned = {row["candidate_id"]: row for row in fresh["candidates"]}
    # Both are listed with their own reasons (this fixture binds no runtime for them); 80 rows stay out.
    assert planned[template["candidate_id"]]["status"] == planned[other["candidate_id"]]["status"] == "unavailable"
    assert "beta_screening_team_binding_invalid" in planned[other["candidate_id"]]["blockers"]
    assert len(fresh["candidates"]) == 56


def test_a_catalog_with_only_left_out_rows_still_plans_and_publishes(tmp_path):
    catalog = load_default_catalog()
    rows = [row for row in catalog["candidates"] if website_never_planned(row)]
    only = seal_catalog({**catalog, "candidates": rows})
    fresh, everything = website_plans(only)
    assert fresh == everything and len(fresh["candidates"]) == 82
    coordinator.enqueue_screening_campaign(fresh, queue_root=tmp_path)
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    assert publication["status"] == "blocked" and publication["budget"]["committed_upper_bound_usd"] == "0"


def test_an_internal_plan_still_lists_every_row():
    catalog = load_default_catalog()
    capabilities, quotes = bound(catalog)
    value = build_screening_plan(catalog, scene(), campaign_id="campaign1", capabilities=capabilities,
                                 cost_quotes=quotes, now=NOW)
    assert len(value["candidates"]) == 136


def _adopt(world):
    [outcome] = worker.enqueue_prepared_scenes(queue_root=world["screening_queue"],
                                               packet_queue_root=world["packet_queue"])["sources"]
    return outcome


class OneAllocates:
    def prepare_candidate(self, **_):
        return {"sealed": "local"}

    def launch_candidate(self, **_):
        return {"status": "blocked", "provider_allocation_performed": True, "provider_zero_verified": True,
                "blockers": ["beta_screening_native_result_missing"]}


def test_a_campaign_planned_before_the_change_keeps_its_rows_through_the_deploy_and_its_replan(tmp_path, monkeypatch):
    """capture-447's campaign was planned with every row; the deploy re-plans it with the same rows."""
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    world = prepared_source_queue(tmp_path, monkeypatch)
    root = world["screening_queue"]
    with monkeypatch.context() as before:
        before.setattr(planner, "website_never_planned", lambda candidate: False)
        first = _adopt(world)["campaign_id"]
    plan_path = root / "plans" / f"{first}.json"
    plan_bytes = plan_path.read_bytes()
    old = json.loads(plan_bytes)
    assert len(old["candidates"]) == 136
    left_out = {row["candidate_id"] for row in load_default_catalog()["candidates"] if website_never_planned(row)}
    # Without a deploy nothing about it changes: same campaign, plan, ledger and publication.
    ledger_bytes = (root / coordinator.LEDGER_FILENAME).read_bytes()
    [publication] = coordinator.pending_publications(queue_root=root)
    replay = _adopt(world)
    assert replay["status"] == "already_enqueued" and replay["campaign_id"] == first
    assert plan_path.read_bytes() == plan_bytes and (root / coordinator.LEDGER_FILENAME).read_bytes() == ledger_bytes
    assert coordinator.pending_publications(queue_root=root) == [publication]
    # One candidate allocates (its money is committed), then this change deploys.
    coordinator.process_screening_queue(queue_root=root, executor=OneAllocates(), require_publication_ack=False)
    spent = coordinator._budget(coordinator._ledger(root), "site-job1")
    world["deployed"]["commit"] = "b" * 40
    replanned = _adopt(world)
    assert replanned["status"] == "enqueued" and replanned["superseded_campaign_id"] == first
    successor = json.loads((root / "plans" / f"{replanned['campaign_id']}.json").read_text())
    ledger = coordinator._ledger(root)
    unspent = [row["candidate_id"] for row in old["candidates"]
               if not coordinator._spent(ledger["campaigns"][first]["candidates"][row["candidate_id"]])]
    # The re-plan lists exactly its predecessor's unspent rows, the left-out kinds included.
    assert sorted(row["candidate_id"] for row in successor["candidates"]) == sorted(unspent)
    assert left_out <= {row["candidate_id"] for row in successor["candidates"]}
    assert all(row["status"] == "unavailable" for row in successor["candidates"] if row["candidate_id"] in left_out)
    # The predecessor's frozen plan is untouched and the site's money carries over unchanged.
    assert plan_path.read_bytes() == plan_bytes
    assert coordinator._budget(ledger, "site-job1") == spent and spent["spent_upper_bound_usd"] == "2"


def test_a_new_campaign_and_its_replan_list_only_rows_that_can_run(tmp_path, monkeypatch):
    from tests.test_beta_policy_screening_worker import prepared_source_queue

    world = prepared_source_queue(tmp_path, monkeypatch)
    root = world["screening_queue"]
    left_out = {row["candidate_id"] for row in load_default_catalog()["candidates"] if website_never_planned(row)}
    first = _adopt(world)["campaign_id"]
    plan = json.loads((root / "plans" / f"{first}.json").read_text())
    assert len(plan["candidates"]) == 54 and not left_out & {row["candidate_id"] for row in plan["candidates"]}
    [publication] = coordinator.pending_publications(queue_root=root)
    assert len(publication["candidates"]) == 54
    coordinator.process_screening_queue(queue_root=root, executor=OneAllocates(), require_publication_ack=False)
    world["deployed"]["commit"] = "b" * 40
    replanned = _adopt(world)
    successor = json.loads((root / "plans" / f"{replanned['campaign_id']}.json").read_text())
    assert replanned["superseded_campaign_id"] == first
    assert not left_out & {row["candidate_id"] for row in successor["candidates"]}
    assert len(successor["candidates"]) == 53  # The one that spent is never planned again.
