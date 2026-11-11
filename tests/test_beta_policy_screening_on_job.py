"""Test chosen policies on one stored job (plan 1.1), never the same pair twice (1.2), the standing allowance (1.6).

Real sealed scene requests, the real planner, coordinator and settlement; only host reads are stubbed
(the scene's planning inputs, the WebApp's task context and job-ceiling read, the completed launch run).
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_on_job as on_job
from blueprint_pipeline import beta_policy_screening_standing_test as standing
from blueprint_pipeline import beta_policy_screening_tested_pairs as tested
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import native_result
from tests.test_beta_policy_screening_settlement import Measured as _Measured, write_run
from tests.test_beta_policy_screening_worker import prepared_source_queue

SITE = "site-job1"


class Measured(_Measured):
    """The settlement suite's finished Vast run, its ledger capped at the attempt's own reservation."""

    def _finish(self, candidate, attempt_dir, prepared):
        self.instance += 1
        write_run(Path(attempt_dir), dict(prepared), instance_id=self.instance,
                  hard_cap=float(candidate["max_cost_usd"]))
        return {"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                "result": native_result(self.value, candidate)}


def _ceiling(*, allowed=None, remaining=40):
    from tests.test_beta_policy_screening_job_ceiling import ceiling_summary

    def read(*, context, capture_root, extras=None):
        if extras is not None and allowed is not None:
            extras[standing.PERMISSION_FIELD] = allowed
        return ceiling_summary(remaining)
    return read


@pytest.fixture()
def world(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_capabilities as capabilities
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
    from tests.test_native_task_arena_policy_canary_session import _activation

    value = prepared_source_queue(tmp_path, monkeypatch)
    original = capabilities.build_planning_inputs
    contract = _activation()["task_success_contract"]["contract_digest"]

    def planning_inputs(**kwargs):
        # The session fixture's task contract, so a scored result binds to these plans.
        from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog

        out = original(**{**kwargs, "catalog": load_default_catalog()})
        out["scene"]["task_success_contract_digest"] = contract
        for binding in out["capabilities"]["candidate_bindings"].values():
            binding["execution_spec"]["task_success_contract_digest"] = contract
            binding["admission_digest"] = canonical_digest(binding["execution_spec"])
        return out

    monkeypatch.setattr(capabilities, "build_planning_inputs", planning_inputs)
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", _ceiling())
    monkeypatch.setenv(standing.ALLOWANCE_FILE_ENV, str(tmp_path / "standing-allowance.json"))
    value["allowance"] = tmp_path / "standing-allowance.json"
    return value


def allow_standing(world, monkeypatch, *, total="20.00", allowed=True):
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    world["allowance"].write_text(json.dumps({
        "schema_version": standing.ALLOWANCE_SCHEMA, "monthly_total_usd": total, "approved_by": "owner",
        "approved_at": "2026-11-11", "authorization_reference": "owner fixture"}))
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", _ceiling(allowed=allowed))


def request(funding="site_allowance", ids=("pi05_droid",)):
    return {"site_id": SITE, "candidate_ids": list(ids), "funding": funding,
            "authorization_reference": "operator fixture"}


def run(world, body, request_id="req-1"):
    return on_job.screen_policies_on_job(body, request_id=request_id, queue_root=world["screening_queue"],
                                         packet_queue_root=world["packet_queue"])


def ledger(world):
    return json.loads((world["screening_queue"] / coordinator.LEDGER_FILENAME).read_text())


def plan_of(world, campaign_id):
    return json.loads((world["screening_queue"] / "plans" / f"{campaign_id}.json").read_text())


def finish(world, campaign_id):
    value = plan_of(world, campaign_id)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=world["screening_queue"],
                                        executor=Measured(value), max_launches=10)
    return ledger(world)["campaigns"][campaign_id]


def test_one_policy_on_one_job_from_the_site_allowance_runs_and_settles(world):
    result = run(world, request())
    assert result["status"] == "enqueued" and result["selected"] == ["pi05_droid"]
    plan = plan_of(world, result["campaign_id"])
    # Exactly the named candidate; a limited plan carries no scripted reference row.
    assert [row["candidate_id"] for row in plan["candidates"]] == ["pi05_droid"]
    assert "standing_test" not in plan and plan["budget_basis"] == "website_job_ceiling"
    state = finish(world, result["campaign_id"])
    row = state["candidates"]["pi05_droid"]
    assert row["status"] == "completed" and row["attempt"].get("settled_cost_usd") == "0.45", row["attempt"].get("settlement_held")
    assert "funding" not in state
    assert coordinator._budget(ledger(world), SITE)["spent_upper_bound_usd"] == "0.45"
    receipt = json.loads((world["screening_queue"] / "job_tests" / "req-1.json").read_text())
    assert receipt["status"] == "enqueued" and receipt["campaign_id"] == result["campaign_id"]
    # The same request again is the same campaign; nothing is planned twice.
    assert run(world, request())["campaign_id"] == result["campaign_id"]


def test_one_policy_from_the_standing_allowance_is_outside_the_site_money(world, monkeypatch):
    allow_standing(world, monkeypatch)
    result = run(world, request("standing_test"))
    assert result["status"] == "enqueued", result
    plan = plan_of(world, result["campaign_id"])
    record = plan["standing_test"]
    assert record["policy_cap_usd"] == "2.00" and record["monthly_total_usd"] == "20.00"
    assert "website_job_ceiling" not in plan and plan["budget_usd"] == plan["maximum_planned_cost_usd"] == "2.00"
    state = finish(world, result["campaign_id"])
    assert state["funding"] == "standing_test"
    assert state["candidates"]["pi05_droid"]["status"] == "completed"
    after = ledger(world)
    # The site's allowance and the job's ceiling never see it; the campaign and the standing ledger do.
    assert coordinator._budget(after, SITE)["committed_upper_bound_usd"] == "0"
    assert coordinator._budget(after, SITE, campaign_id=result["campaign_id"])["spent_upper_bound_usd"] == "0.45"
    usage = standing.standing_usage(world["screening_queue"], month=standing.month_of())
    assert usage["month_usd"] == Decimal("0.45") and usage["by_policy"][(SITE, "pi05_droid")] == Decimal("0.45")


@pytest.mark.parametrize(("change", "reason"), [
    ("unset", standing.MONTHLY_TOTAL_UNSET), ("invalid", standing.MONTHLY_TOTAL_INVALID),
    ("absent", standing.NOT_ALLOWED), ("false", standing.NOT_ALLOWED), ("string", standing.NOT_ALLOWED),
    ("month_spent", standing.MONTHLY_TOTAL_INSUFFICIENT)])
def test_standing_test_refusals_spend_nothing(world, monkeypatch, change, reason):
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    if change != "unset":
        allow_standing(world, monkeypatch, total="1.00" if change == "month_spent" else "20.00")
    if change == "invalid":
        world["allowance"].write_text(json.dumps({"schema_version": standing.ALLOWANCE_SCHEMA}))
    if change in {"absent", "false", "string"}:
        monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary",
                            _ceiling(allowed={"absent": None, "false": False, "string": "true"}[change]))
    result = run(world, request("standing_test"))
    if change == "month_spent":
        assert result["status"] == "refused" and result["reason"] == on_job.NOTHING_FUNDED
        assert result["candidates"]["pi05_droid"]["blockers"] == [reason]
    else:
        assert result["status"] == "refused" and result["reason"] == reason
    assert result["spent_usd"] == "0"
    assert not list((world["screening_queue"] / "plans").glob("*.json"))


def test_the_policy_cap_is_per_policy_per_job_across_standing_tests(world, monkeypatch):
    allow_standing(world, monkeypatch)
    first = run(world, request("standing_test"), "req-1")
    assert first["status"] == "enqueued"
    # Another policy still has its whole $2.00; the same policy's money is held while its test is queued.
    second = run(world, request("standing_test", ("groot_n17_droid",)), "req-2")
    assert second["status"] == "enqueued" and second["candidates"]["groot_n17_droid"]["max_cost_usd"] == "2.00"
    usage = standing.standing_usage(world["screening_queue"], month=standing.month_of())
    assert usage["month_usd"] == Decimal("4.00")


def test_window_closed_refuses_the_site_allowance_only(world, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_worker as worker

    monkeypatch.setattr(worker, "job_window_closed", lambda context: True)
    result = run(world, request())
    assert result["status"] == "refused" and result["reason"] == on_job.WINDOW_CLOSED
    allow_standing(world, monkeypatch)
    assert run(world, request("standing_test"), "req-2")["status"] == "enqueued"


def test_an_allowance_that_cannot_hold_the_quote_refuses(world, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling

    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", _ceiling(remaining=1))
    result = run(world, request())
    assert result["status"] == "refused" and result["reason"] == on_job.NOTHING_FUNDED
    assert result["candidates"]["pi05_droid"]["status"] == "deferred"
    assert not list((world["screening_queue"] / "plans").glob("*.json"))


@pytest.mark.parametrize("stop", ["withdrawn", "unknown", "consent"])
def test_withdrawal_and_consent_refuse_before_planning(world, monkeypatch, stop):
    from blueprint_pipeline import screening_site_withdrawal

    allow_standing(world, monkeypatch)
    if stop == "consent":
        world["context"]["capture_rights"]["consent_revoked"] = True
    else:
        monkeypatch.setattr(screening_site_withdrawal, "site_withdrawal_state", lambda site_id: stop)
    for funding in on_job.FUNDING_MODES:
        result = run(world, request(funding), "req-" + funding)
        assert result["status"] == "refused" and result["reason"] == {
            "withdrawn": on_job.SITE_WITHDRAWN, "unknown": on_job.WITHDRAWAL_UNKNOWN,
            "consent": "beta_screening_current_capture_consent_required"}[stop]
    assert world["planning_calls"] == []


def test_withdrawal_after_enqueue_stops_a_standing_test_before_any_reservation(world, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as module

    allow_standing(world, monkeypatch)
    result = run(world, request("standing_test"))
    monkeypatch.setattr(module, "site_withdrawal_state", lambda site_id: "withdrawn")
    backend = Measured(plan_of(world, result["campaign_id"]))
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=world["screening_queue"],
                                        executor=backend, max_launches=10)
    assert backend.launches == 0
    row = ledger(world)["campaigns"][result["campaign_id"]]["candidates"]["pi05_droid"]
    assert row["status"] not in {"pending", "preparing", "launching", "running"} and not row.get("attempt")


@pytest.mark.parametrize("cold", ["launch_run", "scene_workspace"])
def test_a_cold_job_names_its_restore_operation_and_never_restores(world, cold, tmp_path):
    from blueprint_pipeline.control_plane_evidence_offload import POINTER_SUFFIX

    if cold == "launch_run":
        launch = next(iter(world["results"]))
        launch.parent.mkdir(parents=True, exist_ok=True)
        if launch.exists():
            launch.rmdir()
        (launch.parent / f"{launch.name}{POINTER_SUFFIX}").write_text("{}")
    else:
        (tmp_path / "partition" / "scenes" / f"{SITE}.retired.v1.json").write_text("{}")
    result = run(world, request())
    assert result["status"] == "refused"
    assert (result["reason"], result["restore_operation"]) == (
        (on_job.LAUNCH_RUN_OFFLOADED, "restore-launch-run") if cold == "launch_run"
        else (on_job.SCENE_WORKSPACE_RETIRED, "restore-scene-workspace"))
    assert world["validated"] == []


def test_a_pair_is_never_tested_twice(world, monkeypatch):
    first = run(world, request(), "req-1")
    # Still queued: in progress, not tested again.
    queued = run(world, request(), "req-2")
    assert queued["reason"] == on_job.ALREADY_TESTED and queued["skipped"] == {"in_progress": ["pi05_droid"]}
    finish(world, first["campaign_id"])
    again = run(world, request(), "req-3")
    assert again["reason"] == on_job.ALREADY_TESTED and again["skipped"] == {"already_tested": ["pi05_droid"]}
    allow_standing(world, monkeypatch)
    assert run(world, request("standing_test"), "req-4")["reason"] == on_job.ALREADY_TESTED
    # A mixed request tests only the new pair.
    mixed = run(world, request(ids=("pi05_droid", "groot_n17_droid")), "req-5")
    assert mixed["selected"] == ["groot_n17_droid"] and mixed["skipped"] == {"already_tested": ["pi05_droid"]}
    record = json.loads((world["screening_queue"] / "tested_pairs" / f"{SITE}.json").read_text())
    assert [row["candidate_id"] for row in record["finished"]] == ["pi05_droid"]


def test_an_infrastructure_failure_is_not_a_finished_pair():
    assert not tested.finished({"status": "blocked", "blockers": ["beta_screening_native_result_invalid"],
                                "summary": {"episodes_scored": 0}})
    assert tested.finished({"status": "completed", "summary": {"episodes_scored": 10}})
    assert not tested.finished({"status": "completed", "summary": {"episodes_scored": 0, "episode_outcomes": [
        {"outcome": "void"}]}})


def _complete_every_selected_row(world, campaign_id):
    """Mark a campaign's selected rows finished with a scored result, as a settled run leaves them."""
    from tests.test_beta_policy_screening_settlement import reseal

    def change(value):
        for row in value["campaigns"][campaign_id]["candidates"].values():
            if row["status"] == "pending":
                row.update(status="completed", blockers=[], summary={"episodes_scored": 10, "episodes_succeeded": 0})
    reseal(world["screening_queue"], change)


def test_a_catalog_change_plans_only_new_pairs_and_leaves_earlier_results_untouched(world, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_team as team
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline.beta_policy_candidate_catalog import seal_catalog

    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [first] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    before_plan = plan_of(world, first["campaign_id"])
    funded = sorted(row["candidate_id"] for row in before_plan["candidates"] if row["status"] == "selected")
    _complete_every_selected_row(world, first["campaign_id"])
    earlier = deepcopy(ledger(world)["campaigns"][first["campaign_id"]])
    site_before = coordinator._budget(ledger(world), SITE)
    original = team.catalog_for_profile

    def changed(profile, base):  # a catalog revision: every row the same, a new catalog digest
        value = original(profile, base)
        return seal_catalog({**value, "revision": value["revision"] + "-next"})
    monkeypatch.setattr(team, "catalog_for_profile", changed)
    [second] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert second["status"] == "enqueued" and second["campaign_id"] != first["campaign_id"]
    new_plan = plan_of(world, second["campaign_id"])
    planned = {row["candidate_id"] for row in new_plan["candidates"]}
    # The finished pairs are not planned again; every other row is, as before.
    assert not planned & set(funded)
    assert planned | set(funded) >= {row["candidate_id"] for row in before_plan["candidates"]}
    assert ledger(world)["campaigns"][first["campaign_id"]] == earlier
    assert coordinator._budget(ledger(world), SITE) == site_before


def test_without_earlier_results_the_plan_bytes_are_unchanged(world, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_tested_pairs as pairs
    from blueprint_pipeline import beta_policy_screening_worker as worker

    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [with_filter] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    plan_bytes = (world["screening_queue"] / "plans" / f"{with_filter['campaign_id']}.json").read_bytes()
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
    from tests.test_beta_policy_screening_plan import NOW, inputs, scene

    catalog, capabilities, quotes = inputs()
    plain = build_screening_plan(catalog, scene(), campaign_id="c1", capabilities=capabilities, cost_quotes=quotes,
                                 now=NOW)
    empty = build_screening_plan(catalog, scene(), campaign_id="c1", capabilities=capabilities, cost_quotes=quotes,
                                 now=NOW, finished_pairs=frozenset())
    assert plain == empty
    assert pairs.finished_pair_keys(world["screening_queue"], SITE) == frozenset()
    assert plan_bytes  # the worker's new campaign was planned with an empty filter


def test_every_pair_finished_makes_no_new_campaign_and_holds_nothing(world, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_team as team
    from blueprint_pipeline import beta_policy_screening_worker as worker
    from blueprint_pipeline.beta_policy_candidate_catalog import seal_catalog

    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    monkeypatch.setattr(team, "catalog_for_profile", lambda profile, base: seal_catalog(
        {**base, "candidates": [row for row in base["candidates"] if row["candidate_id"] == "pi05_droid"]}))
    [first] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    _complete_every_selected_row(world, first["campaign_id"])
    monkeypatch.setattr(team, "catalog_for_profile", lambda profile, base: seal_catalog(
        {**base, "revision": base["revision"] + "-next",
         "candidates": [row for row in base["candidates"] if row["candidate_id"] == "pi05_droid"]}))
    [second] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert second["status"] == "already_tested" and not second.get("blockers")
    assert len(list((world["screening_queue"] / "plans").glob("*.json"))) == 1


def test_the_two_dollar_fit_uses_the_reference_rows_rule():
    from blueprint_pipeline.beta_policy_screening_plan import execution_cost_floor, fit_run_to_cap
    from tests.test_beta_policy_screening_plan import admitted, inputs

    catalog, _caps, _quotes = inputs()
    binding, quote = admitted(next(row for row in catalog["candidates"] if row["candidate_id"] == "pi05_droid"),
                              maximum="4.00")
    spec = {**binding["execution_spec"], "max_hourly_rate_usd": "0.80", "hard_ttl_seconds": 14400}
    fitted_spec, fitted = fit_run_to_cap(spec, quote, standing.POLICY_CAP_USD, reason="fixture", suffix="-cap")
    # $0.80/h x 4 h under a $4.00 quote keeps $0.40 of headroom: a 2-hour run at $2.00.
    assert fitted["maximum_cost_usd"] == "2.00" and fitted_spec["hard_ttl_seconds"] == 7200
    assert execution_cost_floor(fitted_spec) == Decimal("1.60")
    assert fit_run_to_cap(spec, quote, Decimal("0.70"), reason="fixture", suffix="-cap") is None


def test_the_plan_field_and_the_standing_module_agree():
    from blueprint_pipeline.beta_policy_screening_plan import STANDING_TEST_FIELD

    assert STANDING_TEST_FIELD == standing.STANDING_TEST_FIELD_NAME == standing.STANDING
    from blueprint_pipeline.beta_policy_screening_job_ceiling import STANDING_PERMISSION_FIELD

    assert STANDING_PERMISSION_FIELD == standing.PERMISSION_FIELD == "standing_tests_allowed"


def _standing_launch(tmp_path, monkeypatch, *, allowed=True, allowance=True, stopped=False):
    from blueprint_pipeline import beta_policy_screening_control_transport as control_transport
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline import beta_policy_screening_worker as service
    from blueprint_pipeline import team_scene_packet_worker as packets
    from tests.test_beta_policy_screening_control_contracts import response
    from tests.test_beta_policy_screening_launch import _preflight_kwargs

    kwargs = _preflight_kwargs(tmp_path, monkeypatch)
    kwargs["plan"]["standing_test"] = {"schema_version": standing.RECORD_SCHEMA}
    # The job's preparation window closed an hour ago.
    window_closed = {"execution_deadline_at_ms": int((time.time() - 3600) * 1000),
                     "capture_rights": {"consent_expires_at_epoch": time.time() + 86400}}
    monkeypatch.setattr(service, "require_current_source", lambda *args, **kw: window_closed)
    monkeypatch.setattr(packets, "resolve_capture_root", lambda source: tmp_path)
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", _ceiling(allowed=allowed))
    monkeypatch.setenv(standing.ALLOWANCE_FILE_ENV, str(tmp_path / "allowance.json"))
    if allowance:
        (tmp_path / "allowance.json").write_text(json.dumps({
            "schema_version": standing.ALLOWANCE_SCHEMA, "monthly_total_usd": "20.00", "approved_by": "owner",
            "approved_at": "2026-11-11", "authorization_reference": "fixture"}))
    seen = []

    def control(*, binding, execution_deadline_epoch=None, **_):
        seen.append(execution_deadline_epoch)
        return response(scope=binding, now=int(time.time()), deadline=int(time.time()) + 3600, stopped=stopped)
    monkeypatch.setattr(control_transport, "request_control", control)
    monkeypatch.setattr("blueprint_pipeline.allocator_invocation.start_invocation", lambda *a, **k: 4242)
    monkeypatch.setattr("blueprint_pipeline.team_candidate_quick10_launch._process_start_ticks", lambda pid: 1)
    return launch.launch_candidate(**kwargs), seen


def test_a_standing_launch_rereads_the_site_permission_and_ignores_the_preparation_window(tmp_path, monkeypatch):
    result, seen = _standing_launch(tmp_path, monkeypatch)
    assert result["status"] == "running"
    # The WebApp's execution control is asked with the consent's expiry, not the closed window.
    assert seen and seen[0] > time.time()


@pytest.mark.parametrize(("change", "code"), [
    ({"allowed": False}, "beta_screening_prelaunch_refused:standing_test_not_allowed_by_site"),
    ({"allowance": False}, "beta_screening_prelaunch_refused:standing_test_monthly_total_unset"),
    # The owner's run closure or a withdrawal reaches the launch as the WebApp's stopped control.
    ({"stopped": True}, "beta_screening_prelaunch_refused:beta_screening_current_source_revoked_before_spawn"),
])
def test_a_standing_launch_is_stopped_like_any_run(tmp_path, monkeypatch, change, code):
    result, _seen = _standing_launch(tmp_path, monkeypatch, **change)
    assert result["status"] == "blocked" and result["blockers"] == [code]
    assert result["provider_allocation_performed"] is False
