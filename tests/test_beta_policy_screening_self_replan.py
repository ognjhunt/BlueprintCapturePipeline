"""A campaign moves and ends by itself: no deploy is needed to plan a free refusal again, wait for
money evidence, or tell the site that it is held."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_holds as holds
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline.account_monthly_spend import LIMIT_REACHED, REPORT_INVALID, REPORT_STALE
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication
from tests.test_beta_policy_screening_offer_shortage import SHORTAGE, _due_now
from tests.test_beta_policy_screening_replan import HARNESS_FAILED, NEW, OLD, successor
from tests.test_beta_policy_screening_worker import prepared_source_queue

REFUSED = {"status": "blocked", "provider_allocation_performed": False, "no_allocation_proven": True}
LOST = RuntimeError("provider response lost")


def _pass(root, observation, **options):
    backend = Backend(observation)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend, **options)
    return backend


def _row(root, campaign_id="campaign-1", candidate_id="candidate-000"):
    return coordinator._ledger(coordinator._root(root))["campaigns"][campaign_id]["candidates"][candidate_id]


def _edit(root, campaign_id, change, candidate_id="candidate-000"):
    queue = coordinator._root(root)
    ledger = coordinator._ledger(queue)
    change(ledger["campaigns"][campaign_id]["candidates"][candidate_id])
    coordinator._save(queue, ledger)


def _hours_ago(hours):
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# --- no GPU offer: planned again after a cool-off, bounded --------------------------------------


def _exhaust_offers(root, monkeypatch):
    """Every search of one row finds no offer: the row closes blocked, with nothing spent."""
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_RETRY_CAP", 1)
    _due_now(monkeypatch)
    for _ in range(2):
        _pass(root, SHORTAGE)


def _search_ended(root, campaign_id, hours):
    _edit(root, campaign_id, lambda row: row["attempt"].update(closed_at=_hours_ago(hours)))


def test_a_row_out_of_gpu_offers_is_planned_again_after_the_cool_off_without_a_deploy(tmp_path, monkeypatch):
    """Before, ``beta_screening_gpu_capacity_unavailable`` stayed until the next deploy planned it again."""
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _exhaust_offers(tmp_path, monkeypatch)
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == ("blocked", [coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER])
    assert _row(tmp_path)["attempt"]["closed_at"]
    # Not during the cool-off.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    _search_ended(tmp_path, "campaign-1", hours=coordinator.OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS / 3600 + 1)
    # After it, at the plan's own commit.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == ["candidate-000"]
    coordinator.supersede_campaign("campaign-1", successor(plan(), commit=OLD), implementation_commit=OLD,
                                   queue_root=tmp_path)
    again = publication(tmp_path, "campaign-1b")
    assert [row["status"] for row in again["candidates"]] == ["pending"]
    # Nothing was spent by the searches, and nothing is reserved until the new row is reserved.
    assert again["budget"]["committed_upper_bound_usd"] == "0"


def test_the_offer_shortage_re_plan_stops_at_its_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_REPLAN_LIMIT", 2)
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS", 0)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    campaigns = ["campaign-1", "campaign-1b", "campaign-1c"]
    for current, following in zip(campaigns, campaigns[1:], strict=False):
        _exhaust_offers(tmp_path, monkeypatch)
        assert coordinator.replan_scope(current, implementation_commit=OLD, queue_root=tmp_path) == ["candidate-000"]
        coordinator.supersede_campaign(current, successor(plan(), campaign_id=following, commit=OLD),
                                       implementation_commit=OLD, queue_root=tmp_path)
    _exhaust_offers(tmp_path, monkeypatch)
    # Two re-plans were made; the third shortage stays, at this commit.
    assert coordinator.replan_scope("campaign-1c", implementation_commit=OLD, queue_root=tmp_path) is None
    assert publication(tmp_path, "campaign-1c")["budget"]["committed_upper_bound_usd"] == "0"
    # A deploy still plans it again, as it always did for a refusal that spent nothing.
    assert coordinator.replan_scope("campaign-1c", implementation_commit=NEW, queue_root=tmp_path) == ["candidate-000"]


def test_a_closed_job_window_ends_the_self_re_plan_and_leaves_a_deploys_re_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS", 0)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _exhaust_offers(tmp_path, monkeypatch)
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == ["candidate-000"]
    # Nothing but the row's own due time asks for this re-plan.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path,
                                    self_replan=False) is None
    before = publication(tmp_path)
    assert coordinator.close_self_replan("campaign-1", queue_root=tmp_path) is True
    assert coordinator.close_self_replan("campaign-1", queue_root=tmp_path) is False
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == ["candidate-000"]
    assert publication(tmp_path) == before  # no row, budget or revision changed


# --- an expired quote: planned again with a fresh one, bounded ----------------------------------


def test_a_row_whose_quote_expired_is_planned_again_at_the_same_commit_bounded(tmp_path, monkeypatch):
    """A plan's quotes last a day. Before, an expired one needed an operator to re-seal the catalog."""
    monkeypatch.setattr(coordinator, "QUOTE_EXPIRED_REPLAN_LIMIT", 1)
    expired = {"campaign-1": True}
    monkeypatch.setattr(coordinator, "_quote_current", lambda _candidate: not expired["campaign-1"])
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _pass(tmp_path, {"status": "running"}, max_launches=2)
    rows = publication(tmp_path)["candidates"]
    assert [(row["status"], row["blockers"]) for row in rows] == [
        ("deferred", [coordinator.QUOTE_EXPIRED_BLOCKER])] * 2
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD), implementation_commit=OLD,
                                   queue_root=tmp_path)
    assert [row["status"] for row in publication(tmp_path, "campaign-1b")["candidates"]] == ["pending", "pending"]
    # The fresh quote expires as well: the bound is reached and the row stays deferred, as it did before.
    _pass(tmp_path, {"status": "running"}, max_launches=2)
    assert coordinator.replan_scope("campaign-1b", implementation_commit=OLD, queue_root=tmp_path) is None
    assert publication(tmp_path, "campaign-1b")["budget"]["committed_upper_bound_usd"] == "0"


def _shortage_beside_an_expired_quote(root, monkeypatch):
    """candidate-000 ran out of GPU offers just now; candidate-001's quote then expired."""
    expired = {"now": False}
    monkeypatch.setattr(coordinator, "_quote_current",
                        lambda candidate: candidate["candidate_id"] != "candidate-001" or not expired["now"])
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=root)
    _exhaust_offers(root, monkeypatch)
    expired["now"] = True
    _pass(root, {"status": "running"})
    assert [(row["status"], row["blockers"]) for row in publication(root)["candidates"]] == [
        ("blocked", [coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER]), ("deferred", [coordinator.QUOTE_EXPIRED_BLOCKER])]


def test_a_due_sibling_never_pulls_a_row_past_its_own_cool_off(tmp_path, monkeypatch):
    _shortage_beside_an_expired_quote(tmp_path, monkeypatch)
    # The expired quote is due; the shortage row's two hours have not passed.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == ["candidate-001"]
    _search_ended(tmp_path, "campaign-1", hours=coordinator.OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS / 3600 + 1)
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]


def test_a_due_sibling_never_pulls_a_row_past_its_own_bound_and_a_deploy_takes_it_up(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_REPLAN_LIMIT", 0)  # its self re-plans are used up
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS", 0)
    _shortage_beside_an_expired_quote(tmp_path, monkeypatch)
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == ["candidate-001"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD), implementation_commit=OLD,
                                       queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-001"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    # A machine fault beside it is no reason either, and the row is not lost: the next deploy plans it again.
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-001", "candidate-000"]


# --- the account's monthly limit: unreadable evidence waits, a reached limit is a money stop ------


@pytest.mark.parametrize("blockers", [
    [REPORT_STALE], [REPORT_INVALID],
    # The G1 lane's allocator wraps a refusal raised at the paid admission.
    [coordinator.G1_ADMISSION_WRAPPER, REPORT_STALE],
    # Beside a stale spend lock it is the same wait: one guard refreshes both.
    ["paid_resource_admission_has_blockers", "paid_resource_admission_not_admitted",
     "spend_admission:spend_admission_lock_stale", REPORT_STALE],
])
def test_an_unreadable_account_report_waits_and_is_read_again_holding_the_reservation(tmp_path, blockers):
    """Before, a report a few minutes too old closed the row as blocked until the next deploy."""
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend({**REFUSED, "blockers": blockers})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == (
        "preparing", [coordinator.CAPACITY_WAIT_BLOCKER, coordinator.CONTROL_WAIT_BLOCKER])
    budget = publication(tmp_path)["budget"]
    assert budget["spent_upper_bound_usd"] == "0" and budget["reserved_usd"] == "4"
    state = _row(tmp_path)
    assert state["spend_lock_retries"] == 1 and state["attempt"]["generation"] == 1
    assert state["attempt_history"][0]["reservation_status"] == "released"
    # Not due yet: the next pass does not ask again.
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 1


def test_the_account_report_wait_shares_the_stale_locks_bound_and_then_keeps_the_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "SPEND_LOCK_RETRY_CAP", 2)
    _due_now(monkeypatch)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _pass(tmp_path, {**REFUSED, "blockers": ["spend_admission_lock_stale"]})
    backend = Backend({**REFUSED, "blockers": [REPORT_STALE]})
    for _ in range(3):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 2  # one wait was left under the shared cap, then the refusal stands
    row = publication(tmp_path)["candidates"][0]
    assert (row["status"], row["blockers"]) == ("blocked", [REPORT_STALE])
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "0"


@pytest.mark.parametrize("blockers", [
    [LIMIT_REACHED], [LIMIT_REACHED, REPORT_STALE], [coordinator.G1_ADMISSION_WRAPPER, LIMIT_REACHED]])
def test_a_reached_account_limit_is_a_money_stop_that_never_retries_and_is_held_by_name(tmp_path, blockers):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    backend = Backend({**REFUSED, "blockers": blockers})
    for _ in range(4):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    # Each candidate asked once and was refused before anything was created: no wait, no loop.
    assert backend.launches == 2
    rows = publication(tmp_path)["candidates"]
    assert [row["status"] for row in rows] == ["blocked", "blocked"] and LIMIT_REACHED in rows[0]["blockers"]
    budget = publication(tmp_path)["budget"]
    assert budget["committed_upper_bound_usd"] == "0" and "spend_lock_retries" not in _row(tmp_path)
    # The campaign says why it stopped, for the hold protocol.
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=tmp_path) == LIMIT_REACHED
    assert holds.campaign_hold(LIMIT_REACHED) == (holds.STALL_HOLD_REASON, LIMIT_REACHED)
    assert holds.STALL_HOLD_REASON in holds.REASONS
    # Nothing plans it again at this commit; a deploy does, once, as for any refusal that spent nothing.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]


def test_a_same_commit_re_plan_never_takes_a_money_stopped_row_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "OFFER_SHORTAGE_REPLAN_COOLOFF_SECONDS", 0)
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _exhaust_offers(tmp_path, monkeypatch)                      # candidate-000: no offer
    _pass(tmp_path, {**REFUSED, "blockers": [LIMIT_REACHED]})   # candidate-001: the account limit
    assert [row["status"] for row in publication(tmp_path)["candidates"]] == ["blocked", "blocked"]
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == ["candidate-000"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), keep={"candidate-000"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    # The next deploy takes the money-stopped row up again from the chain.
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]


def test_the_g1_lane_names_why_its_paid_admission_refused(tmp_path, monkeypatch):
    """The G1 allocator recorded only that the admission raised, so a stale report and a reached
    limit read the same."""
    from blueprint_pipeline import native_g1_beta_paid as paid
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    from tests.test_native_g1_beta_paid import dispatch, setup

    args, _, _, identity, _, _ = setup(tmp_path, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise PaidResourceAdmissionBlocked([REPORT_STALE])

    monkeypatch.setattr(paid, "require_paid_resource_admission", refuse)
    monkeypatch.setattr(paid, "run_arena_native_control_vast", lambda **_: pytest.fail("refused admission spent"))
    result = dispatch(args, identity)
    assert result["status"] == "blocked" and result["provider_mutations_performed"] == 0
    assert result["blockers"] == [coordinator.G1_ADMISSION_WRAPPER, REPORT_STALE]
    assert coordinator._money_evidence_stale(result) == "account_monthly_spend_report_unreadable"


# --- a campaign that cannot move says so; its reservation stays open ------------------------------


def test_a_row_unresolved_past_the_bound_is_held_by_name_and_its_reservation_stays_open(tmp_path, monkeypatch):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _pass(tmp_path, LOST)
    assert publication(tmp_path)["status"] == "reconciliation_required"
    since = _row(tmp_path)[coordinator.RECONCILIATION_SINCE_FIELD]
    # Inside the bound it is still being reconciled: nothing to say yet.
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=tmp_path) is None
    _edit(tmp_path, "campaign-1", lambda row: row.update(
        {coordinator.RECONCILIATION_SINCE_FIELD: _hours_ago(coordinator.RECONCILIATION_HOLD_AFTER_SECONDS / 3600 + 1)}))
    before = (publication(tmp_path)["budget"], _row(tmp_path)["attempt"])
    assert before[0]["reserved_usd"] == "4" and before[1]["reservation_status"] == "reserved"
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=tmp_path) == coordinator.RECONCILIATION_STALLED_DETAIL
    # Saying so released and settled nothing, and reconciliation goes on.
    assert (publication(tmp_path)["budget"], _row(tmp_path)["attempt"]) == before
    monkeypatch.setattr(coordinator, "_capacity_retry_ready", lambda _attempt: True)
    still = _pass(tmp_path, LOST)
    assert still.reconciliations == 1 and still.launches == 0
    assert _row(tmp_path)["attempt"]["reservation_status"] == "reserved"
    assert since  # the clock started when the row first became unresolved


def test_the_unresolved_clock_ends_when_the_row_resolves(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _pass(tmp_path, LOST)
    assert coordinator.RECONCILIATION_SINCE_FIELD in _row(tmp_path)
    _pass(tmp_path, {"status": "running"})
    assert _row(tmp_path)["status"] == "running" and coordinator.RECONCILIATION_SINCE_FIELD not in _row(tmp_path)
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=tmp_path) is None
    assert coordinator.campaign_hold_detail("campaign-unknown", queue_root=tmp_path) is None


def test_a_harness_failure_alone_owes_no_campaign_hold(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    _pass(tmp_path, HARNESS_FAILED)
    assert coordinator.campaign_hold_detail("campaign-1", queue_root=tmp_path) is None


def _world(tmp_path, monkeypatch):
    world = prepared_source_queue(tmp_path, monkeypatch)
    return world, {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}


def test_the_worker_publishes_a_hold_for_a_stalled_campaign_and_releases_it_when_it_moves(tmp_path, monkeypatch):
    world, kwargs = _world(tmp_path, monkeypatch)
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert adopted["status"] == "enqueued" and not holds.pending_holds(queue_root=world["screening_queue"])
    root, campaign_id = world["screening_queue"], adopted["campaign_id"]
    backend = Backend(LOST)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend)
    assert backend.launches == 1
    ledger = coordinator._ledger(coordinator._root(root))
    candidate_id = next(key for key, row in ledger["campaigns"][campaign_id]["candidates"].items()
                        if row["status"] == "reconciliation_required")
    # Still inside the bound: the campaign speaks for itself.
    [early] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert "evaluation_hold" not in early and "campaign_hold" not in early
    _edit(root, campaign_id, lambda row: row.update({coordinator.RECONCILIATION_SINCE_FIELD: _hours_ago(7)}),
          candidate_id)
    reserved = publication(root, campaign_id)["budget"]
    assert reserved["reserved_usd"] != "0"
    [stalled] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert stalled["campaign_hold"] == coordinator.RECONCILIATION_STALLED_DETAIL
    assert stalled["evaluation_hold"] == {"status": "held", "reason_code": holds.STALL_HOLD_REASON, "revision": 1,
                                          "change": "recorded"}
    [held] = holds.pending_holds(queue_root=root)
    # A reason the WebApp accepts (its closed set), with the real cause as the detail.
    assert held["reason_code"] in {"robot_free_probe_failed", "robot_free_probe_not_completed",
                                   "robot_free_probe_unavailable_for_task", "team_scene_packet_blocked"}
    assert (held["status"], held["detail_code"]) == ("held", coordinator.RECONCILIATION_STALLED_DETAIL)
    assert held["source"]["scene_intent_id"] == world["requests"][0]["intent_id"]
    # The reservation is untouched: nothing released, nothing settled.
    assert publication(root, campaign_id)["budget"] == reserved
    attempt = coordinator._ledger(coordinator._root(root))["campaigns"][campaign_id]["candidates"][candidate_id]["attempt"]
    assert attempt["reservation_status"] == "reserved" and "settled_cost_usd" not in attempt
    # The same decision again is the same bytes.
    [again] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert again["evaluation_hold"]["change"] == "unchanged"
    # The provider answers at last: the campaign moves, and the hold is released.
    monkeypatch.setattr(coordinator, "_capacity_retry_ready", lambda _attempt: True)
    _edit(root, campaign_id, lambda row: row["attempt"].pop("reconcile_not_before", None), candidate_id)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root,
                                        executor=Backend({"status": "running"}))
    [moving] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert moving["evaluation_hold"] == {"status": "released", "reason_code": None, "revision": 2,
                                         "change": "recorded"}


def test_the_worker_holds_a_source_whose_planning_keeps_failing_and_releases_it_once_planned(tmp_path, monkeypatch):
    """``website_job_ceiling_unavailable`` was retried every minute and the job read "in progress" for good."""
    from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
    from blueprint_pipeline import team_scene_packet_worker as packets

    world, kwargs = _world(tmp_path, monkeypatch)
    readable = job_ceiling.read_job_ceiling_summary

    def unreadable(**_kwargs):
        raise ValueError(job_ceiling.UNAVAILABLE)

    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", unreadable)
    [first] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert first["blockers"] == [job_ceiling.UNAVAILABLE] and "evaluation_hold" not in first
    state = world["screening_queue"] / "sources" / (packets.request_key(world["requests"][0]) + ".json")
    since = packets._read_json(state)["blocked_since"]
    # The same refusal on the next pass keeps its start; still inside the bound, nothing is said.
    [second] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert packets._read_json(state)["blocked_since"] == since and "evaluation_hold" not in second
    assert not holds.pending_holds(queue_root=world["screening_queue"])
    # Past the bound it is held by name; planning is still tried on every pass.
    coordinator._write(state, {**packets._read_json(state), "blocked_since": _hours_ago(7)})
    [stalled] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert stalled["blockers"] == [job_ceiling.UNAVAILABLE]
    assert stalled["evaluation_hold"] == {"status": "held", "reason_code": holds.STALL_HOLD_REASON, "revision": 1,
                                          "change": "recorded"}
    [held] = holds.pending_holds(queue_root=world["screening_queue"])
    assert (held["reason_code"], held["detail_code"]) == ("team_scene_packet_blocked", job_ceiling.UNAVAILABLE)
    assert held["reason_code"] in holds.REASONS
    # No campaign exists, so there is no reservation to touch.
    assert coordinator._ledger(coordinator._root(world["screening_queue"]))["campaigns"] == {}
    # The ceiling answers: the campaign is planned and the hold released.
    monkeypatch.setattr(job_ceiling, "read_job_ceiling_summary", readable)
    [planned] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert planned["status"] == "enqueued"
    assert planned["evaluation_hold"] == {"status": "released", "reason_code": None, "revision": 2,
                                          "change": "recorded"}
    assert "blocked_since" not in packets._read_json(state)


def test_a_rights_refusal_or_a_closed_window_is_never_a_stall_hold():
    long_ago = _hours_ago(48)
    assert holds.stalled_source_hold("beta_screening_scene_profile_unavailable", long_ago) == (
        holds.STALL_HOLD_REASON, "beta_screening_scene_profile_unavailable")
    # The website ends a withdrawn or revoked job itself; a closed window is the job's own end.
    for code in ("beta_screening_current_capture_consent_required", "team_scene_capture_withdrawn",
                 holds.WINDOW_CLOSED):
        assert holds.stalled_source_hold(code, long_ago) is None
    # Inside the bound, or with no readable start, nothing is held.
    assert holds.stalled_source_hold("beta_screening_scene_profile_unavailable", _hours_ago(1)) is None
    assert holds.stalled_source_hold("beta_screening_scene_profile_unavailable", "not a time") is None
    # A different refusal starts its own clock.
    now = datetime.now(timezone.utc)
    previous = {"blockers": ["a_refusal"], "blocked_since": long_ago}
    assert holds.blocked_since(previous, ["a_refusal"], now=now) == long_ago
    assert holds.blocked_since(previous, ["another_refusal"], now=now) == now.isoformat()
    assert holds.blocked_since(None, ["a_refusal"], now=now) == now.isoformat()


def test_the_worker_ends_a_self_re_plan_once_the_jobs_window_closed(tmp_path, monkeypatch):
    world, kwargs = _world(tmp_path, monkeypatch)
    [adopted] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    root, campaign_id = world["screening_queue"], adopted["campaign_id"]
    # Every funded row ran out of GPU offers long ago.
    queue = coordinator._root(root)
    ledger = coordinator._ledger(queue)
    for row in ledger["campaigns"][campaign_id]["candidates"].values():
        if row["status"] == "pending":
            row.update(status="blocked", blockers=[coordinator.OFFER_SHORTAGE_EXHAUSTED_BLOCKER],
                       attempt={"attempt_id": "screen-" + "0" * 32, "generation": 12, "maximum_cost_usd": "4",
                                "reservation_status": "released", "provider_allocation_observed": False,
                                "closed_at": _hours_ago(5)})
    coordinator._save(queue, ledger)
    commit = world["deployed"]["commit"]
    assert coordinator.replan_scope(campaign_id, implementation_commit=commit, queue_root=root)
    monkeypatch.setattr(worker, "job_window_closed", lambda _context: True)
    [closed] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert closed["blockers"] == [holds.WINDOW_CLOSED] and "evaluation_hold" not in closed
    # It ended once: the next pass plans nothing and reads the campaign as it is.
    assert coordinator.replan_scope(campaign_id, implementation_commit=commit, queue_root=root) is None
    [after] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert after["status"] == "already_enqueued" and after["campaign_id"] == campaign_id
    assert coordinator.latest_campaign(campaign_id, queue_root=root) == campaign_id


def test_the_job_window_reads_the_current_sources_own_deadline():
    now = datetime.now(timezone.utc).timestamp()
    assert worker.job_window_closed({"execution_deadline_at_ms": (now - 60) * 1000}) is True
    assert worker.job_window_closed({"execution_deadline_at_ms": (now + 3600) * 1000}) is False
    assert worker.job_window_closed({"capture_rights": {"consent_expires_at_epoch": now - 60}}) is True
    # Unknown or unreadable is not closed: the launch preflight still decides.
    assert worker.job_window_closed({}) is False
    assert worker.job_window_closed({"execution_deadline_at_ms": "soon"}) is False
