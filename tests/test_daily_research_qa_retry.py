"""503 replay lifecycle against the real fenced bridge; no network/credentials."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.test_daily_research_operator_canary import (
    NOW,
    canary,
    continued_qa_receipt,
    fixture,
    recovered_baseline,
)
from tools.daily_research import qa_retry, recovery
from tools.daily_research.consumer import Consumer
from tools.daily_research.runner import Refusal, digest

_fixture = fixture  # Imported pytest fixture, retained without a second setup implementation.
ERROR = {"stage": "provider_submission", "class": "InternalServerError", "http_status": 503,
         "code": "service_unavailable_error", "request_id": "req_synthetic503receipt"}


class Transient503(Exception):
    pass


def retained_503(fixture, monkeypatch):
    bridge, ledger, api, cache, row = recovered_baseline(fixture, monkeypatch)
    clock = {"now": NOW + timedelta(hours=1)}
    def set_clock(now):
        clock["now"] = now
        bridge.call("test_clock", now=int(now.timestamp() * 1000))
    set_clock(clock["now"])
    canary.authorize_recovered_qa(bridge, continued_qa_receipt(row), clock=lambda: clock["now"])
    def fail(sid, event, key, day, request_digest, deadline_ms):
        bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        raise TimeoutError("synthetic original lost reply")
    api.qa_input = fail
    cfg = canary.render.configured(bridge, cache)
    consumer = Consumer(ledger, cfg, api, clock=lambda: clock["now"])
    consumer.active_day = canary.DAY
    assert consumer.step()["state"] == "qa_input_unresolved"
    with ledger.lock():
        row = ledger.get(canary.DAY)
        row["qa"]["input_error_receipt"] = deepcopy(ERROR)
        ledger.put(row)
    set_clock(clock["now"] + timedelta(seconds=601))
    assert consumer.step()["state"] == "qa_cancel_pending"
    old = ledger.get(canary.DAY)
    provider = object.__new__(canary.CanaryProvider)
    provider.ledger, provider.clock, provider.stopped = ledger, lambda: clock["now"], lambda: False
    provider.safe = lambda _row: None  # Separate tests cover live cost GET admission.
    provider.get, provider.listing = api.get, api.listing
    posts = []
    provider.api = SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=lambda *a, **kw: posts.append((a, kw)))))
    def submit(*args):
        try:
            return provider.qa_retry_input(*args)
        finally:
            api.qa_input_phase = getattr(provider, "qa_input_phase", "preconditions")
    api.qa_retry_input = submit
    real_receipt = recovery.repair_error_receipt
    monkeypatch.setattr(recovery, "repair_error_receipt", lambda error, stage:
                        {**ERROR, "stage": stage} if isinstance(error, Transient503) else real_receipt(error, stage))
    return bridge, ledger, api, cache, consumer, provider, clock, set_clock, old, posts


def arm(ledger, api, clock):
    with ledger.lock():
        row = ledger.get(canary.DAY)
        qa_retry.authorize(row, qa_retry.reconcile(api, ledger, row), clock["now"])
        canary.qa_deadline(row, {})
        ledger.put(row)
    return row


def accept(api, ledger, clock):
    api.qa_exists = True
    row = ledger.get(canary.DAY)
    api.qa_result = {"schema_version": "blueprint.research-qa.v1", "packet_digest": row["packet_digest"],
        "crm_digest": row["qa"]["crm_digest"], "source_support_verified": True,
        "accepted_keys": [c["candidate_key"] for c in row["packet"]["candidates"]],
        "summary": "Synthetic supported QA https://plant.example/tasks; interest unknown.",
        "checks": [{"candidate_key": c["candidate_key"], "source_support_verified": True,
                    "duplicate": False, "reason": "Synthetic exact task"} for c in row["packet"]["candidates"]]}
    original = api.listing
    completed = int(clock["now"].timestamp()) + 1
    def listing(resource, sid=None):
        values = original(resource, sid)
        for turn in values if resource == "turns" else []:
            if turn["id"] == "turn_qa":
                turn["completed_at"] = completed
        return values
    api.listing = listing


@pytest.mark.parametrize("lost_reply", [False, True])
def test_same_key_retry_reaches_qa_and_publication_without_new_session(fixture, monkeypatch, lost_reply):
    bridge, ledger, api, cache, _consumer, provider, clock, tick, old, posts = retained_503(fixture, monkeypatch)
    raw = ledger.read_bytes(canary.DAY + "-artifact.json")
    body = qa_retry.original_event(ledger, old)
    try:
        def post(*args, **kwargs):
            posts.append((args, kwargs))
            if not lost_reply and len(posts) == 1:
                raise Transient503()
            accept(api, ledger, clock)
            provider.listing = api.listing
            if lost_reply:
                raise TimeoutError("accepted, response lost")
        provider.api.sessions.events.create = post
        result = canary.retry_qa_submission(bridge, cache, api_factory=lambda *_: api,
            clock=lambda: clock["now"], sleep=lambda seconds: tick(clock["now"] + timedelta(seconds=seconds)),
            expected_row_digest=digest(old), expected_request_id=ERROR["request_id"])
        assert result["state"] == "completed", result
        assert len(posts) == (1 if lost_reply else 2)
        assert all(args == (old["session_id"],) and kwargs == {"events": [body], "idempotency_key": old["run_key"] + ":qa"}
                   for args, kwargs in posts)
        final = ledger.get(canary.DAY)
        assert final["qa_retry_continuation"]["previous_qa"] == old["qa"]
        assert final["qa_continuation"] == old["qa_continuation"]
        assert final["qa"]["deadline_ms"] == old["qa"]["deadline_ms"]
        assert ledger.read_bytes(canary.DAY + "-artifact.json") == raw
        assert len(api.payloads) == 1 and not getattr(api, "repair_calls", [])
        assert all(value["receipt"]["readback_verified"] for value in final["delivery"].values())
        receipt = deepcopy(final["qa_retry_continuation"])
        repeat = canary.retry_qa_submission(bridge, cache, api_factory=lambda *_: api,
            clock=lambda: clock["now"] + timedelta(hours=1), sleep=lambda _: None)
        assert repeat["state"] == "completed" and ledger.get(canary.DAY)["qa_retry_continuation"] == receipt
        assert len(posts) == (1 if lost_reply else 2)
    finally:
        bridge.close()


@pytest.mark.parametrize("outcome", ["crash", "timeout", "changed_error"])
def test_unknown_attempt_never_reposts_after_restart(fixture, monkeypatch, outcome):
    bridge, ledger, api, _cache, consumer, provider, clock, tick, old, posts = retained_503(fixture, monkeypatch)
    try:
        phase = arm(ledger, api, clock)["qa_retry_continuation"]
        def post(*args, **kwargs):
            posts.append((args, kwargs))
            if outcome == "crash":
                raise SystemExit("synthetic process crash after claim")
            raise TimeoutError() if outcome == "timeout" else Refusal("other_error")
        provider.api.sessions.events.create = post
        tick(clock["now"] + timedelta(seconds=6))
        if outcome == "crash":
            with pytest.raises(SystemExit):
                consumer.step()
        else:
            consumer.step()
        for _ in range(3):
            tick(clock["now"] + timedelta(seconds=20))
            consumer.step()
        assert len(posts) == 1
        row = ledger.get(canary.DAY)
        with ledger.lock(), pytest.raises(Refusal, match="qa_retry_input_not_admitted"):
            bridge.call("qa_retry_check", day=canary.DAY, request_digest=row["qa"]["request_digest"],
                        deadline_ms=int(canary.qa_deadline(row, {}).timestamp()*1000), number=1)
        tick(canary.qa_deadline(row, {}) + timedelta(seconds=1))
        consumer.step()
        assert api.cancellations[-1][1] == old["run_key"] + ":qa:retry-phase"
        assert ledger.get(canary.DAY)["qa_retry_continuation"] == phase
        assert ledger.get(canary.DAY)["qa_retry_continuation"]["previous_qa"] == old["qa"]
        assert len(posts) == 1
    finally:
        bridge.close()


@pytest.mark.parametrize("change", ["turn", "message", "required_action", "artifact", "cancel_uncertain"])
def test_retry_phase_rejects_unreconciled_acceptance_or_cancel(fixture, monkeypatch, change):
    bridge, ledger, api, cache, _consumer, _provider, clock, _tick, old, posts = retained_503(fixture, monkeypatch)
    try:
        if change == "cancel_uncertain":
            with ledger.lock():
                old["qa"]["cancel_reply_unresolved"] = True
                ledger.put(old)
        elif change == "required_action":
            api.actions = [{"type": "environment_connection"}]
        else:
            listing = api.listing
            def changed(resource, sid=None):
                values = listing(resource, sid)
                if resource == {"turn": "turns", "message": "items", "artifact": "artifacts"}[change]:
                    values.append({"id": "new_input_or_effect", "turn_id": None, "status": "cancelled", "path": "/workspace/outputs/daily-research-qa.json"})
                return values
            api.listing = changed
        with pytest.raises(Refusal, match="original_submission_not_admitted"):
            canary.retry_qa_submission(bridge, cache, api_factory=lambda *_: api, clock=lambda: clock["now"],
                                      expected_row_digest=digest(old), expected_request_id=ERROR["request_id"])
        assert not ledger.get(canary.DAY).get("qa_retry_continuation") and not posts
    finally:
        bridge.close()


@pytest.mark.parametrize("change", ["origin", "stop", "deadline", "disabled", "new_message"])
def test_retry_final_guards_after_claim_prevent_post(fixture, monkeypatch, change):
    bridge, ledger, api, _cache, consumer, provider, clock, tick, _old, posts = retained_503(fixture, monkeypatch)
    try:
        arm(ledger, api, clock)
        tick(clock["now"] + timedelta(seconds=6))
        original_call = bridge.call
        def late(op, **fields):
            result = original_call(op, **fields)
            if op == "qa_retry_check":
                if change == "origin":
                    original_call("test_origin_change")
                elif change == "stop":
                    provider.stopped = lambda: True
                elif change == "deadline":
                    clock["now"] += timedelta(seconds=601)
                elif change == "disabled":
                    control = original_call("control")
                    original_call("configure", value={**control, "enabled": False})
                else:
                    listing = provider.listing
                    provider.listing = lambda resource, sid=None: listing(resource, sid) + ([{"id": "accepted_input"}] if resource == "items" else [])
            return result
        monkeypatch.setattr(bridge, "call", late)
        consumer.step()
        assert not posts
        row = ledger.get(canary.DAY)
        receipt = row["qa"]["input_retries"][0]["error_receipt"]
        assert receipt["stage"] == "preconditions" and receipt["class"] == "Refusal"
        assert not qa_retry.transient(receipt)
    finally:
        bridge.close()


def test_retry_after_beyond_phase_stops_and_phase_cannot_be_rearmed(fixture, monkeypatch):
    bridge, ledger, api, _cache, consumer, _provider, clock, _tick, old, posts = retained_503(fixture, monkeypatch)
    try:
        with ledger.lock():
            old["qa"]["input_error_receipt"]["retry_after_seconds"] = 86400
            ledger.put(old)
        first = arm(ledger, api, clock)
        assert consumer.step()["state"] == "qa_blocked" and not posts
        with ledger.lock():
            row = ledger.get(canary.DAY)
            assert qa_retry.authorize(row, {"clear": True}, clock["now"] + timedelta(hours=1)) == row
            row["qa_retry_continuation"]["started_at"] = (clock["now"] + timedelta(hours=1)).isoformat()
            with pytest.raises(Refusal, match="qa_retry_phase_already_bound"):
                ledger.put(row)
        assert ledger.get(canary.DAY)["qa_retry_continuation"] == first["qa_retry_continuation"]
    finally:
        bridge.close()


def test_accepted_reply_persistence_failure_observes_without_repost_or_cancel(fixture, monkeypatch):
    bridge, ledger, api, _cache, consumer, provider, clock, tick, _old, posts = retained_503(fixture, monkeypatch)
    try:
        arm(ledger, api, clock)
        def post(*args, **kwargs):
            posts.append((args, kwargs))
            accept(api, ledger, clock)
            provider.listing = api.listing
        provider.api.sessions.events.create = post
        put, failed = ledger.put, {"once": False}
        def fail_reply(row):
            if row["qa"]["state"] == "qa_running" and not failed["once"]:
                failed["once"] = True
                raise Refusal("firestore_bridge_unavailable")
            return put(row)
        monkeypatch.setattr(ledger, "put", fail_reply)
        cancellations = deepcopy(api.cancellations)
        tick(clock["now"] + timedelta(seconds=6))
        assert consumer.step()["state"] == "qa_input_unresolved"
        row = ledger.get(canary.DAY)
        assert row["qa"]["input_retries"][0]["error_receipt"]["stage"] == "reply_persistence"
        assert api.cancellations == cancellations
        consumer.step()
        assert ledger.get(canary.DAY)["qa"]["state"] == "validated"
        assert len(posts) == 1 and api.cancellations == cancellations
    finally:
        bridge.close()


def test_two_transient_slots_are_the_complete_retry_limit(fixture, monkeypatch):
    bridge, ledger, api, _cache, consumer, provider, clock, tick, _old, posts = retained_503(fixture, monkeypatch)
    try:
        arm(ledger, api, clock)
        def post(*args, **kwargs):
            posts.append((args, kwargs))
            raise Transient503()
        provider.api.sessions.events.create = post
        tick(clock["now"] + timedelta(seconds=6))
        consumer.step()
        tick(clock["now"] + timedelta(seconds=14))
        consumer.step()
        assert len(posts) == 1  # The second slot's 15s floor is enforced.
        tick(clock["now"] + timedelta(seconds=2))
        consumer.step()
        for _ in range(3):
            tick(clock["now"] + timedelta(seconds=20))
            consumer.step()
        assert len(posts) == 2 and len(ledger.get(canary.DAY)["qa"]["input_retries"]) == 2
        assert not ledger.get(canary.DAY)["qa"].get("turn_id")
    finally:
        bridge.close()
