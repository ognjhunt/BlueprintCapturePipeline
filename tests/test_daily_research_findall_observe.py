"""A release that edits FindAll tool text must not wedge a FindAll-pinned row in flight.

observe() asserts the stale-caller fence before its try block. The registry check belongs
inside the try, where a changed registry routes to the existing cancel path.
"""
import copy
from datetime import timedelta

import pytest

from tests import test_daily_research_findall as base
from tests.test_daily_research_findall import DAY, NOW, action, profile_session, respond
from tools.daily_research import findall
from tools.daily_research.consumer import Consumer
from tools.daily_research.runner import Refusal, Runner

# The shared pytest fixtures, registered in this module under their own names.
isolated, runtime = base.isolated, base.runtime


@pytest.mark.parametrize("handler_present", [True, False])
def test_edited_findall_registry_routes_an_overdue_row_to_cancel(runtime, monkeypatch, handler_present):
    row, ledger, _client, _state, api = runtime
    session = profile_session(runtime)
    ledger.put(row)
    api.get = lambda resource, rid: copy.deepcopy(session)
    api.listing = lambda resource, sid: [{"id": row["turn_id"], "status": "running"}] if resource == "turns" else []
    cancels = []
    api.cancel = lambda sid, key: cancels.append((sid, key)) or True
    if not handler_present:
        del api.findall_application_tools
    original = findall.tools

    def edited():
        tools = original()
        tools[0] = {**tools[0], "description": tools[0]["description"] + " (edited by a later release)"}
        return tools

    monkeypatch.setattr(findall, "tools", edited)
    runner = object.__new__(Runner)
    runner.ledger, runner.api, runner.config = ledger, api, {}
    runner.clock = lambda: NOW + timedelta(seconds=5000)  # past the research deadline
    runner.stop_requested = lambda: False
    with ledger.lock():
        runner.observe(row)
    # The overdue session is cancelled instead of raising on every tick.
    assert cancels, "observe must reach the cancel path"
    assert ledger.get(DAY)["state"] != "running"


def edit_registry(monkeypatch):
    original = findall.tools

    def edited():
        tools = original()
        tools[0] = {**tools[0], "description": tools[0]["description"] + " (edited by a later release)"}
        return tools

    monkeypatch.setattr(findall, "tools", edited)


def test_edited_findall_registry_routes_running_qa_to_cancel(runtime, monkeypatch):
    row, ledger, _client, _state, api = runtime
    session = profile_session(runtime)
    row.update(state="awaiting_review", qa={"state": "qa_running", "cancel_attempted": False, "turn_id": "turn_qa"})
    ledger.put(row)
    api.get = lambda resource, rid: copy.deepcopy(session)
    cancels = []
    api.cancel = lambda sid, key: cancels.append(key) or True
    edit_registry(monkeypatch)
    consumer = Consumer(ledger, {}, api, clock=lambda: NOW + timedelta(seconds=3000))
    outcomes = []
    for _ in range(3):  # three scheduler ticks
        with ledger.lock():
            try:
                consumer.qa(ledger.get(DAY))
                outcomes.append("returned")
            except Refusal as exc:
                outcomes.append(str(exc))
    # QA is cancelled instead of raising the registry refusal on every tick.
    assert cancels, outcomes
    assert "findall_tool_registry_binding_changed" not in outcomes[1:]


def test_stale_caller_fence_still_refuses_in_research_observe(runtime):
    row, ledger, _client, _state, api = runtime
    profile_session(runtime)
    ledger.put(row)
    stale = copy.deepcopy(row)
    respond(runtime, action())  # A FindAll claim is now durable, so the copy above is stale.
    runner = object.__new__(Runner)
    runner.ledger, runner.api, runner.config = ledger, api, {}
    runner.clock, runner.stop_requested = (lambda: NOW), (lambda: False)
    before = copy.deepcopy(ledger.get(DAY))
    with ledger.lock(), pytest.raises(Refusal, match="findall_tool_stale_owner_row"):
        runner.observe(stale)
    assert ledger.get(DAY) == before
