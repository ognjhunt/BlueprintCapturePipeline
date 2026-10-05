"""A release that edits FindAll tool text must not wedge a FindAll-pinned row in flight.

observe() asserts the stale-caller fence before its try block. The registry check belongs
inside the try, where a changed registry routes to the existing cancel path.
"""
import copy
from datetime import timedelta

import pytest

from tests.test_daily_research_findall import (  # noqa: F401 - shared fixtures
    DAY,
    NOW,
    isolated,
    profile_session,
    runtime,
)
from tools.daily_research import findall
from tools.daily_research.runner import Runner


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
