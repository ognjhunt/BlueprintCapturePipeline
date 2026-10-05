"""A release that edits FindAll tool text must not wedge a FindAll-pinned row in flight.

observe() asserts the stale-caller fence before its try block. The registry check belongs
inside the try, where a changed registry routes to the existing cancel path.
"""
import copy
import base64
import hashlib
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests import test_daily_research_findall as base
from tests.test_daily_research_findall import DAY, NOW, action, profile_session, respond
from tools.daily_research import findall, recovery, render
from tools.daily_research.consumer import Consumer
from tools.daily_research.runner import Refusal, Runner, digest

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


def prestart_fixture(runtime, monkeypatch, *, failed=False):
    row, ledger, _client, _state, api = runtime
    session = profile_session(runtime)
    raw = b'{"candidates": null}'  # Global invalidity cannot be repaired by item exclusion.
    row.update(state="failed" if failed else "awaiting_review", turn_status="completed",
               artifact_downloaded=True, raw_output_digest=hashlib.sha256(raw).hexdigest(),
               error="synthetic_validation_failure" if failed else None,
               total_runtime_seconds=1800, knowledge_context=None, refresh_policy=None,
               knowledge_context_digest=digest(None), refresh_policy_digest=digest(None),
               packet={"synthetic": True}, packet_digest=digest({"synthetic": True}))
    ledger.write_bytes(DAY + "-artifact.json", raw)
    ledger.put(row)
    get = api.get
    reads, actions = [], []

    def read(resource, rid):
        reads.append(resource)
        return copy.deepcopy(session) if resource == "session" else get(resource, rid)

    api.get = read
    api.listing = lambda *args: actions.append("listing") or []
    api.qa_input = lambda *args: actions.append("qa_input")
    api.repair_input = lambda *args: actions.append("repair_input")
    api.cancel = lambda *args: actions.append("cancel")
    control = {"enabled": True, "workflow": {"enabled": True,
        "qa_authority_reference": "synthetic-qa-authority",
        "publication_authority_reference": "synthetic-publication-authority"}}
    ledger.bridge = SimpleNamespace(call=lambda operation: copy.deepcopy(control))
    monkeypatch.setattr(Consumer, "refresh_crm", lambda self: ({"values": []}, set()))
    return raw, reads, actions


@pytest.mark.parametrize("handler_present", [True, False])
def test_edited_findall_registry_blocks_qa_before_input(runtime, monkeypatch, handler_present):
    raw, reads, actions = prestart_fixture(runtime, monkeypatch)
    row, ledger, client, _state, api = runtime
    before = copy.deepcopy(ledger.get(DAY))
    if not handler_present:
        del api.findall_application_tools
    edit_registry(monkeypatch)
    outcomes, saved = [], []
    for _ in range(3):
        consumer = Consumer(ledger, {}, api, clock=lambda: NOW + timedelta(seconds=10))
        with ledger.lock():
            try:
                outcomes.append(consumer.qa(ledger.get(DAY)))
            except Refusal as exc:
                outcomes.append(str(exc))
        saved.append(ledger.get(DAY))
    assert outcomes == [None, None, None], outcomes
    qa = saved[0]["qa"]
    assert qa["state"] == "qa_blocked" and qa["error"] == "findall_tool_registry_binding_changed"
    assert qa["input_error_receipt"]["code"] == qa["error"]
    assert qa["input_error_receipt"]["stage"] == "preconditions"
    assert saved[0] == saved[1] == saved[2]
    assert {k: v for k, v in saved[-1].items() if k != "qa"} == before
    assert reads.count("session") == 1 and actions == [] and client.posts == 0
    assert ledger.read_bytes(DAY + "-artifact.json") == raw


@pytest.mark.parametrize("handler_present", [True, False])
def test_edited_findall_registry_blocks_repair_before_input(runtime, monkeypatch, handler_present):
    raw, reads, actions = prestart_fixture(runtime, monkeypatch, failed=True)
    row, ledger, client, _state, api = runtime
    before = copy.deepcopy(ledger.get(DAY))
    if not handler_present:
        del api.findall_application_tools
    edit_registry(monkeypatch)
    outcomes, saved = [], []
    for _ in range(3):
        loop = recovery.RepairLoop(ledger, {}, api, clock=lambda: NOW + timedelta(seconds=10))
        try:
            outcomes.append(loop.step(DAY)["state"])
        except Refusal as exc:
            outcomes.append(str(exc))
        saved.append(ledger.get(DAY))
    assert outcomes == ["failed"] * 3, outcomes
    assert saved[0] == saved[1] == saved[2]
    assert len(saved[-1]["validation_repairs"]) == 1
    revision = saved[-1]["validation_repairs"][0]
    assert revision["state"] == "no_progress" and revision["error"] == "findall_tool_registry_binding_changed"
    assert revision["input_attempted"] is False
    assert revision["input_error_receipt"]["code"] == revision["error"]
    assert revision["input_error_receipt"]["stage"] == "preconditions"
    assert saved[-1]["original_validation_failure"]["error"] == before["error"]
    assert saved[-1]["original_validation_failure"]["raw_output_sha256"] == before["raw_output_digest"]
    assert reads.count("session") == 1 and actions == [] and client.posts == 0
    assert ledger.read_bytes(DAY + "-artifact.json") == raw


@pytest.mark.parametrize("phase", ["qa", "repair"])
def test_prestart_keeps_other_session_refusals_fail_closed(runtime, monkeypatch, phase):
    prestart_fixture(runtime, monkeypatch, failed=phase == "repair")
    _row, ledger, _client, _state, api = runtime
    get = api.get

    def wrong_session(resource, rid):
        result = get(resource, rid)
        if resource == "session":
            result["id"] = "session_foreign"
        return result

    api.get = wrong_session
    edit_registry(monkeypatch)
    with pytest.raises(Refusal, match="^agent_qa_session_binding_mismatch$"):
        if phase == "qa":
            with ledger.lock():
                Consumer(ledger, {}, api, clock=lambda: NOW).qa(ledger.get(DAY))
        else:
            recovery.RepairLoop(ledger, {}, api, clock=lambda: NOW).step(DAY)
    saved = ledger.get(DAY)
    assert not saved.get("qa") and not saved.get("validation_repairs")


def test_edited_registry_never_swallows_stale_qa_caller(runtime, monkeypatch):
    prestart_fixture(runtime, monkeypatch)
    row, ledger, _client, _state, api = runtime
    stale = copy.deepcopy(row)
    respond(runtime, action())
    before = copy.deepcopy(ledger.get(DAY))
    edit_registry(monkeypatch)
    with ledger.lock(), pytest.raises(Refusal, match="^findall_tool_stale_owner_row$"):
        Consumer(ledger, {}, api, clock=lambda: NOW).qa(stale)
    assert ledger.get(DAY) == before


def test_repair_ownership_refusal_precedes_any_error_persistence(runtime, monkeypatch):
    prestart_fixture(runtime, monkeypatch, failed=True)
    _row, ledger, _client, state, api = runtime
    before = copy.deepcopy(ledger.get(DAY))
    state["lease"] = False
    edit_registry(monkeypatch)
    with pytest.raises(Refusal, match="^findall_owner_current_lease_required$"):
        recovery.RepairLoop(ledger, {}, api, clock=lambda: NOW).step(DAY)
    assert ledger.get(DAY) == before


def test_prestart_registry_refusal_has_only_the_precise_allowlisted_receipt():
    code = "findall_tool_registry_binding_changed"
    assert recovery.repair_error_receipt(Refusal(code), "preconditions") == {
        "stage": "preconditions", "class": "Refusal", "code": code,
        "http_status": None, "request_id": None}
    assert recovery.repair_error_receipt(Refusal(code + " private diagnostic"), "preconditions")["code"] is None


def test_render_exposes_prestart_repair_block_as_terminal(runtime, monkeypatch, tmp_path):
    raw, reads, actions = prestart_fixture(runtime, monkeypatch, failed=True)
    _row, ledger, _client, _state, api = runtime
    api.client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(render, "FirestoreLedger", lambda bridge: ledger)
    monkeypatch.setattr(render, "configured", lambda *args: {})
    monkeypatch.setattr(render, "datetime", SimpleNamespace(now=lambda zone: NOW + timedelta(seconds=10)))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    edit_registry(monkeypatch)
    outcomes = [render.consume_workflow(ledger.bridge, tmp_path, day=DAY, api_factory=lambda *args: api)
                for _ in range(3)]
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0]["state"] == "validation_repair_blocked"
    assert outcomes[0]["error"] == "findall_tool_registry_binding_changed"
    assert outcomes[0]["feedback"] == ledger.get(DAY)["original_validation_failure"]["feedback"]
    assert reads.count("session") == 1 and actions == []
    assert ledger.read_bytes(DAY + "-artifact.json") == raw


def blocked_repair_snapshot(runtime, monkeypatch):
    raw, _reads, _actions = prestart_fixture(runtime, monkeypatch, failed=True)
    _row, ledger, _client, _state, api = runtime
    edit_registry(monkeypatch)
    row = recovery.RepairLoop(ledger, {}, api, clock=lambda: NOW).step(DAY)
    return {"row": row, "files": {"artifact": base64.b64encode(raw).decode()},
            "missing_files": []}


def test_unstarted_repair_exports_without_fabricated_input(runtime, monkeypatch, tmp_path):
    snapshot = blocked_repair_snapshot(runtime, monkeypatch)
    bridge = SimpleNamespace(call=lambda *args, **kwargs: copy.deepcopy(snapshot))
    result = render.export_snapshot(bridge, DAY, tmp_path / "export")
    assert result["missing_files"] == []
    assert (tmp_path / "export" / (DAY + "-artifact.json")).read_bytes() == runtime[1].read_bytes(DAY + "-artifact.json")


@pytest.mark.parametrize("field,value", [
    ("state", "running"), ("error", "other_refusal"), ("input_attempted", True),
    ("input_attempted", None), ("input_attempted", 0), ("number", True),
    ("turn_id", "turn_sent"), ("request_digest", "0" * 64),
    ("turn_id", None), ("cancel_attempted", False), ("artifact_digest", "0" * 64),
    ("input_file", DAY + "-repair-1-input.json"), ("deadline_ms", 0),
    ("input_error_receipt", {"stage": "provider_submission", "code": "findall_tool_registry_binding_changed"}),
])
def test_unstarted_repair_export_cannot_skip_submitted_input_bindings(runtime, monkeypatch, tmp_path, field, value):
    snapshot = blocked_repair_snapshot(runtime, monkeypatch)
    snapshot["row"]["validation_repairs"][0][field] = value
    bridge = SimpleNamespace(call=lambda *args, **kwargs: copy.deepcopy(snapshot))
    with pytest.raises(Refusal, match="^validation_repair_export_binding_mismatch$"):
        render.export_snapshot(bridge, DAY, tmp_path / "export")


def test_unstarted_repair_export_rejects_retained_input_bytes(runtime, monkeypatch, tmp_path):
    snapshot = blocked_repair_snapshot(runtime, monkeypatch)
    snapshot["files"]["repair-1-input"] = base64.b64encode(b'{}').decode()
    bridge = SimpleNamespace(call=lambda *args, **kwargs: copy.deepcopy(snapshot))
    with pytest.raises(Refusal, match="^validation_repair_export_binding_mismatch$"):
        render.export_snapshot(bridge, DAY, tmp_path / "export")
