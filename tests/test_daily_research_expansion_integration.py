"""The actual daily function boundary, without paid providers or credentials."""
import hashlib
import json
from copy import deepcopy
from datetime import timedelta

import pytest

from tests.test_daily_research_exa_transport import Wire
from tests.test_daily_research_expansion import ARGS, SCHEMA, Transport
from tests.test_daily_research_render import fixture as render_fixture
from tests.test_daily_research_runner import DAY, NOW
from tests.test_daily_research_search import fixture as search_fixture
from tools.daily_research import expansion, render, search
from tools.daily_research.consumer import Consumer
from tools.daily_research.exa_transport import ExaTransport
from tools.daily_research.runner import Refusal, canonical, configuration


@pytest.fixture
def fixture(tmp_path):
    yield from search_fixture.__wrapped__(tmp_path)


@pytest.fixture
def render_context(tmp_path):
    yield from render_fixture.__wrapped__(tmp_path)


def setup(fixture):
    runner, api, ledger = fixture
    runner.config.update(soft_target_usd=5, expansion_profile=expansion.PROFILE)
    api.expansion_context = lambda row, name: {"unavailable_reason": "expansion_remaining_all_in_allocation_unverified"}
    api.expansion_admit = api.tool_admit
    row = runner.start_or_resume()
    return runner, api, ledger, row


def action(row, name=expansion.START, args=ARGS, call_id="call_synthetic_exa"):
    return {"type": "function_call", "name": name, "arguments": args,
            "turn_id": row["turn_id"], "call_id": call_id}


def respond(api, ledger, row):
    return search.respond(row, api.get("session", row["session_id"]), ledger, api,
                          phase="research", clock=lambda: NOW)


def allocation_for(row):
    return {"schema_version": expansion.ALLOCATION, "run_key": row["run_key"],
        "authority_reference": row["recurring_budget_authority_reference"], "all_in_verified": True,
        "usage_unknown": False, "evidence_reference": "company-synthetic-bound-receipt",
        "limit_micros": 5_000_000, "committed_micros": 1_000_000, "reserved_micros": 2_000_000,
        "remaining_micros": 2_000_000, "checked_at": NOW.isoformat(),
        "valid_until": (NOW + timedelta(seconds=30)).isoformat()}


def test_daily_payload_binds_only_guarded_exa_functions_and_survives_qa_session_check(fixture):
    runner, api, _, row = setup(fixture)
    configuration(runner.config)
    tools = row["create_payload"]["agent"]["tools"]
    assert [t["name"] for t in tools][-2:] == [expansion.START, expansion.READ]
    assert not any(t.get("server_label") in {"exa", "parallel_task", "blueprint"} for t in tools)
    assert row["metadata"]["expansion_profile"] == expansion.PROFILE
    Consumer.check_session(row, api.get("session", row["session_id"]))
    runner.config["mcp_profile"] = search.MCP_RESEARCH_PROFILE
    with pytest.raises(Refusal, match="research_expansion_profile_invalid"):
        configuration(runner.config)


def test_unverified_room_is_a_durable_optional_skip_and_ordinary_search_continues(fixture):
    _, api, ledger, row = setup(fixture)
    api.actions = [action(row)]
    respond(api, ledger, row)
    event = api.result_events[-1][1]
    outcome = json.loads(event["output"])
    assert outcome["state"] == "skipped" and "allocation_unverified" in outcome["reason"]
    assert json.loads(event["error"]) == {"code": outcome["reason"], "guidance": outcome["action"]}
    assert event["success"] is False
    assert "exa_expansion" not in ledger.get(DAY) and not api.executions
    api.actions = [action(row, search.SEARCH, {"query": "US laundry folding work"}, "call_synthetic_regular")]
    respond(api, ledger, row)
    assert len(api.executions) == 1


def test_different_function_call_ids_cannot_repeat_paid_start_and_original_read_is_retained(fixture):
    _, api, ledger, row = setup(fixture)
    transport = Transport(ledger, row)
    allocation = allocation_for(row)
    api.expansion_context = lambda row, name: {"transport": transport, "allocation": allocation, "tool_schema": SCHEMA}
    api.actions = [action(row)]
    respond(api, ledger, row)
    api.actions = [action(row, call_id="call_synthetic_exa_second")]
    respond(api, ledger, row)
    assert len(transport.starts) == 1
    transport.result = {"id": "agent_run_synthetic", "status": "completed", "output": {"sites": []}, "cost": 1}
    api.actions = [action(row, expansion.READ, {}, "call_synthetic_exa_read")]
    respond(api, ledger, row)
    assert transport.reads == ["agent_run_synthetic"]
    claim = ledger.get(DAY)["exa_expansion"]
    assert claim["terminal_receipt"] and claim["run_id"] == "agent_run_synthetic"


def test_unknown_native_ack_does_not_stop_next_days_ordinary_research(fixture):
    runner, api, ledger, row = setup(fixture)
    transport = Transport(ledger, row)
    transport.start_error = True
    expansion.execute(expansion.START, ARGS, row, ledger, transport=transport,
        allocation=allocation_for(row), tool_schema=SCHEMA, now=NOW)
    row.update(state="completed", cleanup_required=False)
    ledger.put(row)
    runner.clock = lambda: NOW + timedelta(days=1)
    assert runner.start_or_resume()["date"] == "2026-10-01"
    assert len(api.payloads) == 2
    assert ledger.get(DAY)["exa_expansion"]["state"] == "submission_unresolved"
    assert len(transport.starts) == 1 and not transport.reads


@pytest.mark.parametrize("fail_recovered_pointer", [False, True])
def test_daily_resume_recovers_retained_original_ack_after_deadline_without_credentials_or_post(fixture, monkeypatch, fail_recovered_pointer):
    runner, _api, ledger, row = setup(fixture)
    monkeypatch.setenv("EXA_API_KEY", "synthetic-not-a-real-key")
    wire = Wire(record={"id": "agent_run_synthetic", "status": "completed", "output": {"sites": []}})

    def sink(receipt):
        if receipt["operation"] == "tools/call":
            claim = ledger.get(DAY)["exa_expansion"]
            filename = f"{DAY}-exa-{claim['intent_sha256']}-start-http.json"
            ledger.write_bytes(filename, canonical(receipt).encode())
            raise OSError("synthetic_pointer_failure_after_raw_retention")

    transport = ExaTransport(request_io=wire, receipt_sink=sink)
    schema = transport.discover()
    result = expansion.execute(expansion.START, ARGS, row, ledger, transport=transport,
        allocation=allocation_for(row), tool_schema=schema, now=NOW)
    assert result["state"] == "submission_unresolved" and result["run_id"] is None
    assert len(wire.native_calls) == 1
    row.update(state="completed", cleanup_required=False)
    ledger.put(row)
    monkeypatch.delenv("EXA_API_KEY")
    runner.clock = lambda: NOW + timedelta(days=1)
    if fail_recovered_pointer:
        original_put = ledger.put
        failed = False

        def put(updated):
            nonlocal failed
            if updated.get("exa_expansion", {}).get("run_id") and not failed:
                failed = True
                raise OSError("synthetic_second_pointer_failure")
            return original_put(updated)

        monkeypatch.setattr(ledger, "put", put)
        with pytest.raises(OSError, match="synthetic_second_pointer_failure"):
            runner.start_or_resume()
        assert ledger.get(DAY)["exa_expansion"]["run_id"] is None
    assert runner.start_or_resume()["date"] == "2026-10-01"
    claim = ledger.get(DAY)["exa_expansion"]
    assert claim["run_id"] == "agent_run_synthetic" and claim["state"] == "completed"
    assert claim["terminal_receipt"] and len(wire.native_calls) == 1
    assert ledger.get(DAY)["exa_transport_receipts"][0]["file"].endswith("-start-http.json")


def test_company_store_keeps_float_intent_claim_irreversible_and_exports_native_receipts(render_context, tmp_path):
    _, _, ledger, bridge, _ = render_context
    intent = {"request": {"query": "US laundry caf\u00e9", "budget": {"maxCostDollars": 2.0}}}
    raw_intent = canonical(intent)
    native = b'{"id":"agent_run_synthetic","status":"completed"}'
    filename = DAY + "-exa-" + hashlib.sha256(raw_intent.encode()).hexdigest() + "-start.json"
    receipt = {"file": filename, "sha256": hashlib.sha256(native).hexdigest(), "bytes": len(native)}
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY, "state": "creating",
        "metadata": {"expansion_profile": expansion.PROFILE}, "expansion_profile": expansion.PROFILE,
        "cleanup_required": False, "delivery": {}, "exa_transport_receipts": [receipt],
        "exa_expansion": {"date": DAY, "run_key": "blueprint-researcher:" + DAY,
            "intent": intent, "intent_json": raw_intent, "intent_sha256": hashlib.sha256(raw_intent.encode()).hexdigest(),
            "run_id": "agent_run_synthetic", "terminal_receipt": receipt}}
    with ledger.lock():
        ledger.write_bytes(filename, native)
        ledger.put(row)
        missing = deepcopy(row)
        missing.pop("exa_expansion")
        with pytest.raises(Refusal, match="expansion_start_claim_already_consumed"):
            ledger.put(missing)
        replacement = deepcopy(row)
        replacement["exa_expansion"]["run_id"] = "agent_run_replacement"
        with pytest.raises(Refusal, match="expansion_original_id_changed"):
            ledger.put(replacement)
        with pytest.raises(Refusal, match="artifact_identity_conflict"):
            ledger.write_bytes(filename, b'different')
    render.export_snapshot(bridge, DAY, tmp_path / "export")
    assert (tmp_path / "export" / filename).read_bytes() == native
    snapshot = bridge.call("snapshot", day=DAY)
    snapshot["files"].pop(filename[len(DAY) + 1:-5])
    class Missing:
        def call(self, *args, **kwargs):
            return snapshot
    with pytest.raises(Refusal, match="expansion_export_binding_invalid"):
        render.export_snapshot(Missing(), DAY, tmp_path / "bad-export")
