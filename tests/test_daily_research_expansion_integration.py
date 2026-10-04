"""The actual daily function boundary, without paid providers or credentials."""
import hashlib
import json
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.test_daily_research_exa_transport import Wire
from tests.test_daily_research_expansion import ARGS, COMMIT, SCHEMA, Transport, owner_control
from tests.test_daily_research_render import fixture as render_fixture
from tests.test_daily_research_runner import DAY, NOW
from tests.test_daily_research_search import fixture as search_fixture
from tools.daily_research import allocation, expansion, render, search
from tools.daily_research.consumer import Consumer
from tools.daily_research.exa_transport import ExaTransport
from tools.daily_research.firestore import FencedProvider, FirestoreLedger
from tools.daily_research.runner import AGENT, PROJECT, Refusal, canonical, configuration


@pytest.fixture
def fixture(tmp_path):
    yield from search_fixture.__wrapped__(tmp_path)


@pytest.fixture
def render_context(tmp_path):
    yield from render_fixture.__wrapped__(tmp_path)


def setup(fixture, control=None, *, company=True):
    runner, api, ledger = fixture
    runner.config.update(soft_target_usd=5, expansion_profile=expansion.PROFILE)
    api.expansion_context = lambda row, name: {"unavailable_reason": "expansion_remaining_all_in_allocation_unverified"}
    api.expansion_admit = api.tool_admit
    # The company control the runner freezes the daily grant from (Render's FirestoreLedger).
    ledger.control = owner_control() if control is None else control
    if company:
        ledger.paid_expansion_control = lambda: ledger.control
    row = runner.start_or_resume()
    return runner, api, ledger, row


def action(row, name=expansion.START, args=ARGS, call_id="call_synthetic_exa"):
    return {"type": "function_call", "name": name, "arguments": args,
            "turn_id": row["turn_id"], "call_id": call_id}


def respond(api, ledger, row):
    return search.respond(row, api.get("session", row["session_id"]), ledger, api,
                          phase="research", clock=lambda: NOW)


def test_daily_payload_binds_only_guarded_exa_functions_and_survives_qa_session_check(fixture):
    runner, api, _, row = setup(fixture)
    configuration(runner.config)
    tools = row["create_payload"]["agent"]["tools"]
    assert [t["name"] for t in tools][-2:] == [expansion.START, expansion.READ]
    assert tools[-2:] == expansion.tools() and row["paid_expansion_grant"]["state"] == "granted"
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


def test_stable_expansion_error_code_reaches_the_agent(fixture):
    """ExpansionError codes are stable and secret-free; they must not collapse into a generic error."""
    _, api, ledger, row = setup(fixture)
    api.actions = [action(row, args={**ARGS, "unexpected": True})]
    respond(api, ledger, row)
    event = api.result_events[-1][1]
    assert event["success"] is False
    assert json.loads(event["error"])["code"] == "expansion_start_arguments_invalid"
    assert json.loads(event["output"])["error"]["code"] == "expansion_start_arguments_invalid"
    assert "exa_expansion" not in ledger.get(DAY) and not api.executions


def test_different_function_call_ids_cannot_repeat_paid_start_and_original_read_is_retained(fixture):
    _, api, ledger, row = setup(fixture)
    transport = Transport(ledger, row)
    api.expansion_context = lambda row, name: {"transport": transport, "control": ledger.control, "tool_schema": SCHEMA}
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
        control=ledger.control, tool_schema=SCHEMA, now=NOW)
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
        control=ledger.control, tool_schema=schema, now=NOW)
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


def company_direction(bridge, limit="10.00"):
    """The owner's direction through the real fenced bridge op, as the operator applies it."""
    control = owner_control(limit)
    entry = control["paid_expansion"]["current"]
    current = bridge.call("control")
    previous = (current.get("paid_expansion") or {}).get("current")
    if previous:
        entry["direction"].update(version=previous["version"] + 1, supersedes=previous["sha256"])
        sha = allocation.digest(entry["direction"])
        entry.update(sha256=sha, version=entry["direction"]["version"], uri=allocation.uri(sha))
    with FirestoreLedger(bridge).lock():
        if not previous:
            bridge.call("configure", value={**{k: v for k, v in current.items() if k != "lease"},
                                            "project_id": PROJECT, "agent_id": AGENT, "source_commit": COMMIT})
        bridge.call("paid_expansion_set", expected_sha256=previous["sha256"] if previous else None,
                    value={"enabled": True, "current": entry})
    return bridge.call("control")


def granted_row(control, day=DAY):
    row = {"date": day, "run_key": "blueprint-researcher:" + day, "state": "creating",
           "started_at": NOW.isoformat(), "research_runtime_seconds": 1200,
           "metadata": {"expansion_profile": expansion.PROFILE}, "expansion_profile": expansion.PROFILE,
           "cleanup_required": False, "delivery": {}}
    row["paid_expansion_grant"] = allocation.grant(control, row, NOW)
    assert row["paid_expansion_grant"]["state"] == "granted"
    return row


def claim_for(row, cap_micros=2_000_000, grant=True):
    value = {"request": {"query": "US laundry caf\u00e9", "budget": {"maxCostDollars": cap_micros / 1_000_000}}}
    if grant:
        value["grant"] = deepcopy(row["paid_expansion_grant"])
    raw = canonical(value)
    return {"date": row["date"], "run_key": row["run_key"], "intent": value, "intent_json": raw,
            "intent_sha256": hashlib.sha256(raw.encode()).hexdigest(), "cap_micros": cap_micros,
            "state": "submission_unresolved", "attempted": True, "run_id": None}


def test_company_store_keeps_float_intent_claim_irreversible_and_exports_native_receipts(render_context, tmp_path):
    _, _, ledger, bridge, _ = render_context
    row = granted_row(company_direction(bridge))
    claim = claim_for(row)
    intent, raw_intent = claim["intent"], claim["intent_json"]
    native = b'{"id":"agent_run_synthetic","status":"completed"}'
    filename = DAY + "-exa-" + hashlib.sha256(raw_intent.encode()).hexdigest() + "-start.json"
    receipt = {"file": filename, "sha256": hashlib.sha256(native).hexdigest(), "bytes": len(native)}
    row.update(exa_transport_receipts=[receipt], exa_expansion={**claim, "intent": intent,
               "run_id": "agent_run_synthetic", "terminal_receipt": receipt})
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


def test_below_minimum_start_never_builds_an_authenticated_context(fixture):
    """The cap pre-check needs no allocation read, key or MCP catalog call."""
    _, api, ledger, row = setup(fixture)
    contexts = []
    api.expansion_context = lambda row, name: contexts.append(name) or {"unavailable_reason": "should_not_be_reached"}
    api.actions = [action(row, args={**ARGS, "max_cost_micros": 500_000})]
    respond(api, ledger, row)
    outcome = json.loads(api.result_events[-1][1]["output"])
    assert outcome["reason"] == "expansion_cap_below_ultra_minimum" and contexts == []
    assert "exa_expansion" not in ledger.get(DAY)


def test_grant_is_frozen_across_a_mid_run_amount_change_and_the_next_run_uses_the_new_amount(fixture):
    runner, _api, ledger, row = setup(fixture)
    first = row["paid_expansion_grant"]
    assert (first["limit_micros"], first["per_start_max_micros"], first["version"]) == (10_000_000, 5_000_000, 1)
    ledger.control = owner_control("30.00")
    runner.start_or_resume(allow_create=False)
    assert ledger.get(DAY)["paid_expansion_grant"] == first
    transport = Transport(ledger, row)
    result = expansion.execute(expansion.START, {**ARGS, "max_cost_micros": 5_000_000}, row, ledger,
                               transport=transport, control=ledger.control, tool_schema=SCHEMA, now=NOW)
    assert result["run_id"] == "agent_run_synthetic"
    assert ledger.get(DAY)["exa_expansion"]["intent"]["grant"] == first
    status = expansion.allocation_diagnostic(ledger.get(DAY), ledger.control, now=NOW)
    assert (status["limit_micros"], status["reserved_micros"], status["remaining_micros"]) == (10_000_000, 5_000_000, 5_000_000)
    row = ledger.get(DAY)
    row.update(state="completed", cleanup_required=False)
    ledger.put(row)
    runner.clock = lambda: NOW + timedelta(days=1)
    following = runner.start_or_resume()
    assert following["date"] == "2026-10-01" and following["paid_expansion_grant"]["limit_micros"] == 30_000_000
    assert following["paid_expansion_grant"]["per_start_max_micros"] == 15_000_000
    assert following["paid_expansion_grant"]["grant_id"] != first["grant_id"]


@pytest.mark.parametrize("control", [None, {"source_commit": COMMIT}, owner_control(enabled=False)])
def test_disabled_or_older_control_records_a_refusal_and_research_continues(fixture, control):
    # None: the disk ledger has no company control at all.
    _, api, ledger, row = setup(fixture, control, company=control is not None)
    assert row["paid_expansion_grant"]["state"] == "refused"
    assert row["paid_expansion_grant"]["code"] == "paid_expansion_disabled"
    api.actions = [action(row, search.SEARCH, {"query": "US laundry folding work"}, "call_synthetic_regular")]
    respond(api, ledger, row)
    assert len(api.executions) == 1


def fenced(control, rows=()):
    calls = []

    def call(op, **fields):
        calls.append(op)
        return deepcopy(control) if op == "control" else True

    provider = FencedProvider.__new__(FencedProvider)
    provider.ledger = SimpleNamespace(bridge=SimpleNamespace(call=call), rows=lambda: list(rows))
    provider.clock = lambda: NOW
    return provider, calls


def company_control(row, paid):
    return {"enabled": True, **paid, "config": {"search_provider": search.PROFILE, "expansion_profile": expansion.PROFILE,
            "recurring_budget_authority_reference": row["recurring_budget_authority_reference"],
            "soft_target_usd": row["soft_target_usd"]}}


def test_fenced_context_keeps_todays_skip_for_older_control_and_never_builds_a_transport(fixture, monkeypatch):
    _, api, ledger, row = setup(fixture, {"source_commit": COMMIT})
    monkeypatch.setenv("EXA_API_KEY", "synthetic-not-a-real-key")
    monkeypatch.setattr("tools.daily_research.exa_transport.ExaTransport.discover",
                        lambda self: pytest.fail("no catalog call without an admissible grant"))
    provider, _ = fenced(company_control(row, {"source_commit": COMMIT}))
    api.expansion_context = provider.expansion_context
    api.actions = [action(row)]
    respond(api, ledger, row)
    outcome = json.loads(api.result_events[-1][1]["output"])
    assert outcome["reason"] == "expansion_remaining_all_in_allocation_unverified"
    assert outcome["allocation_status"]["reasons"] == ["paid_expansion_disabled"]
    assert outcome["allocation_reason"] == "paid_expansion_disabled" and outcome["remaining_micros"] is None
    assert "exa_expansion" not in ledger.get(DAY)


def test_fenced_context_and_final_admit_apply_the_live_brake_and_release_fence(fixture, monkeypatch):
    *_, row = setup(fixture)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    live = company_control(row, owner_control())
    provider, _ = fenced(live)
    context = provider.expansion_context(row, expansion.START)
    assert context["unavailable_reason"] == "expansion_worker_exa_binding_missing" and context["control"] == live
    assert context["allocation_status"]["remaining_micros"] == 10_000_000
    braked = company_control(row, owner_control(enabled=False))
    assert fenced(braked)[0].expansion_context(row, expansion.START)["unavailable_reason"] == (
        "expansion_remaining_all_in_allocation_unverified")
    pending = {**row, "exa_expansion": claim_for(row)}
    provider.expansion_admit(pending, "research")
    for changed in (braked, company_control(row, owner_control(commit="e" * 40))):
        with pytest.raises(Refusal, match="expansion_allocation_changed_before_submission"):
            fenced(changed)[0].expansion_admit(pending, "research")
    legacy = {**row, "exa_expansion": claim_for(row, grant=False)}
    with pytest.raises(Refusal, match="expansion_allocation_changed_before_submission"):
        provider.expansion_admit(legacy, "research")


def test_company_store_debits_claims_against_the_frozen_grant_and_live_fences(render_context):
    _, _, ledger, bridge, _ = render_context
    first = company_direction(bridge, "10.00")
    row = granted_row(first)
    running = {**row, "state": "running"}
    with ledger.lock():
        ledger.put(row)  # The durable intent binds the frozen grant.
        with pytest.raises(Refusal, match="paid_expansion_reservation_exceeds_grant"):
            ledger.put({**running, "exa_expansion": claim_for(row, 6_000_000)})
        with pytest.raises(Refusal, match="paid_expansion_grant_required"):
            ledger.put({**running, "exa_expansion": claim_for(row, grant=False)})
        with pytest.raises(Refusal, match="paid_expansion_grant_already_bound"):
            ledger.put({**running, "paid_expansion_grant": {**row["paid_expansion_grant"], "limit_micros": 100_000_000}})
        with pytest.raises(Refusal, match="paid_expansion_grant_already_bound"):
            ledger.put({key: value for key, value in running.items() if key != "paid_expansion_grant"})
    raised = company_direction(bridge, "30.00")  # A raised amount waits for the next run.
    with ledger.lock():
        ledger.put({**running, "exa_expansion": claim_for(row, 5_000_000)})
        assert ledger.get(DAY)["exa_expansion"]["cap_micros"] == 5_000_000
        with pytest.raises(Refusal, match="paid_expansion_grant_not_admitted"):
            ledger.put(granted_row(first, "2026-10-01"))  # A new intent freezes the current direction only.
        following = granted_row(raised, "2026-10-01")
        assert following["paid_expansion_grant"]["limit_micros"] == 30_000_000
        ledger.put(following)
        released = {k: v for k, v in bridge.call("control").items() if k not in {"lease", "paid_expansion"}}
        bridge.call("configure", value={**released, "source_commit": "e" * 40})
        with pytest.raises(Refusal, match="paid_expansion_grant_not_admitted"):
            ledger.put({**following, "state": "running", "exa_expansion": claim_for(following)})
        bridge.call("configure", value={**released, "source_commit": COMMIT})
        current = raised["paid_expansion"]["current"]
        bridge.call("paid_expansion_set", expected_sha256=current["sha256"], value={"enabled": False, "current": current})
        with pytest.raises(Refusal, match="paid_expansion_grant_not_admitted"):
            ledger.put({**following, "state": "running", "exa_expansion": claim_for(following)})
        with pytest.raises(Refusal, match="paid_expansion_grant_not_admitted"):
            ledger.put(granted_row(raised, "2026-10-02"))
    assert bridge.call("control")["paid_expansion"] == {"enabled": False, "current": current}


@pytest.mark.parametrize("amount", ["6.00", "1.00"])
def test_company_store_refuses_a_new_claim_after_live_allowance_decreases(render_context, amount):
    _, _, ledger, bridge, _ = render_context
    first = company_direction(bridge, "30.00")
    row = granted_row(first)
    with ledger.lock():
        ledger.put(row)
    company_direction(bridge, amount)
    with ledger.lock(), pytest.raises(Refusal, match="paid_expansion_reservation_exceeds_grant"):
        ledger.put({**row, "state": "running", "exa_expansion": claim_for(row, 5_000_000)})
    assert not ledger.get(DAY).get("exa_expansion")


@pytest.mark.parametrize("amount, cap", [("6.00", 5_000_000), ("1.00", 5_000_000), ("6.00", 3_000_000)])
def test_final_pre_post_admission_rechecks_live_cap_without_double_debit(render_context, amount, cap):
    _, _, ledger, bridge, _ = render_context
    first = company_direction(bridge, "30.00")
    row = granted_row(first)
    pending = {**row, "state": "running", "exa_expansion": claim_for(row, cap)}
    with ledger.lock():
        ledger.put(row)
        ledger.put(pending)
    lower = company_direction(bridge, amount)
    provider, _ = fenced(lower)
    # Isolate paid allocation; the real Store above already owns and fences the row.
    provider.tool_admit = lambda row, phase: None
    if cap == 3_000_000:
        provider.expansion_admit(pending, "research")
    else:
        with pytest.raises(Refusal, match="expansion_allocation_changed_before_submission"):
            provider.expansion_admit(pending, "research")


@pytest.mark.parametrize("code", ["paid_expansion_grant_not_admitted", "firestore_create_not_admitted"])
def test_store_refusal_of_the_frozen_grant_is_recorded_and_research_continues(fixture, monkeypatch, code):
    runner, api, ledger = fixture
    runner.config.update(soft_target_usd=5, expansion_profile=expansion.PROFILE)
    ledger.paid_expansion_control = owner_control
    original, refused = ledger.put, []

    def put(row):
        if row.get("paid_expansion_grant", {}).get("state") == "granted":
            refused.append(row["paid_expansion_grant"]["grant_id"])
            raise Refusal(code)  # Raised inside the store's transaction: nothing was recorded.
        return original(row)

    monkeypatch.setattr(ledger, "put", put)
    if code != "paid_expansion_grant_not_admitted":
        with pytest.raises(Refusal, match=code):
            runner.start_or_resume()
        assert not api.payloads and ledger.get(DAY) is None
        return
    row = runner.start_or_resume()
    assert len(refused) == 1 and row["paid_expansion_grant"]["state"] == "refused"
    assert row["paid_expansion_grant"]["code"] == code and len(api.payloads) == 1
    assert ledger.get(DAY)["paid_expansion_grant"] == row["paid_expansion_grant"]
