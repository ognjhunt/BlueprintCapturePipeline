"""Hermetic original-run claims, spend admission and lossless Exa receipts."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from tools.daily_research import expansion as e

NOW = datetime(2026, 10, 4, 12, 1, tzinfo=timezone.utc)
ARGS = {"query": "US regional laundry sites with towel handling evidence", "max_cost_micros": 2_000_000}
SCHEMA = {"type": "object", "required": ["query"], "properties": {
    "effort": {"type": "string", "enum": ["low", "auto", "ultra"], "default": "low"},
    "query": {"type": "string"}, "budget": {"type": "object", "properties": {"maxCostDollars": {"type": "number"}}}}}


class Ledger:
    def __init__(self, path):
        self.path = path
        self.fail_next_put = False

    def get(self, day):
        p = self.path / (day + ".json")
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, row):
        if self.fail_next_put:
            self.fail_next_put = False
            raise OSError("synthetic_store_failure")
        (self.path / (row["date"] + ".json")).write_text(json.dumps(row))

    def write_bytes(self, name, raw):
        with (self.path / name).open("xb") as handle:
            handle.write(raw)

    def read_bytes(self, name):
        return (self.path / name).read_bytes()


class Transport:
    def __init__(self, ledger, row):
        self.ledger, self.row = ledger, row
        self.starts, self.reads = [], []
        self.start_error = False
        self.fail_ack_pointer = False
        self.result = {"id": "agent_run_synthetic", "status": "running", "usage": None}

    def start(self, request):
        claim = self.ledger.get(self.row["date"])["exa_expansion"]
        assert claim["attempted"] and claim["run_id"] is None
        assert claim["intent"]["request"] == request
        self.starts.append(copy.deepcopy(request))
        if self.start_error:
            raise TimeoutError("synthetic_ack_unknown")
        if self.fail_ack_pointer:
            self.ledger.fail_next_put = True
        return copy.deepcopy(self.result)

    def read(self, run_id):
        self.reads.append(run_id)
        return copy.deepcopy(self.result)


@pytest.fixture
def context(tmp_path):
    row = {"date": "2026-10-04", "run_key": "synthetic_daily_run", "session_id": "sess_synthetic",
           "turn_id": "turn_synthetic", "state": "running", "started_at": (NOW - timedelta(seconds=60)).isoformat(),
           "research_runtime_seconds": 1200, "recurring_budget_authority_reference": "synthetic_owner_authority",
           "soft_target_usd": 5}
    ledger = Ledger(tmp_path)
    ledger.put(row)
    allocation = {"schema_version": e.ALLOCATION, "run_key": row["run_key"],
                  "authority_reference": row["recurring_budget_authority_reference"],
                  "all_in_verified": True, "usage_unknown": False, "evidence_reference": "company-synthetic-evidence",
                  "limit_micros": 5_000_000, "committed_micros": 1_000_000, "reserved_micros": 2_000_000,
                  "remaining_micros": 2_000_000, "checked_at": NOW.isoformat(),
                  "valid_until": (NOW + timedelta(seconds=30)).isoformat()}
    return row, ledger, Transport(ledger, row), allocation


def call(context, name=e.START, args=None, **overrides):
    row, ledger, transport, allocation = context
    options = {"transport": transport, "allocation": allocation, "tool_schema": SCHEMA, "now": NOW}
    options.update(overrides)
    return e.execute(name, ARGS if args is None else args, row, ledger, **options)


@pytest.mark.parametrize("change", [
    {"usage_unknown": True}, {"all_in_verified": False}, {"remaining_micros": 3_000_000},
    {"reserved_micros": 3_000_000, "remaining_micros": 1_000_000}, {"limit_micros": 10_000_000},
    {"authority_reference": "another_authority"}, {"run_key": "another_run"}, {"evidence_reference": ""},
    {"valid_until": NOW.isoformat()}, {"committed_micros": True},
])
def test_unknown_or_mismatched_allocation_skips_without_claim_or_post(context, change):
    context[3].update(change)
    assert call(context)["state"] == "skipped"
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


@pytest.mark.parametrize("schema", [None, {"properties": {}}, {"properties": None},
    {"properties": {"budget": None}},
    {**SCHEMA, "required": ["query", "previousRunId"]},
    {**SCHEMA, "properties": {**SCHEMA["properties"], "effort": {"type": "string", "enum": ["low", "auto"]}}},
    {**SCHEMA, "properties": {k: v for k, v in SCHEMA["properties"].items() if k != "effort"}},
    {**SCHEMA, "properties": {**SCHEMA["properties"], "effort": {"type": "string", "enum": "ultra"}}},
    {"properties": {"query": {"type": "string"}, "budget": {"properties": {"maxCostDollars": {"type": "number", "maximum": 1}}}}},
])
def test_no_guessed_cap_or_required_native_fields(context, schema):
    assert call(context, tool_schema=schema)["reason"] == "expansion_supported_native_cap_unverified"
    assert not context[2].starts


def test_no_binding_phase_or_original_deadline_extension(context):
    assert call(context, transport=None)["reason"] == "expansion_authenticated_transport_missing"
    assert call(context, phase="qa")["reason"] == "expansion_before_final_qa_only"
    assert call(context, now=NOW + timedelta(minutes=20))["reason"] == "expansion_original_deadline_exhausted"
    assert not context[2].starts


def test_host_unavailable_reason_is_actionable_without_claim(context):
    result = call(context, transport=None, unavailable_reason="expansion_existing_auth_missing")
    assert result["reason"] == "expansion_existing_auth_missing"
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


def test_native_start_is_once_across_restart_and_preserves_exact_cap(context):
    first = call(context, tool_schema={**SCHEMA, "required": ["query", "effort"]})
    row, ledger, transport, _ = context
    assert first["run_id"] == "agent_run_synthetic"
    assert transport.starts == [{"query": ARGS["query"], "effort": "ultra", "budget": {"maxCostDollars": 2.0}}]
    restored = ledger.get(row["date"])
    resumed = (restored, Ledger(ledger.path), transport, context[3])
    assert call(resumed)["run_id"] == first["run_id"] and len(transport.starts) == 1
    with pytest.raises(e.ExpansionError, match="already_claimed_different_request"):
        call(resumed, args={**ARGS, "query": "Another query"})
    with pytest.raises(e.ExpansionError, match="already_claimed_different_request"):
        call(resumed, args={**ARGS, "max_cost_micros": 1_000_000})


def test_documented_ultra_minimum_is_enforced_when_catalog_omits_it(context):
    result = call(context, args={**ARGS, "max_cost_micros": 999_999})
    assert result["reason"] == "expansion_cap_below_ultra_minimum"
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


def test_advertised_cap_bounds_equal_enforced_ultra_bounds():
    """2026-10-04: the agent requested 500000 micros because the tool advertised a 1-micro minimum."""
    cap = e.tools()[0]["parameters"]["properties"]["max_cost_micros"]
    assert (cap["minimum"], cap["maximum"]) == (e.ULTRA_MIN_MICROS, e.LIMIT_MICROS) == (1_000_000, 5_000_000)
    assert "1000000" in e.instructions() and "5000000" in e.instructions()


@pytest.mark.parametrize("overrides", [{}, {"transport": None, "allocation": None,
                                            "unavailable_reason": "expansion_remaining_all_in_allocation_unverified"}])
def test_cap_below_ultra_minimum_is_actionable_and_consumes_no_claim(context, overrides):
    result = call(context, args={**ARGS, "max_cost_micros": 500_000}, **overrides)
    assert result["state"] == "skipped" and result["reason"] == "expansion_cap_below_ultra_minimum"
    assert "1000000" in result["action"] and "5000000" in result["action"] and "no claim" in result["action"].lower()
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


def test_cap_above_remaining_allocation_names_the_headroom(context):
    context[3].update(reserved_micros=3_000_000, remaining_micros=1_000_000)
    result = call(context)
    assert result["state"] == "skipped" and result["reason"] == "expansion_cap_exceeds_remaining_allocation"
    assert result["remaining_micros"] == 1_000_000
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


@pytest.mark.parametrize("bounds", [{"minimum": 3}, {"maximum": 1}, {"exclusiveMinimum": 2},
                                     {"exclusiveMaximum": 2}, {"enum": [1, 3]}])
def test_live_native_cap_bounds_are_not_overridden_by_application_range(context, bounds):
    schema = copy.deepcopy(SCHEMA)
    schema["properties"]["budget"]["properties"]["maxCostDollars"].update(bounds)
    assert call(context, tool_schema=schema)["reason"] == "expansion_supported_native_cap_unverified"
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


@pytest.mark.parametrize("name", [e.START, e.READ])
def test_legacy_no_effort_ack_recovers_unchanged_after_deadline_without_post(context, name):
    row, ledger, transport, _ = context
    original_request = {"query": ARGS["query"], "budget": {"maxCostDollars": 0.5}}
    intent = {"run_key": row["run_key"], "session_id": row["session_id"], "turn_id": row["turn_id"],
              "request": original_request}
    intent_raw = e._bytes(intent)
    claim = {"date": row["date"], "run_key": row["run_key"], "intent": intent,
             "intent_json": intent_raw.decode(), "intent_sha256": e._hash(intent_raw), "cap_micros": 500_000,
             "state": "submission_unresolved", "attempted": True, "run_id": None}
    row["exa_expansion"] = copy.deepcopy(claim)
    ledger.put(row)
    provider_record = {"id": "agent_run_legacy", "status": "completed", "costDollars": None}
    record = {"intent_sha256": claim["intent_sha256"], "run_id": provider_record["id"], "provider_record": provider_record}
    ledger.write_bytes(f"{row['date']}-exa-{claim['intent_sha256']}-start.json", e._bytes(record))
    args = {**ARGS, "max_cost_micros": 500_000} if name == e.START else {}
    result = call(context, name, args, transport=None, now=NOW + timedelta(days=1))
    restored = ledger.get(row["date"])["exa_expansion"]
    assert result["provider_record"] == provider_record and result["billing_verified"] is False
    assert restored["intent_json"].encode() == intent_raw and restored["intent_sha256"] == claim["intent_sha256"]
    assert restored["intent"]["request"] == original_request
    assert not transport.starts and not transport.reads


def test_uncertain_ack_is_permanent_not_a_retry_or_free_budget(context):
    context[2].start_error = True
    assert call(context)["state"] == "submission_unresolved"
    assert call(context)["reserved_micros"] == ARGS["max_cost_micros"]
    assert call(context, e.READ, {})["run_id"] is None
    assert len(context[2].starts) == 1 and not context[2].reads


def test_ack_file_recovers_pointer_failure_without_second_post(context):
    row, ledger, transport, allocation = context
    transport.fail_ack_pointer = True
    with pytest.raises(OSError, match="synthetic_store_failure"):
        call(context)
    restored = ledger.get(row["date"])
    assert restored["exa_expansion"]["run_id"] is None
    assert call((restored, Ledger(ledger.path), transport, allocation))["run_id"] == "agent_run_synthetic"
    assert len(transport.starts) == 1


def test_terminal_ack_recovery_needs_no_provider_read(context):
    row, ledger, transport, allocation = context
    transport.result["status"] = "completed"
    transport.fail_ack_pointer = True
    with pytest.raises(OSError):
        call(context)
    restored = ledger.get(row["date"])
    result = call((restored, Ledger(ledger.path), transport, allocation), e.READ, {}, transport=None)
    assert result["provider_record"] == transport.result
    assert not transport.reads and len(transport.starts) == 1


def test_original_read_retains_full_report_and_uses_terminal_cache(context):
    call(context)
    record = {"id": "agent_run_synthetic", "status": "completed", "outputReady": True,
              "output": {"sources": ["https://operator.example/source"], "unknowns": ["interest"], "future_field": 42},
              "costDollars": {"total": 1.27}, "usage": {"searches": 8}}
    context[2].result = record
    result = call(context, e.READ, {})
    assert result["provider_record"] == record and result["billing_verified"] is False
    saved = context[1].get(context[0]["date"])["exa_expansion"]["terminal_receipt"]
    assert json.loads(context[1].read_bytes(saved["file"]))["provider_record"] == record
    assert call(context, e.READ, {}, transport=None)["provider_record"] == record
    assert context[2].reads == ["agent_run_synthetic"]


def test_pending_identical_reads_are_retained_without_receipt_overwrite(context):
    call(context)
    call(context, e.READ, {})
    call(context, e.READ, {}, now=NOW + timedelta(seconds=1))
    assert context[2].reads == ["agent_run_synthetic", "agent_run_synthetic"]


def test_read_cannot_select_other_ids_or_accept_wrong_provider_identity(context):
    call(context)
    with pytest.raises(e.ExpansionError, match="read_original_id_only"):
        call(context, e.READ, {"runId": "another_run"})
    context[2].result["id"] = "another_run"
    with pytest.raises(e.ExpansionError, match="provider_id_changed"):
        call(context, e.READ, {})


def test_tampered_terminal_receipt_refuses_without_new_call(context):
    call(context)
    context[2].result["status"] = "completed"
    call(context, e.READ, {})
    ref = context[0]["exa_expansion"]["terminal_receipt"]
    (context[1].path / ref["file"]).write_bytes(b"{}")
    with pytest.raises(e.ExpansionError, match="receipt_digest_mismatch"):
        call(context, e.READ, {})
    assert len(context[2].starts) == 1 and len(context[2].reads) == 1


def test_claim_storage_or_final_fence_failure_never_reaches_transport(context):
    context[1].fail_next_put = True
    with pytest.raises(OSError):
        call(context)
    assert not context[2].starts
    calls = []

    def admit(row):
        calls.append(row["run_key"])
        if len(calls) == 2:
            raise e.ExpansionError("synthetic_fence_lost")

    with pytest.raises(e.ExpansionError, match="synthetic_fence_lost"):
        call(context, admit=admit)
    assert not context[2].starts and context[1].get(context[0]["date"])["exa_expansion"]["attempted"]


def test_expired_admission_after_claim_keeps_reservation_without_native_start(context, monkeypatch):
    ticks = iter([0, 31, 31])
    monkeypatch.setattr(e.time, "monotonic", lambda: next(ticks))
    result = call(context)
    assert result["reason"] == "expansion_admission_expired_before_submission"
    assert result["reserved_micros"] == ARGS["max_cost_micros"] and result["replay_permitted"] is False
    assert not context[2].starts
    assert context[1].get(context[0]["date"])["exa_expansion"]["attempted"]
