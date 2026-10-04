"""Hermetic original-run claims, spend admission and lossless Exa receipts."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from tools.daily_research import allocation
from tools.daily_research import expansion as e
from tools.daily_research.runner import digest

NOW = datetime(2026, 10, 4, 12, 1, tzinfo=timezone.utc)
COMMIT = "b" * 40
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


def owner_control(limit="10.00", *, enabled=True, commit=COMMIT):
    """Company control carrying one verified owner direction (allocation.py)."""
    value = {"schema_version": allocation.DIRECTION, "version": 1, "supersedes": None, "per_run_limit_usd": limit,
             "sources": ["exa"], "scope": dict(allocation.SCOPE), "effective_from": "2026-09-01T00:00:00+00:00",
             "expires_at": "2026-12-31T00:00:00+00:00", "approval_reference": "owner-direction-synthetic",
             "approved_by": "owner", "issued_at": "2026-09-01T00:00:00+00:00", "reason": "Synthetic allowance"}
    sha = allocation.digest(value)
    return {"source_commit": commit, "paid_expansion": {"enabled": enabled, "current": {
        "sha256": sha, "version": 1, "uri": allocation.uri(sha), "direction": value}}}


def freeze(row, ledger, limit="10.00"):
    """The runner's one grant per daily row, frozen at the run start."""
    control = owner_control(limit)
    row["paid_expansion_grant"] = allocation.grant(control, row, datetime.fromisoformat(row["started_at"]))
    ledger.put(row)
    return control


@pytest.fixture
def context(tmp_path):
    row = {"date": "2026-10-04", "run_key": "blueprint-researcher:2026-10-04", "session_id": "sess_synthetic",
           "turn_id": "turn_synthetic", "state": "running", "started_at": (NOW - timedelta(seconds=60)).isoformat(),
           "research_runtime_seconds": 1200, "recurring_budget_authority_reference": "synthetic_owner_authority",
           "soft_target_usd": 5}
    ledger = Ledger(tmp_path)
    control = freeze(row, ledger)
    assert row["paid_expansion_grant"]["state"] == "granted"
    return row, ledger, Transport(ledger, row), control


def call(context, name=e.START, args=None, **overrides):
    row, ledger, transport, control = context
    options = {"transport": transport, "control": control, "tool_schema": SCHEMA, "now": NOW}
    options.update(overrides)
    return e.execute(name, ARGS if args is None else args, row, ledger, **options)


def regrant(context, grant):
    row, ledger, _, _ = context
    row["paid_expansion_grant"] = grant
    ledger.put(row)


@pytest.mark.parametrize(("control", "grant", "reason"), [
    (owner_control(enabled=False), None, "paid_expansion_disabled"),
    ({"source_commit": COMMIT}, None, "paid_expansion_disabled"),
    (owner_control(commit="d" * 40), None, "paid_expansion_source_commit_changed"),
    (None, None, "paid_expansion_disabled"),
    (owner_control(), {"limit_micros": 20_000_000}, "paid_expansion_grant_invalid"),
    (owner_control(), {"valid_until": NOW.isoformat()}, "paid_expansion_expired"),
    (owner_control(), {"state": "refused", "code": "paid_expansion_disabled"}, "paid_expansion_disabled"),
])
def test_unknown_braked_or_changed_grant_skips_without_claim_or_post(context, control, grant, reason):
    if grant is not None:
        regrant(context, {**context[0]["paid_expansion_grant"], **grant})
    result = call(context, control=control)
    assert result["state"] == "skipped" and result["reason"] == "expansion_remaining_all_in_allocation_unverified"
    assert result["allocation_reason"] == reason and result["max_start_micros"] in {0, None}
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


@pytest.mark.parametrize("grant", [None, "missing"])
def test_older_rows_without_a_frozen_grant_keep_todays_skip(context, grant):
    row, ledger, transport, _ = context
    if grant:
        row.pop("paid_expansion_grant")
    else:
        row["paid_expansion_grant"] = None
    ledger.put(row)
    result = call(context)
    assert result["reason"] == "expansion_remaining_all_in_allocation_unverified"
    assert result["allocation_reason"] == "paid_expansion_grant_missing" and result["remaining_micros"] is None
    assert not transport.starts and "exa_expansion" not in ledger.get(row["date"])


def test_host_stopped_skip_names_the_same_grant_fields(context):
    result = call(context, transport=None, control=owner_control(enabled=False), allocation_status={},
                  unavailable_reason="expansion_remaining_all_in_allocation_unverified")
    assert result["reason"] == "expansion_remaining_all_in_allocation_unverified"
    assert (result["allocation_reason"], result["remaining_micros"], result["max_start_micros"]) == (
        "paid_expansion_disabled", 10_000_000, 0)
    assert result["allocation_status"]["reasons"] == ["paid_expansion_disabled"]
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


def test_in_memory_grant_must_match_the_durable_row(context):
    context[0]["paid_expansion_grant"] = {**context[0]["paid_expansion_grant"], "limit_micros": 100_000_000}
    with pytest.raises(e.ExpansionError, match="expansion_daily_binding_changed"):
        call(context)
    assert not context[2].starts


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


def test_sixty_minute_row_admits_expansion_until_its_own_forty_five_minute_deadline(tmp_path):
    row = {"date": "2026-10-04", "run_key": "blueprint-researcher:2026-10-04", "session_id": "sess_synthetic",
           "turn_id": "turn_synthetic", "state": "running", "started_at": (NOW - timedelta(seconds=60)).isoformat(),
           "research_runtime_seconds": 2700, "recurring_budget_authority_reference": "synthetic_owner_authority",
           "soft_target_usd": 5}
    ledger = Ledger(tmp_path)
    control = freeze(row, ledger)
    transport = Transport(ledger, row)
    started = datetime.fromisoformat(row["started_at"])
    assert e._deadline(row) == started + timedelta(seconds=2700)
    assert row["paid_expansion_grant"]["valid_until"] == (started + timedelta(seconds=2700)).isoformat()
    context = (row, ledger, transport, control)
    # The old 1200-second research window has closed; this row's own window has not.
    first = call(context, now=started + timedelta(seconds=1201))
    assert first["run_id"] == "agent_run_synthetic" and len(transport.starts) == 1
    # At this row's deadline an original-ID read is not extended or sent.
    assert call(context, name=e.READ, args={}, now=started + timedelta(seconds=2700))["run_id"] == first["run_id"]
    assert not transport.reads and len(transport.starts) == 1
    with pytest.raises(e.ExpansionError, match="^expansion_original_deadline_invalid$"):
        e._deadline({**row, "research_runtime_seconds": 3601})
    assert e._deadline({**row, "research_runtime_seconds": 3600}) == started + timedelta(seconds=3600)


def test_host_unavailable_reason_is_actionable_without_claim(context):
    result = call(context, transport=None, unavailable_reason="expansion_existing_auth_missing")
    assert result["reason"] == "expansion_existing_auth_missing"
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


def test_native_start_is_once_across_restart_and_preserves_exact_cap(context):
    first = call(context, tool_schema={**SCHEMA, "required": ["query", "effort"]})
    row, ledger, transport, _ = context
    assert first["run_id"] == "agent_run_synthetic"
    assert transport.starts == [{"query": ARGS["query"], "effort": "ultra", "budget": {"maxCostDollars": 2.0}}]
    intent = ledger.get(row["date"])["exa_expansion"]["intent"]
    assert intent["grant"] == row["paid_expansion_grant"] and intent["reserved_before_micros"] == 0
    assert "allocation" not in intent
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


def test_tool_bounds_are_the_ultra_minimum_and_the_single_per_start_ceiling():
    """2026-10-04: the agent requested 500000 micros because the tool advertised a 1-micro minimum.
    The owner's per-run amount, not a second hidden schema cap, decides each start."""
    cap = e.tools()[0]["parameters"]["properties"]["max_cost_micros"]
    assert (cap["minimum"], cap["maximum"]) == (e.ULTRA_MIN_MICROS, allocation.PER_START_CEILING_MICROS)
    assert e.LIMIT_MICROS == allocation.PER_START_CEILING_MICROS == 50_000_000
    text = e.instructions()
    assert "1000000" in text and "50000000" in text and "max_start_micros" in text
    assert "$20 allows 10000000" in text and "shares the existing all-in $5" not in text


def test_tool_schema_digest_changes_only_at_an_idle_release_boundary():
    """check_agent compares live session tools with search.tools(): change these bytes only
    when no daily session is in flight, never per run."""
    assert digest(e.tools()) == "6ee756d6ebf1e0b8861b23c4c5eacddc32a47ee43e799735abbeb2b6308183b4"


@pytest.mark.parametrize(("limit", "cap", "reason"), [
    ("20.00", 10_000_000, None), ("20.00", 10_000_001, "expansion_cap_exceeds_per_start_maximum"),
    ("10.00", 5_000_000, None), ("10.00", 6_000_000, "expansion_cap_exceeds_per_start_maximum"),
    ("30.00", 15_000_000, None), ("100.00", 50_000_000, None), ("6.00", 3_000_001, "expansion_cap_exceeds_per_start_maximum"),
    ("1.50", 2_000_000, "expansion_cap_exceeds_remaining_allocation")])
def test_the_owner_amount_alone_binds_each_exa_start(context, limit, cap, reason):
    row, ledger, transport, _ = context
    control = freeze(row, ledger, limit)
    grant = row["paid_expansion_grant"]
    result = call(context, args={**ARGS, "max_cost_micros": cap}, control=control)
    if reason is None:
        assert result["run_id"] == "agent_run_synthetic"
        assert transport.starts[-1]["budget"]["maxCostDollars"] == cap / 1_000_000
        return
    largest = min(grant["per_start_max_micros"], grant["limit_micros"])
    assert result["state"] == "skipped" and result["reason"] == reason
    assert result["max_start_micros"] == largest and result["remaining_micros"] == grant["limit_micros"]
    assert str(largest) in result["action"] and not transport.starts


def test_starts_above_the_hard_ceiling_are_invalid_arguments(context):
    freeze(context[0], context[1], "100.00")
    with pytest.raises(e.ExpansionError, match="expansion_start_arguments_invalid"):
        call(context, args={**ARGS, "max_cost_micros": 50_000_001})
    assert not context[2].starts


def test_a_lower_current_direction_tightens_the_run_in_progress(context):
    row, ledger, transport, _ = context
    assert not transport.starts
    freeze(row, ledger, "30.00")
    lowered = owner_control("6.00")
    lowered["paid_expansion"]["current"]["direction"].update(version=2, supersedes="a" * 64)
    value = lowered["paid_expansion"]["current"]["direction"]
    lowered["paid_expansion"]["current"].update(sha256=allocation.digest(value), version=2,
                                                uri=allocation.uri(allocation.digest(value)))
    result = call(context, args={**ARGS, "max_cost_micros": 4_000_000}, control=lowered)
    assert result["reason"] == "expansion_cap_exceeds_per_start_maximum"
    assert (result["remaining_micros"], result["max_start_micros"]) == (6_000_000, 3_000_000)
    assert call(context, args={**ARGS, "max_cost_micros": 3_000_000}, control=lowered)["run_id"] == "agent_run_synthetic"


@pytest.mark.parametrize("overrides", [{}, {"transport": None, "control": None,
                                            "unavailable_reason": "expansion_remaining_all_in_allocation_unverified"}])
def test_cap_below_ultra_minimum_is_actionable_and_consumes_no_claim(context, overrides):
    result = call(context, args={**ARGS, "max_cost_micros": 500_000}, **overrides)
    assert result["state"] == "skipped" and result["reason"] == "expansion_cap_below_ultra_minimum"
    assert (result["remaining_micros"], result["max_start_micros"]) == (10_000_000, 5_000_000)
    assert "1000000" in result["action"] and "5000000" in result["action"] and "no claim" in result["action"].lower()
    assert not context[2].starts and "exa_expansion" not in context[1].get(context[0]["date"])


def test_cap_above_remaining_allocation_names_the_headroom(context, monkeypatch):
    # FindAll will add claims of its own; any earlier durable reservation debits this run.
    monkeypatch.setattr(allocation, "claims", lambda row: [{"source": "findall", "reserved_micros": 9_000_000}])
    result = call(context)
    assert result["state"] == "skipped" and result["reason"] == "expansion_cap_exceeds_remaining_allocation"
    assert result["remaining_micros"] == 1_000_000 and result["max_start_micros"] == 1_000_000
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
    status = e.allocation_diagnostic(context[1].get(context[0]["date"]), context[3], now=NOW)
    assert status["reserved_micros"] == ARGS["max_cost_micros"] and status["remaining_micros"] == 8_000_000


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
    ticks = iter([0, 1200, 1200])  # The grant ends at the original research deadline.
    monkeypatch.setattr(e.time, "monotonic", lambda: next(ticks))
    result = call(context)
    assert result["reason"] == "expansion_admission_expired_before_submission"
    assert result["reserved_micros"] == ARGS["max_cost_micros"] and result["replay_permitted"] is False
    assert not context[2].starts
    assert context[1].get(context[0]["date"])["exa_expansion"]["attempted"]


def test_cap_supported_enforces_ultra_floor_even_when_catalog_omits_minimum():
    """Direct guard for _cap_supported; the pre-check must not be the only $1 floor."""
    request = {"query": ARGS["query"], "effort": "ultra", "budget": {"maxCostDollars": 0.999999}}
    assert not e._cap_supported(SCHEMA, request)
    assert e._cap_supported(SCHEMA, {**request, "budget": {"maxCostDollars": 1.0}})


@pytest.mark.parametrize(("used", "remaining"), [(9_500_000, 500_000), (10_000_000, 0), (12_000_000, 0)])
def test_no_ultra_cap_fits_remaining_allocation_says_so(context, monkeypatch, used, remaining):
    # Earlier durable reservations (FindAll later) exhaust the combined run allowance.
    monkeypatch.setattr(allocation, "claims", lambda row: [{"source": "findall", "reserved_micros": used}])
    result = call(context)
    assert result["reason"] == "expansion_cap_exceeds_remaining_allocation"
    assert (result["remaining_micros"], result["max_start_micros"]) == (remaining, remaining)
    assert "No Ultra cap fits" in result["action"] and not context[2].starts
    assert "exa_expansion" not in context[1].get(context[0]["date"])


@pytest.mark.parametrize("value", [None, 5, "not-a-time"])
def test_malformed_grant_times_skip_instead_of_crashing(context, value):
    regrant(context, {**context[0]["paid_expansion_grant"], "valid_until": value})
    result = call(context)
    assert result["reason"] == "expansion_remaining_all_in_allocation_unverified"
    assert result["allocation_reason"] == "paid_expansion_grant_invalid"
    assert e.allocation_diagnostic(context[0], context[3], now=NOW)["remaining_micros"] is None
