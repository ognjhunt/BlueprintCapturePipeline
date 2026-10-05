"""Hermetic owner direction command: real private bridge and Store, fake object storage."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.test_daily_research_expansion import ARGS, SCHEMA, Ledger, Transport
from tools.daily_research import allocation, expansion
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import AGENT, PROJECT, Refusal, canonical

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paid_expansion_direction",
                                              ROOT / "tools/daily_research/operators/paid-expansion-direction.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
NOW = datetime(2026, 10, 4, 18, 30, 15, 123456, tzinfo=timezone.utc)
COMMIT = "a" * 40


class Objects:
    """Create-only object storage double keyed by gs:// URI."""

    def __init__(self):
        self.values, self.creates = {}, []

    def create(self, uri, raw):
        self.creates.append(uri)
        if self.values.setdefault(uri, raw) != raw:
            raise Refusal("paid_expansion_object_conflict")

    def read(self, uri):
        if uri not in self.values:
            raise Refusal("paid_expansion_object_missing")
        return self.values[uri]


@pytest.fixture
def fixture(tmp_path):
    script = tmp_path / "bridge.mjs"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline';",
        "import {Store, LeaseChannel} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "const db = new MemoryFirestore(" + json.dumps(str(tmp_path / "firestore.json")) + ");",
        "const channel = new LeaseChannel(new Store(db));",
        "for await (const line of createInterface({input: process.stdin})) {",
        "try {const value = await channel.call(JSON.parse(line)); process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error) {process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}",
        "await channel.close();",
    ]))
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    control.update(source_commit=COMMIT)
    bridge = Bridge(script=script)
    bridge.call("init", value=control)
    yield bridge, Objects(), lambda: Bridge(script=script)
    bridge.close()


def rewrite_store(fixture, tmp_path, change):
    """Edit the persisted company store between processes, as an older release would."""
    bridge, _, open_bridge = fixture
    bridge.close()
    path = tmp_path / "firestore.json"
    values = dict(json.loads(path.read_text()))
    change(values["blueprintDailyResearch/sites-first"])
    path.write_text(json.dumps(list(values.items())))
    return open_bridge()


def setting(bridge, objects, amount="10.00", **changes):
    options = {"apply": True, "now": NOW, "per_run_usd": amount, "approval_reference": "owner-decision-2026-10-04",
               "approved_by": "owner", "reason": "Owner per-run paid expansion allowance", "sleep": lambda _: None}
    options.update(changes)
    return operator.set_direction(bridge, objects, **options)


def test_set_dry_run_is_read_only_and_names_the_exact_next_direction(fixture):
    bridge, objects, _ = fixture
    before = bridge.call("control")
    plan = setting(bridge, objects, "20", apply=False)
    assert bridge.call("control") == before and not objects.creates and bridge.call("paid_expansion_audit") == []
    assert plan["state"] == "planned" and plan["firestore_writes"] == plan["object_writes"] == plan["provider_calls"] == 0
    direction = plan["direction"]
    assert direction == {"schema_version": allocation.DIRECTION, "version": 1, "supersedes": None,
                         "per_run_limit_usd": "20.00", "sources": list(allocation.SOURCES), "scope": allocation.SCOPE,
                         "effective_from": "2026-10-04T18:30:15+00:00", "expires_at": "2027-01-02T18:30:15+00:00",
                         "approval_reference": "owner-decision-2026-10-04", "approved_by": "owner",
                         "issued_at": "2026-10-04T18:30:15+00:00", "reason": "Owner per-run paid expansion allowance"}
    assert plan["next"]["per_start_max_micros"] == 10_000_000 and plan["current_problem"] is None
    assert plan["next"]["sha256"] == allocation.digest(direction) and plan["expected_sha256"] is None


def test_apply_writes_object_then_swaps_control_and_reads_both_back(fixture):
    bridge, objects, _ = fixture
    applied = setting(bridge, objects)
    sha = applied["next"]["sha256"]
    uri = allocation.uri(sha)
    assert applied["state"] == "applied" and applied["readback_verified"] is True
    assert objects.values[uri] == canonical(applied["direction"]).encode() and objects.creates == [uri]
    paid = bridge.call("control")["paid_expansion"]
    assert paid == {"enabled": True, "current": {"sha256": sha, "version": 1, "uri": uri, "direction": applied["direction"]}}
    assert allocation.current(bridge.call("control"))[1] is None
    assert bridge.call("control")["lease"]["expires_at_ms"] == 0  # The lease is held only for the swap.
    # The amount is data: up to $30 and down to $5 without code, deploy or package.
    for version, amount in ((2, "30.00"), (3, "5.00")):
        again = setting(bridge, objects, amount, now=NOW + timedelta(minutes=version))
        assert again["direction"]["version"] == version and again["direction"]["supersedes"] == sha
        sha = again["next"]["sha256"]
    shown = operator.show(bridge, objects)
    assert shown["state"] == "enabled" and shown["per_run_limit_usd"] == "5.00" and shown["version"] == 3
    assert shown["per_start_max_micros"] == 2_500_000 and shown["object_verified"] is True
    assert shown["audit_chain_verified"] is True and [r["per_run_limit_usd"] for r in shown["audit_chain"]] == ["10.00", "30.00", "5.00"]
    assert shown["firestore_writes"] == shown["object_writes"] == shown["provider_calls"] == 0


@pytest.mark.parametrize("amount", ["200", "1000", "10.0", "0.50", "100.01", "$10", "1e1", "10.000", "", "ten"])
def test_typo_amounts_refuse_before_any_write(fixture, amount):
    bridge, objects, _ = fixture
    with pytest.raises(Refusal, match="paid_expansion_limit_invalid"):
        setting(bridge, objects, amount)
    assert not objects.creates and "paid_expansion" not in bridge.call("control")


@pytest.mark.parametrize(("changes", "code"), [
    ({"approval_reference": "PENDING-owner"}, "paid_expansion_direction_invalid"),
    ({"approval_reference": " "}, "paid_expansion_direction_invalid"),
    ({"reason": "café"}, "paid_expansion_direction_invalid"),
    ({"expires_at": NOW + timedelta(days=367)}, "paid_expansion_direction_invalid"),
    ({"expires_at": NOW - timedelta(seconds=1)}, "paid_expansion_direction_invalid"),
])
def test_invalid_direction_fields_refuse_before_any_write(fixture, changes, code):
    bridge, objects, _ = fixture
    with pytest.raises(Refusal, match=code):
        setting(bridge, objects, **changes)
    assert not objects.creates and "paid_expansion" not in bridge.call("control")


def test_stale_expectation_and_a_racing_writer_are_compare_and_swap_conflicts(fixture):
    bridge, objects, _ = fixture
    first = setting(bridge, objects)["next"]["sha256"]
    with pytest.raises(Refusal, match="paid_expansion_direction_conflict"):
        setting(bridge, objects, "20.00", expect="none")
    with pytest.raises(Refusal, match="paid_expansion_direction_conflict"):
        setting(bridge, objects, "20.00", expect="0" * 64)

    class Racing:
        """Another owner command lands between this command's read and its fenced swap."""

        def __init__(self, inner):
            self.inner, self.raced = inner, False

        def call(self, op, **fields):
            if op == "paid_expansion_set" and not self.raced:
                self.raced = True
                rival = operator.next_entry(self.inner.call("control"), per_run_usd="40.00", approval_reference="rival",
                                            approved_by="owner", reason="rival", now=NOW)[1]
                self.inner.call("paid_expansion_set", expected_sha256=first, value={"enabled": True, "current": rival})
            return self.inner.call(op, **fields)

    with pytest.raises(Refusal, match="paid_expansion_direction_conflict"):
        setting(Racing(bridge), objects, "20.00", expect=first)
    current = bridge.call("control")["paid_expansion"]["current"]
    assert current["direction"]["per_run_limit_usd"] == "40.00" and current["version"] == 2
    assert bridge.call("control")["lease"]["expires_at_ms"] == 0
    assert operator.show(bridge, objects)["audit_chain_verified"] is True


def test_disable_is_an_immediate_brake_and_set_re_enables_with_a_new_version(fixture):
    bridge, objects, _ = fixture
    assert operator.disable(bridge, apply=True)["state"] == "already_disabled"
    first = setting(bridge, objects)
    plan = operator.disable(bridge)
    assert plan["state"] == "planned" and bridge.call("control")["paid_expansion"]["enabled"] is True
    braked = operator.disable(bridge, apply=True, sleep=lambda _: None)
    assert braked["state"] == "disabled" and braked["object_writes"] == 0 and braked["readback_verified"] is True
    paid = bridge.call("control")["paid_expansion"]
    assert paid["enabled"] is False and paid["current"]["sha256"] == first["next"]["sha256"]
    assert allocation.current(bridge.call("control")) == (None, "paid_expansion_disabled")
    assert operator.show(bridge, objects)["state"] == "disabled"
    assert operator.disable(bridge, apply=True)["state"] == "already_disabled"
    again = setting(bridge, objects, "10.00", now=NOW + timedelta(minutes=1))
    assert again["direction"]["version"] == 2 and bridge.call("control")["paid_expansion"]["enabled"] is True


def test_apply_waits_for_an_active_run_unless_the_owner_overrides(fixture):
    bridge, objects, _ = fixture
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        bridge.call("configure", value={**{k: v for k, v in bridge.call("control").items() if k != "lease"}, "enabled": True})
        ledger.put({"date": "2026-10-04", "run_key": "blueprint-researcher:2026-10-04", "state": "creating",
                    "metadata": {"purpose": "synthetic_active_run"}, "cleanup_required": True, "delivery": {}})
    with pytest.raises(Refusal, match="paid_expansion_run_active_apply_after_run"):
        setting(bridge, objects)
    assert not objects.creates and "paid_expansion" not in bridge.call("control")
    assert setting(bridge, objects, during_active_run=True)["state"] == "applied"
    assert operator.disable(bridge, apply=True)["state"] == "disabled"  # The brake never waits.


def test_lease_overlap_is_polled_for_and_never_displaced(fixture, monkeypatch):
    bridge, objects, _ = fixture
    calls, sleeps = [], []
    original = bridge.call

    def call(op, **fields):
        calls.append(op)
        if op == "acquire" and calls.count("acquire") < 40:  # A worker pass holds it for ~10 s.
            raise Refusal("runner_overlap")
        return original(op, **fields)

    monkeypatch.setattr(bridge, "call", call)
    assert setting(bridge, objects, sleep=sleeps.append, monotonic=lambda: 0.25 * len(sleeps))["state"] == "applied"
    assert sleeps == [operator.LEASE_POLL_SECONDS] * 39 and calls.count("release") == 1


def test_lease_wait_is_bounded_and_names_the_overlap(fixture, monkeypatch):
    bridge, _, _ = fixture
    ticks, attempts = iter(range(0, 1000, 50)), []

    def overlapping(op, **fields):
        attempts.append(op)
        raise Refusal("runner_overlap")

    monkeypatch.setattr(bridge, "call", overlapping)
    with (pytest.raises(Refusal, match="runner_overlap"),
          operator.lease(bridge, sleep=lambda _: None, monotonic=lambda: next(ticks), wait=200)):
        pytest.fail("the lease was never acquired")
    assert attempts == ["acquire"] * 4


def test_object_readback_mismatch_refuses_before_the_control_swap(fixture):
    bridge, objects, _ = fixture
    objects.read = lambda uri: b"{}"
    with pytest.raises(Refusal, match="paid_expansion_object_readback_failed"):
        setting(bridge, objects)
    assert "paid_expansion" not in bridge.call("control")


def test_show_reports_unset_and_verifies_the_audit_chain(fixture):
    bridge, objects, _ = fixture
    shown = operator.show(bridge, objects)
    assert shown["state"] == "unset" and shown["audit_chain"] == [] and shown["audit_chain_verified"] is True
    setting(bridge, objects)
    assert operator.show(bridge, Objects())["object_verified"] is False
    assert operator.show(bridge, objects)["sources"] == list(allocation.SOURCES) == ["exa", "findall"]
    assert allocation.SCOPE["project_id"] == PROJECT and allocation.SCOPE["agent_id"] == AGENT


def control_from(open_bridge):
    fresh = open_bridge()  # Each CLI call is its own process over the persisted company store.
    try:
        return fresh.call("control")
    finally:
        fresh.close()


def test_cli_defaults_to_dry_run_and_apply_flag_is_required(fixture, capsys):
    bridge, objects, open_bridge = fixture
    bridge.close()
    args = ["set", "--per-run-usd", "20.00", "--approval-reference", "owner-decision-2026-10-04"]
    result = operator.main(args, bridge_factory=open_bridge, objects_factory=lambda _: objects, clock=lambda: NOW)
    assert result["state"] == "planned" and json.loads(capsys.readouterr().out)["state"] == "planned"
    assert "paid_expansion" not in control_from(open_bridge) and not objects.creates
    result = operator.main([*args, "--apply"], bridge_factory=open_bridge, objects_factory=lambda _: objects,
                           clock=lambda: NOW, sleep=lambda _: None)
    assert result["state"] == "applied" and control_from(open_bridge)["paid_expansion"]["enabled"] is True
    shown = operator.main(["show"], bridge_factory=open_bridge, objects_factory=lambda _: objects)
    assert shown["per_run_limit_usd"] == "20.00" and shown["object_verified"] is True
    assert operator.main(["disable"], bridge_factory=open_bridge)["state"] == "planned"
    assert operator.main(["disable", "--apply"], bridge_factory=open_bridge, sleep=lambda _: None)["state"] == "disabled"
    with pytest.raises(Refusal, match="paid_expansion_time_invalid"):
        operator.main([*args, "--expires-at", "2026-12-01T00:00:00"], bridge_factory=open_bridge,
                      objects_factory=lambda _: objects, clock=lambda: NOW)


def test_raising_the_amount_through_set_reaches_the_next_runs_exa_start(fixture, tmp_path):
    """The owner's one number binds: $10 to $30 needs no code change and lifts the next run to $15 per start."""
    bridge, objects, _ = fixture
    setting(bridge, objects, "10.00")
    ledger = FirestoreLedger(bridge)
    first = allocation.grant(ledger.paid_expansion_control(), {"run_key": "blueprint-researcher:2026-10-04",
                             "started_at": NOW.isoformat(), "research_runtime_seconds": 1200}, NOW)
    assert (first["limit_micros"], first["per_start_max_micros"]) == (10_000_000, 5_000_000)
    setting(bridge, objects, "30.00", now=NOW + timedelta(minutes=5))
    raised = ledger.paid_expansion_control()
    assert allocation.effective_limit(first, raised) == 10_000_000  # The run in progress stays frozen.
    day = datetime(2026, 10, 5, 12, 1, tzinfo=timezone.utc)
    row = {"date": "2026-10-05", "run_key": "blueprint-researcher:2026-10-05", "session_id": "sess_synthetic",
           "turn_id": "turn_synthetic", "state": "running", "started_at": day.isoformat(),
           "research_runtime_seconds": 1200, "recurring_budget_authority_reference": "synthetic_owner_authority",
           "soft_target_usd": 5}
    row["paid_expansion_grant"] = allocation.grant(raised, row, day)
    assert row["paid_expansion_grant"]["per_start_max_micros"] == 15_000_000
    store = Ledger(tmp_path / "exa")
    store.path.mkdir()
    store.put(row)
    transport = Transport(store, row)
    options = {"transport": transport, "control": raised, "tool_schema": SCHEMA, "now": day + timedelta(seconds=30)}
    refused = expansion.execute(expansion.START, {**ARGS, "max_cost_micros": 15_000_001}, row, store, **options)
    assert refused["reason"] == "expansion_cap_exceeds_per_start_maximum" and refused["max_start_micros"] == 15_000_000
    admitted = expansion.execute(expansion.START, {**ARGS, "max_cost_micros": 15_000_000}, row, store, **options)
    assert admitted["run_id"] == "agent_run_synthetic" and transport.starts[0]["budget"]["maxCostDollars"] == 15.0


def test_a_rollback_that_drops_the_control_copy_keeps_one_audit_chain(fixture, tmp_path):
    bridge, objects, _ = fixture
    first = setting(bridge, objects, "10.00")["next"]["sha256"]
    reopened = rewrite_store(fixture, tmp_path, lambda control: control.pop("paid_expansion"))
    try:
        assert operator.show(reopened, objects)["state"] == "unset"
        again = setting(reopened, objects, "20.00", now=NOW + timedelta(minutes=1))
        assert again["direction"]["version"] == 2 and again["direction"]["supersedes"] == first
        assert again["expected_sha256"] is None
        shown = operator.show(reopened, objects)
        assert shown["audit_chain_verified"] is True and shown["version"] == 2
    finally:
        reopened.close()


def test_set_replaces_a_current_direction_this_package_cannot_verify(fixture, tmp_path):
    bridge, objects, _ = fixture
    setting(bridge, objects, "10.00")

    def future_release(control):
        control["paid_expansion"]["current"]["direction"]["future_field"] = True

    reopened = rewrite_store(fixture, tmp_path, future_release)
    try:
        unknown = reopened.call("control")["paid_expansion"]["current"]
        assert operator.show(reopened, objects)["state"] == "unverified"
        assert operator.disable(reopened, apply=True)["state"] == "disabled"
        replaced = setting(reopened, objects, "10.00", now=NOW + timedelta(minutes=1))
        assert replaced["current_problem"] == "paid_expansion_direction_invalid"
        assert replaced["direction"]["supersedes"] == unknown["sha256"] and replaced["direction"]["version"] == 2
        assert reopened.call("control")["paid_expansion"]["enabled"] is True
    finally:
        reopened.close()


@pytest.mark.parametrize("value,expected", [("findall", ["findall"]), ("exa", ["exa"]),
                                            ("findall,exa", ["exa", "findall"]), (" exa , findall ", ["exa", "findall"])])
def test_set_admits_only_the_sources_the_owner_names(fixture, value, expected):
    bridge, objects, _ = fixture
    plan = setting(bridge, objects, "10.00", apply=False, sources=value)
    assert plan["direction"]["sources"] == expected == plan["next"]["sources"]


@pytest.mark.parametrize("value", ["", ",", "bing", "findall,findall", "exa,,findall", "FindAll"])
def test_set_refuses_unknown_empty_or_repeated_sources(fixture, value):
    bridge, objects, _ = fixture
    with pytest.raises(Refusal, match="^paid_expansion_sources_invalid$"):
        setting(bridge, objects, "10.00", apply=False, sources=value)


def test_default_sources_stay_every_supported_source(fixture):
    bridge, objects, _ = fixture
    assert setting(bridge, objects, "10.00", apply=False)["direction"]["sources"] == list(allocation.SOURCES)


def test_a_findall_only_direction_refuses_exa_and_admits_findall(fixture):
    bridge, objects, _ = fixture
    applied = setting(bridge, objects, "10.00", sources="findall")
    control = bridge.call("control")
    now = datetime(2026, 10, 5, 12, 5, tzinfo=timezone.utc)
    row = {"run_key": "blueprint-researcher:2026-10-05", "started_at": "2026-10-05T12:00:00+00:00",
           "research_runtime_seconds": 2700}
    grant = allocation.grant({**control, "source_commit": "a" * 40}, row, now)
    assert grant["state"] == "granted" and grant["sources"] == ["findall"]
    live = {**control, "source_commit": "a" * 40}
    assert allocation.standing(grant, live, now, source="exa") == "paid_expansion_source_not_directed"
    assert allocation.standing(grant, live, now, source="findall") is None
    assert applied["next"]["sources"] == ["findall"]


def test_readiness_reports_key_presence_by_name_only(fixture, monkeypatch):
    bridge, objects, _ = fixture
    monkeypatch.setenv("PARALLEL_API_KEY", "synthetic-secret-value")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    shown = operator.show(bridge, objects)
    assert shown["source_readiness"] == {
        "exa": {"credential_binding_name": "EXA_API_KEY", "credential_binding_present": False, "directed": False},
        "findall": {"credential_binding_name": "PARALLEL_API_KEY", "credential_binding_present": True, "directed": False}}
    plan = setting(bridge, objects, "10.00", apply=False, sources="exa,findall")
    assert plan["warnings"] == ["paid_expansion_source_binding_missing:exa"]
    assert "synthetic-secret-value" not in canonical(shown) + canonical(plan)
    assert setting(bridge, objects, "10.00", apply=False, sources="findall")["warnings"] == []


def test_cli_sources_flag_reaches_the_direction(fixture):
    _, objects, open_bridge = fixture
    result = operator.main(["set", "--per-run-usd", "10.00", "--sources", "findall",
                            "--approval-reference", "owner-decision-2026-10-04"],
                           bridge_factory=open_bridge, objects_factory=lambda _bridge: objects,
                           clock=lambda: datetime(2026, 10, 4, 18, 30, 15, tzinfo=timezone.utc))
    assert result["state"] == "planned" and result["direction"]["sources"] == ["findall"]
