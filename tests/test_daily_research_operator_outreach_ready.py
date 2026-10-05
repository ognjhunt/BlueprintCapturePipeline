"""Hermetic outreach-ready direction command: real private bridge and Store, in-memory Firestore, fake object store."""
import base64
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.daily_research import outreach_ready, verification
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import AGENT, PROJECT, Refusal, canonical
from tools.daily_research.standalone import FILES, PREFIX

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("outreach_ready_direction",
                                              ROOT / "tools/daily_research/operators/outreach-ready-direction.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
NOW = datetime(2026, 10, 5, 18, 30, 15, 123456, tzinfo=timezone.utc)
APPROVAL = "owner-decision-outreach-ready-synthetic"


def bridge_script(tmp_path):
    script = tmp_path / "bridge.mjs"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline';",
        "import {Store, LeaseChannel} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "import {FakeBucket} from " + json.dumps((ROOT / "tests/fixtures/daily_research/fake-bucket.mjs").as_uri()) + ";",
        "const db = new MemoryFirestore(" + json.dumps(str(tmp_path / "firestore.json")) + ");",
        "const bucket = new FakeBucket(" + json.dumps(str(tmp_path / "bucket.json")) + ");",
        "const channel = new LeaseChannel(new Store(db, undefined, undefined, null, null, null, null, false, bucket));",
        "for await (const line of createInterface({input: process.stdin})) {",
        "try {const value = await channel.call(JSON.parse(line)); process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error) {process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}",
        "await channel.close();",
    ]))
    return script


@pytest.fixture
def fixture(tmp_path):
    script = bridge_script(tmp_path)
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    bridge = Bridge(script=script)
    bridge.call("init", value=control)
    yield bridge, operator.BridgeObjects(bridge), lambda: Bridge(script=script)
    bridge.close()


def setting(bridge, objects, **changes):
    options = {"apply": True, "now": NOW, "paths": "daily_qa", "max_rows_per_batch": 50,
               "approval_reference": APPROVAL, "approved_by": "owner", "reason": "Owner outreach-ready direction",
               "sleep": lambda _: None}
    options.update(changes)
    return operator.set_direction(bridge, objects, **options)


def test_set_dry_run_is_read_only_and_names_the_exact_next_direction(fixture, tmp_path):
    bridge, objects, _ = fixture
    before = bridge.call("control")
    plan = setting(bridge, objects, apply=False)
    assert bridge.call("control") == before and not (tmp_path / "bucket.json").exists()
    assert plan["state"] == "planned" and plan["firestore_writes"] == plan["object_writes"] == plan["provider_calls"] == 0
    assert plan["direction"] == {
        "schema_version": outreach_ready.DIRECTION, "version": 1, "supersedes": None,
        "rule_version": verification.OUTREACH_RULE_VERSION,
        "scope": {"paths": ["daily_qa"], "label": "hypothesis", "max_rows_per_batch": 50, "sends_authorized": False},
        "binding": {"project_id": PROJECT, "agent_id": AGENT, "firestore_root": "blueprintDailyResearch/sites-first",
                    "run_key_prefix": "blueprint-researcher:", "timezone": "America/Chicago"},
        "effective_from": "2026-10-05T18:30:15+00:00", "expires_at": "2026-11-04T18:30:15+00:00",
        "approval_reference": APPROVAL, "approved_by": "owner", "issued_at": "2026-10-05T18:30:15+00:00",
        "reason": "Owner outreach-ready direction"}
    assert plan["next"]["sha256"] == outreach_ready.digest(plan["direction"]) and plan["expected_sha256"] is None
    assert plan["sends_authorized"] is False and plan["next"]["sends_authorized"] is False


def test_apply_pins_the_create_only_object_by_generation_and_sha_then_reads_both_back(fixture):
    bridge, objects, _ = fixture
    applied = setting(bridge, objects)
    sha, generation = applied["next"]["sha256"], applied["object"]["generation"]
    assert applied["state"] == "applied" and applied["readback_verified"] is True and generation.isdigit()
    pin = bridge.call("control")["outreach_ready"]
    assert pin == {"enabled": True, "current": {"sha256": sha, "generation": generation, "version": 1,
                                                "uri": outreach_ready.uri(sha), "direction": applied["direction"]}}
    assert outreach_ready.current(bridge.call("control"))[1] is None
    assert objects.read(sha, generation) == canonical(applied["direction"]).encode()
    assert bridge.call("control")["lease"]["expires_at_ms"] == 0  # The lease is held only for the swap.
    shown = operator.show(bridge, objects, now=NOW)
    assert shown["state"] == "enabled" and shown["object_verified"] is True and shown["generation"] == generation
    assert shown["next_run"] == {"state": "enabled", "code": None, "paths": ["daily_qa"], "max_rows_per_batch": 50}
    again = setting(bridge, objects, paths="daily_qa,site_screen", max_rows_per_batch=20, now=NOW + timedelta(minutes=1))
    assert again["direction"]["version"] == 2 and again["direction"]["supersedes"] == sha
    assert again["direction"]["scope"]["paths"] == ["daily_qa", "site_screen"] and again["expected_sha256"] == sha
    assert shown["firestore_writes"] == shown["object_writes"] == shown["provider_calls"] == 0


@pytest.mark.parametrize(("changes", "code"), [
    ({"approval_reference": "PENDING-owner"}, "outreach_ready_direction_invalid"),
    ({"approval_reference": " "}, "outreach_ready_direction_invalid"),
    ({"reason": "café"}, "outreach_ready_direction_invalid"),
    ({"expires_at": NOW + timedelta(days=367)}, "outreach_ready_direction_invalid"),
    ({"expires_at": NOW - timedelta(seconds=1)}, "outreach_ready_direction_invalid"),
    ({"paths": "daily_qa,daily_qa"}, "outreach_ready_paths_invalid"),
    ({"paths": "crm"}, "outreach_ready_paths_invalid"),
    ({"paths": ""}, "outreach_ready_paths_invalid"),
    ({"max_rows_per_batch": 0}, "outreach_ready_rows_invalid"),
    ({"max_rows_per_batch": 51}, "outreach_ready_rows_invalid"),
])
def test_invalid_direction_fields_refuse_before_any_write(fixture, tmp_path, changes, code):
    bridge, objects, _ = fixture
    with pytest.raises(Refusal, match=f"^{code}$"):
        setting(bridge, objects, **changes)
    assert not (tmp_path / "bucket.json").exists() and "outreach_ready" not in bridge.call("control")


def test_the_bridge_refuses_a_pin_that_is_not_its_generation_bytes_or_chain(fixture):
    bridge, objects, _ = fixture
    _, entry = operator.next_entry(bridge.call("control"), paths="daily_qa", max_rows_per_batch=50,
                                   approval_reference=APPROVAL, approved_by="owner", reason="Synthetic", now=NOW)
    raw = canonical(entry["direction"]).encode()
    generation = objects.create(entry["sha256"], raw)
    assert objects.create(entry["sha256"], raw) == generation  # Same bytes, same object.
    set_ = lambda value, expected=None: bridge.call("outreach_ready_set", expected_sha256=expected, value=value)
    bridge.call("acquire")
    try:
        for changes, code in (({"generation": "999999"}, "outreach_ready_object_missing"),
                              ({"generation": "01"}, "outreach_ready_direction_digest_mismatch"),
                              ({"version": 2}, "outreach_ready_direction_digest_mismatch"),
                              ({"uri": "gs://other/direction.json"}, "outreach_ready_direction_digest_mismatch"),
                              ({"direction": {**entry["direction"], "scope": {**entry["direction"]["scope"], "sends_authorized": True}}},
                               "outreach_ready_scope_invalid"),
                              ({"direction": {**entry["direction"], "scope": {**entry["direction"]["scope"], "label": "verified"}}},
                               "outreach_ready_scope_invalid")):
            with pytest.raises(Refusal, match=f"^{code}$"):
                set_({"enabled": True, "current": {**entry, "generation": generation, **changes}})
        with pytest.raises(Refusal, match="^outreach_ready_direction_chain_invalid$"):
            set_({"enabled": False, "current": {**entry, "generation": generation}})
        with pytest.raises(Refusal, match="^outreach_ready_request_invalid$"):
            set_({"enabled": True, "current": entry})
        assert "outreach_ready" not in bridge.call("control")
        set_({"enabled": True, "current": {**entry, "generation": generation}})
        with pytest.raises(Refusal, match="^outreach_ready_direction_conflict$"):
            set_({"enabled": True, "current": {**entry, "generation": generation}})
    finally:
        bridge.call("release")
    with pytest.raises(Refusal, match="^outreach_ready_pin_invalid$"):
        bridge.call("outreach_ready_object_get", sha256="../x", generation=generation)
    with pytest.raises(Refusal, match="^outreach_ready_direction_digest_mismatch$"):
        bridge.call("outreach_ready_object_put", sha256="0" * 64, bytes=base64.b64encode(raw).decode("ascii"))
    with pytest.raises(Refusal, match="^outreach_ready_direction_invalid$"):
        bridge.call("outreach_ready_object_put", sha256="0" * 64, bytes=base64.b64encode(b"{}").decode("ascii"))


def test_stale_expectation_and_a_racing_writer_are_compare_and_swap_conflicts(fixture):
    bridge, objects, _ = fixture
    first = setting(bridge, objects)["next"]["sha256"]
    with pytest.raises(Refusal, match="^outreach_ready_direction_conflict$"):
        setting(bridge, objects, expect="none")
    with pytest.raises(Refusal, match="^outreach_ready_direction_conflict$"):
        setting(bridge, objects, expect="0" * 64)

    class Racing:
        """Another owner command lands between this command's read and its fenced swap."""

        def __init__(self, inner):
            self.inner, self.raced = inner, False

        def call(self, op, **fields):
            if op == "outreach_ready_set" and not self.raced:
                self.raced = True
                rival = operator.next_entry(self.inner.call("control"), paths="daily_qa", max_rows_per_batch=5,
                                            approval_reference="rival", approved_by="owner", reason="rival", now=NOW)[1]
                rival["generation"] = objects.create(rival["sha256"], canonical(rival["direction"]).encode())
                self.inner.call("outreach_ready_set", expected_sha256=first, value={"enabled": True, "current": rival})
            return self.inner.call(op, **fields)

    with pytest.raises(Refusal, match="^outreach_ready_direction_conflict$"):
        setting(Racing(bridge), objects, max_rows_per_batch=10, expect=first)
    current = bridge.call("control")["outreach_ready"]["current"]
    assert current["direction"]["scope"]["max_rows_per_batch"] == 5 and current["version"] == 2
    assert bridge.call("control")["lease"]["expires_at_ms"] == 0


def test_disable_is_an_immediate_brake_and_only_a_new_version_re_enables(fixture):
    bridge, objects, _ = fixture
    assert operator.disable(bridge, apply=True)["state"] == "unset"
    first = setting(bridge, objects)
    plan = operator.disable(bridge)
    assert plan["state"] == "planned" and bridge.call("control")["outreach_ready"]["enabled"] is True
    braked = operator.disable(bridge, apply=True, sleep=lambda _: None)
    assert braked["state"] == "disabled" and braked["object_writes"] == 0 and braked["readback_verified"] is True
    pin = bridge.call("control")["outreach_ready"]
    assert pin["enabled"] is False and pin["current"]["sha256"] == first["next"]["sha256"]
    assert outreach_ready.current(bridge.call("control")) == (None, "outreach_ready_disabled")
    assert operator.show(bridge, objects, now=NOW)["state"] == "disabled"
    assert operator.show(bridge, objects, now=NOW)["next_run"] == {"state": "shadow"}
    assert operator.disable(bridge, apply=True)["state"] == "already_disabled"
    bridge.call("acquire")
    try:
        with pytest.raises(Refusal, match="^outreach_ready_reenable_requires_new_direction$"):
            bridge.call("outreach_ready_set", expected_sha256=pin["current"]["sha256"],
                        value={"enabled": True, "current": pin["current"]})
    finally:
        bridge.call("release")
    again = setting(bridge, objects, now=NOW + timedelta(minutes=1))
    assert again["direction"]["version"] == 2 and bridge.call("control")["outreach_ready"]["enabled"] is True


def test_apply_waits_for_an_active_run_unless_the_owner_overrides_and_the_brake_never_waits(fixture):
    bridge, objects, _ = fixture
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        bridge.call("configure", value={**{k: v for k, v in bridge.call("control").items() if k != "lease"}, "enabled": True})
        ledger.put({"date": "2026-10-05", "run_key": "blueprint-researcher:2026-10-05", "state": "creating",
                    "metadata": {"purpose": "synthetic_active_run"}, "cleanup_required": True, "delivery": {}})
    with pytest.raises(Refusal, match="^outreach_ready_run_active_apply_after_run$"):
        setting(bridge, objects)
    assert "outreach_ready" not in bridge.call("control")
    assert setting(bridge, objects, during_active_run=True)["state"] == "applied"
    assert operator.disable(bridge, apply=True)["state"] == "disabled"


def test_object_readback_mismatch_refuses_before_the_control_swap(fixture):
    bridge, _, _ = fixture

    class Wrong(operator.BridgeObjects):
        def read(self, sha256, generation):
            return b"{}"

    with pytest.raises(Refusal, match="^outreach_ready_object_readback_failed$"):
        setting(bridge, Wrong(bridge))
    assert "outreach_ready" not in bridge.call("control")


def test_configure_keeps_the_pin_and_neither_configure_nor_init_can_write_one(fixture, tmp_path):
    bridge, objects, _ = fixture
    setting(bridge, objects)
    pin = bridge.call("control")["outreach_ready"]
    replacement = {k: v for k, v in bridge.call("control").items() if k not in {"lease", "outreach_ready"}}
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        bridge.call("configure", value=replacement)
        assert bridge.call("control")["outreach_ready"] == pin
        bridge.call("configure", value={**replacement, "outreach_ready": pin})
        for value in ({**pin, "enabled": False}, None, {"enabled": True}):
            with pytest.raises(Refusal, match="^outreach_ready_requires_direction_operation$"):
                bridge.call("configure", value={**replacement, "outreach_ready": value})
    assert bridge.call("control")["outreach_ready"] == pin
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    other = Bridge(script=bridge_script(fresh))
    try:
        with pytest.raises(Refusal, match="^outreach_ready_requires_direction_operation$"):
            other.call("init", value={**replacement, "outreach_ready": pin})
        assert other.call("control") is None
    finally:
        other.close()


def test_the_run_freezes_only_an_enabled_daily_qa_direction(fixture):
    bridge, objects, _ = fixture
    ledger = FirestoreLedger(bridge)
    row = {"run_key": "blueprint-researcher:2026-10-05"}
    assert ledger.outreach_ready_control() is None and outreach_ready.freeze(None, row, NOW) is None
    applied = setting(bridge, objects, max_rows_per_batch=12)
    frozen = outreach_ready.freeze(ledger.outreach_ready_control(), row, NOW + timedelta(hours=1))
    assert frozen == {"schema_version": outreach_ready.ADMISSION, "state": "enabled", "run_key": row["run_key"],
                      "frozen_at": (NOW + timedelta(hours=1)).isoformat(), "direction_sha256": applied["next"]["sha256"],
                      "generation": applied["object"]["generation"], "version": 1, "uri": applied["object"]["uri"],
                      "rule_version": verification.OUTREACH_RULE_VERSION, "paths": ["daily_qa"], "label": "hypothesis",
                      "max_rows_per_batch": 12, "sends_authorized": False, "approval_reference": APPROVAL,
                      "valid_until": applied["direction"]["expires_at"]}
    assert outreach_ready.enabled({"outreach_ready": frozen}) and not outreach_ready.enabled({"outreach_ready": frozen}, "site_screen")
    expired = outreach_ready.freeze(ledger.outreach_ready_control(), row, NOW + timedelta(days=31))
    assert expired["state"] == "refused" and expired["code"] == "outreach_ready_expired"
    early = outreach_ready.freeze(ledger.outreach_ready_control(), row, NOW - timedelta(seconds=1))
    assert early["code"] == "outreach_ready_not_yet_effective"
    setting(bridge, objects, paths="site_screen", now=NOW + timedelta(minutes=1))
    # A screen-only direction does not concern daily runs: nothing is frozen, so the row and its manifest
    # stay exactly as without the feature. assess() and show still name the code.
    assert outreach_ready.freeze(ledger.outreach_ready_control(), row, NOW + timedelta(hours=1)) is None
    screen_only = outreach_ready.assess(ledger.outreach_ready_control(), row, NOW + timedelta(hours=1))
    assert screen_only["state"] == "refused" and screen_only["code"] == "outreach_ready_daily_qa_not_directed"
    assert operator.show(bridge, objects, now=NOW + timedelta(hours=1))["next_run"] == {
        "state": "shadow", "code": "outreach_ready_daily_qa_not_directed"}
    operator.disable(bridge, apply=True, sleep=lambda _: None)
    assert outreach_ready.freeze(ledger.outreach_ready_control(), row, NOW + timedelta(hours=1)) is None
    tampered = {"enabled": True, "current": {**bridge.call("control")["outreach_ready"]["current"], "version": 9}}
    assert outreach_ready.freeze(tampered, row, NOW)["code"] == "outreach_ready_direction_digest_mismatch"


def test_live_admission_tightens_and_brakes_but_never_widens_a_frozen_run(fixture):
    bridge, objects, _ = fixture
    ledger = FirestoreLedger(bridge)
    setting(bridge, objects, max_rows_per_batch=10)
    row = {"run_key": "blueprint-researcher:2026-10-05"}
    row["outreach_ready"] = outreach_ready.freeze(ledger.outreach_ready_control(), row, NOW)
    later = NOW + timedelta(hours=1)
    assert outreach_ready.admission(row, bridge.call("control"), later) == (10, None)
    setting(bridge, objects, max_rows_per_batch=50, now=NOW + timedelta(minutes=1))
    assert outreach_ready.admission(row, bridge.call("control"), later) == (10, None)  # Never widened mid-run.
    setting(bridge, objects, max_rows_per_batch=3, now=NOW + timedelta(minutes=2))
    assert outreach_ready.admission(row, bridge.call("control"), later) == (3, None)
    setting(bridge, objects, paths="site_screen", now=NOW + timedelta(minutes=3))
    assert outreach_ready.admission(row, bridge.call("control"), later) == (0, "outreach_ready_path_not_directed")
    setting(bridge, objects, now=NOW + timedelta(minutes=4))
    operator.disable(bridge, apply=True, sleep=lambda _: None)
    assert outreach_ready.admission(row, bridge.call("control"), later) == (0, "outreach_ready_disabled")
    assert outreach_ready.admission({}, bridge.call("control"), later) == (0, "outreach_ready_not_enabled_for_run")
    assert outreach_ready.admission(row, None, later) == (0, "outreach_ready_disabled")


def test_a_row_cannot_gain_or_change_its_frozen_record_after_the_durable_intent(fixture, tmp_path):
    bridge, objects, _ = fixture
    ledger = FirestoreLedger(bridge)
    setting(bridge, objects)
    frozen = outreach_ready.freeze(ledger.outreach_ready_control(), {"run_key": "blueprint-researcher:2026-10-05"}, NOW)
    base = {"date": "2026-10-05", "run_key": "blueprint-researcher:2026-10-05", "state": "creating",
            "metadata": {"purpose": "synthetic_frozen_record"}, "cleanup_required": True, "delivery": {}}
    plain = {**base, "date": "2026-10-06", "run_key": "blueprint-researcher:2026-10-06"}
    with ledger.lock():
        bridge.call("configure", value={**{k: v for k, v in bridge.call("control").items() if k not in {"lease"}}, "enabled": True})
        ledger.put({**base, "outreach_ready": frozen})
        ledger.put({**base, "outreach_ready": frozen, "state": "running"})
        for changed in ({**frozen, "max_rows_per_batch": 50, "version": 2}, None):
            row = {**base, "state": "running", **({"outreach_ready": changed} if changed else {})}
            with pytest.raises(Refusal, match="^outreach_ready_already_bound$"):
                ledger.put(row)
        manifests = lambda: dict(json.loads((tmp_path / "firestore.json").read_text()))
        bound = manifests()["blueprintDailyResearch/sites-first/runs/2026-10-05"]
        assert bound["outreach_ready_digest"] and "outreach_ready_unbound" not in bound
        ledger.put(plain)
        assert not {"outreach_ready_digest", "outreach_ready_unbound"} & set(manifests()["blueprintDailyResearch/sites-first/runs/2026-10-06"])
        # A record added after the intent cannot be told from one whose digest an older bridge dropped. Like
        # paid_expansion_grant_unbound the row is never stranded: it binds the record from then on, stays
        # unbound, and its publication withholds every hypothesis (daily_research_publisher.test.mjs).
        ledger.put({**plain, "state": "running", "outreach_ready": frozen})
        late = manifests()["blueprintDailyResearch/sites-first/runs/2026-10-06"]
        assert late["outreach_ready_unbound"] is True and late["outreach_ready_digest"] == bound["outreach_ready_digest"]
        for changed in ({**frozen, "max_rows_per_batch": 49}, None):
            with pytest.raises(Refusal, match="^outreach_ready_already_bound$"):
                ledger.put({**plain, "state": "running", **({"outreach_ready": changed} if changed else {})})


def test_cli_defaults_to_dry_run_and_apply_flag_is_required(fixture, capsys):
    bridge, _, open_bridge = fixture
    bridge.close()
    objects = operator.BridgeObjects
    args = ["set", "--approval-reference", APPROVAL]
    result = operator.main(args, bridge_factory=open_bridge, objects_factory=objects, clock=lambda: NOW)
    assert result["state"] == "planned" and json.loads(capsys.readouterr().out)["state"] == "planned"
    fresh = open_bridge()
    try:
        assert "outreach_ready" not in fresh.call("control")
    finally:
        fresh.close()
    result = operator.main([*args, "--max-rows-per-batch", "25", "--apply"], bridge_factory=open_bridge,
                           objects_factory=objects, clock=lambda: NOW, sleep=lambda _: None)
    assert result["state"] == "applied" and result["direction"]["scope"]["max_rows_per_batch"] == 25
    shown = operator.main(["show"], bridge_factory=open_bridge, objects_factory=objects, clock=lambda: NOW)
    assert shown["state"] == "enabled" and shown["object_verified"] is True
    assert operator.main(["disable"], bridge_factory=open_bridge)["state"] == "planned"
    assert operator.main(["disable", "--apply"], bridge_factory=open_bridge, sleep=lambda _: None)["state"] == "disabled"
    with pytest.raises(Refusal, match="^outreach_ready_time_invalid$"):
        operator.main([*args, "--expires-at", "2026-12-01T00:00:00"], bridge_factory=open_bridge,
                      objects_factory=objects, clock=lambda: NOW)


def test_release_packages_the_outreach_ready_module_and_owner_command():
    for name in ("outreach_ready.py", "operators/outreach-ready-direction.py"):
        assert name in FILES and (ROOT / PREFIX / name).is_file()
