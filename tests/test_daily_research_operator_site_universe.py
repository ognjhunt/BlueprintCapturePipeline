"""Hermetic site universe owner command: real private bridge and Store, in-memory Firestore, fake object store."""
import base64
import gzip
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.test_daily_research_site_universe import (
    attached_row,
    build_export,
    crm,
    formal,
    record_for,
    sha,
    site,
)
from tools.daily_research import site_universe as su
from tools.daily_research.firestore import Bridge
from tools.daily_research.runner import Refusal, canonical, digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_universe_backlog",
                                              ROOT / "tools/daily_research/operators/site-universe-backlog.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
NOW = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)  # 13:00 America/Chicago: the 2026-10-05 run is due.
CONFIG = {"enabled": False, "first_date": "2026-09-30", "discovery_profile": "adaptive-sites-v1",
          "max_runtime_seconds": 3600, "qa_reserved_seconds": 900, "research_contract_version": 3}
NAMES = ("Synthetic Works", "Synthetic Operator", "Example Road", "Fixture City")


@pytest.fixture
def fixture(tmp_path):
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
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    control["config"] = dict(CONFIG)
    bridge = Bridge(script=script)
    bridge.call("init", value=control)
    raw = build_export([site(number) for number in range(1, 13)])
    path = tmp_path / "backlog.v1.json.gz"
    path.write_bytes(raw)
    yield bridge, path, lambda: Bridge(script=script)
    bridge.close()


def named(result):
    text = canonical(result)
    return [name for name in NAMES if name in text]


def imported(bridge, row):
    bridge.call("acquire")
    try:
        bridge.call("import_run", row=row)
    finally:
        bridge.call("release")


def published(bridge, path):
    return operator.publish(bridge, path, apply=True)


def pinning(bridge, stored, **changes):
    options = {"sha256": stored["export"]["sha256"], "generation": stored["generation"],
               "approval_reference": "owner-synthetic-pin", "slice_size": 20, "now": NOW, "sleep": lambda _: None}
    options.update(changes)
    return operator.pin(bridge, **options)


def test_publish_validates_with_the_runtime_loader_and_is_create_only(fixture, tmp_path):
    bridge, path, _ = fixture
    plan = operator.publish(bridge, path)
    assert plan["state"] == "planned" and plan["object_writes"] == plan["firestore_writes"] == 0
    assert not (tmp_path / "bucket.json").exists() and plan["uri"] == su.object_uri(plan["export"]["sha256"])
    first = published(bridge, path)
    assert first["state"] == "published" and first["readback_verified"] is True and first["object_writes"] == 1
    assert published(bridge, path)["generation"] == first["generation"]  # Same bytes, same object.
    assert first["export"]["counts"]["rows"] == 12 and not named(plan) and not named(first)
    broken = tmp_path / "broken.json.gz"
    broken.write_bytes(gzip.compress(b'{"schema_version":"other"}', mtime=0))
    with pytest.raises(su.SiteUniverseError, match="^site_universe_export_invalid$"):
        operator.publish(bridge, broken, apply=True)
    large = tmp_path / "large.json.gz"
    large.write_bytes(b"\x1f\x8b" + b"\0" * su.MAX_OBJECT_BYTES)
    with pytest.raises(su.SiteUniverseError, match="^site_universe_object_too_large$"):
        operator.publish(bridge, large)
    assert "site_universe" not in bridge.call("control")


def test_pin_runs_the_loader_and_a_dry_run_before_the_fenced_swap(fixture):
    bridge, path, _ = fixture
    stored = published(bridge, path)
    plan = pinning(bridge, stored, apply=False)
    assert plan["state"] == "planned" and "site_universe" not in bridge.call("control")
    assert plan["dry_run"]["run_date"] == "2026-10-05" and len(plan["dry_run"]["site_ids"]) == 12
    assert plan["dry_run"]["crm_snapshot_included"] is False and plan["expected_sha256"] is None and not named(plan)
    applied = pinning(bridge, stored, apply=True)
    value = bridge.call("control")["site_universe"]
    assert applied["state"] == "pinned" and applied["readback_verified"] and value == applied["pin"]
    assert su.pin(value, 2700) == value and value["bytes"] == path.stat().st_size and value["slice_size"] == 20
    assert bridge.call("control")["lease"]["expires_at_ms"] == 0  # The lease is held only for the swap.
    shown = operator.show(bridge, now=NOW)
    assert shown["state"] == "enabled" and shown["ready"] is True and shown["firestore_writes"] == 0
    assert shown["dry_run"]["site_ids"] == applied["dry_run"]["site_ids"] and not named(shown)
    with pytest.raises(Refusal, match="^site_universe_control_conflict$"):
        pinning(bridge, stored, expect="none")
    again = pinning(bridge, stored, slice_size=25, expect=value["sha256"], apply=True)
    assert bridge.call("control")["site_universe"]["slice_size"] == 25 and again["expected_sha256"] == value["sha256"]


@pytest.mark.parametrize(("changes", "code"), [
    ({"approval_reference": "PENDING-owner"}, "site_universe_pin_invalid"),
    ({"slice_size": 31}, "site_universe_pin_invalid"),
    ({"reoffer_after_days": 0}, "site_universe_pin_invalid"),
    ({"generation": "999999"}, "site_universe_object_missing"),
    ({"generation": "01"}, "site_universe_pin_invalid"),
    ({"sha256": "0" * 64}, "site_universe_object_missing"),
])
def test_invalid_pins_refuse_before_any_write(fixture, changes, code):
    bridge, path, _ = fixture
    stored = published(bridge, path)
    with pytest.raises((Refusal, su.SiteUniverseError), match=f"^{code}$"):
        pinning(bridge, stored, apply=True, **changes)
    assert "site_universe" not in bridge.call("control")


def test_pin_respects_the_research_window_and_the_active_run(fixture):
    bridge, path, _ = fixture
    stored = published(bridge, path)
    bridge.call("acquire")
    bridge.call("configure", value={**bridge.call("control"), "config": {**CONFIG, "max_runtime_seconds": 1800,
                                                                       "qa_reserved_seconds": 600}})
    bridge.call("release")
    with pytest.raises(su.SiteUniverseError, match="^site_universe_slice_exceeds_research_window$"):
        pinning(bridge, stored)  # 1200 research seconds hold at most 13 sites.
    assert pinning(bridge, stored, slice_size=13)["state"] == "planned"
    imported(bridge, {"date": "2026-10-04", "run_key": "blueprint-researcher:2026-10-04",
                                   "metadata": {"run_key": "blueprint-researcher:2026-10-04"}, "state": "running",
                                   "cleanup_required": True})
    with pytest.raises(Refusal, match="^site_universe_run_active_apply_after_run$"):
        pinning(bridge, stored, slice_size=13, apply=True)
    assert pinning(bridge, stored, slice_size=13, apply=True, during_active_run=True)["state"] == "pinned"


def test_dry_run_selection_uses_history_and_the_canonical_crm(fixture):
    bridge, path, _ = fixture
    stored = published(bridge, path)
    snapshot = {"sheet_id": "synthetic", "complete": True, "captured_at": NOW.isoformat(),
                "values": crm(("Synthetic Operator 3", "Plant", "Fixture City"))}
    bridge.call("acquire")
    bridge.call("file_put", name="crm.json", bytes=base64.b64encode(json.dumps(snapshot).encode()).decode())
    bridge.call("release")
    prior = attached_row([site(number) for number in range(1, 13)], slice_size=5)
    prior.update(date="2026-10-04", run_key="blueprint-researcher:2026-10-04", state="completed",
                 metadata={**prior["metadata"], "run_key": "blueprint-researcher:2026-10-04"})
    packet = {"candidates": [], "site_universe": su.outcomes(prior, {"discovery_inventory": [record_for(1, "screened")]}, [], [])}
    prior.update(packet=packet, packet_digest=digest(packet))
    imported(bridge, prior)
    plan = pinning(bridge, stored)
    selection = plan["dry_run"]["selection"]
    assert plan["dry_run"]["crm_snapshot_included"] is True
    assert selection["removed"] == {"reoffer_window": 1, "untrusted_history": 0, "crm": 1, "prior_candidate": 0}
    assert sha("synthetic-site-1") not in plan["dry_run"]["site_ids"] and len(plan["dry_run"]["site_ids"]) == 10


def test_disable_keeps_the_pin_and_turns_the_slice_off(fixture):
    bridge, path, _ = fixture
    assert operator.disable(bridge)["state"] == "unset"
    stored = published(bridge, path)
    value = pinning(bridge, stored, apply=True)["pin"]
    plan = operator.disable(bridge)
    assert plan["state"] == "planned" and bridge.call("control")["site_universe"] == value
    done = operator.disable(bridge, apply=True, sleep=lambda _: None)
    assert done["state"] == "disabled" and bridge.call("control")["site_universe"] == {**value, "enabled": False}
    assert operator.show(bridge, now=NOW)["state"] == "disabled" and operator.disable(bridge)["state"] == "already_disabled"
    assert su.pin(bridge.call("control")["site_universe"]) is None  # The runner reads nothing more.


def test_disable_waits_for_an_active_run_unless_overridden(fixture):
    bridge, path, _ = fixture
    value = pinning(bridge, published(bridge, path), apply=True)["pin"]
    imported(bridge, {"date": "2026-10-04", "run_key": "blueprint-researcher:2026-10-04",
                      "metadata": {"run_key": "blueprint-researcher:2026-10-04"}, "state": "running", "cleanup_required": True})
    assert operator.disable(bridge)["state"] == "planned"  # A dry run never contends for the lease.
    with pytest.raises(Refusal, match="^site_universe_run_active_apply_after_run$"):
        operator.disable(bridge, apply=True, sleep=lambda _: None)
    assert bridge.call("control")["site_universe"] == value
    done = operator.disable(bridge, apply=True, during_active_run=True, sleep=lambda _: None)
    assert done["state"] == "disabled" and bridge.call("control")["site_universe"] == {**value, "enabled": False}


def test_show_reports_an_unusable_pin_with_its_code(fixture):
    bridge, path, _ = fixture
    assert operator.show(bridge, now=NOW)["state"] == "unset"
    stored = published(bridge, path)
    pinning(bridge, stored, apply=True)
    bridge.call("acquire")
    bridge.call("configure", value={**bridge.call("control"), "config": {**CONFIG, "max_runtime_seconds": 600,
                                                                       "qa_reserved_seconds": 300}})
    bridge.call("release")
    shown = operator.show(bridge, now=NOW)
    assert shown["state"] == "enabled_unusable" and shown["code"] == "site_universe_slice_exceeds_research_window"
    assert shown["ready"] is False


def test_funnel_reports_counts_and_ids_for_the_last_days_only(fixture):
    bridge, _, _ = fixture
    attached = attached_row([site(number) for number in range(1, 9)], slice_size=5)
    inventory = [record_for(1, "screened"), record_for(2, "candidate"), record_for(3, "rejected")]
    candidates = [formal(inventory[1], "k2")]
    packet = {"candidates": [], "site_universe": su.outcomes(attached, {"discovery_inventory": inventory}, candidates, [])}
    attached.update(date="2026-10-05", run_key="blueprint-researcher:2026-10-05", state="reviewed", packet=packet,
                    packet_digest=digest(packet), metadata={**attached["metadata"], "run_key": "blueprint-researcher:2026-10-05"},
                    review={"accepted_keys": ["k2"], "lead_verification": {"results": [
                        {"candidate_key": "k2", "eligible_for_qualified_promotion": True}]}})
    rows = [attached,
            {"date": "2026-10-04", "site_universe": su.refused("site_universe_object_missing"), "state": "completed"},
            {"date": "2026-10-03", "state": "completed"},
            {"date": "2026-09-20", "site_universe": su.refused("site_universe_pin_invalid"), "state": "completed"}]
    for row in rows:
        row.setdefault("run_key", "blueprint-researcher:" + row["date"])
        row.setdefault("metadata", {"run_key": row["run_key"]})
        imported(bridge, row)
    report = operator.funnel(bridge, days=7, now=NOW)
    assert (report["from"], report["to"]) == ("2026-09-29", "2026-10-05") and report["attached_runs"] == 1
    states = {run["date"]: run["site_universe"] for run in report["runs"]}
    assert set(states) == {"2026-10-03", "2026-10-04", "2026-10-05"} and states["2026-10-03"] == {"state": "off"}
    assert states["2026-10-04"] == {"state": "refused", "code": "site_universe_object_missing"}
    totals = report["totals"]
    assert totals["agent"]["screened"] == totals["agent"]["rejected"] == totals["agent"]["formal_candidates_linked"] == 1
    assert totals["agent"]["untouched"] == 2 and totals["selection"]["offered"] == 5
    assert totals["qa"] == {"eligible_for_promotion": {"slice": 1, "run": 1}, "accepted": {"slice": 1, "run": 1}}
    assert report["not_measured"]["contacted"] is None and not named(report)
    with pytest.raises(Refusal, match="^site_universe_funnel_days_invalid$"):
        operator.funnel(bridge, days=0, now=NOW)


def test_cli_commands_print_stable_results(fixture, capsys):
    bridge, path, open_bridge = fixture
    bridge.close()
    result = operator.main(["publish", "--file", str(path)], bridge_factory=open_bridge)
    assert result["state"] == "planned" and json.loads(capsys.readouterr().out)["state"] == "planned"
    stored = operator.main(["publish", "--file", str(path), "--apply"], bridge_factory=open_bridge)
    args = ["pin", "--sha256", stored["export"]["sha256"], "--generation", stored["generation"],
            "--approval-reference", "owner-synthetic-pin"]
    assert operator.main(args, bridge_factory=open_bridge, clock=lambda: NOW)["state"] == "planned"
    assert operator.main([*args, "--apply"], bridge_factory=open_bridge, clock=lambda: NOW,
                         sleep=lambda _: None)["state"] == "pinned"
    assert operator.main(["show"], bridge_factory=open_bridge, clock=lambda: NOW)["ready"] is True
    assert operator.main(["funnel", "--days", "7"], bridge_factory=open_bridge, clock=lambda: NOW)["runs"] == []
    assert operator.main(["disable", "--apply", "--during-active-run"], bridge_factory=open_bridge,
                         sleep=lambda _: None)["state"] == "disabled"
    assert not named(capsys.readouterr().out)
