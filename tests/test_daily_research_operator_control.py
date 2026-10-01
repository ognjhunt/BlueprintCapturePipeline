"""Hermetic migration checks; real private bridge and Store, no provider."""
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import Refusal, Runner, configuration, due_date

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oct2_control", ROOT / "tools/daily_research/operators/research-oct2-control.py")
oct2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oct2)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    script = tmp_path / "bridge.mjs"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline';",
        "import {Store, LeaseChannel} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "const db = new MemoryFirestore();const channel = new LeaseChannel(new Store(db));",
        "for await (const line of createInterface({input: process.stdin})) {",
        "try {const value = await channel.call(JSON.parse(line)); process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error) {process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}",
        "await channel.close();",
    ]))
    current = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    current.update(source_commit="b9aa5d0d129ecf02082a766e12657a94097e803a", legacy_attempts_reconciled_reference="verified-old-attempts")
    current["config"].update(scheduler_authority_reference="verified-sole-trigger")
    current["workflow"].update(qa_authority_reference="owner-reviewed-qa", publication_authority_reference="owner-reviewed-publication")
    bridge = Bridge(script=script)
    bridge.call("init", value=current)
    ledger = FirestoreLedger(bridge)
    raw = b'{"fixture":"synthetic failed research","candidates":[]}\n'
    monkeypatch.setattr(oct2, "RAW", hashlib.sha256(raw).hexdigest())
    row = {"date": oct2.DAY, "run_key": "blueprint-researcher:" + oct2.DAY,
           "metadata": {"purpose": "synthetic_failed_research", "fixture": "hermetic"},
           "state": "failed", "error": "knowledge_schema_invalid",
           "session_id": oct2.SESSION, "environment_id": oct2.ENVIRONMENT,
           "raw_output_digest": oct2.RAW, "cleanup_required": True, "delivery": {}}
    with ledger.lock():
        current["enabled"] = current["config"]["enabled"] = current["workflow"]["enabled"] = True
        bridge.call("configure", value=current)
        ledger.put({**row, "state": "creating"})
        ledger.put(row)
        ledger.write_bytes(oct2.DAY + "-artifact.json", raw)
    receipt = {"source_commit": oct2.SOURCE, "archive_sha256_reference": oct2.ARCHIVE, "files_verified": 41}
    yield bridge, ledger, current, row, raw, receipt
    bridge.close()


def test_plan_is_read_only_and_removes_legacy_authority(fixture):
    bridge, ledger, control, row, raw, receipt = fixture
    before = bridge.call("control")
    proposal = oct2.plan(bridge, receipt)
    assert bridge.call("control") == before
    assert ledger.get(oct2.DAY) == row
    assert ledger.read_bytes(oct2.DAY + "-artifact.json") == raw
    candidate = proposal["candidate"]
    assert candidate["enabled"] is candidate["config"]["enabled"] is False
    assert candidate["workflow"] == control["workflow"]
    assert candidate["source_commit"] == oct2.SOURCE
    assert candidate["config"]["soft_target_usd"] == 5
    assert candidate["config"]["recurring_budget_authority_reference"] == oct2.BUDGET_AUTHORITY
    assert candidate["config"]["max_runtime_seconds"] == 1800
    assert candidate["config"]["qa_reserved_seconds"] == 600
    assert candidate["config"]["first_date"] == control["config"]["first_date"]
    assert candidate["config"]["approval_reference"] == control["config"]["approval_reference"]
    assert candidate["config"]["scheduler_authority_reference"] == control["config"]["scheduler_authority_reference"]
    assert candidate["workflow"]["publication_authority_reference"] == "owner-reviewed-publication"
    assert proposal["firestore_writes"] == proposal["provider_calls"] == 0


def test_apply_disables_full_profile_and_preserves_failed_history(fixture):
    bridge, ledger, _, row, raw, receipt = fixture
    proposal = oct2.plan(bridge, receipt)
    result = oct2.apply_disabled(bridge, proposal, receipt)
    assert result["state"] == "configured_disabled"
    assert oct2.normalized(bridge.call("control")) == proposal["candidate"]
    assert ledger.get(oct2.DAY) == row
    assert ledger.read_bytes(oct2.DAY + "-artifact.json") == raw
    assert bridge.call("control")["lease"]["expires_at_ms"] == 0


def test_changed_control_refuses_without_overwriting(fixture):
    bridge, _, _, _, _, receipt = fixture
    proposal = oct2.plan(bridge, receipt)
    current = bridge.call("control")
    current["config"]["approval_reference"] = "different-owner-authority"
    with FirestoreLedger(bridge).lock():
        bridge.call("configure", value=current)
    with pytest.raises(Refusal, match="state_changed_replan_required"):
        oct2.apply_disabled(bridge, proposal, receipt)
    assert bridge.call("control")["config"]["approval_reference"] == "different-owner-authority"
    assert bridge.call("control")["enabled"] is True


def test_cleanup_change_refuses_stale_plan(fixture):
    bridge, ledger, _, row, _, receipt = fixture
    proposal = oct2.plan(bridge, receipt)
    changed = {**row, "cleanup_required": False, "cleanup_receipt": {"action_time_approval_reference": "approved-test"}}
    with ledger.lock():
        ledger.put(changed)
    with pytest.raises(Refusal, match="state_changed_replan_required"):
        oct2.apply_disabled(bridge, proposal, receipt)
    assert ledger.get(oct2.DAY) == changed


def test_tampered_candidate_or_package_refuses(fixture):
    bridge, _, _, _, _, receipt = fixture
    proposal = oct2.plan(bridge, receipt)
    proposal["candidate"]["enabled"] = True
    with pytest.raises(Refusal, match="receipt_invalid"):
        oct2.apply_disabled(bridge, proposal, receipt)


def test_pending_budget_cannot_activate(fixture):
    bridge, _, _, _, _, receipt = fixture
    candidate = oct2.plan(bridge, receipt)["candidate"]
    candidate["enabled"] = True
    candidate["config"].update(soft_target_usd=None,
                               recurring_budget_authority_reference="PENDING-owner-budget")
    from tools.daily_research.firestore import control_configuration
    with pytest.raises(Refusal, match="recurring_research_budget_not_approved"):
        configuration(control_configuration(candidate))


def test_approved_budget_still_leaves_runner_disabled(fixture):
    bridge, ledger, _, _, _, receipt = fixture
    candidate = oct2.plan(bridge, receipt)["candidate"]
    from tools.daily_research.firestore import control_configuration
    cfg = configuration(control_configuration(candidate))
    run = Runner(ledger, cfg, None, clock=lambda: datetime.fromisoformat("2026-10-02T12:00:00+00:00"))
    with pytest.raises(Refusal, match="runner_disabled"):
        run.start_or_resume()


@pytest.mark.parametrize(("when", "due"), [
    ("2026-10-01T21:00:00+00:00", "2026-10-01"),
    ("2026-10-02T11:59:59+00:00", "2026-10-01"),
    ("2026-10-02T12:00:00+00:00", "2026-10-02"),
])
def test_chicago_seven_am_preserves_actual_history_date(when, due):
    assert due_date(datetime.fromisoformat(when), "2026-09-30") == due


def test_pre_oct2_restart_returns_immutable_oct1_without_api(fixture):
    bridge, ledger, _, row, _, receipt = fixture
    candidate = oct2.plan(bridge, receipt)["candidate"]
    from tools.daily_research.firestore import control_configuration
    cfg = configuration(control_configuration(candidate))
    run = Runner(ledger, cfg, None, clock=lambda: datetime.fromisoformat("2026-10-02T11:59:59+00:00"))
    assert run.start_or_resume() == row


def test_raw_backup_is_bound(fixture):
    _, _, _, row, _, _ = fixture
    with pytest.raises(Refusal, match="history_or_raw_changed"):
        oct2.verify_failed(row, b"different bytes")
