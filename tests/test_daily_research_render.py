"""Real private-pipe adapter recovery with hermetic Node Firestore transactions."""
import base64
import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.test_daily_research_runner import DAY, NOW, SHEET, FakeAPI
from tools.daily_research import render
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import Refusal, Runner, preflight, save_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path):
    script = tmp_path / "bridge.mjs"
    # Only the driver is fake. Python Bridge/FirestoreLedger and JS Store are real.
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
    bridge = Bridge(script=script)
    bridge.call("init", value={"schema_version": "blueprint.research-control.v1", "enabled": False})
    bridge.call("acquire")
    bridge.call("configure", value={"schema_version": "blueprint.research-control.v1", "enabled": True})
    bridge.call("release")
    headers = ["Prospect ID", "Organization", "Prospect type", "Site / team", "Contact name",
               "", "", "", "", "Task evidence URL", "", "", "", "", "Task / job"]
    crm = tmp_path / "crm.json"
    save_json(crm, {"sheet_id": SHEET, "complete": True, "captured_at": NOW.isoformat(),
                    "values": [["CRM"], [], [], [], headers]})
    config = {"enabled": True, "first_date": DAY, "approval_reference": "owner-soft-one",
              "scheduler_authority_reference": "verified-cutover", "crm_snapshot": str(crm), "soft_target_usd": 1}
    ledger, api = FirestoreLedger(bridge), FakeAPI()
    original = api.create

    def create(payload):
        persisted = ledger.get(DAY)
        assert persisted["state"] == "creating" and persisted["create_payload"] == payload
        assert persisted["crm_snapshot"]["complete"] is True
        bridge.call("create_check", day=DAY, metadata=payload["metadata"])
        return original(payload)

    api.create = create
    run = Runner(ledger, config, api, clock=lambda: NOW)
    yield run, api, ledger, bridge, script
    bridge.close()


def test_restart_without_local_ledger_recovers_exact_terminal_artifact(fixture):
    run, api, _ledger, bridge, script = fixture
    api.lost_create_reply = True
    assert run.start_or_resume()["state"] == "creation_unresolved"
    bridge.close()
    restarted = Bridge(script=script)
    try:
        durable = FirestoreLedger(restarted)
        result = Runner(durable, run.config, api, clock=lambda: NOW).start_or_resume(allow_create=False)
        assert result["state"] == "awaiting_review" and len(api.payloads) == 1
        assert durable.read_bytes(DAY + "-artifact.json") == api.raw
        assert hashlib.sha256(api.raw).hexdigest() == result["raw_output_digest"]
        assert durable.get(DAY)["session_id"] == result["session_id"]
        with pytest.raises(Refusal, match="not_admitted"), durable.lock():
            restarted.call("create_check", day=DAY, metadata=result["metadata"])
    finally:
        restarted.close()


def test_export_verifies_all_run_bindings_and_rejects_replacement(fixture, tmp_path):
    run, _, _, bridge, _ = fixture
    run.start_or_resume()
    assert render.export_snapshot(bridge, DAY, tmp_path / "export")["missing_files"] == []
    snapshot = bridge.call("snapshot", day=DAY)
    snapshot["files"]["artifact"] = base64.b64encode(b"different bytes").decode()
    class Changed:
        def call(self, *args, **kwargs):
            return snapshot
    with pytest.raises(Refusal, match="digest_mismatch"):
        render.export_snapshot(Changed(), DAY, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    snapshot = bridge.call("snapshot", day=DAY)
    snapshot["files"]["output"] = base64.b64encode(b'{"different":"findings"}').decode()
    with pytest.raises(Refusal, match="output_artifact_binding_mismatch"):
        render.export_snapshot(Changed(), DAY, tmp_path / "rejected-output")
    assert not (tmp_path / "rejected-output").exists()


def test_timeout_poisons_pipe_and_cannot_accept_a_late_reply(fixture, monkeypatch):
    _, _, _, bridge, _ = fixture
    from tools.daily_research import firestore
    monkeypatch.setattr(firestore.select, "select", lambda *args: ([], [], []))
    with pytest.raises(Refusal, match="bridge_deadline"):
        bridge.call("control")
    assert bridge.process.poll() is not None
    with pytest.raises(Refusal, match="bridge_unavailable"):
        bridge.call("release")


def test_enabled_render_control_cannot_omit_instruction_pin():
    from tools.daily_research.firestore import control_configuration
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    control["enabled"] = True
    del control["config"]["expected_agent_instructions_sha256"]
    with pytest.raises(Refusal, match="pin_required"):
        control_configuration(control)


def test_cleanup_and_next_date_guard_use_firestore_bytes(fixture):
    run, api, ledger, _, _ = fixture
    result = run.start_or_resume()
    run.clock = lambda: NOW + timedelta(days=1)
    with pytest.raises(Refusal, match="cleanup_unresolved"):
        run.start_or_resume()
    api.absent = True
    receipt = {"session_id": result["session_id"], "environment_id": result["environment_id"],
               "action_time_approval_reference": "exact-session-approval"}
    assert run.record_cleanup(DAY, receipt)["cleanup_required"] is False
    assert ledger.read_bytes(DAY + "-artifact.json") == api.raw


def test_missing_file_is_distinct_from_corrupt_manifest(fixture):
    _, _, ledger, bridge, _ = fixture
    with pytest.raises(FileNotFoundError):
        ledger.read_bytes(DAY + "-artifact.json")
    # Invalid/missing blob manifest must propagate, never masquerade as a missing file.
    class Corrupt:
        def call(self, *args, **kwargs):
            raise Refusal("firestore_blob_digest_mismatch")
    with pytest.raises(Refusal, match="digest_mismatch"):
        FirestoreLedger(Corrupt()).read_bytes(DAY + "-artifact.json")
    with ledger.lock():
        bridge.call("file_put", name=DAY + "-artifact.json", bytes=base64.b64encode(b"retained").decode())
    assert ledger.read_bytes(DAY + "-artifact.json") == b"retained"


@pytest.mark.parametrize(("now", "expected"), [
    ("2026-10-31T23:00:00+00:00", "2026-11-01T13:00:00+00:00"),
    ("2027-03-13T23:00:00+00:00", "2027-03-14T12:00:00+00:00"),
])
def test_render_next_seven_am_handles_dst(now, expected):
    from datetime import datetime
    assert render.next_wake(datetime.fromisoformat(now)).isoformat() == expected


def test_idle_scheduler_does_not_repeat_ledger_work(monkeypatch):
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    clock = [NOW]
    calls, invoked = [], []

    class IdleBridge:
        def call(self, op):
            calls.append(op)
            assert op == "control"
            return control
        def close(self):
            calls.append("close")

    class Stopped:
        ticks = 0
        def is_set(self):
            return self.ticks >= 4
        def wait(self, delay):
            self.ticks += 1
            clock[0] += timedelta(seconds=delay)

    monkeypatch.setattr(render, "invoke", lambda *args, **kwargs: invoked.append(args[0]) or {"state": "nothing_to_reconcile"})
    render.scheduler(Stopped(), bridge_factory=IdleBridge, clock=lambda: clock[0])
    assert invoked == ["reconcile"] and calls == ["control"] * 4 + ["close"]


def test_selected_instruction_hash_drift_refuses_before_create():
    api = FakeAPI()
    api.agent["instructions"] = "reviewed instructions"
    sha = hashlib.sha256(api.agent["instructions"].encode()).hexdigest()
    assert preflight(api, sha)["instructions_sha256"] == sha
    api.agent["instructions"] = "changed"
    with pytest.raises(Refusal, match="pin_mismatch"):
        preflight(api, sha)


@pytest.mark.parametrize("command", ["run", "reconcile"])
@pytest.mark.parametrize("boundary", [
    "eligible", "stopped", "workflow_disabled", "control_disabled",
    "canary", "artifact_missing", "turn_failed", "qa_started", "publication_started",
])
def test_completed_validation_failure_enters_existing_workflow_only_when_eligible(
        fixture, tmp_path, monkeypatch, command, boundary):
    run, api, ledger, bridge, _ = fixture
    api.raw = b"{invalid retained research"
    failed = run.start_or_resume()
    assert failed["state"] == "failed" and failed["turn_status"] == "completed"
    assert failed["artifact_downloaded"] is True
    retained = ledger.read_bytes(DAY + "-artifact.json")
    with ledger.lock():
        control = bridge.call("control")
        control["workflow"] = {"enabled": boundary != "workflow_disabled",
                               "qa_authority_reference": "approved-same-session-QA",
                               "publication_authority_reference": "approved-fixed-targets"}
        control["enabled"] = boundary != "control_disabled"
        bridge.call("configure", value=control)
        if boundary == "canary":
            failed["canary"] = {"test_id": "retained-canary"}
        elif boundary == "artifact_missing":
            failed["artifact_downloaded"] = False
        elif boundary == "turn_failed":
            failed["turn_status"] = "failed"
        elif boundary == "qa_started":
            failed["qa"] = {"state": "validated"}
        elif boundary == "publication_started":
            failed["delivery"] = {"notion": {"state": "pending"}}
        ledger.put(failed)
    delegated, creation_modes = [], []
    def stopped():
        return boundary == "stopped"
    def api_factory(*_args):
        return api
    def runner_factory(*args):
        actual = Runner(*args, clock=lambda: NOW)
        start = actual.start_or_resume
        def observed_start(*, allow_create):
            creation_modes.append(allow_create)
            return start(allow_create=allow_create)
        actual.start_or_resume = observed_start
        return actual
    def consume(actual_bridge, cache, **options):
        assert actual_bridge is bridge and cache == tmp_path
        assert options == {"stopped": stopped, "day": DAY, "api_factory": api_factory}
        assert ledger.get(DAY)["session_id"] == failed["session_id"]
        delegated.append(DAY)
        return {"state": "existing_workflow_called", "date": DAY}
    monkeypatch.setattr(render, "configured", lambda *_args: run.config)
    monkeypatch.setattr(render, "Runner", runner_factory)
    monkeypatch.setattr(render, "consume_workflow", consume)
    result = render.invoke(command, bridge, tmp_path, stopped=stopped, api_factory=api_factory)
    assert delegated == ([DAY] if boundary == "eligible" else [])
    assert result["state"] == ("existing_workflow_called" if boundary == "eligible" else "failed")
    assert creation_modes == [command == "run"] and len(api.payloads) == 1
    assert ledger.read_bytes(DAY + "-artifact.json") == retained


@pytest.fixture
def adjustable_control():
    return json.loads((ROOT/"tools/daily_research/render.control.example.json").read_text())


@pytest.mark.parametrize("minutes",[60,120,180,240])
def test_runtime_minutes_are_owner_adjustable_without_changing_budget(adjustable_control,minutes):
    control=adjustable_control
    control["config"].update(discovery_profile="adaptive-sites-v1",max_runtime_seconds=3600,qa_reserved_seconds=900)
    original=copy.deepcopy(control)
    configured=render.runtime_configuration(control,minutes)
    assert configured["max_runtime_seconds"] ==minutes*60 and configured["qa_reserved_seconds"] ==900
    assert configured["soft_target_usd"] ==control["config"]["soft_target_usd"]
    assert control ==original
    assert render.runtime_configuration(control,minutes,20)["qa_reserved_seconds"] ==1200


@pytest.mark.parametrize("minutes,qa",[(None,None),(True,None),(1,None),(241,None),(120,True),(120,120)])
def test_runtime_minutes_and_qa_refuse_invalid_values(adjustable_control,minutes,qa):
    adjustable_control["config"].update(discovery_profile="adaptive-sites-v1",max_runtime_seconds=3600,qa_reserved_seconds=900)
    with pytest.raises(Refusal):
        render.runtime_configuration(adjustable_control,minutes,qa)


def test_real_private_pipe_adjusts_runtime_without_provider_start(fixture,adjustable_control,monkeypatch):
    _,api,ledger,bridge,_=fixture
    control=adjustable_control
    control["source_commit"]="c"*40
    control["config"].update(discovery_profile="adaptive-sites-v1",max_runtime_seconds=3600,qa_reserved_seconds=900)
    with ledger.lock():
        bridge.call("configure",value=control)
    original=render.read_json
    monkeypatch.setattr(render,"read_json",lambda path: {"source_commit":"c"*40} if Path(path).name=="manifest.json" else original(path))
    result=render.set_runtime(bridge,ledger,240)
    assert result["total_minutes"] ==240 and result["qa_minutes"] ==15 and result["research_minutes"] ==225
    assert result["existing_rows_changed"] is False and result["paid_allowance_changed"] is False
    assert bridge.call("control")["config"]["max_runtime_seconds"] ==14400
    assert api.payloads ==[]


def test_invoke_wires_the_stop_signal_into_the_provider_for_findall(monkeypatch, tmp_path):
    """The FindAll handler reads provider.stopped, so a SIGTERM must reach it before a create POST."""
    seen = {}

    class Provider:
        pass

    class FakeRunner:
        def __init__(self, ledger, cfg, api):
            seen["api"] = api

        def start_or_resume(self, *, allow_create):
            return {"state": "completed"}

    class FakeBridge:
        def call(self, op, **_kwargs):
            return None if op == "active_qa" else {}

    def stopped():
        return True

    monkeypatch.setattr(render, "configured", lambda *_args: {"enabled": False})
    monkeypatch.setattr(render, "Runner", FakeRunner)
    render.invoke("reconcile", FakeBridge(), tmp_path, stopped=stopped, api_factory=lambda *_args: Provider())
    assert seen["api"].stopped is stopped
