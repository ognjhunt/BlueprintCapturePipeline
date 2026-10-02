"""Real private-pipe adapter recovery with hermetic Node Firestore transactions."""
import base64
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.test_daily_research_runner import DAY, NOW, SHEET, FakeAPI
from tools.daily_research import render
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import Refusal, Runner, digest, preflight, save_json

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


@pytest.mark.parametrize(("now", "expected"), [
    ("2026-11-01T12:44:00+00:00", "2026-11-01T12:45:00+00:00"),
    ("2027-03-14T11:44:00+00:00", "2027-03-14T11:45:00+00:00"),
    ("2026-10-02T11:46:00+00:00", "2026-10-02T12:00:00+00:00"),
])
def test_learning_wake_is_six_forty_five_before_seven_with_dst(now, expected):
    from datetime import datetime
    assert render.next_wake(datetime.fromisoformat(now), learning=True).isoformat() == expected


def test_learning_aggregation_runs_once_before_research_and_replays_after_restart(monkeypatch):
    from datetime import datetime
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    control["enabled"] = True
    control["config"]["scheduler_authority_reference"] = "synthetic-reviewed-cutover"
    control["learning"] = {"enabled": True, "startDate": "2026-09-30"}
    calls, daily = [], []
    clock = [datetime.fromisoformat("2026-09-30T11:44:00+00:00")]

    class ClockBridge:
        def call(self, op, **kwargs):
            calls.append(op)
            if op == "control":
                return control
            if op == "learning_daily":
                daily.append(kwargs)
                return {"overviewId": "verified-fake-overview"}
            assert op in {"acquire", "release", "learning_reconcile"}
        def close(self):
            pass

    class Stopped:
        ticks = 0
        def is_set(self):
            return self.ticks >= 18
        def wait(self, delay):
            self.ticks += 1
            clock[0] += timedelta(seconds=delay)

    monkeypatch.setattr(render, "invoke", lambda *args, **kwargs: calls.append("research") or {"state": "not_due"})
    render.scheduler(Stopped(), bridge_factory=ClockBridge, clock=lambda: clock[0])
    assert daily == [{"day": "2026-09-30", "as_of": "2026-09-30T11:45:00Z"}]
    assert calls.count("learning_reconcile") == 1
    assert calls.index("learning_daily") < len(calls) - calls[::-1].index("research") - 1
    render.scheduler(Stopped(), bridge_factory=ClockBridge, clock=lambda: clock[0])
    assert daily[0] == daily[1]  # Same immutable daily request/job identity.
    assert len(daily) == 2


def test_scoped_learning_is_durable_before_create_and_recovery_keeps_exact_bytes(fixture):
    run, api, ledger, bridge, _ = fixture
    scope = {"enabled": True, **{key: {"expiresAt": "2099-01-01T00:00:00.000Z"}
                                for key in ("binding", "businessScope", "learningGrant")}}
    with ledger.lock():
        bridge.call("configure", value={**bridge.call("control"), "learning": scope})
    content = json.dumps({"date": DAY, "paidAnalysisCalls": 0, "sendsAuthorized": False,
                          "evidence": "Original café observation; interest unknown", "confidence": 0.9}, ensure_ascii=False)
    learning = {"version": "blueprint.research-learning-input.v1", "date": DAY,
                "paidAnalysisCalls": 0, "sendsAuthorized": False, "content_json": content,
                "bindingHash": digest(scope),
                "inputHash": hashlib.sha256(content.encode()).hexdigest()}
    contexts = []
    ledger.learning_context = lambda day: contexts.append(day) or learning
    api.lost_create_reply = True
    assert run.start_or_resume()["state"] == "creation_unresolved"
    persisted = ledger.get(DAY)
    assert persisted["learning_context"] == learning
    assert persisted["learning_context_digest"] == learning["inputHash"]
    assert json.dumps(content) in persisted["create_payload"]["input"]
    learning["content_json"] = "changed after claim"
    assert run.start_or_resume(allow_create=False)["state"] == "awaiting_review"
    assert contexts == [DAY] and len(api.payloads) == 1
    assert ledger.get(DAY)["learning_context"]["content_json"] == content


def test_invalid_learning_cannot_claim_a_date_or_call_provider(fixture):
    run, api, ledger, _, _ = fixture
    ledger.learning_context = lambda day: {"version": "blueprint.research-learning-input.v1", "date": day,
                                           "paidAnalysisCalls": 0, "sendsAuthorized": False,
                                           "content_json": "{}", "inputHash": "a" * 64}
    with pytest.raises(Refusal, match="learning_input_invalid"):
        run.start_or_resume()
    assert ledger.get(DAY) is None and api.payloads == []


@pytest.mark.parametrize('enabled', [False, True])
def test_learning_failure_never_blocks_existing_intent_observation(monkeypatch, enabled):
    control = json.loads((ROOT / 'tools/daily_research/render.control.example.json').read_text())
    control['enabled'] = enabled
    control['config']['scheduler_authority_reference'] = 'synthetic-cutover'
    control['learning'] = {'enabled': True, 'startDate': DAY}
    clock, calls = [NOW], []

    class FailingLearning:
        def call(self, op, **kwargs):
            calls.append(op)
            if op == 'control':
                return control
            if op == 'learning_reconcile':
                raise Refusal('research_learning_scope_expired')
            assert op in {'acquire', 'release'}
        def close(self):
            calls.append('close')

    class Stopped:
        ticks = 0
        def is_set(self):
            return self.ticks >= 7
        def wait(self, delay):
            self.ticks += 1
            clock[0] += timedelta(seconds=delay)

    def observe(command, *args, **kwargs):
        calls.append(command)
        return {'state': 'running'}  # Recovery remains due after five minutes.

    monkeypatch.setattr(render, 'invoke', observe)
    monkeypatch.setattr(render, 'emit', lambda result: None)
    render.scheduler(Stopped(), bridge_factory=FailingLearning, clock=lambda: clock[0])
    command = 'run' if enabled else 'reconcile'
    assert calls.count(command) == 2
    assert calls.count('learning_reconcile') == 2  # Separate bounded retry, not every minute.


def test_learning_projection_failure_preserves_terminal_result(monkeypatch):
    class FailedProjection:
        def call(self, op, **kwargs):
            if op == 'learning_terminal':
                raise Refusal('research_learning_scope_expired')
    monkeypatch.setattr(render, 'emit', lambda result: None)
    result = {'date': DAY, 'state': 'failed', 'error': 'preserved-root-failure'}
    assert render.learning_terminal(FailedProjection(), result) is result
