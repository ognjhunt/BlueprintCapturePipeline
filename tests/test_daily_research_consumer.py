"""Actual durable QA adapter with fake provider; no credentials or inference."""
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_daily_research_knowledge import policy_bundle, v3
from tests.test_daily_research_runner import AGENT, DAY, NOW, SHEET, FakeAPI
from tools.daily_research import render
from tools.daily_research.consumer import QA_PATH, Consumer, qa_decision, qa_text
from tools.daily_research.firestore import Bridge, FencedProvider, FirestoreLedger
from tools.daily_research.runner import Refusal, Runner, canonical, digest, save_json

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ["Prospect ID", "Organization", "Prospect type", "Site / team", "Contact name",
           "Contact details", "Verification", "Contact source URL", "Robot-team fit", "Task evidence URL",
           "Stage", "Owner", "Next action", "Next action date", "Task / job",
           "Robot capability evidence URL", "Evidence maturity", "Geography", "Evidence checked date"]


class QAAPI(FakeAPI):
    def __init__(self, ledger):
        super().__init__()
        self.ledger, self.inputs = ledger, []
        self.qa_status, self.lost_reply, self.qa_exists = "completed", False, False
        self.qa_result, self.fail_observation = None, False
        self.client = SimpleNamespace(close=lambda: None)

    def qa_input(self, sid, event, key, day, request_digest, deadline_ms):
        row = self.ledger.get(day)
        assert row["qa"]["event"] == event and row["qa"]["request_digest"] == request_digest
        self.ledger.bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        self.inputs.append((sid, key))
        self.qa_exists = True
        self.qa_result = {"schema_version": "blueprint.research-qa.v1", "packet_digest": row["packet_digest"],
                          "crm_digest": row["qa"]["crm_digest"], "source_support_verified": True,
                          "accepted_keys": [c["candidate_key"] for c in row["packet"]["candidates"]],
                          "summary": "Supported task claim: https://plant.example/tasks; orderability unknown.",
                          "checks": [{"candidate_key": c["candidate_key"], "source_support_verified": True,
                                      "duplicate": False, "reason": "Operator task evidence checked"}
                                     for c in row["packet"]["candidates"]]}
        if self.lost_reply:
            raise TimeoutError()

    def get(self, resource, resource_id):
        if self.fail_observation and resource == "session" and self.qa_exists:
            raise TimeoutError()
        return super().get(resource, resource_id)

    def listing(self, resource, session_id=None):
        values = super().listing(resource, session_id)
        if not self.qa_exists:
            return values
        if resource == "turns":
            values.append({"id": "turn_qa", "session_id": "sess_1", "agent_id": AGENT,
                           "subagent_id": None, "status": self.qa_status,
                           "completed_at": int((NOW + timedelta(seconds=20)).timestamp())})
        if resource == "artifacts":
            values.append({"id": "artifact_qa", "turn_id": "turn_qa", "path": QA_PATH})
        if resource == "items":
            values.append({"id": "qa_web", "turn_id": "turn_qa", "type": "web_search_call"})
        return values

    def artifact(self, sid, aid):
        return canonical(self.qa_result).encode() if aid == "artifact_qa" else super().artifact(sid, aid)


def consumer_setup(tmp_path, *, failed=False):
    crm = tmp_path / "crm.json"
    save_json(crm, {"sheet_id": SHEET, "complete": True, "captured_at": NOW.isoformat(),
                    "values": [["CRM"], [], [], [], HEADERS]})
    script = tmp_path / "bridge.mjs"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline'; import {readFileSync} from 'node:fs';",
        "import {Store,LeaseChannel} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {Publisher} from " + json.dumps((ROOT / "tools/daily_research/publisher.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "const db=new MemoryFirestore(" + json.dumps(str(tmp_path / "db.json")) + ");",
        "const crmReader=async()=>JSON.parse(readFileSync(" + json.dumps(str(crm)) + ",'utf8'));",
        "const pages=[]; const google=async(method,path,body)=>{if(method==='GET')return {sheets:[]}; const crm=await crmReader();crm.values.push(...body.values);await import('node:fs').then(fs=>fs.writeFileSync(" + json.dumps(str(crm)) + ",JSON.stringify(crm)));return {};};",
        "const notion=async(method,path,body)=>{if(method==='POST'){pages.push(body);return {id:'page-result'};}if(path==='/pages/3eb80154161d8116858ed5f376b4b7a9')return {object:'page',id:'3eb80154161d8116858ed5f376b4b7a9'};if(path.startsWith('/blocks/3eb80154161d8116858ed5f376b4b7a9/'))return {has_more:false,results:pages.map(p=>({id:'page-result',type:'child_page',child_page:{title:p.properties.title.title[0].text.content}}))};if(path==='/pages/page-result')return {parent:{page_id:'3eb80154161d8116858ed5f376b4b7a9'}};return {has_more:false,results:pages[0].children};};",
        "const publisher=new Publisher({crmReader,google,notion});",
        "let testNow=" + str(int(NOW.timestamp()*1000)) + ";const channel=new LeaseChannel(new Store(db,()=>testNow,undefined,crmReader,publisher));",
        "for await (const line of createInterface({input:process.stdin})) {try {const r=JSON.parse(line);if(r.op==='test_clock'){testNow=r.now;process.stdout.write(JSON.stringify({ok:true,value:true})+'\\n');continue;}const value=await channel.call(r);process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error){process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}} await channel.close();",
    ]))
    bridge = Bridge(script=script)
    bridge.call("init", value={"enabled": False, "schema_version": "blueprint.research-control.v1"})
    with FirestoreLedger(bridge).lock():
        bridge.call("configure", value={"enabled": True, "schema_version": "blueprint.research-control.v1",
                                        "workflow": {"enabled": True, "qa_authority_reference": "approved-same-session-QA",
                                                     "publication_authority_reference": "approved-fixed-targets"}})
    ledger = FirestoreLedger(bridge)
    api = QAAPI(ledger)
    cfg = {"enabled": True, "first_date": DAY, "approval_reference": "owner-soft-total-one",
           "scheduler_authority_reference": "one-standalone-trigger", "crm_snapshot": str(crm), "soft_target_usd": 1}
    _knowledge, raw, policy, context = policy_bundle()
    (tmp_path / "knowledge.json").write_bytes(raw)
    save_json(tmp_path / "policy.json", policy)
    cfg.update(research_contract_version=3, knowledge_snapshot=str(tmp_path / "knowledge.json"),
               knowledge_refresh_policy=str(tmp_path / "policy.json"))
    output = v3(context)
    if failed:
        from tests.test_daily_research_knowledge import delta
        proposal = delta()
        proposal["evidence"][0].update(classification="operator", evidence_level=None)
        output["proposed_knowledge_deltas"] = [proposal]
    api.raw = canonical(output).encode()
    assert Runner(ledger, cfg, api, clock=lambda: NOW).start_or_resume()["state"] == ("failed" if failed else "awaiting_review")
    consumer = Consumer(ledger, cfg, api, clock=lambda: NOW + timedelta(seconds=30))
    yield consumer, api, ledger, bridge, script
    bridge.close()


@pytest.fixture
def fixture(tmp_path):
    yield from consumer_setup(tmp_path)


def test_automatic_qa_exact_artifact_review_and_private_export(fixture, tmp_path):
    consumer, api, ledger, bridge, _ = fixture
    assert consumer.step()["state"] == "reviewed"
    row = ledger.get(DAY)
    assert row["qa"]["state"] == "validated" and len(api.payloads) == len(api.inputs) == 1
    assert api.inputs == [("sess_1", "blueprint-researcher:" + DAY + ":qa")]
    assert ledger.read_bytes(DAY + "-qa.json") == canonical(api.qa_result).encode()
    assert row["review"]["reviewer_reference"] == "agent-turn:sess_1:turn_qa"
    assert row["delivery"]["sheets"]["payload_digest"] == digest(row["delivery"]["sheets"]["payload"])
    assert render.export_snapshot(bridge, DAY, tmp_path / "export")["missing_files"] == []


def test_adaptive_qa_uses_reserved_total_time_and_checks_coverage_without_quota_or_legacy_activity_cap(fixture):
    consumer, api, ledger, _, _ = fixture
    with ledger.lock():
        row = ledger.get(DAY)
        row.update(discovery_profile="adaptive-sites-v1", total_runtime_seconds=1800,
                   research_runtime_seconds=1200, web_tool_activities=17)
        ledger.put(row)
    original_listing = api.listing
    def listing(resource, session_id=None):
        values = original_listing(resource, session_id)
        if resource == "turns":
            for value in values:
                if value["id"] == "turn_qa":
                    value["completed_at"] = int((NOW + timedelta(seconds=320)).timestamp())
        return values
    api.listing = listing
    consumer.clock = lambda: NOW + timedelta(seconds=300)
    assert consumer.step()["state"] == "reviewed"
    saved = ledger.get(DAY)
    assert saved["qa"]["state"] == "validated" and len(api.inputs) == 1
    text = saved["qa"]["event"]["input"][0]["content"][0]["text"]
    assert "target of 10 new" not in text and "Existing deployments" in text
    assert "count never establishes completion" in text and "Check contact relevance" in text
    assert "At most" not in text and "why work stopped" in text


def test_long_qa_reasoning_is_retained_and_published_without_another_paid_turn(fixture, tmp_path):
    consumer, api, ledger, bridge, _ = fixture
    api.lost_reply = True
    assert consumer.step()["state"] == "qa_input_unresolved"
    summary = ("Evidence https://plant.example/tasks; unknown interest remains explicit. " * 120)[:7301]
    reason = ("Exact employer/task source verified; buying intent remains unknown. " * 20)[:1103]
    assert len(summary) == 7301 and len(reason) == 1103
    api.qa_result["summary"] = summary
    api.qa_result["checks"][0]["reason"] = reason
    assert consumer.step()["state"] == "reviewed"
    row = ledger.get(DAY)
    assert row["review"]["summary"] == row["delivery"]["notion"]["payload"]["summary"] == summary
    assert json.loads(ledger.read_bytes(DAY + "-qa.json"))["checks"][0]["reason"] == reason
    while row["state"] != "completed":
        consumer.step()
        row = ledger.get(DAY)
    assert all(d["receipt"]["readback_verified"] for d in row["delivery"].values())
    assert all(c["qualification_status"] == "unqualified" for c in row["delivery"]["sheets"]["payload"]["candidates"])
    assert len(api.inputs) == len(api.payloads) == 1 and api.cancellations == []
    assert render.export_snapshot(bridge, DAY, tmp_path / "export")["missing_files"] == []


def test_fenced_qa_is_collected_losslessly_without_repeating_completed_review(fixture):
    consumer, api, ledger, _, _ = fixture
    api.lost_reply = True
    assert consumer.step()["state"] == "qa_input_unresolved"
    artifact = api.artifact
    raw = ("```json\n" + canonical(api.qa_result) + "\n```").encode()
    api.artifact = lambda sid, aid: raw if aid == "artifact_qa" else artifact(sid, aid)
    assert consumer.step()["state"] == "reviewed"
    row = ledger.get(DAY)
    assert ledger.read_bytes(DAY + "-qa.json") == raw
    assert row["qa"]["artifact_digest"] == row["qa"]["artifact_format_normalization"]["raw_sha256"]
    assert row["review"]["summary"] == api.qa_result["summary"]
    assert len(api.inputs) == len(api.payloads) == 1 and api.cancellations == []


def test_run_entrypoint_automatically_finishes_qa_and_both_publication_receipts(fixture, monkeypatch, tmp_path):
    consumer, api, ledger, bridge, _ = fixture
    monkeypatch.setattr(render, "configured", lambda *args: consumer.config)
    monkeypatch.setattr(render, "Runner", lambda *args: Runner(*args, clock=lambda: NOW))
    monkeypatch.setattr(render, "Consumer", lambda *args, **kwargs: Consumer(*args, **kwargs, clock=consumer.clock))
    monkeypatch.setattr(render.time, "sleep", lambda *args: None)
    result = render.invoke("run", bridge, tmp_path, api_factory=lambda *args: api)
    assert result["state"] == "completed"
    row = ledger.get(DAY)
    assert row["qa"]["state"] == "validated"
    assert all(d["receipt"]["readback_verified"] is True for d in row["delivery"].values())
    assert row["cleanup_required"] is True and api.cancellations == []
    assert len(api.payloads) == len(api.inputs) == 1


def test_uncertain_input_restart_observes_without_second_event(fixture):
    consumer, api, _ledger, bridge, script = fixture
    api.lost_reply = True
    assert consumer.step()["state"] == "qa_input_unresolved"
    bridge.close()
    restarted = Bridge(script=script)
    try:
        durable = FirestoreLedger(restarted)
        api.ledger = durable
        result = Consumer(durable, consumer.config, api, clock=consumer.clock).step()
        assert result["state"] == "reviewed" and len(api.inputs) == len(api.payloads) == 1
    finally:
        restarted.close()


def test_slow_preflight_cannot_start_qa_after_total_deadline(fixture):
    consumer, api, ledger, _, _ = fixture
    now = [NOW + timedelta(seconds=20)]
    consumer.clock = lambda: now[0]
    original = api.get
    def delayed(resource, rid):
        value = original(resource, rid)
        now[0] = NOW + timedelta(seconds=200)
        return value
    api.get = delayed
    assert consumer.step()["state"] == "qa_blocked"
    assert api.inputs == [] and ledger.get(DAY)["qa"]["cancel_attempted"] is False


def test_cold_disabled_recovery_cancels_existing_qa_without_input(fixture, monkeypatch, tmp_path):
    consumer, api, ledger, bridge, script = fixture
    api.lost_reply = True
    api.qa_status = "in_progress"
    assert consumer.step()["state"] == "qa_input_unresolved"
    with ledger.lock():
        control = bridge.call("control")
        control["enabled"] = False
        bridge.call("configure", value=control)
    bridge.close()
    restarted = Bridge(script=script)
    try:
        api.ledger = FirestoreLedger(restarted)
        monkeypatch.setattr(render, "configured", lambda *args: {**consumer.config, "enabled": False})
        original_cancel = api.cancel
        def cancel(sid, key):
            original_cancel(sid, key)
            api.qa_status = "cancelled"
        api.cancel = cancel
        result = render.invoke("reconcile", restarted, tmp_path, api_factory=lambda *args: api)
        assert result["state"] == "qa_blocked"
        assert api.cancellations == [("sess_1", "blueprint-researcher:" + DAY + ":qa")]
        assert len(api.inputs) == 1 and api.ledger.get(DAY)["state"] == "awaiting_review"
    finally:
        restarted.close()


def test_late_recovery_collects_in_time_completion_without_cancelling(fixture):
    consumer, api, _, _, _ = fixture
    api.lost_reply = True
    assert consumer.step()["state"] == "qa_input_unresolved"
    consumer.clock = lambda: NOW + timedelta(seconds=190)
    assert consumer.step()["state"] == "reviewed"
    assert api.cancellations == []


def test_observation_failure_persists_one_cancel_and_never_claims_success(fixture):
    consumer, api, ledger, _, _ = fixture
    api.fail_observation = True
    assert consumer.step()["state"] == "qa_cancel_pending"
    assert consumer.step()["state"] == "qa_cancel_pending"
    assert len(api.cancellations) == 1 and ledger.get(DAY)["qa"]["cancel_attempted"] is True
    assert ledger.get(DAY)["state"] == "awaiting_review"


def test_deadline_cancel_preserves_history_and_collects_an_earlier_terminal_result(fixture):
    consumer, api, ledger, _, _ = fixture
    api.lost_reply, api.qa_status = True, "in_progress"
    assert consumer.step()["state"] == "qa_input_unresolved"
    consumer.clock = lambda: NOW + timedelta(seconds=190)
    assert consumer.step()["state"] == "qa_cancel_pending"
    cancelled = deepcopy(ledger.get(DAY)["qa"])
    assert cancelled["cancel_record"]["reason"] == "agent_qa_deadline"
    assert cancelled["cancel_record"]["requested_at"] == consumer.clock().isoformat()
    api.qa_status = "completed"
    assert consumer.step()["state"] == "reviewed"
    qa = ledger.get(DAY)["qa"]
    for key in ("cancel_attempted", "cancel_record", "cancel_idempotency_key", "cancel_reply_received"):
        assert qa[key] == cancelled[key]
    assert qa["terminal_collection_receipt"]["cancel_record_digest"] == digest(cancelled["cancel_record"])
    assert len(api.inputs) == len(api.cancellations) == 1


@pytest.mark.parametrize("change", ["early", "stopped", "disabled", "unknown", "late_completion", "same_second",
                                    "disabled_race", "stopped_race"])
def test_terminal_collection_never_waives_early_unknown_or_nondeadline_cancellation(fixture, change):
    consumer, api, ledger, _, _ = fixture
    api.lost_reply, api.qa_status = True, "in_progress"
    assert consumer.step()["state"] == "qa_input_unresolved"
    consumer.clock = lambda: NOW + timedelta(seconds=190 if change != "early" else 30)
    with ledger.lock():
        row = ledger.get(DAY)
        if change == "disabled_race":
            control = ledger.bridge.call("control")
            control["enabled"] = False
            ledger.bridge.call("configure", value=control)
        if change == "stopped_race":
            consumer.stopped = lambda: True
        consumer.cancel(row, "agent_qa_" + change if change in {"stopped", "disabled"} else "agent_qa_deadline")
        if change in {"disabled_race", "stopped_race"}:
            assert row["qa"]["cancel_record"]["reason"] == "agent_qa_" + change.split("_")[0]
        if change == "unknown":
            row["qa"].pop("cancel_record")
        if change == "same_second":
            row["qa"]["cancel_record"]["requested_at"] = (NOW + timedelta(seconds=180.5)).isoformat()
        ledger.put(row)
    original_listing = api.listing
    def listing(resource, session_id=None):
        values = original_listing(resource, session_id)
        if resource == "turns" and change in {"late_completion", "same_second"}:
            for turn in values:
                if turn["id"] == "turn_qa":
                    turn["completed_at"] = int((NOW + timedelta(seconds=181 if change == "late_completion" else 180)).timestamp())
        return values
    api.listing, api.qa_status = listing, "completed"
    assert consumer.step()["state"] == "qa_blocked"
    assert ledger.get(DAY)["state"] == "awaiting_review"
    assert len(api.inputs) == len(api.cancellations) == 1


def test_fresh_crm_dedupe_and_source_attestation_required(fixture):
    consumer, api, ledger, _, _ = fixture
    api.lost_reply = True
    consumer.step()
    row = ledger.get(DAY)
    row["qa"]["turn_id"] = "turn_qa"
    row["qa"]["artifact_digest"] = "a" * 64
    candidate = row["packet"]["candidates"][0]
    assert qa_decision(row, api.qa_result, set(candidate["identity_keys"]))["accepted_keys"] == []
    rejected = deepcopy(api.qa_result)
    rejected["checks"][0]["source_support_verified"] = False
    with pytest.raises(Refusal, match="candidate_checks_invalid"):
        qa_decision(row, rejected, set())
    row["qa"]["state"] = "qa_running"
    with ledger.lock():
        ledger.put(row)
    with pytest.raises(Refusal, match="cleanup_not_terminal"):
        Runner(ledger, consumer.config, api).record_cleanup(DAY, {})


def test_qa_extra_metadata_is_retained_without_weakening_dispositions(fixture):
    consumer, api, ledger, _, _ = fixture
    api.lost_reply = True
    consumer.step()
    row = ledger.get(DAY)
    row["qa"]["turn_id"] = "turn_qa"
    row["qa"]["artifact_digest"] = "a" * 64
    result = deepcopy(api.qa_result)
    result["source_notes"] = {"dates_unknown": True}
    result["checks"][0]["source_notes"] = {"employer_affiliation": "primary public page"}
    preserved = deepcopy(result)
    assert qa_decision(row, result, set())["accepted_keys"] == result["accepted_keys"]
    assert result == preserved
    result["checks"][0]["duplicate"] = 0
    with pytest.raises(Refusal, match="candidate_checks_invalid"):
        qa_decision(row, result, set())


def test_qa_context_excludes_crm_contacts(fixture):
    _, _, ledger, _, _ = fixture
    row = ledger.get(DAY)
    snapshot = {"values": [[], [], [], [], HEADERS, ["BP-000001", "Existing", "Facility", "Site",
                "PRIVATE NAME", "PRIVATE EMAIL", "", "", "", "https://example.com/task", "", "", "", "", "Task"]]}
    text = qa_text(row, snapshot, "b" * 64)
    assert "PRIVATE NAME" not in text and "PRIVATE EMAIL" not in text
    assert "Existing" in text and "UNTRUSTED DATA" in text


def _sdk_wire_probe():
    import httpx2 as httpx
    from openai import OpenAI
    requests, checks = [], []
    def send(request):
        requests.append(request)
        return httpx.Response(202, json={})
    client = OpenAI(api_key="offline-fake-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(send)))
    provider = FencedProvider.__new__(FencedProvider)
    provider.api = client.beta.agents
    provider.ledger = SimpleNamespace(get=lambda _day: {},
        bridge=SimpleNamespace(call=lambda *args, **kwargs: checks.append((args, kwargs))))
    event = {"type": "agent.session.input.message", "input": [{"role": "user", "content": [{"type": "input_text", "text": "QA"}]}]}
    future = int((datetime.now(timezone.utc) + timedelta(seconds=30)).timestamp() * 1000)
    try:
        provider.qa_input("sess_offline", event, "one-qa-key", DAY, digest(event), future)
        assert len(requests) == 1 and requests[0].url.path == "/v1/agents/sessions/sess_offline/events"
        assert requests[0].headers["Idempotency-Key"] == "one-qa-key"
        assert json.loads(requests[0].content)["events"] == [event]
        assert checks[0][0] == ("qa_check",)
        with pytest.raises(Refusal, match="runtime_exhausted"):
            provider.qa_input("sess_offline", event, "one-qa-key", DAY, digest(event), 1)
        assert len(requests) == 1
    finally:
        client.close()


def test_actual_sdk_event_wire_and_deadline_gate():
    runtime = os.environ.get("BLUEPRINT_RESEARCH_SDK_PYTHON", sys.executable)
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([runtime, "-c", "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                             str(Path(__file__).resolve())],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=True)
    assert result.stdout.strip() == "qa_sdk_wire_contract_verified"


if __name__ == "__main__":
    _sdk_wire_probe()
    print("qa_sdk_wire_contract_verified")
