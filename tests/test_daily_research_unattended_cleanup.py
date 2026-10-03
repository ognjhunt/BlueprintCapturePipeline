"""Real lease/bridge/archive/DELETE integration with offline transport doubles."""
import base64
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_daily_research_runner import FakeAPI, NotFound, output
from tools.daily_research import render
from tools.daily_research.firestore import Bridge, FencedProvider, FirestoreLedger
from tools.daily_research.runner import AGENT, PROJECT, SHEET, TEMPLATE, Runner, Refusal, canonical, digest, save_json

DAY = "2026-10-03"
NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    script = tmp_path / "bridge.mjs"
    objects = tmp_path / "objects.json"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline';",
        "import {existsSync,readFileSync,writeFileSync} from 'node:fs';",
        "import {Store,LeaseChannel} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "const objectFile=" + json.dumps(str(objects)) + ";let objects=existsSync(objectFile)?JSON.parse(readFileSync(objectFile)):{};",
        "const bucket={name:'blueprint-8c1ca.appspot.com',file(name,opts={}){return {name,async save(raw,options){if(options.preconditionOpts.ifGenerationMatch!==0)throw Error('not create-only');if(objects[name])throw Object.assign(Error('exists'),{code:412});objects[name]={bytes:raw.toString('base64'),generation:'1'};writeFileSync(objectFile,JSON.stringify(objects));},async getMetadata(){const x=objects[name];if(!x||opts.generation&&opts.generation!==x.generation)throw Error('missing generation');return [{generation:x.generation,size:Buffer.from(x.bytes,'base64').length}];},async download(){return [Buffer.from(objects[name].bytes,'base64')];}};}};",
        "const db=new MemoryFirestore(" + json.dumps(str(tmp_path / "firestore.json")) + ");",
        "const store=new Store(db,()=>" + str(int(NOW.timestamp() * 1000)) + ",undefined,null,null,null,null,false,bucket);const channel=new LeaseChannel(store);",
        "for await(const line of createInterface({input:process.stdin})){try{const r=JSON.parse(line);if(r.op==='test_corrupt_archive'){objects[Object.keys(objects)[0]].bytes=Buffer.from('tampered').toString('base64');writeFileSync(objectFile,JSON.stringify(objects));process.stdout.write(JSON.stringify({ok:true,value:true})+'\\n');continue;}const value=await channel.call(r);process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}catch(error){process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}await channel.close();",
    ]))
    bridge = Bridge(script=script)
    bridge.call("init", value={"schema_version": "blueprint.research-control.v1", "enabled": False})
    ledger = FirestoreLedger(bridge)
    api = FakeAPI()
    api.raw = canonical(output(DAY)).encode()
    api.agent["instructions"] = "Synthetic reviewed research instructions"
    crm = tmp_path / "crm.json"
    headers = ["Prospect ID", "Organization", "Prospect type", "Site / team", "Contact name",
               "", "", "", "", "Task evidence URL", "", "", "", "", "Task / job"]
    save_json(crm, {"sheet_id": SHEET, "complete": True,
                    "captured_at": NOW.isoformat(), "values": [["CRM"], [], [], [], headers]})
    config = {"enabled": True, "first_date": DAY, "approval_reference": "synthetic-research-owner",
              "scheduler_authority_reference": "synthetic-recurring", "crm_snapshot": str(crm), "soft_target_usd": 1,
              "expected_agent_instructions_sha256": hashlib.sha256(api.agent["instructions"].encode()).hexdigest()}
    policy = {"enabled": True, "approval_reference": "synthetic-standing-deletion-owner", "first_date": DAY,
              "expires_at": "2027-01-01T00:00:00Z", "project_id": PROJECT, "agent_id": AGENT,
              "template_id": TEMPLATE, "bucket": "blueprint-8c1ca.appspot.com"}
    with ledger.lock():
        bridge.call("configure", value={"schema_version": "blueprint.research-control.v1", "enabled": True,
                   "project_id": PROJECT, "agent_id": AGENT, "template_id": TEMPLATE, "model": "gpt-6.1-sol", "config": config})
    runner = Runner(ledger, config, api, clock=lambda: NOW)
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review"
    runner.review(DAY, {"packet_digest": row["packet_digest"], "reviewer_reference": "synthetic-same-agent",
        "source_support_verified": True, "crm_rechecked": True,
        "accepted_keys": [c["candidate_key"] for c in row["packet"]["candidates"]], "summary": "Retained source checked"})
    with ledger.lock():
        row = ledger.get(DAY)
        for name in ("notion", "sheets"):
            delivery = row["delivery"][name]
            delivery.update(state="acknowledged", receipt={"destination": name, "key": delivery["key"],
                "payload_digest": delivery["payload_digest"], "readback_verified": True, "reference": "synthetic-" + name})
        qa_raw = canonical({"schema_version": "blueprint.research-qa.v1", "packet_digest": row["packet_digest"]}).encode()
        qa_input = {"type": "agent.session.input.message", "input": "Synthetic existing QA turn"}
        pub_input = {"type": "agent.session.input.message", "input": "Synthetic existing publication turn"}
        for name, data in {"qa": qa_raw, "qa-input": (canonical(qa_input) + "\n").encode(), "qa-evidence": b"[]",
                           "publication-input": (canonical(pub_input) + "\n").encode(), "publication-evidence": b"[]"}.items():
            ledger.write_bytes(f"{DAY}-{name}.json", data)
        row["qa"] = {"state": "validated", "turn_id": "turn_qa", "artifact_file": DAY + "-qa.json",
            "artifact_digest": hashlib.sha256(qa_raw).hexdigest(), "evidence_file": DAY + "-qa-evidence.json",
            "evidence_digest": digest([]), "input_file": DAY + "-qa-input.json", "request_digest": digest(qa_input)}
        row["publication"] = {"state": "completed", "turn_id": "turn_pub", "turn_status": "completed",
            "idempotency_key": row["run_key"] + ":publication", "session_id": row["session_id"],
            "input_file": DAY + "-publication-input.json", "request_digest": digest(pub_input),
            "evidence_file": DAY + "-publication-evidence.json", "evidence_digest": digest([])}
        row["state"] = "completed"
        ledger.put(row)
    get, listing, artifact = api.get, api.listing, api.artifact
    state = {"session_absent": False, "environment_absent": False, "delete_calls": [], "mode": "ack", "closed": 0}
    def retrieve(resource, rid):
        if state.get(resource + "_absent"):
            api.calls.append(("GET", resource, rid))
            raise NotFound()
        return get(resource, rid)
    def inventory(resource, sid=None):
        values = listing(resource, sid)
        if resource == "turns":
            values += [{"id": "turn_qa", "status": "completed"}, {"id": "turn_pub", "status": "completed"}]
        if resource == "artifacts":
            values += [{"id": "artifact_qa", "turn_id": "turn_qa", "path": "/workspace/outputs/qa.json"}]
        return values
    api.get, api.listing = retrieve, inventory
    api.artifact = lambda sid, aid: qa_raw if aid == "artifact_qa" else artifact(sid, aid)
    def provider_factory(l, key):
        provider = object.__new__(FencedProvider)
        provider.ledger, provider.get, provider.listing = l, api.get, api.listing
        provider.artifact = api.artifact
        def delete(sid):
            saved = l.get(DAY)
            assert saved["cleanup"]["delete_claimed"] is True
            receipt = saved["cleanup"]["archive"]
            actual = json.loads(objects.read_text())
            assert len(actual) == len(receipt["objects"]) and actual
            assert all(hashlib.sha256(base64.b64decode(actual[o["name"]]["bytes"])).hexdigest() == o["sha256"] for o in receipt["objects"])
            assert any(base64.b64decode(o["bytes"]) == api.raw for name, o in actual.items() if name.endswith(".bin"))
            state["delete_calls"].append(sid)
            if state["mode"] != "lost_unaccepted":
                state["session_absent"] = True
                state["environment_absent"] = state["mode"] != "environment_alive"
            if state["mode"].startswith("lost"):
                raise TimeoutError()
            return SimpleNamespace(model_dump=lambda mode: {"id": sid, "deleted": True})
        provider.api = SimpleNamespace(sessions=SimpleNamespace(delete=delete))
        provider.client = SimpleNamespace(close=lambda: state.update(closed=state["closed"] + 1))
        return provider
    def enable(value=None):
        with ledger.lock():
            control = bridge.call("control")
            control["cleanup_policy"] = value or policy
            bridge.call("configure", value=control)
    yield SimpleNamespace(bridge=bridge, ledger=ledger, api=api, state=state, factory=provider_factory,
                          enable=enable, policy=policy, objects=objects, cache=tmp_path, script=script)
    bridge.close()


def run(s):
    return render.cleanup_completed(s.bridge, s.cache, api_factory=s.factory)


def test_complete_archive_precedes_only_delete_and_preserves_unknown_cost(setup):
    s = setup
    assert run(s)["state"] == "cleanup_disabled"
    assert not s.objects.exists() and s.state["delete_calls"] == []
    s.enable()
    original = s.ledger.get(DAY)
    assert original["create_payload"]["agent_id"] == AGENT
    assert original["create_payload"]["environment"]["environment_template_id"] == TEMPLATE
    result = run(s)
    assert result["state"] == "cleanup_completed" and result["billing_stop_verified"] is False
    row = s.ledger.get(DAY)
    assert row["cleanup_required"] is False and row["usage"] == original["usage"]
    assert row["cost_status"] == original["cost_status"] and s.state["delete_calls"] == [row["session_id"]]
    assert s.state["closed"] == 1
    export = render.export_snapshot(s.bridge, DAY, s.cache / "portable")
    assert export["missing_files"] == []
    manifest = json.loads((s.cache / "portable/cleanup-manifest.json").read_text())
    assert manifest["delete_claimed"] is True and manifest["archive"] == row["cleanup"]["archive"]
    assert run(s)["state"] == "cleanup_not_due" and len(s.state["delete_calls"]) == 1


@pytest.mark.parametrize("change", ["expired", "wrong_agent", "wrong_project", "early_date", "qa_active", "receipt_missing", "running"])
def test_authority_and_active_work_never_archive_or_delete(setup, change):
    s = setup
    policy = deepcopy(s.policy)
    if change == "expired": policy["expires_at"] = NOW.isoformat()
    if change == "wrong_agent": policy["agent_id"] = "foreign-agent"
    if change == "wrong_project": policy["project_id"] = "foreign-project"
    if change == "early_date": policy["first_date"] = "2026-10-02"
    s.enable(policy)
    if change in {"qa_active", "receipt_missing"}:
        with s.ledger.lock():
            row = s.ledger.get(DAY)
            if change == "qa_active": row["qa"]["state"] = "running"
            else: row["delivery"]["sheets"]["receipt"]["readback_verified"] = False
            s.ledger.put(row)
    if change == "running": s.api.session_status = "running"
    with pytest.raises(Refusal):
        run(s)
    assert not s.objects.exists() and s.state["delete_calls"] == []


@pytest.mark.parametrize("mode", ["lost_accepted", "lost_unaccepted", "environment_alive"])
def test_unknown_delete_restart_and_revocation_are_get_only(setup, mode):
    s = setup
    s.enable()
    s.state["mode"] = mode
    assert run(s)["state"] == "cleanup_pending"
    assert s.ledger.get(DAY)["cleanup_required"] is True
    assert len(s.state["delete_calls"]) == 1
    s.bridge.close()
    s.bridge = Bridge(script=s.script)
    s.ledger = FirestoreLedger(s.bridge)
    with s.ledger.lock():
        control = s.bridge.call("control")
        del control["cleanup_policy"]
        s.bridge.call("configure", value=control)
    result = run(s)
    assert result["state"] == ("cleanup_completed" if mode == "lost_accepted" else "cleanup_pending")
    assert len(s.state["delete_calls"]) == 1
    if mode != "lost_accepted":
        assert s.ledger.get(DAY)["cleanup_required"] is True
        s.state.update(session_absent=True, environment_absent=True)
        assert run(s)["state"] == "cleanup_completed"
        assert len(s.state["delete_calls"]) == 1


def test_changed_inventory_after_archive_and_durable_claim_does_not_delete(setup):
    s = setup
    s.enable()
    original = s.bridge.call
    def crossing(op, **kw):
        result = original(op, **kw)
        if op == "cleanup_claim":
            s.api.session_status = "running"
        return result
    s.bridge.call = crossing
    assert run(s)["state"] == "cleanup_pending"
    assert not s.state["delete_calls"]
    s.api.session_status = "idle"
    assert run(s)["state"] == "cleanup_pending"
    assert not s.state["delete_calls"]


@pytest.mark.parametrize("boundary", ["revoked", "stopped", "lost_row_put"])
def test_last_action_fence_and_durable_claim_survive_failure(setup, boundary):
    s = setup
    s.enable()
    original = s.bridge.call
    stopping = [False]
    def race(op, **kw):
        result = original(op, **kw)
        if op == "cleanup_claim":
            if boundary == "revoked":
                control = original("control")
                control["cleanup_policy"]["enabled"] = False
                original("configure", value=control)
            if boundary == "lost_row_put":
                raise Refusal("synthetic_persistence_lost_after_claim")
        if op == "cleanup_delete_check" and boundary == "stopped": stopping[0] = True
        return result
    s.bridge.call = race
    if boundary == "lost_row_put":
        with pytest.raises(Refusal, match="synthetic_persistence_lost_after_claim"):
            render.cleanup_completed(s.bridge, s.cache, api_factory=s.factory)
    else:
        assert render.cleanup_completed(s.bridge, s.cache, api_factory=s.factory, stopped=lambda: stopping[0])["state"] == "cleanup_pending"
    s.bridge.call = original
    assert run(s)["state"] == "cleanup_pending"
    assert s.state["delete_calls"] == [] and s.ledger.get(DAY)["cleanup_required"] is True


def test_archive_readback_must_pass_before_deletion_claim(setup):
    s = setup
    s.enable()
    original = s.bridge.call
    def tamper(op, **kw):
        result = original(op, **kw)
        if op == "cleanup_archive": original("test_corrupt_archive")
        return result
    s.bridge.call = tamper
    with pytest.raises(Refusal, match="cleanup_archive_readback_mismatch"):
        run(s)
    with s.ledger.lock():
        assert s.bridge.call("cleanup_status", day=DAY)["delete_claimed"] is False
    assert s.state["delete_calls"] == []


def test_later_receipts_allow_cleanup_without_rewriting_original_phase_result(setup):
    s = setup
    with s.ledger.lock():
        row = s.ledger.get(DAY)
        row["publication"]["state"] = "agent_finished_without_complete_receipts"
        s.ledger.put(row)
    s.enable()
    assert run(s)["state"] == "cleanup_completed"
    assert s.ledger.get(DAY)["publication"]["state"] == "agent_finished_without_complete_receipts"


def test_saved_session_id_mismatch_never_archives_or_deletes(setup):
    s = setup
    s.enable()
    get = s.api.get
    def mismatch(resource, rid):
        value = get(resource, rid)
        if resource == "session": value["id"] = "foreign_session"
        return value
    s.api.get = mismatch
    with pytest.raises(Refusal, match="session_binding_mismatch"):
        run(s)
    assert s.state["delete_calls"] == [] and not s.objects.exists()


def test_archive_tampering_never_clears_guard_after_unknown_delete(setup):
    s = setup
    s.enable()
    s.state["mode"] = "lost_accepted"
    assert run(s)["state"] == "cleanup_pending"
    s.bridge.call("test_corrupt_archive")
    with pytest.raises(Refusal, match="cleanup_archive_readback_mismatch"):
        run(s)
    assert s.ledger.get(DAY)["cleanup_required"] is True and len(s.state["delete_calls"]) == 1


def test_completed_correction_keeps_original_qa_turn_in_exact_inventory(setup):
    s = setup
    row = s.ledger.get(DAY)
    row["qa"]["turn_id"] = "turn_qa_fixed"
    row["qa"]["corrections"] = [{"turn_id": "turn_qa_fixed", "previous_review": {"turn_id": "turn_qa"}}]
    listing = s.api.listing
    def corrected(resource, sid=None):
        values = listing(resource, sid)
        if resource == "turns": values.append({"id": "turn_qa_fixed", "status": "completed"})
        return values
    s.api.listing = corrected
    assert render.cleanup_inventory(s.api, row, contents=False)["provider-turns.json"]
    s.api.listing = lambda resource, sid=None: corrected(resource, sid) + ([{"id": "foreign_turn", "status": "completed"}] if resource == "turns" else [])
    with pytest.raises(Refusal, match="cleanup_turn_inventory_changed"):
        render.cleanup_inventory(s.api, row, contents=False)
