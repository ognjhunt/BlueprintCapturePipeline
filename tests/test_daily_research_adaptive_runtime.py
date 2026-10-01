"""One-time claim/recovery rehearsal with the real private pipe; zero live calls."""
import hashlib
import json
import os
import subprocess
import sys
from copy import copy, deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_daily_research_adaptive import result
from tests.test_daily_research_consumer import fixture as consumer_fixture
from tests.test_daily_research_runner import AGENT, NOW
from tools.daily_research import adaptive, adaptive_runtime
from tools.daily_research.consumer import QA_PATH
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import REMOTE_OUTPUT, Refusal, canonical, digest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    yield from consumer_fixture.__wrapped__(tmp_path)


def setup(fixture, monkeypatch):
    consumer, old_api, daily_ledger, bridge, script = fixture
    clock = {"now": NOW + timedelta(days=1)}
    original = daily_ledger.get("2026-09-30")
    original.update(date="2026-10-01", run_key="blueprint-researcher:2026-10-01", state="failed", qa=None, delivery={})
    original.pop("packet", None)
    original.pop("packet_digest", None)
    with daily_ledger.lock():
        daily_ledger.put({**original,"state":"creating"})
        daily_ledger.put(original)
    monkeypatch.setattr(adaptive, "SESSION", original["session_id"])
    monkeypatch.setattr(adaptive, "DAILY_STATUS_SHA", digest(original))
    monkeypatch.setattr(adaptive, "DAILY_ARTIFACT_SHA", original["raw_output_digest"])
    profile = json.loads(adaptive.PROFILE.read_text())
    profile.update(spend_admission_reference="synthetic-approved-watchdog-and-usage",session_state_receipt="synthetic-exact-GETs")
    # The existing fake credential remains outside the event/intent contract.
    source = "a" * 40
    out, _, _ = result()
    out["checked_date"] = "2026-10-01"
    for candidate in out["candidates"]:
        for evidence in candidate["evidence"]:
            if evidence["origin"] == "live":
                evidence.update(checked_date="2026-10-01",source_checked_at=clock["now"].isoformat())
    raw = canonical(out).encode()
    instructions = "Hermetic reviewed agent"
    monkeypatch.setattr(adaptive_runtime, "INSTRUCTIONS", hashlib.sha256(instructions.encode()).hexdigest())
    api = copy(old_api)
    api.agent, api.sessions = deepcopy(old_api.agent), deepcopy(old_api.sessions)
    api.agent.update(instructions=instructions,service_tier="default")
    api.sessions[0]["environment"]["container_size"]="small"
    posted, turns = [], [{"id":original["turn_id"],"status":"completed","subagent_id":None,
                         "session_id":original["session_id"],"agent_id":AGENT}]
    options={"lost_reply":False,"unknown_usage":False,"research_status":"completed"}
    qa_result={}
    original_get=api.get
    def get(resource, resource_id):
        if resource=="session":
            session=original_get(resource, resource_id)
            session["usage"]={"input_tokens":900000,"output_tokens":1000}
            return session
        return original_get(resource, resource_id)
    def listing(resource, session_id=None):
        if resource=="sessions":
            raise AssertionError("unrelated session listing forbidden")
        if resource=="turns":
            return deepcopy(turns)
        if resource=="items":
            return [{"id":"web"+str(n),"turn_id":"turn_test","type":"web_search_call"} for n in range(17)]
        if resource=="artifacts":
            return [{"id":"artifact_old","turn_id":original["turn_id"],"path":REMOTE_OUTPUT},
                    {"id":"artifact_test","turn_id":"turn_test","path":"/workspace/outputs/adaptive-discovery-20261001.json"},
                    {"id":"artifact_qa","turn_id":"turn_qa","path":QA_PATH}]
        raise AssertionError(resource)
    ledger=adaptive_runtime.TestLedger(bridge)
    def create(sid, events, idempotency_key):
        row=ledger.get("2026-10-01")
        assert row is not None and row["test_intent"]["event"]
        assert sid==original["session_id"] and len(events)==1
        phase="qa" if idempotency_key.endswith(":qa") else "research"
        posted.append((phase,idempotency_key))
        tid="turn_qa" if phase=="qa" else "turn_test"
        turns.append({"id":tid,"session_id":sid,"agent_id":AGENT,"subagent_id":None,
                      "status":"completed" if phase=="qa" else options["research_status"],
                      "completed_at":int(clock["now"].timestamp()),
                      "usage":None if options["unknown_usage"] else {"input_tokens":20000,"output_tokens":1000}})
        if phase=="qa":
            qa_result.update(schema_version="blueprint.research-qa.v1",packet_digest=row["packet_digest"],
                crm_digest=row["qa"]["crm_digest"],source_support_verified=True,
                accepted_keys=[c["candidate_key"] for c in row["packet"]["candidates"]],summary="Synthetic source-supported QA",
                checks=[{"candidate_key":c["candidate_key"],"source_support_verified":True,"duplicate":False,
                         "reason":"Synthetic checked operator, incumbent and geography sources"} for c in row["packet"]["candidates"]])
        if options["lost_reply"]:
            raise TimeoutError()
    api.api=SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=create)))
    api.get, api.listing=get, listing
    api.artifact=lambda sid, aid:canonical(qa_result).encode() if aid=="artifact_qa" else raw
    cancellations=[]
    def cancel(sid,key):
        cancellations.append((sid,key))
        for turn in turns[1:]:
            if turn["status"]=="in_progress":
                turn["status"]="cancelled"
    api.cancel=cancel
    # Full canonical reader is the existing fake service binding, refreshed here
    # to the test date so this rehearsal exercises freshness rather than bypassing.
    crm_path=Path(consumer.config["crm_snapshot"])
    crm=json.loads(crm_path.read_text())
    crm["captured_at"]=clock["now"].isoformat()
    crm_path.write_text(canonical(crm))
    return SimpleNamespace(bridge=bridge,script=script,api=api,profile=profile,source=source,clock=clock,
                           original=deepcopy(original),posted=posted,options=options,cancellations=cancellations,
                           ledger=ledger,turns=turns)


def run(f, cache, **kwargs):
    return adaptive_runtime.invoke(f.bridge,f.profile,f.source,cache,f.api,
        clock=lambda:f.clock["now"],sleep=lambda _:f.clock.update(now=f.clock["now"]+timedelta(seconds=3)),**kwargs)


def test_real_store_collects_ten_then_agent_qa_without_touching_failed_daily(fixture,tmp_path,monkeypatch):
    f=setup(fixture,monkeypatch)
    result=run(f,tmp_path,execute=True)
    assert result["state"]=="test_qa_validated" and result["accepted_new_count"]==10
    assert [phase for phase,_ in f.posted]==["research","qa"]
    assert FirestoreLedger(f.bridge).get("2026-10-01")==f.original
    assert result["publication_performed"] is False and not f.cancellations
    assert f.ledger.get("2026-10-01")["artifact_id"]=="artifact_test"
    assert len(f.api.payloads)==1  # Historical fake daily create only; never a second create.


def test_lost_input_replies_reconcile_and_reinvocation_does_not_post(fixture,tmp_path,monkeypatch):
    f=setup(fixture,monkeypatch)
    f.options["lost_reply"]=True
    assert run(f,tmp_path,execute=True)["state"]=="test_qa_validated"
    assert len(f.posted)==2
    f.bridge.close()
    f.bridge=Bridge(script=f.script)
    try:
        assert run(f,tmp_path,execute=True)["state"]=="test_qa_validated"
        assert len(f.posted)==2
        assert FirestoreLedger(f.bridge).get("2026-10-01")==f.original
    finally:
        f.bridge.close()


def test_missing_usage_cancels_exact_new_turn_without_qa(fixture,tmp_path,monkeypatch):
    f=setup(fixture,monkeypatch)
    f.options.update(unknown_usage=True,research_status="in_progress")
    result=run(f,tmp_path,execute=True)
    assert result["state"]=="cancelled" and len(f.posted)==len(f.cancellations)==1
    assert f.ledger.get("2026-10-01")["model_cost_estimate"]["known"] is False
    assert FirestoreLedger(f.bridge).get("2026-10-01")==f.original


def test_pending_admission_and_reconcile_have_zero_input_and_no_test_record(fixture,tmp_path,monkeypatch):
    f=setup(fixture,monkeypatch)
    assert run(f,tmp_path)["state"]=="no_adaptive_test"
    f.profile["spend_admission_reference"]="PENDING"
    assert run(f,tmp_path,execute=True)["state"]=="admission_blocked"
    assert not f.posted and f.ledger.get("2026-10-01") is None
    with pytest.raises(Refusal,match="create_forbidden"):
        adaptive_runtime.TestProvider(f.api,f.ledger,{"session_id":"sess_1"},lambda:NOW).create({})


@pytest.mark.parametrize("phase",["research","qa"])
@pytest.mark.parametrize("stop_during",["session_get","claim"])
def test_stop_during_final_read_or_claim_prevents_both_paid_inputs(fixture,tmp_path,monkeypatch,phase,stop_during):
    f=setup(fixture,monkeypatch)
    stop={"requested":False}
    with f.ledger.lock():
        row=adaptive_runtime.stage(f.bridge,f.profile,f.source,tmp_path,f.api,lambda:f.clock["now"])
        f.ledger.put(row)
        event=row["test_intent"]["event"]
        deadline=row["research_deadline_ms"]
        if phase=="qa":
            row["state"]="awaiting_review"
            row["qa"]={"request_digest":digest(event),"deadline_ms":deadline}
        f.ledger.put(row)
        scoped=adaptive_runtime.TestProvider(f.api,f.ledger,row["test_intent"],lambda:f.clock["now"],lambda:stop["requested"])
        if stop_during=="session_get":
            get=f.api.get
            def interrupted_get(resource,resource_id):
                value=get(resource,resource_id)
                stop["requested"]=True
                return value
            f.api.get=interrupted_get
        else:
            call=f.bridge.call
            def interrupted_claim(op,**fields):
                value=call(op,**fields)
                if op=="adaptive_claim":
                    stop["requested"]=True
                return value
            monkeypatch.setattr(f.bridge,"call",interrupted_claim)
        with pytest.raises(Refusal,match="before_input"):
            scoped.input(phase,event,"synthetic-once",deadline)
        assert not f.posted
    assert FirestoreLedger(f.bridge).get("2026-10-01")==f.original


@pytest.mark.parametrize("guard",["disabled","expired"])
def test_late_control_or_deadline_change_after_claim_prevents_post(fixture,tmp_path,monkeypatch,guard):
    f=setup(fixture,monkeypatch)
    with f.ledger.lock():
        row=adaptive_runtime.stage(f.bridge,f.profile,f.source,tmp_path,f.api,lambda:f.clock["now"])
        f.ledger.put(row)
        call=f.bridge.call
        claimed={"value":False}
        def changed_after_claim(op,**fields):
            value=call(op,**fields)
            if op=="adaptive_claim":
                claimed["value"]=True
                if guard=="expired":
                    f.clock["now"]+=timedelta(seconds=1201)
            if op=="control" and claimed["value"] and guard=="disabled":
                return {**value,"enabled":False}
            return value
        monkeypatch.setattr(f.bridge,"call",changed_after_claim)
        scoped=adaptive_runtime.TestProvider(f.api,f.ledger,row["test_intent"],lambda:f.clock["now"])
        with pytest.raises(Refusal,match="before_input"):
            scoped.input("research",row["test_intent"]["event"],"synthetic-once",row["research_deadline_ms"])
        assert not f.posted


def test_entrypoint_requires_actual_bounded_timeout_parent_without_provider_access():
    root=Path(__file__).resolve().parents[1]
    probe="from tools.daily_research.adaptive_runtime import verify_process_watchdog;verify_process_watchdog()"
    refused=subprocess.run([sys.executable,"-c",probe],cwd=root,capture_output=True,check=False,env={"PATH":os.environ["PATH"]})
    assert refused.returncode!=0 and b"watchdog_required" in refused.stderr
    admitted=subprocess.run(["timeout","--signal=TERM","--kill-after=60s","1860s",sys.executable,"-c",probe],
                            cwd=root,capture_output=True,check=False,env={"PATH":os.environ["PATH"]})
    assert admitted.returncode==0, admitted.stderr
