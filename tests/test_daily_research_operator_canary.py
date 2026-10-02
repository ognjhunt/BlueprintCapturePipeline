"""Hermetic canary claims, namespace/restart, terminal QA and publication."""
import base64
import hashlib
import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_daily_research_consumer import HEADERS
from tests.test_daily_research_knowledge import policy_bundle, v3
from tests.test_daily_research_runner import AGENT, SHEET
from tests.test_daily_research_search import SearchAPI
from tools.daily_research import render
from tools.daily_research.consumer import QA_PATH
from tools.daily_research.firestore import FirestoreLedger
from tools.daily_research.runner import Refusal, Runner, canonical, digest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "tools/daily_research/operators"
spec = importlib.util.spec_from_file_location("canary_operator", HELPERS / "research-perplexity-canary.py")
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)
NOW = datetime(2026, 10, 1, 22, tzinfo=timezone.utc)
APPROVAL = {"schema_version": "blueprint.perplexity-canary-admission.v1", "test_id": canary.TEST,
            "authority_reference": "synthetic-one-time-25-approval", "ceiling_usd": 25, "scope": canary.SCOPE}


class API(SearchAPI):
    def __init__(self, ledger, context):
        super().__init__()
        self.ledger, self.inputs, self.qa_exists = ledger, [], False
        self.turn_status = "completed"
        self.completed_at = int(NOW.timestamp()) + 10
        self.client = SimpleNamespace(close=lambda: None)
        out = v3(context)
        out["checked_date"] = canary.DAY
        for candidate in out["candidates"]:
            for evidence in candidate["evidence"]:
                if evidence["origin"] == "live":
                    evidence.update(checked_date=canary.DAY, source_checked_at=canary.DAY)
        out["coverage"] = {"search_queries": 3, "pages_opened": 2, "branches_checked": ["Synthetic exact task"],
                           "rejection_reasons": [], "stop_reason": "Synthetic scope covered", "shortfall_reason": None,
                           "defined_run_scope": ["Synthetic exact operator/site/task"],
                           "unresolved_promising_branches": [], "completion_state": "coverage_complete"}
        self.raw = canonical(out).encode()

    def create(self, payload):
        row = self.ledger.get(canary.DAY)
        assert row["state"] == "creating" and row["create_payload"] == payload
        assert row["metadata"] == payload["metadata"]
        assert row["canary"]["admission"]["ceiling_usd"] == 25
        self.ledger.bridge.call("create_check", day=canary.DAY, metadata=payload["metadata"])
        return super().create(payload)

    def listing(self, resource, session_id=None):
        result = super().listing(resource, session_id)
        if resource == "turns":
            for turn in result:
                turn.update(session_id="sess_1", agent_id=AGENT, usage={"input_tokens": 100, "output_tokens": 50})
            if self.qa_exists:
                result.append({"id": "turn_qa", "session_id": "sess_1", "agent_id": AGENT, "subagent_id": None,
                               "status": "completed", "completed_at": int(NOW.timestamp()) + 20,
                               "usage": {"input_tokens": 150, "output_tokens": 70}})
        if resource == "artifacts" and self.qa_exists:
            result.append({"id": "artifact_qa", "turn_id": "turn_qa", "path": QA_PATH})
        return result

    def qa_input(self, sid, event, key, day, request_digest, deadline_ms):
        row = self.ledger.get(day)
        assert json.loads(self.ledger.read_bytes(row["qa"]["input_file"])) == event
        self.ledger.bridge.call("qa_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        self.inputs.append((sid, key))
        self.qa_exists = True
        self.qa_result = {"schema_version": "blueprint.research-qa.v1", "packet_digest": row["packet_digest"],
                          "crm_digest": row["qa"]["crm_digest"], "source_support_verified": True,
                          "accepted_keys": [c["candidate_key"] for c in row["packet"]["candidates"]],
                          "summary": "One-time synthetic canary: supported task https://plant.example/tasks; interest unknown.",
                          "checks": [{"candidate_key": c["candidate_key"], "source_support_verified": True,
                                      "duplicate": False, "reason": "Synthetic exact operator task evidence"}
                                     for c in row["packet"]["candidates"]]}

    def artifact(self, sid, aid):
        return canonical(self.qa_result).encode() if aid == "artifact_qa" else super().artifact(sid, aid)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    # The historic one-time admission keeps its real expiry in production;
    # staging tests use their synthetic date even when CI runs after that day.
    monkeypatch.setattr(canary, "datetime", FixtureClock)
    _knowledge, raw, policy, context = policy_bundle(now=NOW)
    original_raw = b'{"fixture":"synthetic original failed run","candidates":[]}\n'
    monkeypatch.setattr(canary.migration, "RAW", hashlib.sha256(original_raw).hexdigest())
    instructions = SearchAPI().agent["instructions"]
    monkeypatch.setattr(canary.migration, "INSTRUCTIONS", hashlib.sha256(instructions.encode()).hexdigest())
    crm = tmp_path / "canonical-crm.json"
    crm.write_text(canonical({"sheet_id": SHEET, "complete": True, "captured_at": NOW.isoformat(),
                              "values": [["Synthetic CRM"], [], [], [], HEADERS]}))
    control = json.loads((ROOT / "tools/daily_research/render.control.example.json").read_text())
    control.update(source_commit=canary.migration.SOURCE, legacy_attempts_reconciled_reference="synthetic-legacy-receipt")
    control["config"].update(search_provider="perplexity-fast-v1", discovery_profile="adaptive-sites-v1",
        soft_target_usd=5, recurring_budget_authority_reference=canary.migration.BUDGET_AUTHORITY,
        max_runtime_seconds=1800, qa_reserved_seconds=600, scheduler_authority_reference="synthetic-old-cutover",
        expected_agent_instructions_sha256=canary.migration.INSTRUCTIONS)
    control["workflow"].update(enabled=True, qa_authority_reference="synthetic-qa-authority",
                               publication_authority_reference="synthetic-publication-authority")
    origin = {"date": canary.DAY, "run_key": "blueprint-researcher:" + canary.DAY,
              "metadata": {"purpose": "synthetic_original_failed"}, "state": "failed",
              "error": "knowledge_schema_invalid", "session_id": canary.migration.SESSION,
              "environment_id": canary.migration.ENVIRONMENT, "raw_output_digest": canary.migration.RAW,
              "cleanup_required": False, "cleanup_receipt": {"action_time_approval_reference": "synthetic-exact-cleanup"}, "delivery": {}}
    generated = canary.driver(ROOT, tmp_path)
    database = tmp_path / "db.json"
    pages_file = tmp_path / "pages.json"
    driver = tmp_path / "hermetic-driver.mjs"
    driver.write_text("\n".join([
        "import {createInterface} from 'node:readline'; import {readFileSync,writeFileSync,existsSync} from 'node:fs';",
        "import {Store} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {Publisher} from " + json.dumps((ROOT / "tools/daily_research/publisher.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "import {CanaryChannel} from " + json.dumps(generated.as_uri()) + ";",
        "const db=new MemoryFirestore(" + json.dumps(str(database)) + "); const normal=new Store(db);",
        "const disabled=" + canonical(control) + ";",
        "if(!db.values.size) {await normal.dispatch({op:'init',value:disabled});await normal.acquire();",
        "await normal.dispatch({op:'configure',value:{...disabled,enabled:true}});",
        "await normal.put(" + canonical({**origin, "state": "creating"}) + ");await normal.put(" + canonical(origin) + ");",
        "await normal.filePut('2026-10-01-artifact.json'," + json.dumps(base64.b64encode(original_raw).decode()) + ");",
        "await normal.filePut('knowledge.json'," + json.dumps(base64.b64encode(raw).decode()) + ");",
        "await normal.filePut('refresh-policy.json'," + json.dumps(base64.b64encode((canonical(policy)+'\n').encode()).decode()) + ");",
        "await normal.dispatch({op:'configure',value:disabled});await normal.release();}",
        "const crmPath=" + json.dumps(str(crm)) + ", pagesPath=" + json.dumps(str(pages_file)) + ";",
        "const crmReader=async()=>JSON.parse(readFileSync(crmPath,'utf8'));",
        "const google=async(method,path,body)=>{if(method==='GET')return {sheets:[]};const c=await crmReader();c.values.push(...body.values);writeFileSync(crmPath,JSON.stringify(c));return {};};",
        "const notion=async(method,path,body)=>{const p=existsSync(pagesPath)?JSON.parse(readFileSync(pagesPath,'utf8')):[];if(method==='POST'){p.push(body);writeFileSync(pagesPath,JSON.stringify(p));return {id:'synthetic-page-'+p.length};}",
        "const number=Number(path.match(/synthetic-page-(\\d+)/)?.[1]);if(method==='PATCH'){p[number-1].children.push(...body.children);writeFileSync(pagesPath,JSON.stringify(p));return {};}",
        "if(path==='/pages/3eb80154161d8116858ed5f376b4b7a9')return {object:'page',id:'3eb80154161d8116858ed5f376b4b7a9'};if(path.startsWith('/blocks/3eb80154161d8116858ed5f376b4b7a9/'))return {has_more:false,results:p.map((x,i)=>({id:'synthetic-page-'+(i+1),type:'child_page',child_page:{title:x.properties.title.title[0].text.content}}))};if(path.startsWith('/pages/synthetic-page-'))return {parent:{page_id:'3eb80154161d8116858ed5f376b4b7a9'}};const start=Number(new URL('https://fixture.invalid'+path).searchParams.get('start_cursor')||0),all=p[number-1]?.children||[],results=all.slice(start,start+100).map((b,i)=>({id:'block-'+number+'-'+(start+i),...b})),next=start+results.length;return {has_more:next<all.length,next_cursor:String(next),results};};",
        "let testNow=" + str(int(NOW.timestamp()*1000)) + ";const channel=new CanaryChannel(db,crmReader,new Publisher({crmReader,google,notion}),()=>testNow);",
        "for await(const line of createInterface({input:process.stdin})){try{const r=JSON.parse(line);",
        "if(r.op==='test_clock'){testNow=r.now;process.stdout.write(JSON.stringify({ok:true,value:true})+'\\n');continue;}",
        "if(r.op==='test_origin_change'){await normal.acquire();await normal.put({...await normal.get('2026-10-01'),cleanup_receipt:{action_time_approval_reference:'changed-synthetic'}});await normal.release();process.stdout.write(JSON.stringify({ok:true,value:true})+'\\n');continue;}",
        "const value=await channel.call(r);process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}catch(error){process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}await channel.close();",
    ]))
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text(canonical({"source_commit": canary.migration.SOURCE}))
    monkeypatch.setattr(render, "__file__", str(release / "tools/daily_research/render.py"))
    bridge = canary.CanaryBridge(script=driver)
    ledger = FirestoreLedger(bridge)
    api = API(ledger, context)
    receipt = {"source_commit": canary.migration.SOURCE, "files_verified": 41, "fixture": "synthetic-package-receipt"}
    plan = canary.inspect(bridge, APPROVAL, receipt, api, tmp_path, now=NOW)
    yield bridge, ledger, api, receipt, plan, tmp_path, driver, crm, pages_file
    bridge.close()


def test_inspection_is_read_only_and_separates_authorities(fixture):
    bridge, ledger, api, _, plan, _, _, _, _ = fixture
    assert ledger.rows() == [] and bridge.call("control") is None
    assert not api.payloads and not api.inputs
    assert plan["firestore_writes"] == plan["provider_mutations"] == 0
    candidate = plan["candidate"]
    assert candidate["config"]["soft_target_usd"] == 5
    assert candidate["config"]["recurring_budget_authority_reference"] == canary.migration.BUDGET_AUTHORITY
    assert candidate["canary"]["admission"]["ceiling_usd"] == 25
    assert candidate["enabled"] is candidate["config"]["enabled"] is candidate["workflow"]["enabled"] is False


@pytest.mark.parametrize("captured_offset,accepted", [(2, True), (4, False), (-26*3600, False)])
def test_crm_read_validates_at_completion_without_rewriting_timestamp(fixture, monkeypatch, captured_offset, accepted):
    bridge, ledger, api, receipt, _, cache, _, crm, _ = fixture
    snapshot = json.loads(crm.read_text())
    snapshot["captured_at"] = (NOW+timedelta(seconds=captured_offset)).isoformat()
    calls = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append("clock")
            return NOW if len(calls) == 1 else NOW+timedelta(seconds=3)

    original_call = bridge.call

    def read(op, **fields):
        if op == "read_crm":
            assert calls == ["clock"]  # Only the earlier admission time exists.
            return snapshot
        return original_call(op, **fields)

    monkeypatch.setattr(canary, "datetime", Clock)
    monkeypatch.setattr(bridge, "call", read)
    if accepted:
        plan = canary.inspect(bridge, APPROVAL, receipt, api, cache)
        retained = json.loads(base64.b64decode(plan["inputs"]["crm.json"]))
        assert retained == snapshot and plan["crm_values_digest"] == canary.digest(snapshot["values"])
    else:
        with pytest.raises(Refusal, match="crm_snapshot_missing_incomplete_or_stale"):
            canary.inspect(bridge, APPROVAL, receipt, api, cache)
    assert calls == ["clock", "clock"]
    assert ledger.rows() == [] and bridge.call("control") is None
    assert not api.payloads and not api.inputs


def test_original_row_drift_refuses_staging(fixture):
    bridge, ledger, _, receipt, plan, _, _, _, _ = fixture
    bridge.call("test_origin_change")
    with pytest.raises(Refusal, match="daily_guard_unreconciled_or_changed"):
        canary.stage(bridge, plan, receipt)
    assert ledger.rows() == []


def test_other_dates_and_unrelated_origin_files_are_refused(fixture):
    bridge, ledger, _, receipt, plan, _, _, _, _ = fixture
    with pytest.raises(Refusal, match="date_scope_invalid"):
        ledger.get("2026-10-02")
    with pytest.raises(Refusal, match="origin_file_scope_invalid"):
        bridge.call("origin_file", name="unrelated-private.json")
    canary.stage(bridge, plan, receipt)
    with ledger.lock(), pytest.raises(Refusal, match="date_scope_invalid"):
        ledger.put({"date": "2026-10-02", "state": "failed"})
    assert ledger.rows() == []


def test_full_qa_publication_and_restart_preserve_normal_history(fixture):
    bridge, ledger, api, receipt, plan, cache, script, crm, pages = fixture
    before = bridge.call("origin")
    canary.stage(bridge, plan, receipt)
    cycles = []
    def tick(_):
        cycles.append(1)
        current = ledger.get(canary.DAY)
        assert len(cycles) < 12, {"state": current["state"], "qa": {k: v for k, v in current.get("qa", {}).items() if k in {"state", "error", "artifact_checks", "observation_failures"}}, "delivery": {k: {a:b for a,b in v.items() if a in {"state", "error"}} for k,v in current.get("delivery", {}).items()}}
    result = canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: NOW, sleep=tick)
    assert result["state"] == "completed" and result["qa_state"] == "validated", (result, ledger.get(canary.DAY).get("qa"))
    assert result["root_turn_status"] == result["qa_turn_status"] == "completed"
    assert all(value["receipt"]["readback_verified"] for value in result["delivery"].values())
    assert len(api.payloads) == len(api.inputs) == 1
    assert len(json.loads(crm.read_text())["values"]) == 6 and len(json.loads(pages.read_text())) == 1
    assert bridge.call("origin") == before
    row = ledger.get(canary.DAY)
    assert row["metadata"]["purpose"] == "blueprint_research_perplexity_canary"
    assert row["create_payload"] == api.payloads[0]
    assert row["cleanup_required"] is True and row["budget_is_hard_cap"] is False
    assert api.payloads[0]["agent"]["tools"] == canary.search.tools()
    assert "no prospect-count stopping rule" in api.payloads[0]["input"]
    assert api.payloads[0]["environment"]["network"] == {"access": "disabled"}
    assert render.export_snapshot(bridge, canary.DAY, cache / "export")["missing_files"] == []
    bridge.close()
    restarted = canary.CanaryBridge(script=script)
    try:
        api.ledger = FirestoreLedger(restarted)
        recovered = canary.run(restarted, cache, execute=False, api_factory=lambda *_: api,
                               clock=lambda: NOW+timedelta(days=1))
        assert recovered["state"] == "completed" and len(api.payloads) == len(api.inputs) == 1
        assert len(json.loads(crm.read_text())["values"]) == 6 and len(json.loads(pages.read_text())) == 1
        assert restarted.call("origin") == before
        row = api.ledger.get(canary.DAY)
        receipt = {"session_id": row["session_id"], "environment_id": row["environment_id"],
                   "action_time_approval_reference": "synthetic-exact-canary-deletion-after-portable-backup"}
        with pytest.raises(Refusal, match="still_present"):
            canary.record_cleanup(restarted, cache, receipt, api_factory=lambda *_: api)
        api.absent = True
        cleaned = canary.record_cleanup(restarted, cache, receipt, api_factory=lambda *_: api)
        assert cleaned["cleanup_required"] is False
        assert not any(call[0] == "DELETE" for call in api.calls)
        assert restarted.call("origin") == before
    finally:
        restarted.close()


def test_unknown_create_reply_never_creates_twice(fixture):
    bridge, ledger, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    api.lost_create_reply = True
    first = canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: NOW)
    assert first["state"] == "creation_unresolved" and len(api.payloads) == 1
    api.lost_create_reply = False
    cycles = []
    def tick(_):
        cycles.append(1)
        assert len(cycles) < 12, ledger.get(canary.DAY).get("qa")
    final = canary.run(bridge, cache, execute=False, api_factory=lambda *_: api, clock=lambda: NOW, sleep=tick)
    assert final["state"] == "completed" and len(api.payloads) == len(api.inputs) == 1, (final, ledger.get(canary.DAY).get("qa"))
    assert ledger.get(canary.DAY)["create_payload"] == api.payloads[0]


@pytest.mark.parametrize("phase", ["create", "qa"])
@pytest.mark.parametrize("stop_at", ["safe", "claim"])
def test_late_stop_after_slow_reads_or_claim_never_posts(phase, stop_at, monkeypatch):
    flag, posts = {"stop": False}, []
    row = {"started_at": NOW.isoformat(), "research_runtime_seconds": 1200}
    control = {"enabled": True, "workflow": {"enabled": True}, "canary": {"admission": APPROVAL}}
    class Bridge:
        def call(self, op, **_):
            if op == ("create_check" if phase == "create" else "qa_check") and stop_at == "claim":
                flag["stop"] = True
            return control if op == "control" else True
    provider = canary.CanaryProvider.__new__(canary.CanaryProvider)
    provider.ledger = SimpleNamespace(bridge=Bridge(), get=lambda _: row)
    provider.stopped = lambda: flag["stop"]
    provider.clock = lambda: NOW
    provider.safe = lambda *_: flag.update(stop=True) if stop_at == "safe" else None
    provider.api = SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=lambda *_args, **kwargs: posts.append(kwargs))))
    monkeypatch.setattr(canary.Provider, "create", lambda *_: posts.append("create"))
    with pytest.raises(Refusal, match="canary_stopped"):
        if phase == "create":
            provider.create({"metadata": {}})
        else:
            provider.qa_input("sess-test", {}, "key", canary.DAY, "digest", int((NOW+timedelta(seconds=600)).timestamp()*1000))
    assert posts == []


def test_expired_immutable_research_deadline_never_creates(monkeypatch):
    posts = []
    row = {"started_at": (NOW-timedelta(seconds=1201)).isoformat(), "research_runtime_seconds": 1200}
    control = {"enabled": True, "canary": {"admission": APPROVAL}}
    provider = canary.CanaryProvider.__new__(canary.CanaryProvider)
    provider.ledger = SimpleNamespace(bridge=SimpleNamespace(call=lambda op, **_: control if op=="control" else True), get=lambda _: row)
    provider.safe = lambda *_: None
    provider.clock = lambda: NOW
    monkeypatch.setattr(canary.Provider, "create", lambda *_: posts.append("create"))
    with pytest.raises(Refusal, match="before_create"):
        provider.create({"metadata": {}})
    assert posts == []


def test_unknown_usage_is_pending_and_never_zero():
    api = SimpleNamespace(listing=lambda *_: [{"id": "root", "usage": None}])
    observation = canary.spend(api, {"session_id": "sess-test"})
    assert observation["usage_state"] == "pending"
    assert observation["known"] is observation["hard_total_cap"] is False
    assert observation["estimate_usd"] is observation["reported_estimate_usd"] is None


def test_partial_usage_does_not_claim_complete_cost():
    api = SimpleNamespace(listing=lambda *_: [
        {"id": "root", "status": "completed", "usage": {"input_tokens": 100, "output_tokens": 50}},
        {"id": "qa", "status": "in_progress", "usage": None}])
    observation = canary.spend(api, {"session_id": "sess-test"})
    assert observation["known"] is False and observation["estimate_usd"] is None
    assert float(observation["reported_estimate_usd"]) > 0
    assert observation["reported_turn_count"] == 1
    assert observation["pending_turns"] == [{"turn_id": "qa", "status": "in_progress"}]


def test_usage_read_failure_is_unavailable_and_never_zero():
    def unavailable(*_):
        raise RuntimeError("upstream private error")
    observation = canary.spend(SimpleNamespace(listing=unavailable), {"session_id": "sess-test"})
    assert observation == {"known": False, "estimate_usd": None, "reported_estimate_usd": None,
                           "usage_state": "unavailable", "hard_total_cap": False}


@pytest.mark.parametrize("usage", [None, {"input_tokens": 2_000_000, "output_tokens": 50}])
def test_baseline_paid_guard_does_not_invent_a_spend_cap(fixture, usage):
    bridge, _, _, receipt, plan, _, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    provider = canary.CanaryProvider.__new__(canary.CanaryProvider)
    provider.ledger = FirestoreLedger(bridge)
    provider.listing = lambda *_: [{"id": "root", "status": "in_progress", "usage": usage}]
    row = {"session_id": "sess-test", "canary": bridge.call("control")["canary"]}
    provider.safe(row)
    observation = row["canary_model_estimate"]
    assert observation["hard_total_cap"] is False
    if usage is None:
        assert observation["estimate_usd"] is None and observation["usage_state"] == "pending"
    else:
        assert float(observation["estimate_usd"]) > 8  # Observed, not an arbitrary cancellation threshold.


def test_missing_in_progress_and_terminal_usage_allows_normal_tools_qa_and_publication(fixture, monkeypatch):
    bridge, _, api, receipt, plan, cache, _, crm, pages = fixture
    canary.stage(bridge, plan, receipt)
    api.turn_status = "in_progress"
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "baseline_root",
                    "name": canary.search.SEARCH, "arguments": {"query": "synthetic exact operator task"}}]
    state = {"qa": "in_progress", "accounting_arrived": False, "cycles": 0}
    original_listing, original_input = api.listing, api.qa_input

    def listing(resource, sid=None):
        values = original_listing(resource, sid)
        if resource == "turns":
            for turn in values:
                if not state["accounting_arrived"]:
                    turn["usage"] = None
                if turn["id"] == "turn_qa":
                    turn["status"] = state["qa"]
        return values

    def qa_input(*args):
        original_input(*args)
        api.actions = [{"type": "function_call", "turn_id": "turn_qa", "call_id": "baseline_qa",
                        "name": canary.search.SEARCH, "arguments": {"query": "synthetic task evidence QA"}}]

    def tick(_):
        state["cycles"] += 1
        assert state["cycles"] < 12 and not api.cancellations
        if api.qa_exists:
            if any(event[1]["turn_id"] == "turn_qa" for event in api.result_events):
                state["qa"], api.actions = "completed", []
        elif api.result_events:
            api.turn_status, api.actions = "completed", []

    monkeypatch.setattr(api, "listing", listing)
    monkeypatch.setattr(api, "qa_input", qa_input)
    result = canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: NOW, sleep=tick)
    assert result["state"] == "completed" and result["qa_state"] == "validated"
    assert len(api.payloads) == len(api.inputs) == 1 and len(api.executions) == 2
    assert not api.cancellations
    assert result["canary_model_estimate"]["estimate_usd"] is None
    assert result["canary_model_estimate"]["usage_state"] == "pending"
    assert len(json.loads(crm.read_text())["values"]) == 6 and len(json.loads(pages.read_text())) == 1
    assert all(value["receipt"]["readback_verified"] for value in result["delivery"].values())
    state["accounting_arrived"] = True
    reconciled = canary.run(bridge, cache, execute=False, api_factory=lambda *_: api, clock=lambda: NOW)
    assert reconciled["state"] == "completed"
    assert reconciled["canary_model_estimate"]["usage_state"] == "reported_best_effort"
    assert reconciled["canary_model_estimate"]["reported_turn_count"] == 2
    assert float(reconciled["canary_model_estimate"]["estimate_usd"]) > 0
    assert len(api.payloads) == len(api.inputs) == 1 and len(api.executions) == 2


def test_pending_usage_does_not_bypass_the_original_research_deadline(fixture, monkeypatch):
    bridge, _, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    api.turn_status = "in_progress"
    original_listing = api.listing
    state = {"now": NOW, "cycles": 0}

    def listing(resource, sid=None):
        values = original_listing(resource, sid)
        if resource == "turns":
            for turn in values:
                turn["usage"] = None
        return values

    def tick(_):
        state["cycles"] += 1
        assert state["cycles"] < 5
        if state["cycles"] == 1:
            assert not api.cancellations
            state["now"] = NOW+timedelta(seconds=1201)
        else:
            assert len(api.cancellations) == 1
            api.turn_status = "cancelled"

    monkeypatch.setattr(api, "listing", listing)
    result = canary.run(bridge, cache, execute=True, api_factory=lambda *_: api,
                        clock=lambda: state["now"], sleep=tick)
    assert result["state"] == "cancelled" and len(api.cancellations) == len(api.payloads) == 1
    assert not api.inputs and result["canary_model_estimate"]["estimate_usd"] is None


def test_pending_qa_usage_preserves_total_deadline_and_terminal_cancel_recovery(fixture, monkeypatch):
    bridge, ledger, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    state = {"now": NOW, "qa": "in_progress"}
    original_listing = api.listing

    def listing(resource, sid=None):
        values = original_listing(resource, sid)
        if resource == "turns":
            for turn in values:
                turn["usage"] = None
                if turn["id"] == "turn_qa":
                    turn["status"] = state["qa"]
        return values

    def tick(_):
        assert api.qa_exists and not api.cancellations
        state["now"] = NOW+timedelta(seconds=1801)

    monkeypatch.setattr(api, "listing", listing)
    first = canary.run(bridge, cache, execute=True, api_factory=lambda *_: api,
                       clock=lambda: state["now"], sleep=tick)
    assert first["qa_state"] == "qa_cancel_pending"
    assert first["observer_error"] == "canary_total_observation_deadline"
    assert len(api.payloads) == len(api.inputs) == len(api.cancellations) == 1
    original_deadline = ledger.get(canary.DAY)["qa"]["deadline_ms"]
    state["qa"] = "cancelled"
    terminal = canary.run(bridge, cache, execute=False, api_factory=lambda *_: api, clock=lambda: state["now"])
    assert terminal["qa_state"] == "qa_blocked" and terminal["qa_turn_status"] == "cancelled"
    assert ledger.get(canary.DAY)["qa"]["deadline_ms"] == original_deadline
    assert len(api.payloads) == len(api.inputs) == len(api.cancellations) == 1
    assert not api.executions


@pytest.mark.parametrize("execute", [False, True])
def test_existing_cancelled_attempt_is_never_resurrected_or_recreated(fixture, execute):
    bridge, ledger, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    api.turn_status = "in_progress"
    runner = Runner(ledger, canary.render.configured(bridge, cache), api, clock=lambda: NOW)
    runner.start_or_resume()
    runner.cancel_current(canary.DAY, "historical_unknown_usage_guard")
    api.turn_status = "cancelled"
    result = canary.run(bridge, cache, execute=execute, api_factory=lambda *_: api, clock=lambda: NOW)
    assert result["state"] == "cancelled" and len(api.payloads) == len(api.cancellations) == 1
    assert ledger.get(canary.DAY)["cancel_attempted"] is True
    assert not api.inputs and not api.executions


def test_test_approval_is_not_recurring_or_delete_approval():
    assert canary.admission(APPROVAL, NOW)["scope"] == canary.SCOPE
    with pytest.raises(Refusal, match="admission"):
        canary.admission({**APPROVAL, "scope": "delete-session"}, NOW)
    with pytest.raises(Refusal, match="admission"):
        canary.admission({**APPROVAL, "ceiling_usd": 26}, NOW)


def closed_original(fixture):
    """Real fake-provider cancellation/cleanup path, not an empty namespace."""
    bridge, ledger, api, receipt, plan, cache, *_ = fixture
    canary.stage(bridge, plan, receipt)
    api.turn_status = "in_progress"
    runner = Runner(ledger, canary.render.configured(bridge, cache), api, clock=lambda: NOW)
    runner.start_or_resume()
    runner.cancel_current(canary.DAY, "synthetic_historical_stop")
    api.turn_status = "cancelled"
    assert canary.run(bridge, cache, api_factory=lambda *_: api, clock=lambda: NOW)["state"] == "cancelled"
    cleanup_fake(bridge, ledger, api, cache)
    retained = ledger.get(canary.DAY)
    bridge.close()
    return retained


def cleanup_fake(bridge, ledger, api, cache):
    class Missing(Exception):
        status_code = 404

    def absent(*_):
        raise Missing

    get = api.get
    api.get = absent
    try:
        row = ledger.get(canary.DAY)
        canary.record_cleanup(bridge, cache, {"session_id": row["session_id"],
            "environment_id": row["environment_id"], "action_time_approval_reference": "synthetic-separate-cleanup"},
            api_factory=lambda *_: api)
    finally:
        api.get = get


def attempt_bridge(fixture, monkeypatch, number, date="2026-10-01"):
    for name in ("TEST", "ROOT", "DAY", "BASELINE"):
        monkeypatch.setattr(canary, name, getattr(canary, name))
    canary.select_attempt(number, date)
    cache, source = fixture[5], fixture[6]
    target = cache / f"attempt-{number}"
    target.mkdir(exist_ok=True)
    generated = canary.driver(ROOT, target)
    driver = target / "hermetic-driver.mjs"
    driver.write_text(source.read_text().replace(
        json.dumps((cache / "canary-bridge.mjs").as_uri()), json.dumps(generated.as_uri())))
    bridge = canary.CanaryBridge(script=driver)
    ledger = FirestoreLedger(bridge)
    api = API(ledger, policy_bundle(now=NOW)[3])
    approval = {**APPROVAL, "test_id": canary.TEST, "scope": canary.BASELINE_SCOPE,
                "authority_reference": canary.BASELINE_AUTHORITY}
    return bridge, ledger, api, approval, driver


def complete_baseline(bridge, api, cache):
    """Exercise the actual application tool/result path before terminal output."""
    api.turn_status = "in_progress"
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "search_" + canary.TEST,
                    "name": canary.search.SEARCH, "arguments": {"query": "synthetic exact operator task"}}]

    def tick(_):
        if api.result_events:
            api.turn_status, api.actions = "completed", []

    return canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: NOW, sleep=tick)


def test_baseline_repeat_restart_and_success_close_retries(fixture, monkeypatch):
    prior = closed_original(fixture)
    bridge, ledger, api, approval, driver = attempt_bridge(fixture, monkeypatch, 1)
    receipt, cache = fixture[3], fixture[5]
    try:
        before = bridge.call("origin")
        plan = canary.inspect(bridge, approval, receipt, api, cache, now=NOW)
        assert plan["baseline_budget"]["attempts"] == []
        assert bridge.call("control") is None  # inspection has no writes
        canary.stage(bridge, plan, receipt)
        canary.stage(bridge, plan, receipt)  # same unconsumed intent/inputs only
        assert len(bridge.call("baseline_status")["attempts"]) == 1
        result = complete_baseline(bridge, api, cache)
        assert result["state"] == "completed" and len(api.payloads) == len(api.inputs) == 1
        assert bridge.call("origin") == before
        cleanup_fake(bridge, ledger, api, cache)
        finished = ledger.get(canary.DAY)
        bridge.close()
        bridge = canary.CanaryBridge(script=driver)
        api.ledger = FirestoreLedger(bridge)
        assert canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: NOW)["state"] == "completed"
        assert len(api.payloads) == len(api.inputs) == 1
        budget = bridge.call("baseline_status")
        assert budget["complete_model_estimate_usd"] is not None and budget["prior_scope_included"] is False
        assert budget["baseline"]["soft_total_usd"] == 25 and len(budget["attempts"]) == 1
        assert float(budget["reported_search_estimate_usd"]) > 0 and budget["total_billed_usd"] is None
        assert finished["delivery"]["notion"]["key"] == "blueprint-research-canary:baseline-20261002-attempt-0001:notion"
        assert prior["state"] == "cancelled" and prior["cleanup_required"] is False
        assert finished["canary"]["baseline"]["attempt_number"] == 1
    finally:
        bridge.close()
    next_bridge, _, next_api, next_approval, _ = attempt_bridge(fixture, monkeypatch, 2)
    try:
        with pytest.raises(Refusal, match="baseline_already_completed"):
            canary.inspect(next_bridge, next_approval, receipt, next_api, cache, now=NOW)
        assert next_bridge.call("control") is None and not next_api.payloads
    finally:
        next_bridge.close()


def test_baseline_cancelled_retry_preserves_claims_and_cumulative_pending_usage(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    receipt, cache = fixture[3], fixture[5]
    try:
        canary.stage(bridge, canary.inspect(bridge, approval, receipt, api, cache, now=NOW), receipt)
        api.turn_status = "in_progress"
        runner = Runner(ledger, canary.render.configured(bridge, cache), api, clock=lambda: NOW)
        runner.start_or_resume()
        first = canary.run(bridge, cache, execute=False, api_factory=lambda *_: api, clock=lambda: NOW, stopped=lambda: True)
        assert first["state"] == "cancel_pending" and len(api.payloads) == 1
        api.turn_status = "cancelled"
        assert canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: NOW)["state"] == "cancelled"
        assert len(api.payloads) == 1  # same identity never recreated
        with ledger.lock():
            row = ledger.get(canary.DAY)
            row["canary_model_estimate"] = {"known": False, "usage_state": "pending", "estimate_usd": None,
                                            "reported_estimate_usd": "0.15"}
            ledger.put(row)
        bridge.close()
        blocked, _, blocked_api, blocked_approval, _ = attempt_bridge(fixture, monkeypatch, 2)
        try:
            with pytest.raises(Refusal, match="predecessor_unreconciled"):
                canary.inspect(blocked, blocked_approval, receipt, blocked_api, cache, now=NOW)
            assert not blocked_api.payloads
        finally:
            blocked.close()
        # Reopen the exact previous attempt and perform separately approved
        # synthetic cleanup; selecting attempt 2 never mutates its predecessor.
        bridge, ledger, _, _, _ = attempt_bridge(fixture, monkeypatch, 1)
        cleanup_fake(bridge, ledger, api, cache)
        retained = ledger.get(canary.DAY)
    finally:
        bridge.close()
    bridge, ledger, api2, approval2, _ = attempt_bridge(fixture, monkeypatch, 2)
    try:
        canary.stage(bridge, canary.inspect(bridge, approval2, receipt, api2, cache, now=NOW), receipt)
        result = complete_baseline(bridge, api2, cache)
        assert result["state"] == "completed" and len(api2.payloads) == len(api2.inputs) == 1
        budget = bridge.call("baseline_status")
        assert len(budget["attempts"]) == 2 and budget["complete_model_estimate_usd"] is None
        assert float(budget["reported_model_estimate_usd"]) > 0
        assert budget["attempts"][0]["legacy_model_estimate_usd"] == "0.15"
        assert budget["attempts"][0]["usage_state"] == "pending"
        assert budget["baseline"]["soft_total_usd"] == 25  # not $25 per retry
        assert retained["state"] == "cancelled" and retained["canary"]["baseline"]["attempt_number"] == 1
    finally:
        bridge.close()


@pytest.mark.parametrize("number,date", [(0, "2026-10-01"), (-1, "2026-10-01"), (True, "2026-10-01"),
    (1, None), (1, "2026-10-01/other"), (1, "2026-10-01T00:00:00")])
def test_baseline_identity_rejects_invalid_host_parameters(number, date):
    with pytest.raises(Refusal, match="attempt_identity_invalid"):
        canary.select_attempt(number, date)


def test_baseline_attempt_order_and_exact_approval(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 2)
    try:
        with pytest.raises(Refusal, match="attempt_order_invalid"):
            canary.inspect(bridge, approval, fixture[3], api, fixture[5], now=NOW)
        assert ledger.rows() == [] and bridge.call("control") is None
        with pytest.raises(Refusal, match="admission"):
            canary.admission({**approval, "authority_reference": APPROVAL["authority_reference"]}, NOW)
        with pytest.raises(Refusal, match="admission"):
            canary.admission({**approval, "scope": canary.SCOPE}, NOW)
        assert not api.payloads and not api.inputs
    finally:
        bridge.close()


def test_baseline_concurrent_stage_transaction_keeps_one_attempt(fixture, monkeypatch):
    closed_original(fixture)
    bridge, _, api, approval, driver = attempt_bridge(fixture, monkeypatch, 1)
    try:
        plan = canary.inspect(bridge, approval, fixture[3], api, fixture[5], now=NOW)
    finally:
        bridge.close()
    source = driver.read_text()
    old = "const value=await channel.call(r);"
    probe = ("if(r.op==='test_stage_race'){let arrived=0,release;const gate=new Promise(resolve=>release=resolve);"
        "const guard=channel.guard.bind(channel);channel.guard=async(...a)=>{await guard(...a);"
        "if(++arrived===2)release();await gate;};"
        "const outcomes=await Promise.allSettled([channel.call(r.request),channel.call(r.request)]);"
        "channel.guard=guard;process.stdout.write(JSON.stringify({ok:true,value:outcomes.map(x=>"
        "({status:x.status,value:x.value,error:x.reason?.message}))})+'\\n');continue;}" + old)
    driver.write_text(source.replace(old, probe))
    bridge = canary.CanaryBridge(script=driver)
    try:
        outcomes = bridge.call("test_stage_race", request={"op": "stage", "value": plan["candidate"]})
        assert sum(x["status"] == "fulfilled" for x in outcomes) == 1
        assert any(x.get("error") == "canary_baseline_predecessor_changed" for x in outcomes)
        assert len(bridge.call("baseline_status")["attempts"]) == 1
        assert not api.payloads
        canary.stage(bridge, plan, fixture[3])
        assert len(bridge.call("baseline_status")["attempts"]) == 1
    finally:
        bridge.close()


def test_baseline_completed_partial_report_allows_retry_with_distinct_publication_identity(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    receipt, cache = fixture[3], fixture[5]
    try:
        canary.stage(bridge, canary.inspect(bridge, approval, receipt, api, cache, now=NOW), receipt)
        report = json.loads(api.raw)
        report["coverage"].update(completion_state="time_interrupted", stop_reason="Synthetic interruption",
            shortfall_reason="Synthetic unresolved branch", unresolved_promising_branches=["Synthetic second task"])
        api.raw = canonical(report).encode()
        result = complete_baseline(bridge, api, cache)
        assert result["state"] == "completed" and len(api.payloads) == len(api.inputs) == 1
        cleanup_fake(bridge, ledger, api, cache)
        first = ledger.get(canary.DAY)
        assert first["packet"]["coverage"]["completion_state"] == "time_interrupted"
    finally:
        bridge.close()
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 2)
    try:
        canary.stage(bridge, canary.inspect(bridge, approval, receipt, api, cache, now=NOW), receipt)
        result = complete_baseline(bridge, api, cache)
        assert result["state"] == "completed" and len(api.payloads) == len(api.inputs) == 1
        second = ledger.get(canary.DAY)
        assert first["delivery"]["notion"]["key"] != second["delivery"]["notion"]["key"]
        assert first["delivery"]["notion"]["receipt"]["reference"] != second["delivery"]["notion"]["receipt"]["reference"]
        assert len(json.loads(fixture[8].read_text())) == 2
        # Fresh CRM dedupe excludes the already-published task from the retry.
        assert len(json.loads(fixture[7].read_text())["values"]) == 6
        assert len(bridge.call("baseline_status")["attempts"]) == 2
    finally:
        bridge.close()


def test_scoped_baseline_large_qa_report_preserves_identity_content_and_export_claims(fixture, monkeypatch, tmp_path):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    receipt, cache = fixture[3], fixture[5]
    summary = "Scoped supported source https://plant.example/tasks; interest unknown. " * 3500
    original = api.qa_input
    def large(*args, **kwargs):
        original(*args, **kwargs)
        api.qa_result["summary"] = summary
    api.qa_input = large
    try:
        canary.stage(bridge, canary.inspect(bridge, approval, receipt, api, cache, now=NOW), receipt)
        result = complete_baseline(bridge, api, cache)
        for _ in range(10):
            if result["state"] == "completed":
                break
            assert result["observer_state"] == "publication_pending"
            result = canary.run(bridge, cache, api_factory=lambda *_: api, clock=lambda: NOW, sleep=lambda _: None)
        assert result["state"] == "completed"
        row = ledger.get(canary.DAY)
        delivery = row["delivery"]["notion"]
        assert delivery["key"] == "blueprint-research-canary:" + canary.TEST + ":notion"
        assert delivery["payload"]["summary"] == "Blueprint baseline attempt " + canary.TEST + "\n" + summary
        assert summary in "".join(delivery["plan"]["paragraphs"][1:])
        assert len(delivery["plan"]["batches"]) > 1
        exported = tmp_path / "scoped-large-export"
        assert render.export_snapshot(bridge, canary.DAY, exported)["missing_files"] == []
        proof = json.loads(json.loads((exported / "publication-manifest.json").read_bytes())["manifest_json"])
        assert len(proof["publication_batches"]["notion"]) == len(delivery["plan"]["batches"])
        assert len(api.payloads) == len(api.inputs) == 1
    finally:
        bridge.close()


def test_scoped_pagination_rechecks_origin_after_inventory_before_append(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, driver = attempt_bridge(fixture, monkeypatch, 1)
    bridge.close()
    source = driver.read_text()
    needle = "for await(const line of createInterface({input:process.stdin}))"
    hook = ("const originalProgress=channel.store.publisher?.notionProgress;"
        "if(originalProgress)channel.store.publisher.notionProgress=async(...args)=>{"
        "const result=await originalProgress(...args);if(result.step?.number>0){"
        "await normal.acquire();const origin=await normal.get('2026-10-01');"
        "await normal.put({...origin,cleanup_receipt:{action_time_approval_reference:'changed-after-inventory'}});"
        "await normal.release();}return result;};")
    assert needle in source
    driver.write_text(source.replace(needle, hook + needle))
    bridge = canary.CanaryBridge(script=driver)
    ledger = api.ledger = FirestoreLedger(bridge)
    original = api.qa_input
    def large(*args, **kwargs):
        original(*args, **kwargs)
        api.qa_result["summary"] = "Scoped supported source https://plant.example/tasks; interest unknown. " * 3500
    api.qa_input = large
    try:
        canary.stage(bridge, canary.inspect(bridge, approval, fixture[3], api, fixture[5], now=NOW), fixture[3])
        result = complete_baseline(bridge, api, fixture[5])
        assert result["observer_state"] == "publication_pending"
        with ledger.lock(), pytest.raises(Refusal, match="publication_canary_authority_unavailable_or_changed"):
            bridge.call("publish", day=canary.DAY)
        proof = json.loads(bridge.call("snapshot", day=canary.DAY)["publication_manifest"]["manifest_json"])
        assert set(proof["publication_batches"]["notion"]) == {"0", "1"}
        pages = json.loads(fixture[8].read_bytes())
        assert len(pages[-1]["children"]) == ledger.get(canary.DAY)["delivery"]["notion"]["plan"]["batches"][0]["end"]
        assert len(api.payloads) == len(api.inputs) == 1
    finally:
        bridge.close()


def test_identical_empty_partial_reports_have_distinct_notion_titles(fixture, monkeypatch):
    closed_original(fixture)
    records = []
    for number in (1, 2):
        bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, number)
        try:
            canary.stage(bridge, canary.inspect(bridge, approval, fixture[3], api, fixture[5], now=NOW), fixture[3])
            report = json.loads(api.raw)
            report["candidates"] = []
            report["coverage"].update(completion_state="time_interrupted", stop_reason="Synthetic interruption",
                shortfall_reason="Synthetic unresolved branch", unresolved_promising_branches=["Synthetic second task"])
            api.raw = canonical(report).encode()
            assert complete_baseline(bridge, api, fixture[5])["state"] == "completed"
            cleanup_fake(bridge, ledger, api, fixture[5])
            records.append(ledger.get(canary.DAY))
            assert len(api.payloads) == len(api.inputs) == 1
        finally:
            bridge.close()
    assert records[0]["review"]["summary"] == records[1]["review"]["summary"]
    deliveries = [row["delivery"]["notion"] for row in records]
    assert deliveries[0]["plan"]["title"] != deliveries[1]["plan"]["title"]
    assert deliveries[0]["receipt"]["reference"] != deliveries[1]["receipt"]["reference"]
    assert len(json.loads(fixture[8].read_text())) == 2
    assert len(json.loads(fixture[7].read_text())["values"]) == 5


def test_staged_uncreated_attempt_can_be_closed_after_due_date_advances(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    try:
        plan = canary.inspect(bridge, approval, fixture[3], api, fixture[5], now=NOW)
        canary.stage(bridge, plan, fixture[3])
        origin = bridge.call("origin")
        with pytest.raises(Refusal, match="admission"):
            canary.admission(approval, datetime(2026, 10, 2, 13, tzinfo=timezone.utc))
        with ledger.lock(), pytest.raises(Refusal, match="runner_overlap"):
            bridge.call("abandon_unstarted")
        closed = bridge.call("abandon_unstarted")
        assert closed["state"] == "abandoned_unstarted" and closed["provider_mutations"] == 0
        assert bridge.call("abandon_unstarted")["existing"] is True
        assert ledger.rows() == [] and not api.payloads
        assert bridge.call("origin") == origin
        assert bridge.call("control")["enabled"] is False
        assert bridge.call("baseline_status")["attempts"][0]["usage_state"] == "not_started_no_provider_intent"
        with pytest.raises(Refusal, match="attempt_abandoned"):
            canary.stage(bridge, plan, fixture[3])
    finally:
        bridge.close()
    next_bridge, next_ledger, next_api, next_approval, _ = attempt_bridge(fixture, monkeypatch, 2, "2026-10-02")
    try:
        assert canary.admission(next_approval, datetime(2026, 10, 2, 13, tzinfo=timezone.utc))
        assert next_bridge.call("baseline_check") is True
        assert next_ledger.rows() == [] and next_bridge.call("control") is None
        assert not next_api.payloads
    finally:
        next_bridge.close()


def test_no_create_abandonment_never_releases_existing_uncertain_intent(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    try:
        canary.stage(bridge, canary.inspect(bridge, approval, fixture[3], api, fixture[5], now=NOW), fixture[3])
        api.turn_status = "in_progress"
        api.lost_create_reply = True
        runner = Runner(ledger, canary.render.configured(bridge, fixture[5]), api, clock=lambda: NOW)
        runner.start_or_resume()
        before, control, budget = ledger.get(canary.DAY), bridge.call("control"), bridge.call("baseline_status")
        assert before["state"] == "creation_unresolved"
        with pytest.raises(Refusal, match="abandonment_not_proven"):
            bridge.call("abandon_unstarted")
        assert ledger.get(canary.DAY) == before and bridge.call("control") == control
        assert bridge.call("baseline_status") == budget and len(api.payloads) == 1
    finally:
        bridge.close()


def null_operator_output(api):
    from copy import deepcopy

    from tests.test_daily_research_knowledge import delta
    out = json.loads(api.raw)
    vendor = delta()
    vendor['evidence'][0]['source_checked_at'] = canary.DAY
    operator = deepcopy(vendor)
    operator.update(record_id=None, fact_id=None, reason='discovery',
                    proposed_statement='Synthetic ordinary site observation; robot maturity unknown')
    operator['evidence'] *= 3
    for evidence in operator['evidence']:
        evidence.update(classification='operator', evidence_level=None)
    out['proposed_knowledge_deltas'] = [vendor, deepcopy(vendor), operator]
    return out


def recovery_receipt(row):
    return {'session_id': row['session_id'], 'turn_id': row['turn_id'],
            'raw_output_sha256': row['raw_output_digest'], 'approval_reference': 'synthetic-recovery-approval',
            'scope': 'quarantine-null-operator-deltas-no-inference-no-publication'}


def test_retained_null_operator_proposal_recovery_is_offline_strict_and_exportable(fixture):
    bridge, ledger, api, receipt, plan, cache, _, _crm, pages = fixture
    canary.stage(bridge, plan, receipt)
    original = null_operator_output(api)
    api.raw = canonical(original).encode()
    runner = Runner(ledger, canary.render.configured(bridge, cache), api, clock=lambda: NOW)
    failed = runner.start_or_resume()
    assert failed['state'] == 'failed' and failed['error'] == 'knowledge_delta_evidence_invalid'
    admission = recovery_receipt(failed)
    before = ledger.read_bytes(canary.DAY + '-artifact.json')
    # No API is available for the recovery; a provider request would fail this test.
    offline = Runner(ledger, runner.config, None, clock=lambda: NOW)
    recovered = offline.recover_output(canary.DAY, admission)
    assert recovered['state'] == 'awaiting_review' and recovered['turn_id'] == failed['turn_id']
    assert ledger.read_bytes(canary.DAY + '-artifact.json') == before
    assert json.loads(ledger.read_bytes(canary.DAY + '-output.json')) == original
    assert recovered['packet']['proposed_knowledge_deltas'] == original['proposed_knowledge_deltas'][:2]
    quarantined = recovered['packet']['output_recovery']['quarantined_proposals']
    assert quarantined[0]['delta_index'] == 2 and quarantined[0]['approved'] is False
    assert len(quarantined[0]['invalid_fields']) == 3
    assert all(e['evidence_level'] is None for e in quarantined[0]['proposal']['evidence'])
    assert len(api.payloads) == 1 and not api.inputs and not pages.exists()
    assert offline.recover_output(canary.DAY, admission) == recovered
    exported = cache / 'recovered-export'
    assert not canary.render.export_snapshot(bridge, canary.DAY, exported)['missing_files']
    assert (exported / (canary.DAY + '-recovery.json')).exists()
    with pytest.raises(Refusal, match='binding_or_state_invalid'):
        offline.recover_output(canary.DAY, {**admission, 'raw_output_sha256': '0' * 64})


def test_quarantine_does_not_weaken_candidate_or_remaining_delta_evidence(fixture):
    bridge, ledger, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    original = null_operator_output(api)
    api.raw = canonical(original).encode()
    runner = Runner(ledger, canary.render.configured(bridge, cache), api, clock=lambda: NOW)
    failed = runner.start_or_resume()
    # Another invalid, nonnull delta must still block recovery.
    original['proposed_knowledge_deltas'][0]['evidence'][0]['evidence_level'] = 'invented_deployment'
    from tools.daily_research import recovery
    from tools.daily_research.runner import validate_output
    derived, _ = recovery.quarantine_null_operator_deltas(original)
    with pytest.raises(Refusal, match='knowledge_delta_evidence_invalid'):
        validate_output(derived, canary.DAY, set(), contract_version=3,
                        knowledge_context=failed['knowledge_context'], refresh_policy=failed['refresh_policy'], observed_at=NOW)
    assert ledger.get(canary.DAY)['state'] == 'failed' and not api.inputs


def continued_qa_receipt(row):
    return {'authority_reference': 'Sentinel_dac3e21091cc819196cb4e5799b7229d',
            'scope': 'same-session-recovered-qa-and-existing-publication-no-new-research',
            'baseline_id': 'baseline-20261002', 'soft_total_usd': 25,
            'session_id': row['session_id'], 'root_turn_id': row['turn_id'],
            'raw_output_sha256': row['raw_output_digest'], 'packet_digest': row['packet_digest'],
            'model_observation_digest': canary.digest(row['canary_model_estimate'])}


class RepairAPI(API):
    """Same-session event with optional lost reply, followed by ordinary QA."""
    def __init__(self, ledger, context):
        super().__init__(ledger, context)
        self.repair_calls, self.repair_turns = [], []
        self.repaired_raw, self.lose_repair_reply = self.raw, False

    def repair_input(self, session_id, event, key, day, request_digest, deadline_ms):
        current = self.ledger.get(day)['validation_repairs'][-1]
        assert current['input_attempted'] and current['request_digest'] == request_digest
        assert json.loads(self.ledger.read_bytes(current['input_file'])) == event
        assert key.endswith(':repair:' + str(current['number']))
        self.ledger.bridge.call('repair_check', day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        self.repair_calls.append(key)
        self.repair_turns.append({'id': 'turn_repair_' + str(current['number']), 'session_id': session_id,
            'agent_id': AGENT, 'subagent_id': None, 'status': 'completed', 'completed_at': int(NOW.timestamp()) + 15,
            'usage': None})  # Expected pending usage must not stop useful work.
        if self.lose_repair_reply:
            raise TimeoutError('synthetic accepted repair reply lost')

    def listing(self, resource, session_id=None):
        result = super().listing(resource, session_id)
        if resource == 'turns':
            result.extend(self.repair_turns)
        if resource == 'artifacts':
            from tools.daily_research.recovery import REPAIR_PATH
            result.extend({'id': 'artifact_' + t['id'], 'turn_id': t['id'], 'path': REPAIR_PATH}
                          for t in self.repair_turns)
        return result

    def artifact(self, session_id, artifact_id):
        return self.repaired_raw if artifact_id.startswith('artifact_turn_repair_') else super().artifact(session_id, artifact_id)


def failed_repair_baseline(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, original_api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    canary.stage(bridge, canary.inspect(bridge, approval, fixture[3], original_api, fixture[5], now=NOW), fixture[3])
    context = original_api.ledger.bridge.call('control')
    # Use the already-bound knowledge context, not a new approved snapshot.
    cfg = canary.render.configured(bridge, fixture[5])
    from tools.daily_research.runner import load_knowledge_bundle
    selected, _ = load_knowledge_bundle(cfg, NOW)
    api = RepairAPI(ledger, selected)
    api.raw = canonical(null_operator_output(api)).encode()
    failed = Runner(ledger, cfg, api, clock=lambda: NOW).start_or_resume()
    assert failed['state'] == 'failed' and failed['error'] == 'knowledge_delta_evidence_invalid'
    assert context['canary']['baseline']['soft_total_usd'] == 25
    return bridge, ledger, api, cfg, failed


def test_same_session_agent_repair_then_existing_qa_publication(fixture, monkeypatch):
    bridge, ledger, api, _cfg, failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        original = ledger.read_bytes(canary.DAY + '-artifact.json')
        later = NOW + timedelta(hours=1)
        result = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        assert result['state'] == 'completed', result
        final = ledger.get(canary.DAY)
        assert final['started_at'] == failed['started_at'] and final['turn_id'] == failed['turn_id']
        assert ledger.read_bytes(canary.DAY + '-artifact.json') == original
        assert len(api.payloads) == len(api.repair_calls) == len(api.inputs) == 1
        assert final['packet']['research_revision']['turn_id'] == 'turn_repair_1'
        assert final['qa']['baseline_turn_ids'] == ['turn_1', 'turn_repair_1']
        assert all(value['receipt']['readback_verified'] for value in final['delivery'].values())
        exported = fixture[5] / 'agent-repaired-export'
        assert not canary.render.export_snapshot(bridge, canary.DAY, exported)['missing_files']
        assert (exported / (canary.DAY + '-repair-1-artifact.json')).exists()
    finally:
        bridge.close()


def test_same_session_repair_accepted_unacknowledged_event_is_not_resent(fixture, monkeypatch):
    bridge, ledger, api, cfg, failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        from tools.daily_research.recovery import RepairLoop
        later = NOW + timedelta(hours=1)
        with ledger.lock():
            row = ledger.get(canary.DAY)
            row['validation_repair_authority'] = {'started_at': later.isoformat(), 'duration_seconds': 1800,
                'request': {'scope': 'same-session-validation-repair-and-qa-no-outreach',
                    'authority_reference': 'Sentinel_3b6171ff167c8191b378202c5f0c54c0',
                    'budget_authority_reference': canary.BASELINE['authority_reference'],
                    'baseline_id': 'baseline-20261002', 'soft_total_usd': 25,
                    'session_id': row['session_id'], 'root_turn_id': row['turn_id'],
                    'raw_output_sha256': row['raw_output_digest']}}
            ledger.put(row)
        api.lose_repair_reply = True
        loop = RepairLoop(ledger, cfg, api, clock=lambda: later)
        uncertain = loop.step(canary.DAY)
        assert uncertain['validation_repairs'][-1]['state'] == 'input_unresolved'
        # New controller object resumes durable intent through GETs only.
        recovered = RepairLoop(ledger, cfg, api, clock=lambda: later).step(canary.DAY)
        assert recovered['state'] == 'awaiting_review' and len(api.repair_calls) == 1
        assert len(api.payloads) == 1 and not api.inputs
        assert recovered['raw_output_digest'] == failed['raw_output_digest']
    finally:
        bridge.close()


def test_repair_loop_stops_repeated_failure_without_new_session(fixture, monkeypatch):
    bridge, ledger, api, _cfg, _failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        api.repaired_raw = api.raw
        result = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api,
                                     clock=lambda: NOW + timedelta(hours=1), sleep=lambda _: None,
                                     authority_reference='synthetic-recorded-repair-approval')
        # The repeated failure stops correction; its located proposal is excluded
        # so the valid candidate still reaches the same QA and publication gates.
        assert result['validation_repairs'][-1]['state'] == 'no_progress'
        assert result['state'] == 'completed' and result['qa_state'] == 'validated'
        assert len(api.repair_calls) == len(api.payloads) == len(api.inputs) == 1
        final = ledger.get(canary.DAY)
        assert final['validation_repair_outcome']['state'] == 'accepted_with_exclusions'
        assert [e['field'] for e in final['validation_repair_outcome']['excluded']] == ['proposed_knowledge_deltas']
        request = final['validation_repair_authority']['request']
        assert request['authority_reference'] == 'synthetic-recorded-repair-approval'
    finally:
        bridge.close()


def test_repair_output_restart_observes_lost_qa_reply_without_repair_or_qa_resend(fixture, monkeypatch):
    bridge, ledger, api, cfg, _failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)
        original_run, original_input = canary.run, api.qa_input
        def accepted_unacknowledged(*args):
            original_input(*args)
            raise TimeoutError("synthetic accepted QA reply lost")
        api.qa_input = accepted_unacknowledged
        def begin_only(*_args, **_kwargs):
            from tools.daily_research.consumer import Consumer
            return Consumer(ledger, cfg, api, clock=lambda: later).step()
        monkeypatch.setattr(canary, "run", begin_only)
        partial = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        assert partial["state"] == "qa_input_unresolved"
        assert ledger.get(canary.DAY)["validation_repairs"][-1]["state"] == "validated"
        monkeypatch.setattr(canary, "run", original_run)
        api.qa_input = original_input
        final = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        assert final["state"] == "completed"
        assert len(api.repair_calls) == len(api.inputs) == len(api.payloads) == 1
    finally:
        bridge.close()


def test_repair_output_restart_reconciles_publication_after_lost_receipt(fixture, monkeypatch):
    bridge, _ledger, api, _cfg, _failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)
        original_call, lost = bridge.call, [False]
        def lose_receipt(op, **fields):
            value = original_call(op, **fields)
            if op == "publish" and not lost[0]:
                lost[0] = True
                raise TimeoutError("synthetic publication accepted; receipt lost")
            return value
        monkeypatch.setattr(bridge, "call", lose_receipt)
        with pytest.raises(TimeoutError):
            canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        final = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        assert final["state"] == "completed"
        assert len(api.repair_calls) == len(api.inputs) == len(api.payloads) == 1
        assert len(json.loads(fixture[8].read_text())) == 1
        assert len(json.loads(fixture[7].read_text())["values"]) == 6
    finally:
        bridge.close()


def test_changed_feedback_gets_another_same_session_correction_and_exports_every_revision(fixture, monkeypatch):
    bridge, ledger, api, _cfg, _failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)
        good = api.repaired_raw
        incomplete = json.loads(good)
        incomplete["coverage"].pop("defined_run_scope")
        artifact = api.artifact
        api.artifact = lambda sid, aid: canonical(incomplete).encode() if aid == "artifact_turn_repair_1" else artifact(sid, aid)
        result = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        assert result["state"] == "completed"
        final = ledger.get(canary.DAY)
        assert [r["state"] for r in final["validation_repairs"]] == ["invalid", "validated"]
        assert len(api.payloads) == len(api.inputs) == 1 and len(api.repair_calls) == 2
        assert any(i["path"] == "/coverage" for i in final["validation_repairs"][0]["feedback"])
        exported = fixture[5] / "two-revision-export"
        assert not canary.render.export_snapshot(bridge, canary.DAY, exported)["missing_files"]
        assert (exported / (canary.DAY + "-repair-1-artifact.json")).read_bytes() == canonical(incomplete).encode()
        assert (exported / (canary.DAY + "-repair-2-artifact.json")).read_bytes() == good
        snapshot = bridge.call("snapshot", day=canary.DAY)
        snapshot["files"]["repair-1-artifact"] = base64.b64encode(b"corrupt old revision").decode()
        class Changed:
            def call(self, *_args, **_kwargs):
                return snapshot
        with pytest.raises(Refusal, match="validation_repair_export_binding_mismatch"):
            canary.render.export_snapshot(Changed(), canary.DAY, fixture[5] / "corrupt-old-revision-export")
    finally:
        bridge.close()


def test_late_completed_repair_is_retained_but_never_qualified(fixture, monkeypatch):
    bridge, ledger, api, _cfg, _failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)
        listing = api.listing
        def late(resource, sid=None):
            values = listing(resource, sid)
            if resource == "turns":
                for value in values:
                    if value["id"].startswith("turn_repair_"):
                        value["completed_at"] = int((later + timedelta(seconds=1801)).timestamp())
            return values
        api.listing = late
        result = canary.repair_report(bridge, fixture[5], api_factory=lambda *_: api, clock=lambda: later, sleep=lambda _: None)
        final = ledger.get(canary.DAY)
        revision = final["validation_repairs"][-1]
        assert revision["state"] == "no_progress"
        assert revision["error"] == "validation_repair_terminal_guard_failed"
        assert ledger.read_bytes(revision["artifact_file"]) == api.repaired_raw
        # The late correction is retained but never qualified: the in-window
        # original continues with only its located failure excluded.
        assert final["validation_repair_outcome"]["revision"] == 0
        assert final["packet"]["research_exclusions"]["artifact_sha256"] == final["raw_output_digest"]
        assert result["state"] == "completed" and len(api.inputs) == 1
    finally:
        bridge.close()


@pytest.mark.parametrize("after_claim", ["stop", "deadline", "origin"])
def test_repair_input_rechecks_action_time_after_durable_claim(fixture, monkeypatch, after_claim):
    from tools.daily_research.runner import digest
    bridge, ledger, _api, _cfg, failed = failed_repair_baseline(fixture, monkeypatch)
    provider = object.__new__(canary.CanaryProvider)
    later, posts = NOW + timedelta(hours=1), []
    provider.ledger, provider.clock, provider.stopped = ledger, lambda: later, lambda: False
    provider.safe = lambda _row: None  # Cost GET was separately exercised; no credentials.
    provider.api = SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=lambda *a, **kw: posts.append(a))))
    event = {"type": "agent.session.input.message", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Repair the retained report"}]}]}
    deadline = int((later + timedelta(seconds=1800)).timestamp() * 1000)
    try:
        original_call = bridge.call
        with ledger.lock():
            row = ledger.get(canary.DAY)
            row["validation_repairs"] = [{"number": 1, "state": "input_unresolved", "input_attempted": True,
                "request_digest": digest(event), "deadline_ms": deadline}]
            ledger.put(row)
            def delayed_claim(op, **fields):
                result = original_call(op, **fields)
                if op == "repair_check":
                    if after_claim == "stop":
                        provider.stopped = lambda: True
                    elif after_claim == "deadline":
                        provider.clock = lambda: later + timedelta(seconds=1801)
                    else:
                        original_call("test_origin_change")
                return result
            monkeypatch.setattr(bridge, "call", delayed_claim)
            with pytest.raises(Refusal, match="stopped_disabled_expired|guard_unreconciled_or_changed"):
                provider.repair_input(failed["session_id"], event, failed["run_key"] + ":repair:1", canary.DAY, digest(event), deadline)
        assert not posts
    finally:
        bridge.close()


def recovered_baseline(fixture, monkeypatch):
    closed_original(fixture)
    bridge, ledger, api, approval, _ = attempt_bridge(fixture, monkeypatch, 1)
    receipt, cache = fixture[3], fixture[5]
    canary.stage(bridge, canary.inspect(bridge, approval, receipt, api, cache, now=NOW), receipt)
    api.raw = canonical(null_operator_output(api)).encode()
    runner = Runner(ledger, canary.render.configured(bridge, cache), api, clock=lambda: NOW)
    failed = runner.start_or_resume()
    assert failed['error'] == 'knowledge_delta_evidence_invalid'
    recovered = Runner(ledger, runner.config, None, clock=lambda: NOW).recover_output(canary.DAY, recovery_receipt(failed))
    from tools.daily_research import discovery
    with ledger.lock():
        discovery.preserve_estimate(recovered, 'canary_model_estimate', canary.spend(api, recovered))
        ledger.put(recovered)
    return bridge, ledger, api, cache, recovered


@pytest.mark.parametrize("expired_repair", [False, True])
def test_recovered_qa_has_its_own_bounded_window_without_resetting_research(fixture, monkeypatch, expired_repair):
    bridge, ledger, api, cache, row = recovered_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)  # Original total/research window is exhausted.
        repair = None
        if expired_repair:
            repair = {"number": 1, "state": "cancel_pending", "input_attempted": True,
                "request_digest": "a" * 64, "deadline_ms": int((NOW + timedelta(seconds=1800)).timestamp() * 1000),
                "cancel_attempted": True, "cancel_reply_unresolved": True}
            row["validation_repair_authority"] = {"started_at": NOW.isoformat(), "duration_seconds": 1800,
                "request": {"scope": "same-session-validation-repair-and-qa-no-outreach",
                    "authority_reference": "Sentinel_3b6171ff167c8191b378202c5f0c54c0",
                    "budget_authority_reference": canary.BASELINE["authority_reference"],
                    "baseline_id": "baseline-20261002", "soft_total_usd": 25,
                    "session_id": row["session_id"], "root_turn_id": row["turn_id"],
                    "raw_output_sha256": row["raw_output_digest"]}}
            row["validation_repairs"] = [deepcopy(repair)]
            with ledger.lock():
                ledger.put(row)
                bridge.call("repair_check", day=canary.DAY, request_digest=repair["request_digest"], deadline_ms=repair["deadline_ms"])
            assert canary.qa_deadline(row, {}) < later
        before = {key: row.get(key) for key in ('started_at', 'research_deadline_ms', 'research_runtime_seconds', 'total_runtime_seconds', 'turn_id', 'raw_output_digest')}
        receipt = continued_qa_receipt(row)
        armed = canary.authorize_recovered_qa(bridge, receipt, clock=lambda: later)
        assert canary.qa_deadline(armed, {}) == later + timedelta(seconds=600)
        assert canary.authorize_recovered_qa(bridge, receipt, clock=lambda: later + timedelta(minutes=5)) == armed
        original_listing, original_input = api.listing, api.qa_input
        state = {'qa': 'in_progress'}
        def listing(resource, sid=None):
            values = original_listing(resource, sid)
            for turn in values if resource == 'turns' else []:
                if turn['id'] == 'turn_qa':
                    turn['status'] = state['qa']
                    turn['completed_at'] = int(later.timestamp()) + 20
            return values
        def qa_input(*args):
            original_input(*args)
            api.actions = [{'type': 'function_call', 'turn_id': 'turn_qa', 'call_id': 'recovered_qa_search',
                            'name': canary.search.SEARCH, 'arguments': {'query': 'Synthetic recovered QA source'}}]
        def tick(_):
            assert not api.cancellations
            if api.result_events:
                state['qa'], api.actions = 'completed', []
        api.listing, api.qa_input = listing, qa_input
        result = canary.run(bridge, cache, recovery_only=True, api_factory=lambda *_: api, clock=lambda: later, sleep=tick)
        assert len(api.executions) == len(api.result_events) == 1 and not api.cancellations
        assert result['state'] == 'completed' and len(api.payloads) == len(api.inputs) == 1
        final = ledger.get(canary.DAY)
        assert {key: final.get(key) for key in before} == before
        assert final['qa']['deadline_ms'] == int((later + timedelta(seconds=600)).timestamp() * 1000)
        assert final['delivery']['notion']['key'].endswith('baseline-20261002-attempt-0001:notion')
        assert final['qa_continuation']['model_observation'] == armed['qa_continuation']['model_observation']
        if repair:
            assert final["validation_repairs"] == [repair]
            assert final["validation_repair_authority"] == row["validation_repair_authority"]
            with ledger.lock(), pytest.raises(Refusal, match="not_admitted"):
                bridge.call("repair_check", day=canary.DAY, request_digest=repair["request_digest"], deadline_ms=repair["deadline_ms"])
    finally:
        bridge.close()


@pytest.mark.parametrize("pending", ["session", "connection", "turn"])
def test_recovered_qa_refuses_concurrent_or_waiting_work(fixture, monkeypatch, pending):
    bridge, ledger, api, cache, row = recovered_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)
        canary.authorize_recovered_qa(bridge, continued_qa_receipt(row), clock=lambda: later)
        if pending == "session":
            api.session_status = "in_progress"
        elif pending == "connection":
            api.actions = [{"type": "environment_connection"}]
        else:
            listing = api.listing
            def with_correction(resource, sid=None):
                values = listing(resource, sid)
                if resource == "turns":
                    values.append({"id": "turn_unexpected_correction", "status": "in_progress", "subagent_id": None})
                return values
            api.listing = with_correction
        from tools.daily_research.consumer import Consumer
        consumer = Consumer(ledger, canary.render.configured(bridge, cache), api, clock=lambda: later)
        consumer.active_day = canary.DAY
        with pytest.raises(Refusal, match="session_not_idle|turn_scope_mismatch"):
            consumer.step()
        assert not api.inputs and not ledger.get(canary.DAY).get("qa")
        assert len(api.payloads) == 1
    finally:
        bridge.close()


def test_recovered_qa_cannot_renew_expired_window_or_use_wrong_authority(fixture, monkeypatch):
    bridge, ledger, api, cache, row = recovered_baseline(fixture, monkeypatch)
    try:
        later = NOW + timedelta(hours=1)
        receipt = continued_qa_receipt(row)
        with pytest.raises(Refusal, match='authority_or_binding_invalid'):
            canary.authorize_recovered_qa(bridge, {**receipt, 'authority_reference': 'unapproved'}, clock=lambda: later)
        assert not ledger.get(canary.DAY).get('qa_continuation')
        canary.authorize_recovered_qa(bridge, receipt, clock=lambda: later)
        result = canary.run(bridge, cache, recovery_only=True, api_factory=lambda *_: api,
                            clock=lambda: later + timedelta(seconds=601))
        assert result['observer_error'] == 'canary_total_observation_deadline'
        assert not api.inputs and len(api.payloads) == 1
        assert ledger.get(canary.DAY)['started_at'] == row['started_at']
    finally:
        bridge.close()


@pytest.mark.parametrize("after_claim", ["session", "connection", "turn", "stop", "deadline", "origin"])
def test_recovered_qa_rechecks_session_after_its_durable_claim(fixture, monkeypatch, after_claim):
    from tools.daily_research.consumer import Consumer
    bridge, ledger, api, cache, row = recovered_baseline(fixture, monkeypatch)
    later, posts = NOW + timedelta(hours=1), []
    provider = object.__new__(canary.CanaryProvider)
    provider.ledger, provider.clock, provider.stopped = ledger, lambda: later, lambda: False
    provider.safe = lambda _row: None
    provider.get, provider.listing = api.get, api.listing
    provider.api = SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=lambda *a, **kw: posts.append(a))))
    try:
        canary.authorize_recovered_qa(bridge, continued_qa_receipt(row), clock=lambda: later)
        original_call, original_listing = bridge.call, api.listing
        def delayed_claim(op, **fields):
            result = original_call(op, **fields)
            if op == "qa_check":
                if after_claim == "session":
                    api.session_status = "in_progress"
                elif after_claim == "connection":
                    api.actions = [{"type": "environment_connection"}]
                elif after_claim == "turn":
                    def late_turn(resource, sid=None):
                        values = original_listing(resource, sid)
                        if resource == "turns":
                            values.append({"id": "turn_late_correction", "status": "in_progress", "subagent_id": None})
                        return values
                    provider.listing = api.listing = late_turn
                else:
                    def changed_during_inventory(resource, sid=None):
                        values = original_listing(resource, sid)
                        if resource == "turns":
                            if after_claim == "stop":
                                provider.stopped = lambda: True
                            elif after_claim == "origin":
                                original_call("test_origin_change")
                            else:
                                provider.clock = lambda: later + timedelta(seconds=601)
                        return values
                    provider.listing = changed_during_inventory
            return result
        monkeypatch.setattr(bridge, "call", delayed_claim)
        api.qa_input = provider.qa_input
        consumer = Consumer(ledger, canary.render.configured(bridge, cache), api, clock=lambda: later)
        consumer.active_day = canary.DAY
        result = consumer.step()
        final = ledger.get(canary.DAY)
        assert result["state"] == "qa_input_unresolved" and not posts
        assert final["qa"]["input_error_receipt"]["class"] == "Refusal"
        assert final["qa"]["input_error_receipt"]["code"] in {
            "recovered_qa_session_or_turn_changed", "recovered_qa_stopped_disabled_or_expired",
            "canary_daily_guard_unreconciled_or_changed"}
        assert final["qa"]["input_error_receipt"]["stage"] == "preconditions"
        with ledger.lock(), pytest.raises(Refusal, match="agent_qa_input_not_admitted|canary_daily_guard_unreconciled_or_changed"):
            original_call("qa_check", day=canary.DAY, request_digest=final["qa"]["request_digest"],
                          deadline_ms=final["qa"]["deadline_ms"])
        assert final["raw_output_digest"] == row["raw_output_digest"] and len(api.payloads) == 1
    finally:
        bridge.close()


def test_one_command_recovers_original_without_resending_expired_correction(fixture, monkeypatch):
    bridge, ledger, api, _cfg, failed = failed_repair_baseline(fixture, monkeypatch)
    later = NOW + timedelta(hours=1)
    repair = {"number": 1, "state": "cancel_pending", "input_attempted": True,
              "request_digest": "a" * 64, "deadline_ms": int((NOW + timedelta(seconds=1800)).timestamp() * 1000),
              "cancel_attempted": True, "cancel_reply_unresolved": True}
    try:
        original_raw = ledger.read_bytes(canary.DAY + "-artifact.json")
        with ledger.lock():
            failed["validation_repairs"] = [deepcopy(repair)]
            failed["validation_repair_authority"] = {"started_at": NOW.isoformat(), "duration_seconds": 1800,
                "request": {"scope": "same-session-validation-repair-and-qa-no-outreach",
                    "authority_reference": "Sentinel_3b6171ff167c8191b378202c5f0c54c0",
                    "budget_authority_reference": canary.BASELINE["authority_reference"],
                    "baseline_id": "baseline-20261002", "soft_total_usd": 25,
                    "session_id": failed["session_id"], "root_turn_id": failed["turn_id"],
                    "raw_output_sha256": failed["raw_output_digest"]}}
            ledger.put(failed)
            bridge.call("repair_check", day=canary.DAY, request_digest=repair["request_digest"], deadline_ms=repair["deadline_ms"])
        result = canary.recover_original_and_qa(bridge, fixture[5], api_factory=lambda *_: api,
                                               clock=lambda: later, sleep=lambda _: None)
        assert result["state"] == "completed", result
        final = ledger.get(canary.DAY)
        assert final["qa_continuation"]["started_at"] == later.isoformat()
        assert final["qa"]["deadline_ms"] == int((later + timedelta(seconds=600)).timestamp() * 1000)
        assert final["validation_repairs"] == [repair]
        assert final["validation_repair_authority"] == failed["validation_repair_authority"]
        assert final["started_at"] == failed["started_at"] and final["turn_id"] == failed["turn_id"]
        assert ledger.read_bytes(canary.DAY + "-artifact.json") == original_raw
        assert not api.repair_calls and len(api.payloads) == len(api.inputs) == 1
        assert all(delivery["receipt"]["readback_verified"] for delivery in final["delivery"].values())
        replay = canary.recover_original_and_qa(bridge, fixture[5], api_factory=lambda *_: api,
                                               clock=lambda: later + timedelta(minutes=20), sleep=lambda _: None)
        assert replay["state"] == "completed" and len(api.inputs) == 1 and not api.repair_calls
        assert ledger.get(canary.DAY)["qa_continuation"] == final["qa_continuation"]
    finally:
        bridge.close()


@pytest.mark.parametrize("pending", ["connection", "turn", "stop"])
def test_one_command_checks_for_late_work_before_recovery_writes(fixture, monkeypatch, pending):
    bridge, ledger, api, _cfg, failed = failed_repair_baseline(fixture, monkeypatch)
    try:
        if pending == "connection":
            api.actions = [{"type": "environment_connection"}]
        elif pending == "turn":
            listing = api.listing
            def late(resource, sid=None):
                values = listing(resource, sid)
                if resource == "turns":
                    values.append({"id": "turn_late_correction", "status": "in_progress", "subagent_id": None})
                return values
            api.listing = late
        with pytest.raises(Refusal, match="recovered_qa_session_or_turn_changed|recovered_qa_state_not_admitted"):
            canary.recover_original_and_qa(bridge, fixture[5], api_factory=lambda *_: api,
                clock=lambda: NOW + timedelta(hours=1), stopped=lambda: pending == "stop", sleep=lambda _: None)
        assert ledger.get(canary.DAY) == failed
        assert not api.inputs and not api.repair_calls and len(api.payloads) == 1
    finally:
        bridge.close()


def test_saved_artifact_diagnosis_checks_actual_bindings_without_store_writes(fixture):
    bridge, ledger, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    api.raw = canonical(null_operator_output(api)).encode()
    cfg = canary.render.configured(bridge, cache)
    row = Runner(ledger, cfg, api, clock=lambda: NOW).start_or_resume()
    before = canary.digest(row)
    result = canary.diagnose_saved_output(ledger, cfg, now=NOW)
    assert result['original_validation'] == {'valid': False, 'error': 'knowledge_delta_evidence_invalid'}
    assert result['quarantined_derivation_validation']['valid'] is True
    assert result['knowledge_context_attached'] is True and result['crm_snapshot_attached_to_research'] is False
    assert result['store_writes'] == result['provider_mutations'] == 0
    assert canary.digest(ledger.get(canary.DAY)) == before
    assert 'candidate_count' in result and not result['newness_verified']


def test_recovery_receipt_is_immutable_even_when_crash_precedes_row_pointer(fixture, monkeypatch):
    bridge, ledger, api, receipt, plan, cache, _, _, _ = fixture
    canary.stage(bridge, plan, receipt)
    api.raw = canonical(null_operator_output(api)).encode()
    cfg = canary.render.configured(bridge, cache)
    row = Runner(ledger, cfg, api, clock=lambda: NOW).start_or_resume()
    original_put = ledger.put
    def crash(_):
        raise RuntimeError('synthetic crash after derivation file commit')
    monkeypatch.setattr(ledger, 'put', crash)
    runner = Runner(ledger, cfg, None, clock=lambda: NOW)
    with pytest.raises(RuntimeError, match='synthetic crash'):
        runner.recover_output(canary.DAY, recovery_receipt(row))
    raw = ledger.read_bytes(canary.DAY + '-recovery.json')
    assert not ledger.get(canary.DAY).get('output_recovery')
    with ledger.lock(), pytest.raises(Refusal, match='artifact_identity_conflict'):
        ledger.write_bytes(canary.DAY + '-recovery.json', b'{}')
    assert ledger.read_bytes(canary.DAY + '-recovery.json') == raw
    monkeypatch.setattr(ledger, 'put', original_put)
    assert runner.recover_output(canary.DAY, recovery_receipt(row))['state'] == 'awaiting_review'


@pytest.mark.parametrize("change", ["disabled", "authority", "budget"])
def test_qa_correction_origin_guard_rechecks_control_before_provider_post(monkeypatch, change):
    row = {"packet": {}, "packet_digest": digest({}), "search_provider": "perplexity-fast-v1",
           "recurring_budget_authority_reference": "approved", "soft_target_usd": 5,
           "qa": {"state": "qa_correction_input_unresolved", "submission_binding": {"authority_reference": "qa-approved"}, "corrections": [
               {"state": "input_unresolved", "authority_reference": "qa-approved"}]}}
    control = {"enabled": True, "workflow": {"enabled": True, "qa_authority_reference": "qa-approved"},
               "config": {"search_provider": "perplexity-fast-v1", "recurring_budget_authority_reference": "approved", "soft_target_usd": 5}}
    class Bridge:
        def call(self, op, **_):
            if op == "guard":
                if change == "disabled":
                    control["enabled"] = False
                elif change == "authority":
                    control["workflow"]["qa_authority_reference"] = "changed"
                else:
                    control["config"]["soft_target_usd"] = 7
            return control if op == "control" else True
    provider = canary.CanaryProvider.__new__(canary.CanaryProvider)
    provider.ledger = SimpleNamespace(bridge=Bridge(), get=lambda _: row)
    provider.safe, provider.clock = lambda *_: None, lambda: NOW
    # Generic inventory and final fence have separate real-provider race coverage.
    monkeypatch.setattr(canary.FencedProvider, "qa_correction_action_guard", lambda *_: None)
    with pytest.raises(Refusal, match="authority_changed|budget_authority_changed"):
        provider.qa_correction_action_guard(canary.DAY, int((NOW + timedelta(seconds=60)).timestamp() * 1000))
