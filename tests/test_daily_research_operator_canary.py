"""Hermetic canary claims, namespace/restart, terminal QA and publication."""
import base64
import hashlib
import importlib.util
import json
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
from tools.daily_research.runner import Refusal, canonical

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
        "const notion=async(method,path,body)=>{const p=existsSync(pagesPath)?JSON.parse(readFileSync(pagesPath,'utf8')):[];if(method==='POST'){p.push(body);writeFileSync(pagesPath,JSON.stringify(p));return {id:'synthetic-page'};}",
        "if(path==='/pages/3eb80154161d8116858ed5f376b4b7a9')return {object:'page',id:'3eb80154161d8116858ed5f376b4b7a9'};if(path.startsWith('/blocks/3eb80154161d8116858ed5f376b4b7a9/'))return {has_more:false,results:p.map(x=>({id:'synthetic-page',type:'child_page',child_page:{title:x.properties.title.title[0].text.content}}))};if(path==='/pages/synthetic-page')return {parent:{page_id:'3eb80154161d8116858ed5f376b4b7a9'}};return {has_more:false,results:p[0]?.children||[]};};",
        "const channel=new CanaryChannel(db,crmReader,new Publisher({crmReader,google,notion}),()=>" + str(int(NOW.timestamp()*1000)) + ");",
        "for await(const line of createInterface({input:process.stdin})){try{const r=JSON.parse(line);",
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


def test_unknown_usage_is_not_zero_or_admitted():
    api = SimpleNamespace(listing=lambda *_: [{"usage": None}])
    assert canary.spend(api, {"session_id": "sess-test"}) == {"known": False, "estimate_usd": None, "hard_total_cap": False}


def test_test_approval_is_not_recurring_or_delete_approval():
    assert canary.admission(APPROVAL, NOW)["scope"] == canary.SCOPE
    with pytest.raises(Refusal, match="admission"):
        canary.admission({**APPROVAL, "scope": "delete-session"}, NOW)
    with pytest.raises(Refusal, match="admission"):
        canary.admission({**APPROVAL, "ceiling_usd": 26}, NOW)
