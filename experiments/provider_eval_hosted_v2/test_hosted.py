"""Hermetic real hosted-session flow with wire-shaped API and HTTP mocks."""

import base64
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentTask, AgentExecutionError, digest as task_digest
from blueprint_pipeline.agent_execution.journal import AgentJournal
from blueprint_pipeline.agent_execution.operations import AgentOperations, OperationPending, ToolRefused
from experiments.provider_eval_recovery.harness import Ledger, digest, write_once
from experiments.provider_eval_recovery.live_http import MODEL, PROJECT
from experiments.provider_eval_recovery.public_inputs import load_public
from tests.test_agent_execution_sessions import FakeAPI, make_task
from .evidence import Evidence, ResearchTools, search_request
from .hosted import HostedRuntime, INSTRUCTIONS, OUTPUT_SCHEMA, live_admission, planning_cost, preflight
from .routes import ExistingSearchRoute, public_fetch

PUBLIC = Path(__file__).parents[1] / "provider_eval_recovery/real_public/inputs.parent-message.json"


def fixture(tmp_path, *, authorize=lambda _: None):
    public = load_public(PUBLIC)[0]
    evidence = Evidence(tmp_path / "sources", public["cases"][0], "parallel_fast")
    evidence.ingest({"results": [{"url": "https://example.com/vendor", "title": "Public",
        "excerpts": ["X" * 20000 + "Late utilities evidence: 120V and compressed air."]}]}, {"fixture": True})
    tools = ResearchTools(evidence, search=lambda *_: {"results": [{"url": "https://example.com/followup", "excerpts": ["Full followup evidence"]}]},
                          fetch=lambda req, _: {"url": req["url"], "text": "Y" * 25000 + "Late deployment process.",
                                                "retrieved_at": "2026-10-01", "body_sha256": "a" * 64})
    api = FakeAPI()
    api.project_id = PROJECT
    original = api.request
    def request(method, path, **kwargs):
        result = original(method, path, **kwargs)
        if path == "/agents/sessions" and method == "POST":
            api.session["environment"] = {"type": "openai_hosted", "container_size": "small", "id": "env_1",
                                          "network": {"access": "disabled"}}
            return dict(api.session)
        return result
    api.request = request
    api.output = {"answer": "Unknown site fit.", "citations": ["https://example.com/vendor"], "unknowns": ["site qualification"]}
    journal = AgentJournal(tmp_path / "agent-journal")
    ops = AgentOperations(journal, tools.tools(), authorize=lambda *_: None, clock=lambda: 1000)
    runtime = HostedRuntime(evidence=evidence, common_prompt=public["common_prompt"], transport=api,
        project_id=PROJECT, journal=journal, operations=ops, validate_admission=authorize, clock=lambda: 1000)
    original_task = make_task(tools.tools())
    values = original_task.model_dump()
    values.update(model=MODEL, instructions=INSTRUCTIONS, reasoning_effort="low", output_schema=OUTPUT_SCHEMA,
                  context_revision=task_digest({"case": evidence.case, "mode": evidence.mode, "files": runtime.files()}))
    values["admission"]["project_id"] = PROJECT
    # This authority/guard is explicitly synthetic and never used for paid work.
    task = AgentTask.model_validate(values)
    return runtime, api, task, evidence, tools


def test_real_hosted_payload_full_late_evidence_and_no_secrets_or_hidden_search(tmp_path):
    runtime, api, task, evidence, tools = fixture(tmp_path)
    runtime.start(task)
    payload = api.calls[0][2]
    assert payload["environment"]["type"] == "openai_hosted"
    assert payload["environment"]["network"] == {"access": "disabled"}
    assert payload["agent"]["model"] == MODEL
    assert payload["agent"]["multi_agent"] == {"enabled": False}
    assert {t["type"] for t in payload["agent"]["tools"]} == {"function"}
    files = {f["path"]: base64.b64decode(f["data"]).decode() for f in payload["environment"]["files"]}
    assert any("Late utilities evidence" in content for content in files.values())
    assert not any(key in payload for key in ("vault_ids", "agent_id"))
    source = evidence.sources()[0]
    read = evidence.read({"source_id": source["id"], "start": 19995, "length": 100})
    assert "Late utilities evidence" in read["text"] and read["next_start"] is None
    assert evidence.find({"source_id": source["id"], "term": "utilities", "start": 0})["offsets"][0] > 20000


def test_durable_tool_search_restart_and_final_turn_status(tmp_path):
    runtime, api, task, evidence, tools = fixture(tmp_path)
    runtime.start(task)
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "call_search",
                    "name": "search", "arguments": {"query": "Chef Robotics utilities commercial deployment"}}]
    api.reply_uncertain = True
    with pytest.raises(Exception):
        runtime.step(task.task_id)
    original_search = tools.search_route
    def never_search(*_):
        raise AssertionError("duplicate provider call")
    tools.search_route = never_search
    # Same pending call is reconciled from the saved result after disconnect.
    runtime.step(task.task_id)
    tools.search_route = original_search
    assert len(list((evidence.root / "operations/search").glob("*/intent.json"))) == 1
    state = runtime.step(task.task_id)
    assert state["state"] == "completed"
    assert state["result"]["session_id"] == "session_1" and state["result"]["turn_id"] == "turn_1"
    assert state["result"]["scientific_acceptance_granted"] is False
    assert state["result"]["cost_status"] == "official_reconciliation_required"
    # New tool evidence cannot invalidate the immutable initial hosted file set.
    restarted = HostedRuntime(evidence=evidence, common_prompt=runtime.common_prompt, transport=api, project_id=PROJECT,
        journal=runtime.journal, operations=runtime.operations, validate_admission=lambda _: None, clock=lambda: 1000)
    assert restarted.files() == runtime.files()
    assert restarted.step(task.task_id)["state"] == "completed"


def test_default_live_gate_refuses_before_any_session_call(tmp_path):
    runtime, api, task, _, _ = fixture(tmp_path, authorize=live_admission)
    with pytest.raises(AgentExecutionError, match="hard_spend_bound"):
        runtime.start(task)
    assert api.calls == []
    with pytest.raises(AgentExecutionError, match="permanent_deletion_not_authorized"):
        runtime.cleanup(task.task_id)
    assert api.calls == []


def test_model_project_file_binding_and_environment_fail_closed(tmp_path):
    runtime, api, task, _, _ = fixture(tmp_path)
    changed = task.model_dump()
    changed["model"] = "gpt-6-astra"
    with pytest.raises(AgentExecutionError, match="sol_and"):
        runtime.start(AgentTask.model_validate(changed))
    assert api.calls == []
    runtime.start(task)
    api.session["environment"]["type"] = "none"
    with pytest.raises(AgentExecutionError, match="isolated_small_hosted"):
        runtime.step(task.task_id)


@pytest.mark.parametrize("policy", [None, {"access": "enabled"}, {"access": "restricted"}])
def test_hosted_returned_network_policy_must_be_explicitly_disabled(tmp_path, policy):
    runtime, api, task, _, _ = fixture(tmp_path)
    runtime.start(task)
    api.session["environment"]["network"] = policy
    with pytest.raises(AgentExecutionError, match="isolated_small_hosted"):
        runtime.step(task.task_id)


def test_same_queries_preserve_ordinary_ampersand_product_names():
    public = load_public(PUBLIC)[0]
    for mode in ("parallel_fast", "parallel_advanced", "perplexity_fast", "perplexity_standard"):
        request = search_request(mode, public["cases"][5], "GrayMatter Scan&Sand utilities deployment")
        assert "Scan&Sand" in json.dumps(request)
        assert public["cases"][5]["question"] in json.dumps(request)
    with pytest.raises(ToolRefused):
        search_request("parallel_fast", public["cases"][5], "query\ncontrol")


def test_evidence_isolation_quarantine_and_uncertain_search_no_retry(tmp_path):
    _, _, _, evidence, tools = fixture(tmp_path)
    other = Evidence(tmp_path / "sources", evidence.case, "perplexity_fast")
    assert other.sources() == []
    with pytest.raises(FileNotFoundError):
        other.source(evidence.sources()[0]["id"])
    evidence.ingest({"results": [{"url": "https://user:secret@example.com/private?token=secret", "excerpts": ["UNSAFE"]}]}, {"fixture": True})
    assert "UNSAFE" not in json.dumps(evidence.sources())
    attempts = []
    def uncertain(*_):
        attempts.append(1)
        raise TimeoutError("PRIVATE_EXCEPTION")
    tools.search_route = uncertain
    ctx = SimpleNamespace(operation_id="uncertain")
    with pytest.raises(TimeoutError):
        tools.operation("search", {"query": "Chef Robotics deployment"}, ctx)
    with pytest.raises(OperationPending, match="uncertain"):
        tools.operation("search", {"query": "Chef Robotics deployment"}, ctx)
    assert attempts == [1]
    assert tools.reconcile("search", {"query": "Chef Robotics deployment"}, ctx).status == "pending"


def test_exact_used_budget_checkpoint_and_conditional_pilot_plan(tmp_path):
    Ledger(tmp_path / "live_journal.jsonl", "10.00").append("reserved", "prior", amount_usd="3.934105")
    before = (tmp_path / "live_journal.jsonl").read_bytes()
    result = preflight(tmp_path)
    assert result["remaining_original_cap_usd"] == "6.065895"
    assert result["pilot_target_within_remaining_reservation"] is True
    assert result["status"] == "blocked_no_network" and result["new_paid_calls"] == 0
    assert planning_cost(1)["incremental_planning_target_usd"] == "0.468980"
    assert planning_cost(2)["incremental_planning_target_usd"] == "0.937960"
    assert planning_cost(20)["incremental_planning_target_usd"] == "9.379600"
    assert Decimal("3.934105") + Decimal(planning_cost(20)["incremental_planning_target_usd"]) == Decimal("13.313705")
    assert (tmp_path / "live_journal.jsonl").read_bytes() == before


class FetchResponse:
    def __init__(self, status=200, content_type="text/html", body=b"<p>Public</p><p>Late evidence</p>"):
        self.status, self.kind, self.body = status, content_type, body
    def getheader(self, *_): return self.kind
    def read(self, limit): return self.body[:limit]


class FetchConnection:
    def __init__(self, response): self.response, self.requests = response, []
    def request(self, *args, **kwargs): self.requests.append((args, kwargs))
    def getresponse(self): return self.response
    def close(self): pass


def test_public_source_fetch_pins_dns_retains_whole_text_and_refuses_redirects():
    called = []
    def resolve(host, port, **_):
        return [(None, None, None, None, ("93.184.216.34", port))]
    connection = FetchConnection(FetchResponse())
    def connect(parsed, ip):
        called.append((parsed.hostname, ip))
        return connection
    result = public_fetch({"method": "GET", "url": "https://example.com/public"}, None,
        authorize=lambda _: None, resolve=resolve, connect=connect)
    assert "Late evidence" in result["text"] and result["complete_body"] is True
    assert called == [("example.com", "93.184.216.34")]
    assert "Authorization" not in connection.requests[0][1]["headers"]
    connection.response.status = 302
    with pytest.raises(ToolRefused, match="redirect"):
        public_fetch({"method": "GET", "url": "https://example.com/public"}, None,
                     authorize=lambda _: None, resolve=resolve, connect=connect)
    with pytest.raises(ToolRefused, match="nonpublic"):
        public_fetch({"method": "GET", "url": "http://localhost/public"}, None,
            authorize=lambda _: None, resolve=lambda *_args, **_kwargs: [(None,None,None,None,("127.0.0.1",80))], connect=connect)


def test_full_source_fetch_and_equal_opportunity_stop(tmp_path):
    _, _, _, evidence, tools = fixture(tmp_path)
    sid = evidence.sources()[0]["id"]
    out = tools.operation("fetch", {"source_id": sid}, SimpleNamespace(operation_id="fetch_1"))
    late = evidence.source(out["new_source_ids"][0])
    assert late["text"].endswith("Late deployment process.") and len(late["text"]) > 25000
    for number in range(3):
        tools.operation("search", {"query": "Chef Robotics deployment " + str(number)}, SimpleNamespace(operation_id="search_" + str(number)))
    with pytest.raises(ToolRefused, match="budget_stop"):
        tools.operation("search", {"query": "Chef Robotics fourth search"}, SimpleNamespace(operation_id="search_4"))


def test_existing_routes_default_gate_blocks_without_key_read_or_ledger(tmp_path):
    route = ExistingSearchRoute(tmp_path)
    public = load_public(PUBLIC)[0]
    with pytest.raises(AgentExecutionError, match="hard_spend_bound"):
        route("parallel_fast", search_request("parallel_fast", public["cases"][0], "Chef Robotics deployment"), SimpleNamespace(operation_id="call"))
    assert not (tmp_path / "live_journal.jsonl").exists()


def test_valid_retained_search_reuse_preserves_root_and_cannot_pool_arms(tmp_path):
    public = load_public(PUBLIC)[0]
    evidence = Evidence(tmp_path / "protocols", public["cases"][0], "parallel_fast")
    request = search_request("parallel_fast", evidence.case, "Chef Robotics portioning deployment")
    raw = {"results": [{"url": "https://example.com/primary", "excerpts": ["X" * 25000 + "Tail availability"]}], "warnings": None}
    envelope = {"request": request, "raw": raw}
    key = digest(envelope)
    write_once(tmp_path / "protocols/bounded_adaptive_v1/raw" / (key + ".json"), envelope)
    ledger = Ledger(tmp_path / "live_journal.jsonl", "10.00")
    ledger.append("reserved", key, amount_usd="0.004125", protocol="bounded_adaptive_v1", cell="01_parallel_fast",
                  step="search1", request_sha256=digest(request))
    ledger.append("completed", key, raw_sha256=digest(envelope))
    before = (tmp_path / "live_journal.jsonl").read_bytes()
    for _ in range(2):
        evidence.reuse(tmp_path)
    assert evidence.reused_count() == 1
    assert evidence.sources()[0]["text"].endswith("Tail availability")
    assert (tmp_path / "live_journal.jsonl").read_bytes() == before
    other = Evidence(tmp_path / "protocols", evidence.case, "perplexity_fast")
    assert other.reuse(tmp_path) == []


def test_actual_provider_http_route_shares_old_ledger_and_retained_calls(tmp_path):
    from io import BytesIO
    from unittest.mock import patch
    from blueprint_pipeline.paid_resource_admission import (PAID_LANE_ADMISSION_SCHEMA_VERSION, require_paid_resource_admission)
    class Response(BytesIO):
        status = 200
    class Opener:
        def __init__(self): self.calls = []
        def open(self, request, **kwargs):
            self.calls.append(request)
            return Response(b'{"results": [], "warnings": null}')
    ctx = SimpleNamespace(operation_id="paid_mock", authority_digest="fixture_binding")
    def authority(_):
        return require_paid_resource_admission({"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
            "status": "admitted", "resource_class": "evaluator_api", "blockers": [],
            "allocation_binding_digest": ctx.authority_digest}, resource_class="evaluator_api",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    opener = Opener()
    original, replacement, alias = tmp_path / "original", tmp_path / "replacement", tmp_path / "alias"
    original.mkdir()
    replacement.mkdir()
    alias.symlink_to(original, target_is_directory=True)
    route = ExistingSearchRoute(alias, authorize=authority, opener=opener)
    Ledger(original / "live_journal.jsonl", "10.00").append("reserved", "prior", amount_usd="3.934105")
    request = search_request("parallel_fast", load_public(PUBLIC)[0]["cases"][0], "Chef Robotics utilities")
    with patch(ExistingSearchRoute.__module__ + ".existing_key", return_value="MOCK_ONLY"):
        route("parallel_fast", request, ctx)
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        route("parallel_fast", request, ctx)
    assert len(opener.calls) == 1 and opener.calls[0].get_header("X-api-key") == "MOCK_ONLY"
    assert Ledger(original / "live_journal.jsonl", "10.00").exposure == Decimal("3.938230")
    assert not (replacement / "live_journal.jsonl").exists()
    with pytest.raises(ToolRefused, match="isolation"):
        route("perplexity_fast", request, ctx)
    evidence = Evidence(tmp_path / "arm", load_public(PUBLIC)[0]["cases"][0], "parallel_fast")
    tools = ResearchTools(evidence, search=route, fetch=lambda *_: None)
    args = {"query": "Chef Robotics utilities"}
    # Completed search adoption uses only the original retained envelope.
    with patch(ExistingSearchRoute.__module__ + ".existing_key", side_effect=AssertionError("key lookup on replay")):
        tools.operation("search", args, ctx)
        assert tools.operation("search", args, ctx)["new_source_ids"] == []
    journal_before = (original / "live_journal.jsonl").read_bytes()
    folder = evidence.root / "operations/search/paid_mock"
    forged = {"results": [{"url": "https://example.com/forged", "title": "FORGED_TITLE", "excerpts": ["FORGED_TEXT"]}], "warnings": None}
    ids = evidence.ingest(forged, {"kind": "hosted_agent_search", "operation_id": ctx.operation_id})
    (folder / "raw.json").write_text(json.dumps(forged))
    saved = json.loads((folder / "result.json").read_text())
    saved.update(raw_sha256=digest(forged), output={"new_source_ids": ids, "sources": evidence.manifest()})
    (folder / "result.json").write_text(json.dumps(saved))
    with pytest.raises(ToolRefused, match="paid_response_anchor_integrity"):
        tools.operation("search", args, ctx)
    assert tools.reconcile("search", args, ctx).status == "pending"
    assert (original / "live_journal.jsonl").read_bytes() == journal_before
    assert len(opener.calls) == 1


def test_artificial_admission_never_claims_documented_hard_cap(tmp_path):
    from .hosted import prepare_task
    from blueprint_pipeline.agent_execution.contracts import AgentAdmission
    runtime, _, task, _, _ = fixture(tmp_path)
    inputs = [{"role": "user", "content": [{"type": "input_text", "text":
        "Research this public case using the complete evidence files and registered tools: "
        + json.dumps(runtime.evidence.case, sort_keys=True)}]}]
    admitted = task.admission.model_dump()
    admitted["allowed_input_digests"] = (task_digest(inputs), task_digest(runtime.files()))
    prepared = prepare_task(runtime, AgentAdmission.model_validate(admitted), task_id="hosted_mock",
                            source_commit="a" * 40, deadline=1800)
    assert prepared.model == MODEL and prepared.max_input_tokens == 12000
    assert prepared.admission.budget_policy == "project_guard_accepted_uncertainty"
    with pytest.raises(AgentExecutionError, match="hard_spend_bound"):
        live_admission(prepared)


def test_source_tampering_cannot_enter_manifest_or_initial_hosted_files(tmp_path):
    _, _, _, evidence, _ = fixture(tmp_path)
    source = evidence.sources()[0]
    path = evidence.root / "sources" / (source["id"] + ".json")
    changed = {**source, "text": "tampered text"}
    path.write_text(json.dumps(changed))
    with pytest.raises(ToolRefused, match="integrity"):
        evidence.manifest()
    with pytest.raises(ToolRefused, match="integrity"):
        evidence.sources()
    path.write_text(json.dumps({**source, "id": "0" * 64}))
    with pytest.raises(ToolRefused, match="integrity"):
        evidence.sources()


def test_uncertain_provider_dispatch_remains_unsettled_in_actual_durable_runtime(tmp_path):
    from unittest.mock import patch
    from blueprint_pipeline.paid_resource_admission import (PAID_LANE_ADMISSION_SCHEMA_VERSION, require_paid_resource_admission)
    runtime, api, task, evidence, tools = fixture(tmp_path)
    class UncertainHTTP:
        calls = 0
        def open(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError("UNSAFE_UPSTREAM_PROSE")
    http = UncertainHTTP()
    def authorize(ctx):
        return require_paid_resource_admission({"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
            "status": "admitted", "resource_class": "evaluator_api", "blockers": [],
            "allocation_binding_digest": ctx.authority_digest}, resource_class="evaluator_api",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    route = ExistingSearchRoute(tmp_path, authorize=authorize, opener=http)
    tools.search_route = route
    runtime.start(task)
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "call_uncertain",
                    "name": "search", "arguments": {"query": "Chef Robotics utilities deployment"}}]
    with patch(ExistingSearchRoute.__module__ + ".existing_key", return_value="MOCK_ONLY"):
        assert runtime.step(task.task_id)["state"] == "reconciling"
        assert runtime.step(task.task_id)["state"] == "reconciling"
    assert http.calls == 1
    assert runtime.journal.unsettled_operations(task.task_id)
    assert Ledger(tmp_path / "live_journal.jsonl", "10.00").exposure == Decimal("0.004125")
    assert api.tool_results == []


@pytest.mark.parametrize("changed", ["manifest", "new_ids", "raw", "intent"])
def test_cached_replay_validates_source_manifest_and_raw_request_linkage(tmp_path, changed):
    _, _, _, evidence, tools = fixture(tmp_path)
    calls = []
    original = tools.search_route
    def route(*args):
        calls.append(1)
        return original(*args)
    tools.search_route = route
    args, ctx = {"query": "Chef Robotics deployment"}, SimpleNamespace(operation_id="replay_test")
    expected = tools.operation("search", args, ctx)
    assert tools.operation("search", args, ctx) == expected
    assert tools.reconcile("search", args, ctx).status == "completed"
    folder = evidence.root / "operations/search/replay_test"
    target = folder / ((changed if changed in {"raw", "intent"} else "result") + ".json")
    saved = json.loads(target.read_text())
    if changed == "manifest":
        saved["output"]["sources"][0]["title"] = "FORGED_EVIDENCE_TITLE"
    elif changed == "new_ids":
        saved["output"]["new_source_ids"] = []
    elif changed == "raw":
        saved["results"][0]["excerpts"] = ["FORGED_RAW_TEXT"]
    else:
        saved["request"]["body"]["search_queries"] = ["Different company"]
    target.write_text(json.dumps(saved))
    with pytest.raises(ToolRefused, match="integrity"):
        tools.operation("search", args, ctx)
    assert tools.reconcile("search", args, ctx).status == "pending"
    assert calls == [1]


def test_accepted_search_validation_warning_is_not_pre_dispatch_refusal(tmp_path):
    _, _, _, evidence, tools = fixture(tmp_path)
    calls = []
    def route(*_):
        calls.append(1)
        return {"results": [], "warnings": ["Provider input truncated"]}
    tools.search_route = route
    args, ctx = {"query": "Chef Robotics deployment"}, SimpleNamespace(operation_id="warned")
    with pytest.raises(OperationPending, match="completed_response_validation"):
        tools.operation("search", args, ctx)
    assert (evidence.root / "operations/search/warned/raw.json").exists()
    with pytest.raises(OperationPending):
        tools.operation("search", args, ctx)
    assert tools.reconcile("search", args, ctx).status == "pending"
    assert calls == [1]
