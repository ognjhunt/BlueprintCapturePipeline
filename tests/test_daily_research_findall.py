"""Real daily-call receipts and owner journal with synthetic transport only."""
import copy
import json
import socket
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import parallel_findall_execution as execution
from blueprint_pipeline import parallel_findall_owner as owner
from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant
from tools.daily_research import findall, search
from tools.daily_research import capabilities
from tools.daily_research.consumer import Consumer
from tools.daily_research.runner import AGENT, MODEL, TEMPLATE, Ledger, Refusal, Runner, check_agent, digest, preflight

DAY = "2026-10-04"
NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
RUN_ID = "findall_synthetic_daily"
GRANT = object()  # No admission issuer or real credential is used.
SPEC = {"objective": "US operating sites with sourced physical tasks", "entity_type": "company",
        "generator": "base", "match_limit": 5,
        "match_conditions": [{"name": "task", "description": "Exact named-site task evidence"}]}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    def forbid(*args, **kwargs):
        raise AssertionError("network forbidden")

    def validate(grant, **binding):
        if grant is None:
            require_paid_resource_admission_grant(grant, **binding)
        assert grant is GRANT
        assert binding["resource_class"] == "parallel_findall"
        assert binding["require_allocation_binding"] is True

    monkeypatch.setattr(socket, "create_connection", forbid)
    monkeypatch.setattr(execution.safe_outbound_http, "open_request", forbid)
    monkeypatch.setattr(owner, "require_paid_resource_admission_grant", validate)
    monkeypatch.setattr(execution, "require_paid_resource_admission_grant", validate)


class Client(execution.AdmittedFindAllClient):
    def __init__(self, ledger):
        super().__init__("synthetic-never-sent")
        self.ledger, self.posts, self.reads = ledger, 0, []
        self.fail_after_post = None

    def _post(self, url, body=None):
        self.posts += 1
        entry = next(iter(self.ledger.get(DAY)[owner.SUBMISSIONS_FIELD].values()))
        assert entry["state"] == "submission_unresolved"
        assert entry["prepared"]["body_json"] == body
        if self.fail_after_post:
            self.fail_after_post()
        return json.dumps({"findall_id": RUN_ID, "generator": "base",
                           "status": {"status": "queued", "is_active": True},
                           "unknown": {"unchanged": [1, 2]}}).encode()

    def status(self, value):
        self.reads.append(("status", value))
        return {"findall_id": value, "status": {"status": "future_provider_status"}}

    def result(self, value):
        self.reads.append(("result", value))
        return {"run": {"findall_id": value}, "candidates": [{"status": "matched",
            "basis": [{"reasoning": "conditional", "citations": [{"url": "https://example.test/site"}]}],
            "future": {"unknown": True}}], "last_event_id": "event_future"}


@pytest.fixture
def runtime(tmp_path):
    ledger = Ledger(tmp_path)
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY, "state": "running",
           "session_id": "session_synthetic", "turn_id": "turn_synthetic",
           "started_at": NOW.isoformat(), "search_provider": search.PROFILE,
           "findall_profile": findall.PROFILE,
           "metadata": {"findall_tools_digest": digest(findall.tools())},
           "preflight": {"findall_tools_digest": digest(findall.tools())}}
    ledger.put(row)
    client = Client(ledger)
    state = {"lease": True, "authority": True, "grant": GRANT, "replies": [], "grant_requests": []}

    def grant_provider(prepared):
        state["grant_requests"].append(copy.deepcopy(prepared))
        return state["grant"]

    handler = findall.FindAllApplicationTools(
        ledger=ledger, client=client, grant_provider=grant_provider,
        current_authority=lambda row, prepared: state["authority"],
        assert_current_lease=lambda: state["lease"],
    )
    api = SimpleNamespace(findall_application_tools=handler,
        tool_admit=lambda row, phase: None,
        tool_result=lambda sid, event, key: state["replies"].append(copy.deepcopy(event)))
    yield row, ledger, client, state, api
    ledger.db.close()


def action(name=findall.CREATE, cid="call_create", arguments=None):
    return {"type": "function_call", "turn_id": "turn_synthetic", "call_id": cid,
            "name": name, "arguments": arguments if arguments is not None else {**SPEC, "maximum_cost_usd": "1.00"}}


def respond(runtime, pending):
    row, ledger, _, _, api = runtime
    with ledger.lock():
        return search.respond(row, {"required_actions": [pending]}, ledger, api,
                              phase="research", clock=lambda: NOW)


def test_real_dispatch_commits_claim_and_does_not_reacquire_lease(runtime, monkeypatch):
    row, ledger, client, state, _ = runtime
    with ledger.lock():
        monkeypatch.setattr(ledger, "lock", lambda: (_ for _ in ()).throw(AssertionError("nested lock")))
        search.respond(row, {"required_actions": [action()]}, ledger, runtime[-1],
                       phase="research", clock=lambda: NOW)
    assert client.posts == 1 and state["replies"][-1]["success"] is True
    stored = ledger.get(DAY)
    assert stored[owner.SUBMISSIONS_FIELD] == row[owner.SUBMISSIONS_FIELD]
    entry = next(iter(stored[owner.SUBMISSIONS_FIELD].values()))
    assert entry["state"] == "receipt_retained" and entry["findall_id"] == RUN_ID
    assert json.loads(ledger.read_bytes(entry["receipt_file"]))["unknown"] == {"unchanged": [1, 2]}


def test_repoll_and_process_resume_do_not_restart_provider(runtime):
    respond(runtime, action())
    first = copy.deepcopy(runtime[3]["replies"][-1])
    runtime[0].update(runtime[1].get(DAY))
    respond(runtime, action())
    assert runtime[2].posts == 1 and runtime[3]["replies"][-1] == first


@pytest.mark.parametrize("pending_name", [findall.CREATE, search.SEARCH])
def test_stale_row_cannot_erase_prior_claim_or_restart_provider(runtime, pending_name):
    row, ledger, client, state, _ = runtime
    stale = copy.deepcopy(row)
    respond(runtime, action())
    stored = copy.deepcopy(ledger.get(DAY))
    row.clear()
    row.update(stale)
    pending = action() if pending_name == findall.CREATE else action(search.SEARCH, "call_search", {"query": "site"})
    with pytest.raises(Refusal, match="findall_tool_stale_owner_row"):
        respond(runtime, pending)
    assert client.posts == 1 and len(state["grant_requests"]) == 1
    assert ledger.get(DAY) == stored


@pytest.mark.parametrize("field,value", [("lease", False), ("authority", False), ("grant", None)])
def test_missing_current_scope_lease_or_grant_has_zero_posts(runtime, field, value):
    runtime[3][field] = value
    if field == "lease":
        with pytest.raises(Refusal, match="findall_owner_current_lease_required"):
            respond(runtime, action())
        assert "application_tool_calls" not in runtime[1].get(DAY)
    else:
        respond(runtime, action())
        assert runtime[3]["replies"][-1]["success"] is False
    assert runtime[2].posts == 0


def test_no_handler_refuses_before_any_provider_attempt(runtime):
    del runtime[-1].findall_application_tools
    with pytest.raises(Refusal, match="findall_tool_handler_missing"):
        respond(runtime, action())
    assert runtime[2].posts == 0


def test_injection_cannot_enable_findall_on_an_unpinned_existing_session(runtime):
    row, ledger, client, _, _ = runtime
    del row["findall_profile"]
    ledger.put(row)
    with pytest.raises(Refusal, match="research_tool_action_binding_invalid"):
        respond(runtime, action())
    assert client.posts == 0 and "application_tool_calls" not in ledger.get(DAY)


@pytest.mark.parametrize("name", [findall.STATUS, findall.RESULT])
def test_only_owned_runs_are_read_and_raw_snapshots_are_immutable(runtime, name):
    respond(runtime, action())
    pending = action(name, "call_read", {"findall_id": RUN_ID})
    respond(runtime, pending)
    output = json.loads(runtime[3]["replies"][-1]["output"])
    assert output["evidence_scope"] == "discovery_only"
    assert json.loads(runtime[1].read_bytes(output["receipt"]["file"])) == output["snapshot"]
    assert runtime[1].get(DAY)[findall.READS_FIELD] == runtime[0][findall.READS_FIELD]
    respond(runtime, pending)
    assert len(runtime[2].reads) == 1
    if name == findall.RESULT:
        assert output["snapshot"]["candidates"][0]["status"] == "matched"
        assert output["snapshot"]["candidates"][0]["basis"][0]["reasoning"] == "conditional"


def test_foreign_run_has_zero_gets(runtime):
    respond(runtime, action(findall.STATUS, arguments={"findall_id": "findall_other_owner"}))
    assert runtime[2].reads == [] and runtime[3]["replies"][-1]["success"] is False


def test_post_response_store_failure_keeps_known_id_and_claim_in_caller(runtime, monkeypatch):
    row, ledger, client, state, _ = runtime
    original = ledger.get
    client.fail_after_post = lambda: monkeypatch.setattr(ledger, "get", lambda day: (_ for _ in ()).throw(OSError("synthetic")))
    respond(runtime, action())
    assert state["replies"][-1]["findall_id"] == RUN_ID
    assert next(iter(row[owner.SUBMISSIONS_FIELD].values()))["state"] == "submission_unresolved"
    monkeypatch.setattr(ledger, "get", original)
    respond(runtime, action())
    assert client.posts == 1 and owner.SUBMISSIONS_FIELD in ledger.get(DAY)


def test_metadata_does_not_read_credential_values(runtime, monkeypatch):
    class NamesOnly:
        def __contains__(self, name):
            return name == "PARALLEL_API_KEY"

        def __getitem__(self, name):
            raise AssertionError("secret value read")

        def get(self, *args):
            raise AssertionError("secret value read")

    monkeypatch.setattr(findall, "os", SimpleNamespace(environ=NamesOnly()))
    status = findall.runtime_status(runtime[-1])
    assert status["credential_binding_present"] is True
    assert status["callable_handler_installed"] is True
    assert status["production_binding_verified"] is False
    assert status["paid_operations_authorized"] is False
    assert {tool["name"] for tool in findall.tools()} == findall.NAMES


def test_caller_cannot_supply_an_uncommitted_tool_claim(runtime):
    row, _, client, _, api = runtime
    pending = action()
    binding = {k: pending[k] for k in ("turn_id", "call_id", "name", "arguments")}
    row["application_tool_calls"] = {"call_create": {"attempted": True, "request_digest": digest(binding)}}
    with pytest.raises(Exception, match="durable_call_claim_required"):
        api.findall_application_tools.execute(pending, row=row, phase="research")
    assert client.posts == 0


def profile_session(runtime, *, expansion_profile=None):
    row, _, _, _, api = runtime
    agent = {"id": AGENT, "model": MODEL, "reasoning": {"effort": "medium"},
             "multi_agent": {"enabled": False}, "tools": [{"type": "web_search"}],
             "instructions": "Synthetic existing owner instructions"}
    template = {"id": TEMPLATE, "network": {"access": "disabled"},
                "capability_directories": [capabilities.ROOT], "skills": [], "plugins": [],
                "files": [{"type": "inline", "path": capabilities.ROOT + "/" + name,
                           "size_bytes": size} for name, (size, _) in capabilities.TEMPLATE_FILES.items()]}
    api.get = lambda resource, rid: copy.deepcopy(agent if resource == "agent" else template)
    api.search_binding_present = lambda: True
    checked = preflight(api, search_provider=search.PROFILE, expansion_profile=expansion_profile)
    assert checked["findall_profile"] == findall.PROFILE
    assert checked["session_agent_override"]["tools"] == search.tools(expansion_profile=expansion_profile, findall_profile=findall.PROFILE)
    row["preflight"] = checked
    row["environment_id"] = "env_synthetic"
    row["create_payload"] = {"agent": copy.deepcopy(checked["session_agent_override"])}
    if expansion_profile:
        row["expansion_profile"] = expansion_profile
    return {"id": row["session_id"], "metadata": copy.deepcopy(row["metadata"]),
            "environment": {"id": row["environment_id"], "type": "openai_hosted"},
            "agent": {**agent, **checked["session_agent_override"]}}


@pytest.mark.parametrize("expansion_profile", [None, "exa-guarded-v1"])
def test_opt_in_registry_survives_normal_research_qa_and_repair_validation(runtime, expansion_profile):
    session = profile_session(runtime, expansion_profile=expansion_profile)
    check_agent(session["agent"], search_provider=search.PROFILE, expansion_profile=expansion_profile, findall_profile=findall.PROFILE)
    # QA and recovery/repair both use this same exact session validator.
    Consumer.check_session(runtime[0], session)
    without_handler = SimpleNamespace(get=runtime[-1].get, search_binding_present=lambda: True)
    default = preflight(without_handler, search_provider=search.PROFILE)
    assert "findall_profile" not in default
    assert default["session_agent_override"]["tools"] == search.tools()
    with pytest.raises(Refusal, match="agent_search_profile_mismatch"):
        check_agent(session["agent"], search_provider=search.PROFILE)


@pytest.mark.parametrize("changed", ["metadata", "preflight", "profile", "tool"])
def test_optional_registry_drift_refused(runtime, changed):
    session = profile_session(runtime)
    if changed in {"metadata", "preflight"}:
        runtime[0][changed]["findall_tools_digest"] = "0" * 64
        session["metadata"] = copy.deepcopy(runtime[0]["metadata"])
    elif changed == "profile":
        runtime[0]["findall_profile"] = "foreign-profile"
    else:
        session["agent"]["tools"][-1]["description"] = "foreign definition"
    with pytest.raises(Refusal, match="findall_tool_registry_binding_changed|agent_search_profile_mismatch"):
        Consumer.check_session(runtime[0], session)


@pytest.mark.parametrize("entry", ["research_observe", "research_cancel", "research_collect",
                                  "qa", "qa_observe", "qa_cancel", "qa_correction"])
def test_lifecycle_entry_refuses_stale_row_before_first_write_and_error_persistence(runtime, entry):
    row, ledger, client, state, api = runtime
    session = profile_session(runtime)
    ledger.put(row)
    stale = copy.deepcopy(row)
    respond(runtime, action())
    stored = copy.deepcopy(ledger.get(DAY))
    row.clear()
    row.update(stale)
    api.get = lambda resource, rid: copy.deepcopy(session)
    api.listing = lambda resource, sid: ([{"id": row["turn_id"], "status": "running"}]
                                        if resource == "turns" else [])
    session["required_actions"] = [action()]
    runner = object.__new__(Runner)
    runner.ledger, runner.api, runner.config, runner.clock = ledger, api, {}, lambda: NOW
    runner.stop_requested = lambda: False
    consumer = Consumer(ledger, {}, api, clock=lambda: NOW)
    calls = {"research_observe": lambda: runner.observe(row),
             "research_cancel": lambda: runner.cancel(row, "synthetic"),
             "research_collect": lambda: runner.collect(row),
             "qa": lambda: consumer.qa(row),
             "qa_observe": lambda: consumer.observe(row, NOW),
             "qa_cancel": lambda: consumer.cancel(row, "synthetic"),
             "qa_correction": lambda: consumer.correct_qa(row, {}, session, NOW)}
    with ledger.lock(), pytest.raises(Refusal, match="findall_tool_stale_owner_row"):
        calls[entry]()
    assert client.posts == 1 and len(state["grant_requests"]) == 1
    assert ledger.get(DAY) == stored


def test_foreign_ledger_cannot_use_injected_handler(runtime, tmp_path):
    other = Ledger(tmp_path / "other")
    try:
        with other.lock(), pytest.raises(Refusal, match="findall_tool_ledger_binding_changed"):
            search.respond(runtime[0], {"required_actions": [action()]}, other, runtime[-1],
                           phase="research", clock=lambda: NOW)
        assert other.get(DAY) is None and runtime[2].posts == 0
    finally:
        other.db.close()
