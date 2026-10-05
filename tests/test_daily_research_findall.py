"""Daily FindAll calls admitted by the shared paid expansion allowance; synthetic transport only.

Ports ognjhunt/BlueprintCapturePipeline#2577's caller contracts and adds the owner-directed
allocation, production wiring and original-deadline settlement. No network, credential,
real grant or provider call is used: every grant comes from the real canonical issuer only
after the real allocation admits the exact synthetic request.
"""
import copy
import hashlib
import json
import signal
import socket
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import parallel_findall_execution as execution
from blueprint_pipeline import parallel_findall_owner as owner
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from tools.daily_research import allocation, capabilities, findall, render, search
from tools.daily_research.consumer import Consumer
from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.runner import (
    AGENT,
    MODEL,
    PROJECT,
    TEMPLATE,
    Ledger,
    Refusal,
    Runner,
    check_agent,
    digest,
    preflight,
)

DAY = "2026-10-04"
NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)  # 07:00 in America/Chicago on the run date
COMMIT = "c" * 40
RUN_ID = "findall_synthetic_daily"
SPEC = {"objective": "US operating sites with sourced physical tasks", "entity_type": "company",
        "generator": "base", "match_limit": 5,
        "match_conditions": [{"name": "task", "description": "Exact named-site task evidence"}]}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    def forbid(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "create_connection", forbid)
    monkeypatch.setattr(execution.safe_outbound_http, "open_request", forbid)


def owner_control(limit="10.00", *, sources=("exa", "findall"), enabled=True, commit=COMMIT):
    """Company control carrying one verified owner direction (allocation.py)."""
    value = {"schema_version": allocation.DIRECTION, "version": 1, "supersedes": None, "per_run_limit_usd": limit,
             "sources": sorted(sources), "scope": dict(allocation.SCOPE),
             "effective_from": "2026-09-01T00:00:00+00:00", "expires_at": "2026-12-31T00:00:00+00:00",
             "approval_reference": "owner-chat-2026-10-04-findall-enabled", "approved_by": "owner",
             "issued_at": "2026-09-01T00:00:00+00:00", "reason": "Synthetic combined allowance"}
    sha = allocation.digest(value)
    return {"source_commit": commit, "paid_expansion": {"enabled": enabled, "current": {
        "sha256": sha, "version": 1, "uri": allocation.uri(sha), "direction": value}}}


class Client(execution.AdmittedFindAllClient):
    def __init__(self, ledger):
        super().__init__("synthetic-never-sent")
        self.ledger, self.posts, self.reads, self.cancels = ledger, 0, [], []
        self.fail_after_post = None
        self.post_error = None
        self.cancel_error = None
        self.active = True

    def _post(self, url, body=None):
        if url.endswith("/cancel"):
            assert body is None
            # The cancel claim is durable before its one POST.
            record = next(iter(self.ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))
            assert record["state"] == "cancel_unresolved"
            self.cancels.append(url.rsplit("/", 2)[-2])
            if self.cancel_error:
                raise execution.FindAllError(self.cancel_error)
            self.active = False
            return b""
        self.posts += 1
        entries = self.ledger.get(DAY)[owner.SUBMISSIONS_FIELD]
        # The one claim for this exact body is durable and unresolved before its single POST.
        assert any(e["prepared"]["body_json"] == body and e["state"] == "submission_unresolved"
                   for e in entries.values())
        if self.post_error:
            raise TimeoutError("synthetic_uncertain_reply")
        if self.fail_after_post:
            self.fail_after_post()
        return json.dumps({"findall_id": RUN_ID if self.posts == 1 else f"{RUN_ID}_{self.posts}",
                           "generator": body["generator"],
                           "status": {"status": "queued", "is_active": True},
                           "unknown": {"unchanged": [1, 2]}}).encode()

    def status(self, value):
        self.reads.append(("status", value))
        return {"findall_id": value, "status": {"status": "running" if self.active else "cancelled",
                                                "is_active": self.active}, "future": {"unknown": True}}

    def result(self, value):
        self.reads.append(("result", value))
        return {"run": {"findall_id": value}, "candidates": [{"status": "matched",
            "basis": [{"reasoning": "conditional", "citations": [{"url": "https://example.test/site"}]}],
            "future": {"unknown": True}}], "last_event_id": "event_future"}


def pinned_row(control, **changes):
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY, "state": "running",
           "session_id": "session_synthetic", "turn_id": "turn_synthetic",
           "started_at": NOW.isoformat(), "research_runtime_seconds": 1200, "search_provider": search.PROFILE,
           "findall_profile": findall.PROFILE,
           "metadata": {"findall_tools_digest": digest(findall.tools())},
           "preflight": {"findall_tools_digest": digest(findall.tools())}}
    row.update(changes)
    row["paid_expansion_grant"] = allocation.grant(control, row, NOW)
    return row


@pytest.fixture
def runtime(tmp_path):
    ledger = Ledger(tmp_path)
    state = {"lease": True, "control": owner_control(), "replies": [], "stopped": False, "now": NOW}
    row = pinned_row(state["control"])
    assert row["paid_expansion_grant"]["state"] == "granted"
    ledger.put(row)
    client = Client(ledger)
    handler = findall.FindAllApplicationTools(
        ledger=ledger, client=client, control=lambda: copy.deepcopy(state["control"]),
        assert_current_lease=lambda: state["lease"], clock=lambda: state["now"], stopped=lambda: state["stopped"])
    api = SimpleNamespace(findall_application_tools=handler,
        tool_admit=lambda row, phase: None,
        tool_result=lambda sid, event, key: state["replies"].append(copy.deepcopy(event)))
    yield row, ledger, client, state, api
    ledger.db.close()


def action(name=findall.CREATE, cid="call_create", arguments=None, cost="1.00"):
    return {"type": "function_call", "turn_id": "turn_synthetic", "call_id": cid,
            "name": name, "arguments": arguments if arguments is not None else {**SPEC, "maximum_cost_usd": cost}}


def respond(runtime, pending, phase="research"):
    row, ledger, _, state, api = runtime
    with ledger.lock():
        return search.respond(row, {"required_actions": [pending]}, ledger, api,
                              phase=phase, clock=lambda: state["now"])


def reply(runtime):
    event = runtime[3]["replies"][-1]
    return event, json.loads(event["output"]) if "output" in event else None


def findall_claims(ledger):
    return [claim for claim in allocation.claims(ledger.get(DAY)) if claim["source"] == "findall"]


# --- Ported #2577 caller contracts -------------------------------------------------------


def test_real_dispatch_commits_claim_and_does_not_reacquire_lease(runtime, monkeypatch):
    row, ledger, client, state, api = runtime
    with ledger.lock():
        monkeypatch.setattr(ledger, "lock", lambda: (_ for _ in ()).throw(AssertionError("nested lock")))
        search.respond(row, {"required_actions": [action()]}, ledger, api, phase="research", clock=lambda: NOW)
    assert client.posts == 1 and state["replies"][-1]["success"] is True
    stored = ledger.get(DAY)
    assert stored[owner.SUBMISSIONS_FIELD] == row[owner.SUBMISSIONS_FIELD]
    entry = next(iter(stored[owner.SUBMISSIONS_FIELD].values()))
    assert entry["state"] == "receipt_retained" and entry["findall_id"] == RUN_ID
    assert json.loads(ledger.read_bytes(entry["receipt_file"]))["unknown"] == {"unchanged": [1, 2]}
    output = reply(runtime)[1]
    assert output["findall_id"] == RUN_ID and output["reserved_micros"] == 1_000_000
    assert output["billing_verified"] is False and output["evidence_scope"] == "discovery_only"


def test_repoll_and_process_resume_do_not_restart_provider(runtime):
    respond(runtime, action())
    first = copy.deepcopy(runtime[3]["replies"][-1])
    runtime[0].update(runtime[1].get(DAY))
    respond(runtime, action())
    assert runtime[2].posts == 1 and runtime[3]["replies"][-1] == first


@pytest.mark.parametrize("pending_name", [findall.CREATE, search.SEARCH])
def test_stale_row_cannot_erase_prior_claim_or_restart_provider(runtime, pending_name):
    row, ledger, client, _, _ = runtime
    stale = copy.deepcopy(row)
    respond(runtime, action())
    stored = copy.deepcopy(ledger.get(DAY))
    row.clear()
    row.update(stale)
    pending = action() if pending_name == findall.CREATE else action(search.SEARCH, "call_search", {"query": "site"})
    with pytest.raises(Refusal, match="findall_tool_stale_owner_row"):
        respond(runtime, pending)
    assert client.posts == 1
    assert ledger.get(DAY) == stored


def test_missing_current_lease_writes_nothing_and_posts_nothing(runtime):
    runtime[3]["lease"] = False
    with pytest.raises(Refusal, match="findall_owner_current_lease_required"):
        respond(runtime, action())
    assert "application_tool_calls" not in runtime[1].get(DAY) and runtime[2].posts == 0


def test_no_handler_reports_unavailable_and_ordinary_research_continues(runtime):
    _row, ledger, client, state, api = runtime
    del api.findall_application_tools
    api.application_tool = lambda name, arguments: {"results": []}
    respond(runtime, action())
    event, output = reply(runtime)
    assert event["success"] is False and output["reason"] == "findall_runtime_unavailable"
    assert json.loads(event["error"])["guidance"].startswith("No FindAll run was started")
    respond(runtime, action(search.SEARCH, "call_search", {"query": "US laundry sites"}))
    assert state["replies"][-1]["success"] is True
    assert client.posts == 0 and owner.SUBMISSIONS_FIELD not in ledger.get(DAY)


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
    respond(runtime, pending, phase="research")
    output = reply(runtime)[1]
    assert output["evidence_scope"] == "discovery_only"
    assert json.loads(runtime[1].read_bytes(output["receipt"]["file"])) == output["snapshot"]
    assert runtime[1].get(DAY)[findall.READS_FIELD] == runtime[0][findall.READS_FIELD]
    respond(runtime, pending)
    assert len(runtime[2].reads) == 1
    if name == findall.RESULT:
        assert output["snapshot"]["candidates"][0]["status"] == "matched"
        assert output["snapshot"]["candidates"][0]["basis"][0]["reasoning"] == "conditional"


def test_foreign_run_has_zero_gets_and_names_the_refusal(runtime):
    respond(runtime, action(findall.STATUS, arguments={"findall_id": "findall_other_owner"}))
    event = runtime[3]["replies"][-1]
    assert runtime[2].reads == [] and event["success"] is False
    assert json.loads(event["error"]) == {"code": "findall_tool_run_not_owned"}


def test_post_response_store_failure_keeps_known_id_and_claim_in_caller(runtime, monkeypatch):
    row, ledger, client, _state, _ = runtime
    original = ledger.get
    client.fail_after_post = lambda: monkeypatch.setattr(ledger, "get", lambda day: (_ for _ in ()).throw(OSError("synthetic")))
    respond(runtime, action())
    event, output = reply(runtime)
    assert event["success"] is False and output["state"] == "submission_unresolved"
    assert output["findall_id"] == RUN_ID and output["reserved_micros"] == 1_000_000
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
    assert checked["session_agent_override"]["instructions"].count(findall.instructions()) == 1
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
    row, ledger, client, _state, api = runtime
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
    assert client.posts == 1
    assert ledger.get(DAY) == stored


def test_a_registry_change_never_blocks_cancellation(runtime):
    row, ledger, _, _, api = runtime
    row["metadata"]["findall_tools_digest"] = "0" * 64  # A mid-run release changed the definitions.
    ledger.put(row)
    with pytest.raises(Refusal, match="findall_tool_registry_binding_changed"):
        search.assert_findall_caller(row, ledger, api)
    search.assert_findall_caller(row, ledger, api, registry=False)  # Cancellation still proceeds.
    del api.findall_application_tools
    search.assert_findall_caller(row, ledger, api, registry=False)


def test_foreign_ledger_cannot_use_injected_handler(runtime, tmp_path):
    other = Ledger(tmp_path / "other")
    try:
        with other.lock(), pytest.raises(Refusal, match="findall_tool_ledger_binding_changed"):
            search.respond(runtime[0], {"required_actions": [action()]}, other, runtime[-1],
                           phase="research", clock=lambda: NOW)
        assert other.get(DAY) is None and runtime[2].posts == 0
    finally:
        other.db.close()


# --- Shared allocation admission ----------------------------------------------------------


def test_create_debits_its_whole_maximum_through_the_canonical_exact_grant(runtime, monkeypatch):
    seen = []
    original = execution.require_paid_resource_admission_grant

    def observe(grant, **binding):
        seen.append((grant, binding))
        return original(grant, **binding)

    monkeypatch.setattr(execution, "require_paid_resource_admission_grant", observe)
    respond(runtime, action(cost="4.00"))
    (grant, binding), = seen
    entry = next(iter(runtime[1].get(DAY)[owner.SUBMISSIONS_FIELD].values()))
    assert isinstance(grant, PaidResourceAdmissionGrant) and grant.resource_class == "parallel_findall"
    assert grant.allocation_binding_digest == entry["allocation_binding_digest"] == binding["allocation_binding_digest"]
    assert findall_claims(runtime[1]) == [{"source": "findall", "operation_sha256": next(iter(
        runtime[1].get(DAY)[owner.SUBMISSIONS_FIELD])), "reserved_micros": 4_000_000}]
    status = allocation.diagnostic(runtime[1].get(DAY), runtime[3]["control"], now=NOW)
    assert status["reserved_micros"] == 4_000_000 and status["remaining_micros"] == 6_000_000


def test_exa_and_findall_together_never_exceed_the_one_owner_limit(runtime):
    row, ledger, client, state, _ = runtime
    # An Exa claim already holds $4 of the $10 run allowance.
    row["exa_expansion"] = {"cap_micros": 4_000_000, "intent_sha256": "e" * 64, "state": "submission_unresolved"}
    ledger.put(row)
    respond(runtime, action(cid="call_over_per_start", cost="5.50"))
    output = reply(runtime)[1]
    assert output["reason"] == "paid_expansion_cap_exceeds_per_start_maximum"
    assert (output["remaining_micros"], output["max_start_micros"]) == (6_000_000, 5_000_000)
    respond(runtime, action(cid="call_first", cost="5.00"))
    assert reply(runtime)[1]["ok"] is True and client.posts == 1
    respond(runtime, action(cid="call_over_remaining", cost="2.00"))
    output = reply(runtime)[1]
    assert output["reason"] == "paid_expansion_cap_exceeds_remaining"
    assert (output["remaining_micros"], output["max_start_micros"]) == (1_000_000, 1_000_000)
    assert output["claim_created"] is False and client.posts == 1
    respond(runtime, action(cid="call_fits", cost="1.00"))
    assert reply(runtime)[1]["ok"] is True and client.posts == 2
    found = allocation.claims(ledger.get(DAY))
    assert sum(claim["reserved_micros"] for claim in found) == 10_000_000
    assert allocation.problem(row["paid_expansion_grant"], found, 1_000_000, NOW, control=state["control"],
                              source="exa") == "paid_expansion_cap_exceeds_remaining"


@pytest.mark.parametrize(("change", "code"), [
    ({"grant_sources": ("exa",)}, "paid_expansion_source_not_directed"),
    ({"live_sources": ("exa",)}, "paid_expansion_source_not_directed"),
    ({"enabled": False}, "paid_expansion_disabled"),
    ({"commit": "d" * 40}, "paid_expansion_source_commit_changed"),
    ({"stopped": True}, "findall_stopped"),
    ({"later": 1200}, "findall_original_deadline_exhausted"),
    ({"limit": "3.00", "cost": "2.00", "grant_limit": "3.00"}, "paid_expansion_cap_exceeds_per_start_maximum"),
])
def test_every_refusal_is_actionable_and_starts_nothing(runtime, change, code):
    row, ledger, client, state, _ = runtime
    if "grant_sources" in change or "grant_limit" in change:
        row["paid_expansion_grant"] = allocation.grant(owner_control(change.get("grant_limit", "10.00"),
            sources=change.get("grant_sources", ("exa", "findall"))), row, NOW)
        ledger.put(row)
    state["control"] = owner_control(change.get("limit", "10.00"), sources=change.get("live_sources", ("exa", "findall")),
                                     enabled=change.get("enabled", True), commit=change.get("commit", COMMIT))
    state["stopped"] = change.get("stopped", False)
    state["now"] = NOW + timedelta(seconds=change.get("later", 0))
    if "later" in change:
        with pytest.raises(Refusal, match="research_tool_stopped_or_expired"):
            respond(runtime, action())
        # The handler's own original-deadline fence agrees, without any claim.
        assert row["paid_expansion_grant"] and runtime[-1].findall_application_tools.create_problem(
            ledger.get(DAY), {"maximum_cost_usd": "1"}, "research") == code
    else:
        respond(runtime, action(cost=change.get("cost", "1.00")))
        event, output = reply(runtime)
        assert event["success"] is False and output["reason"] == code
        assert json.loads(event["error"]) == {"code": code, "guidance": output["action"]}
        assert output["action"].startswith("No FindAll run was started and no claim was consumed.")
    assert client.posts == 0 and owner.SUBMISSIONS_FIELD not in ledger.get(DAY)


def test_creates_belong_to_research_before_final_output(runtime):
    row, ledger, client, _, _ = runtime
    row.update(qa={"turn_id": "turn_qa"})
    ledger.put(row)
    pending = {**action(cid="call_in_qa"), "turn_id": "turn_qa"}
    respond(runtime, pending, phase="qa")
    assert reply(runtime)[1]["reason"] == "findall_before_final_qa_only" and client.posts == 0


def test_an_uncertain_create_holds_its_reservation_without_replay(runtime):
    _row, ledger, client, state, _ = runtime
    client.post_error = True
    respond(runtime, action(cid="call_uncertain", cost="3.00"))
    event, output = reply(runtime)
    assert event["success"] is False and output["state"] == "submission_unresolved"
    assert output["findall_id"] is None and output["replay_permitted"] is False
    assert findall_claims(ledger)[0]["reserved_micros"] == 3_000_000
    assert allocation.diagnostic(ledger.get(DAY), state["control"], now=NOW)["remaining_micros"] == 7_000_000
    client.post_error = False
    respond(runtime, action(cid="call_uncertain", cost="3.00"))  # A repoll never restarts it.
    assert client.posts == 1


def test_price_and_format_problems_are_named_before_any_claim(runtime):
    respond(runtime, action(cid="call_cheap", arguments={**SPEC, "match_limit": 50, "maximum_cost_usd": "1.00"}))
    output = reply(runtime)[1]
    assert output["reason"] == "findall_spend_ceiling_below_estimated_cost"
    assert output["estimated_maximum_cost_usd"] == "1.75"
    respond(runtime, action(cid="call_format", cost="$1"))
    assert json.loads(runtime[3]["replies"][-1]["error"]) == {"code": "findall_maximum_cost_usd_invalid"}
    assert runtime[2].posts == 0 and owner.SUBMISSIONS_FIELD not in runtime[1].get(DAY)


# --- Production wiring ----------------------------------------------------------------------


def test_default_factory_builds_no_handler_without_the_worker_binding(monkeypatch, tmp_path):
    from tools.daily_research.firestore import FencedProvider
    provider = SimpleNamespace(ledger=SimpleNamespace(bridge=None))
    assert findall.owner_handler(provider) is None
    monkeypatch.setenv("PARALLEL_API_KEY", "contains whitespace")  # Unusable binding: fail closed, never echo.
    assert findall.owner_handler(provider) is None
    monkeypatch.setenv("PARALLEL_API_KEY", "synthetic-offline-value")
    handler = findall.owner_handler(provider)
    assert isinstance(handler, findall.FindAllApplicationTools)
    assert "synthetic-offline-value" not in json.dumps(findall.runtime_status(SimpleNamespace(findall_application_tools=handler)))
    assert FencedProvider.findall_from_worker_binding is True
    from importlib import util
    from pathlib import Path
    spec = util.spec_from_file_location("canary_findall_opt_out", Path(__file__).resolve().parents[1]
                                        / "tools/daily_research/operators/research-perplexity-canary.py")
    canary = util.module_from_spec(spec)
    spec.loader.exec_module(canary)
    assert canary.CanaryProvider.findall_from_worker_binding is False


@pytest.mark.parametrize(("search_provider", "mcp_profile"), [(None, None), (search.PROFILE, search.MCP_RESEARCH_PROFILE)])
def test_a_handler_never_changes_other_profiles(runtime, search_provider, mcp_profile, monkeypatch):
    _, _, _, _, api = runtime
    profile_session(runtime)
    if mcp_profile:
        monkeypatch.setattr(search, "mcp_connections", lambda tools, profile: [])
        api.resolve_mcp_vaults = lambda connections: []
    checked = preflight(api, search_provider=search_provider, mcp_profile=mcp_profile)
    assert "findall_profile" not in checked
    assert all(tool.get("name") not in findall.NAMES for tool in checked.get("session_agent_override", {}).get("tools", []))


def test_findall_only_profile_freezes_the_shared_grant(tmp_path):
    from tests.test_daily_research_search import fixture as search_fixture
    generator = search_fixture.__wrapped__(tmp_path)
    runner, api, ledger = next(generator)
    try:
        ledger.paid_expansion_control = lambda: owner_control()
        ledger.control = owner_control()
        api.findall_application_tools = findall.FindAllApplicationTools(
            ledger=ledger, client=Client(ledger), control=lambda: ledger.control,
            assert_current_lease=lambda: True, clock=lambda: runner.clock())
        row = runner.start_or_resume()
        assert row["findall_profile"] == findall.PROFILE and "expansion_profile" not in row
        assert row["metadata"]["findall_tools_digest"] == digest(findall.tools())
        assert row["paid_expansion_grant"]["state"] == "granted"
        assert row["paid_expansion_grant"]["sources"] == ["exa", "findall"]
        assert api.payloads[0]["agent"]["tools"][-3:] == findall.tools()
    finally:
        try:
            next(generator)
        except StopIteration:
            pass


# --- Original-deadline settlement ---------------------------------------------------------


def started(runtime, cost="1.00", cid="call_create"):
    respond(runtime, action(cid=cid, cost=cost))
    assert reply(runtime)[1]["ok"] is True
    runtime[0].update(runtime[1].get(DAY))


def test_settlement_waits_for_research_end_or_the_original_deadline(runtime):
    row = runtime[0]
    assert not findall.settlement_due(row, NOW)
    started(runtime)
    assert not findall.settlement_due(row, NOW + timedelta(seconds=1199))
    assert findall.settlement_due(row, NOW + timedelta(seconds=1200))
    assert findall.settlement_due({**row, "state": "awaiting_review"}, NOW)


def test_a_run_still_active_at_the_deadline_is_cancelled_once(runtime):
    row, ledger, client, state, api = runtime
    started(runtime)
    state["now"] = NOW + timedelta(seconds=1200)
    with ledger.lock():
        api.findall_application_tools.settle(row, reason="original_research_deadline")
        api.findall_application_tools.settle(row, reason="original_research_deadline")
    record = next(iter(ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))
    assert client.cancels == [RUN_ID] and record["state"] == "terminal"
    assert record["provider_status"] == "cancelled" and record["cancel_attempts"] == 1
    statuses = [ref for ref in record["receipts"] if ref != record["result_receipt"]]
    assert all(json.loads(ledger.read_bytes(ref["file"]))["findall_id"] == RUN_ID for ref in statuses)
    # One result read, after the cancel, keeps the matches; a second pass reads nothing more.
    assert [kind for kind, _ in client.reads] == ["status", "status", "result"]
    assert json.loads(ledger.read_bytes(record["result_receipt"]["file"]))["candidates"][0]["status"] == "matched"
    assert findall_claims(ledger)[0]["reserved_micros"] == 1_000_000  # Cancellation is not a refund.
    assert not findall.settlement_due(ledger.get(DAY), state["now"])


def test_a_completed_run_needs_no_cancel_and_an_unknown_id_stays_reserved(runtime):
    row, ledger, client, _state, api = runtime
    started(runtime)
    client.active = False
    client.post_error = True
    respond(runtime, action(cid="call_uncertain", cost="2.00"))
    row.update(ledger.get(DAY), state="awaiting_review")
    ledger.put(row)
    with ledger.lock():
        api.findall_application_tools.settle(row, reason="research_ended")
    records = {record["operation_id"].rsplit(":", 1)[-1]: record
               for record in ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()}
    states = {name: record["state"] for name, record in records.items()}
    assert states == {"call_create": "terminal", "call_uncertain": "provider_id_unknown"}
    assert client.cancels == []
    # A run found terminal keeps one free result snapshot; an unknown ID has nothing to read.
    assert client.reads.count(("result", RUN_ID)) == 1 and "result_receipt" in records["call_create"]
    assert "result_receipt" not in records["call_uncertain"]
    assert sorted(claim["reserved_micros"] for claim in findall_claims(ledger)) == [1_000_000, 2_000_000]


def test_failed_cancels_are_bounded_and_the_reservation_stays_held(runtime):
    row, ledger, client, _state, api = runtime
    started(runtime)
    client.cancel_error = "findall_transport_failed"
    row.update(state="awaiting_review")
    ledger.put(row)
    for _ in range(findall.MAX_CANCEL_ATTEMPTS + 2):
        with ledger.lock():
            api.findall_application_tools.settle(row, reason="research_ended")
    record = next(iter(ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))
    assert client.cancels == [RUN_ID] * findall.MAX_CANCEL_ATTEMPTS
    assert record["state"] == "cancel_attempts_exhausted" and record["cancel_error"] == "findall_transport_failed"
    assert ("result", RUN_ID) not in client.reads and "result_receipt" not in record  # Never cancelled or terminal.
    assert findall_claims(ledger)[0]["reserved_micros"] == 1_000_000


def test_runner_settles_at_research_end_and_never_without_a_client(runtime):
    row, ledger, client, state, api = runtime
    started(runtime)
    runner = object.__new__(Runner)
    runner.ledger, runner.api, runner.clock = ledger, SimpleNamespace(), lambda: state["now"]
    row.update(state="awaiting_review")
    ledger.put(row)
    runner.settle_findall(row)  # No client in this process: nothing is sent; the claim stays reserved.
    assert client.cancels == [] and findall.SETTLEMENTS_FIELD not in ledger.get(DAY)
    runner.api = api
    with ledger.lock():
        runner.settle_findall(row)
    assert client.cancels == [RUN_ID]
    assert next(iter(ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))["state"] == "terminal"


@pytest.mark.parametrize("change", ["brake", "source", "commit", "expiry", "deadline", "limit", "lease"])
def test_post_commit_authority_loss_never_posts_and_never_releases_claim(runtime, monkeypatch, change):
    _row, ledger, client, state, _ = runtime
    original = ledger.put
    changed = False

    def commit(value):
        nonlocal changed
        original(value)
        if value.get(findall.SUBMISSIONS_FIELD) and not changed:
            changed = True
            if change == "brake":
                state["control"]["paid_expansion"]["enabled"] = False
            elif change == "source":
                state["control"] = owner_control(sources=("exa",))
            elif change == "commit":
                state["control"]["source_commit"] = "d" * 40
            elif change == "expiry":
                current = state["control"]["paid_expansion"]["current"]
                current["direction"]["expires_at"] = (NOW + timedelta(seconds=1)).isoformat()
                current["sha256"] = allocation.digest(current["direction"])
                current["uri"] = allocation.uri(current["sha256"])
                state["now"] = NOW + timedelta(seconds=2)
            elif change == "deadline":
                state["now"] = NOW + timedelta(seconds=1200)
            elif change == "limit":
                state["control"] = owner_control("1.00")
            else:
                state["lease"] = False

    monkeypatch.setattr(ledger, "put", commit)
    if change == "deadline":
        with pytest.raises(Refusal, match="research_tool_stopped_or_expired"):
            respond(runtime, action(cost="2.00"))
    else:
        respond(runtime, action(cost="2.00"))
    assert client.posts == 0
    assert findall_claims(ledger)[0]["reserved_micros"] == 2_000_000
    assert next(iter(ledger.get(DAY)[findall.SUBMISSIONS_FIELD].values()))["state"] == "submission_unresolved"
    event = json.loads(ledger.read_bytes(ledger.get(DAY)["application_tool_calls"]["call_create"]["result_file"]))
    assert event["success"] is False


@pytest.mark.parametrize("change", ["brake", "source", "commit", "expiry", "limit", "stop"])
def test_running_row_cancels_on_fresh_paid_authority_loss(runtime, change):
    row, ledger, client, state, api = runtime
    started(runtime, cost="2.00")
    if change == "brake":
        state["control"]["paid_expansion"]["enabled"] = False
    elif change == "source":
        state["control"] = owner_control(sources=("exa",))
    elif change == "commit":
        state["control"]["source_commit"] = "d" * 40
    elif change == "expiry":
        current = state["control"]["paid_expansion"]["current"]
        current["direction"]["expires_at"] = (NOW + timedelta(seconds=1)).isoformat()
        current["sha256"] = allocation.digest(current["direction"])
        current["uri"] = allocation.uri(current["sha256"])
        state["now"] = NOW + timedelta(seconds=2)
    elif change == "limit":
        state["control"] = owner_control("1.00")
    else:
        state["stopped"] = True
    runner = object.__new__(Runner)
    runner.ledger, runner.api, runner.clock = ledger, api, lambda: state["now"]
    with ledger.lock():
        runner.settle_findall(row)
    assert row["state"] == "running" and client.cancels == [RUN_ID]
    record = next(iter(ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))
    assert record["state"] == "terminal" and record["reason"] not in {"research_ended", "original_research_deadline"}
    assert findall_claims(ledger)[0]["reserved_micros"] == 2_000_000


def findall_writes(ledger, monkeypatch):
    """Record each FindAll snapshot file written; the tool's own result event is not one."""
    names = []
    original = ledger.write_bytes

    def write(name, raw):
        if "-tool-findall-" in name:
            assert len(raw) <= owner.MAX_SNAPSHOT_BYTES < 8 * 1024 * 1024  # One file below the store ceiling.
            names.append(name)
        return original(name, raw)

    monkeypatch.setattr(ledger, "write_bytes", write)
    return names


@pytest.mark.parametrize("payload", ["x" * 550_000, "\U0001f680" * 100_000, "x" * 6_000_000],
                         ids=["550kB", "four-byte", "6MB"])
def test_large_results_are_one_file_served_in_receipt_bound_pages_with_lossless_export(runtime, monkeypatch, payload):
    _row, ledger, client, _, _api = runtime
    started(runtime)
    snapshot = {"run": {"findall_id": RUN_ID}, "candidates": [{"status": "matched", "future": payload}],
                "future": {"citations": [{"url": "https://example.test/raw"}], "unknown": True}}
    monkeypatch.setattr(client, "result", lambda value: client.reads.append(("result", value)) or snapshot)
    writes = findall_writes(ledger, monkeypatch)
    respond(runtime, action(findall.RESULT, "call_result", {"findall_id": RUN_ID}))
    event, first = reply(runtime)
    assert event["success"] is True and len(event["output"].encode()) < search.MAX_RESPONSE
    assert first["page"] == 0 and first["next_page"] == 1 and first["evidence_scope"] == "discovery_only"
    stored = ledger.get(DAY)
    receipt = stored[findall.READS_FIELD]["call_result"]
    assert writes == [receipt["file"]] and "parts" not in receipt  # ONE immutable file, one write.
    raw = ledger.read_bytes(receipt["file"])
    rebuilt = raw.decode("utf-8")
    if len(raw) < 1_000_000:  # Larger files are rebuilt in the owner tests; here, spot pages below.
        rebuilt = "".join(owner.snapshot_page(receipt, ledger.read_bytes, page)["json_fragment"]
                          for page in range(first["page_count"]))
    assert rebuilt.encode("utf-8") == raw and json.loads(rebuilt) == snapshot
    assert first["json_fragment"] == rebuilt[:owner.SNAPSHOT_FRAGMENT_CHARS]
    owner.validate_snapshot(receipt, ledger.read_bytes)
    findall.validate_snapshot_exports(stored, ledger.read_bytes)
    assert {key: receipt[key] for key in ("file", "sha256", "bytes")} in findall.receipt_refs(stored)
    size = owner.SNAPSHOT_FRAGMENT_CHARS
    for page in {1, first["page_count"] - 1}:
        respond(runtime, action(findall.RESULT, f"call_page_{page}", {
            "findall_id": RUN_ID, "receipt_sha256": first["receipt"]["sha256"], "page": page}))
        event, shown = reply(runtime)
        assert event["success"] is True and shown["page"] == page
        assert shown["json_fragment"] == rebuilt[page * size:(page + 1) * size]
        assert len(event["output"].encode()) < search.MAX_RESPONSE
    # Continuation slices the same file: no provider read and no new snapshot.
    assert client.reads.count(("result", RUN_ID)) == 1 and writes == [receipt["file"]]
    for cid, cursor, code in (("call_wrong_cursor", "0" * 64, "findall_snapshot_cursor_not_owned"),
                              ("call_bad_cursor", "not-a-sha256", "findall_snapshot_cursor_not_owned"),
                              ("call_past_end", first["receipt"]["sha256"], "findall_snapshot_page_invalid")):
        respond(runtime, action(findall.RESULT, cid, {
            "findall_id": RUN_ID, "receipt_sha256": cursor, "page": first["page_count"]}))
        event, _ = reply(runtime)
        assert event["success"] is False and json.loads(event["error"]) == {"code": code}
    original_read = ledger.read_bytes
    monkeypatch.setattr(ledger, "read_bytes", lambda name: b"changed" if name == receipt["file"] else original_read(name))
    with pytest.raises(execution.FindAllError, match="snapshot_binding_invalid"):
        findall.validate_snapshot_exports(stored, ledger.read_bytes)


def test_a_result_above_the_retention_limit_is_refused_fast_with_no_writes_and_the_run_kept(runtime, monkeypatch):
    _row, ledger, client, _, _ = runtime
    started(runtime)
    entry = copy.deepcopy(next(iter(ledger.get(DAY)[owner.SUBMISSIONS_FIELD].values())))
    snapshot = {"run": {"findall_id": RUN_ID}, "candidates": [{"status": "unmatched", "future": "x" * (8 * 1024 * 1024)}]}
    monkeypatch.setattr(client, "result", lambda value: client.reads.append(("result", value)) or snapshot)
    writes = findall_writes(ledger, monkeypatch)
    began = time.monotonic()
    respond(runtime, action(findall.RESULT, "call_result", {"findall_id": RUN_ID}))
    elapsed = time.monotonic() - began
    event, output = reply(runtime)
    assert event["success"] is False and output["reason"] == "findall_snapshot_too_large"
    assert json.loads(event["error"]) == {"code": "findall_snapshot_too_large", "guidance": output["action"]}
    assert "narrower or smaller request" in output["action"] and "findall_id" in output["action"]
    assert output["findall_id"] == RUN_ID and output["snapshot_retained"] is False
    assert output["snapshot_bytes"] > output["maximum_snapshot_bytes"] == owner.MAX_SNAPSHOT_BYTES
    assert writes == [] and findall.READS_FIELD not in ledger.get(DAY)  # Nothing stored.
    # The provider run stays recorded with its whole reservation, so a later recovery can read it.
    assert next(iter(ledger.get(DAY)[owner.SUBMISSIONS_FIELD].values())) == entry
    assert findall_claims(ledger)[0]["reserved_micros"] == 1_000_000
    assert elapsed < 10 and client.reads.count(("result", RUN_ID)) == 1


def test_findall_tool_schemas_carry_no_pattern_keyword():
    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    definitions = findall.tools()
    assert all("pattern" not in set(keys(tool["parameters"])) for tool in definitions)
    result = next(tool for tool in definitions if tool["name"] == findall.RESULT)
    assert result["parameters"]["properties"]["receipt_sha256"] == {"type": "string"}
    create = next(tool for tool in definitions if tool["name"] == findall.CREATE)
    assert create["parameters"]["properties"]["maximum_cost_usd"] == {"type": "string"}  # Host-validated.


def test_instructions_recommend_small_first_requests_one_read_and_receipt_paging():
    text = findall.instructions()
    assert "match_limit of about 50" in text and "two base requests of 150" not in text and "$4.75" not in text
    assert "one base request of 50 matches has a listed estimate of $1.75" in text and "leaving $8.25" in text
    assert "Poll blueprint_findall_status until the run is no longer active before reading its result" in text
    assert "Read each completed run's result once" in text and "receipt_sha256 and page" in text
    assert "including non-matches, so it is much larger than match_limit suggests" in text
    assert "7 MiB retention limit is refused as findall_snapshot_too_large" in text
    execution.prepare_submission({**SPEC, "match_limit": findall.FIRST_MATCH_LIMIT}, operation_id="synthetic",
                                 maximum_cost_usd="1.75")  # The stated estimate is the host's own estimate.
    with pytest.raises(execution.FindAllError, match="below_estimated_cost"):
        execution.prepare_submission({**SPEC, "match_limit": findall.FIRST_MATCH_LIMIT}, operation_id="synthetic",
                                     maximum_cost_usd="1.74")
    descriptions = {tool["name"]: tool["description"] for tool in findall.tools()}
    assert "match_limit of about 50" in descriptions[findall.CREATE]
    assert "no longer active" in descriptions[findall.STATUS] and "once" in descriptions[findall.RESULT]
    assert "non-matches" in descriptions[findall.CREATE] and "non-matches" in descriptions[findall.RESULT]


# --- Read time bound: only the provider GET is alarmed ------------------------------------


def watch(timers, label, function):
    def wrapped(*args, **kwargs):
        timers.append((label, signal.getitimer(signal.ITIMER_REAL)[0]))
        return function(*args, **kwargs)
    return wrapped


@pytest.mark.parametrize("name", [findall.STATUS, findall.RESULT])
def test_reads_alarm_only_the_provider_get_and_every_store_call_runs_outside_it(runtime, monkeypatch, name):
    _row, ledger, client, _state, api = runtime
    timers = []
    for operation in ("get", "put", "write_bytes", "read_bytes"):
        monkeypatch.setattr(ledger, operation, watch(timers, operation, getattr(ledger, operation)))
    handler = api.findall_application_tools
    monkeypatch.setattr(handler, "assert_current_lease", watch(timers, "assert_lease", handler.assert_current_lease))
    monkeypatch.setattr(client, "_post", watch(timers, "post", client._post))
    monkeypatch.setattr(client, name.rsplit("_", 1)[-1], watch(timers, "get_provider", getattr(client, name.rsplit("_", 1)[-1])))
    started(runtime)
    posts = [seconds for label, seconds in timers if label == "post"]
    assert len(posts) == 1 and 0 < posts[0] <= findall.TOOL_BOUND_SECONDS  # A create keeps the 15 s bound.
    timers.clear()
    respond(runtime, action(name, "call_read", {"findall_id": RUN_ID}))
    assert reply(runtime)[0]["success"] is True
    gets = [seconds for label, seconds in timers if label == "get_provider"]
    assert len(gets) == 1 and findall.TOOL_BOUND_SECONDS < gets[0] <= findall.FINDALL_READ_BOUND_SECONDS
    stores = [(label, seconds) for label, seconds in timers if label != "get_provider"]
    assert {label for label, _ in stores} >= {"get", "put", "write_bytes", "read_bytes", "assert_lease"}
    assert all(seconds == 0.0 for _, seconds in stores), stores  # No store call ever runs under the alarm.
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_the_read_bound_is_the_remaining_phase_time_when_that_is_shorter(runtime, monkeypatch):
    _row, _ledger, client, state, _api = runtime
    started(runtime)
    timers = []
    monkeypatch.setattr(client, "result", watch(timers, "get_provider", client.result))
    state["now"] = NOW + timedelta(seconds=1200 - 30)  # 30 s of research remain.
    respond(runtime, action(findall.RESULT, "call_read", {"findall_id": RUN_ID}))
    assert reply(runtime)[0]["success"] is True
    assert len(timers) == 1 and 0 < timers[0][1] <= 30


def test_a_get_that_outlives_its_bound_fails_before_any_store_write(runtime, monkeypatch):
    _row, ledger, client, _state, _api = runtime
    started(runtime)
    bound, provider_result = findall.FINDALL_READ_BOUND_SECONDS, client.result
    monkeypatch.setattr(findall, "FINDALL_READ_BOUND_SECONDS", 0.2)

    def slow(value):
        client.reads.append(("result", value))
        time.sleep(5)  # The alarm interrupts this provider read.
        raise AssertionError("the read bound did not fire")

    monkeypatch.setattr(client, "result", slow)
    writes = findall_writes(ledger, monkeypatch)
    began = time.monotonic()
    respond(runtime, action(findall.RESULT, "call_slow", {"findall_id": RUN_ID}))
    assert time.monotonic() - began < 3
    event, _ = reply(runtime)
    assert event["success"] is False and json.loads(event["error"]) == {"code": "research_tool_absolute_deadline"}
    assert writes == [] and findall.READS_FIELD not in ledger.get(DAY)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    # The store and call journal stay usable: the next read is retained.
    monkeypatch.setattr(findall, "FINDALL_READ_BOUND_SECONDS", bound)
    monkeypatch.setattr(client, "result", provider_result)
    respond(runtime, action(findall.RESULT, "call_after", {"findall_id": RUN_ID}))
    assert reply(runtime)[0]["success"] is True and "call_after" in ledger.get(DAY)[findall.READS_FIELD]


# --- Settlement keeps the matches that were paid for -----------------------------------------


def test_settlement_reads_the_result_only_after_the_cancel_and_exports_it(runtime, monkeypatch):
    row, ledger, client, state, api = runtime
    started(runtime)
    order = []
    original_post, original_result = client._post, client.result
    monkeypatch.setattr(client, "_post", lambda url, body=None: order.append("cancel") or original_post(url, body))
    monkeypatch.setattr(client, "result", lambda value: order.append("result") or original_result(value))
    state["now"] = NOW + timedelta(seconds=1200)
    with ledger.lock():
        api.findall_application_tools.settle(row, reason="original_research_deadline")
    stored = ledger.get(DAY)
    record = next(iter(stored[findall.SETTLEMENTS_FIELD].values()))
    assert order == ["cancel", "result"] and record["state"] == "terminal"
    receipt = record["result_receipt"]
    assert receipt in record["receipts"] and "-tool-findall-settle-result-" in receipt["file"]
    assert json.loads(ledger.read_bytes(receipt["file"])) == original_result(RUN_ID)
    assert findall_claims(ledger)[0]["reserved_micros"] == 1_000_000  # No refund semantics.
    findall.validate_snapshot_exports(stored, ledger.read_bytes)
    assert {key: receipt[key] for key in ("file", "sha256", "bytes")} in findall.receipt_refs(stored)


def test_a_cancelled_run_that_still_reports_active_gets_exactly_one_result_read(runtime, monkeypatch):
    row, ledger, client, _state, api = runtime
    started(runtime)
    row.update(state="awaiting_review")
    ledger.put(row)
    original = client._post

    def accepted_but_active(url, body=None):
        value = original(url, body)
        client.active = True  # The provider accepted the cancel but still reports the run active.
        return value

    monkeypatch.setattr(client, "_post", accepted_but_active)
    with ledger.lock():
        for _ in range(findall.MAX_SETTLEMENT_PASSES + 1):
            api.findall_application_tools.settle(row, reason="research_ended")
    record = next(iter(ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))
    assert client.cancels == [RUN_ID] and record["state"] == "settlement_unconfirmed"
    assert client.reads.count(("result", RUN_ID)) == 1 and record["result_receipt"] in record["receipts"]


@pytest.mark.parametrize("failure", ["provider", "timeout", "too_large", "store"])
def test_a_failed_settlement_result_read_is_recorded_and_never_blocks_the_cancel(runtime, monkeypatch, failure):
    row, ledger, client, state, api = runtime
    started(runtime)
    state["now"] = NOW + timedelta(seconds=1200)
    if failure == "provider":
        def fail(value):
            raise execution.FindAllError("findall_transport_failed")
        monkeypatch.setattr(client, "result", fail)
        code = "findall_transport_failed"
    elif failure == "timeout":
        monkeypatch.setattr(findall, "FINDALL_READ_BOUND_SECONDS", 0.2)
        monkeypatch.setattr(client, "result", lambda value: time.sleep(5))
        code = "research_tool_absolute_deadline"
    elif failure == "too_large":
        monkeypatch.setattr(client, "result", lambda value: {"run": {"findall_id": value},
                                                             "candidates": [{"future": "x" * (8 * 1024 * 1024)}]})
        code = "findall_snapshot_too_large"
    else:
        original = ledger.write_bytes

        def refuse(name, raw):
            if "-settle-result-" in name:
                raise Refusal("firestore_blob_too_large")
            return original(name, raw)

        monkeypatch.setattr(ledger, "write_bytes", refuse)
        code = "firestore_blob_too_large"
    with ledger.lock():
        for _ in range(3):
            api.findall_application_tools.settle(row, reason="original_research_deadline")
    record = next(iter(ledger.get(DAY)[findall.SETTLEMENTS_FIELD].values()))
    assert client.cancels == [RUN_ID] and record["state"] == "terminal"  # The cancel path completed.
    assert record["result_observation_error"] == code and "result_receipt" not in record
    assert not any("-settle-result-" in ref["file"] for ref in record["receipts"])
    assert (record.get("result_snapshot_bytes", 0) > owner.MAX_SNAPSHOT_BYTES) is (failure == "too_large")
    assert findall_claims(ledger)[0]["reserved_micros"] == 1_000_000
    assert not findall.settlement_due(ledger.get(DAY), state["now"])


# --- The real store path: Python Bridge/FirestoreLedger and the JS Store ------------------------

ROOT = Path(__file__).resolve().parents[1]
# A Buffer-native hermetic Firestore driver with MemoryFirestore's transaction rules. The shared
# double JSON-clones every 256 KiB Buffer chunk as a number array (seconds per MB), which would
# measure the double instead of the Python pipe and JS Store code under test.
BUFFER_FIRESTORE = r"""
const clone = value => Buffer.isBuffer(value) ? Buffer.from(value) : Array.isArray(value) ? value.map(clone)
  : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value)
    .filter(([, item]) => item !== undefined).map(([key, item]) => [key, clone(item)])) : value;
class Snapshot {
  constructor(path, value) {this.ref = {path, id: path.split('/').at(-1)}; this.id = this.ref.id; this.value = value;
    this.exists = value !== undefined;}
  data() {return clone(this.value);}
}
export class BufferFirestore {
  constructor() {this.values = new Map(); this.tail = Promise.resolve();}
  doc(path) {return {path, id: path.split('/').at(-1), get: async () => new Snapshot(path, this.values.get(path))};}
  collection(path) {
    const depth = path.split('/').length + 1;
    const query = count => ({limit: next => query(next), get: async () => ({docs: [...this.values]
      .filter(([key]) => key.startsWith(path + '/') && key.split('/').length === depth).slice(0, count)
      .map(([key, value]) => new Snapshot(key, value))})});
    return query(Infinity);
  }
  runTransaction(fn) {
    const run = async () => {
      const writes = [];
      const result = await fn({get: async ref => {
        if (writes.length) throw new Error('read after write');
        return ref.get();
      }, set: (ref, value, options) => writes.push({ref, value: clone(value), merge: options?.merge})});
      for (const {ref, value, merge} of writes)
        this.values.set(ref.path, merge ? {...this.values.get(ref.path), ...value} : value);
      return result;
    };
    const result = this.tail.then(run); this.tail = result.catch(() => {}); return result;
  }
}
"""


@pytest.fixture
def real_store(tmp_path):
    """The real private pipe, LeaseChannel and JS Store; only Firestore is a hermetic double (fixed clock)."""
    driver = tmp_path / "buffer-firestore.mjs"
    driver.write_text(BUFFER_FIRESTORE)
    script = tmp_path / "bridge.mjs"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline';",
        "import {Store, LeaseChannel} from "
        + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {BufferFirestore} from " + json.dumps(driver.as_uri()) + ";",
        f"const channel = new LeaseChannel(new Store(new BufferFirestore(), () => {int(NOW.timestamp() * 1000)}));",
        "for await (const line of createInterface({input: process.stdin})) {",
        "try {const value = await channel.call(JSON.parse(line)); process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error) {process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}}",
        "await channel.close();",
    ]))
    bridge = Bridge(script=script)
    try:
        ledger = FirestoreLedger(bridge)
        bridge.call("init", value={"schema_version": "blueprint.research-control.v1", "enabled": False})
        with ledger.lock():
            bridge.call("configure", value={"schema_version": "blueprint.research-control.v1", "enabled": True,
                                            "project_id": PROJECT, "agent_id": AGENT, "source_commit": COMMIT})
            bridge.call("paid_expansion_set", expected_sha256=None,
                        value={"enabled": True, "current": owner_control()["paid_expansion"]["current"]})
            row = pinned_row(bridge.call("control"), state="creating")
            ledger.put(row)  # The real store admits the frozen grant at the durable intent.
            row["state"] = "running"
            ledger.put(row)
        replies = []

        def assert_lease():
            bridge.call("assert_lease")
            return True

        client = Client(ledger)
        handler = findall.FindAllApplicationTools(ledger=ledger, client=client, control=lambda: bridge.call("control"),
                                                  assert_current_lease=assert_lease, clock=lambda: NOW)
        api = SimpleNamespace(findall_application_tools=handler, tool_admit=lambda row, phase: None,
                              tool_result=lambda sid, event, key: replies.append(copy.deepcopy(event)))
        yield SimpleNamespace(row=row, ledger=ledger, client=client, replies=replies, api=api, bridge=bridge)
    finally:
        bridge.close()


def respond_through(store, pending):
    with store.ledger.lock():
        search.respond(store.row, {"required_actions": [pending]}, store.ledger, store.api,
                       phase="research", clock=lambda: NOW)
    return store.replies[-1]


def synthetic_result(megabytes):
    """A FindAll-shaped result listing every evaluated candidate, mostly non-matches. Hex reasoning
    compresses poorly, so the store writes many gzip chunks for the one file."""
    candidates, size = [], 0
    while size < megabytes * 1_000_000:
        index = len(candidates)
        reasoning = "".join(hashlib.sha256(f"{index}:{part}".encode()).hexdigest() for part in range(24))
        candidate = {"candidate_id": f"candidate_{index}", "name": f"Operating site {index}",
                     "url": f"https://operator.example/sites/{index}",
                     "match_status": "matched" if index % 7 == 0 else "unmatched",
                     "basis": [{"field": "task", "reasoning": reasoning, "citations": [
                         {"url": f"https://evidence.example/{index}", "excerpts": [reasoning[:640]]}]}]}
        candidates.append(candidate)
        size += len(json.dumps(candidate))
    return {"run": {"findall_id": RUN_ID, "status": {"status": "completed", "is_active": False}},
            "candidates": candidates, "last_event_id": "event_synthetic"}


@pytest.mark.parametrize("megabytes", [2, 6])
def test_large_results_retain_through_the_real_store_in_one_write_within_the_bound(
        real_store, monkeypatch, tmp_path, megabytes):
    store = real_store
    assert respond_through(store, action())["success"] is True
    store.row.update(store.ledger.get(DAY))
    snapshot = synthetic_result(megabytes)
    monkeypatch.setattr(store.client, "result", lambda value: store.client.reads.append(("result", value)) or snapshot)
    ops, alarmed = [], []
    call = store.bridge.call

    def counted(op, **fields):
        began = time.monotonic()
        try:
            return call(op, **fields)
        finally:
            ops.append((op, fields.get("name"), time.monotonic() - began))

    monkeypatch.setattr(store.bridge, "call", counted)
    handler = store.api.findall_application_tools
    provider_read = handler.provider_read

    def timed(*args):
        began = time.monotonic()
        try:
            return provider_read(*args)
        finally:
            alarmed.append(time.monotonic() - began)

    monkeypatch.setattr(handler, "provider_read", timed)
    began = time.monotonic()
    event = respond_through(store, action(findall.RESULT, "call_result", {"findall_id": RUN_ID}))
    elapsed, during = time.monotonic() - began, list(ops)
    first = json.loads(event["output"])
    receipt = store.ledger.get(DAY)[findall.READS_FIELD]["call_result"]
    raw = store.ledger.read_bytes(receipt["file"])
    text = raw.decode("utf-8")
    assert event["success"] is True and first["page"] == 0 and first["page_count"] == receipt["page_count"] > 1
    assert raw == owner._json_bytes(snapshot) and first["json_fragment"] == text[:owner.SNAPSHOT_FRAGMENT_CHARS]
    # ONE immutable file: one write and one readback; the first page needs no further store read.
    assert [name for op, name, _ in during if op == "file_put" and "-tool-findall-" in name] == [receipt["file"]]
    assert [op for op, name, _ in during if name == receipt["file"]] == ["file_put", "file_get"]
    assert len(alarmed) == 1 and alarmed[0] < elapsed < findall.FINDALL_READ_BOUND_SECONDS
    assert store.bridge.broken is False
    counts = dict(sorted(Counter(op for op, _, _ in during).items()))
    print(f"\nreal store {megabytes} MB result: {len(raw)} bytes, {receipt['page_count']} pages; tool call "
          f"{elapsed:.2f}s, alarmed GET+encode {alarmed[0]:.2f}s; {len(during)} store calls "
          f"({sum(seconds for *_, seconds in during):.2f}s) {counts}; snapshot file_put "
          f"{next(s for op, name, s in during if op == 'file_put' and name == receipt['file']):.2f}s")
    ops.clear()
    last = receipt["page_count"] - 1
    event = respond_through(store, action(findall.RESULT, "call_last_page", {
        "findall_id": RUN_ID, "receipt_sha256": receipt["sha256"], "page": last}))
    shown = json.loads(event["output"])
    assert event["success"] is True and shown["next_page"] is None
    assert shown["json_fragment"] == text[last * owner.SNAPSHOT_FRAGMENT_CHARS:]
    assert [op for op, name, _ in ops if name == receipt["file"]] == ["file_get"]  # Same file, no new snapshot.
    assert store.client.reads.count(("result", RUN_ID)) == 1
    render.export_snapshot(store.bridge, DAY, tmp_path / "export")  # The JS export and Python validation.
    assert (tmp_path / "export" / receipt["file"]).read_bytes() == raw
    assert not any("-part-" in path.name for path in (tmp_path / "export").iterdir())


def test_a_get_timeout_never_breaks_the_real_store_bridge(real_store, monkeypatch):
    store = real_store
    assert respond_through(store, action())["success"] is True
    store.row.update(store.ledger.get(DAY))
    monkeypatch.setattr(findall, "FINDALL_READ_BOUND_SECONDS", 0.3)
    monkeypatch.setattr(store.client, "result", lambda value: time.sleep(5))
    event = respond_through(store, action(findall.RESULT, "call_slow", {"findall_id": RUN_ID}))
    assert event["success"] is False and json.loads(event["error"]) == {"code": "research_tool_absolute_deadline"}
    assert store.bridge.broken is False  # The alarm fired in the GET, before any store write.
    assert findall.READS_FIELD not in store.ledger.get(DAY)
    assert store.ledger.get(DAY)["application_tool_calls"]["call_slow"]["success"] is False
