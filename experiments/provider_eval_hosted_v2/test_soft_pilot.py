"""Runnable pilot admission, stop placement and replay, with no network calls."""

from decimal import Decimal
import json
from pathlib import Path

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError
from experiments.provider_eval_recovery.harness import Ledger, digest, write_once
from experiments.provider_eval_recovery.live_http import MODEL, PROJECT
from experiments.provider_eval_recovery.public_inputs import DECLARED_ORIGINAL_SHA256
from . import soft_pilot as soft


class Clock:
    def __init__(self): self.now = 1000
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(soft, "code_identity", lambda: ("a" * 40, "c" * 64))
    access = json.loads((Path(__file__).parents[1] / "provider_eval_recovery/ACCESS_RECEIPT.example.json").read_text())
    access["journal_root"] = str(tmp_path)
    write_once(tmp_path / "live_access.json", access)
    write_once(tmp_path / "live_scope.json", {"budget_usd": "10.00", "journal_root": str(tmp_path),
        "model": MODEL, "public_inputs_sha256": DECLARED_ORIGINAL_SHA256})
    ledger = Ledger(tmp_path / "live_journal.jsonl", "10.00")
    ledger.append("reserved", "prior", amount_usd="3.934105")
    ledger.append("completed", "prior", raw_sha256="prior_proof")
    clock = Clock()
    receipt, sha = soft.prepare_receipt(tmp_path, soft.OWNER, clock=clock)
    return tmp_path, receipt, sha, clock


class API:
    project_id = PROJECT
    def __init__(self, mode, clock, *, usage="known", uncertain=False, ignore_cancel=False, pending=False):
        self.mode, self.clock, self.usage = mode, clock, usage
        self.uncertain, self.ignore_cancel = uncertain, ignore_cancel
        self.pending = pending
        self.calls, self.session, self.cancelled = [], None, False
    def request(self, method, path, *, body=None, query=None):
        self.calls.append((method, path, body))
        if path == "/agents/sessions":
            if method == "GET":
                return {"data": [self.session] if self.session else [], "has_more": False}
            self.session = {"id": "session_" + self.mode, "agent": body["agent"], "metadata": body["metadata"],
                "environment": {"type": "openai_hosted", "container_size": "small", "network": {"access": "disabled"},
                                "id": "env_" + self.mode}, "usage": None}
            if self.uncertain:
                raise AgentTransportError("agents_api_connection_uncertain")
            return self.session
        if path.endswith("/events"):
            event = body["events"][0]
            if event["type"] == "agent.session.input.cancel":
                self.cancelled = not self.ignore_cancel
                return {}
            raise AssertionError("no function reply authorized in this fixture")
        usage = {"input_tokens": 100, "output_tokens": 10} if self.usage == "known" else self.usage
        if path == "/agents/sessions/" + self.session["id"]:
            self.session["usage"] = usage
            actions = [] if self.usage == "known" or self.cancelled else [{"type": "function_call", "turn_id": "turn_1",
                "call_id": "unfunded_search", "name": "search", "arguments": {"query": "Chef Robotics deployment"}}]
            return {**self.session, "status": "idle" if not actions else "in_progress", "required_actions": actions}
        if path.endswith("/turns"):
            status = "cancelled" if self.cancelled else ("completed" if self.usage == "known" and not self.pending else "in_progress")
            return {"data": [{"id": "turn_1", "session_id": self.session["id"], "subagent_id": None,
                "status": status, "usage": usage, "completed_at": self.clock()}], "has_more": False}
        if path.endswith("/items"):
            return {"data": [{"id": "answer_1", "type": "message", "turn_id": "turn_1", "role": "assistant",
                "phase": "final_answer", "status": "completed", "content": [{"type": "output_text",
                "text": json.dumps({"answer": "Fixture unknown.", "citations": [], "unknowns": ["site fit"]})}]}], "has_more": False}
        raise AssertionError((method, path))


def factory_for(clock, **api_options):
    apis = []
    def factory(root, receipt, monitor, mode):
        api = API(mode, clock, **api_options)
        apis.append(api)
        return soft.make_runtime(root, receipt, monitor, mode, transport=api)
    return factory, apis


def test_exact_receipt_preserves_old_journal_and_rejects_loose_overrides(admitted):
    root, receipt, sha, clock = admitted
    before = (root / "live_journal.jsonl").read_bytes()
    assert soft.validate_receipt(root, soft.OWNER, sha) == receipt
    assert soft.prepare_receipt(root, soft.OWNER, clock=clock)[1] == sha
    assert (root / "live_journal.jsonl").read_bytes() == before
    changed = {**receipt, "soft_target_usd": "100.00"}
    (root / "protocols" / soft.PROTOCOL / "soft_pilot_approval.json").write_text(json.dumps(changed))
    with pytest.raises(AgentExecutionError, match="exact_reviewed"):
        soft.validate_receipt(root, soft.OWNER, digest(changed))


def test_missing_prior_journal_cannot_issue_zero_baseline_receipt(admitted):
    root, _, _, clock = admitted
    (root / "protocols" / soft.PROTOCOL / "soft_pilot_approval.json").unlink()
    (root / "live_journal.jsonl").unlink()
    with pytest.raises(AgentExecutionError, match="prior_journal"):
        soft.prepare_receipt(root, soft.OWNER, clock=clock)
    assert not (root / "live_journal.jsonl").exists()


@pytest.mark.parametrize("support,probe", [(False, True), ("unverified", False)])
def test_parallel_header_boundary_is_preserved(admitted, support, probe):
    root, _, _, clock = admitted
    access = json.loads((root / "live_access.json").read_text())
    access.update(parallel_x_api_key_supported=support, parallel_header_pilot_probe_authorized=probe)
    (root / "live_access.json").write_text(json.dumps(access))
    with pytest.raises(AgentExecutionError, match="existing_frozen_scope_access"):
        soft.prepare_receipt(root, soft.OWNER, clock=clock)


def test_four_mode_cli_core_finishes_durably_and_never_recreates_sessions(admitted):
    root, receipt, _, clock = admitted
    factory, apis = factory_for(clock)
    old_prefix = (root / "live_journal.jsonl").read_bytes()
    result = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert [o["mode"] for o in result["outcomes"]] == list(soft.MODES)
    assert all(o["state"]["state"] == "completed" for o in result["outcomes"])
    assert not result["budget"]["stopped"]
    assert len(result["budget"]["sessions"]) == 4
    assert result["budget"]["aggregate_reserved_usd"] == "4.569585"
    assert result["budget"]["projected_with_remaining_search_opportunity_usd"] == "0.708980"
    assert (root / "live_journal.jsonl").read_bytes().startswith(old_prefix)
    assert sum(m == "POST" and p == "/agents/sessions" for a in apis for m, p, _ in a.calls) == 4
    # A restart uses the four original durable task/result records, no creation.
    again = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert len(again["outcomes"]) == 4
    assert sum(m == "POST" and p == "/agents/sessions" for a in apis for m, p, _ in a.calls) == 4


@pytest.mark.parametrize("usage", [None, {"input_tokens": 500000, "output_tokens": 0}])
def test_fresh_usage_stops_before_tool_dispatch_or_result_reply(admitted, monkeypatch, usage):
    root, receipt, _, clock = admitted
    # No provider key may be read: guard must act before the paid route.
    import importlib
    route_module = importlib.import_module(soft.ExistingSearchRoute.__module__)
    monkeypatch.setattr(route_module, "existing_key", lambda *_: (_ for _ in ()).throw(AssertionError("provider key read")))
    factory, apis = factory_for(clock, usage=usage)
    result = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["budget"]["stopped"] and len(apis) == 1
    assert apis[0].cancelled
    assert not any(e.get("role") == soft.PROTOCOL + ":search" for e in Ledger(root / "live_journal.jsonl", "10.00").events)
    assert not any(body and any(e.get("type") == "agent.session.input.tool_result" for e in body.get("events", [])) for _, _, body in apis[0].calls)


def test_creation_uncertainty_keeps_hold_and_never_redispatches(admitted):
    root, receipt, _, clock = admitted
    factory, apis = factory_for(clock, uncertain=True)
    result = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["budget"]["stopped"] and len(apis) == 1
    assert sum(m == "POST" and p == "/agents/sessions" for m, p, _ in apis[0].calls) == 1
    soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert len(apis) == 1
    assert Ledger(root / "live_journal.jsonl", "10.00").exposure == Decimal("4.569585")


def test_container_carry_continues_after_completion_and_restart(admitted):
    root, receipt, _, clock = admitted
    factory, _ = factory_for(clock)
    soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    clock.now += 4801
    monitor = soft.SoftMonitor(root, receipt, clock=clock, notify=lambda _: None)
    report = monitor.report()
    assert report["stopped"]  # > one hour carry and approval expired soon
    assert all(s["container_elapsed_estimate_usd"] == "0.15" for s in report["sessions"])
    assert all(s["ongoing_container_cost_unreconciled"] for s in report["sessions"])
    assert monitor.report()["aggregate_reserved_usd"] == report["aggregate_reserved_usd"]


def test_overrun_is_reported_even_when_nominal_ten_ceiling_cannot_reserve_it(admitted):
    root, receipt, _, clock = admitted
    factory, _ = factory_for(clock, usage={"input_tokens": 2500000, "output_tokens": 0})
    result = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert Decimal(result["budget"]["unreserved_observed_overrun_usd"]) > 10
    assert result["budget"]["stop"]["reason"] == "observed_overrun_beyond_original_aggregate_ceiling"
    assert Ledger(root / "live_journal.jsonl", "10.00").exposure == Decimal("4.569585")


@pytest.mark.parametrize("tamper", ["reset_time", "delete_start"])
def test_mutable_creation_sidecar_cannot_reset_or_hide_retained_container(admitted, tamper):
    root, receipt, _, clock = admitted
    factory, _ = factory_for(clock)
    soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    clock.now += 4801
    monitor = soft.SoftMonitor(root, receipt, clock=clock, notify=lambda _: None)
    for mode in soft.MODES:
        path = monitor.path / monitor.task_id(mode) / "started.json"
        if tamper == "delete_start":
            path.unlink()
        else:
            intent = json.loads(path.read_text())
            intent["at"] = clock.now
            path.write_text(json.dumps(intent))
    report = monitor.report()
    assert report["stopped"] and len(report["sessions"]) == 4
    assert all(s["session_id"] and s["container_elapsed_estimate_usd"] == "0.15" for s in report["sessions"])
    assert report["stop"]["reason"] == "durable_creation_proof_missing_or_changed"


def test_receipt_cannot_absorb_existing_hosted_spend_into_new_baseline(admitted):
    root, receipt, _, clock = admitted
    factory, _ = factory_for(clock)
    soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    changed = {**receipt, "baseline_events": len(ledger.events), "baseline_head": ledger.previous,
               "baseline_reserved_usd": str(ledger.exposure), "created_at": clock.now + 20, "expires_at": clock.now + 3620}
    (root / "protocols" / soft.PROTOCOL / "soft_pilot_approval.json").write_text(json.dumps(changed))
    with pytest.raises(AgentExecutionError, match="original_journal_prefix_changed"):
        soft.validate_receipt(root, soft.OWNER, digest(changed))


def test_deadline_cancel_is_bounded_and_no_next_arm_starts(admitted):
    root, receipt, _, clock = admitted
    factory, apis = factory_for(clock, pending=True, ignore_cancel=True)
    result = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["budget"]["stopped"] and len(apis) == 1
    assert clock.now <= 1000 + soft.ARM_SECONDS + 3 * soft.POLL_SECONDS
    assert any(body and body.get("events", [{}])[0].get("type") == "agent.session.input.cancel" for _, _, body in apis[0].calls)
    assert result["outcomes"][0]["state"]["cancel_requested"]


def test_pinned_public_sdk_read_only_wire_uses_default_project_and_agents_beta():
    import httpx2
    from openai import OpenAI
    seen = []
    def response(request):
        seen.append(request)
        return httpx2.Response(200, json={"data": [], "has_more": False, "object": "list"})
    with OpenAI(api_key="MOCK_SDK_ONLY", project=PROJECT, max_retries=0, timeout=20,
        default_headers={"OpenAI-Beta": "agents=v1"}, http_client=httpx2.Client(transport=httpx2.MockTransport(response))) as client:
        assert client.beta.agents.sessions.list(limit=1).data == []
    assert len(seen) == 1 and seen[0].method == "GET"
    assert seen[0].url.host == "api.openai.com" and seen[0].url.path == "/v1/agents/sessions"
    assert seen[0].headers["OpenAI-Project"] == PROJECT and seen[0].headers["OpenAI-Beta"] == "agents=v1"


def test_canonical_paid_model_gate_refusal_prevents_session_dispatch(admitted, monkeypatch):
    root, receipt, _, clock = admitted
    def refuse(*args, **kwargs):
        raise AgentExecutionError("canonical_paid_model_grant_rejected")
    monkeypatch.setattr(soft, "require_paid_resource_admission_grant", refuse)
    factory, apis = factory_for(clock)
    result = soft.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["budget"]["stopped"] and len(apis) == 1
    assert not any(m == "POST" and p == "/agents/sessions" for m, p, _ in apis[0].calls)


def test_previous_receipt_is_readonly_not_a_paid_scope_migration(admitted):
    root, receipt, _, _ = admitted
    previous = {**receipt, "source_commit": soft.PREVIOUS_SOURCE_IDENTITY[0],
                "code_sha256": soft.PREVIOUS_SOURCE_IDENTITY[1]}
    (root / "protocols" / soft.PROTOCOL / "soft_pilot_approval.json").write_text(json.dumps(previous))
    assert soft.validate_receipt(root, soft.OWNER, digest(previous), readonly_previous=True) == previous
    with pytest.raises(AgentExecutionError, match="exact_reviewed"):
        soft.validate_receipt(root, soft.OWNER, digest(previous))


@pytest.fixture
def unresolved_stopped(admitted):
    root, receipt, _, clock = admitted
    monitor = soft.SoftMonitor(root, receipt, clock=clock)
    monitor.reserve()
    api = API("parallel_fast", clock, usage={"input_tokens": 49484, "output_tokens": 261})
    original = api.request
    def request(method, path, **kwargs):
        value = original(method, path, **kwargs)
        if method == "POST" and path == "/agents/sessions":
            value["environment"]["network"]["allowed_domains"] = []
        return value
    api.request = request
    runtime, task = soft.make_runtime(root, receipt, monitor, "parallel_fast", transport=api)
    runtime._validate_session = lambda *_: (_ for _ in ()).throw(AgentExecutionError("old_exact_policy_comparison"))
    with pytest.raises(AgentExecutionError, match="old_exact"):
        runtime.start(task)
    monitor.stop("pilot_interrupted_no_additional_paid_dispatch")
    api.cancelled = True  # Execution owner already cancelled remotely.
    assert runtime.journal.task(task.task_id)["state"] == "creation_unresolved"
    api.calls.clear()
    return root, receipt, clock, api, task


def test_getonly_reconciliation_binds_old_id_preserves_receipt_and_cost(unresolved_stopped):
    from .reconcile import reconcile_session
    root, receipt, clock, api, task = unresolved_stopped
    receipt_bytes = (root / "protocols" / soft.PROTOCOL / "soft_pilot_approval.json").read_bytes()
    original_prefix = (root / "live_journal.jsonl").read_bytes()
    result = reconcile_session(root, receipt, api.session["id"], transport=api, clock=clock)
    assert result["local_state"] == "cancelled" and result["cancellation_settled"]
    assert result["budget"]["aggregate_reserved_usd"] == "4.765920"
    assert result["budget"]["projected_with_remaining_search_opportunity_usd"] == "0.905315"
    assert result["budget"]["stopped"] and result["new_paid_calls"] == 0
    assert all(method == "GET" for method, _, _ in api.calls)
    assert (root / "live_journal.jsonl").read_bytes().startswith(original_prefix)
    assert (root / "protocols" / soft.PROTOCOL / "soft_pilot_approval.json").read_bytes() == receipt_bytes
    retained = json.loads((Path(result["raw_evidence_directory"]) / "snapshot.json").read_text())
    assert retained["original_state"]["state"] == "creation_unresolved"
    assert retained["session"]["environment"]["network"]["allowed_domains"] == []
    held = result["budget"]["aggregate_reserved_usd"]
    again = reconcile_session(root, receipt, api.session["id"], transport=api, clock=clock)
    assert again["budget"]["aggregate_reserved_usd"] == held
    assert all(method == "GET" for method, _, _ in api.calls)
    assert len(soft.AgentJournal(soft.SoftMonitor(root, receipt).path / "agent_journal").tasks(active_only=False)) == 1


@pytest.mark.parametrize("tamper", ["wrong_id", "enabled_network", "nonempty_domains", "wrong_metadata", "lost_start"])
def test_reconciliation_invalid_identity_or_policy_never_binds(unresolved_stopped, tamper):
    from .reconcile import reconcile_session
    root, receipt, clock, api, task = unresolved_stopped
    session_id = api.session["id"]
    if tamper == "wrong_id":
        session_id = "other_session"
    elif tamper == "enabled_network":
        api.session["environment"]["network"]["access"] = "enabled"
    elif tamper == "nonempty_domains":
        api.session["environment"]["network"]["allowed_domains"] = ["example.com"]
    elif tamper == "wrong_metadata":
        api.session["metadata"]["blueprint_task_digest"] = "wrong"
    else:
        (soft.SoftMonitor(root, receipt).path / task.task_id / "started.json").unlink()
    with pytest.raises(AgentExecutionError):
        reconcile_session(root, receipt, session_id, transport=api, clock=clock)
    state = soft.AgentJournal(soft.SoftMonitor(root, receipt).path / "agent_journal").task(task.task_id)
    assert state["session_id"] is None and state["state"] == "creation_unresolved"
    assert all(method == "GET" for method, _, _ in api.calls)


def test_remote_idle_alone_does_not_prove_cancellation(unresolved_stopped):
    from .reconcile import reconcile_session
    root, receipt, clock, api, _ = unresolved_stopped
    original = api.request
    def no_turns(method, path, **kwargs):
        response = original(method, path, **kwargs)
        if path.endswith("/turns"):
            return {"data": [], "has_more": False}
        return response
    api.request = no_turns
    result = reconcile_session(root, receipt, api.session["id"], transport=api, clock=clock)
    assert not result["cancellation_settled"] and result["local_state"] == "cancelling"
    assert all(method == "GET" for method, _, _ in api.calls)


@pytest.mark.parametrize("pending", ["earlier_root", "required_action"])
def test_any_active_remote_work_prevents_cancellation_settlement(unresolved_stopped, pending):
    from .reconcile import reconcile_session
    root, receipt, clock, api, _ = unresolved_stopped
    original = api.request
    def active(method, path, **kwargs):
        response = original(method, path, **kwargs)
        if pending == "earlier_root" and path.endswith("/turns"):
            response["data"].insert(0, {**response["data"][0], "id": "earlier", "status": "in_progress"})
        elif pending == "required_action" and path == "/agents/sessions/" + api.session["id"]:
            response["required_actions"] = [{"type": "function_call", "call_id": "pending"}]
        return response
    api.request = active
    result = reconcile_session(root, receipt, api.session["id"], transport=api, clock=clock)
    assert not result["cancellation_settled"] and result["local_state"] == "reconciling"
    assert all(method == "GET" for method, _, _ in api.calls)


@pytest.mark.parametrize("fresh", ["active", "missing"])
def test_local_cancelled_cannot_override_fresh_remote_uncertainty(unresolved_stopped, fresh):
    from .reconcile import reconcile_session
    root, receipt, clock, api, _ = unresolved_stopped
    assert reconcile_session(root, receipt, api.session["id"], transport=api, clock=clock)["cancellation_settled"]
    original = api.request
    def response(method, path, **kwargs):
        value = original(method, path, **kwargs)
        if path.endswith("/turns"):
            if fresh == "active":
                value["data"][0]["status"] = "in_progress"
            else:
                value["data"] = []
        return value
    api.request = response
    result = reconcile_session(root, receipt, api.session["id"], transport=api, clock=clock)
    assert result["local_state"] == "cancelled" and not result["cancellation_settled"]
    assert all(method == "GET" for method, _, _ in api.calls)


def test_readonly_transport_blocks_every_mutation_and_other_session():
    from .reconcile import ReadOnlySession
    api = API("parallel_fast", Clock())
    readonly = ReadOnlySession(api, "session_parallel_fast")
    for method, path in (("POST", "/agents/sessions"), ("POST", "/agents/sessions/session_parallel_fast/events"),
                         ("DELETE", "/agents/sessions/session_parallel_fast"), ("GET", "/agents/sessions/other")):
        with pytest.raises(AgentExecutionError, match="mutations_and_other"):
            readonly.request(method, path)
    assert api.calls == []
