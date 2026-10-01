"""Reporting lag and same-session resumption; all HTTP is mocked."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from . import soft_pilot as base
from . import retry_pilot as retry
from . import usage_recovery as recovery
from .test_retry_pilot import ready_retry  # noqa: F401 - fixture registration
from .test_soft_pilot import admitted, API, Clock, factory_for  # noqa: F401 - fixture registration

RETAINED = json.loads((Path(__file__).parent / "fixtures/usage_lag_retained.json").read_text())


def test_actual_retained_usage_observations_settle_within_grace(admitted):  # noqa: F811
    root, receipt, _, clock = admitted
    rows = RETAINED["observations"]
    clock.now = rows[0]["at"]
    # The test's fake authority/task is retimed/rebound; the retained records
    # themselves stay exact. No missing wire body/status/network is invented.
    receipt = {**receipt, "created_at": clock(), "expires_at": clock() + 3600}
    (root / "protocols" / base.PROTOCOL / "soft_pilot_approval.json").write_text(json.dumps(receipt))
    monitor = base.SoftMonitor(root, receipt, clock=clock, sleep=clock.sleep)
    monitor.reserve()
    runtime, task = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=API("parallel_fast", clock))
    runtime.start(task)
    for row in rows:
        clock.now = row["at"]
        known = monitor.observe(task.task_id, {"id": row["session_id"], "usage": row["usage"],
            "environment": {"id": row["environment_id"]}})
        assert known == (row["usage"] is not None)
        actual = max(monitor.observations(), key=lambda o: o["at"])
        assert actual["at"] == row["at"] and actual["usage"] == row["usage"]
        assert actual["model_estimate_usd"] == row["model_estimate_usd"]
        assert not monitor.report()["stopped"]
        if not known:
            assert monitor.usage_deadline(task.task_id) == rows[0]["at"] + 30
    assert rows[-1]["at"] - rows[0]["at"] < 15
    assert RETAINED["provenance"]["initial_full_session_bodies_retained"] is False


def test_actual_later_session_projection_and_turn_defaults_are_accepted(admitted):  # noqa: F811
    root, receipt, _, clock = admitted
    monitor = base.SoftMonitor(root, receipt, clock=clock)
    runtime, _ = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=API("parallel_fast", clock))
    # Validate the included exact owned binding only; the projection does not
    # contain original instructions/input/full task and cannot prove those bytes.
    binding = SimpleNamespace(parent_task_id=None, model="gpt-6.1-sol",
        task_id="hosted_retry1_01_parallel_fast_bb3a1e8b8e1b96a9",
        task_digest="sha256:bbb805b0ccec703986ab036333a8753aa9af8fda0683d49ebfa67b68786eb2d9",
        run_id="hosted_agent_research_v2_retry1")
    assert runtime._validate_session(binding, RETAINED["session"]) == recovery.SESSION
    class Projection:
        project_id = base.PROJECT
        def request(self, method, path, **kwargs):
            assert method == "GET" and path == "/agents/sessions/" + recovery.SESSION + "/turns"
            return RETAINED["turns"]
    runtime.transport = Projection()
    roots = runtime._root_turns(recovery.SESSION)
    assert roots == RETAINED["turns"]["data"]
    assert roots[0]["status"] == "cancelled" and roots[0]["subagent_id"] is None
    assert RETAINED["session"]["required_actions"] == []
    assert "turn_id" not in RETAINED["session"] and "latest_turn" not in RETAINED["session"]


class LagAPI(API):
    def request(self, method, path, **kwargs):
        if method == "POST" and path == "/agents/sessions":
            self.created = self.clock()
        if path == "/agents/sessions/" + (self.session or {}).get("id", ""):
            self.usage = None if self.clock() - self.created < 14.732338 else "known"
        return super().request(method, path, **kwargs)


def test_initial_missing_usage_waits_readonly_then_finishes(admitted):  # noqa: F811
    root, receipt, _, clock = admitted
    apis = []
    def factory(root, receipt, monitor, mode):
        api = LagAPI(mode, clock)
        apis.append(api)
        return base.make_runtime(root, receipt, monitor, mode, transport=api)
    result = base.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert len(result["outcomes"]) == 4 and not result["budget"]["stopped"]
    assert all(not a.cancelled for a in apis)
    assert all(sum(m == "POST" and p == "/agents/sessions" for m, p, _ in a.calls) == 1 for a in apis)
    assert all(len([1 for m, p, _ in a.calls if m == "GET" and p == "/agents/sessions/" + a.session["id"]]) == 4 for a in apis)


def test_persistently_missing_usage_has_one_durable_grace_and_no_paid_tools(admitted):  # noqa: F811
    root, receipt, _, clock = admitted
    factory, apis = factory_for(clock, usage=None)
    result = base.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["budget"]["stop"]["reason"] == "hosted_usage_reporting_grace_expired"
    assert clock() == 1030 and apis[0].cancelled and len(apis) == 1
    monitor = base.SoftMonitor(root, receipt, clock=clock, sleep=clock.sleep)
    assert monitor.usage_deadline(monitor.task_id("parallel_fast")) == 1030
    again = base.run_pilot(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert again["budget"]["stopped"] and len(apis) == 1 and clock() == 1030


@pytest.mark.parametrize("bad", [{"input_tokens": -1, "output_tokens": 0},
    {"input_tokens": True, "output_tokens": 0}, {"input_tokens": "100", "output_tokens": 10},
    {"input_tokens": 99, "output_tokens": 10}, {"input_tokens": 100, "output_tokens": 9}])
def test_malformed_and_nonmonotonic_usage_refused(admitted, bad):  # noqa: F811
    root, receipt, _, clock = admitted
    monitor = base.SoftMonitor(root, receipt, clock=clock)
    task_id = monitor.task_id("parallel_fast")
    # Actual durable owned task/creation proof rather than an unbound row.
    monitor.reserve()
    runtime, task = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=API("parallel_fast", clock))
    runtime.start(task)
    monitor.observe(task_id, {"id": runtime.journal.task(task_id)["session_id"], "usage": {"input_tokens": 100, "output_tokens": 10}})
    with pytest.raises(AgentExecutionError, match="malformed_or_nonmonotonic"):
        monitor.observe(task_id, {"usage": bad})


def test_stale_usage_cannot_admit_paid_work_and_unchanged_fresh_usage_is_valid(admitted):  # noqa: F811
    root, receipt, _, clock = admitted
    monitor = base.SoftMonitor(root, receipt, clock=clock)
    monitor.reserve()
    api = API("parallel_fast", clock)
    runtime, task = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=api)
    runtime.start(task)
    known = {"id": api.session["id"], "usage": {"input_tokens": 100, "output_tokens": 10}}
    monitor.observe(task.task_id, known)
    clock.sleep(31)
    with pytest.raises(AgentExecutionError, match="fresh_hosted_usage_required"):
        monitor.require_fresh_usage()
    monitor.observe(task.task_id, known)
    monitor.require_fresh_usage()
    clock.now -= 1
    with pytest.raises(AgentExecutionError, match="clock_regressed"):
        monitor.observe(task.task_id, known)


class ResumeAPI(API):
    def __init__(self, clock):
        super().__init__("parallel_fast", clock, usage="known")
        self.followup = None
    def request(self, method, path, *, body=None, query=None):
        if method == "POST" and path.endswith("/events") and body["events"][0]["type"] == "agent.session.input.message":
            self.calls.append((method, path, body))
            self.followup = body["events"][0]["input"]
            self.cancelled = False
            return {}
        value = super().request(method, path, body=body, query=query)
        if method == "POST" and path == "/agents/sessions":
            value["id"] = recovery.SESSION
        if path.endswith("/turns"):
            first = {**value["data"][0], "id": recovery.TURN, "status": "cancelled"}
            value["data"] = [first]
            if self.followup:
                value["data"].append({**first, "id": "turn_resume1", "status": "completed"})
        if path.endswith("/items") and self.followup:
            value["data"] = [{**value["data"][0], "turn_id": "turn_resume1"},
                {"id": "saved_followup", "type": "message", "role": "user", "status": "completed",
                 "turn_id": "turn_resume1", "content": [p for m in self.followup for p in m["content"]]}]
        if path == "/agents/sessions/" + recovery.SESSION:
            value["usage"] = {"input_tokens": 8594 if not self.followup else 15000, "output_tokens": 84}
        return value


@pytest.fixture
def cancelled_retry(ready_retry, monkeypatch):  # noqa: F811
    root, receipt, _, clock, *_ = ready_retry
    old_identity = ("7394bc57f285060fd7e3a96f2a4390a18ffe69ea", "39c6f96c0564f1228dc82cb6e4dcf313acd78035974261470d34c03da37ac77c")
    monkeypatch.setattr(base, "code_identity", lambda: old_identity)
    receipt = {**receipt, "source_commit": old_identity[0], "code_sha256": old_identity[1]}
    retry.receipt_path(root).write_text(json.dumps(receipt))
    monitor = retry.RetryMonitor(root, receipt, clock=clock, sleep=clock.sleep)
    monitor.reserve()
    api = ResumeAPI(clock)
    runtime, task = base.make_runtime(root, receipt, monitor, "parallel_fast", transport=api)
    runtime.start(task)
    runtime.cancel(task.task_id)
    runtime.step(task.task_id)
    assert runtime.journal.task(task.task_id)["state"] == "cancelled"
    monitor.stop(recovery.LAG_STOP)
    before = runtime.journal.task(task.task_id)
    prefix = (root / "live_journal.jsonl").read_bytes()
    monkeypatch.setattr(base, "code_identity", lambda: ("b" * 40, "d" * 64))
    return root, receipt, clock, monitor, api, before, prefix


def test_same_session_followup_finishes_four_arms_without_recreate_or_accounting_reset(cancelled_retry):
    root, receipt, clock, old_monitor, api, before, prefix = cancelled_retry
    proof = recovery.prepare(root, receipt, transport=api, clock=clock)
    assert proof["new_paid_calls"] == 0 and recovery.validate_recovery(root, receipt, clock=clock) == proof
    assert old_monitor.report()["aggregate_reserved_usd"] == "6.221400"
    assert (old_monitor.path / "retained_usage_lag_stop.json").is_file()
    other_apis = []
    def factory(root, receipt, monitor, mode):
        transport = api if mode == "parallel_fast" else API(mode, clock)
        if mode != "parallel_fast":
            other_apis.append(transport)
        return recovery.continuation_factory(root, receipt, monitor, mode, transport=transport)
    result = recovery.execute(root, receipt, clock=clock, sleep=clock.sleep, factory=factory, notify=lambda _: None)
    assert len(result["outcomes"]) == 4 and not result["budget"]["stopped"]
    assert result["outcomes"][0]["state"]["session_id"] == recovery.SESSION
    assert result["outcomes"][0]["state"]["turn_id"] == "turn_resume1"
    assert result["outcomes"][0]["state"]["parent_task_id"] == before["task_id"]
    assert sum(m == "POST" and p == "/agents/sessions" for m, p, _ in api.calls) == 1
    assert sum(m == "POST" and b["events"][0]["type"] == "agent.session.input.message" for m, p, b in api.calls if p.endswith("/events")) == 1
    assert base.AgentJournal(old_monitor.path / "agent_journal").task(before["task_id"]) == before
    assert (root / "live_journal.jsonl").read_bytes().startswith(prefix)
    assert result["budget"]["aggregate_reserved_usd"] == "6.221400"
    # Existing child identity/result is reused, no repeat message or session create.
    again = recovery.execute(root, receipt, clock=clock, sleep=clock.sleep, factory=factory, notify=lambda _: None)
    assert len(again["outcomes"]) == 4
    assert sum(m == "POST" and b["events"][0]["type"] == "agent.session.input.message" for m, p, b in api.calls if p.endswith("/events")) == 1


def test_same_session_recovery_does_not_clear_a_different_stop(cancelled_retry):
    root, receipt, clock, monitor, api, *_ = cancelled_retry
    (monitor.path / "stop.json").write_text(json.dumps({"reason": "soft_target_approaching_stop_threshold", "soft_receipt": base.digest(receipt)}))
    with pytest.raises(AgentExecutionError, match="only_exact_initial_usage_lag"):
        recovery.prepare(root, receipt, transport=api, clock=clock)
    assert api.followup is None


def test_recovery_deadline_uses_one_realistic_clock_sample(cancelled_retry):
    root, receipt, clock, _, api, *_ = cancelled_retry
    def advancing_clock():
        clock.now += 0.000001
        return clock()
    proof = recovery.prepare(root, receipt, transport=api, clock=advancing_clock)
    assert proof["followup_deadline"] == proof["prepared_at"] + base.ARM_SECONDS
    assert recovery.validate_recovery(root, receipt, clock=advancing_clock) == proof


def test_followup_never_renews_deadline_or_resets_cumulative_model_hold(cancelled_retry):
    root, receipt, clock, _, api, *_ = cancelled_retry
    clock.sleep(400)  # Original cancelled task's five-minute deadline has expired.
    proof = recovery.prepare(root, receipt, transport=api, clock=clock)
    monitor = retry.RetryMonitor(root, receipt, clock=clock, sleep=clock.sleep)
    runtime, child = recovery.continuation_factory(root, receipt, monitor, "parallel_fast", transport=api)
    runtime.start(child)
    assert child.deadline == proof["followup_deadline"] == 1700
    assert monitor.report()["sessions"][0]["model_high_water_estimate_usd"] == "0.26"
    before = (root / "live_journal.jsonl").read_bytes()
    clock.sleep(31)
    runtime2, child2 = recovery.continuation_factory(root, receipt, monitor, "parallel_fast", transport=api)
    assert child2.task_digest == child.task_digest and child2.deadline == child.deadline
    assert (root / "live_journal.jsonl").read_bytes() == before
    with pytest.raises(AgentExecutionError, match="fresh_hosted_usage_required"):
        monitor.require_fresh_usage()
    # An unknown fresh snapshot after resumption is bounded by the new immutable
    # child deadline and its first unknown observation, not the old expired turn.
    monitor.observe(child.parent_task_id, {"id": recovery.SESSION, "usage": None})
    assert monitor.usage_deadline(child.parent_task_id) == 1461
    assert runtime2.journal.task(child.parent_task_id)["state"] == "cancelled"


def test_changed_followup_input_is_refused_before_message_post(cancelled_retry):
    root, receipt, clock, _, api, *_ = cancelled_retry
    recovery.prepare(root, receipt, transport=api, clock=clock)
    monitor = retry.RetryMonitor(root, receipt, clock=clock, sleep=clock.sleep)
    runtime, child = recovery.continuation_factory(root, receipt, monitor, "parallel_fast", transport=api)
    data = child.model_dump(mode="json")
    data["instructions"] += " altered"
    changed = base.AgentTask.model_validate(data)
    with pytest.raises(AgentExecutionError, match="exact_authorized_followup_task"):
        runtime.start(changed)
    assert api.followup is None


def test_unknown_followup_send_is_not_repeated(cancelled_retry):
    from blueprint_pipeline.agent_execution.openai_transport import AgentTransportError
    root, receipt, clock, _, api, *_ = cancelled_retry
    recovery.prepare(root, receipt, transport=api, clock=clock)
    monitor = retry.RetryMonitor(root, receipt, clock=clock, sleep=clock.sleep)
    runtime, child = recovery.continuation_factory(root, receipt, monitor, "parallel_fast", transport=api)
    request = api.request
    sends = []
    def ambiguous(method, path, **kwargs):
        if method == "POST" and kwargs["body"]["events"][0]["type"] == "agent.session.input.message":
            sends.append(path)
            raise AgentTransportError("agents_api_connection_uncertain")
        return request(method, path, **kwargs)
    api.request = ambiguous
    runtime.start(child)
    runtime.start(child)
    assert len(sends) == 1
    assert runtime.journal.continuation(child.task_id)["delivery_state"] == "sent_unknown"


@pytest.mark.parametrize("during", ["prepare", "execute"])
def test_mismatched_session_get_never_authorizes_followup(cancelled_retry, during):
    root, receipt, clock, _, api, *_ = cancelled_retry
    if during == "execute":
        recovery.prepare(root, receipt, transport=api, clock=clock)
    request = api.request
    def wrong_session(method, path, **kwargs):
        value = request(method, path, **kwargs)
        if method == "GET" and path == "/agents/sessions/" + recovery.SESSION:
            return {**value, "id": "sess_foreign"}
        return value
    api.request = wrong_session
    with pytest.raises(AgentExecutionError, match="identity_mismatch"):
        if during == "prepare":
            recovery.prepare(root, receipt, transport=api, clock=clock)
        else:
            monitor = retry.RetryMonitor(root, receipt, clock=clock, sleep=clock.sleep)
            runtime, child = recovery.continuation_factory(root, receipt, monitor, "parallel_fast", transport=api)
            runtime.start(child)
    assert api.followup is None
