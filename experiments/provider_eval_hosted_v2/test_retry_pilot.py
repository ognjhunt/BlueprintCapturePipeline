"""Synthetic retry lifecycle preserves deleted history and cumulative budgets."""

from decimal import Decimal
import json

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from experiments.provider_eval_recovery.harness import Ledger, digest
from . import soft_pilot as base
from . import retry_pilot as retry
from .reconcile import reconcile_cleanup
from .test_soft_pilot import admitted, API, factory_for, cleanup_fixture  # noqa: F401 - pytest fixture registration


@pytest.fixture
def ready_retry(admitted):  # noqa: F811 - imported pytest fixture injection
    root, initial, _, clock = admitted
    old = {**initial, "source_commit": base.PREVIOUS_SOURCE_IDENTITY[0], "code_sha256": base.PREVIOUS_SOURCE_IDENTITY[1]}
    old_path = root / "protocols" / base.PROTOCOL / "soft_pilot_approval.json"
    old_path.write_text(json.dumps(old))
    old_monitor = base.SoftMonitor(root, old, clock=clock)
    old_monitor.reserve()
    api = API("parallel_fast", clock, usage={"input_tokens": 49484, "output_tokens": 261})
    create = api.request
    def request(method, path, **kwargs):
        value = create(method, path, **kwargs)
        if method == "POST" and path == "/agents/sessions":
            value["id"] = retry.PRIOR_SESSION
            value["environment"]["network"]["allowed_domains"] = []
        return value
    api.request = request
    runtime, task = base.make_runtime(root, old, old_monitor, "parallel_fast", transport=api)
    runtime._validate_session = lambda *_: (_ for _ in ()).throw(AgentExecutionError("old_validator"))
    with pytest.raises(AgentExecutionError):
        runtime.start(task)
    old_monitor.stop("old_failed_pilot")
    api.cancelled = True
    path, sha, _ = cleanup_fixture(root, old, clock, api)
    reconcile_cleanup(root, old, path, sha, clock=clock)
    old_state = runtime.journal.task(task.task_id)
    old_ledger = (root / "live_journal.jsonl").read_bytes()
    old_receipt = old_path.read_bytes()
    receipt, receipt_sha = retry.prepare_receipt(root, base.OWNER, digest(old), clock=clock)
    return root, receipt, receipt_sha, clock, old_monitor, old_state, old_ledger, old_receipt


def test_retry_new_receipt_keeps_prior_exposure_and_exact_approval(ready_retry):
    root, receipt, sha, clock, _, _, prefix, _ = ready_retry
    assert receipt["approval"] == retry.APPROVAL
    assert receipt["baseline_reserved_usd"] == "4.765920"
    assert receipt["prior_hosted_soft_exposure_usd"] == "0.341335"
    assert receipt["initial_projected_soft_total_usd"] == "1.870315"
    assert retry.validate_receipt(root, base.OWNER, sha, clock=clock) == receipt
    assert (root / "live_journal.jsonl").read_bytes() == prefix
    assert retry.prepare_receipt(root, base.OWNER, receipt["prior_approval_sha256"], clock=clock)[1] == sha


def test_four_new_retry_sessions_are_durable_and_old_deleted_state_is_unchanged(ready_retry):
    root, receipt, _, clock, old_monitor, old_state, prefix, old_bytes = ready_retry
    factory, apis = factory_for(clock)
    result = retry.run_retry(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["protocol"] == retry.PROTOCOL and len(result["outcomes"]) == 4
    assert all(o["state"]["state"] == "completed" for o in result["outcomes"])
    assert result["budget"]["soft_target_usd"] == "2.00" and result["budget"]["stop_threshold_usd"] == "1.90"
    assert result["budget"]["projected_with_remaining_search_opportunity_usd"] == "1.870315"
    assert result["budget"]["aggregate_reserved_usd"] == "6.221400"
    assert Decimal(result["budget"]["aggregate_reserved_usd"]) + base.SEARCH_HOLD == Decimal("6.294900")
    assert (root / "live_journal.jsonl").read_bytes().startswith(prefix)
    assert (root / "protocols" / base.PROTOCOL / "soft_pilot_approval.json").read_bytes() == old_bytes
    assert base.AgentJournal(old_monitor.path / "agent_journal").task(old_state["task_id"]) == old_state
    for api in apis:
        creates = [b for m, p, b in api.calls if m == "POST" and p == "/agents/sessions"]
        assert len(creates) == 1
        assert creates[0]["metadata"]["blueprint_task_id"].startswith("hosted_retry1_01_")
        assert creates[0]["metadata"]["blueprint_run_id"] == retry.PROTOCOL
        assert "narrow ranges" in creates[0]["agent"]["instructions"]
        assert "All retained text remains accessible" in creates[0]["agent"]["instructions"]
        assert creates[0]["agent"]["model"] == base.MODEL
    again = retry.run_retry(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert len(again["outcomes"]) == 4
    assert sum(m == "POST" and p == "/agents/sessions" for a in apis for m, p, _ in a.calls) == 4
    assert len(list((root / "protocols" / retry.PROTOCOL / "evidence").glob("**/hosted_files.json"))) == 4


def test_cumulative_prior_plus_new_usage_stops_before_another_arm_or_tool(ready_retry, monkeypatch):
    root, receipt, _, clock, *_ = ready_retry
    import importlib
    route = importlib.import_module(base.ExistingSearchRoute.__module__)
    monkeypatch.setattr(route, "existing_key", lambda *_: (_ for _ in ()).throw(AssertionError("provider key read")))
    factory, apis = factory_for(clock, usage={"input_tokens": 500000, "output_tokens": 0})
    result = retry.run_retry(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    assert result["budget"]["stopped"] and len(apis) == 1 and apis[0].cancelled
    assert Decimal(result["budget"]["projected_with_remaining_search_opportunity_usd"]) > Decimal("2")
    assert not any(e.get("role") == base.PROTOCOL + ":search" for e in Ledger(root / "live_journal.jsonl", "10.00").events)
    assert not any(body and any(e.get("type") == "agent.session.input.tool_result" for e in body.get("events", []))
                   for _, _, body in apis[0].calls)


@pytest.mark.parametrize("tamper", ["target", "approval", "reset_baseline", "omit_prior", "change_cleanup"])
def test_retry_cannot_drop_prior_spend_or_reset_authority(ready_retry, tamper):
    root, receipt, _, clock, old_monitor, old_state, *_ = ready_retry
    changed = dict(receipt)
    if tamper == "target":
        changed["soft_target_usd"] = "20"
    elif tamper == "approval":
        changed["approval"] = base.APPROVAL
    elif tamper == "reset_baseline":
        changed["baseline_events"] = 0
    elif tamper == "omit_prior":
        changed["prior_hosted_soft_exposure_usd"] = "0"
    else:
        (old_monitor.path / old_state["task_id"] / "approved_cleanup.json").write_text("{}")
    retry.receipt_path(root).write_text(json.dumps(changed))
    with pytest.raises(AgentExecutionError):
        retry.validate_receipt(root, base.OWNER, digest(changed), clock=clock)


def test_used_retry_cannot_absorb_spend_into_another_fresh_receipt(ready_retry):
    root, receipt, _, clock, *_ = ready_retry
    factory, _ = factory_for(clock)
    retry.run_retry(root, receipt, clock=clock, sleep=clock.sleep, notify=lambda _: None, factory=factory)
    retry.receipt_path(root).unlink()
    with pytest.raises(AgentExecutionError, match="already_used_no_new_baseline"):
        retry.prepare_receipt(root, base.OWNER, receipt["prior_approval_sha256"], clock=clock)


def test_python_entry_and_live_guard_cannot_use_mutated_retry_scope(ready_retry):
    root, receipt, _, clock, *_ = ready_retry
    bad = {**receipt, "prior_hosted_soft_exposure_usd": "0"}
    factory, apis = factory_for(clock)
    with pytest.raises(AgentExecutionError):
        retry.run_retry(root, bad, clock=clock, sleep=clock.sleep, factory=factory)
    assert apis == []
    monitor = retry.RetryMonitor(root, dict(receipt), clock=clock)
    monitor.reserve()
    monitor.receipt["baseline_reserved_usd"] = "10"
    with pytest.raises(AgentExecutionError, match="in_memory_scope_changed"):
        monitor.guard()
