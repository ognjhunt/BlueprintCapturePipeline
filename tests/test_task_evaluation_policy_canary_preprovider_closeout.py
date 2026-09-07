"""R12: preserve typed no-execution outcomes before the provider boundary.

These tests drive the real policy-canary dispatcher and queue consumer.  They
assert that a dry-run or a preprovider refusal remains a bounded diagnostic
record, with no invented policy projection or provider-zero receipt, and that
the paid retry boundary stays at zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as terminal_reconciler
from blueprint_pipeline import task_evaluation_scene_terminal_result_index as terminal_index
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    dispatch_policy_canary_activation,
    process_policy_canary_dispatch_queue,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import reconcile_launches
from tests.test_task_evaluation_policy_canary_dispatcher import (
    COMMIT,
    _inputs,
    _record,
    _write,
)
from tests.test_task_evaluation_scene_terminal_result_index import _owner_launch_run
from tests.test_task_evaluation_scene_terminal_reconciler import _env


def _fake_bundle(monkeypatch):
    def build(**kwargs):
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64, "bundle_path": str(job / "bundle.zip")}
        _write(job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json", receipt)
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        build,
    )


def _envelope(activation_result: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": _record(activation_result),
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    value["envelope_digest"] = canonical_digest(value, digest_field="envelope_digest")
    return value


def test_preprovider_block_is_sealed_after_notification_and_has_no_fake_closure(
    tmp_path, monkeypatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope_path = _write(queue / "pending" / "activation-1.json", _envelope(activation_result))
    setups = tmp_path / "setups"
    setups.mkdir()
    # The template producer is a real queue consumer; this malformed template
    # forces its typed preprovider refusal before the dispatcher/allocator.
    template = _write(tmp_path / "invalid-template.json", {})

    def blocked_sync(**_kwargs):
        return {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted", "terminal_state": "blocked"},
        }

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=blocked_sync,
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_before_paid_dispatch"
    assert result["allocator_invoked"] is False
    assert result["provider_mutation_performed"] is False
    assert result["provider_allocation_performed"] is False
    assert result["automatic_retry_performed"] is False
    assert result["retry_cap"] == 0
    assert not envelope_path.exists()
    assert (queue / "blocked" / envelope_path.name).is_file()

    path = tmp_path / "dispatches" / "activation-1" / "preprovider_blocked.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == result
    assert record["blocked_result_digest"] == canonical_digest(
        record, digest_field="blocked_result_digest"
    )
    assert record["envelope_digest"] == _envelope(activation_result)["envelope_digest"]
    assert record["run_id"] == "scene-839873-canary-1"
    assert "policy_canary_result_projection" not in record
    assert "provider_zero" not in record


def test_dry_run_receipt_is_explicit_nonexecution_without_provider_zero_or_projection(
    tmp_path, monkeypatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    _fake_bundle(monkeypatch)

    def dry_run_allocator(argv):
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        _write(
            adapter,
            {
                "status": "dry_run_ready",
                "vast_instance_ids": [],
                "provider_mutations_performed": 0,
                "provider_create_attempted": False,
                "vast_side_effects_may_have_occurred": False,
                "continuing_spend_from_this_run": False,
            },
        )
        return 0

    receipt = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=tmp_path / "dispatch",
        implementation_commit=COMMIT,
        execute=False,
        allocator_runner=dry_run_allocator,
    )

    persisted = json.loads(
        (tmp_path / "dispatch" / "dispatch_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt == persisted
    assert persisted["status"] == "prepared_no_execution"
    assert persisted["provider_allocation_performed"] is False
    assert persisted["provider_mutation_performed"] is False
    assert persisted["provider_zero_required"] is False
    assert persisted["automatic_retry_performed"] is False
    assert persisted["retry_cap"] == 0
    assert persisted["terminal_result_kind"] == "prepared_no_execution"
    assert persisted["receipt_digest"] == canonical_digest(
        persisted, digest_field="receipt_digest"
    )
    assert "policy_canary_result_projection" not in persisted
    assert "provider_zero" not in persisted


def test_preprovider_producer_is_indexed_and_owner_status_closes_as_blocked(
    tmp_path, monkeypatch
) -> None:
    env = _env(tmp_path)
    launch_root = _owner_launch_run(
        tmp_path / "state", env, run_id="scene-839873-canary-1"
    )
    assert terminal_index.index_launch_bridge(
        launch_run_root=launch_root,
        scene_intent_root=env["config"]["intent_root"],
        terminal_result_root=env["config"]["terminal_result_root"],
    )["status"] == "launch_bridge_indexed"
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope_path = _write(queue / "pending" / "activation-1.json", _envelope(activation_result))
    setups = tmp_path / "setups"
    setups.mkdir()
    template = _write(tmp_path / "invalid-template.json", {})

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted", "terminal_state": "blocked"},
        },
    )
    assert observed["results"][0]["status"] == "blocked_before_paid_dispatch"
    producer_root = tmp_path / "dispatches" / "activation-1"

    indexed = terminal_index.index_policy_canary_nonexecution(
        canary_run_root=producer_root,
        terminal_result_root=env["config"]["terminal_result_root"],
    )
    assert indexed["status"] == "policy_canary_nonexecution_indexed"
    owner_root = Path(env["config"]["terminal_result_root"]) / env["intent_id"]
    assert (owner_root / terminal_reconciler.NONEXECUTION_FILENAME).is_file()
    joined = terminal_reconciler.reconcile_terminal_owner_result(
        intent=env["intent"],
        config=env["config"],
        release=env["release"],
        now=env["now"],
        output=env["output"],
    )
    assert joined["terminal"] is True
    assert joined["status"] == "blocked"
    assert joined["phase"] == "policy_canary_preprovider_blocked"
    assert joined["result_reference"] is None
    assert "policy_canary_execution_template_invalid" in joined["blockers"]
    assert not envelope_path.exists()


def test_prepared_no_execution_is_indexed_without_becoming_completed(
    tmp_path, monkeypatch
) -> None:
    env = _env(tmp_path)
    launch_root = _owner_launch_run(
        tmp_path / "state", env, run_id="scene-839873-canary-1"
    )
    assert terminal_index.index_launch_bridge(
        launch_run_root=launch_root,
        scene_intent_root=env["config"]["intent_root"],
        terminal_result_root=env["config"]["terminal_result_root"],
    )["status"] == "launch_bridge_indexed"
    activation_result, setup_path, _activation = _inputs(tmp_path)
    _fake_bundle(monkeypatch)

    def dry_run_allocator(argv):
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        _write(
            adapter,
            {
                "status": "dry_run_ready",
                "vast_instance_ids": [],
                "provider_mutations_performed": 0,
                "provider_create_attempted": False,
                "vast_side_effects_may_have_occurred": False,
                "continuing_spend_from_this_run": False,
            },
        )
        return 0

    dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=tmp_path / "dispatches" / "activation-1",
        implementation_commit=COMMIT,
        execute=False,
        allocator_runner=dry_run_allocator,
    )
    indexed = terminal_index.index_policy_canary_terminal(
        canary_run_root=tmp_path / "dispatches" / "activation-1",
        terminal_result_root=env["config"]["terminal_result_root"],
    )
    assert indexed["status"] == "policy_canary_nonexecution_indexed"
    joined = terminal_reconciler.reconcile_terminal_owner_result(
        intent=env["intent"],
        config=env["config"],
        release=env["release"],
        now=env["now"],
        output=env["output"],
    )
    assert joined["terminal"] is False
    assert joined["status"] == "awaiting_execution"
    assert joined["phase"] == "policy_canary_prepared"
    assert joined["result_reference"] is None


def test_launch_reconciler_tick_consumes_preprovider_record_without_provider_zero(
    tmp_path, monkeypatch
) -> None:
    env = _env(tmp_path)
    launch_root = _owner_launch_run(
        tmp_path / "state", env, run_id="scene-839873-canary-1"
    )
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope_path = _write(queue / "pending" / "activation-1.json", _envelope(activation_result))
    setups = tmp_path / "setups"
    setups.mkdir()
    template = _write(tmp_path / "invalid-template.json", {})
    process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted", "terminal_state": "blocked"},
        },
    )

    report = reconcile_launches(
        queue_root=tmp_path / "reconcile-queue",
        state_root=launch_root.parent,
        guard_report_path=tmp_path / "missing-guard.json",
        policy_canary_dispatch_root=tmp_path / "dispatches",
        terminal_result_root=env["config"]["terminal_result_root"],
        scene_intent_root=env["config"]["intent_root"],
        publish_progress=False,
    )
    assert report["status"] == "passed", report
    assert [row["status"] for row in report["terminal_index"]] == [
        "launch_bridge_indexed",
        "policy_canary_nonexecution_indexed",
    ]
    assert report.get("terminal_provider_zero") == []
    assert report["allocator_invoked"] is False
    assert not envelope_path.exists()


def test_nonexecution_index_rejects_two_conflicting_producer_records(tmp_path, monkeypatch) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope_path = _write(queue / "pending" / "activation-1.json", _envelope(activation_result))
    setups = tmp_path / "setups"
    setups.mkdir()
    template = _write(tmp_path / "invalid-template.json", {})
    process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted", "terminal_state": "blocked"},
        },
    )
    producer_root = tmp_path / "dispatches" / "activation-1"
    blocked = producer_root / "preprovider_blocked.json"
    (producer_root / "no_provider_allocation_blocked.json").write_bytes(blocked.read_bytes())
    with pytest.raises(terminal_index.TerminalResultIndexError, match="nonexecution_record_ambiguous"):
        terminal_index.index_policy_canary_nonexecution(
            canary_run_root=producer_root,
            terminal_result_root=tmp_path / "terminal",
        )
    assert not (tmp_path / "terminal").exists()
    assert not envelope_path.exists()
