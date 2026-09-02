from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    _projection,
    TaskEvaluationPolicyCanaryDispatchError,
    dispatch_policy_canary_activation,
    process_policy_canary_dispatch_queue,
)


COMMIT = "a" * 40


def _public_artifact(character: str, artifact_id: str) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "digest": "sha256:" + character * 64,
        "size_bytes": 10,
    }


def test_projection_derives_reset_identity_only_for_blocked_legacy_episodes() -> None:
    episodes = []
    for candidate_id in ("pi05_droid", "groot_n17_droid"):
        for index in range(10):
            episodes.append(
                {
                    "candidate_id": candidate_id,
                    "cell_id": f"quick-cell-{index}",
                    "seed": 3100 + index,
                    "resolved_scenario": {
                        "family": "canonical_anchor",
                        "ordinal": index,
                    },
                    "status": "blocked",
                    "candidate_policy_queried": False,
                    "actions_reached_robot": False,
                    "arm_moved": False,
                    "policy_outcome_interpretable": False,
                    "typed_harness_failure": "RuntimeError",
                    "checkpoint_digest": "sha256:" + "a" * 64,
                    "runtime_identity_digest": "sha256:" + "b" * 64,
                    "visual_evidence": {
                        "media_gap": {
                            "type": "before_first_observation",
                            "reason": "policy_canary_episode_runner_failed",
                        }
                    },
                }
            )
    report = {
        "run_id": "scene-839873-canary-legacy",
        "result_status": "blocked",
        "delivery_digest": "sha256:" + "c" * 64,
        "report": {
            "machine_readable_report": _public_artifact("d", "report"),
            "evidence_manifest": _public_artifact("e", "manifest"),
        },
        "closure": {
            "billing": _public_artifact("f", "billing"),
            "teardown": _public_artifact("1", "teardown"),
            "provider_zero": _public_artifact("2", "provider-zero"),
        },
        "candidate_results": [],
        "artifacts": [],
    }
    result = {
        "run_id": report["run_id"],
        "configuration_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "status": "blocked",
        "episodes": episodes,
        "blockers": ["policy_canary_episode_runner_failed"],
    }

    projected = _projection(
        setup={
            "scene_id": "839873",
            "request_digest": "sha256:" + "5" * 64,
        },
        result=result,
        delivery=report,
    )

    expected = canonical_digest(
        {
            "resolved_scenario": episodes[0]["resolved_scenario"],
            "seed": episodes[0]["seed"],
            "execution_performed": False,
        }
    )
    assert projected["episodes"][0]["evidence"]["reset_state_digest"] == expected
    assert projected["counts"]["completed_learned_policy_rollout_count"] == 0


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_id = "scene-839873-canary-1"
    units = [
        {
            "campaign_unit_id": f"unit-{index}",
            "cell_id": f"quick-cell-{index}",
            "seed": 3100 + index,
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        }
        for index in range(10)
    ]
    activation: dict[str, object] = {
        "schema_version": "task_evaluation_policy_campaign_activation.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "campaign_unit_count": 10,
        "campaign_units": units,
        "activation_digest": "",
    }
    activation["activation_digest"] = canonical_digest(
        activation, digest_field="activation_digest"
    )
    activation_root = tmp_path / "activation"
    activation_path = _write(
        activation_root / "task_evaluation_policy_campaign_activation.v1.json",
        activation,
    )
    packet = _write(
        tmp_path / "packet" / "native_task_arena_packet_receipt.v1.json", {}
    )
    runtime_source = _write(tmp_path / "runtime-source.json", {})
    construction = _write(tmp_path / "construction.json", {})
    cells = []
    for index in range(10):
        scenario = {"family": "canonical", "ordinal": index}
        cells.append(
            {
                "cell_id": f"quick-cell-{index}",
                "seed": 3100 + index,
                "family": (
                    "canonical_anchor" if index < 2 else "placement_approach"
                ),
                "cell_spec_digest": "sha256:" + f"{index:064x}",
                "resolved_scenario": scenario,
                "resolved_scenario_digest": canonical_digest(scenario),
                "control_diagnostic": {
                    "mode": "nonblocking_diagnostic_pending",
                    "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False,
                },
            }
        )
    runtime: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_id": run_id,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_revision_digest": "sha256:" + "9" * 64,
        "matrix_digest": "sha256:" + "8" * 64,
        "configuration_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "activation_digest": activation["activation_digest"],
        "base_native_packet": _record(packet),
        "runtime_source": _record(runtime_source),
        "construction_result": _record(construction),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "cells": cells,
        "execution_authority": {
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "single_warm_provider_session_required": True,
            "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True,
        },
        "resource_authority": {
            "resource_name": "blueprint-native-task-policy-canary-0123456789abcdef",
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 4.0,
            "hard_ttl_seconds": 14_400,
            "user_confirmed": True,
        },
        "runtime_inputs_digest": "",
    }
    runtime["runtime_inputs_digest"] = canonical_digest(
        runtime, digest_field="runtime_inputs_digest"
    )
    runtime_path = _write(
        activation_root / "task_evaluation_policy_canary_runtime_inputs.v1.json",
        runtime,
    )
    activation_result: dict[str, object] = {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "policy_campaign_queue_materialized_no_execution",
        "activation_id": "activation-1",
        "source_commit": COMMIT,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "policy_canary_runtime_inputs_path": str(runtime_path),
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    activation_result["result_digest"] = canonical_digest(
        activation_result, digest_field="result_digest"
    )
    activation_result_path = _write(tmp_path / "activation-result.json", activation_result)
    records = {}
    for name in (
        "pi05_execution_spec",
        "groot_execution_spec",
        "pi05_checkpoint_inventory",
    ):
        records[name] = _record(_write(tmp_path / f"{name}.json", {"name": name}))
    setup: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_execution_setup.v1",
        "status": "verified_runnable",
        "scene_id": "839873",
        "configured_source_launch_id": "configured-scene-839873-r4",
        "scene_revision_digest": runtime["scene_revision_digest"],
        "activation_digest": activation["activation_digest"],
        "source_commit": COMMIT,
        "provider": "vast",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": "sha256:" + "7" * 64,
        "records": records,
        "setup_digest": "",
    }
    setup["setup_digest"] = canonical_digest(setup, digest_field="setup_digest")
    setup_path = _write(tmp_path / "execution-setup.json", setup)
    return activation_result_path, setup_path, activation_path


def test_dispatcher_materializes_one_authority_bundle_and_allocator_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch"
    observed: dict[str, object] = {}

    def fake_bundle(**kwargs):
        observed["bundle"] = kwargs
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {
            "bundle_sha256": "sha256:" + "b" * 64,
            "bundle_path": str(job / "bundle.zip"),
        }
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )

    def fake_allocator(argv):
        observed["argv"] = list(argv)
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        _write(adapter, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0

    receipt = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        allocator_runner=fake_allocator,
    )

    argv = observed["argv"]
    assert receipt["status"] == "prepared_no_execution"
    assert argv.count("native-task-arena-policy-canary-session") == 1
    assert "--execute" not in argv
    assert argv[0] == "gpu-canary"
    assert receipt["retry_cap"] == 0
    assert receipt["provider_mutation_performed"] is False
    assert Path(observed["bundle"]["session_authority_path"]).is_file()


def test_dispatcher_refuses_absent_scene839873_setup_before_allocator(
    tmp_path: Path,
) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="policy_canary_scene839873_setup_receipt_missing",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=tmp_path / "missing.json",
            output_root=tmp_path / "dispatch",
            implementation_commit=COMMIT,
            allocator_runner=lambda _argv: pytest.fail("allocator must not run"),
        )


def test_allocator_invocation_marker_prevents_unrecorded_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch-crashed-allocator"

    def fake_bundle(**kwargs):
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64}
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )

    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="policy_canary_allocator_exit_17_without_result",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=setup_path,
            output_root=output,
            implementation_commit=COMMIT,
            execute=True,
            allocator_runner=lambda _argv: 17,
        )

    started = json.loads(
        (output / "allocator_invocation_started.json").read_text(encoding="utf-8")
    )
    finished = json.loads(
        (output / "allocator_invocation_finished.json").read_text(encoding="utf-8")
    )
    assert started["allocator_invoked"] is True
    assert started["automatic_retry_authorized"] is False
    assert finished["exit_code"] == 17
    assert finished["adapter_result_present"] is False
    with pytest.raises(
        TaskEvaluationPolicyCanaryDispatchError,
        match="policy_canary_allocator_previous_invocation_without_result",
    ):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result,
            execution_setup_path=setup_path,
            output_root=output,
            implementation_commit=COMMIT,
            execute=True,
            allocator_runner=lambda _argv: pytest.fail("allocator invoked twice"),
        )


def test_live_shaped_result_waits_for_billing_and_never_launches_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch-live"
    calls = {"allocator": 0, "bundle": 0}

    def fake_bundle(**kwargs):
        calls["bundle"] += 1
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64}
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )

    def fake_allocator(argv):
        calls["allocator"] += 1
        adapter_path = Path(argv[argv.index("--adapter-output") + 1])
        attempt = output / "allocator" / "attempts" / "attempt-1"
        evidence = attempt / "immutable_execution"
        evidence.mkdir(parents=True, exist_ok=True)
        inner = {
            "schema_version": "native_task_arena_policy_canary_session_result.v1",
            "status": "runtime_completed_unqualified_pending_closeout",
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "episodes": [
                {
                    "candidate_id": candidate,
                    "cell_id": f"quick-cell-{index}",
                    "seed": 3100 + index,
                    "status": "completed",
                }
                for candidate in ("pi05_droid", "groot_n17_droid")
                for index in range(10)
            ],
            "artifact_inventory": [],
            "result_digest": "",
        }
        inner["result_digest"] = canonical_digest(
            inner, digest_field="result_digest"
        )
        inner_path = _write(
            evidence / "native_task_arena_policy_canary_session_result.v1.json",
            inner,
        )
        teardown = _write(attempt / "vast_teardown_manifest.json", {"status": "completed"})
        _write(
            adapter_path,
            {
                "schema_version": "native_task_arena_policy_canary_session_result.v1",
                "status": "completed",
                "vast_instance_ids": [49247792],
                "native_control_result_path": str(inner_path),
                "teardown_manifest_path": str(teardown),
                "continuing_spend_from_this_run": False,
                "all_staged_objects_absent": True,
                "provider_closeout": {
                    "provider_zero_confirmed": True,
                    "warm_session_retained": False,
                    "all_staged_objects_absent": True,
                },
            },
        )
        return 0

    zero = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "blockers": [],
        "receipt_digest": "sha256:" + "c" * 64,
    }
    progress_updates = []

    def sync_progress(**kwargs):
        progress_updates.append(kwargs["progress"])
        return {"status": "succeeded", "response": {"status": "recorded"}}

    first = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=fake_allocator,
        provider_zero_collector=lambda: zero,
        progress_sync_runner=sync_progress,
    )
    second = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=lambda _argv: pytest.fail("allocator invoked twice"),
        provider_zero_collector=lambda: zero,
        progress_sync_runner=sync_progress,
    )

    assert first["status"] == second["status"] == "awaiting_official_billing"
    assert first["allocator_invoked"] is True
    assert second["allocator_invoked"] is False
    assert first["website_progress_sync"]["status"] == "succeeded"
    assert progress_updates[0]["phase"] == "awaiting_official_billing"
    assert calls == {"allocator": 1, "bundle": 1}

    def post_billing(**kwargs):
        _write(Path(kwargs["output_path"]), {"status": "reconciled_official_posted_charges"})
        return True

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher._materialize_official_billing_if_posted",
        post_billing,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_vast_official_same_goal_reconciliation",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.materialize_policy_canary_result_delivery",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "result_status": kwargs["result_status"],
            "delivery_digest": "sha256:" + "d" * 64,
            "report": {},
            "closure": {},
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher._projection",
        lambda **_kwargs: {"projection_digest": "sha256:" + "e" * 64},
    )
    third = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=lambda _argv: pytest.fail("allocator invoked on billing resume"),
        provider_zero_collector=lambda: zero,
        sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {
                "status": "failed",
                "run_result_digest": "sha256:" + "e" * 64,
            },
        },
    )

    assert third["status"] == "awaiting_website_sync_or_notification"
    assert third["allocator_invoked"] is False
    assert not (output / "dispatch_receipt.json").exists()

    fourth = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=lambda _argv: pytest.fail("allocator invoked on sync resume"),
        provider_zero_collector=lambda: zero,
        sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {
                "status": "delivered",
                "run_result_digest": "sha256:" + "e" * 64,
            },
        },
    )

    assert fourth["status"] == "completed_unqualified"
    assert fourth["allocator_invoked"] is False
    assert calls == {"allocator": 1, "bundle": 1}


def test_invalid_envelope_is_quarantined_without_allocator(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    pending = _write(queue / "pending" / "000-invalid.json", {"secret": "redacted"})
    setups = tmp_path / "setups"
    setups.mkdir()

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_invalid_envelope"
    assert result["allocator_invoked"] is False
    assert result["provider_mutation_performed"] is False
    assert "secret" not in json.dumps(result)
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()


def test_proven_zero_allocation_terminalizes_without_billing_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    output = tmp_path / "dispatch-no-allocation"

    def fake_bundle(**kwargs):
        job = Path(kwargs["job_dir"])
        job.mkdir(parents=True, exist_ok=True)
        receipt = {"bundle_sha256": "sha256:" + "b" * 64}
        _write(
            job / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.build_policy_canary_session_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )

    def allocator_without_instance(argv):
        adapter_path = Path(argv[argv.index("--adapter-output") + 1])
        _write(
            adapter_path,
            {
                "status": "blocked",
                "vast_instance_ids": [],
                "provider_mutations_performed": 0,
                "provider_create_attempted": False,
                "vast_side_effects_may_have_occurred": False,
                "continuing_spend_from_this_run": False,
                "blockers": ["policy_canary_provider_capacity_unavailable"],
            },
        )
        return 2

    synced = []
    result = dispatch_policy_canary_activation(
        activation_result_path=activation_result,
        execution_setup_path=setup_path,
        output_root=output,
        implementation_commit=COMMIT,
        execute=True,
        allocator_runner=allocator_without_instance,
        blocked_sync_runner=lambda **kwargs: synced.append(kwargs)
        or {
            "status": "succeeded",
            "notification_delivery": {"status": "accepted"},
        },
    )

    assert result["status"] == "blocked_without_provider_allocation"
    assert result["provider_allocation_performed"] is False
    assert result["provider_mutation_performed"] is False
    assert result["terminal_sync"]["status"] == "succeeded"
    assert synced[0]["blockers"] == ["policy_canary_provider_capacity_unavailable"]
    assert not (output / "official_billing_reconciliation.json").exists()


def test_post_allocator_failure_is_not_labeled_preprovider_or_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
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
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())

    def crash_after_allocator(**kwargs):
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        marker = {
            "schema_version": "task_evaluation_policy_canary_allocator_invocation.v1",
            "status": "started",
            "run_id": "scene-839873-canary-1",
            "allocator_invoked": True,
            "invocation_digest": "sha256:" + "a" * 64,
        }
        _write(output / "allocator_invocation_started.json", marker)
        raise TaskEvaluationPolicyCanaryDispatchError(
            "policy_canary_allocator_exit_17_without_result"
        )

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.dispatch_policy_canary_activation",
        crash_after_allocator,
    )
    zero = {
        "status": "provider_zero_confirmed",
        "provider_zero_verified": True,
        "live_instance_count": 0,
    }
    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: pytest.fail(
            "post-allocator failure cannot use preprovider sync"
        ),
        provider_zero_collector=lambda: zero,
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_after_allocator_invocation_provider_zero"
    assert result["allocator_invoked"] is True
    assert result["provider_mutation_status"] == "unknown_after_allocator_invocation"
    assert result["automatic_retry_performed"] is False
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()


def test_paid_queue_waits_for_setup_without_invoking_dispatcher(tmp_path: Path) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
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
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    envelope_path = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
    )

    assert observed["results"][0]["status"] == (
        "waiting_for_scene839873_execution_setup"
    )
    assert observed["results"][0]["allocator_invoked"] is False
    assert envelope_path.is_file()
    assert (
        tmp_path / "dispatches/activation-1/preprovider_waiting.json"
    ).is_file()


def test_nonretryable_setup_refusal_moves_queue_only_after_blocked_email_sync(
    tmp_path: Path,
) -> None:
    activation_result, _setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
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
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    invalid_template = _write(tmp_path / "invalid-template.json", {})

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=invalid_template,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit=COMMIT,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "delivered"},
        },
    )

    assert observed["results"][0]["status"] == "blocked_before_paid_dispatch"
    assert observed["results"][0]["allocator_invoked"] is False
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()
    assert (
        tmp_path / "dispatches/activation-1/preprovider_blocked.json"
    ).is_file()


def test_stale_commit_setup_is_terminalized_before_allocator(
    tmp_path: Path,
) -> None:
    activation_result, setup_path, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
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
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    setups = tmp_path / "setups"
    setups.mkdir()
    (setups / "activation-1.json").write_bytes(setup_path.read_bytes())

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        dispatch_root=tmp_path / "dispatches",
        implementation_commit="b" * 40,
        execute=True,
        blocked_sync_runner=lambda **_kwargs: {
            "status": "succeeded",
            "notification_delivery": {"status": "delivered"},
        },
    )

    result = observed["results"][0]
    assert result["status"] == "blocked_before_paid_dispatch"
    assert result["allocator_invoked"] is False
    assert result["blockers"] == ["policy_canary_dispatch_activation_setup_mismatch"]
    assert not pending.exists()
    assert (queue / "blocked" / pending.name).is_file()


def test_paid_queue_materializes_setup_from_staged_template_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation_result, source_setup, _activation = _inputs(tmp_path)
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    envelope = {
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
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = _write(queue / "pending" / "activation-1.json", envelope)
    template = _write(tmp_path / "template.json", {"static": True})
    setups = tmp_path / "setups"
    setups.mkdir()
    dispatches = tmp_path / "dispatches"

    def materialize(*, output_dir, **_kwargs):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "task_evaluation_policy_canary_execution_setup.v1.json"
        target.write_bytes(source_setup.read_bytes())
        return json.loads(target.read_text(encoding="utf-8"))

    def dispatch(**kwargs):
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        _write(output / "dispatch_receipt.json", {"status": "prepared_no_execution"})
        return {"status": "prepared_no_execution", "allocator_invoked": True}

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.materialize_scene839873_policy_canary_setup_from_template",
        materialize,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.dispatch_policy_canary_activation",
        dispatch,
    )

    observed = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue,
        execution_setup_root=setups,
        execution_setup_template_path=template,
        dispatch_root=dispatches,
        implementation_commit=COMMIT,
        execute=False,
    )

    assert observed["results"][0]["status"] == "prepared_no_execution"
    assert (
        setups
        / "activation-1/task_evaluation_policy_canary_execution_setup.v1.json"
    ).is_file()
    assert not pending.exists()
    assert (queue / "completed" / pending.name).is_file()
