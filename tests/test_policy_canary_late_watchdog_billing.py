"""A later watchdog close may settle an immutable pre-close allocator snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    _adapter_instance_ids,
    _record,
    _sealed_provider_zero,
    dispatch_policy_canary_activation,
    process_policy_canary_dispatch_queue,
)
from blueprint_pipeline.policy_canary_retained_billing import retained_sparse_billing_gap
from blueprint_pipeline.vast_official_billing_extractor import (
    VastOfficialBillingExtractionError,
    _terminal_evidence,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _case(root: Path, *, instance_id: int = 52854484) -> tuple[Path, Path]:
    attempt = root / "allocator" / "attempts" / "attempt_001"
    provider = attempt / "vast_provider_run"
    watchdog = attempt / "independent_vast_watchdog" / "groot_oscar_runpod_canary_watchdog.json"
    prefix = "blueprint-native-task-policy-canary-test"
    adapter = _write(provider / "vast_provider_adapter_result.json", {
        "vast_instance_ids": [instance_id], "continuing_spend_from_this_run": False,
    })
    teardown = _write(provider / "vast_teardown_manifest.json", {
        "vast_instance_ids": [instance_id], "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
    })
    artifact = _write(attempt / "artifact_manifest.json", {"status": "completed"})
    _write(root / "post_teardown_global_provider_zero.json", {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "provider_zero_verified": True, "live_instance_count": 0,
    })
    snapshot = {
        "status": "retained_until_hard_ttl", "instance_ids": [instance_id],
        "watchdog_evidence_path": str(watchdog), "watchdog_out_dir": str(watchdog.parent),
        "pod_name_prefix": prefix, "resource_name_exact": prefix,
    }
    result = _write(root / "allocator_result.json", {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked", "retry_cap": 0, "continuing_spend_from_this_run": False,
        "independent_watchdog": snapshot,
        "adapter_result_path": str(adapter), "teardown_manifest_path": str(teardown),
        "artifact_manifest_path": str(artifact),
        "provider_closeout": {
            "provider_zero_confirmed": True, "warm_session_retained": False,
            "all_staged_objects_absent": True,
        },
    })
    _write(watchdog, {
        "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "provider_terminal", "provider_absence_confirmed": True,
        "provider_absence_scope": "recorded_instance_and_lane_prefix",
        "raw_secret_values_recorded": False,
        "pod_name_prefix": prefix, "resource_name_exact": prefix,
        "watchdog_out_dir": str(watchdog.parent),
        "recorded_vast_instance": {
            "instance_id": str(instance_id), "scope_confirmed": True,
            "pod_name_prefix": prefix,
        },
        "recorded_vast_instance_teardown": {
            "instance_id": str(instance_id), "status": "absent",
            "provider_absence_confirmed": True,
        },
    })
    return result, watchdog


def test_late_watchdog_recovers_exact_billing_lineage_without_mutating_allocator(
    tmp_path: Path,
) -> None:
    result, watchdog = _case(tmp_path)
    original = result.read_bytes()

    assert _adapter_instance_ids(json.loads(original), result_path=result) == [52854484]
    evidence = _terminal_evidence(instance_id=52854484, terminal_result_path=result)

    assert evidence["provider_zero_verified"] is True
    assert evidence["independent_watchdog_terminal"]["path"] == str(watchdog)
    assert result.read_bytes() == original


@pytest.mark.parametrize("defect", ["wrong_instance", "wrong_scope", "symlink"])
def test_late_watchdog_refuses_broken_terminal_lineage(tmp_path: Path, defect: str) -> None:
    result, watchdog = _case(tmp_path)
    terminal = json.loads(watchdog.read_text())
    if defect == "wrong_instance":
        terminal["recorded_vast_instance_teardown"]["instance_id"] = "52854485"
    elif defect == "wrong_scope":
        terminal["provider_absence_scope"] = "informational_global_inventory"
    else:
        target = watchdog.with_suffix(".target")
        watchdog.rename(target)
        watchdog.symlink_to(target)
    if defect != "symlink":
        _write(watchdog, terminal)
    with pytest.raises(VastOfficialBillingExtractionError):
        _terminal_evidence(instance_id=52854484, terminal_result_path=result)


def test_old_no_query_billing_gap_is_delivery_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_task_evaluation_policy_canary_dispatcher import _inputs

    result, _watchdog = _case(tmp_path / "dispatches" / "activation-1")
    root = result.parent
    for name in (
        "allocator_invocation_started.json", "allocator_invocation_finished.json",
        "policy_canary_session_authority.json",
        "bundle/native_task_arena_policy_canary_session_bundle_receipt.v1.json",
    ):
        _write(root / name, {})
    _write(root / "dispatch_pending.json", {"status": "awaiting_official_billing"})
    rows = [{
        "status": "blocked", "candidate_policy_queried": False, "arm_moved": False,
        "typed_harness_failure": "cell_not_completed_before_terminal_failure",
        "visual_evidence": {"media_gap": {"type": "cell_not_completed_before_terminal_failure"}},
    } for _ in range(20)]
    joined = {
        "status": "blocked", "candidate_policy_queried": False,
        "episodes": rows, "result_digest": "",
    }
    joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
    _write(root / "policy_canary_terminal_result.json", joined)
    zero = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "status": "provider_zero_confirmed", "api_confirmed": True,
        "provider_zero_verified": True, "live_instance_count": 0,
        "blockers": [], "receipt_digest": "",
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    _write(root / "post_teardown_global_provider_zero.json", zero)

    def gate() -> bool:
        return retained_sparse_billing_gap(
            root, read_json=lambda path, _code: json.loads(path.read_text()),
            sealed_provider_zero=_sealed_provider_zero,
        )
    assert gate() is True
    activation_result, setup_path, activation_path = _inputs(tmp_path / "inputs")
    queue = tmp_path / "queue"
    for name in ("pending", "processing", "completed", "blocked"):
        (queue / name).mkdir(parents=True)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": root.name, "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution", "source_commit": "a" * 40,
        "activation_result": _record(activation_result), "maximum_provider_allocations": 1,
        "retry_cap": 0, "automatic_retry_authorized": False,
        "provider_mutation_performed": False, "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    _write(queue / "blocked" / "activation-1.json", envelope)
    setup_dir = tmp_path / "setups"
    _write(setup_dir / "activation-1.json", json.loads(setup_path.read_text()))
    called = []
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.dispatch_policy_canary_activation",
        lambda **kwargs: called.append(kwargs) or {"status": "awaiting_official_billing", "allocator_invoked": False},
    )
    selected = process_policy_canary_dispatch_queue(
        dispatch_queue_root=queue, execution_setup_root=setup_dir,
        dispatch_root=root.parent, implementation_commit="b" * 40, execute=True,
    )
    assert selected["processed_count"] == 1
    assert len(called) == 1
    assert called[0]["retained_delivery_only"] is True

    from blueprint_pipeline.native_task_arena_policy_canary_session import build_session_authority
    runtime_path = Path(json.loads(activation_result.read_text())["policy_canary_runtime_inputs_path"])
    runtime = json.loads(runtime_path.read_text())
    activation = json.loads(activation_path.read_text())
    resource = runtime["resource_authority"]
    authority = build_session_authority(
        activation_manifest=activation, activation_record=_record(activation_path),
        runtime_inputs=runtime, runtime_input_record=_record(runtime_path),
        resource_name=resource["resource_name"], hard_cap_usd=resource["hard_cap_usd"],
        hard_ttl_seconds=resource["hard_ttl_seconds"],
    )
    (root / "policy_canary_session_authority.json").write_bytes(
        (json.dumps(authority, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    _write(root / "bundle/native_task_arena_policy_canary_session_bundle_receipt.v1.json",
           {"bundle_sha256": "sha256:" + "b" * 64})
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher.validate_provider_bundle",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_policy_canary_dispatcher._materialize_official_billing_if_posted",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("reached_billing_without_allocator")),
    )
    with pytest.raises(RuntimeError, match="reached_billing_without_allocator"):
        dispatch_policy_canary_activation(
            activation_result_path=activation_result, execution_setup_path=setup_path,
            output_root=root, implementation_commit="b" * 40, execute=True,
            allocator_runner=lambda _argv: pytest.fail("old attempt reallocated a GPU"),
            progress_sync_runner=lambda **_kwargs: {"status": "succeeded"},
        )
    rows[0]["candidate_policy_queried"] = True
    joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
    _write(root / "policy_canary_terminal_result.json", joined)
    assert gate() is False
