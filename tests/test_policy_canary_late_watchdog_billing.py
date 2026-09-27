"""A later watchdog close may settle an immutable pre-close allocator snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    _adapter_instance_ids,
    _sealed_provider_zero,
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


def test_old_no_query_billing_gap_is_delivery_only(tmp_path: Path) -> None:
    result, _watchdog = _case(tmp_path)
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
    rows[0]["candidate_policy_queried"] = True
    joined["result_digest"] = canonical_digest(joined, digest_field="result_digest")
    _write(root / "policy_canary_terminal_result.json", joined)
    assert gate() is False
