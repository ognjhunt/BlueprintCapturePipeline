from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.policy_canary_provider_null_closeout import proven_provider_null_closeout


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_provider_null_requires_definite_refusal_and_sealed_closeout(tmp_path: Path) -> None:
    inner_path = tmp_path / "provider.json"
    teardown_path = tmp_path / "teardown.json"
    inner: dict[str, object] = {
        "status": "failed",
        "provider_create_attempted": True,
        "vast_instance_ids": [],
        "vast_side_effects_may_have_occurred": False,
        "continuing_spend_from_this_run": False,
        "create_failure_diagnosis": {
            "http_status_code": 410,
            "definite_create_refusal": True,
            "create_inventory_verified": True,
            "create_inventory_http_status_code": 200,
            "create_produced_no_instance": True,
            "matching_attempt_instance_ids": [],
            "attempted_labels": ["test-policy"],
        },
        "provider_attempt_classification": {
            "classification": "pre_execution_provider_null",
            "scientific_attempt_consumed": False,
        },
    }
    _write(inner_path, inner)
    _write(teardown_path, {"status": "completed", "vast_instance_ids": [], "continuing_spend_from_this_run": False})
    allocator = {
        "status": "blocked",
        "scientific_attempt_started": False,
        "candidate_policy_queried": False,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "adapter_result_path": str(inner_path),
        "teardown_manifest_path": str(teardown_path),
        "provider_closeout": {
            "provider_zero_confirmed": True,
            "adapter_result": _record(inner_path),
            "teardown_manifest": _record(teardown_path),
        },
        "independent_watchdog_close": {
            "status": "cancelled_no_allocation",
            "provider_mutations_performed": 0,
        },
    }
    assert proven_provider_null_closeout(allocator, root=tmp_path, record_file=_record)

    allocator["independent_watchdog_close"] = {"status": "retained_until_hard_ttl", "provider_mutations_performed": 0}
    assert proven_provider_null_closeout(allocator, root=tmp_path, record_file=_record) is None
    allocator["independent_watchdog_close"].update({
        "reason": "provider_allocation_identity_ambiguous",
        "watchdog_retention_liveness_confirmed": True,
        "watchdog_armed_before_allocation": True,
        "instance_ids": [],
    })
    retained = proven_provider_null_closeout(allocator, root=tmp_path, record_file=_record)
    assert retained is not None and retained["watchdog_retained_until_hard_ttl"] is True
    allocator["independent_watchdog_close"] = {"status": "cancelled_no_allocation", "provider_mutations_performed": 0}

    diagnosis = inner["create_failure_diagnosis"]
    assert isinstance(diagnosis, dict)
    diagnosis["http_status_code"] = 400
    diagnosis["error_preview"] = ""
    _write(inner_path, inner)
    allocator["provider_closeout"]["adapter_result"] = _record(inner_path)
    assert proven_provider_null_closeout(allocator, root=tmp_path, record_file=_record) is None

    diagnosis["http_status_code"] = 410
    diagnosis["matching_attempt_instance_ids"] = [99]
    _write(inner_path, inner)
    allocator["provider_closeout"]["adapter_result"] = _record(inner_path)
    assert proven_provider_null_closeout(allocator, root=tmp_path, record_file=_record) is None
