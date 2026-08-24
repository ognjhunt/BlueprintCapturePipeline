from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_warm_closeout import (
    materialize_expired_warm_closeout,
)
from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_provider_zero,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    authority = {
        "schema_version": "native_task_arena_paid_attempt_authority.v1",
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = _write(tmp_path / "authority.json", authority)
    watchdog_path = _write(
        tmp_path / "watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "provider": "vast",
            "status": "provider_terminal",
            "completed_at": "2026-08-22T06:52:00+00:00",
            "provider_absence_confirmed": True,
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "recorded_vast_instance_teardown": {
                "instance_id": 42,
                "provider_absence_confirmed": True,
            },
            "terminations": [{"instance_id": 42, "status": "absent"}],
        },
    )
    guard_path = _write(
        tmp_path / "guard.json",
        {
            "schema_version": "gpu_spend_guard.v1",
            "generated_at": "2026-08-22T06:53:00+00:00",
            "status": "passed",
            "blockers": [],
            "provider_zero_verified": True,
            "live_instance_count": 0,
            "inventory_results": [
                {"provider": "vast", "status": "succeeded", "row_count": 0}
            ],
        },
    )
    adapter_path = _write(
        tmp_path / "adapter.json",
        {
            "schema_version": "vast_provider_adapter_result.v1",
            "status": "completed",
            "vast_instance_ids": [42],
            "retained_owned": True,
            "continuing_spend_from_this_run": True,
        },
    )
    teardown_path = _write(
        tmp_path / "teardown.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "retained_owned",
            "vast_instance_ids": [42],
            "continuing_spend_from_this_run": True,
        },
    )
    cleanup_path = _write(
        tmp_path / "cleanup.json",
        {"all_objects_absent": True, "signed_url_files_removed": True},
    )
    result_path = _write(
        tmp_path / "result.json",
        {
            "schema_version": "native_task_arena_vast_run.v1",
            "status": "blocked",
            "authorization_consumption": {
                "status": "consumed",
                "authorization_digest": authority["authorization_digest"],
            },
            "bundle_sha256": "sha256:" + "a" * 64,
            "hard_cap_usd": 2.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
            "estimated_cost_usd": 0.2,
            "all_staged_objects_absent": True,
            "continuing_spend_from_this_run": True,
            "warm_session": {
                "status": "ready",
                "instance_id": 42,
                "continuing_spend": True,
                "watchdog_pid": 100,
                "watchdog_deadline_epoch": 2_000.0,
                "watchdog_out_dir": str(watchdog_path.parent),
                "watchdog_pod_name_prefix": "blueprint-original-",
            },
            "adapter_result_path": str(adapter_path),
            "teardown_manifest_path": str(teardown_path),
            "watchdog_receipt_path": str(watchdog_path),
            "object_store_cleanup_path": str(cleanup_path),
        },
    )
    return authority_path, result_path, guard_path


def test_materializes_terminal_derivatives_without_provider_mutation(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    output = tmp_path / "closeout"

    receipt = materialize_expired_warm_closeout(
        authority_path=authority,
        retained_result_path=retained,
        provider_zero_guard_path=guard,
        output_dir=output,
    )

    result = json.loads((output / "adp_arena_vast_result.json").read_text())
    teardown = json.loads((output / "vast_teardown_manifest.json").read_text())
    adapter = json.loads((output / "vast_provider_adapter_result.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["provider_mutation_performed"] is False
    assert result["status"] == "blocked"
    assert result["continuing_spend_from_this_run"] is False
    assert result["warm_session"]["continuing_spend"] is False
    assert teardown["status"] == "completed"
    assert teardown["provider_instance_absent"] is True
    assert adapter["retained_owned"] is False
    assert adapter["continuing_spend_from_this_run"] is False
    provider_zero = materialize_native_task_arena_provider_zero(
        authority_path=authority,
        result_path=output / "adp_arena_vast_result.json",
        output_path=tmp_path / "provider_zero.json",
    )
    assert provider_zero["provider_zero_confirmed"] is True
    assert provider_zero["continuing_spend_from_this_run"] is False


def test_refuses_nonterminal_watchdog(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    result = json.loads(retained.read_text())
    watchdog = Path(result["watchdog_receipt_path"])
    value = json.loads(watchdog.read_text())
    value["status"] = "armed"
    watchdog.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="expired_warm_closeout_invalid"):
        materialize_expired_warm_closeout(
            authority_path=authority,
            retained_result_path=retained,
            provider_zero_guard_path=guard,
            output_dir=tmp_path / "closeout",
        )


def test_materializes_after_exact_watchdog_supersession(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    retained_value = json.loads(retained.read_text())
    original_watchdog = Path(retained_value["watchdog_receipt_path"])
    _write(
        original_watchdog,
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "armed",
            "provider": "vast",
            "pid": 100,
            "deadline_epoch": 2_000.0,
            "watchdog_out_dir": str(original_watchdog.parent),
            "pod_name_prefix": "blueprint-original-",
        },
    )
    successor_dir = tmp_path / "successor" / "independent_vast_watchdog"
    successor_dir.mkdir(parents=True)
    (successor_dir / "started_vast_instance_id.txt").write_text(
        "42\n", encoding="utf-8"
    )
    successor_watchdog = _write(
        successor_dir / "groot_oscar_runpod_canary_watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "provider_terminal",
            "provider": "vast",
            "pid": 200,
            "deadline_epoch": 3_000.0,
            "watchdog_out_dir": str(successor_dir),
            "completed_at": "2026-08-22T06:52:00+00:00",
            "provider_absence_confirmed": True,
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "recorded_vast_instance_teardown": {
                "instance_id": 42,
                "provider_absence_confirmed": True,
            },
            "terminations": [{"instance_id": 42, "status": "absent"}],
        },
    )
    inspection = {
        "api_confirmed": True,
        "instance_id": "42",
        "actual_status": "running",
    }
    supersession = _write(
        tmp_path / "successor" / "vast_independent_watchdog_supersession.json",
        {
            "schema_version": "vast_independent_watchdog_supersession.v1",
            "status": "superseded",
            "instance_id": 42,
            "predecessor_watchdog_pid": 100,
            "predecessor_watchdog_deadline_epoch": 2_000.0,
            "predecessor_watchdog_retired": True,
            "successor_watchdog_pid": 200,
            "successor_watchdog_deadline_epoch": 3_000.0,
            "successor_watchdog_out_dir": str(successor_dir),
            "provider_inspect_before": inspection,
            "provider_inspect_successor_armed": inspection,
            "provider_inspect_after_transfer": inspection,
            "provider_instance_running_after_transfer": True,
        },
    )

    receipt = materialize_expired_warm_closeout(
        authority_path=authority,
        retained_result_path=retained,
        provider_zero_guard_path=guard,
        watchdog_supersession_path=supersession,
        successor_watchdog_path=successor_watchdog,
        output_dir=tmp_path / "closeout",
    )

    assert receipt["status"] == "completed"
    assert receipt["watchdog_supersession"]["path"] == str(supersession)


def test_refuses_supersession_bound_to_different_instance(tmp_path: Path) -> None:
    authority, retained, guard = _fixture(tmp_path)
    successor_dir = tmp_path / "successor" / "independent_vast_watchdog"
    successor_dir.mkdir(parents=True)
    (successor_dir / "started_vast_instance_id.txt").write_text(
        "42\n", encoding="utf-8"
    )
    successor = _write(
        successor_dir / "groot_oscar_runpod_canary_watchdog.json",
        {"schema_version": "groot_oscar_runpod_canary_watchdog.v1"},
    )
    supersession = _write(
        tmp_path / "supersession.json",
        {
            "schema_version": "vast_independent_watchdog_supersession.v1",
            "status": "superseded",
            "instance_id": 99,
        },
    )

    with pytest.raises(ValueError, match="expired_warm_closeout_invalid"):
        materialize_expired_warm_closeout(
            authority_path=authority,
            retained_result_path=retained,
            provider_zero_guard_path=guard,
            watchdog_supersession_path=supersession,
            successor_watchdog_path=successor,
            output_dir=tmp_path / "closeout",
        )
