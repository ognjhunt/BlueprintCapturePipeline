from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import blueprint_pipeline.native_task_arena_paid_authority as paid


COMMIT = "a" * 40


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _predecessor(root: Path) -> dict[str, Path]:
    root.mkdir()
    authority = {
        "schema_version": "paired_target_native_import_paid_attempt_authority.v1",
        "bundle_sha256": "sha256:" + "b" * 64,
        "hard_attempt_spend_cap_usd": 0.75,
        "maximum_single_resource_ttl_seconds": 3600,
        "aggregate_goal_spend_before_attempt_usd": 11.211507,
        "aggregate_goal_spend_cap_usd": 12.0,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = root / "authority.json"
    write_json(authority_path, authority)
    result = {
        "schema_version": "paired_target_native_import_vast_run.v1",
        "status": "completed",
        "bundle_sha256": authority["bundle_sha256"],
        "estimated_cost_usd": 0.092936,
        "hard_cap_usd": 0.75,
        "hard_ttl_seconds": 3600,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "authorization_consumption": {
            "authorization_digest": authority["authorization_digest"]
        },
    }
    canonical_result_path = root / "canonical_result.json"
    write_json(canonical_result_path, result)
    alias_result_path = root / "allocator_alias_result.json"
    alias_result_path.write_bytes(canonical_result_path.read_bytes())
    zero = {
        "schema_version": "paired_target_native_import_provider_zero.v1",
        "status": "completed",
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(canonical_result_path),
        "provider_zero_confirmed": True,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    zero_path = root / "provider_zero.json"
    write_json(zero_path, zero)
    return {
        "authority": authority_path,
        "result": alias_result_path,
        "canonical_result": canonical_result_path,
        "zero": zero_path,
    }


def _prepared_bundle(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    bundle_path = root / "native_task_arena_provider_bundle.zip"
    bundle_path.write_bytes(b"bundle")
    receipt_path = root / "native_task_arena_provider_bundle_receipt.v1.json"
    write_json(receipt_path, {"execution_mode": "construction_canary"})
    return receipt_path, {
        "execution_mode": "construction_canary",
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha(bundle_path),
        "input_digest": "sha256:" + "c" * 64,
        "packet_receipt_digest": "sha256:" + "d" * 64,
        "runtime_source_packet": {"receipt_digest": "sha256:" + "e" * 64},
        "implementation_commit": COMMIT,
        "container_image": "nvcr.io/nvidia/isaac-sim:5.0.0",
        "policy_candidate_id": None,
    }


def test_real_shape_predecessor_alias_and_authority_are_digest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = _predecessor(tmp_path / "predecessor")
    receipt_path, prepared = _prepared_bundle(tmp_path / "bundle")
    monkeypatch.setattr(paid, "_bundle_loader", lambda _mode: lambda *_args, **_kwargs: prepared)
    reconciled = {
        "prior_terminal_attempts": [{"result": _record(predecessor["canonical_result"])}],
        "reconciliation": {
            "path": str(tmp_path / "reconciliation.json"),
            "sha256": "sha256:" + "9" * 64,
        },
        "actual_total_usd": 0.025,
    }
    monkeypatch.setattr(paid, "bind_lane_prior_spend", lambda **_kwargs: reconciled)
    monkeypatch.setattr(
        paid, "validate_bound_lane_prior_spend", lambda *_args, **_kwargs: reconciled
    )

    authority = paid.materialize_native_task_arena_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_authority_path=predecessor["authority"],
        prior_result_path=predecessor["result"],
        prior_provider_zero_path=predecessor["zero"],
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
        authorization_reference="user-directed native construction",
        authorized_by="user",
        authorized_on="2026-08-13",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.6,
        hard_cap_usd=0.6,
        hard_ttl_seconds=3600,
        output_path=tmp_path / "attempt_authority.json",
    )

    assert authority["aggregate_goal_spend_before_attempt_usd"] == 11.236507
    assert authority["prior_terminal_attempt"]["attempt_cost_usd"] == 0.092936
    assert authority["prior_terminal_attempt"]["actual_provider_charge_usd"] == 0.025
    assert authority["prior_terminal_attempt"]["terminal_result"]["path"] == str(
        predecessor["canonical_result"]
    )
    assert paid.validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=prepared,
        max_hourly_rate_usd=0.6,
        hard_cap_usd=0.6,
        hard_ttl_seconds=3600,
    )["authorization_digest"] == authority["authorization_digest"]

    tampered = dict(authority)
    tampered["bundle_sha256"] = "sha256:" + "f" * 64
    tampered["authorization_digest"] = canonical_digest(
        tampered, digest_field="authorization_digest"
    )
    with pytest.raises(ValueError, match="bundle_sha256_mismatch"):
        paid.validate_native_task_arena_paid_attempt_authority(
            tampered,
            prepared_bundle=prepared,
            max_hourly_rate_usd=0.6,
            hard_cap_usd=0.6,
            hard_ttl_seconds=3600,
        )


def test_authority_consumption_is_one_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paid, "consumption_root", lambda: tmp_path / "consumption")
    authority = {"authorization_digest": "sha256:" + "a" * 64, "bundle_sha256": "b"}
    assert paid.consume_native_task_arena_authority_once(authority)["status"] == "consumed"
    second = paid.consume_native_task_arena_authority_once(authority)
    assert second == {
        "status": "blocked",
        "blockers": ["native_task_arena_authority_consumed"],
    }


def test_provider_zero_requires_watchdog_api_inventory(tmp_path: Path) -> None:
    authority = {
        "schema_version": paid.AUTHORITY_SCHEMA_VERSION,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "authority.json"
    write_json(authority_path, authority)
    cleanup = tmp_path / "cleanup.json"
    adapter = tmp_path / "adapter.json"
    teardown = tmp_path / "teardown.json"
    watchdog = tmp_path / "watchdog.json"
    write_json(
        cleanup, {"all_objects_absent": True, "signed_url_files_removed": True}
    )
    write_json(adapter, {"continuing_spend_from_this_run": False})
    write_json(teardown, {"continuing_spend_from_this_run": False})
    write_json(
        watchdog,
        {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "final_global_inventory": {
                "api_confirmed": True,
                "live_resource_count": 0,
            },
        },
    )
    result = {
        "schema_version": "native_task_arena_vast_run.v1",
        "status": "completed",
        "authorization_consumption": {
            "authorization_digest": authority["authorization_digest"]
        },
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "watchdog_receipt_path": str(watchdog),
        "object_store_cleanup_path": str(cleanup),
        "adapter_result_path": str(adapter),
        "teardown_manifest_path": str(teardown),
        "estimated_cost_usd": 0.1,
    }
    result_path = tmp_path / "result.json"
    write_json(result_path, result)

    receipt = paid.materialize_native_task_arena_provider_zero(
        authority_path=authority_path,
        result_path=result_path,
        output_path=tmp_path / "provider_zero.json",
    )
    assert receipt["provider_zero_confirmed"] is True

    watchdog_value = json.loads(watchdog.read_text())
    watchdog_value["final_global_inventory"]["live_resource_count"] = 1
    write_json(watchdog, watchdog_value)
    with pytest.raises(ValueError, match="native_task_arena_provider_zero_invalid"):
        paid.materialize_native_task_arena_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            output_path=tmp_path / "invalid_zero.json",
        )
