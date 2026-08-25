from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import blueprint_pipeline.native_task_arena_paid_authority as paid
import blueprint_pipeline.task_evaluation_launch_dispatcher as dispatcher
from blueprint_pipeline.task_evaluation_immutable_input_resolver import (
    STAGING_RECEIPT_ENV,
)


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
        max_hourly_rate_usd=0.64,
        hard_cap_usd=0.5,
        hard_ttl_seconds=2_800,
        output_path=tmp_path / "attempt_authority.json",
    )

    assert authority["aggregate_goal_spend_before_attempt_usd"] == 11.236507
    # A current program-level ceiling may explicitly supersede the lower
    # immutable ceiling recorded by the predecessor.  Per-attempt limits and
    # the predecessor's spend still remain digest-bound.
    assert authority["aggregate_goal_spend_cap_usd"] == 50.0
    assert authority["prior_terminal_attempt"]["attempt_cost_usd"] == 0.092936
    assert authority["prior_terminal_attempt"]["actual_provider_charge_usd"] == 0.025
    assert authority["prior_terminal_attempt"]["terminal_result"]["path"] == str(
        predecessor["canonical_result"]
    )
    assert authority["retain_warm_session"] is False
    assert paid.validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=prepared,
        max_hourly_rate_usd=0.64,
        hard_cap_usd=0.5,
        hard_ttl_seconds=2_800,
    )["authorization_digest"] == authority["authorization_digest"]

    declared = [
        receipt_path,
        predecessor["authority"],
        predecessor["canonical_result"],
        predecessor["zero"],
    ]
    profile = {
        "profile_id": "authority-closure-test",
        "profile_digest": "sha256:" + "0" * 64,
        "immutable_inputs": [
            {
                "name": f"input-{index}",
                "path": str(path.resolve()),
                "digest": _sha(path),
            }
            for index, path in enumerate(declared)
        ],
    }
    dispatcher._stage_profile_immutable_inputs(
        profile=profile,
        run_root=tmp_path / "staged-run",
        allocator_argv=[],
    )
    monkeypatch.setenv(
        STAGING_RECEIPT_ENV,
        str(tmp_path / "staged-run" / "immutable_input_staging_receipt.json"),
    )
    for path in declared:
        path.write_bytes(b"tampered-after-staging")
    assert paid.validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=prepared,
        max_hourly_rate_usd=0.64,
        hard_cap_usd=0.5,
        hard_ttl_seconds=2_800,
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
            max_hourly_rate_usd=0.64,
            hard_cap_usd=0.5,
            hard_ttl_seconds=2_800,
        )


def test_new_lane_genesis_binds_project_spend_and_fresh_provider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, prepared = _prepared_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        paid, "_bundle_loader", lambda _mode: lambda *_args, **_kwargs: prepared
    )
    reconciliation_path = tmp_path / "project-spend.json"
    write_json(reconciliation_path, {"sealed": True})
    project_record = _record(reconciliation_path)
    project_spend = {
        "receipt_digest": "sha256:" + "8" * 64,
        "total_cost_usd": 39.791914,
        "entries": [{"attempt_id": "prior-project-attempt"}],
    }
    monkeypatch.setattr(
        paid,
        "validate_project_spend_reconciliation",
        lambda *_args, **_kwargs: (project_spend, project_record),
    )
    zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "observed_at_utc": "2026-08-25T14:25:00+00:00",
        "api_confirmed": True,
        "global_live_resource_count": 0,
        "provider_zero": True,
        "inventory": [],
        "api_command": ["vastai", "show", "instances", "--raw"],
        "raw_secret_values_recorded": False,
        "stderr_present": False,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    zero_path = tmp_path / "provider-zero.json"
    write_json(zero_path, zero)

    authority = paid.materialize_native_task_arena_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        project_spend_reconciliation_path=reconciliation_path,
        initial_provider_zero_path=zero_path,
        authorization_reference="user-authorized independent scene lane",
        authorized_by="user",
        authorized_on="2026-08-25T14:30:00+00:00",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.8,
        hard_cap_usd=0.75,
        hard_ttl_seconds=3_300,
        output_path=tmp_path / "authority.json",
    )

    assert authority["lineage_kind"] == "project_spend_genesis"
    assert authority["prior_terminal_attempts"] == []
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 39.791914
    assert authority["project_spend_reconciliation"] == project_record
    assert authority["initial_provider_zero"]["provider_zero_digest"] == zero[
        "provider_zero_digest"
    ]
    assert paid.validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=prepared,
        max_hourly_rate_usd=0.8,
        hard_cap_usd=0.75,
        hard_ttl_seconds=3_300,
    )["authorization_digest"] == authority["authorization_digest"]


def test_terminal_continuation_uses_newer_conservative_project_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = _predecessor(tmp_path / "predecessor")
    receipt_path, prepared = _prepared_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        paid, "_bundle_loader", lambda _mode: lambda *_args, **_kwargs: prepared
    )
    reconciled = {
        "prior_terminal_attempts": [
            {"result": _record(predecessor["canonical_result"])}
        ],
        "reconciliation": _record(predecessor["canonical_result"]),
        "actual_total_usd": 0.025,
    }
    monkeypatch.setattr(paid, "bind_lane_prior_spend", lambda **_kwargs: reconciled)
    monkeypatch.setattr(
        paid, "validate_bound_lane_prior_spend", lambda *_args, **_kwargs: reconciled
    )
    project_path = tmp_path / "project-spend.json"
    write_json(project_path, {"total_cost_usd": 43.197914})
    project_record = _record(project_path)
    monkeypatch.setattr(
        paid,
        "validate_project_spend_reconciliation",
        lambda *_args, **_kwargs: (
            {"total_cost_usd": 43.197914},
            project_record,
        ),
    )

    authority = paid.materialize_native_task_arena_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_authority_path=predecessor["authority"],
        prior_result_path=predecessor["result"],
        prior_provider_zero_path=predecessor["zero"],
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
        project_spend_reconciliation_path=project_path,
        authorization_reference="user-authorized conservative continuation",
        authorized_by="user",
        authorized_on="2026-08-25T20:41:55Z",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.8,
        hard_cap_usd=0.75,
        hard_ttl_seconds=3_300,
        output_path=tmp_path / "authority.json",
    )

    assert authority["lineage_kind"] == "terminal_predecessor"
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 43.197914
    assert authority["project_spend_reconciliation"] == project_record
    assert paid.validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=prepared,
        max_hourly_rate_usd=0.8,
        hard_cap_usd=0.75,
        hard_ttl_seconds=3_300,
    )["authorization_digest"] == authority["authorization_digest"]


def test_new_lane_genesis_refuses_stale_provider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, prepared = _prepared_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        paid, "_bundle_loader", lambda _mode: lambda *_args, **_kwargs: prepared
    )
    reconciliation_path = tmp_path / "project-spend.json"
    write_json(reconciliation_path, {"sealed": True})
    monkeypatch.setattr(
        paid,
        "validate_project_spend_reconciliation",
        lambda *_args, **_kwargs: (
            {"receipt_digest": "sha256:" + "8" * 64, "total_cost_usd": 1.0},
            _record(reconciliation_path),
        ),
    )
    zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "observed_at_utc": "2026-08-25T13:00:00+00:00",
        "api_confirmed": True,
        "global_live_resource_count": 0,
        "provider_zero": True,
        "inventory": [],
        "api_command": ["vastai", "show", "instances", "--raw"],
        "raw_secret_values_recorded": False,
        "stderr_present": False,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    zero_path = tmp_path / "provider-zero.json"
    write_json(zero_path, zero)

    with pytest.raises(ValueError, match="initial_authority_invalid"):
        paid.materialize_native_task_arena_paid_attempt_authority(
            bundle_receipt_path=receipt_path,
            project_spend_reconciliation_path=reconciliation_path,
            initial_provider_zero_path=zero_path,
            authorization_reference="new lane",
            authorized_by="user",
            authorized_on="2026-08-25T14:30:00+00:00",
            blueprint_commit=COMMIT,
            max_hourly_rate_usd=0.8,
            hard_cap_usd=0.75,
            hard_ttl_seconds=3_300,
            output_path=tmp_path / "authority.json",
        )


def test_warm_retention_intent_is_digest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = _predecessor(tmp_path / "predecessor")
    receipt_path, prepared = _prepared_bundle(tmp_path / "bundle")
    prepared["execution_mode"] = "controls"
    write_json(receipt_path, {"execution_mode": "controls"})
    monkeypatch.setattr(
        paid, "_bundle_loader", lambda _mode: lambda *_args, **_kwargs: prepared
    )
    reconciled = {
        "prior_terminal_attempts": [
            {"result": _record(predecessor["canonical_result"])}
        ],
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
        authorization_reference="user-directed warm controls session",
        authorized_by="user",
        authorized_on="2026-08-21",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.6,
        hard_cap_usd=0.6,
        hard_ttl_seconds=3600,
        output_path=tmp_path / "attempt_authority.json",
        retain_warm_session=True,
    )

    assert authority["retain_warm_session"] is True
    assert paid.validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=prepared,
        max_hourly_rate_usd=0.6,
        hard_cap_usd=0.6,
        hard_ttl_seconds=3600,
        retain_warm_session=True,
    )["authorization_digest"] == authority["authorization_digest"]
    with pytest.raises(ValueError, match="retain_warm_session_mismatch"):
        paid.validate_native_task_arena_paid_attempt_authority(
            authority,
            prepared_bundle=prepared,
            max_hourly_rate_usd=0.6,
            hard_cap_usd=0.6,
            hard_ttl_seconds=3600,
            retain_warm_session=False,
        )


def test_authority_consumption_is_one_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    consumption = tmp_path / "consumption"
    consumption.mkdir(mode=0o700)
    monkeypatch.setattr(paid, "prepare_consumption_root", lambda: consumption)
    authority = {"authorization_digest": "sha256:" + "a" * 64, "bundle_sha256": "b"}
    assert paid.consume_native_task_arena_authority_once(authority)["status"] == "consumed"
    second = paid.consume_native_task_arena_authority_once(authority)
    assert second == {
        "status": "blocked",
        "blockers": ["native_task_arena_authority_consumed"],
    }


def test_authority_consumption_tightens_owned_reconciler_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spend_root = tmp_path / "spend-authority"
    consumption = spend_root / "consumed"
    consumption.mkdir(parents=True)
    consumption.chmod(0o710)
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(spend_root))
    authority = {"authorization_digest": "sha256:" + "b" * 64, "bundle_sha256": "c"}

    result = paid.consume_native_task_arena_authority_once(authority)

    assert result["status"] == "consumed"
    assert consumption.stat().st_mode & 0o777 == 0o700


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


def test_provider_zero_scopes_to_this_lane_when_the_watchdog_says_so(
    tmp_path: Path,
) -> None:
    """An unrelated instance on the account must not block this lane's seal.

    The watchdog sweeps the whole provider account but marks that sweep
    ``global_inventory_informational_only`` and records a lane-scoped
    ``final_inventory`` matched on the run's own name prefix. Gating on the
    global sweep meant a debug pod -- or any concurrent lane -- permanently
    blocked every later arena run, because the receipt is frozen at write time
    and can never re-observe a now-quiet account. r14 was blocked exactly this
    way by an interactive pod that had nothing to do with the run.

    The lane-scoped inventory still gates: this must fail closed when THIS
    run's own resources are still alive.
    """

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
            "global_inventory_informational_only": True,
            "final_global_inventory": {
                "api_confirmed": True,
                "live_resource_count": 1,
                "resources": [{"instance_id": "48118775", "status": "running"}],
            },
            "final_inventory": {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
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
    assert receipt["inventory_scope"] == "recorded_instance_and_lane_prefix"
    # the unrelated instance is still recorded as evidence, just not as the gate
    assert receipt["global_inventory"]["live_resource_count"] == 1

    watchdog_value = json.loads(watchdog.read_text())
    watchdog_value["final_inventory"]["live_resource_count"] = 1
    write_json(watchdog, watchdog_value)
    with pytest.raises(ValueError, match="native_task_arena_provider_zero_invalid"):
        paid.materialize_native_task_arena_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            output_path=tmp_path / "still_spending.json",
        )


def test_every_accepted_predecessor_can_reconcile_its_own_posted_charges() -> None:
    """A chained authority is only issuable if the predecessor reconciles.

    Issuing the next attempt's authority requires the predecessor's official
    posted charges, which are rebuilt by the Vast official-billing extractor
    from the predecessor's terminal result. If the extractor does not accept
    that result's schema, the predecessor can complete, tear down, and confirm
    provider zero -- and still make every later attempt unissuable.

    That is exactly what happened on 2026-08-18: the Arena lane's own
    ``native_task_arena_vast_run.v1`` was missing from the registry, so the
    lane whose contract is "every later attempt follows the prior zero-closed
    native Arena attempt" could not follow its own attempt.
    """

    from blueprint_pipeline.native_task_arena_paid_authority import (
        PREDECESSOR_RESULT_SCHEMAS,
    )
    from blueprint_pipeline.vast_official_billing_extractor import (
        _SUPPORTED_TERMINAL_RESULT_SCHEMAS,
    )

    unreconcilable = sorted(
        set(PREDECESSOR_RESULT_SCHEMAS.values()) - _SUPPORTED_TERMINAL_RESULT_SCHEMAS
    )
    assert unreconcilable == [], (
        "these predecessor result schemas cannot be reconciled, so any "
        f"authority chained off them is unissuable: {unreconcilable}"
    )

    # the Arena lane must be able to follow its own attempt, not only the
    # import probe that precedes the very first one
    assert (
        PREDECESSOR_RESULT_SCHEMAS["native_task_arena_paid_attempt_authority.v1"]
        == "native_task_arena_vast_run.v1"
    )


def test_the_arena_transport_binds_its_closure_artifacts_by_digest() -> None:
    """The extractor refuses closure it cannot pin, so paths are not enough.

    The generic extractor layout reads ``provider_closeout`` and rejects a
    record without path, size, and sha256. The Arena transport reported only
    ``adapter_result_path`` and ``teardown_manifest_path``.
    """

    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "blueprint_pipeline"
        / "adp_isaac_lab_arena_vast.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    closeout_keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "provider_closeout"
                and isinstance(value, ast.Dict)
            ):
                closeout_keys = {
                    inner.value
                    for inner in value.keys
                    if isinstance(inner, ast.Constant)
                }

    assert {
        "adapter_result",
        "teardown_manifest",
        "provider_zero_confirmed",
        "all_staged_objects_absent",
    } <= closeout_keys, closeout_keys


def _failed_sweep_attempt(tmp_path: Path) -> tuple[Path, Path]:
    """An attempt that tore down cleanly but could not finish its account sweep."""

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
    write_json(cleanup, {"all_objects_absent": True, "signed_url_files_removed": True})
    write_json(adapter, {"continuing_spend_from_this_run": False})
    write_json(
        teardown, {"status": "completed", "continuing_spend_from_this_run": False}
    )
    write_json(
        watchdog,
        {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            # this attempt's own instance was observed gone over a real response
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
            # the account-wide sweep did not complete
            "final_global_inventory": {
                "api_confirmed": False,
                "live_resource_count": None,
                "blockers": ["vast_billable_inventory_failed"],
            },
        },
    )
    result = {
        "schema_version": "native_task_arena_vast_run.v1",
        "status": "blocked",
        "authorization_consumption": {
            "authorization_digest": authority["authorization_digest"]
        },
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "watchdog_receipt_path": str(watchdog),
        "object_store_cleanup_path": str(cleanup),
        "adapter_result_path": str(adapter),
        "teardown_manifest_path": str(teardown),
        "estimated_cost_usd": 0.184878,
    }
    result_path = tmp_path / "result.json"
    write_json(result_path, result)
    return authority_path, result_path


def _guard(tmp_path: Path, *, generated_at: str, name: str = "guard.json") -> Path:
    path = tmp_path / name
    write_json(
        path,
        {
            "schema_version": "gpu_spend_guard.v1",
            "status": "passed",
            "provider_zero_verified": True,
            "live_instance_count": 0,
            "total_burn_per_hour_usd": 0,
            "generated_at": generated_at,
            "inventory_results": [
                {
                    "provider": provider,
                    "required": True,
                    "status": "succeeded",
                    "row_count": 0,
                }
                for provider in ("vast", "runpod", "digitalocean")
            ],
        },
    )
    return path


def test_a_failed_account_sweep_is_recoverable_from_a_fresh_global_zero(
    tmp_path: Path,
) -> None:
    """An attempt that spent money must be accountable even if its sweep failed.

    On 2026-08-18 an Arena attempt destroyed its instance, confirmed absence
    for its own label over a 200 response, and then could not complete the
    account-wide inventory. It could not seal its own zero, and an unsealed
    attempt blocks every authority chained after it -- so the ledger would have
    had to either stall or omit $0.18 that was really spent.
    """

    now = datetime(2026, 8, 18, 19, 30, tzinfo=timezone.utc)
    authority_path, result_path = _failed_sweep_attempt(tmp_path)
    guard = _guard(tmp_path, generated_at="2026-08-18T19:25:00+00:00")

    receipt = paid.materialize_native_task_arena_recovered_provider_zero(
        authority_path=authority_path,
        result_path=result_path,
        global_zero_guard_path=guard,
        output_path=tmp_path / "recovered.json",
        now=now,
    )

    assert receipt["status"] == "completed_recovered_provider_zero"
    assert receipt["provider_zero_confirmed"] is True
    assert receipt["recovery_reason"] == "attempt_global_inventory_sweep_failed"
    # the spend is carried, not dropped
    assert receipt["estimated_cost_usd"] == 0.184878
    # and the receipt says which fresh evidence replaced the failed sweep
    assert receipt["recovered_global_zero_guard"]["path"] == str(guard)


def test_recovery_accepts_additional_required_providers_only_when_zero(
    tmp_path: Path,
) -> None:
    """Adding AWS to the required fleet strengthens rather than breaks zero."""

    now = datetime(2026, 8, 18, 19, 30, tzinfo=timezone.utc)
    authority_path, result_path = _failed_sweep_attempt(tmp_path)
    guard = _guard(tmp_path, generated_at="2026-08-18T19:29:00+00:00")
    value = json.loads(guard.read_text())
    value["inventory_results"].append(
        {
            "provider": "aws",
            "required": True,
            "status": "succeeded",
            "row_count": 0,
        }
    )
    write_json(guard, value)

    receipt = paid.materialize_native_task_arena_recovered_provider_zero(
        authority_path=authority_path,
        result_path=result_path,
        global_zero_guard_path=guard,
        output_path=tmp_path / "recovered-with-aws.json",
        now=now,
    )

    assert receipt["provider_zero_confirmed"] is True

    value["inventory_results"][-1]["row_count"] = 1
    write_json(guard, value)
    with pytest.raises(ValueError, match="recovered_provider_zero_invalid"):
        paid.materialize_native_task_arena_recovered_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            global_zero_guard_path=guard,
            output_path=tmp_path / "must-not-exist.json",
            now=now,
        )


def test_recovery_refuses_stale_or_non_zero_evidence(tmp_path: Path) -> None:
    """Recovery is not a way around proving the account is empty now."""

    now = datetime(2026, 8, 18, 19, 30, tzinfo=timezone.utc)
    authority_path, result_path = _failed_sweep_attempt(tmp_path)

    # older than the freshness bound: proves the past, not the present
    stale = _guard(
        tmp_path, generated_at="2026-08-18T19:00:00+00:00", name="stale.json"
    )
    with pytest.raises(ValueError, match="recovered_provider_zero_invalid"):
        paid.materialize_native_task_arena_recovered_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            global_zero_guard_path=stale,
            output_path=tmp_path / "a.json",
            now=now,
        )

    # a guard that did not itself pass
    failing = _guard(
        tmp_path, generated_at="2026-08-18T19:29:00+00:00", name="failing.json"
    )
    value = json.loads(failing.read_text())
    value["inventory_results"][0]["row_count"] = 1
    write_json(failing, value)
    with pytest.raises(ValueError, match="recovered_provider_zero_invalid"):
        paid.materialize_native_task_arena_recovered_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            global_zero_guard_path=failing,
            output_path=tmp_path / "b.json",
            now=now,
        )

    # a guard missing a required provider is not account-wide
    partial = _guard(
        tmp_path, generated_at="2026-08-18T19:29:00+00:00", name="partial.json"
    )
    value = json.loads(partial.read_text())
    value["inventory_results"] = value["inventory_results"][:2]
    write_json(partial, value)
    with pytest.raises(ValueError, match="recovered_provider_zero_invalid"):
        paid.materialize_native_task_arena_recovered_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            global_zero_guard_path=partial,
            output_path=tmp_path / "c.json",
            now=now,
        )


def test_recovery_still_requires_the_attempt_instance_to_be_observed_gone(
    tmp_path: Path,
) -> None:
    """A fresh account-wide zero does not excuse never seeing this instance go."""

    now = datetime(2026, 8, 18, 19, 30, tzinfo=timezone.utc)
    authority_path, result_path = _failed_sweep_attempt(tmp_path)
    guard = _guard(tmp_path, generated_at="2026-08-18T19:29:00+00:00")

    result = json.loads(result_path.read_text())
    watchdog_path = Path(result["watchdog_receipt_path"])
    watchdog = json.loads(watchdog_path.read_text())
    watchdog["final_inventory"]["api_confirmed"] = False
    write_json(watchdog_path, watchdog)

    with pytest.raises(ValueError, match="recovered_provider_zero_invalid"):
        paid.materialize_native_task_arena_recovered_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            global_zero_guard_path=guard,
            output_path=tmp_path / "d.json",
            now=now,
        )


def test_a_recovered_predecessor_zero_is_still_chainable() -> None:
    """A recovered zero proves the same absence, so it cannot be a dead end.

    The chain validator admitted only ``completed``. Sealing a recovered zero
    would then have produced a receipt no later authority could consume, which
    is the same stall it exists to prevent -- and the import lane, which has
    had a recovered seal all along, was already exposed to it.
    """

    assert paid.ACCEPTED_PREDECESSOR_ZERO_STATUSES == {
        "completed",
        "completed_recovered_provider_zero",
        "completed_preallocation_provider_zero",
    }


def test_authority_adds_digest_bound_supplemental_actuals_without_assigning_them_to_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = _predecessor(tmp_path / "predecessor")
    receipt_path, prepared = _prepared_bundle(tmp_path / "bundle")
    monkeypatch.setattr(
        paid, "_bundle_loader", lambda _mode: lambda *_args, **_kwargs: prepared
    )
    supplemental = [tmp_path / "old-pi.json", tmp_path / "retained-controls.json"]
    for path in supplemental:
        write_json(path, {"status": "blocked"})
    reconciled = {
        "prior_terminal_attempts": [
            {
                "result": _record(predecessor["canonical_result"]),
                "actual_provider_charge_usd": 0.1,
            },
            {
                "result": _record(supplemental[0]),
                "actual_provider_charge_usd": 0.3,
            },
            {
                "result": _record(supplemental[1]),
                "actual_provider_charge_usd": 1.157,
            },
        ],
        "reconciliation": {
            "path": str(tmp_path / "reconciliation.json"),
            "sha256": "sha256:" + "9" * 64,
        },
        "actual_total_usd": 1.557,
    }
    observed = {}

    def bind(**kwargs):
        observed.update(kwargs)
        return reconciled

    monkeypatch.setattr(paid, "bind_lane_prior_spend", bind)
    monkeypatch.setattr(
        paid, "validate_bound_lane_prior_spend", lambda *_args, **_kwargs: reconciled
    )
    authority = paid.materialize_native_task_arena_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        prior_authority_path=predecessor["authority"],
        prior_result_path=predecessor["result"],
        prior_provider_zero_path=predecessor["zero"],
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
        supplemental_prior_result_paths=supplemental,
        authorization_reference="complete historical spend",
        authorized_by="user",
        authorized_on="2026-08-25",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.2,
        hard_cap_usd=0.5,
        hard_ttl_seconds=9000,
        output_path=tmp_path / "attempt_authority.json",
    )

    assert len(observed["prior_result_paths"]) == 3
    assert authority["prior_terminal_attempt"]["actual_provider_charge_usd"] == 0.1
    assert authority["prior_actual_provider_spend_usd"] == 1.557
    assert authority["aggregate_goal_spend_before_attempt_usd"] == 12.768507


def _watchdog_not_armed_fixture(root: Path) -> dict[str, Path]:
    root.mkdir()
    attempt_root = (
        root
        / "allocator"
        / "arena-policy-diagnostic-job"
        / "attempts"
        / "attempt_001"
    )
    (attempt_root / "object_store_staging").mkdir(parents=True)
    observed_at = datetime.now(timezone.utc)
    authority = {
        "schema_version": paid.AUTHORITY_SCHEMA_VERSION,
        "execution_mode": "policy_diagnostic",
        "bundle_sha256": "sha256:" + "b" * 64,
        "hard_attempt_spend_cap_usd": 0.5,
        "maximum_single_resource_ttl_seconds": 9000,
        "aggregate_goal_spend_before_attempt_usd": 39.540914,
        "aggregate_goal_spend_cap_usd": 50.0,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = root / "authority.json"
    write_json(authority_path, authority)
    watchdog = {
        "schema_version": paid.WATCHDOG_HANDOFF_SCHEMA_VERSION,
        "status": "blocked",
        "watchdog_armed_before_allocation": False,
        "independent_process": False,
        "provider_mutations_performed": 0,
    }
    watchdog_path = attempt_root / "vast_independent_watchdog_handoff.json"
    write_json(watchdog_path, watchdog)
    cleanup = {
        "schema_version": "wam_provider_object_store_cleanup.v1",
        "status": "completed",
        "blockers": [],
        "cleanup_attempts": 1,
        "exact_object_count": 2,
        "staging_manifest_sha256": "c" * 64,
        "raw_secret_values_recorded": False,
        "objects": [
            {
                "key_sha256": "d" * 64,
                "absence": {
                    "absence_confirmed": True,
                    "http_status_code": 404,
                    "status": "passed",
                    "raw_secret_values_recorded": False,
                },
            },
            {
                "key_sha256": "e" * 64,
                "absence": {
                    "absence_confirmed": True,
                    "http_status_code": 404,
                    "status": "passed",
                    "raw_secret_values_recorded": False,
                },
            },
        ],
        "all_objects_absent": True,
        "signed_url_files_removed": True,
    }
    cleanup_path = (
        attempt_root
        / "object_store_staging"
        / "wam_provider_object_store_cleanup.json"
    )
    write_json(cleanup_path, cleanup)
    result = {
        "schema_version": "native_task_arena_vast_run.v1",
        "generated_at": observed_at.isoformat(),
        "status": "blocked",
        "blockers": [
            "native_task_arena_policy_diagnostic_independent_watchdog_not_armed"
        ],
        "provider_mutations_performed": 0,
        "all_staged_objects_absent": True,
        "attempt_root": str(attempt_root),
        "retry_cap": 0,
        "authorization_consumption": {
            "status": "consumed",
            "authorization_digest": authority["authorization_digest"],
        },
        "independent_watchdog": watchdog,
    }
    result_path = attempt_root / "adp_arena_vast_result.json"
    write_json(result_path, result)
    api_zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "observed_at_utc": observed_at.isoformat(),
        "api_confirmed": True,
        "global_live_resource_count": 0,
        "provider_zero": True,
        "inventory": [],
        "api_command": ["vastai", "show", "instances", "--raw"],
        "raw_secret_values_recorded": False,
        "stderr_present": False,
        "provider_zero_digest": "",
    }
    api_zero["provider_zero_digest"] = canonical_digest(
        api_zero, digest_field="provider_zero_digest"
    )
    api_zero_path = root / "api_zero.json"
    write_json(api_zero_path, api_zero)
    return {
        "authority": authority_path,
        "result": result_path,
        "watchdog": watchdog_path,
        "cleanup": cleanup_path,
        "api_zero": api_zero_path,
    }


def _pre_spend_blocked_fixture(root: Path) -> dict[str, Path]:
    root.mkdir()
    attempt_root = (
        root
        / "allocator"
        / "arena-policy-diagnostic-job"
        / "attempts"
        / "attempt_001"
    )
    provider_run = attempt_root / "vast_provider_run"
    provider_run.mkdir(parents=True)
    observed_at = datetime.now(timezone.utc)
    authority = {
        "schema_version": paid.AUTHORITY_SCHEMA_VERSION,
        "execution_mode": "policy_diagnostic",
        "bundle_sha256": "sha256:" + "b" * 64,
        "blueprint_commit": COMMIT,
        "hard_attempt_spend_cap_usd": 0.5,
        "maximum_single_resource_ttl_seconds": 2800,
        "aggregate_goal_spend_before_attempt_usd": 39.540914,
        "aggregate_goal_spend_cap_usd": 50.0,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = root / "authority.json"
    write_json(authority_path, authority)
    preflight = {
        "schema_version": "pre_spend_preflight.v1",
        "generated_at": observed_at.isoformat(),
        "status": "FAIL",
        "spend_allowed": False,
        "blockers": ["spend_gate_closed:explicit_spend_approval_missing"],
    }
    preflight_path = provider_run / "pre_spend_preflight.json"
    write_json(preflight_path, preflight)
    result = {
        "schema_version": "native_task_arena_vast_run.v1",
        "generated_at": observed_at.isoformat(),
        "status": "blocked",
        "attempt_root": str(attempt_root),
        "provider_mutations_performed": 0,
        "pre_spend_preflight": preflight,
        "blockers": [
            "native_task_arena_policy_diagnostic_pre_spend_preflight_not_passed",
            "spend_gate_closed:explicit_spend_approval_missing",
        ],
    }
    result_path = attempt_root / "adp_arena_vast_result.json"
    write_json(result_path, result)
    consumption = {
        "schema_version": paid.CONSUMPTION_SCHEMA_VERSION,
        "authorization_digest": authority["authorization_digest"],
        "bundle_sha256": authority["bundle_sha256"],
        "blueprint_commit": authority["blueprint_commit"],
        "consumed_at": observed_at.isoformat(),
        "maximum_provider_allocations": 1,
    }
    consumption_path = (
        root
        / f"native-task-arena-{authority['authorization_digest'][7:]}.json"
    )
    write_json(consumption_path, consumption)
    api_zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "observed_at_utc": observed_at.isoformat(),
        "api_confirmed": True,
        "global_live_resource_count": 0,
        "provider_zero": True,
        "inventory": [],
        "api_command": ["vastai", "show", "instances", "--raw"],
        "raw_secret_values_recorded": False,
        "stderr_present": False,
        "provider_zero_digest": "",
    }
    api_zero["provider_zero_digest"] = canonical_digest(
        api_zero, digest_field="provider_zero_digest"
    )
    api_zero_path = root / "api_zero.json"
    write_json(api_zero_path, api_zero)
    return {
        "authority": authority_path,
        "result": result_path,
        "consumption": consumption_path,
        "api_zero": api_zero_path,
    }


def test_pre_spend_block_closes_without_claiming_policy_execution(
    tmp_path: Path,
) -> None:
    fixture = _pre_spend_blocked_fixture(tmp_path / "attempt")
    value = paid.materialize_native_task_arena_pre_spend_closeout(
        authority_path=fixture["authority"],
        allocator_result_path=fixture["result"],
        authority_consumption_path=fixture["consumption"],
        api_provider_zero_path=fixture["api_zero"],
        output_dir=tmp_path / "closeout",
    )

    result_path = Path(value["terminal_result_path"])
    result = json.loads(result_path.read_text())
    assert result["status"] == "sealed_blocked_attempt"
    assert result["closeout_kind"] == paid.PRE_SPEND_CLOSEOUT_KIND
    assert result["candidate_policy_queried"] is False
    assert result["first_observation_reached"] is False
    assert result["visual_evidence"] == {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "spend_gate_closed:explicit_spend_approval_missing",
        },
    }
    chain = paid.validate_terminal_spend_chain(
        authority_path=fixture["authority"],
        result_path=result_path,
        provider_zero_path=value["provider_zero_path"],
    )
    assert chain["attempt_cost_usd"] == 0.0
    assert chain["aggregate_goal_spend_after_attempt_usd"] == 39.540914


def test_watchdog_not_armed_preallocation_failure_closes_without_claiming_execution(
    tmp_path: Path,
) -> None:
    fixture = _watchdog_not_armed_fixture(tmp_path / "attempt")
    output = tmp_path / "closeout"
    value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=fixture["authority"],
        allocator_result_path=fixture["result"],
        watchdog_handoff_path=fixture["watchdog"],
        object_store_cleanup_path=fixture["cleanup"],
        api_provider_zero_path=fixture["api_zero"],
        output_dir=output,
    )

    closed_result = Path(value["terminal_result_path"])
    provider_zero = Path(value["provider_zero_path"])
    result = json.loads(closed_result.read_text())
    assert result["status"] == "sealed_blocked_attempt"
    assert result["estimated_cost_usd"] == 0.0
    assert result["scientific_attempt_started"] is False
    assert result["candidate_policy_queried"] is False
    assert result["visual_evidence"] == {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "native_task_arena_policy_diagnostic_independent_watchdog_not_armed",
        },
    }
    chain = paid.validate_terminal_spend_chain(
        authority_path=fixture["authority"],
        result_path=closed_result,
        provider_zero_path=provider_zero,
    )
    assert chain["attempt_cost_usd"] == 0.0
    assert chain["aggregate_goal_spend_after_attempt_usd"] == 39.540914


def test_preallocation_closeout_accepts_hardened_vast_api_zero_receipt(
    tmp_path: Path,
) -> None:
    fixture = _watchdog_not_armed_fixture(tmp_path / "attempt")
    api_zero = json.loads(fixture["api_zero"].read_text())
    api_zero["api_command"] = [
        "blueprint_pipeline.gpu_render_providers.VastRenderProvider.billable_inventory",
        "name_prefix=",
    ]
    api_zero["provider_zero_digest"] = canonical_digest(
        api_zero, digest_field="provider_zero_digest"
    )
    write_json(fixture["api_zero"], api_zero)

    value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=fixture["authority"],
        allocator_result_path=fixture["result"],
        watchdog_handoff_path=fixture["watchdog"],
        object_store_cleanup_path=fixture["cleanup"],
        api_provider_zero_path=fixture["api_zero"],
        output_dir=tmp_path / "closeout",
    )

    assert Path(value["terminal_result_path"]).is_file()


def test_preallocation_closeout_rejects_missing_typed_media_gap(tmp_path: Path) -> None:
    fixture = _watchdog_not_armed_fixture(tmp_path / "attempt")
    value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=fixture["authority"],
        allocator_result_path=fixture["result"],
        watchdog_handoff_path=fixture["watchdog"],
        object_store_cleanup_path=fixture["cleanup"],
        api_provider_zero_path=fixture["api_zero"],
        output_dir=tmp_path / "closeout",
    )
    result_path = Path(value["terminal_result_path"])
    result = json.loads(result_path.read_text())
    result.pop("visual_evidence")
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    write_json(result_path, result)
    zero_path = Path(value["provider_zero_path"])
    zero = json.loads(zero_path.read_text())
    zero["terminal_result"] = _record(result_path)
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    write_json(zero_path, zero)

    with pytest.raises(ValueError, match="preallocation_closeout_invalid"):
        paid.validate_terminal_spend_chain(
            authority_path=fixture["authority"],
            result_path=result_path,
            provider_zero_path=zero_path,
        )


def test_preallocation_closeout_chain_validates_from_dispatcher_staged_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling = _watchdog_not_armed_fixture(tmp_path / "sibling")
    sibling_value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=sibling["authority"],
        allocator_result_path=sibling["result"],
        watchdog_handoff_path=sibling["watchdog"],
        object_store_cleanup_path=sibling["cleanup"],
        api_provider_zero_path=sibling["api_zero"],
        output_dir=tmp_path / "sibling-closeout",
    )
    primary = _watchdog_not_armed_fixture(tmp_path / "primary")
    authority = json.loads(primary["authority"].read_text())
    authority["authority_reference"] = "primary"
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(primary["authority"], authority)
    result = json.loads(primary["result"].read_text())
    result["authorization_consumption"]["authorization_digest"] = authority[
        "authorization_digest"
    ]
    write_json(primary["result"], result)
    primary_value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=primary["authority"],
        allocator_result_path=primary["result"],
        watchdog_handoff_path=primary["watchdog"],
        object_store_cleanup_path=primary["cleanup"],
        api_provider_zero_path=primary["api_zero"],
        sibling_preallocation_closeout_paths=[sibling_value["provider_zero_path"]],
        output_dir=tmp_path / "primary-closeout",
    )

    primary_result = Path(primary_value["terminal_result_path"])
    primary_zero = Path(primary_value["provider_zero_path"])
    sibling_result = Path(sibling_value["terminal_result_path"])
    sibling_zero = Path(sibling_value["provider_zero_path"])
    primary_teardown = Path(json.loads(primary_zero.read_text())["teardown"]["path"])
    sibling_teardown = Path(json.loads(sibling_zero.read_text())["teardown"]["path"])
    declared = list(
        dict.fromkeys(
            (
                primary["authority"],
                primary_result,
                primary_zero,
                primary["result"],
                primary["watchdog"],
                primary["cleanup"],
                primary["api_zero"],
                primary_teardown,
                sibling["authority"],
                sibling_result,
                sibling_zero,
                sibling["result"],
                sibling["watchdog"],
                sibling["cleanup"],
                sibling["api_zero"],
                sibling_teardown,
            )
        )
    )
    records = {path: _record(path) for path in declared}
    profile = {
        "profile_id": "preallocation-closeout-staging-test",
        "profile_digest": "sha256:" + "0" * 64,
        "immutable_inputs": [
            {"name": f"input-{index}", "path": str(path), "digest": _sha(path)}
            for index, path in enumerate(declared)
        ],
    }
    run_root = tmp_path / "staged-run"
    dispatcher._stage_profile_immutable_inputs(
        profile=profile,
        run_root=run_root,
        allocator_argv=[],
    )
    monkeypatch.setenv(
        STAGING_RECEIPT_ENV,
        str(run_root / "immutable_input_staging_receipt.json"),
    )
    staged_authority = paid._bound_record(records[primary["authority"]], "authority")[0]
    staged_result = paid._bound_record(records[primary_result], "result")[0]
    staged_zero = paid._bound_record(records[primary_zero], "zero")[0]

    for path in declared:
        path.unlink()

    chain = paid.validate_terminal_spend_chain(
        authority_path=staged_authority,
        result_path=staged_result,
        provider_zero_path=staged_zero,
    )
    assert chain["attempt_cost_usd"] == 0.0
    assert chain["aggregate_goal_spend_after_attempt_usd"] == 39.540914


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("estimated_cost_usd",), 0.01),
        (("continuing_spend_from_this_run",), True),
        (("retry_cap",), 1),
        (("authorization_consumption", "status"), "available"),
        (("instance_id",), 48600001),
    ],
)
def test_preallocation_closeout_refuses_contradictory_original_state(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    fixture = _watchdog_not_armed_fixture(tmp_path / "attempt")
    result = json.loads(fixture["result"].read_text())
    target = result
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    write_json(fixture["result"], result)
    with pytest.raises(ValueError, match="preallocation_evidence_invalid"):
        paid.materialize_native_task_arena_preallocation_closeout(
            authority_path=fixture["authority"],
            allocator_result_path=fixture["result"],
            watchdog_handoff_path=fixture["watchdog"],
            object_store_cleanup_path=fixture["cleanup"],
            api_provider_zero_path=fixture["api_zero"],
            output_dir=tmp_path / "closeout",
        )


def test_preallocation_closeout_requires_the_canonical_authenticated_api_contract(
    tmp_path: Path,
) -> None:
    fixture = _watchdog_not_armed_fixture(tmp_path / "attempt")
    zero = json.loads(fixture["api_zero"].read_text())
    zero["api_command"] = ["echo", "[]"]
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    write_json(fixture["api_zero"], zero)
    with pytest.raises(ValueError, match="preallocation_api_zero_invalid"):
        paid.materialize_native_task_arena_preallocation_closeout(
            authority_path=fixture["authority"],
            allocator_result_path=fixture["result"],
            watchdog_handoff_path=fixture["watchdog"],
            object_store_cleanup_path=fixture["cleanup"],
            api_provider_zero_path=fixture["api_zero"],
            output_dir=tmp_path / "closeout",
        )


def test_preallocation_closeout_rejects_broad_attempt_root_and_empty_cleanup_claim(
    tmp_path: Path,
) -> None:
    broad = _watchdog_not_armed_fixture(tmp_path / "broad")
    result = json.loads(broad["result"].read_text())
    result["attempt_root"] = "/"
    write_json(broad["result"], result)
    with pytest.raises(ValueError, match="preallocation_evidence_invalid"):
        paid.materialize_native_task_arena_preallocation_closeout(
            authority_path=broad["authority"],
            allocator_result_path=broad["result"],
            watchdog_handoff_path=broad["watchdog"],
            object_store_cleanup_path=broad["cleanup"],
            api_provider_zero_path=broad["api_zero"],
            output_dir=tmp_path / "broad-closeout",
        )

    empty = _watchdog_not_armed_fixture(tmp_path / "empty")
    cleanup = json.loads(empty["cleanup"].read_text())
    cleanup["objects"] = []
    cleanup["exact_object_count"] = 0
    write_json(empty["cleanup"], cleanup)
    with pytest.raises(ValueError, match="preallocation_evidence_invalid"):
        paid.materialize_native_task_arena_preallocation_closeout(
            authority_path=empty["authority"],
            allocator_result_path=empty["result"],
            watchdog_handoff_path=empty["watchdog"],
            object_store_cleanup_path=empty["cleanup"],
            api_provider_zero_path=empty["api_zero"],
            output_dir=tmp_path / "empty-closeout",
        )


def test_preallocation_closeout_binds_one_zero_cost_sibling_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    sibling = _watchdog_not_armed_fixture(tmp_path / "sibling")
    sibling_value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=sibling["authority"],
        allocator_result_path=sibling["result"],
        watchdog_handoff_path=sibling["watchdog"],
        object_store_cleanup_path=sibling["cleanup"],
        api_provider_zero_path=sibling["api_zero"],
        output_dir=tmp_path / "sibling-closeout",
    )
    primary = _watchdog_not_armed_fixture(tmp_path / "primary")
    # The two concurrent authorities have distinct identities.
    authority = json.loads(primary["authority"].read_text())
    authority["authority_reference"] = "primary"
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(primary["authority"], authority)
    result = json.loads(primary["result"].read_text())
    result["authorization_consumption"]["authorization_digest"] = authority[
        "authorization_digest"
    ]
    write_json(primary["result"], result)
    value = paid.materialize_native_task_arena_preallocation_closeout(
        authority_path=primary["authority"],
        allocator_result_path=primary["result"],
        watchdog_handoff_path=primary["watchdog"],
        object_store_cleanup_path=primary["cleanup"],
        api_provider_zero_path=primary["api_zero"],
        sibling_preallocation_closeout_paths=[sibling_value["provider_zero_path"]],
        output_dir=tmp_path / "primary-closeout",
    )
    zero_path = Path(value["provider_zero_path"])
    zero = json.loads(zero_path.read_text())
    assert len(zero["sibling_preallocation_closeouts"]) == 1

    sibling_zero = Path(sibling_value["provider_zero_path"])
    sibling_payload = json.loads(sibling_zero.read_text())
    sibling_payload["scientific_attempt_started"] = True
    write_json(sibling_zero, sibling_payload)
    with pytest.raises(ValueError, match="sibling_unbound"):
        paid.validate_terminal_spend_chain(
            authority_path=primary["authority"],
            result_path=value["terminal_result_path"],
            provider_zero_path=zero_path,
        )


def _no_allocation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A run that ended before allocating anything, as r16 actually did."""

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
            "schema_version": paid.WATCHDOG_HANDOFF_SCHEMA_VERSION,
            "status": "cancelled_no_allocation",
            "provider_mutations_performed": 0,
            "watchdog_armed_before_allocation": True,
        },
    )
    result = {
        "schema_version": "native_task_arena_vast_run.v1",
        "status": "blocked",
        "authorization_consumption": {
            "authorization_digest": authority["authorization_digest"]
        },
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "watchdog_receipt_path": str(watchdog),
        "object_store_cleanup_path": str(cleanup),
        "adapter_result_path": str(adapter),
        "teardown_manifest_path": str(teardown),
        "estimated_cost_usd": 0.0,
    }
    result_path = tmp_path / "result.json"
    write_json(result_path, result)
    return authority_path, result_path, watchdog


def test_a_run_that_never_allocated_can_still_seal(tmp_path: Path) -> None:
    """Otherwise a pre-allocation failure wedges the whole chain.

    When no offer meets the lane's constraints the run ends before creating
    anything, so the watchdog records a handoff rather than a canary receipt
    and there is no inventory to prove empty. The successor's first step is
    sealing its predecessor, so an unsealable predecessor blocks every later
    attempt permanently. Arena r16 ended exactly this way at $0.00.
    """

    authority_path, result_path, _ = _no_allocation_fixture(tmp_path)
    receipt = paid.materialize_native_task_arena_provider_zero(
        authority_path=authority_path,
        result_path=result_path,
        output_path=tmp_path / "provider_zero.json",
    )
    assert receipt["provider_zero_confirmed"] is True
    assert receipt["inventory_scope"] == "no_provider_allocation"


def test_legacy_preflight_exit_with_missing_cost_can_seal_and_chain(
    tmp_path: Path,
) -> None:
    """The provider's own no-create evidence repairs the old absent scalar."""

    authority_path, result_path, _ = _no_allocation_fixture(tmp_path)
    authority = json.loads(authority_path.read_text())
    authority.update(
        {
            "bundle_sha256": "sha256:" + "b" * 64,
            "hard_attempt_spend_cap_usd": 2.0,
            "maximum_single_resource_ttl_seconds": 3600,
            "aggregate_goal_spend_before_attempt_usd": 1.0,
            "aggregate_goal_spend_cap_usd": 50.0,
        }
    )
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    write_json(authority_path, authority)

    result = json.loads(result_path.read_text())
    result.update(
        {
            "bundle_sha256": authority["bundle_sha256"],
            "hard_cap_usd": 2.0,
            "hard_ttl_seconds": 3600,
            "retry_cap": 0,
            "estimated_cost_usd": None,
            "authorization_consumption": {
                "authorization_digest": authority["authorization_digest"]
            },
        }
    )
    adapter_path = Path(result["adapter_result_path"])
    write_json(
        adapter_path,
        {
            "api_call_performed": False,
            "provider_create_attempted": False,
            "vast_instance_ids": [],
        },
    )
    teardown_path = Path(result["teardown_manifest_path"])
    write_json(
        teardown_path,
        {"continuing_spend_from_this_run": False, "vast_instance_ids": []},
    )
    write_json(result_path, result)

    zero_path = tmp_path / "legacy_provider_zero.json"
    receipt = paid.materialize_native_task_arena_provider_zero(
        authority_path=authority_path,
        result_path=result_path,
        output_path=zero_path,
    )
    assert receipt["inventory_scope"] == "no_provider_allocation"
    chain = paid.validate_terminal_spend_chain(
        authority_path=authority_path,
        result_path=result_path,
        provider_zero_path=zero_path,
    )
    assert chain["attempt_cost_usd"] == 0.0


def test_the_no_allocation_seal_demands_proof_that_nothing_was_allocated(
    tmp_path: Path,
) -> None:
    """An orphan needs an allocation to exist, so each proof must be required.

    Every one of these mutations describes a run that DID touch the provider,
    and each must fail closed rather than ride the no-allocation path.
    """

    for field, value in (
        ("provider_mutations_performed", 1),
        ("watchdog_armed_before_allocation", False),
        ("status", "provider_running"),
    ):
        authority_path, result_path, watchdog = _no_allocation_fixture(tmp_path)
        payload = json.loads(watchdog.read_text())
        payload[field] = value
        write_json(watchdog, payload)
        with pytest.raises(
            ValueError, match="native_task_arena_provider_zero_invalid"
        ):
            paid.materialize_native_task_arena_provider_zero(
                authority_path=authority_path,
                result_path=result_path,
                output_path=tmp_path / f"invalid_{field}.json",
            )

    # A non-zero, missing, or unparseable cost is not evidence of zero spend.
    for cost in (0.17, None, "free"):
        authority_path, result_path, _ = _no_allocation_fixture(tmp_path)
        payload = json.loads(result_path.read_text())
        payload["estimated_cost_usd"] = cost
        write_json(result_path, payload)
        with pytest.raises(
            ValueError, match="native_task_arena_provider_zero_invalid"
        ):
            paid.materialize_native_task_arena_provider_zero(
                authority_path=authority_path,
                result_path=result_path,
                output_path=tmp_path / f"invalid_cost_{cost}.json",
            )

    # Continuing spend still fails closed even with nothing allocated.
    authority_path, result_path, _ = _no_allocation_fixture(tmp_path)
    payload = json.loads(result_path.read_text())
    payload["continuing_spend_from_this_run"] = True
    write_json(result_path, payload)
    with pytest.raises(ValueError, match="native_task_arena_provider_zero_invalid"):
        paid.materialize_native_task_arena_provider_zero(
            authority_path=authority_path,
            result_path=result_path,
            output_path=tmp_path / "invalid_continuing.json",
        )
