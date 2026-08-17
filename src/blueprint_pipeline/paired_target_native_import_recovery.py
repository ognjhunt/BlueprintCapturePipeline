"""Seal post-attempt paired-target provider-zero from recovered terminal evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .vast_official_billing_extractor import extract_vast_official_instance_charge


SCHEMA_VERSION = "paired_target_native_import_provider_zero.v1"
AUTHORITY_SCHEMA = "paired_target_native_import_paid_attempt_authority.v1"
RESULT_SCHEMA = "paired_target_native_import_vast_run.v1"
MAX_GUARD_AGE_SECONDS = 900


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not path.is_file() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    candidate = Path(str(value.get("path") or "")).expanduser()
    absolute = candidate.absolute()
    path = candidate.resolve()
    if (
        candidate.is_symlink()
        or path != absolute
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def _time(value: Any, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(code) from exc
    if parsed.tzinfo is None:
        raise ValueError(code)
    return parsed.astimezone(timezone.utc)


def _global_guard_zero(value: Mapping[str, Any]) -> bool:
    rows = value.get("inventory_results")
    required = {
        row.get("provider"): row
        for row in rows or []
        if isinstance(row, Mapping) and row.get("required") is True
    }
    return (
        value.get("schema_version") == "gpu_spend_guard.v1"
        and value.get("status") == "passed"
        and value.get("provider_zero_verified") is True
        and value.get("live_instance_count") == 0
        and value.get("total_burn_per_hour_usd") == 0
        and set(required) == {"runpod", "vast", "digitalocean"}
        and all(
            row.get("status") == "succeeded" and row.get("row_count") == 0
            for row in required.values()
        )
    )


def _zero_inventory(value: Any, *, prefix: str | None) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("status") == "observed"
        and value.get("provider") == "vast"
        and value.get("api_confirmed") is True
        and value.get("live_resource_count") == 0
        and value.get("resources") == []
        and (prefix is None or value.get("name_prefix") == prefix)
    )


def _launch_identity(result_path: Path) -> dict[str, Any]:
    if (
        result_path.name != "paired_target_native_import_vast_result.v1.json"
        or result_path.parent.name != "paired-target-native-import-job"
        or result_path.parent.parent.name != "allocator"
    ):
        raise ValueError("paired_target_recovery_result_layout_invalid")
    run_root = result_path.parents[2]
    names = {
        "launch_request": ("launch_request.json", "task_evaluation_launch_request.v1", "request_digest"),
        "launch_profile": ("launch_profile.json", "task_evaluation_launch_profile.v1", "profile_digest"),
        "launch_binding": ("launch_binding.json", "task_evaluation_launch_binding.v1", "binding_digest"),
        "launch_started": ("launch_started.json", "task_evaluation_launch_started.v1", "started_digest"),
        "launch_receipt": ("launch_receipt.json", "task_evaluation_launch_receipt.v1", "receipt_digest"),
    }
    values: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for role, (name, schema, digest_field) in names.items():
        path = run_root / name
        value = _read(path, "paired_target_recovery_launch_identity_invalid")
        if (
            value.get("schema_version") != schema
            or value.get(digest_field)
            != canonical_digest(value, digest_field=digest_field)
        ):
            raise ValueError("paired_target_recovery_launch_identity_invalid")
        values[role] = value
        records[role] = {**_record(path), digest_field: value[digest_field]}
    request = values["launch_request"]
    profile = values["launch_profile"]
    binding = values["launch_binding"]
    started = values["launch_started"]
    receipt = values["launch_receipt"]
    launch_id = request.get("launch_id")
    run_id = request.get("run_id")
    request_digest = request.get("request_digest")
    profile_id = request.get("launch_profile_id")
    profile_digest = request.get("launch_profile_digest")
    if (
        not isinstance(launch_id, str)
        or not launch_id
        or run_id != launch_id
        or run_root.name != launch_id
        or profile.get("profile_id") != profile_id
        or profile.get("profile_digest") != profile_digest
        or binding.get("launch_id") != launch_id
        or binding.get("run_id") != run_id
        or binding.get("request_digest") != request_digest
        or binding.get("profile_digest") != profile_digest
        or started.get("launch_id") != launch_id
        or started.get("run_id") != run_id
        or started.get("request_digest") != request_digest
        or started.get("binding_digest") != binding.get("binding_digest")
        or started.get("automatic_retry_authorized") is not False
        or receipt.get("launch_id") != launch_id
        or receipt.get("run_id") != run_id
        or receipt.get("request_digest") != request_digest
        or receipt.get("launch_profile_digest") != profile_digest
        or receipt.get("binding_digest") != binding.get("binding_digest")
    ):
        raise ValueError("paired_target_recovery_launch_identity_invalid")
    return {
        "launch_id": launch_id,
        "run_id": run_id,
        "request_digest": request_digest,
        "profile_id": profile_id,
        "profile_digest": profile_digest,
        "records": records,
    }


def _validated_payload(
    records: Mapping[str, Any],
    *,
    launch_label: str,
    check_fresh: bool,
    now: datetime,
) -> dict[str, Any]:
    required = {
        "attempt_authority",
        "terminal_result",
        "original_teardown",
        "original_watchdog_handoff",
        "original_armed_watchdog",
        "owner_cancel_receipt",
        "recovered_watchdog",
        "object_store_cleanup",
        "provider_adapter",
        "session_budget",
        "failure_machine_avoidlist",
        "fresh_global_guard",
        "webapp_sync",
        "provider_billing_source_receipt",
    }
    if set(records) != required:
        raise ValueError("paired_target_recovered_provider_zero_invalid")
    paths = {
        role: _bound(record, "paired_target_recovered_provider_zero_unbound")[0]
        for role, record in records.items()
    }
    values = {
        role: _read(path, "paired_target_recovered_provider_zero_unreadable")
        for role, path in paths.items()
    }
    authority = values["attempt_authority"]
    bundle_receipt_path, _ = _bound(
        authority.get("bundle_receipt"),
        "paired_target_recovery_bundle_receipt_unbound",
    )
    bundle_receipt = _read(
        bundle_receipt_path, "paired_target_recovery_bundle_receipt_unreadable"
    )
    result = values["terminal_result"]
    original_teardown = values["original_teardown"]
    handoff = values["original_watchdog_handoff"]
    armed = values["original_armed_watchdog"]
    owner_cancel = values["owner_cancel_receipt"]
    watchdog = values["recovered_watchdog"]
    cleanup = values["object_store_cleanup"]
    adapter = values["provider_adapter"]
    session = values["session_budget"]
    machine_avoidlist = values["failure_machine_avoidlist"]
    guard = values["fresh_global_guard"]
    webapp = values["webapp_sync"]
    instance_ids = adapter.get("vast_instance_ids")
    instance_id = instance_ids[0] if isinstance(instance_ids, list) and len(instance_ids) == 1 else None
    recorded = watchdog.get("recorded_vast_instance")
    teardown = watchdog.get("recorded_vast_instance_teardown")
    attempts = teardown.get("inspect_attempts") if isinstance(teardown, Mapping) else None
    prefix = recorded.get("pod_name_prefix") if isinstance(recorded, Mapping) else None
    classification = adapter.get("provider_attempt_classification")
    session_attempts = session.get("attempts")
    session_attempt = (
        session_attempts[0]
        if isinstance(session_attempts, list) and len(session_attempts) == 1
        else None
    )
    machine_id = session_attempt.get("machine_id") if isinstance(session_attempt, Mapping) else None
    offer_id = session_attempt.get("offer_id") if isinstance(session_attempt, Mapping) else None
    avoid_entries = machine_avoidlist.get("entries")
    avoid_entry = (
        avoid_entries[0]
        if isinstance(avoid_entries, list) and len(avoid_entries) == 1
        else None
    )
    estimate = result.get("estimated_cost_usd")
    launch = _launch_identity(paths["terminal_result"])
    official_charge = extract_vast_official_instance_charge(
        provider_billing_source_receipt_path=paths[
            "provider_billing_source_receipt"
        ],
        instance_id=int(instance_id or 0),
        launch_label=launch_label,
    )
    completed_at = _time(watchdog.get("completed_at"), "paired_target_recovery_watchdog_time_invalid")
    guard_at = _time(guard.get("generated_at"), "paired_target_recovery_guard_time_invalid")
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or authority.get("maximum_paid_attempts") != 1
        or authority.get("maximum_provider_allocations") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("automatic_paid_retry_authorized") is not False
        or result.get("bundle_sha256") != authority.get("bundle_sha256")
        or bundle_receipt.get("receipt_digest")
        != authority.get("bundle_receipt_digest")
        or bundle_receipt.get("bundle_sha256") != authority.get("bundle_sha256")
        or result.get("request_digest") != bundle_receipt.get("request_digest")
        or result.get("hard_cap_usd") != authority.get("hard_attempt_spend_cap_usd")
        or result.get("hard_ttl_seconds")
        != authority.get("maximum_single_resource_ttl_seconds")
        or result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "blocked"
        or result.get("provider_mutations_performed") != 1
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not True
        or result.get("all_staged_objects_absent") is not True
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("adapter_result_path") != str(paths["provider_adapter"])
        or result.get("watchdog_receipt_path") != str(paths["recovered_watchdog"])
        or result.get("object_store_cleanup_path") != str(paths["object_store_cleanup"])
        or result.get("teardown_manifest_path") != str(paths["original_teardown"])
        or isinstance(estimate, bool)
        or not isinstance(estimate, (int, float))
        or not math.isfinite(float(estimate))
        or float(estimate) < 0
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or cleanup.get("blockers") != []
        or original_teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or original_teardown.get("status") != "blocked"
        or original_teardown.get("vast_instance_ids") != [instance_id]
        or original_teardown.get("continuing_spend_from_this_run") is not True
        or original_teardown.get("teardown_actions_performed")
        != [
            {
                "instance_id": instance_id,
                "action": "destroy_instance",
                "status": "failed",
                "http_status_code": 429,
                "error": "HTTPError: HTTP Error 429: Too Many Requests",
            }
        ]
        or adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("status") != "failed"
        or adapter.get("provider_create_attempted") is not True
        or adapter.get("continuing_spend_from_this_run") is not True
        or adapter.get("estimated_cost_usd") != estimate
        or not isinstance(instance_id, int)
        or isinstance(instance_id, bool)
        or instance_id <= 0
        or adapter.get("raw_api_key_stored") is not False
        or adapter.get("secret_values_in_artifact") is not False
        or adapter.get("provider_bundle_kind") != "paired_target_native_import"
        or not isinstance(classification, Mapping)
        or classification.get("classification") != "pre_execution_provider_null"
        or classification.get("provider_bundle_started") is not False
        or classification.get("provider_entrypoint_started") is not False
        or classification.get("provider_output_returned") is not False
        or classification.get("scientific_attempt_consumed") is not False
        or classification.get("automatic_requeue_executed") is not False
        or classification.get("maximum_automatic_requeues") != 0
        or session.get("schema_version") != "vast_session_cost_summary.v4"
        or session.get("status") != "completed"
        or session.get("attempt_count") != 1
        or not isinstance(session_attempt, Mapping)
        or session_attempt.get("vast_instance_ids") != [instance_id]
        or session_attempt.get("estimated_cost_usd") != estimate
        or session_attempt.get("continuing_spend_from_this_run") is not True
        or session_attempt.get("blockers") != ["vast_heartbeat_container_missing"]
        or isinstance(machine_id, bool)
        or not isinstance(machine_id, int)
        or machine_id <= 0
        or isinstance(offer_id, bool)
        or not isinstance(offer_id, int)
        or offer_id <= 0
        or machine_avoidlist.get("schema_version") != "vast_machine_avoidlist.v1"
        or machine_avoidlist.get("status") != "completed"
        or machine_avoidlist.get("machine_ids") != [machine_id]
        or machine_avoidlist.get("raw_secret_values_recorded") is not False
        or not isinstance(avoid_entry, Mapping)
        or avoid_entry.get("machine_id") != machine_id
        or avoid_entry.get("offer_id") != offer_id
        or avoid_entry.get("instance_id") != instance_id
        or avoid_entry.get("reason")
        != "vast_startup_control_plane_did_not_reach_onstart_heartbeat"
        or avoid_entry.get("blockers") != ["vast_heartbeat_container_missing"]
        or handoff.get("schema_version") != "vast_independent_watchdog_handoff.v1"
        or handoff.get("status") != "retained_until_hard_ttl"
        or handoff.get("watchdog_armed_before_allocation") is not True
        or handoff.get("instance_ids") != [instance_id]
        or handoff.get("provider_mutations_performed") != 0
        or handoff.get("raw_secret_values_recorded") is not False
        or armed.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or armed.get("status") != "armed"
        or armed.get("provider") != "vast"
        or armed.get("provider_mutations_performed") != 0
        or armed.get("raw_secret_values_recorded") is not False
        or armed.get("allowed_active_instance_ids") != []
        or isinstance(armed.get("pid"), bool)
        or not isinstance(armed.get("pid"), int)
        or armed.get("pid") <= 0
        or isinstance(watchdog.get("pid"), bool)
        or not isinstance(watchdog.get("pid"), int)
        or watchdog.get("pid") <= 0
        or armed.get("pid") == watchdog.get("pid")
        or watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("provider_absence_scope") != "recorded_instance_and_lane_prefix"
        or watchdog.get("provider_mutations_performed") != 0
        or watchdog.get("raw_secret_values_recorded") is not False
        or not isinstance(recorded, Mapping)
        or recorded.get("status") != "recorded"
        or recorded.get("instance_id") != str(instance_id)
        or recorded.get("scope_confirmed") is not True
        or not isinstance(prefix, str)
        or not prefix.startswith("blueprint-adp-paired-native-import-")
        or armed.get("pod_name_prefix") != prefix
        or armed.get("deadline_epoch") != watchdog.get("deadline_epoch")
        or owner_cancel.get("schema_version")
        != "groot_oscar_runpod_canary_watchdog_cancel.v1"
        or owner_cancel.get("provider") != "vast"
        or owner_cancel.get("instance_id") != str(instance_id)
        or owner_cancel.get("pod_name_prefix") != prefix
        or owner_cancel.get("provider_absence_confirmed") is not True
        or owner_cancel.get("raw_secret_values_recorded") is not False
        or watchdog.get("owner_teardown_cancel_requested") is not True
        or watchdog.get("owner_teardown_cancel_request_valid") is not True
        or not isinstance(teardown, Mapping)
        or teardown.get("status") != "absent"
        or teardown.get("instance_id") != str(instance_id)
        or teardown.get("provider_absence_confirmed") is not True
        or teardown.get("provider_mutations_performed") != 0
        or not isinstance(attempts, list)
        or len(attempts) < 2
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "absent"
            or row.get("provider") != "vast"
            or row.get("instance_id") != str(instance_id)
            or row.get("api_confirmed") is not True
            or row.get("provider_absence_confirmed") is not True
            for row in attempts
        )
        or not _zero_inventory(watchdog.get("final_inventory"), prefix=prefix)
        or not _zero_inventory(watchdog.get("final_global_inventory"), prefix="")
        or not _global_guard_zero(guard)
        or guard_at < completed_at
        or (check_fresh and (guard_at > now or (now - guard_at).total_seconds() > MAX_GUARD_AGE_SECONDS))
        or webapp.get("schema_version")
        != "task_evaluation_launch_webapp_sync_result.v1"
        or webapp.get("status") != "succeeded"
        or webapp.get("launch_id") != launch["launch_id"]
        or webapp.get("run_id") != launch["run_id"]
        or webapp.get("request_digest") != launch["request_digest"]
        or webapp.get("receipt_digest")
        != launch["records"]["launch_receipt"]["receipt_digest"]
        or webapp.get("sync_result_digest")
        != canonical_digest(webapp, digest_field="sync_result_digest")
        or webapp.get("provider_mutation_performed") is not False
        or official_charge["provider_billing_source_receipt"]["path"]
        != str(paths["provider_billing_source_receipt"])
    ):
        raise ValueError("paired_target_recovered_provider_zero_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_recovered_provider_zero",
        "closure_kind": "post_attempt_recovery_without_original_rewrite",
        "attempt_authority_digest": authority["authorization_digest"],
        "provider_instance_id": instance_id,
        "provider_launch_label": launch_label,
        "attempt_cost_estimate_usd": round(float(estimate), 6),
        "official_cost_usd": official_charge["official_charge_usd"],
        "official_charge": official_charge,
        "failed_machine_id": machine_id,
        "failed_offer_id": offer_id,
        "recommended_excluded_machine_ids": [machine_id],
        "original_watchdog_pid": armed["pid"],
        "recovery_watchdog_pid": watchdog["pid"],
        "watchdog_process_recovered": True,
        "original_result_reported_continuing_spend": True,
        "original_adapter_reported_continuing_spend": True,
        "original_teardown_reported_continuing_spend": True,
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "lane_provider_zero_confirmed": True,
        "global_provider_zero_confirmed": True,
        "provider_absence_observation_count": len(attempts),
        "allocation_count": 1,
        "retry_cap": 0,
        "science_execution_started": False,
        "launch_identity": launch,
        "records": {role: _record(path) for role, path in paths.items()},
        "authority_bundle_receipt": _record(bundle_receipt_path),
    }


def materialize_paired_target_native_import_recovered_provider_zero(
    *,
    attempt_authority_path: str | Path,
    result_path: str | Path,
    recovered_watchdog_path: str | Path,
    original_watchdog_handoff_path: str | Path,
    original_armed_watchdog_path: str | Path,
    owner_cancel_receipt_path: str | Path,
    original_teardown_path: str | Path,
    cleanup_path: str | Path,
    adapter_path: str | Path,
    session_budget_path: str | Path,
    failure_machine_avoidlist_path: str | Path,
    fresh_global_guard_path: str | Path,
    webapp_sync_path: str | Path,
    provider_billing_source_receipt_path: str | Path,
    launch_label: str,
    output_path: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    paths = {
        "attempt_authority": Path(attempt_authority_path).expanduser().resolve(),
        "terminal_result": Path(result_path).expanduser().resolve(),
        "original_teardown": Path(original_teardown_path).expanduser().resolve(),
        "original_watchdog_handoff": Path(
            original_watchdog_handoff_path
        ).expanduser().resolve(),
        "original_armed_watchdog": Path(
            original_armed_watchdog_path
        ).expanduser().resolve(),
        "owner_cancel_receipt": Path(owner_cancel_receipt_path).expanduser().resolve(),
        "recovered_watchdog": Path(recovered_watchdog_path).expanduser().resolve(),
        "object_store_cleanup": Path(cleanup_path).expanduser().resolve(),
        "provider_adapter": Path(adapter_path).expanduser().resolve(),
        "session_budget": Path(session_budget_path).expanduser().resolve(),
        "failure_machine_avoidlist": Path(
            failure_machine_avoidlist_path
        ).expanduser().resolve(),
        "fresh_global_guard": Path(fresh_global_guard_path).expanduser().resolve(),
        "webapp_sync": Path(webapp_sync_path).expanduser().resolve(),
        "provider_billing_source_receipt": Path(
            provider_billing_source_receipt_path
        ).expanduser().resolve(),
    }
    records = {role: _record(path) for role, path in paths.items()}
    value: dict[str, Any] = {
        **_validated_payload(
            records,
            launch_label=launch_label,
            check_fresh=True,
            now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc),
        ),
        "generated_at": utc_now_iso(),
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_recovered_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, value)
    validate_paired_target_native_import_recovered_provider_zero(output)
    return value


def validate_paired_target_native_import_recovered_provider_zero(
    path: str | Path,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = _read(source, "paired_target_recovered_provider_zero_unreadable")
    records = value.get("records")
    if (
        not isinstance(records, Mapping)
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("provider_mutation_performed") is not False
        or value.get("raw_secret_values_recorded") is not False
    ):
        raise ValueError("paired_target_recovered_provider_zero_invalid")
    expected = {
        **_validated_payload(
            records,
            launch_label=str(value.get("provider_launch_label") or ""),
            check_fresh=False,
            now=datetime.now(timezone.utc),
        ),
        "generated_at": value.get("generated_at"),
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": value["receipt_digest"],
    }
    if value != expected:
        raise ValueError("paired_target_recovered_provider_zero_invalid")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--recovered-watchdog", required=True)
    parser.add_argument("--original-watchdog-handoff", required=True)
    parser.add_argument("--original-armed-watchdog", required=True)
    parser.add_argument("--owner-cancel-receipt", required=True)
    parser.add_argument("--original-teardown", required=True)
    parser.add_argument("--cleanup", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--session-budget", required=True)
    parser.add_argument("--failure-machine-avoidlist", required=True)
    parser.add_argument("--fresh-global-guard", required=True)
    parser.add_argument("--webapp-sync", required=True)
    parser.add_argument("--provider-billing-source-receipt", required=True)
    parser.add_argument("--launch-label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = materialize_paired_target_native_import_recovered_provider_zero(
            attempt_authority_path=args.attempt_authority,
            result_path=args.result,
            recovered_watchdog_path=args.recovered_watchdog,
            original_watchdog_handoff_path=args.original_watchdog_handoff,
            original_armed_watchdog_path=args.original_armed_watchdog,
            owner_cancel_receipt_path=args.owner_cancel_receipt,
            original_teardown_path=args.original_teardown,
            cleanup_path=args.cleanup,
            adapter_path=args.adapter,
            session_budget_path=args.session_budget,
            failure_machine_avoidlist_path=args.failure_machine_avoidlist,
            fresh_global_guard_path=args.fresh_global_guard,
            webapp_sync_path=args.webapp_sync,
            provider_billing_source_receipt_path=(
                args.provider_billing_source_receipt
            ),
            launch_label=args.launch_label,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "output": str(Path(args.output).expanduser().resolve()),
                "provider_instance_id": value["provider_instance_id"],
                "receipt_digest": value["receipt_digest"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
