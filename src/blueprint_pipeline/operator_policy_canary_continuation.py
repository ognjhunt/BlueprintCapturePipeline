"""CP continuation of one already allocated policy canary; no allocation path.

The immutable intent and explicit owner handoff gate every collection/cleanup
step. A separate canonical watchdog retains the original absolute deadline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import get_render_provider
from . import groot_oscar_runpod_watchdog as watchdog
from .operator_policy_canary_handoff import (
    ContinuationError, build_continuation_intent as build_continuation_intent,
    cloud_readiness, commit_owner_handoff, host_identity, metadata_root,
    process_identity, read_json, seal_value, validate_continuation_intent,
    validate_owner_handoff, verified_record, validate_control_plane_host, record_watchdog_process,
)
from .operator_policy_canary_terminal_delivery import (
    OperatorTerminalDeliveryAdapters, file_record, finalize_operator_policy_canary, _seal,
)
from .provider_output_range_ingestion import ingest_provider_output
from .task_evaluation_artifact_manifest import build_task_evaluation_artifact_manifest
from .task_evaluation_policy_canary_dispatcher import collect_policy_canary_vast_provider_zero, _sealed_provider_zero
from .wam_provider_object_store import cleanup_staged_wam_provider_objects


class _ExactExistingProvider:
    """Expose only read/terminate for the recorded ID, even if labels collide."""
    name = "vast"

    def __init__(self, provider, instance_id):
        self._provider, self._instance_id = provider, str(instance_id)

    def _key(self):
        return self._provider._key()

    def inspect(self, instance_id):
        if str(instance_id) != self._instance_id:
            raise ContinuationError("continuation_unowned_resource_forbidden")
        return self._provider.inspect(instance_id)

    def terminate(self, instance_id):
        if str(instance_id) != self._instance_id:
            raise ContinuationError("continuation_unowned_resource_forbidden")
        return self._provider.terminate(instance_id)


def live_inventory(intent):
    return watchdog._billable_inventory(provider=get_render_provider("vast"), provider_name="vast",
        name_prefix=intent["resource_name"], resource_name_exact=intent["resource_name"])


def terminate_owned_instance(intent):
    armed = read_json(Path(intent["watchdog_root"]) / watchdog.EVIDENCE_NAME)
    if (armed.get("deadline_epoch") != intent["deadline_epoch"]
            or armed.get("pod_name_prefix") != intent["resource_name"]
            or Path(intent["watchdog_root"]).joinpath(watchdog.VAST_STARTED_INSTANCE_ID_NAME).read_text().strip() != str(intent["instance_id"])):
        raise ContinuationError("continuation_watchdog_identity_changed")
    return watchdog.terminate_canary_resources(
        provider=_ExactExistingProvider(get_render_provider("vast"), intent["instance_id"]),
        pod_name_prefix=intent["resource_name"], armed=armed, provider_name="vast",
        resource_name_exact=intent["resource_name"])


def cleanup_owned_objects(intent):
    staging = verified_record(intent["ingestion_binding"]["staging_manifest"])
    allowed = {"access_key_id_file", "secret_access_key_file", "endpoint_url_file", "bucket_file", "region_file"}
    config = intent["cleanup_configuration"]
    if not set(config) <= allowed:
        raise ContinuationError("continuation_cleanup_configuration_invalid")
    return cleanup_staged_wam_provider_objects(staging.parent, **config)


def refresh_official_billing(*, intent, adapter, continuation_intent):
    del adapter
    from .provider_billing_reconciler import reconcile_provider_billing
    return reconcile_provider_billing(secrets_dir=continuation_intent["billing_secrets_dir"],
        billing_export_path=metadata_root(continuation_intent) / "official_billing_export.json",
        audit_root=intent["billing_audit_root"], start_at=continuation_intent["billing_period_start_at"],
        required_providers=("vast",))


def canonical_delivery_adapters(intent):
    from .task_evaluation_run_webapp_sync import sync_task_evaluation_policy_canary_to_webapp
    from .task_evaluation_delivery_readback import verify_website_delivery
    return OperatorTerminalDeliveryAdapters(sync_runner=sync_task_evaluation_policy_canary_to_webapp,
        download_readback=verify_website_delivery,
        official_billing_refresher=lambda **kwargs: refresh_official_billing(**kwargs, continuation_intent=intent))


@dataclass(frozen=True)
class ExistingRunContinuationAdapters:
    collector: Callable[..., dict] = ingest_provider_output
    terminator: Callable[..., dict] = terminate_owned_instance
    cleanup: Callable[..., dict] = cleanup_owned_objects
    provider_zero_reader: Callable[..., dict] = collect_policy_canary_vast_provider_zero
    finalizer: Callable[..., dict] = finalize_operator_policy_canary
    delivery_adapters_factory: Callable[..., Any] = canonical_delivery_adapters
    inputs_validator: Callable[..., dict] | None = None
    clock: Callable[[], float] = time.time
    host_reader: Callable[[], dict] = host_identity


def _pending(intent, phases, blocker):
    result = seal_value({"schema_version": "operator_existing_policy_canary_continuation.v1",
        "status": "pending", "run_id": intent["run_id"], "intent_digest": intent["intent_digest"],
        "phases": phases, "blockers": [blocker], "new_provider_allocations": 0,
        "policy_execution_repeated": False, "raw_secret_values_recorded": False})
    root = metadata_root(intent)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "pending.tmp"
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n")
    temporary.replace(root / "pending.json")
    return result


def _safe_failure(intent, phase, exc):
    root = metadata_root(intent) / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    _seal(root / f"{phase}-{time.time_ns()}.json", {"phase": phase, "failure_type": type(exc).__name__,
                                                "raw_exception_recorded": False})


def _validate_collected(intent, receipt, *, rehash=False):
    binding = intent["ingestion_binding"]
    root = Path(intent["terminal_delivery_intent"]["run_root"]) / "provider_output"
    inventory = read_json(root / ".ingestion/member_inventory.json")
    native = read_json(root / "native" / binding["result_document"])
    if (receipt.get("status") != "collected_pending_finalization" or receipt.get("blockers") != []
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("binding_digest") != binding["binding_digest"]
            or str(receipt.get("instance_id")) != str(intent["instance_id"])
            or receipt.get("runtime_inputs_digest") != binding["runtime_inputs_digest"]
            or receipt.get("local_archive_copy_created") is not False
            or inventory.get("inventory_digest") != canonical_digest(inventory, digest_field="inventory_digest")
            or inventory.get("inventory_digest") != receipt.get("member_inventory_digest")
            or inventory.get("binding_digest") != binding["binding_digest"]
            or inventory.get("archive_sha256") != receipt.get("archive_sha256")
            or native.get("result_digest") != canonical_digest(native, digest_field="result_digest")
            or native.get("result_digest") != receipt.get("native_inventory", {}).get("result_document_digest")):
        raise ContinuationError("continuation_terminal_archive_unverified")
    if rehash:
        for relative, record in inventory["members"].items():
            from .provider_output_native_inventory import safe_member_name
            actual = file_record(root / "native" / safe_member_name(relative))
            if any(actual[key] != record.get(key) for key in ("sha256", "size_bytes")):
                raise ContinuationError("continuation_collected_evidence_changed")
    return native


def _teardown_proven(intent, result):
    recorded = result.get("recorded_vast_instance_teardown", {})
    return (result.get("status") == "provider_terminal" and result.get("provider_absence_confirmed") is True
        and result.get("resource_name_exact") == intent["resource_name"]
        and result.get("deadline_epoch") == intent["deadline_epoch"]
        and str(result.get("recorded_vast_instance", {}).get("instance_id")) == str(intent["instance_id"])
        and str(recorded.get("instance_id")) == str(intent["instance_id"])
        and recorded.get("provider_absence_confirmed") is True
        and result.get("final_inventory", {}).get("api_confirmed") is True
        and result.get("final_inventory", {}).get("live_resource_count") == 0)


def _teardown(intent, adapters, *, terminal_archive_verified):
    root = Path(intent["terminal_delivery_intent"]["run_root"])
    path = root / "vast_provider_run/vast_teardown_manifest.json"
    if path.exists():
        value = read_json(path)
        evidence = read_json(verified_record(value["canonical_termination"]))
        if not _teardown_proven(intent, evidence) or value.get("vast_instance_ids") != [intent["instance_id"]]:
            raise ContinuationError("continuation_teardown_receipt_changed")
        return value
    if not terminal_archive_verified and adapters.clock() < intent["deadline_epoch"]:
        raise ContinuationError("continuation_early_teardown_forbidden")
    result = adapters.terminator(intent)
    evidence_path = metadata_root(intent) / "attempts" / f"termination-{time.time_ns()}.json"
    _seal(evidence_path, result)
    if not _teardown_proven(intent, result):
        raise ContinuationError("continuation_exact_resource_absence_unproven")
    value = {"schema_version": "vast_teardown_manifest.v1", "status": "completed",
        "generated_at": result.get("completed_at"), "vast_instance_ids": [intent["instance_id"]],
        "teardown_actions_performed": result.get("terminations", []),
        "runner_gpu_teardown_completed": True, "continuing_spend_from_this_run": False,
        "provider_instance_absent": True, "canonical_termination": file_record(evidence_path),
        "trigger": "verified_terminal_archive" if terminal_archive_verified else "original_hard_deadline",
        "original_deadline_epoch": intent["deadline_epoch"], "original_watchdog": intent["original_watchdog"],
        "raw_secret_values_recorded": False}
    return _seal(path, value)


def _cleanup_valid(intent, cleanup):
    staging_record = intent["ingestion_binding"]["staging_manifest"]
    staging = read_json(verified_record(staging_record))
    keys = [staging["output_key"]] if staging.get("bundle_object_retained_for_reuse") is True else [staging["bundle_key"], staging["output_key"]]
    witness = staging.get("paired_witness") or {}
    if witness.get("status") == "ready":
        from .native_task_arena_paired_witness_staging import SUFFIX
        if witness.get("witness_key") != staging["output_key"] + SUFFIX:
            return False
        keys.append(witness["witness_key"])
    rows = cleanup.get("objects")
    expected = sorted(hashlib.sha256(key.encode()).hexdigest() for key in keys)
    return (cleanup.get("schema_version") == "wam_provider_object_store_cleanup.v1"
        and cleanup.get("staging_manifest_sha256") == staging_record["sha256"].removeprefix("sha256:")
        and cleanup.get("status") == "completed" and cleanup.get("blockers") == []
        and cleanup.get("all_objects_absent") is True and cleanup.get("all_ephemeral_objects_absent") is True
        and cleanup.get("exact_object_count") == len(keys) and isinstance(rows, list)
        and sorted(row.get("key_sha256", "") for row in rows) == expected
        and all(row.get("absence", {}).get("absence_confirmed") is True for row in rows))


def _build_recovery_adapter(intent, values, native, collected, teardown, cleanup_path, zero):
    root = Path(intent["terminal_delivery_intent"]["run_root"])
    path = root / "allocator_result.json"
    if path.exists():
        result = read_json(path)
        if result.get("continuation_intent_digest") != intent["intent_digest"]:
            raise ContinuationError("continuation_adapter_identity_changed")
        return result
    provider_path = root / "vast_provider_run/vast_provider_adapter_result.json"
    teardown_path = root / "vast_provider_run/vast_teardown_manifest.json"
    provider = {"schema_version": "vast_provider_adapter_result.v1", "status": "blocked",
        "vast_instance_ids": [intent["instance_id"]], "continuing_spend_from_this_run": False,
        "existing_instance_recovered": True, "provider_allocations_performed": 0,
        "canonical_termination": teardown["canonical_termination"], "raw_secret_values_recorded": False}
    _seal(provider_path, provider)
    manifest = build_task_evaluation_artifact_manifest(attempt_root=root,
        artifact_roots={"provider_runtime_evidence": root / "provider_output/native",
            "allocator_adapter_result": provider_path, "teardown_manifest": teardown_path,
            "staged_object_cleanup": cleanup_path},
        required_roles=["provider_runtime_evidence", "allocator_adapter_result", "teardown_manifest", "staged_object_cleanup"],
        binding={"continuation_intent_digest": intent["intent_digest"], "instance_id": intent["instance_id"],
                 "bundle_sha256": values["bundle"]["bundle_sha256"], "retry_cap": 0})
    if manifest["status"] != "completed":
        raise ContinuationError("continuation_artifact_manifest_incomplete")
    result = {"schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
        "run_id": intent["run_id"], "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "vast_instance_ids": [intent["instance_id"]], "retry_cap": 0, "hard_cap_usd": intent["hard_cap_usd"],
        "bundle_sha256": values["bundle"]["bundle_sha256"], "continuing_spend_from_this_run": False,
        "continuation_intent_digest": intent["intent_digest"], "provider_allocations_performed": 0,
        "existing_instance_recovered": True, "attempt_root": str(root),
        "native_control_result_path": intent["terminal_delivery_intent"]["native_result_path"],
        "native_control_result_digest": native["result_digest"], "ingestion_receipt_digest": collected["receipt_digest"],
        "adapter_result_path": str(provider_path), "teardown_manifest_path": str(teardown_path),
        "artifact_manifest_path": str(root / "artifact_manifest.json"), "object_store_cleanup_path": str(cleanup_path),
        "provider_closeout": {"teardown_manifest": file_record(teardown_path),
            "provider_zero_confirmed": zero["provider_zero_verified"], "warm_session_retained": False,
            "all_staged_objects_absent": read_json(cleanup_path)["all_objects_absent"]},
        "blockers": ["scientific_finalization_pending"], "raw_secret_values_recorded": False}
    return _seal(path, result)


def continue_existing_run(intent, *, adapters=ExistingRunContinuationAdapters()):
    """One resumable tick. Side-effect callbacks never include create/allocate."""
    validate_control_plane_host(intent, host_reader=adapters.host_reader)
    values = validate_continuation_intent(intent, inputs_validator=adapters.inputs_validator)
    root = Path(intent["terminal_delivery_intent"]["run_root"])
    meta = metadata_root(intent)
    meta.mkdir(parents=True, exist_ok=True)
    phases = {}
    with (meta / ".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "pending", "blockers": ["continuation_already_running"]}
        _seal(meta / "intent.json", dict(intent))
        try:
            validate_owner_handoff(intent)
        except (OSError, ValueError, KeyError):
            return _pending(intent, phases, "continuation_explicit_owner_handoff_pending")
        phases["owner_handoff"] = "complete"
        completed_path = meta / "completed.json"
        if completed_path.exists():
            completed = read_json(completed_path)
            terminal = read_json(root / "operator_terminal_delivery/completed.json")
            if (completed.get("intent_digest") != intent["intent_digest"]
                    or completed.get("receipt_digest") != canonical_digest(completed, digest_field="receipt_digest")
                    or terminal.get("result_digest") != canonical_digest(terminal, digest_field="result_digest")
                    or terminal.get("result_digest") != completed.get("terminal_delivery_digest")):
                raise ContinuationError("continuation_completed_receipt_changed")
            return completed
        collected_path = meta / "collected.json"
        phase = "ingestion"
        try:
            if not collected_path.exists() and adapters.clock() >= intent["deadline_epoch"]:
                _teardown(intent, adapters, terminal_archive_verified=False)
                phases["teardown"] = "complete_at_hard_deadline"
            if collected_path.exists():
                collected = read_json(collected_path)
                native = _validate_collected(intent, collected, rehash=not (root / "vast_provider_run/vast_teardown_manifest.json").exists())
            else:
                collected = adapters.collector(binding=intent["ingestion_binding"],
                    signed_get_url_file=intent["signed_get_url_file"], output_root=root / "provider_output",
                    # CPU/object-store custody has its own bounded attempt. The
                    # independent watchdog still enforces the original GPU deadline.
                    deadline_seconds=3600)
                if collected.get("status") != "collected_pending_finalization":
                    if adapters.clock() >= intent["deadline_epoch"]:
                        _teardown(intent, adapters, terminal_archive_verified=False)
                        phases["teardown"] = "complete_at_hard_deadline"
                    return _pending(intent, phases, "continuation_terminal_archive_pending")
                native = _validate_collected(intent, collected)
                _seal(collected_path, collected)
            phases["ingestion"] = "complete"
            phase = "teardown"
            teardown = _teardown(intent, adapters, terminal_archive_verified=True)
            phases[phase] = "complete"
            phase = "object_cleanup"
            cleanup_path = meta / "staged_object_cleanup.json"
            if cleanup_path.exists():
                cleanup = read_json(cleanup_path)
            else:
                original = Path(intent["ingestion_binding"]["staging_manifest"]["path"]).parent / "wam_provider_object_store_cleanup.json"
                cleanup = read_json(original) if original.exists() else {}
                if not _cleanup_valid(intent, cleanup):
                    cleanup = adapters.cleanup(intent)
                if not _cleanup_valid(intent, cleanup):
                    return _pending(intent, phases, "continuation_exact_object_cleanup_pending")
                _seal(cleanup_path, cleanup)
            if not _cleanup_valid(intent, cleanup):
                raise ContinuationError("continuation_cleanup_receipt_changed")
            phases[phase] = "complete"
            phase = "provider_zero"
            zero_path = Path(intent["terminal_delivery_intent"]["provider_zero_path"])
            zero = _sealed_provider_zero(zero_path)
            if zero is None:
                zero = adapters.provider_zero_reader()
                candidate = meta / "attempts" / f"zero-{time.time_ns()}.json"
                _seal(candidate, zero)
                if _sealed_provider_zero(candidate) is None:
                    return _pending(intent, phases, "continuation_global_provider_zero_pending")
                _seal(zero_path, zero)
            phases[phase] = "complete"
            phase = "recovery_adapter"
            _build_recovery_adapter(intent, values, native, collected, teardown, cleanup_path, zero)
            phases[phase] = "complete"
            phase = "terminal_delivery"
            result = adapters.finalizer(intent["terminal_delivery_intent"], adapters=adapters.delivery_adapters_factory(intent))
            if (result.get("status") != "completed" or result.get("all_required_phases_done") is not True
                    or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")):
                phases[phase] = result.get("phases", {})
                return _pending(intent, phases, "continuation_billing_or_website_finalization_pending")
            phases[phase] = "complete"
            final = seal_value({"schema_version": "operator_existing_policy_canary_continuation.v1",
                "status": "completed", "intent_digest": intent["intent_digest"], "run_id": intent["run_id"],
                "phases": phases, "terminal_delivery_digest": result["result_digest"], "blockers": [],
                "new_provider_allocations": 0, "policy_execution_repeated": False, "raw_secret_values_recorded": False})
            return _seal(meta / "completed.json", final)
        except Exception as exc:
            _safe_failure(intent, phase, exc)
            return _pending(intent, phases, "continuation_" + phase + "_pending")


def run_existing_watchdog(intent, *, intent_path, host_reader=host_identity):
    """Adopt an original absolute deadline; restart after expiry cannot extend it."""
    validate_control_plane_host(intent, host_reader=host_reader)
    validate_continuation_intent(intent)
    root = Path(intent["watchdog_root"])
    root.mkdir(parents=True, exist_ok=True)
    started = root / watchdog.VAST_STARTED_INSTANCE_ID_NAME
    if started.exists() and started.read_text().strip() != str(intent["instance_id"]):
        raise ContinuationError("continuation_watchdog_started_id_changed")
    if not started.exists():
        descriptor = os.open(started, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(intent["instance_id"]) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    _seal(root / "existing_run_adoption.json", {"intent_digest": intent["intent_digest"],
        "original_watchdog": intent["original_watchdog"], "original_started_instance": intent["original_started_instance"],
        "deadline_epoch": intent["deadline_epoch"], "armed_before_original_allocation": False,
        "resource_creation_permitted": False})
    record_watchdog_process(intent, intent_path)
    if time.time() + 60 < intent["deadline_epoch"]:
        return watchdog.run_watchdog(out_dir=root, pod_name_prefix=intent["resource_name"],
            resource_name_exact=intent["resource_name"], deadline_epoch=intent["deadline_epoch"], provider_name="vast",
            provider_factory=lambda name: _ExactExistingProvider(get_render_provider(name), intent["instance_id"]))
    # Canonical arm refuses a fresh lease under 60 seconds. An existing-run
    # restart waits out only the original remainder, then uses canonical teardown.
    while time.time() < intent["deadline_epoch"]:
        time.sleep(min(10, intent["deadline_epoch"] - time.time()))
    original = read_json(verified_record(intent["original_watchdog"]))
    armed = {**original, "watchdog_out_dir": str(root), "resource_name_exact": intent["resource_name"],
             "status": "existing_run_deadline_elapsed", "pid": os.getpid()}
    result = watchdog.terminate_canary_resources(provider=_ExactExistingProvider(get_render_provider("vast"), intent["instance_id"]),
        pod_name_prefix=intent["resource_name"], resource_name_exact=intent["resource_name"], armed=armed, provider_name="vast")
    _seal(root / f"deadline_recovery_result-{time.time_ns()}.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "readiness", "watchdog", "process-identity", "commit-handoff", "host-identity"))
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--marker")
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "host-identity":
            result = host_identity()
        elif args.action == "process-identity":
            result = process_identity(args.pid, required_tokens=("blueprint_pipeline.paid_resource_allocator", args.marker))
        else:
            intent = read_json(args.intent)
            if args.action == "run":
                result = continue_existing_run(intent)
            elif args.action == "readiness":
                result = cloud_readiness(intent, inventory_reader=live_inventory)
            elif args.action == "watchdog":
                result = run_existing_watchdog(intent, intent_path=args.intent)
            else:
                result = commit_owner_handoff(intent, read_json(args.readiness))
        if args.output:
            _seal(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key in (
            "status", "blockers", "run_id", "receipt_digest", "process_identity_digest", "platform", "hostname", "machine_id_sha256")}, sort_keys=True))
        return 2 if args.action == "watchdog" and result.get("status") != "provider_terminal" else 0
    except Exception as exc:
        print(json.dumps({"status": "blocked", "failure_type": type(exc).__name__, "raw_exception_recorded": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
