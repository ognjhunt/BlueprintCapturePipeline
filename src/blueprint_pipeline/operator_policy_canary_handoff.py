"""Read-only cloud readiness and explicit transfer of an existing allocator owner.

No signal is sent here. The local owner must pause its exact allocator before
committing handoff; the GPU worker is outside this process protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .operator_policy_canary_terminal_delivery import file_record, _seal

INTENT_SCHEMA = "operator_existing_policy_canary_continuation_intent.v1"
READY_SCHEMA = "operator_existing_policy_canary_cloud_readiness.v1"
HANDOFF_SCHEMA = "operator_existing_policy_canary_owner_handoff.v1"
MODULE = "blueprint_pipeline.operator_policy_canary_continuation"


class ContinuationError(ValueError):
    pass


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    file_record(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContinuationError("continuation_document_invalid")
    return value


def verified_record(record: Mapping[str, Any]) -> Path:
    actual = file_record(record["path"])
    if any(actual[key] != record.get(key) for key in ("sha256", "size_bytes")):
        raise ContinuationError("continuation_immutable_input_changed")
    return Path(actual["path"])


def seal_value(value: dict, field="receipt_digest") -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def host_identity() -> dict[str, Any]:
    machine = Path("/etc/machine-id")
    return {"platform": platform.system(), "hostname": socket.gethostname(),
        "machine_id_sha256": "sha256:" + hashlib.sha256(machine.read_bytes()).hexdigest() if machine.is_file() else None}


def process_identity(pid: int, *, required_tokens=(), runner=subprocess.run) -> dict[str, Any]:
    """Record a narrow process identity while keeping raw argv out of receipts."""
    if type(pid) is not int or pid <= 0:
        raise ContinuationError("continuation_process_id_invalid")
    rows = {}
    for field in ("lstart", "stat", "command"):
        result = runner(["ps", "-p", str(pid), "-o", field + "="], check=False,
                        capture_output=True, text=True, timeout=10)
        if result.returncode or not result.stdout.strip():
            raise ContinuationError("continuation_process_not_observed")
        rows[field] = result.stdout.strip()
    command = rows["command"]
    if any(str(token) not in command.split() for token in required_tokens):
        raise ContinuationError("continuation_process_command_mismatch")
    result = {"pid": pid, "started_at_local": rows["lstart"], "state": rows["stat"],
              "command_sha256": "sha256:" + hashlib.sha256(command.encode()).hexdigest(),
              "hostname": socket.gethostname(), "required_command_tokens_verified": True}
    result["process_identity_digest"] = canonical_digest(
        {key: value for key, value in result.items() if key != "state"})
    return result


def validate_continuation_intent(intent, *, inputs_validator=None):
    if (intent.get("schema_version") != INTENT_SCHEMA
            or intent.get("intent_digest") != canonical_digest(intent, digest_field="intent_digest")
            or intent.get("provider") != "vast" or intent.get("new_provider_allocations_permitted") != 0
            or intent.get("automatic_paid_retry_permitted") is not False
            or type(intent.get("instance_id")) is not int or intent["instance_id"] <= 0
            or not isinstance(intent.get("deadline_epoch"), (int, float))
            or not math.isfinite(intent["deadline_epoch"])):
        raise ContinuationError("continuation_intent_invalid")
    terminal = intent["terminal_delivery_intent"]
    if inputs_validator is None:
        from .operator_policy_canary_terminal_delivery import validate_operator_terminal_delivery_inputs
        values = validate_operator_terminal_delivery_inputs(terminal, require_relocated=True)
    else:
        values = inputs_validator(terminal)
    authority, bundle, runtime = (values[name] for name in ("session_authority", "bundle", "runtime_inputs"))
    original = read_json(verified_record(intent["original_watchdog"]))
    started = verified_record(intent["original_started_instance"])
    binding = intent["ingestion_binding"]
    staging = read_json(verified_record(binding["staging_manifest"]))
    root = Path(terminal["run_root"])
    if (not root.is_absolute() or root.is_symlink()
            or terminal["run_id"] != intent["run_id"] or authority["run_id"] != intent["run_id"]
            or authority["resource_name"] != intent["resource_name"]
            or authority["hard_cap_usd"] != intent["hard_cap_usd"]
            or values["operator_authorization"].get("policy_cap_usd") != intent["hard_cap_usd"]
            or bundle["implementation_commit"] != intent["scientific_commit"]
            or authority.get("retry_cap") != 0 or bundle.get("retry_cap") != 0
            or original.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
            or original.get("deadline_epoch") != intent["deadline_epoch"]
            or original.get("pod_name_prefix") != intent["resource_name"]
            or original.get("provider") != "vast" or original.get("independent_process") is not True
            or started.read_text().strip() != str(intent["instance_id"])
            or binding.get("run_id") != intent["run_id"]
            or str(binding.get("instance_id")) != str(intent["instance_id"])
            or binding.get("runtime_inputs_digest") != runtime["runtime_inputs_digest"]
            or binding.get("binding_digest") != canonical_digest(binding, digest_field="binding_digest")
            or staging.get("schema_version") != "wam_provider_object_store_staging.v1"
            or staging.get("status") != "completed" or staging.get("output_key_run_unique") is not True
            or staging.get("bundle_sha256", "").removeprefix("sha256:") != bundle["bundle_sha256"].removeprefix("sha256:")
            or Path(terminal["allocator_result_path"]) != root / "allocator_result.json"
            or Path(terminal["provider_zero_path"]) != root / "post_teardown_global_provider_zero.json"
            or Path(terminal["native_result_path"]) != root / "provider_output/native" / binding["result_document"]):
        raise ContinuationError("continuation_original_execution_identity_mismatch")
    if not intent.get("control_plane_machine_id_sha256") or not intent.get("local_allocator_identity", {}).get("process_identity_digest"):
        raise ContinuationError("continuation_owner_identity_missing")
    return values


def build_continuation_intent(*, terminal_delivery_intent, ingestion_binding, instance_id,
        resource_name, deadline_epoch, scientific_commit, hard_cap_usd, original_watchdog,
        original_started_instance, local_allocator_identity, local_allocator_marker,
        control_plane_machine_id_sha256, watchdog_root, signed_get_url_file,
        billing_period_start_at, billing_secrets_dir="/etc/blueprint/provider-secrets",
        cleanup_configuration=None) -> dict:
    return seal_value({"schema_version": INTENT_SCHEMA, "provider": "vast",
        "run_id": terminal_delivery_intent["run_id"], "terminal_delivery_intent": terminal_delivery_intent,
        "ingestion_binding": ingestion_binding, "instance_id": instance_id, "resource_name": resource_name,
        "deadline_epoch": deadline_epoch, "scientific_commit": scientific_commit, "hard_cap_usd": hard_cap_usd,
        "original_watchdog": original_watchdog, "original_started_instance": original_started_instance,
        "local_allocator_identity": local_allocator_identity, "local_allocator_marker": local_allocator_marker,
        "control_plane_machine_id_sha256": control_plane_machine_id_sha256,
        "watchdog_root": str(watchdog_root), "signed_get_url_file": str(signed_get_url_file),
        "billing_period_start_at": billing_period_start_at, "billing_secrets_dir": str(billing_secrets_dir),
        "cleanup_configuration": dict(cleanup_configuration or {}),
        "new_provider_allocations_permitted": 0, "automatic_paid_retry_permitted": False,
        "raw_secret_values_recorded": False}, "intent_digest")


def metadata_root(intent):
    return Path(intent["terminal_delivery_intent"]["run_root"]) / "existing_run_continuation"


def cloud_readiness(intent, *, inventory_reader, process_reader=process_identity,
                    host_reader=host_identity, clock=time.time, inputs_validator=None) -> dict:
    """Prove CP custody and an independently running exact-deadline watchdog."""
    validate_continuation_intent(intent, inputs_validator=inputs_validator)
    from .provider_output_range_ingestion import validate_ingestion_binding
    validate_ingestion_binding(intent["ingestion_binding"], intent["signed_get_url_file"])
    host = host_reader()
    watchdog_path = Path(intent["watchdog_root"]) / "groot_oscar_runpod_canary_watchdog.json"
    watchdog = read_json(watchdog_path)
    started = Path(intent["watchdog_root"]) / "started_vast_instance_id.txt"
    process = process_reader(watchdog["pid"], required_tokens=("-m", MODULE, "watchdog"))
    inventory = inventory_reader(intent)
    rows = inventory.get("resources")
    if (host.get("platform") != "Linux" or host.get("machine_id_sha256") != intent["control_plane_machine_id_sha256"]
            or watchdog.get("status") != "armed" or watchdog.get("independent_process") is not True
            or watchdog.get("provider") != "vast" or watchdog.get("resource_name_exact") != intent["resource_name"]
            or Path(watchdog.get("watchdog_out_dir", "")) != Path(intent["watchdog_root"])
            or watchdog.get("pod_name_prefix") != intent["resource_name"]
            or watchdog.get("deadline_epoch") != intent["deadline_epoch"]
            or started.is_symlink() or started.read_text().strip() != str(intent["instance_id"])
            or process.get("pid") == os.getpid() or process.get("state", "").startswith(("T", "Z"))
            or process.get("required_command_tokens_verified") is not True
            or clock() >= intent["deadline_epoch"]
            or inventory.get("api_confirmed") is not True or inventory.get("live_resource_count") != 1
            or not isinstance(rows, list) or len(rows) != 1
            or str(rows[0].get("instance_id")) != str(intent["instance_id"])
            or rows[0].get("name") != intent["resource_name"]):
        raise ContinuationError("continuation_cloud_readiness_unproven")
    snapshot = metadata_root(intent) / "watchdog_arm_snapshots" / (file_record(watchdog_path)["sha256"].removeprefix("sha256:") + ".json")
    _seal(snapshot, watchdog)
    value = seal_value({"schema_version": READY_SCHEMA, "status": "ready_for_owner_handoff",
        "intent_digest": intent["intent_digest"], "observed_epoch": clock(), "host": host,
        "instance_id": intent["instance_id"], "resource_name": intent["resource_name"],
        "deadline_epoch": intent["deadline_epoch"], "hard_cap_usd": intent["hard_cap_usd"],
        "scientific_commit": intent["scientific_commit"], "immutable_inputs_readable": True,
        "watchdog": file_record(snapshot), "watchdog_live_evidence_path": str(watchdog_path), "started_instance": file_record(started),
        "watchdog_process": process, "inventory": inventory,
        "original_watchdog": intent["original_watchdog"], "cp_watchdog_armed_before_original_allocation": False,
        "provider_mutations_performed": 0, "raw_secret_values_recorded": False})
    meta = metadata_root(intent)
    if (meta / "owner_handoff_commit.json").exists():
        raise ContinuationError("continuation_owner_already_handed_off")
    _seal(meta / "readiness_attempts" / (value["receipt_digest"].removeprefix("sha256:") + ".json"), value)
    temporary = meta / "cloud_readiness.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(meta / "cloud_readiness.json")
    return value


def commit_owner_handoff(intent, readiness, *, process_reader=process_identity, clock=time.time):
    """Observe an already-paused owner; this function never pauses a process."""
    process = process_reader(intent["local_allocator_identity"]["pid"], required_tokens=(
        "blueprint_pipeline.paid_resource_allocator", intent["local_allocator_marker"]))
    if (readiness.get("schema_version") != READY_SCHEMA
            or readiness.get("receipt_digest") != canonical_digest(readiness, digest_field="receipt_digest")
            or readiness.get("intent_digest") != intent["intent_digest"]
            or readiness.get("status") != "ready_for_owner_handoff"
            or not 0 <= clock() - readiness["observed_epoch"] <= 300
            or process.get("process_identity_digest") != intent["local_allocator_identity"]["process_identity_digest"]
            or not process.get("state", "").startswith("T")):
        raise ContinuationError("continuation_owner_pause_not_proven")
    return seal_value({"schema_version": HANDOFF_SCHEMA, "status": "owner_handoff_committed",
        "intent_digest": intent["intent_digest"], "cloud_readiness_digest": readiness["receipt_digest"],
        "committed_epoch": clock(), "allocator_process": process, "allocator_signals_sent_by_coordinator": 0,
        "gpu_worker_mutated": False, "new_paid_attempt_authorized": False})


def validate_owner_handoff(intent):
    root = metadata_root(intent)
    ready, commit = (read_json(root / name) for name in ("cloud_readiness.json", "owner_handoff_commit.json"))
    if (ready.get("schema_version") != READY_SCHEMA or ready.get("status") != "ready_for_owner_handoff"
            or ready.get("receipt_digest") != canonical_digest(ready, digest_field="receipt_digest")
            or ready.get("intent_digest") != intent["intent_digest"]
            or ready.get("deadline_epoch") != intent["deadline_epoch"]
            or ready.get("instance_id") != intent["instance_id"]
            or commit.get("schema_version") != HANDOFF_SCHEMA or commit.get("status") != "owner_handoff_committed"
            or commit.get("receipt_digest") != canonical_digest(commit, digest_field="receipt_digest")
            or commit.get("intent_digest") != intent["intent_digest"]
            or commit.get("cloud_readiness_digest") != ready["receipt_digest"]
            or not 0 <= commit["committed_epoch"] - ready["observed_epoch"] <= 300
            or commit.get("allocator_process", {}).get("process_identity_digest") != intent["local_allocator_identity"]["process_identity_digest"]
            or not commit.get("allocator_process", {}).get("state", "").startswith("T")
            or commit.get("gpu_worker_mutated") is not False or commit.get("new_paid_attempt_authorized") is not False):
        raise ContinuationError("continuation_owner_handoff_invalid")
    return commit
