from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.vast_official_billing_extractor import _terminal_evidence


INSTANCE_ID = 48_901_234
RUN_ID = "adp-new-scene-simple-relocation-839873-billing-fixture"


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, value: dict, digest_field: str) -> Path:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    return _write(path, value)


def _scene_configuration_terminal_fixture(tmp_path: Path) -> Path:
    run_root = tmp_path / "task-evaluation-launch-runs" / RUN_ID
    allocator = run_root / "allocator"
    job = allocator / "scene-configuration-job"
    provider_run = job / "provider_run"
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "task-evaluation-scene-configuration-fixture",
        "reconciliation": {"required_providers": ["vast"]},
        "profile_digest": "",
    }
    _identity(run_root / "launch_profile.json", profile, "profile_digest")
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": RUN_ID,
        "run_id": RUN_ID,
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "request_digest": "",
    }
    _identity(run_root / "launch_request.json", request, "request_digest")
    binding = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": RUN_ID,
        "run_id": RUN_ID,
        "request_digest": request["request_digest"],
        "profile_digest": profile["profile_digest"],
        "binding_digest": "",
    }
    _identity(run_root / "launch_binding.json", binding, "binding_digest")
    started = {
        "schema_version": "task_evaluation_launch_started.v1",
        "launch_id": RUN_ID,
        "run_id": RUN_ID,
        "request_digest": request["request_digest"],
        "binding_digest": binding["binding_digest"],
        "automatic_retry_authorized": False,
        "started_digest": "",
    }
    _identity(run_root / "launch_started.json", started, "started_digest")

    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": "completed",
        "provider_bundle_kind": "task_evaluation_scene_configuration",
        "provider_create_attempted": True,
        "vast_instance_ids": [INSTANCE_ID],
        "continuing_spend_from_this_run": False,
        "final_validation_status": "passed",
        "retained_owned": False,
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
    }
    adapter_path = _write(
        provider_run / "vast_provider_adapter_result.json", adapter
    )
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "vast_instance_ids": [INSTANCE_ID],
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "retention_authorized": False,
        "raw_secret_values_recorded": False,
    }
    teardown_path = _write(provider_run / "vast_teardown_manifest.json", teardown)
    runtime_path = job / "immutable_execution" / "provider-failure.log"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text("typed provider blocker\n", encoding="utf-8")
    rows = [
        {
            "relative_path": "provider_run/vast_provider_adapter_result.json",
            "roles": ["allocator_adapter_result", "provider_run_diagnostics"],
            "size_bytes": adapter_path.stat().st_size,
            "sha256": _sha256(adapter_path),
        },
        {
            "relative_path": "provider_run/vast_teardown_manifest.json",
            "roles": ["provider_run_diagnostics", "teardown_manifest"],
            "size_bytes": teardown_path.stat().st_size,
            "sha256": _sha256(teardown_path),
        },
        {
            "relative_path": "immutable_execution/provider-failure.log",
            "roles": ["provider_runtime_evidence"],
            "size_bytes": runtime_path.stat().st_size,
            "sha256": _sha256(runtime_path),
        },
    ]
    bundle_digest = "sha256:" + "4" * 64
    manifest = {
        "schema_version": "task_evaluation_artifact_manifest.v1",
        "status": "completed",
        "binding": {
            "allocator_lane": "task_evaluation_scene_configuration",
            "bundle_sha256": bundle_digest,
            "provider": "vast",
            "retry_cap": 0,
        },
        "required_roles": [
            "allocator_adapter_result",
            "provider_runtime_evidence",
            "teardown_manifest",
        ],
        "observed_roles": [
            "allocator_adapter_result",
            "provider_run_diagnostics",
            "provider_runtime_evidence",
            "teardown_manifest",
        ],
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "files": rows,
        "blockers": [],
        "raw_secret_values_recorded": False,
        "manifest_digest": "",
    }
    manifest_path = _identity(
        job / "artifact_manifest.json", manifest, "manifest_digest"
    )
    watchdog = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "status": "provider_terminal",
        "instance_ids": [INSTANCE_ID],
        "provider_absence_confirmed": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    result = {
        "schema_version": "task_evaluation_scene_configuration_vast_result.v1",
        "status": "blocked",
        "bundle_sha256": bundle_digest,
        "provider_adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "artifact_manifest_path": str(manifest_path),
        "provider_mutations_performed": 1,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "independent_watchdog": watchdog,
        "object_store_cleanup": {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
            "raw_secret_values_recorded": False,
            "blockers": [],
        },
        "blockers": ["provider_result_blocker:fixture"],
        "raw_secret_values_recorded": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path = _write(
        job / "task_evaluation_scene_configuration_vast_result.v1.json", result
    )
    allocator_result = _write(allocator / "result.json", result)
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "blocked",
        "launch_id": RUN_ID,
        "run_id": RUN_ID,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "binding_digest": binding["binding_digest"],
        "execute_requested": True,
        "raw_secret_values_recorded": False,
        "terminal_evidence": {
            "result": {
                "path": str(allocator_result),
                "digest": _sha256(allocator_result),
                "exists": True,
            },
            "artifacts": {
                "teardown_manifest_path": {
                    "path": str(teardown_path),
                    "digest": _sha256(teardown_path),
                    "exists": True,
                },
                "artifact_manifest_path": {
                    "path": str(manifest_path),
                    "digest": _sha256(manifest_path),
                    "exists": True,
                },
            },
        },
        "receipt_digest": "",
    }
    _identity(run_root / "launch_receipt.json", receipt, "receipt_digest")
    zero = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "launch_id": RUN_ID,
        "run_id": RUN_ID,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "automatic_retry_performed": False,
        "provider_mutation_performed": False,
        "required_providers": ["vast"],
        "blockers": [],
        "provider_zero_receipt_digest": "",
    }
    _identity(
        run_root / "post_teardown_provider_zero_receipt.json",
        zero,
        "provider_zero_receipt_digest",
    )
    sync = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "attempt_number": 1,
        "launch_id": RUN_ID,
        "run_id": RUN_ID,
        "request_digest": request["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "provider_mutation_performed": False,
        "response": {
            "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
            "status": "blocked",
            "already_exists": False,
            "launch_id": RUN_ID,
            "run_id": RUN_ID,
            "request_digest": request["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "sync_result_digest": "",
    }
    _identity(run_root / "webapp_sync_succeeded.json", sync, "sync_result_digest")
    return result_path


def test_blocked_scene_configuration_attempt_is_officially_reconcilable(
    tmp_path: Path,
) -> None:
    result_path = _scene_configuration_terminal_fixture(tmp_path)

    evidence = _terminal_evidence(
        instance_id=INSTANCE_ID,
        terminal_result_path=result_path,
    )

    assert evidence["terminal_status"] == "blocked"
    assert evidence["provider_zero_verified"] is True
    assert evidence["artifact_manifest"]["status"] == "completed"
    assert evidence["webapp_terminal_binding"]["status"] == "succeeded"
