from __future__ import annotations

import json
import math
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_billing_reconciler import (
    BILLING_SOURCE_SCHEMA_VERSION,
    VAST_CHARGES_URL,
)
from blueprint_pipeline.vast_official_billing_extractor import (
    ENTRY_SCHEMA_VERSION,
    RECONCILIATION_SCHEMA_VERSION,
    VastOfficialBillingExtractionError,
    main,
    materialize_vast_official_same_goal_reconciliation,
    validate_vast_official_same_goal_reconciliation,
)


INSTANCE_A = 47_912_530
INSTANCE_B = 47_913_976
LABEL_A = "blueprint-groot-oscar-canary-adp-artifixer3d-1786935680"
LABEL_B = "blueprint-groot-oscar-canary-adp-artifixer3d-1786937589"
PAIRED_INSTANCE = 47_933_056
PAIRED_LABEL = "blueprint-adp-paired-native-import-1786959124"
PAIRED_RUN_ID = (
    "adp-paired-native-840920-846bce86-r3-api-20260817T093112Z-902a4451"
)
CONTENT_AGENTS_INSTANCE = 47_940_042
CONTENT_AGENTS_LABEL = (
    "blueprint-adp-content-agents-20260817t113743803190000-1786966665"
)
CONTENT_AGENTS_RUN_ID = (
    "adp-content-agents-840920-task-a-6bcc65db-r3-api-"
    "20260817T113646Z-bfedb8c1"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPOSITORY_ROOT / "scripts/materialize_vast_official_same_goal_reconciliation.py"


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _charge(
    *,
    instance_id: int,
    label: str,
    total: float,
    gpu: float,
    disk: float,
    bandwidth_download: float = 0.0,
    bandwidth_upload: float = 0.0,
) -> dict:
    return {
        "source": f"instance-{instance_id}",
        "amount": total,
        "description": f"Instance {instance_id} Charges - 1 day",
        "type": "instance",
        "start": 1_786_924_800,
        "end": 1_786_924_800,
        "items": [
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "gpu",
                "source": None,
                "description": "production-shaped GPU line",
                "amount": gpu,
                "metadata": {},
                "items": [],
            },
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "disk",
                "source": None,
                "description": "production-shaped disk line",
                "amount": disk,
                "metadata": {},
                "items": [],
            },
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "bwd",
                "source": None,
                "description": "0.0 GB Downloaded",
                "amount": bandwidth_download,
                "metadata": {},
                "items": [],
            },
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "bwu",
                "source": None,
                "description": "0.0 GB Uploaded",
                "amount": bandwidth_upload,
                "metadata": {},
                "items": [],
            },
        ],
        "metadata": {"label": label},
    }


def _bound(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _terminal_fixture(
    tmp_path: Path,
    *,
    instance_id: int,
    status: str,
    paired_native: bool = False,
    run_id_override: str | None = None,
    profile_id_override: str | None = None,
) -> Path:
    run_id = run_id_override or (
        PAIRED_RUN_ID if paired_native else f"adp-artifixer3d-fixture-{instance_id}"
    )
    profile_id = profile_id_override or (
        "adp-paired-target-native-import-live-846bce86-r3"
        if paired_native
        else f"adp-artifixer3d-live-fixture-{instance_id}"
    )
    run_root = tmp_path / "task-evaluation-launch-runs" / run_id
    allocator = run_root / "allocator"
    job = allocator / (
        "paired-target-native-import-job" if paired_native else "artifixer3d-job"
    )
    provider_run = job / "vast_provider_run"

    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "execution_admission": {"live_enabled": True, "blockers": []},
        "reconciliation": {"required_providers": ["vast"]},
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    profile_path = _write(run_root / "launch_profile.json", profile)
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "launch_profile_id": profile_id,
        "launch_profile_digest": profile["profile_digest"],
        "idempotency_key": run_id,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path = _write(run_root / "launch_request.json", request)
    binding = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "profile_digest": profile["profile_digest"],
        "execute_requested": True,
        "binding_digest": "",
    }
    binding["binding_digest"] = canonical_digest(
        binding, digest_field="binding_digest"
    )
    binding_path = _write(run_root / "launch_binding.json", binding)
    started = {
        "schema_version": "task_evaluation_launch_started.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "binding_digest": binding["binding_digest"],
        "automatic_retry_authorized": False,
        "started_digest": "",
    }
    started["started_digest"] = canonical_digest(
        started, digest_field="started_digest"
    )
    started_path = _write(run_root / "launch_started.json", started)

    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": status,
        "provider_bundle_kind": (
            "paired_target_native_import" if paired_native else "public_scene_artifixer3d"
        ),
        "provider_create_attempted": True,
        "vast_instance_ids": [instance_id],
        "continuing_spend_from_this_run": False,
        "final_validation_status": "passed",
        "retained_owned": False,
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
        "blockers": [] if status == "completed" else ["runtime_result_missing"],
    }
    adapter_path = _write(provider_run / "vast_provider_adapter_result.json", adapter)
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "vast_instance_ids": [instance_id],
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "retention_authorized": False,
        "raw_secret_values_recorded": False,
        "zero_continuing_spend_scope": "all Vast instances created were destroyed",
    }
    teardown_path = _write(provider_run / "vast_teardown_manifest.json", teardown)
    independent_watchdog = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "status": "provider_terminal",
        "instance_ids": [instance_id],
        "provider_absence_confirmed": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    result = {
        "schema_version": (
            "paired_target_native_import_vast_run.v1"
            if paired_native
            else "public_scene_artifixer3d_vast_run.v1"
        ),
        "status": status,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "raw_secret_values_recorded": False,
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "independent_watchdog": independent_watchdog,
        "blockers": [] if status == "completed" else ["runtime_result_missing"],
    }
    if paired_native:
        prefix = "blueprint-adp-paired-native-import-20260817t093112000000000-"
        lane_zero = {
            "status": "observed",
            "provider": "vast",
            "name_prefix": prefix,
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
        }
        global_zero = {**lane_zero, "name_prefix": ""}
        inspect = {
            "status": "absent",
            "provider": "vast",
            "instance_id": str(instance_id),
            "http": 404,
            "api_confirmed": True,
            "provider_absence_confirmed": True,
        }
        watchdog_receipt = {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "provider_terminal",
            "provider": "vast",
            "pod_name_prefix": prefix,
            "provider_absence_confirmed": True,
            "owner_teardown_cancel_requested": True,
            "owner_teardown_cancel_request_valid": True,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
            "recorded_vast_instance_teardown": {
                "status": "absent",
                "instance_id": str(instance_id),
                "provider_absence_confirmed": True,
                "provider_mutations_performed": 0,
                "inspect_attempts": [
                    {**inspect, "attempt": 1},
                    {**inspect, "attempt": 2},
                ],
            },
            "initial_inventory": lane_zero,
            "final_inventory": lane_zero,
            "initial_global_inventory": global_zero,
            "final_global_inventory": global_zero,
        }
        watchdog_path = _write(
            job
            / "independent_vast_watchdog"
            / "groot_oscar_runpod_canary_watchdog.json",
            watchdog_receipt,
        )
        cleanup_path = _write(
            job
            / "object_store_staging"
            / "wam_provider_object_store_cleanup.json",
            {
                "schema_version": "wam_provider_object_store_cleanup.v1",
                "status": "completed",
                "all_objects_absent": True,
                "signed_url_files_removed": True,
                "blockers": [],
            },
        )
        result.update(
            {
                "request_digest": "sha256:" + "9" * 64,
                "replacement_count": 2,
                "all_staged_objects_absent": True,
                "watchdog_receipt_path": str(watchdog_path),
                "object_store_cleanup_path": str(cleanup_path),
            }
        )
        result_path = _write(job / "paired_target_native_import_vast_result.v1.json", result)
    else:
        result["provider_closeout"] = {
            "adapter_result": _bound(adapter_path),
            "teardown_manifest": _bound(teardown_path),
            "provider_zero_confirmed": True,
            "all_staged_objects_absent": True,
        }
        result_path = _write(job / "public_scene_artifixer3d_vast_result.json", result)
    allocator_result_path = _write(allocator / "result.json", result)
    terminal_status = "passed" if status == "completed" else "blocked"
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": status,
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "binding_digest": binding["binding_digest"],
        "execute_requested": True,
        "raw_secret_values_recorded": False,
        "terminal_evidence": {
            "status": terminal_status,
            "result": {
                "path": str(allocator_result_path),
                "digest": _sha256(allocator_result_path),
                "exists": True,
            },
            "artifacts": {
                "teardown_manifest_path": {
                    "path": str(teardown_path),
                    "digest": _sha256(teardown_path),
                    "exists": True,
                }
            },
            "blockers": [] if status == "completed" else ["terminal_blocked"],
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = _write(run_root / "launch_receipt.json", receipt)
    zero = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "launch_id": run_id,
        "run_id": run_id,
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
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    zero_path = _write(run_root / "post_teardown_provider_zero_receipt.json", zero)
    assert all(
        path.is_file()
        for path in (
            profile_path,
            request_path,
            binding_path,
            started_path,
            receipt_path,
            zero_path,
        )
    )
    return result_path


def _refresh_terminal_bindings(result_path: Path) -> None:
    run_root = result_path.parents[2]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    adapter_path = Path(result["adapter_result_path"])
    teardown_path = Path(result["teardown_manifest_path"])
    if "provider_closeout" in result:
        result["provider_closeout"]["adapter_result"] = _bound(adapter_path)
        result["provider_closeout"]["teardown_manifest"] = _bound(teardown_path)
    _write(result_path, result)
    allocator_result_path = _write(run_root / "allocator" / "result.json", result)
    receipt_path = run_root / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["terminal_evidence"]["result"]["digest"] = _sha256(
        allocator_result_path
    )
    receipt["terminal_evidence"]["artifacts"]["teardown_manifest_path"][
        "digest"
    ] = _sha256(teardown_path)
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(receipt_path, receipt)
    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    zero["receipt_digest"] = receipt["receipt_digest"]
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    _write(zero_path, zero)


def _fixture(tmp_path: Path) -> dict[str, object]:
    audit = tmp_path / "billing-audit" / "20260817T055421.193507Z"
    responses = [
        _write(
            audit / "response-004-vast.json",
            {
                "success": True,
                "next_token": "page-two",
                "results": [
                    _charge(
                        instance_id=INSTANCE_A,
                        label=LABEL_A,
                        total=0.123,
                        gpu=0.112,
                        disk=0.011,
                    ),
                    {
                        "source": "instance-1",
                        "amount": 9.0,
                        "type": "instance",
                        "items": [],
                        "metadata": {"label": "blueprint-unrelated-private-label"},
                        "unretained_secret": "raw-response-secret",
                    },
                ],
            },
        ),
        _write(
            audit / "response-005-vast.json",
            {
                "success": True,
                "next_token": None,
                "results": [
                    _charge(
                        instance_id=INSTANCE_B,
                        label=LABEL_B,
                        total=2.183,
                        gpu=2.056,
                        disk=0.127,
                    )
                ],
            },
        ),
    ]
    receipt = {
        "schema_version": BILLING_SOURCE_SCHEMA_VERSION,
        "status": "reconciled",
        "generated_at": "2026-08-17T05:54:21.193507+00:00",
        "cohort_start_at": "2026-07-01T00:00:00+00:00",
        "cohort_end_at": "2026-08-17T05:54:21.193507+00:00",
        "provider_totals_usd": {
            "runpod": 98.563962,
            "vast": 281.889,
            "digitalocean": 152.25,
        },
        "sources": [
            {
                "provider": "vast",
                "endpoint": VAST_CHARGES_URL,
                "request_query_digest": "sha256:" + "1" * 64,
                "response_digest": _sha256(path),
                "response_size_bytes": path.stat().st_size,
                "retained_path": str(path),
            }
            for path in responses
        ],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = _write(audit / "provider_billing_source_receipt.json", receipt)
    terminals = {
        INSTANCE_A: _terminal_fixture(tmp_path, instance_id=INSTANCE_A, status="blocked"),
        INSTANCE_B: _terminal_fixture(
            tmp_path, instance_id=INSTANCE_B, status="completed"
        ),
    }
    return {
        "audit": audit,
        "responses": responses,
        "receipt": receipt_path,
        "receipt_value": receipt,
        "terminals": terminals,
    }


def _paired_native_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    payload = json.loads(response.read_text(encoding="utf-8"))
    payload["results"].append(
        _charge(
            instance_id=PAIRED_INSTANCE,
            label=PAIRED_LABEL,
            total=0.086,
            gpu=0.081,
            disk=0.005,
        )
    )
    _write(response, payload)
    _refresh_response_binding(fixture, 0)
    terminals = fixture["terminals"]
    assert isinstance(terminals, dict)
    terminals[PAIRED_INSTANCE] = _terminal_fixture(
        tmp_path,
        instance_id=PAIRED_INSTANCE,
        status="completed",
        paired_native=True,
    )
    return fixture


def _content_agents_terminal_fixture(tmp_path: Path) -> Path:
    seed_result_path = _terminal_fixture(
        tmp_path,
        instance_id=CONTENT_AGENTS_INSTANCE,
        status="completed",
        run_id_override=CONTENT_AGENTS_RUN_ID,
        profile_id_override=(
            "adp-content-agents-live-scene840920-task-a-paired-registered-"
            "6bcc65db-r2"
        ),
    )
    run_root = seed_result_path.parents[2]
    allocator = run_root / "allocator"
    job = allocator / "content-agents-job"
    provider_run = job / "vast_provider_run"

    seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
    seed_adapter_path = Path(seed_result["adapter_result_path"])
    adapter = json.loads(seed_adapter_path.read_text(encoding="utf-8"))
    adapter["provider_bundle_kind"] = "adp_content_agents"
    adapter_path = _write(provider_run / "vast_provider_adapter_result.json", adapter)
    seed_teardown_path = Path(seed_result["teardown_manifest_path"])
    teardown = json.loads(seed_teardown_path.read_text(encoding="utf-8"))
    teardown_path = _write(provider_run / "vast_teardown_manifest.json", teardown)

    execution_result_path = _write(
        job / "immutable_execution" / "adp_content_agents_vast_result.json",
        {
            "schema_version": "adp_content_agents_vast_result.v1",
            "status": "completed",
            "retry_cap": 0,
            "material_agent_executed": True,
            "texture_agent_executed": True,
            "physics_agent_executed": True,
            "validation_agent_executed": True,
            "blockers": [],
            "raw_secret_values_recorded": False,
        },
    )
    prefix = "blueprint-adp-content-agents-20260817t113743803190000-"
    lane_zero = {
        "status": "observed",
        "provider": "vast",
        "name_prefix": prefix,
        "api_confirmed": True,
        "live_resource_count": 0,
        "resources": [],
    }
    global_zero = {**lane_zero, "name_prefix": ""}
    inspect = {
        "status": "absent",
        "provider": "vast",
        "instance_id": str(CONTENT_AGENTS_INSTANCE),
        "http": 200,
        "api_confirmed": True,
        "provider_absence_confirmed": True,
    }
    watchdog = {
        "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "provider_terminal",
        "provider": "vast",
        "pod_name_prefix": prefix,
        "provider_absence_confirmed": True,
        "owner_teardown_cancel_requested": True,
        "owner_teardown_cancel_request_valid": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "recorded_vast_instance_teardown": {
            "status": "absent",
            "instance_id": str(CONTENT_AGENTS_INSTANCE),
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 0,
            "inspect_attempts": [
                {**inspect, "attempt": 1},
                {**inspect, "attempt": 2},
            ],
        },
        "initial_inventory": lane_zero,
        "final_inventory": lane_zero,
        "initial_global_inventory": global_zero,
        "final_global_inventory": global_zero,
    }
    _write(
        job
        / "independent_vast_watchdog"
        / "groot_oscar_runpod_canary_watchdog.json",
        watchdog,
    )
    _write(
        job / "object_store_staging" / "wam_provider_object_store_cleanup.json",
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
            "raw_secret_values_recorded": False,
            "blockers": [],
        },
    )
    watchdog_handoff = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "status": "provider_terminal",
        "watchdog_armed_before_allocation": True,
        "instance_ids": [CONTENT_AGENTS_INSTANCE],
        "provider_absence_confirmed": True,
        "watchdog_process_exit_code": 0,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    _write(job / "vast_independent_watchdog_handoff.json", watchdog_handoff)

    bundle_digest = "sha256:" + "4" * 64
    empty_log_path = job / "immutable_execution" / "empty-provider-log.txt"
    empty_log_path.write_bytes(b"")
    file_rows = [
        {
            "relative_path": "immutable_execution/adp_content_agents_vast_result.json",
            "roles": ["provider_runtime_evidence"],
            "size_bytes": execution_result_path.stat().st_size,
            "sha256": _sha256(execution_result_path),
        },
        {
            "relative_path": "vast_provider_run/vast_provider_adapter_result.json",
            "roles": ["allocator_adapter_result", "provider_run_diagnostics"],
            "size_bytes": adapter_path.stat().st_size,
            "sha256": _sha256(adapter_path),
        },
        {
            "relative_path": "vast_provider_run/vast_teardown_manifest.json",
            "roles": ["provider_run_diagnostics", "teardown_manifest"],
            "size_bytes": teardown_path.stat().st_size,
            "sha256": _sha256(teardown_path),
        },
        {
            "relative_path": "immutable_execution/empty-provider-log.txt",
            "roles": ["provider_runtime_evidence"],
            "size_bytes": 0,
            "sha256": _sha256(empty_log_path),
        },
    ]
    artifact_manifest = {
        "schema_version": "task_evaluation_artifact_manifest.v1",
        "status": "completed",
        "binding": {
            "allocator_lane": "adp_content_agents",
            "retry_cap": 0,
            "bundle_sha256": bundle_digest,
            "provider": "vast",
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
        "file_count": len(file_rows),
        "total_size_bytes": sum(row["size_bytes"] for row in file_rows),
        "files": file_rows,
        "blockers": [],
        "raw_secret_values_recorded": False,
        "manifest_digest": "",
    }
    artifact_manifest["manifest_digest"] = canonical_digest(
        artifact_manifest, digest_field="manifest_digest"
    )
    artifact_manifest_path = _write(job / "artifact_manifest.json", artifact_manifest)

    result = {
        "schema_version": "adp_content_agents_vast_run.v1",
        "status": "completed",
        "bundle_sha256": bundle_digest,
        "execution_result_path": str(execution_result_path),
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "artifact_manifest_path": str(artifact_manifest_path),
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "independent_watchdog": watchdog_handoff,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    result_path = _write(job / "adp_content_agents_vast_result.json", result)
    allocator_result_path = _write(allocator / "result.json", result)

    receipt_path = run_root / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["terminal_evidence"]["result"] = {
        "path": str(allocator_result_path),
        "digest": _sha256(allocator_result_path),
        "exists": True,
    }
    receipt["terminal_evidence"]["artifacts"] = {
        "artifact_manifest_path": {
            "path": str(artifact_manifest_path),
            "digest": _sha256(artifact_manifest_path),
            "exists": True,
        },
        "teardown_manifest_path": {
            "path": str(teardown_path),
            "digest": _sha256(teardown_path),
            "exists": True,
        },
    }
    receipt["terminal_evidence"]["blockers"] = []
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(receipt_path, receipt)

    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    zero["receipt_digest"] = receipt["receipt_digest"]
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    _write(zero_path, zero)

    sync = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "attempt_number": 1,
        "launch_id": CONTENT_AGENTS_RUN_ID,
        "run_id": CONTENT_AGENTS_RUN_ID,
        "request_digest": json.loads(
            (run_root / "launch_request.json").read_text(encoding="utf-8")
        )["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "provider_mutation_performed": False,
        "response": {
            "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
            "status": "completed",
            "already_exists": False,
            "launch_id": CONTENT_AGENTS_RUN_ID,
            "run_id": CONTENT_AGENTS_RUN_ID,
            "request_digest": json.loads(
                (run_root / "launch_request.json").read_text(encoding="utf-8")
            )["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "sync_result_digest": "",
    }
    sync["sync_result_digest"] = canonical_digest(
        sync, digest_field="sync_result_digest"
    )
    _write(run_root / "webapp_sync_succeeded.json", sync)
    return result_path


ARENA_INSTANCE = 48_032_653
ARENA_LABEL = "blueprint-native-task-arena-20260818t150350615100000-1787065437"
ARENA_RUN_ID = "adp-arena-construction-fixture-48032653"


def _arena_terminal_fixture(tmp_path: Path) -> Path:
    """One Arena attempt in its real shape: attempt-scoped, provider-clean, lane-blocked.

    Modelled on the 2026-08-18 construction attempt. The provider side
    completed and tore down; the lane blocked on its own qualification. That
    combination is the normal shape of an attempt worth retrying, and it must
    reconcile or the retry can never be funded.
    """

    seed_result_path = _terminal_fixture(
        tmp_path,
        instance_id=ARENA_INSTANCE,
        status="blocked",
        run_id_override=ARENA_RUN_ID,
        profile_id_override="arena-construction-live-840920-task-a-fixture",
    )
    run_root = seed_result_path.parents[2]
    allocator = run_root / "allocator"
    job = allocator / "arena-construction-job"
    # the Arena transport seals under a numbered attempt
    attempt_root = job / "attempts" / "attempt_001"
    provider_run = attempt_root / "vast_provider_run"

    seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
    adapter = json.loads(
        Path(seed_result["adapter_result_path"]).read_text(encoding="utf-8")
    )
    adapter["provider_bundle_kind"] = "native_task_arena"
    # the provider attempt itself completed, unlike the lane verdict below
    adapter["status"] = "completed"
    adapter_path = _write(provider_run / "vast_provider_adapter_result.json", adapter)
    teardown = json.loads(
        Path(seed_result["teardown_manifest_path"]).read_text(encoding="utf-8")
    )
    teardown_path = _write(provider_run / "vast_teardown_manifest.json", teardown)

    watchdog_handoff = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "status": "provider_terminal",
        "watchdog_armed_before_allocation": True,
        "instance_ids": [ARENA_INSTANCE],
        "provider_absence_confirmed": True,
        "watchdog_process_exit_code": 0,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }

    result = {
        "schema_version": "native_task_arena_vast_run.v1",
        # the lane refused its own output; the GPU time was still spent
        "status": "blocked",
        "attempt_number": 1,
        "attempt_root": str(attempt_root),
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "independent_watchdog": watchdog_handoff,
        "blockers": ["native_task_construction_failed_at_dependencies_qualified"],
        "raw_secret_values_recorded": False,
    }
    result_path = _write(job / "adp_arena_vast_result.json", result)
    allocator_result_path = _write(allocator / "result.json", result)

    receipt_path = run_root / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["terminal_evidence"]["result"] = {
        "path": str(allocator_result_path),
        "digest": _sha256(allocator_result_path),
        "exists": True,
    }
    receipt["terminal_evidence"]["artifacts"] = {
        "teardown_manifest_path": {
            "path": str(teardown_path),
            "digest": _sha256(teardown_path),
            "exists": True,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(receipt_path, receipt)

    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    zero["receipt_digest"] = receipt["receipt_digest"]
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    _write(zero_path, zero)
    return result_path


def _arena_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    payload = json.loads(response.read_text(encoding="utf-8"))
    payload["results"].append(
        _charge(
            instance_id=ARENA_INSTANCE,
            label=ARENA_LABEL,
            total=0.072,
            gpu=0.068,
            disk=0.004,
        )
    )
    _write(response, payload)
    _refresh_response_binding(fixture, 0)
    terminals = fixture["terminals"]
    assert isinstance(terminals, dict)
    terminals[ARENA_INSTANCE] = _arena_terminal_fixture(tmp_path)
    return fixture


def _content_agents_fixture(tmp_path: Path) -> dict[str, object]:
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    payload = json.loads(response.read_text(encoding="utf-8"))
    payload["results"].append(
        _charge(
            instance_id=CONTENT_AGENTS_INSTANCE,
            label=CONTENT_AGENTS_LABEL,
            total=0.329,
            gpu=0.285,
            disk=0.005,
            bandwidth_download=0.036,
            bandwidth_upload=0.003,
        )
    )
    _write(response, payload)
    _refresh_response_binding(fixture, 0)
    terminals = fixture["terminals"]
    assert isinstance(terminals, dict)
    terminals[CONTENT_AGENTS_INSTANCE] = _content_agents_terminal_fixture(tmp_path)
    return fixture


def _refresh_content_agents_launch_bindings(result_path: Path) -> None:
    run_root = result_path.parents[2]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    allocator_result_path = _write(run_root / "allocator" / "result.json", result)
    receipt_path = run_root / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["terminal_evidence"]["result"]["digest"] = _sha256(
        allocator_result_path
    )
    for field, result_field in (
        ("teardown_manifest_path", "teardown_manifest_path"),
        ("artifact_manifest_path", "artifact_manifest_path"),
    ):
        path = Path(result[result_field])
        receipt["terminal_evidence"]["artifacts"][field] = {
            "path": str(path),
            "digest": _sha256(path),
            "exists": True,
        }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(receipt_path, receipt)
    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    zero["receipt_digest"] = receipt["receipt_digest"]
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    _write(zero_path, zero)
    sync_path = run_root / "webapp_sync_succeeded.json"
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    sync["receipt_digest"] = receipt["receipt_digest"]
    sync["response"]["receipt_digest"] = receipt["receipt_digest"]
    sync["sync_result_digest"] = canonical_digest(
        sync, digest_field="sync_result_digest"
    )
    _write(sync_path, sync)


def _refresh_response_binding(fixture: dict[str, object], index: int) -> None:
    path = fixture["responses"][index]
    assert isinstance(path, Path)
    receipt_path = fixture["receipt"]
    assert isinstance(receipt_path, Path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sources"][index]["response_digest"] = _sha256(path)
    receipt["sources"][index]["response_size_bytes"] = path.stat().st_size
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(receipt_path, receipt)


def _materialize(
    fixture: dict[str, object],
    output: Path,
    *,
    expected: list[tuple[int, str, Path]] | None = None,
    prior: Path | None = None,
) -> dict:
    return materialize_vast_official_same_goal_reconciliation(
        provider_billing_source_receipt_path=fixture["receipt"],
        expected_instances=expected
        or [
            _spec(fixture, INSTANCE_A, LABEL_A),
            _spec(fixture, INSTANCE_B, LABEL_B),
        ],
        output_path=output,
        prior_reconciliation_path=prior,
    )


def _spec(
    fixture: dict[str, object], instance_id: int, label: str
) -> tuple[int, str, Path]:
    terminals = fixture["terminals"]
    assert isinstance(terminals, dict)
    terminal = terminals[instance_id]
    assert isinstance(terminal, Path)
    return instance_id, label, terminal


def test_extracts_production_shaped_posted_instance_charges_exactly(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "official" / "same-goal.json"
    value = _materialize(fixture, output)

    assert value["schema_version"] == RECONCILIATION_SCHEMA_VERSION
    assert value["entry_count"] == 2
    assert value["provider_instance_ids"] == [INSTANCE_A, INSTANCE_B]
    assert value["official_total_usd"] == pytest.approx(2.306)
    assert value["provider_mutation_performed"] is False
    assert value["paid_resource_allocated"] is False
    assert value["raw_secret_values_recorded"] is False
    assert value["receipt_digest"] == canonical_digest(
        value, digest_field="receipt_digest"
    )
    assert value["current_provider_billing_source_receipt"]["receipt_digest"] == (
        fixture["receipt_value"]["receipt_digest"]
    )
    first, second = value["entries"]
    assert first["schema_version"] == ENTRY_SCHEMA_VERSION
    assert first["provider_instance_id"] == INSTANCE_A
    assert first["official_charge_usd"] == pytest.approx(0.123)
    assert first["official_line_items_usd"] == {
        "gpu": 0.112,
        "disk": 0.011,
        "bandwidth_download": 0.0,
        "bandwidth_upload": 0.0,
    }
    assert second["provider_instance_id"] == INSTANCE_B
    assert second["official_charge_usd"] == pytest.approx(2.183)
    assert second["official_line_items_usd"] == {
        "gpu": 2.056,
        "disk": 0.127,
        "bandwidth_download": 0.0,
        "bandwidth_upload": 0.0,
    }
    assert first["entry_digest"] == canonical_digest(
        first, digest_field="entry_digest"
    )
    first_terminal = first["terminal_execution_evidence"]
    second_terminal = second["terminal_execution_evidence"]
    assert first_terminal["terminal_status"] == "blocked"
    assert second_terminal["terminal_status"] == "completed"
    for terminal, instance_id in (
        (first_terminal, INSTANCE_A),
        (second_terminal, INSTANCE_B),
    ):
        assert terminal["retry_cap"] == 0
        assert terminal["continuing_spend_from_this_run"] is False
        assert terminal["provider_absence_confirmed"] is True
        assert terminal["provider_zero_verified"] is True
        assert terminal["launch_id"] == terminal["run_id"]
        assert terminal["request_digest"].startswith("sha256:")
        assert terminal["profile_id"].startswith("adp-artifixer3d-live-")
        assert terminal["profile_digest"].startswith("sha256:")
        assert terminal["provider_adapter_result"]["status"] in {
            "blocked",
            "completed",
        }
        assert terminal["teardown_manifest"]["status"] == "completed"
        assert str(instance_id) in terminal["terminal_result"]["path"]
        terminal_path = Path(terminal["terminal_result"]["path"])
        assert terminal_path.name == "public_scene_artifixer3d_vast_result.json"
        assert terminal_path.parent.name == "artifixer3d-job"
        assert terminal_path.parent.parent.name == "allocator"
    assert validate_vast_official_same_goal_reconciliation(output) == value
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    serialized = output.read_text(encoding="utf-8")
    assert "raw-response-secret" not in serialized
    assert "production-shaped GPU line" not in serialized


def test_accepts_exact_paired_native_terminal_layout_and_official_charge(
    tmp_path: Path,
) -> None:
    fixture = _paired_native_fixture(tmp_path)
    output = tmp_path / "paired-native-official.json"
    terminal_path = _spec(fixture, PAIRED_INSTANCE, PAIRED_LABEL)[2]

    value = _materialize(
        fixture,
        output,
        expected=[(PAIRED_INSTANCE, PAIRED_LABEL, terminal_path)],
    )

    assert value["provider_instance_ids"] == [PAIRED_INSTANCE]
    assert value["launch_labels"] == [PAIRED_LABEL]
    assert value["official_total_usd"] == pytest.approx(0.086)
    terminal = value["entries"][0]["terminal_execution_evidence"]
    assert terminal["launch_id"] == PAIRED_RUN_ID
    assert terminal["run_id"] == PAIRED_RUN_ID
    assert terminal["profile_id"] == (
        "adp-paired-target-native-import-live-846bce86-r3"
    )
    assert terminal["terminal_result"]["schema_version"] == (
        "paired_target_native_import_vast_run.v1"
    )
    assert terminal["provider_adapter_result"]["status"] == "completed"
    assert terminal["teardown_manifest"]["status"] == "completed"
    assert terminal["independent_watchdog"]["status"] == "provider_terminal"
    assert terminal["object_store_cleanup"]["status"] == "completed"
    assert terminal["post_teardown_provider_zero"]["status"] == (
        "provider_zero_confirmed"
    )
    assert validate_vast_official_same_goal_reconciliation(output) == value


def test_accepts_exact_content_agents_terminal_layout_and_official_charge(
    tmp_path: Path,
) -> None:
    fixture = _content_agents_fixture(tmp_path)
    output = tmp_path / "content-agents-official.json"
    terminal_path = _spec(
        fixture, CONTENT_AGENTS_INSTANCE, CONTENT_AGENTS_LABEL
    )[2]

    value = _materialize(
        fixture,
        output,
        expected=[
            (CONTENT_AGENTS_INSTANCE, CONTENT_AGENTS_LABEL, terminal_path)
        ],
    )

    assert value["provider_instance_ids"] == [CONTENT_AGENTS_INSTANCE]
    assert value["launch_labels"] == [CONTENT_AGENTS_LABEL]
    assert value["official_total_usd"] == pytest.approx(0.329)
    terminal = value["entries"][0]["terminal_execution_evidence"]
    assert terminal["launch_id"] == CONTENT_AGENTS_RUN_ID
    assert terminal["run_id"] == CONTENT_AGENTS_RUN_ID
    assert terminal["terminal_result"]["schema_version"] == (
        "adp_content_agents_vast_run.v1"
    )
    assert terminal["provider_adapter_result"]["status"] == "completed"
    assert terminal["teardown_manifest"]["status"] == "completed"
    assert terminal["independent_watchdog"]["status"] == "provider_terminal"
    assert terminal["object_store_cleanup"]["status"] == "completed"
    assert terminal["artifact_manifest"]["status"] == "completed"
    assert terminal["provider_runtime_result"]["status"] == "completed"
    assert terminal["webapp_terminal_binding"]["status"] == "succeeded"
    assert terminal["post_teardown_provider_zero"]["status"] == (
        "provider_zero_confirmed"
    )
    assert validate_vast_official_same_goal_reconciliation(output) == value


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("result_path", "vast_official_artifact_manifest_invalid"),
        ("result_schema", "vast_official_terminal_result_invalid"),
        ("watchdog_tamper", "vast_official_content_agents_terminal_closure_invalid"),
        ("artifact_tamper", "vast_official_artifact_manifest_invalid"),
        ("webapp_tamper", "vast_official_webapp_terminal_binding_invalid"),
    ],
)
def test_rejects_hostile_content_agents_path_schema_and_tamper(
    tmp_path: Path, mutation: str, blocker: str
) -> None:
    fixture = _content_agents_fixture(tmp_path)
    terminal_path = _spec(
        fixture, CONTENT_AGENTS_INSTANCE, CONTENT_AGENTS_LABEL
    )[2]
    run_root = terminal_path.parents[2]
    result = json.loads(terminal_path.read_text(encoding="utf-8"))
    if mutation == "result_path":
        hostile = terminal_path.parent / "hostile-artifact-manifest.json"
        hostile.write_bytes(Path(result["artifact_manifest_path"]).read_bytes())
        result["artifact_manifest_path"] = str(hostile)
        _write(terminal_path, result)
        _refresh_content_agents_launch_bindings(terminal_path)
    elif mutation == "result_schema":
        result["schema_version"] = "adp_content_agents_vast_run.v2"
        _write(terminal_path, result)
        _refresh_content_agents_launch_bindings(terminal_path)
    elif mutation == "watchdog_tamper":
        watchdog_path = (
            terminal_path.parent
            / "independent_vast_watchdog"
            / "groot_oscar_runpod_canary_watchdog.json"
        )
        watchdog = json.loads(watchdog_path.read_text(encoding="utf-8"))
        watchdog["provider_absence_confirmed"] = False
        _write(watchdog_path, watchdog)
    elif mutation == "artifact_tamper":
        artifact_path = Path(result["artifact_manifest_path"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["binding"]["bundle_sha256"] = "sha256:" + "f" * 64
        artifact["manifest_digest"] = canonical_digest(
            artifact, digest_field="manifest_digest"
        )
        _write(artifact_path, artifact)
        _refresh_content_agents_launch_bindings(terminal_path)
    else:
        sync_path = run_root / "webapp_sync_succeeded.json"
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        sync["response"]["run_id"] = "hostile-run-id"
        sync["sync_result_digest"] = canonical_digest(
            sync, digest_field="sync_result_digest"
        )
        _write(sync_path, sync)

    with pytest.raises(VastOfficialBillingExtractionError, match=blocker):
        _materialize(
            fixture,
            tmp_path / "hostile-content-output.json",
            expected=[
                (CONTENT_AGENTS_INSTANCE, CONTENT_AGENTS_LABEL, terminal_path)
            ],
        )


@pytest.mark.parametrize("mutation", ["path", "schema", "launch_digest"])
def test_rejects_hostile_paired_native_path_schema_and_digest(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _paired_native_fixture(tmp_path)
    terminal_path = _spec(fixture, PAIRED_INSTANCE, PAIRED_LABEL)[2]
    run_root = terminal_path.parents[2]
    if mutation == "launch_digest":
        request_path = run_root / "launch_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["launch_profile_id"] = "hostile-profile"
        _write(request_path, request)
    else:
        result = json.loads(terminal_path.read_text(encoding="utf-8"))
        if mutation == "path":
            result["object_store_cleanup_path"] = str(
                run_root / "allocator" / "hostile-cleanup.json"
            )
        else:
            result["schema_version"] = "paired_target_native_import_vast_run.v2"
        _write(terminal_path, result)
        _refresh_terminal_bindings(terminal_path)

    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(
            fixture,
            tmp_path / "hostile-output.json",
            expected=[(PAIRED_INSTANCE, PAIRED_LABEL, terminal_path)],
        )


def test_rejects_paired_native_watchdog_digest_drift_after_sealing(
    tmp_path: Path,
) -> None:
    fixture = _paired_native_fixture(tmp_path)
    terminal_path = _spec(fixture, PAIRED_INSTANCE, PAIRED_LABEL)[2]
    output = tmp_path / "paired-native-official.json"
    _materialize(
        fixture,
        output,
        expected=[(PAIRED_INSTANCE, PAIRED_LABEL, terminal_path)],
    )
    result = json.loads(terminal_path.read_text(encoding="utf-8"))
    watchdog_path = Path(result["watchdog_receipt_path"])
    watchdog = json.loads(watchdog_path.read_text(encoding="utf-8"))
    watchdog["completed_at"] = "2026-08-17T09:43:41+00:00"
    _write(watchdog_path, watchdog)

    with pytest.raises(VastOfficialBillingExtractionError):
        validate_vast_official_same_goal_reconciliation(output)


def test_prior_reconciliation_extends_without_repricing_prior_entry(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prior_path = tmp_path / "prior.json"
    prior = _materialize(
        fixture, prior_path, expected=[_spec(fixture, INSTANCE_A, LABEL_A)]
    )
    output = tmp_path / "extended.json"
    value = _materialize(
        fixture,
        output,
        expected=[_spec(fixture, INSTANCE_B, LABEL_B)],
        prior=prior_path,
    )

    assert value["entry_count"] == 2
    assert value["new_entry_count"] == 1
    assert value["prior_entry_count"] == 1
    assert value["official_total_usd"] == pytest.approx(2.306)
    assert value["entries"][0] == prior["entries"][0]
    assert value["predecessor_reconciliation"]["receipt_digest"] == prior[
        "receipt_digest"
    ]


def test_rejects_tampered_or_overlapping_prior_reconciliation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prior_path = tmp_path / "prior.json"
    _materialize(
        fixture, prior_path, expected=[_spec(fixture, INSTANCE_A, LABEL_A)]
    )
    prior_path.chmod(0o600)
    tampered = json.loads(prior_path.read_text(encoding="utf-8"))
    tampered["official_total_usd"] = 0.0
    _write(prior_path, tampered)
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(
            fixture,
            tmp_path / "tampered-output.json",
            expected=[_spec(fixture, INSTANCE_B, LABEL_B)],
            prior=prior_path,
        )

    prior_path.unlink()
    _materialize(
        fixture, prior_path, expected=[_spec(fixture, INSTANCE_A, LABEL_A)]
    )
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_prior_overlap"
    ):
        _materialize(
            fixture,
            tmp_path / "overlap-output.json",
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
            prior=prior_path,
        )


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("duplicate", "vast_official_charge_duplicate"),
        ("wrong_label", "vast_official_charge_identity_invalid"),
        ("non_instance", "vast_official_charge_identity_invalid"),
        ("negative_amount", "vast_official_charge_amount_invalid"),
        ("nonfinite_amount", "vast_official_charge_amount_invalid"),
        ("negative_item", "vast_official_charge_item_amount_invalid"),
        ("nonfinite_item", "vast_official_charge_item_amount_invalid"),
        ("missing_item", "vast_official_charge_items_invalid"),
        ("duplicate_item", "vast_official_charge_items_invalid"),
        ("contradictory_total", "vast_official_charge_total_contradiction"),
    ],
)
def test_rejects_ambiguous_or_invalid_official_rows(
    tmp_path: Path, mutation: str, blocker: str
) -> None:
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    payload = json.loads(response.read_text(encoding="utf-8"))
    row = payload["results"][0]
    if mutation == "duplicate":
        payload["results"].append(dict(row))
    elif mutation == "wrong_label":
        row["metadata"]["label"] = "blueprint-wrong-label"
    elif mutation == "non_instance":
        row["type"] = "storage"
    elif mutation == "negative_amount":
        row["amount"] = -0.123
    elif mutation == "nonfinite_amount":
        row["amount"] = math.inf
    elif mutation == "negative_item":
        row["items"][0]["amount"] = -0.112
    elif mutation == "nonfinite_item":
        row["items"][0]["amount"] = math.inf
    elif mutation == "missing_item":
        row["items"].pop()
    elif mutation == "duplicate_item":
        row["items"][-1]["type"] = "gpu"
    elif mutation == "contradictory_total":
        row["items"][0]["amount"] = 0.111
    _write(response, payload)
    _refresh_response_binding(fixture, 0)

    with pytest.raises(VastOfficialBillingExtractionError, match=blocker):
        _materialize(
            fixture,
            tmp_path / "output.json",
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unsupported_terminal_schema",
        "retry_cap",
        "continuing_spend",
        "adapter_instance",
        "teardown_incomplete",
        "provider_zero_false",
        "launch_identity",
        "wrong_result_path",
        "terminal_symlink",
        "alternate_depth",
    ],
)
def test_rejects_unbound_or_incomplete_terminal_execution_evidence(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    result_path = _spec(fixture, INSTANCE_A, LABEL_A)[2]
    expected = [_spec(fixture, INSTANCE_A, LABEL_A)]
    run_root = result_path.parents[2]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "unsupported_terminal_schema":
        result["schema_version"] = "unknown_vast_run.v1"
        _write(result_path, result)
        _refresh_terminal_bindings(result_path)
    elif mutation == "retry_cap":
        result["retry_cap"] = 1
        _write(result_path, result)
        _refresh_terminal_bindings(result_path)
    elif mutation == "continuing_spend":
        result["continuing_spend_from_this_run"] = True
        _write(result_path, result)
        _refresh_terminal_bindings(result_path)
    elif mutation == "adapter_instance":
        adapter_path = Path(result["adapter_result_path"])
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter["vast_instance_ids"] = [INSTANCE_B]
        _write(adapter_path, adapter)
        _refresh_terminal_bindings(result_path)
    elif mutation == "teardown_incomplete":
        teardown_path = Path(result["teardown_manifest_path"])
        teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
        teardown["runner_gpu_teardown_completed"] = False
        _write(teardown_path, teardown)
        _refresh_terminal_bindings(result_path)
    elif mutation == "provider_zero_false":
        zero_path = run_root / "post_teardown_provider_zero_receipt.json"
        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        zero["provider_zero_verified"] = False
        zero["provider_zero_receipt_digest"] = canonical_digest(
            zero, digest_field="provider_zero_receipt_digest"
        )
        _write(zero_path, zero)
    elif mutation == "launch_identity":
        request_path = run_root / "launch_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["run_id"] = "different-run"
        request["request_digest"] = canonical_digest(
            request, digest_field="request_digest"
        )
        _write(request_path, request)
    elif mutation == "wrong_result_path":
        expected = [
            (INSTANCE_A, LABEL_A, _spec(fixture, INSTANCE_B, LABEL_B)[2])
        ]
    elif mutation == "terminal_symlink":
        real_result = result_path.with_suffix(".real.json")
        result_path.rename(real_result)
        result_path.symlink_to(real_result)
    elif mutation == "alternate_depth":
        alternate = run_root / "allocator" / result_path.name
        alternate.write_bytes(result_path.read_bytes())
        expected = [(INSTANCE_A, LABEL_A, alternate)]
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(fixture, tmp_path / "output.json", expected=expected)


@pytest.mark.parametrize("mutation", ["digest", "size", "path", "receipt_digest"])
def test_rejects_response_or_source_receipt_binding_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    receipt_path = fixture["receipt"]
    assert isinstance(receipt_path, Path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "digest":
        receipt["sources"][0]["response_digest"] = "sha256:" + "0" * 64
    elif mutation == "size":
        receipt["sources"][0]["response_size_bytes"] += 1
    elif mutation == "path":
        receipt["sources"][0]["retained_path"] = receipt["sources"][1][
            "retained_path"
        ]
    elif mutation == "receipt_digest":
        receipt["provider_totals_usd"]["vast"] += 1
    if mutation != "receipt_digest":
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
    _write(receipt_path, receipt)

    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(fixture, tmp_path / "output.json")


def test_rejects_symlinked_source_receipt_and_response(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "receipt-case")
    receipt = fixture["receipt"]
    assert isinstance(receipt, Path)
    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(receipt)
    with pytest.raises(VastOfficialBillingExtractionError):
        materialize_vast_official_same_goal_reconciliation(
            provider_billing_source_receipt_path=receipt_link,
            expected_instances=[_spec(fixture, INSTANCE_A, LABEL_A)],
            output_path=tmp_path / "receipt-output.json",
        )

    fixture = _fixture(tmp_path / "response-case")
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    real = response.with_suffix(".real.json")
    response.rename(real)
    response.symlink_to(real)
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(
            fixture,
            tmp_path / "response-output.json",
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
        )


def test_rejects_unposted_duplicate_expectation_and_existing_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_charge_unposted"
    ):
        _materialize(
            fixture,
            tmp_path / "missing.json",
            expected=[
                (
                    99_999_999,
                    "blueprint-missing-instance",
                    _spec(fixture, INSTANCE_A, LABEL_A)[2],
                )
            ],
        )
    with pytest.raises(
        VastOfficialBillingExtractionError,
        match="vast_official_expected_instances_duplicate",
    ):
        _materialize(
            fixture,
            tmp_path / "duplicate.json",
            expected=[
                _spec(fixture, INSTANCE_A, LABEL_A),
                (INSTANCE_A, LABEL_B, _spec(fixture, INSTANCE_B, LABEL_B)[2]),
            ],
        )
    output = tmp_path / "exists.json"
    output.write_text("user-owned", encoding="utf-8")
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_output_invalid"
    ):
        _materialize(
            fixture,
            output,
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
        )
    assert output.read_text(encoding="utf-8") == "user-owned"


def test_cli_materializes_without_provider_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "cli.json"
    assert (
        main(
            [
                "--provider-billing-source-receipt",
                str(fixture["receipt"]),
                "--expected-instance",
                f"{INSTANCE_A}={LABEL_A}={_spec(fixture, INSTANCE_A, LABEL_A)[2]}",
                "--expected-instance",
                f"{INSTANCE_B}={LABEL_B}={_spec(fixture, INSTANCE_B, LABEL_B)[2]}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "materialized"
    assert summary["official_total_usd"] == pytest.approx(2.306)
    assert summary["provider_mutation_performed"] is False


def test_cli_accepts_production_paired_native_result_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _paired_native_fixture(tmp_path)
    output = tmp_path / "paired-cli.json"
    terminal_path = _spec(fixture, PAIRED_INSTANCE, PAIRED_LABEL)[2]

    assert (
        main(
            [
                "--provider-billing-source-receipt",
                str(fixture["receipt"]),
                "--expected-instance",
                f"{PAIRED_INSTANCE}={PAIRED_LABEL}={terminal_path}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "materialized"
    assert summary["entry_count"] == 1
    assert summary["official_total_usd"] == pytest.approx(0.086)
    assert summary["provider_mutation_performed"] is False


def test_script_accepts_production_content_agents_result_path(
    tmp_path: Path,
) -> None:
    fixture = _content_agents_fixture(tmp_path)
    output = tmp_path / "content-agents-cli.json"
    terminal_path = _spec(
        fixture, CONTENT_AGENTS_INSTANCE, CONTENT_AGENTS_LABEL
    )[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--provider-billing-source-receipt",
            str(fixture["receipt"]),
            "--expected-instance",
            f"{CONTENT_AGENTS_INSTANCE}={CONTENT_AGENTS_LABEL}={terminal_path}",
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "materialized"
    assert summary["entry_count"] == 1
    assert summary["official_total_usd"] == pytest.approx(0.329)
    assert summary["provider_mutation_performed"] is False
    assert validate_vast_official_same_goal_reconciliation(output)[
        "receipt_digest"
    ].startswith("sha256:")


def test_script_entrypoint_is_reachable_without_provider_access() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "--provider-billing-source-receipt" in completed.stdout
    assert "--expected-instance" in completed.stdout


def test_accepts_an_attempt_scoped_arena_layout_whose_lane_blocked(
    tmp_path: Path,
) -> None:
    """The Arena lane reconciles an attempt its own verdict rejected.

    Chaining the next Arena authority requires the previous attempt's official
    posted charges. The 2026-08-18 construction attempt completed on the
    provider, tore down, confirmed provider zero, and posted $0.072 -- and the
    lane blocked on its own qualification. If that shape cannot reconcile, the
    attempts most in need of a retry are exactly the ones that can never fund
    one.
    """

    fixture = _arena_fixture(tmp_path)
    output = tmp_path / "arena-official.json"
    terminal_path = _spec(fixture, ARENA_INSTANCE, ARENA_LABEL)[2]

    value = _materialize(
        fixture,
        output,
        expected=[(ARENA_INSTANCE, ARENA_LABEL, terminal_path)],
    )

    assert value["provider_instance_ids"] == [ARENA_INSTANCE]
    assert value["official_total_usd"] == pytest.approx(0.072)
    terminal = value["entries"][0]["terminal_execution_evidence"]
    assert terminal["terminal_result"]["schema_version"] == (
        "native_task_arena_vast_run.v1"
    )
    # the provider completed while the lane blocked, and both are recorded
    assert terminal["terminal_result"]["status"] == "blocked"
    assert terminal["provider_adapter_result"]["status"] == "completed"
    assert validate_vast_official_same_goal_reconciliation(output) == value


def test_rejects_an_arena_result_naming_closure_outside_its_own_attempt(
    tmp_path: Path,
) -> None:
    """Attempt-scoped discovery must not let a result point anywhere it likes."""

    fixture = _arena_fixture(tmp_path)
    terminal_path = _spec(fixture, ARENA_INSTANCE, ARENA_LABEL)[2]
    result = json.loads(terminal_path.read_text(encoding="utf-8"))
    job = terminal_path.parent

    # a sibling attempt's adapter is not this attempt's evidence
    stray = job / "attempts" / "attempt_002" / "vast_provider_run"
    stray_adapter = _write(
        stray / "vast_provider_adapter_result.json",
        json.loads(Path(result["adapter_result_path"]).read_text(encoding="utf-8")),
    )
    result["adapter_result_path"] = str(stray_adapter)
    _write(terminal_path, result)
    _write(job.parent / "result.json", result)

    with pytest.raises(VastOfficialBillingExtractionError) as excinfo:
        _materialize(
            fixture,
            tmp_path / "arena-stray.json",
            expected=[(ARENA_INSTANCE, ARENA_LABEL, terminal_path)],
        )

    assert "adapter_result_invalid" in str(excinfo.value)


def test_rejects_an_arena_attempt_root_outside_its_job_directory(
    tmp_path: Path,
) -> None:
    """``attempt_root`` is pinned to this job, not merely to some attempts dir."""

    fixture = _arena_fixture(tmp_path)
    terminal_path = _spec(fixture, ARENA_INSTANCE, ARENA_LABEL)[2]
    result = json.loads(terminal_path.read_text(encoding="utf-8"))
    job = terminal_path.parent
    result["attempt_root"] = str(
        job.parent / "arena-policy-job" / "attempts" / "attempt_001"
    )
    _write(terminal_path, result)
    _write(job.parent / "result.json", result)

    with pytest.raises(VastOfficialBillingExtractionError) as excinfo:
        _materialize(
            fixture,
            tmp_path / "arena-wrong-job.json",
            expected=[(ARENA_INSTANCE, ARENA_LABEL, terminal_path)],
        )

    assert "terminal_result_invalid" in str(excinfo.value)
