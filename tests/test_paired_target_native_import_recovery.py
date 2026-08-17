from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_import_recovery import (
    materialize_paired_target_native_import_recovered_provider_zero,
    validate_paired_target_native_import_recovered_provider_zero,
)


INSTANCE_ID = 47_925_871
MACHINE_ID = 140_718
OFFER_ID = 41_254_717
LABEL = "blueprint-adp-paired-native-import-1786951229"
NOW = datetime(2026, 8, 17, 7, 37, tzinfo=timezone.utc)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPOSITORY_ROOT / "scripts/seal_paired_target_native_import_recovered_zero.py"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _self(path: Path, value: dict, field: str) -> Path:
    value[field] = canonical_digest(value, digest_field=field)
    write_json(path, value)
    return path


def _fixture(
    tmp_path: Path,
    *,
    source_request_digest: str = "sha256:" + "4" * 64,
) -> dict[str, Path]:
    launch_id = "adp-paired-native-840920-d6506694-r2-api-20260817T072001Z-09b46828"
    run = tmp_path / "task-evaluation-launch-runs" / launch_id
    job = run / "allocator" / "paired-target-native-import-job"
    provider = job / "vast_provider_run"
    watchdog_root = job / "independent_vast_watchdog"
    request_digest = "sha256:" + "1" * 64
    bundle_sha256 = "sha256:" + "2" * 64
    bundle_receipt = {
        "schema_version": "paired_target_native_import_bundle_receipt.v1",
        "receipt_digest": "sha256:" + "3" * 64,
        "bundle_sha256": bundle_sha256,
        "request_digest": request_digest,
    }
    bundle_receipt_path = tmp_path / "bundle-receipt.json"
    write_json(bundle_receipt_path, bundle_receipt)
    authority = {
        "schema_version": "paired_target_native_import_paid_attempt_authority.v1",
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "bundle_receipt": _record(bundle_receipt_path),
        "bundle_receipt_digest": bundle_receipt["receipt_digest"],
        "bundle_sha256": bundle_sha256,
        "source_request_digest": source_request_digest,
        "hard_attempt_spend_cap_usd": 2.0,
        "maximum_single_resource_ttl_seconds": 6000,
        "aggregate_goal_spend_before_attempt_usd": 2.318914,
        "prior_paired_attempts": [],
        "paired_attempt_ordinal": 1,
        "authorization_digest": "",
    }
    authority_path = _self(
        tmp_path / "authority.json", authority, "authorization_digest"
    )

    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "adp-paired-target-native-import-live-d6506694-r1",
        "profile_digest": "",
    }
    profile_path = _self(run / "launch_profile.json", profile, "profile_digest")
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": launch_id,
        "run_id": launch_id,
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "request_digest": "",
    }
    request_path = _self(run / "launch_request.json", request, "request_digest")
    binding = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "profile_digest": profile["profile_digest"],
        "binding_digest": "",
    }
    binding_path = _self(run / "launch_binding.json", binding, "binding_digest")
    started = {
        "schema_version": "task_evaluation_launch_started.v1",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "binding_digest": binding["binding_digest"],
        "automatic_retry_authorized": False,
        "started_digest": "",
    }
    started_path = _self(run / "launch_started.json", started, "started_digest")
    launch_receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "blocked",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "binding_digest": binding["binding_digest"],
        "receipt_digest": "",
    }
    launch_receipt_path = _self(
        run / "launch_receipt.json", launch_receipt, "receipt_digest"
    )
    webapp = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": request["request_digest"],
        "receipt_digest": launch_receipt["receipt_digest"],
        "provider_mutation_performed": False,
        "sync_result_digest": "",
    }
    webapp_path = _self(
        run / "webapp_sync_succeeded.json", webapp, "sync_result_digest"
    )

    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": "failed",
        "provider_bundle_kind": "paired_target_native_import",
        "provider_create_attempted": True,
        "continuing_spend_from_this_run": True,
        "estimated_cost_usd": 0.058517,
        "vast_instance_ids": [INSTANCE_ID],
        "excluded_machine_ids": [MACHINE_ID],
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
        "provider_attempt_classification": {
            "classification": "pre_execution_provider_null",
            "provider_bundle_started": False,
            "provider_entrypoint_started": False,
            "provider_output_returned": False,
            "scientific_attempt_consumed": False,
            "automatic_requeue_executed": False,
            "maximum_automatic_requeues": 0,
        },
    }
    adapter_path = job / "vast_provider_run" / "vast_provider_adapter_result.json"
    write_json(adapter_path, adapter)
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "blocked",
        "vast_instance_ids": [INSTANCE_ID],
        "continuing_spend_from_this_run": True,
        "teardown_actions_performed": [
            {
                "instance_id": INSTANCE_ID,
                "action": "destroy_instance",
                "status": "failed",
                "http_status_code": 429,
                "error": "HTTPError: HTTP Error 429: Too Many Requests",
            }
        ],
    }
    teardown_path = provider / "vast_teardown_manifest.json"
    write_json(teardown_path, teardown)
    cleanup = {
        "schema_version": "wam_provider_object_store_cleanup.v1",
        "status": "completed",
        "all_objects_absent": True,
        "signed_url_files_removed": True,
        "blockers": [],
    }
    cleanup_path = job / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    write_json(cleanup_path, cleanup)
    session = {
        "schema_version": "vast_session_cost_summary.v4",
        "status": "completed",
        "attempt_count": 1,
        "attempts": [
            {
                "vast_instance_ids": [INSTANCE_ID],
                "estimated_cost_usd": 0.058517,
                "continuing_spend_from_this_run": True,
                "blockers": ["vast_heartbeat_container_missing"],
                "machine_id": MACHINE_ID,
                "offer_id": OFFER_ID,
            }
        ],
    }
    session_path = job / "paired_target_native_import_session_budget.json"
    write_json(session_path, session)
    avoidlist = {
        "schema_version": "vast_machine_avoidlist.v1",
        "status": "completed",
        "machine_ids": [MACHINE_ID],
        "entries": [
            {
                "machine_id": MACHINE_ID,
                "offer_id": OFFER_ID,
                "instance_id": INSTANCE_ID,
                "reason": "vast_startup_control_plane_did_not_reach_onstart_heartbeat",
                "blockers": ["vast_heartbeat_container_missing"],
            }
        ],
        "raw_secret_values_recorded": False,
    }
    avoidlist_path = job / "vast_machine_avoidlist.json"
    write_json(avoidlist_path, avoidlist)

    prefix = "blueprint-adp-paired-native-import-20260817t072028288615000-"
    deadline = 1_786_957_228.2893279
    armed = {
        "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "armed",
        "provider": "vast",
        "pid": 1_096_161,
        "deadline_epoch": deadline,
        "pod_name_prefix": prefix,
        "allowed_active_instance_ids": [],
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    armed_path = watchdog_root / "groot_oscar_runpod_canary_watchdog.original-armed.json"
    write_json(armed_path, armed)
    handoff = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "status": "retained_until_hard_ttl",
        "watchdog_armed_before_allocation": True,
        "instance_ids": [INSTANCE_ID],
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    handoff_path = job / "vast_independent_watchdog_handoff.json"
    write_json(handoff_path, handoff)
    cancel = {
        "schema_version": "groot_oscar_runpod_canary_watchdog_cancel.v1",
        "provider": "vast",
        "instance_id": str(INSTANCE_ID),
        "pod_name_prefix": prefix,
        "provider_absence_confirmed": True,
        "raw_secret_values_recorded": False,
    }
    cancel_path = watchdog_root / "groot_oscar_runpod_canary_watchdog_cancel.json"
    write_json(cancel_path, cancel)
    zero_inventory = {
        "status": "observed",
        "provider": "vast",
        "name_prefix": prefix,
        "live_resource_count": 0,
        "resources": [],
        "api_confirmed": True,
    }
    recovered = {
        "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "provider_terminal",
        "provider": "vast",
        "pid": 1_098_022,
        "deadline_epoch": deadline,
        "pod_name_prefix": prefix,
        "completed_at": "2026-08-17T07:35:00+00:00",
        "provider_absence_confirmed": True,
        "provider_absence_scope": "recorded_instance_and_lane_prefix",
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "owner_teardown_cancel_requested": True,
        "owner_teardown_cancel_request_valid": True,
        "recorded_vast_instance": {
            "status": "recorded",
            "instance_id": str(INSTANCE_ID),
            "scope_confirmed": True,
            "pod_name_prefix": prefix,
        },
        "recorded_vast_instance_teardown": {
            "status": "absent",
            "instance_id": str(INSTANCE_ID),
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 0,
            "inspect_attempts": [
                {
                    "status": "absent",
                    "provider": "vast",
                    "instance_id": str(INSTANCE_ID),
                    "api_confirmed": True,
                    "provider_absence_confirmed": True,
                },
                {
                    "status": "absent",
                    "provider": "vast",
                    "instance_id": str(INSTANCE_ID),
                    "api_confirmed": True,
                    "provider_absence_confirmed": True,
                },
            ],
        },
        "final_inventory": zero_inventory,
        "final_global_inventory": {**zero_inventory, "name_prefix": ""},
    }
    recovered_path = watchdog_root / "groot_oscar_runpod_canary_watchdog.json"
    write_json(recovered_path, recovered)
    guard = {
        "schema_version": "gpu_spend_guard.v1",
        "status": "passed",
        "generated_at": "2026-08-17T07:36:11+00:00",
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0,
        "inventory_results": [
            {
                "provider": name,
                "required": True,
                "status": "succeeded",
                "row_count": 0,
            }
            for name in ("runpod", "vast", "digitalocean")
        ],
    }
    guard_path = tmp_path / "gpu_spend_guard" / "latest.json"
    write_json(guard_path, guard)

    result = {
        "schema_version": "paired_target_native_import_vast_run.v1",
        "status": "blocked",
        "bundle_sha256": bundle_sha256,
        "request_digest": request_digest,
        "hard_cap_usd": 2.0,
        "hard_ttl_seconds": 6000,
        "provider_mutations_performed": 1,
        "retry_cap": 0,
        "continuing_spend_from_this_run": True,
        "all_staged_objects_absent": True,
        "estimated_cost_usd": 0.058517,
        "authorization_consumption": {
            "status": "consumed",
            "authorization_digest": authority["authorization_digest"],
        },
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "watchdog_receipt_path": str(recovered_path),
        "object_store_cleanup_path": str(cleanup_path),
    }
    result_path = job / "paired_target_native_import_vast_result.v1.json"
    write_json(result_path, result)

    response = {
        "success": True,
        "results": [
            {
                "source": f"instance-{INSTANCE_ID}",
                "type": "instance",
                "amount": 0.007,
                "metadata": {"label": LABEL},
                "items": [
                    {"type": "gpu", "source": None, "items": [], "amount": 0.0},
                    {"type": "disk", "source": None, "items": [], "amount": 0.007},
                    {"type": "bwd", "source": None, "items": [], "amount": 0.0},
                    {"type": "bwu", "source": None, "items": [], "amount": 0.0},
                ],
            }
        ],
    }
    audit = tmp_path / "billing-audit"
    response_path = audit / "response-004-vast.json"
    write_json(response_path, response)
    billing_source = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "provider_totals_usd": {"vast": 1.0},
        "sources": [
            {
                "provider": "vast",
                "endpoint": "https://console.vast.ai/api/v0/charges/",
                "retained_path": str(response_path),
                "response_size_bytes": response_path.stat().st_size,
                "response_digest": _sha256(response_path),
            }
        ],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    billing_path = _self(
        audit / "provider_billing_source_receipt.json",
        billing_source,
        "receipt_digest",
    )
    return {
        "authority": authority_path,
        "result": result_path,
        "teardown": teardown_path,
        "handoff": handoff_path,
        "armed": armed_path,
        "cancel": cancel_path,
        "recovered": recovered_path,
        "cleanup": cleanup_path,
        "adapter": adapter_path,
        "session": session_path,
        "avoidlist": avoidlist_path,
        "guard": guard_path,
        "webapp": webapp_path,
        "billing": billing_path,
        "request": request_path,
        "profile": profile_path,
        "binding": binding_path,
        "started": started_path,
        "launch_receipt": launch_receipt_path,
    }


def _materialize(paths: dict[str, Path], output: Path) -> dict:
    return materialize_paired_target_native_import_recovered_provider_zero(
        attempt_authority_path=paths["authority"],
        result_path=paths["result"],
        recovered_watchdog_path=paths["recovered"],
        original_watchdog_handoff_path=paths["handoff"],
        original_armed_watchdog_path=paths["armed"],
        owner_cancel_receipt_path=paths["cancel"],
        original_teardown_path=paths["teardown"],
        cleanup_path=paths["cleanup"],
        adapter_path=paths["adapter"],
        session_budget_path=paths["session"],
        failure_machine_avoidlist_path=paths["avoidlist"],
        fresh_global_guard_path=paths["guard"],
        webapp_sync_path=paths["webapp"],
        provider_billing_source_receipt_path=paths["billing"],
        launch_label=LABEL,
        output_path=output,
        now=NOW,
    )


def test_live_shaped_recovery_seals_official_zero_without_rewriting_original(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    before = {role: _sha256(path) for role, path in paths.items()}
    output = tmp_path / "recovered-zero.json"
    value = _materialize(paths, output)
    assert value["status"] == "completed_recovered_provider_zero"
    assert value["provider_instance_id"] == INSTANCE_ID
    assert value["attempt_cost_estimate_usd"] == 0.058517
    assert value["official_cost_usd"] == 0.007
    assert value["failed_machine_id"] == MACHINE_ID
    assert value["failed_offer_id"] == OFFER_ID
    assert value["recommended_excluded_machine_ids"] == [MACHINE_ID]
    assert value["provider_absence_observation_count"] == 2
    assert value["science_execution_started"] is False
    assert value["continuing_spend_from_this_run"] is False
    assert value["original_result_reported_continuing_spend"] is True
    assert value["original_teardown_reported_continuing_spend"] is True
    assert value["watchdog_process_recovered"] is True
    assert value["official_charge"]["official_line_items_usd"]["disk"] == 0.007
    assert value["receipt_digest"] == canonical_digest(
        value, digest_field="receipt_digest"
    )
    assert validate_paired_target_native_import_recovered_provider_zero(output) == value
    assert {role: _sha256(path) for role, path in paths.items()} == before


def test_recovery_cli_is_reachable_without_provider_mutation() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "--recovered-watchdog" in completed.stdout
    assert "--provider-billing-source-receipt" in completed.stdout


@pytest.mark.parametrize(
    ("role", "mutation"),
    [
        ("result", lambda value: value.__setitem__("retry_cap", 1)),
        ("adapter", lambda value: value.__setitem__("vast_instance_ids", [1])),
        ("teardown", lambda value: value.__setitem__("status", "completed")),
        ("armed", lambda value: value.__setitem__("deadline_epoch", 1.0)),
        ("cancel", lambda value: value.__setitem__("provider_absence_confirmed", False)),
        (
            "recovered",
            lambda value: value["recorded_vast_instance_teardown"][
                "inspect_attempts"
            ].pop(),
        ),
        ("guard", lambda value: value.__setitem__("live_instance_count", 1)),
        ("session", lambda value: value["attempts"][0].__setitem__("machine_id", 2)),
        ("avoidlist", lambda value: value.__setitem__("machine_ids", [2])),
        (
            "billing",
            lambda value: value["sources"][0].__setitem__(
                "response_digest", "sha256:" + "0" * 64
            ),
        ),
        ("webapp", lambda value: value.__setitem__("run_id", "wrong")),
    ],
)
def test_recovery_rejects_mutated_live_chain(
    tmp_path: Path, role: str, mutation
) -> None:
    paths = _fixture(tmp_path)
    path = paths[role]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    digest_fields = {
        "billing": "receipt_digest",
        "webapp": "sync_result_digest",
    }
    field = digest_fields.get(role)
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    write_json(path, value)
    with pytest.raises(ValueError):
        _materialize(paths, tmp_path / "blocked.json")
