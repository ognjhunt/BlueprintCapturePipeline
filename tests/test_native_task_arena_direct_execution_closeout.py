from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_direct_execution_closeout import (
    FILENAME,
    materialize_native_direct_execution_adoption,
    validate_native_direct_execution_adoption,
)
from blueprint_pipeline.project_spend_reconciliation import (
    materialize_project_spend_reconciliation,
    project_spend_dependency_records,
    validate_project_spend_reconciliation,
)
from blueprint_pipeline.provider_billing_reconciler import (
    BILLING_SOURCE_SCHEMA_VERSION,
    VAST_CHARGES_URL,
)
from blueprint_pipeline.same_goal_spend_reconciliation import (
    materialize_same_goal_spend_reconciliation,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import (
    DIRECT_EXECUTION_WEBAPP_SYNC_SUCCEEDED_FILENAME,
    reconcile_launches,
    validated_succeeded_webapp_sync_row,
)
from blueprint_pipeline.task_evaluation_launch_webapp_sync import (
    sync_launch_receipt_to_webapp,
)
from blueprint_pipeline.vast_evidence_contracts import VAST_PROVIDER_ZERO_API_CALL
from blueprint_pipeline.vast_official_billing_extractor import (
    _terminal_evidence,
    materialize_vast_official_same_goal_reconciliation,
)


INSTANCE_ID = 49_349_649
LAUNCH_ID = "blueprint-scene-839873-construction-launch-r2"
LAUNCH_LABEL = "blueprint-native-task-arena-scene-839873-1788154047"
BLOCKERS = [
    "native_rigid_construction_gate_failed:base_collision_clearance",
    "native_rigid_construction_gate_failed:destination_containment",
    "native_rigid_construction_gate_failed:push_contact_maintained",
    "native_rigid_construction_gate_failed:push_path",
]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _identity(path: Path, value: dict, field: str) -> Path:
    value[field] = canonical_digest(value, digest_field=field)
    return _write(path, value)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "pipeline-control-plane" / "task-evaluation-launch-runs" / LAUNCH_ID
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "scene-839873-construction",
        "allocator": {"max_spend_usd": 2.0},
        "webapp_sync": {"max_attempts": 20},
        "profile_digest": "",
    }
    profile_path = _identity(root / "launch_profile.json", profile, "profile_digest")
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": LAUNCH_ID,
        "run_id": LAUNCH_ID,
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "request_digest": "",
    }
    request_path = _identity(root / "launch_request.json", request, "request_digest")
    binding = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": LAUNCH_ID,
        "run_id": LAUNCH_ID,
        "request_digest": request["request_digest"],
        "profile_digest": profile["profile_digest"],
        "execute_requested": True,
        "binding_digest": "",
    }
    binding_path = _identity(root / "launch_binding.json", binding, "binding_digest")
    started = {
        "schema_version": "task_evaluation_launch_started.v1",
        "launch_id": LAUNCH_ID,
        "run_id": LAUNCH_ID,
        "request_digest": request["request_digest"],
        "binding_digest": binding["binding_digest"],
        "automatic_retry_authorized": False,
        "started_digest": "",
    }
    started_path = _identity(root / "launch_started.json", started, "started_digest")
    dispatcher = {
        "status": "blocked",
        "provider_mutations_performed": 0,
        "blockers": ["paid_resource_admission_not_admitted"],
        "result_digest": "",
    }
    dispatcher_path = _identity(
        root / "allocator" / "result.json", dispatcher, "result_digest"
    )
    launch_receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "blocked",
        "launch_id": LAUNCH_ID,
        "run_id": LAUNCH_ID,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "binding_digest": binding["binding_digest"],
        "execute_requested": True,
        "terminal_evidence": {
            "result": {
                "path": str(dispatcher_path),
                "digest": _sha256(dispatcher_path),
                "exists": True,
            }
        },
        "receipt_digest": "",
    }
    receipt_path = _identity(
        root / "launch_receipt.json", launch_receipt, "receipt_digest"
    )
    standing_path = _write(
        tmp_path
        / "pipeline-control-plane"
        / "standing-authorizations"
        / "consumed"
        / str(profile["profile_id"])
        / f"{LAUNCH_ID}.json",
        {
            "schema_version": "task_evaluation_standing_launch_authorization.v1",
            "profile_id": profile["profile_id"],
            "launch_id": LAUNCH_ID,
            "max_spend_usd": 2.0,
        },
    )

    direct_root = root / "allocator" / "direct-execute-r4"
    authority = {
        "schema_version": "native_task_arena_paid_attempt_authority.v1",
        "provider": "vast",
        "paid_compute_authorized": True,
        "blueprint_commit": "a" * 40,
        "bundle_sha256": "sha256:" + "b" * 64,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "retain_warm_session": False,
        "authorization_digest": "",
    }
    authority_path = _identity(
        direct_root / "authority.json", authority, "authorization_digest"
    )
    authority_consumption = {
        "schema_version": "native_task_arena_authority_consumption.v1",
        "authorization_digest": authority["authorization_digest"],
        "blueprint_commit": authority["blueprint_commit"],
        "bundle_sha256": authority["bundle_sha256"],
        "consumed_at": "2026-08-31T05:27:27+00:00",
        "maximum_provider_allocations": 1,
    }
    authority_consumption_path = _write(
        tmp_path / "spend" / "consumed.json", authority_consumption
    )
    attempt = direct_root / "arena-construction-job" / "attempts" / "attempt_001"
    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": "completed",
        "provider_create_attempted": True,
        "vast_instance_ids": [INSTANCE_ID],
        "continuing_spend_from_this_run": False,
        "final_validation_status": "passed",
        "retained_owned": False,
    }
    adapter_path = _write(
        attempt / "vast_provider_run" / "vast_provider_adapter_result.json", adapter
    )
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "generated_at": "2026-08-31T05:41:47+00:00",
        "vast_instance_ids": [INSTANCE_ID],
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "retention_authorized": False,
    }
    teardown_path = _write(
        attempt / "vast_provider_run" / "vast_teardown_manifest.json", teardown
    )
    watchdog_path = _write(
        attempt
        / "independent_vast_watchdog"
        / "groot_oscar_runpod_canary_watchdog.json",
        {"schema_version": "groot_oscar_runpod_canary_watchdog.v1", "status": "provider_terminal"},
    )
    cleanup_path = _write(
        attempt
        / "object_store_staging"
        / "wam_provider_object_store_cleanup.json",
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
        },
    )
    artifact = {
        "schema_version": "task_evaluation_artifact_manifest.v1",
        "status": "completed",
        "manifest_digest": "",
    }
    artifact_path = _identity(attempt / "artifact_manifest.json", artifact, "manifest_digest")
    native = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "blocked",
        "construction_gate_qualified": False,
        "blockers": BLOCKERS,
        "result_digest": "",
    }
    native_path = _identity(
        attempt
        / "immutable_execution"
        / "native_task_arena_construction_result.v1.json",
        native,
        "result_digest",
    )
    direct = {
        "schema_version": "native_task_arena_vast_run.v1",
        "status": "blocked",
        "generated_at": "2026-08-31T05:41:47+00:00",
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "raw_secret_values_recorded": False,
        "estimated_cost_usd": 0.095082,
        "all_staged_objects_absent": True,
        "bundle_sha256": authority["bundle_sha256"],
        "authorization_consumption": {
            "status": "consumed",
            "authorization_digest": authority["authorization_digest"],
            "consumption_record_sha256": _sha256(authority_consumption_path),
            "record_location_disclosed": False,
        },
        "attempt_root": str(attempt),
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "watchdog_receipt_path": str(watchdog_path),
        "object_store_cleanup_path": str(cleanup_path),
        "artifact_manifest_path": str(artifact_path),
        "native_control_result_path": str(native_path),
        "independent_watchdog": {
            "schema_version": "vast_independent_watchdog_handoff.v1",
            "status": "provider_terminal",
            "instance_ids": [INSTANCE_ID],
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 0,
        },
        "blockers": BLOCKERS,
        "result_digest": "",
    }
    direct_path = _identity(direct_root / "result.json", direct, "result_digest")
    _write(
        direct_root / "arena-construction-job" / "adp_arena_vast_result.json",
        direct,
    )
    zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "api_command": VAST_PROVIDER_ZERO_API_CALL,
        "api_confirmed": True,
        "provider_zero": True,
        "global_live_resource_count": 0,
        "inventory": [],
        "stderr_present": False,
        "raw_secret_values_recorded": False,
        "provider_zero_digest": "",
    }
    zero_path = _identity(
        direct_root / "post_teardown_provider_zero_receipt.json",
        zero,
        "provider_zero_digest",
    )
    return {
        "root": root,
        "request": request_path,
        "profile": profile_path,
        "binding": binding_path,
        "started": started_path,
        "receipt": receipt_path,
        "dispatcher": dispatcher_path,
        "standing": standing_path,
        "direct": direct_path,
        "authority": authority_path,
        "authority_consumption": authority_consumption_path,
        "teardown": teardown_path,
        "zero": zero_path,
    }


def _adopt(paths: dict[str, Path]) -> tuple[Path, dict]:
    output = paths["root"] / FILENAME
    value = materialize_native_direct_execution_adoption(
        run_root=paths["root"],
        standing_consumption_path=paths["standing"],
        direct_allocator_result_path=paths["direct"],
        direct_attempt_authority_path=paths["authority"],
        direct_authority_consumption_path=paths["authority_consumption"],
        post_teardown_provider_zero_path=paths["zero"],
        output_path=output,
    )
    return output, value


def _billing(tmp_path: Path) -> tuple[Path, Path]:
    response = _write(
        tmp_path / "billing" / "response-004-vast.json",
        {
            "success": True,
            "results": [
                {
                    "source": f"instance-{INSTANCE_ID}",
                    "amount": 0.085,
                    "type": "instance",
                    "metadata": {"label": LAUNCH_LABEL},
                    "items": [
                        {"type": "gpu", "amount": 0.08, "description": "0.1 hours GPU", "source": None, "items": []},
                        {"type": "disk", "amount": 0.005, "description": "disk", "source": None, "items": []},
                        {"type": "bwd", "amount": 0.0, "description": "download", "source": None, "items": []},
                        {"type": "bwu", "amount": 0.0, "description": "upload", "source": None, "items": []},
                    ],
                }
            ]
        },
    )
    source = {
        "schema_version": BILLING_SOURCE_SCHEMA_VERSION,
        "status": "reconciled",
        "provider_totals_usd": {"vast": 0.085},
        "sources": [
            {
                "provider": "vast",
                "endpoint": VAST_CHARGES_URL,
                "retained_path": str(response),
                "response_size_bytes": response.stat().st_size,
                "response_digest": _sha256(response),
            }
        ],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    source_path = _identity(
        tmp_path / "billing" / "provider_billing_source_receipt.json",
        source,
        "receipt_digest",
    )
    return response, source_path


def test_direct_execution_adoption_closes_billing_project_and_webapp(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    adoption_path, adoption = _adopt(paths)

    assert validate_native_direct_execution_adoption(adoption_path) == adoption
    assert adoption["provider_instance_id"] == INSTANCE_ID
    assert adoption["blockers"] == BLOCKERS
    assert adoption["website_projection"]["configured_scene_offering_status"] == (
        "configured_controls_pending"
    )
    assert adoption["evaluation_ready"] is False

    terminal = _terminal_evidence(
        instance_id=INSTANCE_ID,
        terminal_result_path=adoption_path,
    )
    assert terminal["terminal_status"] == "blocked"
    assert terminal["native_construction_result"]["status"] == "blocked"

    response, source = _billing(tmp_path)
    official_path = tmp_path / "official-vast.json"
    official = materialize_vast_official_same_goal_reconciliation(
        provider_billing_source_receipt_path=source,
        expected_instances=[(INSTANCE_ID, LAUNCH_LABEL, adoption_path)],
        output_path=official_path,
    )
    assert official["official_total_usd"] == 0.085

    same_goal_path = tmp_path / "same-goal.json"
    same_goal = materialize_same_goal_spend_reconciliation(
        lane="native_task_arena",
        terminal_result_paths=[adoption_path],
        teardown_manifest_paths=[paths["teardown"]],
        provider_zero_paths=[paths["zero"]],
        official_billing_response_paths=[response],
        provider_billing_source_receipt_paths=[source],
        output_path=same_goal_path,
    )
    assert same_goal["total_cost_usd"] == 0.085
    baseline = {
        "schema_version": "native_task_arena_paid_attempt_authority.v1",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "aggregate_goal_spend_before_attempt_usd": 79.595914,
        "authorized_on": "2026-08-31T05:00:00+00:00",
        "authorization_digest": "",
    }
    baseline_path = _identity(
        tmp_path / "baseline-authority.json", baseline, "authorization_digest"
    )
    project_path = tmp_path / "project-spend.json"
    project = materialize_project_spend_reconciliation(
        baseline_authority_path=baseline_path,
        posted_reconciliation_paths=[same_goal_path],
        expected_coverage_ids=[LAUNCH_ID],
        completeness_reference="Scene 839873 direct r4 adoption",
        authorized_by="Blueprint owner",
        authorized_on="2026-08-31T06:00:00+00:00",
        output_path=project_path,
    )
    assert project["total_cost_usd"] == pytest.approx(79.680914)
    validate_project_spend_reconciliation(project_path)
    dependency_roles = {
        role for role, _record in project_spend_dependency_records(project)
    }
    assert "posted_entry_0_adoption_original_dispatcher_result" in dependency_roles
    assert "posted_entry_0_adoption_native_construction_result" in dependency_roles

    skipped = sync_launch_receipt_to_webapp(receipt=adoption)
    assert skipped["status"] == "skipped"
    assert skipped["native_construction_blockers"] == BLOCKERS
    attempt = {
        **skipped,
        "status": "succeeded",
        "attempt_number": 1,
        "attempted_at": "2026-08-31T06:00:00+00:00",
        "provider_mutation_performed": False,
        "response": {
            "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
            "status": "blocked",
            "already_exists": True,
            "launch_id": LAUNCH_ID,
            "run_id": LAUNCH_ID,
            "request_digest": adoption["request_digest"],
            "receipt_digest": adoption["receipt_digest"],
        },
        "sync_result_digest": "",
    }
    attempt.pop("reason", None)
    attempt["sync_result_digest"] = canonical_digest(
        attempt, digest_field="sync_result_digest"
    )
    row = validated_succeeded_webapp_sync_row(receipt=adoption, attempt=attempt)
    assert row["receipt"]["native_construction_blockers"] == BLOCKERS
    assert row["receipt"]["qualification_upgrade_performed"] is False


def test_adoption_refuses_dispatcher_that_mutated_a_provider(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    dispatcher = json.loads(paths["dispatcher"].read_text(encoding="utf-8"))
    dispatcher["provider_mutations_performed"] = 1
    _identity(paths["dispatcher"], dispatcher, "result_digest")
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["terminal_evidence"]["result"]["digest"] = _sha256(paths["dispatcher"])
    _identity(paths["receipt"], receipt, "receipt_digest")

    with pytest.raises(ValueError, match="dispatcher_refusal_invalid"):
        _adopt(paths)


def test_reconciler_syncs_adoption_without_overwriting_original_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _adoption_path, adoption = _adopt(paths)

    def sync(*, receipt, **_kwargs):
        assert receipt["receipt_digest"] == adoption["receipt_digest"]
        return {
            "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
            "status": "succeeded",
            "launch_id": LAUNCH_ID,
            "run_id": LAUNCH_ID,
            "request_digest": adoption["request_digest"],
            "receipt_digest": adoption["receipt_digest"],
            "configured_scene_offering_status": "configured_controls_pending",
            "native_construction_status": "blocked",
            "native_construction_blockers": BLOCKERS,
            "qualification_upgrade_performed": False,
            "response": {
                "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
                "status": "blocked",
                "already_exists": True,
                "launch_id": LAUNCH_ID,
                "run_id": LAUNCH_ID,
                "request_digest": adoption["request_digest"],
                "receipt_digest": adoption["receipt_digest"],
            },
        }

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_launch_webapp_sync.sync_launch_receipt_to_webapp",
        sync,
    )
    guard = _write(tmp_path / "guard.json", {})
    report = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=paths["root"].parent,
        guard_report_path=guard,
        publish_progress=False,
    )

    assert report["webapp_sync"][0]["status"] == "webapp_sync_succeeded"
    assert (
        paths["root"] / DIRECT_EXECUTION_WEBAPP_SYNC_SUCCEEDED_FILENAME
    ).is_file()
    assert not (paths["root"] / "webapp_sync_succeeded.json").exists()
    assert json.loads(paths["receipt"].read_text(encoding="utf-8"))[
        "receipt_digest"
    ] != adoption["receipt_digest"]
