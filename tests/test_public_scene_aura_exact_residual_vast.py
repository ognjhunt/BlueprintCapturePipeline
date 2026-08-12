from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.public_scene_aura_exact_residual_vast import (
    RESULT_SCHEMA_VERSION,
    materialize_aura_exact_residual_provider_runtime_campaign_abstention,
    materialize_aura_exact_residual_runtime_abstention,
    run_aura_exact_residual_vast,
)


def test_exact_residual_vast_dry_run_has_no_provider_mutation(tmp_path: Path) -> None:
    result = run_aura_exact_residual_vast(
        job_dir=tmp_path,
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle={
            "bundle_sha256": "sha256:" + "a" * 64,
            "preflight_digest": "sha256:" + "b" * 64,
            "allowed_active_instance_ids": [47373597],
        },
        max_hourly_rate_usd=1.5,
        hard_cap_usd=6.0,
        hard_ttl_seconds=14_400,
    )

    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0
    retained = json.loads(
        (tmp_path / "public_scene_aura_exact_residual_vast_result.json").read_text()
    )
    assert retained == result


def test_runtime_abstention_reopens_file_backed_bundle_and_closeout(
    tmp_path: Path, monkeypatch
) -> None:
    """A provider null is only a valid abstention with real zero receipts."""

    module = "blueprint_pipeline.public_scene_aura_exact_residual_vast"
    bundle_receipt = tmp_path / "bundle-receipt.json"
    bundle_receipt.write_text("{}", encoding="utf-8")
    bundle = {
        "receipt_path": str(bundle_receipt),
        "receipt_sha256": "sha256:" + "a" * 64,
        "bundle_sha256": "sha256:" + "b" * 64,
        "preflight_digest": "sha256:" + "c" * 64,
        "replacement_object_count": 2,
        "shared_camera_count": 16,
        "task_count": 2,
    }
    monkeypatch.setattr(f"{module}.validate_aura_exact_residual_bundle", lambda _path: bundle)

    def write(name: str, value: dict) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    budget = write(
        "vast_provider_run/vast_budget_ledger.json",
        {
            "schema_version": "vast_budget_ledger.v1",
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "vast_instance_ids": [47533282],
            "estimated_cost_usd": 0.125,
            "actual_live_runtime_seconds_observed_by_adapter": 180.0,
        },
    )
    adapter = write(
        "vast_provider_run/vast_provider_adapter_result.json",
        {
            "schema_version": "vast_provider_adapter_result.v1",
            "status": "failed",
            "reason": "vast_probe_failed",
            "provider_bundle_kind": "adp_aura_exact_residual",
            "api_call_performed": True,
            "provider_create_attempted": True,
            "continuing_spend_from_this_run": False,
            "machine_avoidlist_path": str(tmp_path / "vast_machine_avoidlist.json"),
            "vast_instance_ids": [47533282],
            "estimated_cost_usd": 0.125,
            "artifacts": {"vast_budget_ledger": str(budget)},
            "blockers": ["vast_heartbeat_instance_exited"],
            "provider_attempt_classification": {
                "classification": "pre_execution_provider_null",
                "provider_bundle_started": False,
                "provider_entrypoint_started": False,
                "provider_output_returned": False,
                "automatic_requeue_authorized": False,
                "automatic_requeue_executed": False,
                "maximum_automatic_requeues": 0,
            },
            "session_budget_summary": {
                "attempts": [{"vast_instance_ids": [47533282], "machine_id": 27753}]
            },
        },
    )
    teardown = write(
        "vast_provider_run/vast_teardown_manifest.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "runner_gpu_teardown_completed": True,
            "vast_instance_ids": [47533282],
        },
    )
    watchdog = write(
        "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "recorded_vast_instance": {"instance_id": "47533282"},
            "recorded_vast_instance_teardown": {"status": "absent"},
            "final_inventory": {"live_resource_count": 0},
        },
    )
    staging = tmp_path / "object_store_staging"
    staging.mkdir()
    write_cleanup = staging / "wam_provider_object_store_cleanup.json"
    write_cleanup.write_text(
        json.dumps(
            {
                "schema_version": "wam_provider_object_store_cleanup.v1",
                "status": "completed",
                "all_objects_absent": True,
                "signed_url_files_removed": True,
            }
        ),
        encoding="utf-8",
    )
    avoidlist = write(
        "vast_machine_avoidlist.json",
        {
            "schema_version": "vast_machine_avoidlist.v1",
            "status": "completed",
            "entries": [
                {
                    "instance_id": 47533282,
                    "reason": "vast_startup_control_plane_did_not_reach_onstart_heartbeat",
                    "retry_policy": "exclude_persistently_across_sibling_jobs_until_manual_review",
                }
            ],
        },
    )
    assert avoidlist.is_file()
    admission = write(
        "admission.json",
        {
            "schema_version": "paid_lane_admission.v1",
            "status": "admitted",
            "retry_cap": 0,
            "private_derived_upload_only": True,
            "raw_interiorgs_upload_authorized": False,
            "provider_training_authorized": False,
            "exact_mask_only_edits_required": True,
            "allocation_binding": {"bundle_receipt_sha256": bundle["receipt_sha256"]},
        },
    )
    result = write(
        "result.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "retry_cap": 0,
            "hard_cap_usd": 6.0,
            "raw_result_path": None,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "bundle_sha256": bundle["bundle_sha256"],
            "preflight_digest": bundle["preflight_digest"],
            "adapter_result_path": str(adapter),
            "teardown_manifest_path": str(teardown),
            "watchdog_receipt_path": str(watchdog),
        },
    )

    receipt = materialize_aura_exact_residual_runtime_abstention(
        execution_result_path=result,
        paid_admission_path=admission,
        bundle_receipt_path=bundle_receipt,
        output_path=tmp_path / "abstention.json",
    )

    assert receipt["status"] == "abstained_provider_runtime_before_aura_entrypoint"
    assert receipt["replacement_object_count"] == 2
    assert receipt["aura_inpainting_executed"] is False
    assert receipt["provider_zero_confirmed"] is True

    result.write_text(
        json.dumps({**json.loads(result.read_text()), "bundle_sha256": "sha256:" + "d" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime_abstention_result_invalid"):
        materialize_aura_exact_residual_runtime_abstention(
            execution_result_path=result,
            paid_admission_path=admission,
            bundle_receipt_path=bundle_receipt,
            output_path=tmp_path / "abstention-tampered.json",
        )


def test_campaign_abstention_requires_two_distinct_zero_closed_hosts(
    tmp_path: Path, monkeypatch
) -> None:
    module = "blueprint_pipeline.public_scene_aura_exact_residual_vast"
    bundle_receipt = tmp_path / "bundle-receipt.json"
    bundle_receipt.write_text("{}", encoding="utf-8")
    bundle = {
        "receipt_path": str(bundle_receipt),
        "receipt_sha256": "sha256:" + "a" * 64,
        "bundle_sha256": "sha256:" + "b" * 64,
        "preflight_digest": "sha256:" + "c" * 64,
        "replacement_object_count": 2,
        "shared_camera_count": 16,
        "task_count": 2,
    }
    monkeypatch.setattr(f"{module}.validate_aura_exact_residual_bundle", lambda _path: bundle)

    def record(path: Path) -> dict[str, object]:
        import hashlib

        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def make_attempt(root: Path, instance_id: int, machine_id: int) -> Path:
        def write(relative: str, value: dict) -> Path:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            return path

        budget = write(
            "vast_provider_run/vast_budget_ledger.json",
            {
                "schema_version": "vast_budget_ledger.v1",
                "status": "completed",
                "continuing_spend_from_this_run": False,
                "vast_instance_ids": [instance_id],
                "estimated_cost_usd": 0.125,
                "actual_live_runtime_seconds_observed_by_adapter": 180.0,
            },
        )
        adapter = write(
            "vast_provider_run/vast_provider_adapter_result.json",
            {
                "schema_version": "vast_provider_adapter_result.v1",
                "status": "failed",
                "reason": "vast_probe_failed",
                "provider_bundle_kind": "adp_aura_exact_residual",
                "api_call_performed": True,
                "provider_create_attempted": True,
                "continuing_spend_from_this_run": False,
                "machine_avoidlist_path": str(root / "vast_machine_avoidlist.json"),
                "vast_instance_ids": [instance_id],
                "estimated_cost_usd": 0.125,
                "artifacts": {"vast_budget_ledger": str(budget)},
                "blockers": ["vast_heartbeat_instance_exited"],
                "provider_attempt_classification": {
                    "classification": "pre_execution_provider_null",
                    "provider_bundle_started": False,
                    "provider_entrypoint_started": False,
                    "provider_output_returned": False,
                    "automatic_requeue_authorized": False,
                    "automatic_requeue_executed": False,
                    "maximum_automatic_requeues": 0,
                },
                "session_budget_summary": {
                    "attempts": [{"vast_instance_ids": [instance_id], "machine_id": machine_id}]
                },
            },
        )
        teardown = write(
            "vast_provider_run/vast_teardown_manifest.json",
            {
                "schema_version": "vast_teardown_manifest.v1",
                "status": "completed",
                "continuing_spend_from_this_run": False,
                "runner_gpu_teardown_completed": True,
                "vast_instance_ids": [instance_id],
            },
        )
        watchdog = write(
            "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json",
            {
                "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
                "status": "provider_terminal",
                "provider_absence_confirmed": True,
                "recorded_vast_instance": {"instance_id": str(instance_id)},
                "recorded_vast_instance_teardown": {"status": "absent"},
                "final_inventory": {"live_resource_count": 0},
            },
        )
        cleanup = write(
            "object_store_staging/wam_provider_object_store_cleanup.json",
            {
                "schema_version": "wam_provider_object_store_cleanup.v1",
                "status": "completed",
                "all_objects_absent": True,
                "signed_url_files_removed": True,
            },
        )
        avoidlist = write(
            "vast_machine_avoidlist.json",
            {
                "schema_version": "vast_machine_avoidlist.v1",
                "status": "completed",
                "entries": [
                    {
                        "instance_id": instance_id,
                        "reason": "vast_startup_control_plane_did_not_reach_onstart_heartbeat",
                        "retry_policy": "exclude_persistently_across_sibling_jobs_until_manual_review",
                    }
                ],
            },
        )
        write(
            "admission.json",
            {
                "schema_version": "paid_lane_admission.v1",
                "status": "admitted",
                "retry_cap": 0,
                "private_derived_upload_only": True,
                "raw_interiorgs_upload_authorized": False,
                "provider_training_authorized": False,
                "exact_mask_only_edits_required": True,
                "allocation_binding": {"bundle_receipt_sha256": bundle["receipt_sha256"]},
            },
        )
        result = write(
            "public_scene_aura_exact_residual_vast_result.json",
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "retry_cap": 0,
                "raw_result_path": None,
                "continuing_spend_from_this_run": False,
                "all_staged_objects_absent": True,
                "bundle_sha256": bundle["bundle_sha256"],
                "preflight_digest": bundle["preflight_digest"],
                "adapter_result_path": str(adapter),
                "teardown_manifest_path": str(teardown),
                "watchdog_receipt_path": str(watchdog),
            },
        )
        receipt = {
            "schema_version": "public_scene_aura_exact_residual_runtime_abstention.v1",
            "status": "abstained_provider_runtime_before_aura_entrypoint",
            "bundle_sha256": bundle["bundle_sha256"],
            "preflight_digest": bundle["preflight_digest"],
            "replacement_object_count": 2,
            "shared_camera_count": 16,
            "task_count": 2,
            "bundle_receipt": record(bundle_receipt),
            "execution_result": record(result),
            "provider_adapter": record(adapter),
            "teardown": record(teardown),
            "independent_watchdog": record(watchdog),
            "object_store_cleanup": record(cleanup),
            "machine_avoidlist": record(avoidlist),
            "provider_budget_ledger": record(budget),
            "provider_instance_id": instance_id,
            "estimated_cost_usd": 0.125,
            "aura_inpainting_executed": False,
            "provider_bundle_started": False,
            "provider_entrypoint_started": False,
            "provider_output_returned": False,
            "automatic_paid_retry_allowed": False,
            "automatic_paid_retry_executed": False,
            "continuing_spend_from_this_run": False,
            "provider_zero_confirmed": True,
            "receipt_digest": "",
        }
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest

        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        path = write("runtime-abstention.json", receipt)
        assert path.is_file()
        return path

    first = make_attempt(tmp_path / "first", 47533282, 27753)
    second = make_attempt(tmp_path / "second", 47534099, 35172)
    receipt = materialize_aura_exact_residual_provider_runtime_campaign_abstention(
        runtime_abstention_paths=[first, second],
        bundle_receipt_path=bundle_receipt,
        output_path=tmp_path / "campaign.json",
    )

    assert receipt["attempt_count"] == 2
    assert receipt["total_estimated_cost_usd"] == 0.25
    assert receipt["provider_zero_confirmed_all"] is True
    assert {item["machine_id"] for item in receipt["attempts"]} == {27753, 35172}

    with pytest.raises(ValueError, match="runtime_abstention_path_invalid"):
        materialize_aura_exact_residual_provider_runtime_campaign_abstention(
            runtime_abstention_paths=[first, first],
            bundle_receipt_path=bundle_receipt,
            output_path=tmp_path / "campaign-duplicate.json",
        )

    same_machine = make_attempt(tmp_path / "same-machine", 47534100, 27753)
    with pytest.raises(ValueError, match="attempts_not_independent"):
        materialize_aura_exact_residual_provider_runtime_campaign_abstention(
            runtime_abstention_paths=[first, same_machine],
            bundle_receipt_path=bundle_receipt,
            output_path=tmp_path / "campaign-same-machine.json",
        )
