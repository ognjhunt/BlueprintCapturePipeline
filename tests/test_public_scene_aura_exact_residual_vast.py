from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.public_scene_aura_exact_residual_vast import (
    GPU_SELECTION_POLICY,
    PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    consume_aura_exact_residual_paid_attempt_authority_once,
    materialize_aura_exact_residual_provider_runtime_campaign_abstention,
    materialize_aura_exact_residual_paid_attempt_authority,
    materialize_aura_exact_residual_runtime_abstention,
    run_aura_exact_residual_vast,
    validate_aura_exact_residual_paid_attempt_authority,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _corrected_attempt_authority(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Build file-backed, zero-closed evidence for a manual corrected attempt."""

    previous = _write(
        tmp_path / "prior" / "execution.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "retry_cap": 0,
            "raw_result_path": None,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "bundle_sha256": "sha256:" + "a" * 64,
            "preflight_digest": "sha256:" + "b" * 64,
            "estimated_cost_usd": 0.03428,
        },
    )
    runtime = _write(
        tmp_path / "prior" / "runtime.json",
        {
            "schema_version": "public_scene_aura_exact_residual_runtime_result.v1",
            "status": "blocked",
            "aura_inpainting_executed": False,
            "blockers": ["aura_exact_residual_runtime_wonderworld_bytes_changed"],
        },
    )
    teardown = _write(
        tmp_path / "prior" / "teardown.json",
        {"status": "completed", "continuing_spend_from_this_run": False},
    )
    watchdog = _write(
        tmp_path / "prior" / "watchdog.json",
        {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "final_inventory": {"live_resource_count": 0},
        },
    )
    cleanup = _write(
        tmp_path / "prior" / "cleanup.json",
        {"status": "completed", "all_objects_absent": True},
    )
    campaign: dict[str, object] = {
        "schema_version": "public_scene_aura_exact_residual_provider_runtime_campaign_abstention.v1",
        "status": "abstained_shared_provider_runtime_before_aura_entrypoint",
        "preflight_digest": "sha256:" + "b" * 64,
        "provider_zero_confirmed_all": True,
        "aura_inpainting_executed": False,
        "total_estimated_cost_usd": 0.127344,
        "receipt_digest": "",
    }
    campaign["receipt_digest"] = canonical_digest(campaign, digest_field="receipt_digest")
    campaign_path = _write(tmp_path / "prior" / "campaign.json", campaign)
    parent_authority = _write(
        tmp_path / "parent-authority.json",
        {"paid_compute": {"hard_total_spend_cap_usd": 12.0}},
    )
    bundle = {
        "receipt_sha256": "sha256:" + "c" * 64,
        "bundle_sha256": "sha256:" + "d" * 64,
        "preflight_digest": "sha256:" + "b" * 64,
        "execution_authority_digest": "sha256:" + "e" * 64,
        "execution_authority_path": str(parent_authority),
        "allowed_active_instance_ids": [47373597],
    }
    authority: dict[str, object] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "fixture-manual-corrected-aura-attempt",
        "authorized_by": "fixture-user",
        "authorized_on": "2026-08-12",
        "purpose": "manual_corrected_aura_exact_residual_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "manual_corrected_reissue_after_terminal_attempt": True,
        "automatic_paid_retry_authorized": False,
        "maximum_automatic_retries": 0,
        "maximum_paid_attempts": 1,
        "zero_retry": True,
        "parent_execution_authority_digest": bundle["execution_authority_digest"],
        "bundle_receipt_sha256": bundle["receipt_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "preflight_digest": bundle["preflight_digest"],
        "hard_attempt_spend_cap_usd": 6.0,
        "maximum_hourly_rate_usd": 3.0,
        "maximum_single_resource_ttl_seconds": 7200,
        "external_active_instance_allowlist": [47373597],
        "private_derived_upload_only": True,
        "raw_interiorgs_upload_authorized": False,
        "provider_training_authorized": False,
        "publication_authorized": False,
        "exact_mask_only_edits_required": True,
        "previous_bundle_sha256": "sha256:" + "a" * 64,
        "previous_terminal_execution_result": _record(previous),
        "previous_runtime_result": _record(runtime),
        "previous_teardown": _record(teardown),
        "previous_watchdog": _record(watchdog),
        "previous_object_store_cleanup": _record(cleanup),
        "prior_provider_runtime_campaign": _record(campaign_path),
        "prior_goal_spend_usd": 0.161624,
        "aggregate_goal_spend_cap_usd": 12.0,
        "corrective_blueprint_commit": "f" * 40,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    return authority, bundle


def test_exact_residual_uses_only_previously_observed_aura_gpu_classes() -> None:
    assert GPU_SELECTION_POLICY == {
        "policy_id": "aura_exact_residual_observed_cuda_control",
        "allowed_gpu_keywords": ("L40S", "RTX 4090"),
        "denied_gpu_keywords": (),
        "reason": (
            "released Aura author controls previously reached their entrypoint on "
            "both L40S and RTX 4090; no task input or scene claim depends on GPU class"
        ),
    }


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


def test_manual_corrected_attempt_binds_zero_closed_history_and_is_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, bundle = _corrected_attempt_authority(tmp_path)
    validated = validate_aura_exact_residual_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        allowed_active_instance_ids=[47373597],
    )
    assert validated["prior_goal_spend_usd"] == 0.161624

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_exact_residual_vast.AUTHORIZATION_CONSUMPTION_ROOT",
        tmp_path / "consumed",
    )
    first = consume_aura_exact_residual_paid_attempt_authority_once(
        validated, blueprint_commit="f" * 40
    )
    second = consume_aura_exact_residual_paid_attempt_authority_once(
        validated, blueprint_commit="f" * 40
    )
    assert first["status"] == "consumed"
    assert second == {
        "status": "blocked",
        "blockers": ["aura_exact_residual_paid_attempt_authority_consumed"],
    }


def test_manual_corrected_attempt_rejects_digest_shaped_prior_history(tmp_path: Path) -> None:
    authority, bundle = _corrected_attempt_authority(tmp_path)
    authority["previous_runtime_result"] = {
        "path": "/not/file-backed.json",
        "size_bytes": 1,
        "sha256": "sha256:" + "0" * 64,
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    with pytest.raises(ValueError, match="previous_terminal_evidence_unbound"):
        validate_aura_exact_residual_paid_attempt_authority(
            authority,
            prepared_bundle=bundle,
            max_hourly_rate_usd=3.0,
            hard_cap_usd=6.0,
            hard_ttl_seconds=7200,
            allowed_active_instance_ids=[47373597],
        )


def test_materializes_manual_corrected_attempt_from_file_backed_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, bundle = _corrected_attempt_authority(tmp_path)
    bundle_receipt = _write(
        tmp_path / "bundle-receipt.json",
        {"untrusted": "bundle bytes opened by mocked validator"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_exact_residual_vast.validate_aura_exact_residual_bundle",
        lambda _path: {**bundle, "receipt_path": str(bundle_receipt)},
    )
    materialized = materialize_aura_exact_residual_paid_attempt_authority(
        bundle_receipt_path=bundle_receipt,
        previous_terminal_execution_result_path=authority["previous_terminal_execution_result"]["path"],
        previous_runtime_result_path=authority["previous_runtime_result"]["path"],
        previous_teardown_path=authority["previous_teardown"]["path"],
        previous_watchdog_path=authority["previous_watchdog"]["path"],
        previous_object_store_cleanup_path=authority["previous_object_store_cleanup"]["path"],
        prior_provider_runtime_campaign_path=authority["prior_provider_runtime_campaign"]["path"],
        authorization_reference="fixture-manual-corrected-aura-attempt",
        authorized_by="fixture-user",
        authorized_on="2026-08-12",
        corrective_blueprint_commit="f" * 40,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        output_path=tmp_path / "authority.json",
    )
    assert materialized["authorization_digest"]
    assert materialized["prior_goal_spend_usd"] == 0.161624
    assert materialized["previous_runtime_result"] == authority["previous_runtime_result"]


def test_scientific_successor_binds_completed_run_new_preflight_and_extra_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new repair input may follow completion without erasing any goal spend."""

    prior_authority, old_bundle = _corrected_attempt_authority(tmp_path)
    prior_authority_path = _write(tmp_path / "prior-authority.json", prior_authority)
    raw_result = _write(
        tmp_path / "completed" / "raw-result.json",
        {"schema_version": "public_scene_aura_exact_residual_raw_result.v1"},
    )
    completed = _write(
        tmp_path / "completed" / "execution.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "completed",
            "retry_cap": 0,
            "raw_result_path": str(raw_result),
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "bundle_sha256": old_bundle["bundle_sha256"],
            "preflight_digest": old_bundle["preflight_digest"],
            "estimated_cost_usd": 0.279869,
        },
    )
    runtime = _write(
        tmp_path / "completed" / "runtime.json",
        {
            "schema_version": "public_scene_aura_exact_residual_runtime_result.v1",
            "status": "completed",
            "aura_inpainting_executed": True,
            "blockers": [],
        },
    )
    renderer: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_vast_run.v1",
        "status": "completed",
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "estimated_cost_usd": 0.018506,
        "receipt_digest": "",
    }
    renderer["receipt_digest"] = canonical_digest(
        renderer, digest_field="receipt_digest"
    )
    renderer_path = _write(tmp_path / "renderer.json", renderer)
    bundle = {
        **old_bundle,
        "receipt_sha256": "sha256:" + "1" * 64,
        "bundle_sha256": "sha256:" + "2" * 64,
        "preflight_digest": "sha256:" + "3" * 64,
    }
    bundle_receipt = _write(tmp_path / "bundle-receipt.json", {"fixture": True})
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_exact_residual_vast.validate_aura_exact_residual_bundle",
        lambda _path: {**bundle, "receipt_path": str(bundle_receipt)},
    )

    successor = materialize_aura_exact_residual_paid_attempt_authority(
        bundle_receipt_path=bundle_receipt,
        previous_terminal_execution_result_path=completed,
        previous_runtime_result_path=runtime,
        previous_teardown_path=prior_authority["previous_teardown"]["path"],
        previous_watchdog_path=prior_authority["previous_watchdog"]["path"],
        previous_object_store_cleanup_path=prior_authority[
            "previous_object_store_cleanup"
        ]["path"],
        prior_provider_runtime_campaign_path=prior_authority[
            "prior_provider_runtime_campaign"
        ]["path"],
        prior_manual_corrected_attempt_authority_path=prior_authority_path,
        scientific_input_changed_after_terminal_attempt=True,
        additional_terminal_spend_receipt_paths=[renderer_path],
        authorization_reference="fixture-new-broad-repair-input",
        authorized_by="fixture-user",
        authorized_on="2026-08-12",
        corrective_blueprint_commit="f" * 40,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        output_path=tmp_path / "successor-authority.json",
    )

    assert successor["purpose"] == "manual_successor_aura_exact_residual_execution"
    assert successor["previous_preflight_digest"] == old_bundle["preflight_digest"]
    assert successor["preflight_digest"] == bundle["preflight_digest"]
    assert successor["previous_raw_result"] == _record(raw_result)
    assert successor["additional_terminal_spend_receipts"] == [_record(renderer_path)]
    assert successor["prior_goal_spend_usd"] == 0.459999
    assert validate_aura_exact_residual_paid_attempt_authority(
        successor,
        prepared_bundle=bundle,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        allowed_active_instance_ids=[47373597],
    )["prior_goal_spend_usd"] == 0.459999

    same_preflight = dict(successor)
    same_preflight["preflight_digest"] = old_bundle["preflight_digest"]
    same_preflight["authorization_digest"] = canonical_digest(
        same_preflight, digest_field="authorization_digest"
    )
    with pytest.raises(ValueError, match="previous_terminal_execution_invalid"):
        validate_aura_exact_residual_paid_attempt_authority(
            same_preflight,
            prepared_bundle={**bundle, "preflight_digest": old_bundle["preflight_digest"]},
            max_hourly_rate_usd=3.0,
            hard_cap_usd=6.0,
            hard_ttl_seconds=7200,
            allowed_active_instance_ids=[47373597],
        )


def test_second_manual_attempt_chains_the_first_spend_and_zero_closeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrective reissue never erases the prior manual attempt's spend."""

    first_authority, bundle = _corrected_attempt_authority(tmp_path)
    first_path = _write(tmp_path / "first-attempt-authority.json", first_authority)
    second_execution = _write(
        tmp_path / "second" / "execution.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "retry_cap": 0,
            "raw_result_path": None,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "bundle_sha256": bundle["bundle_sha256"],
            "preflight_digest": bundle["preflight_digest"],
            "estimated_cost_usd": 0.134197,
        },
    )
    second_runtime = _write(
        tmp_path / "second" / "runtime.json",
        {
            "schema_version": "public_scene_aura_exact_residual_runtime_result.v1",
            "status": "blocked",
            "aura_inpainting_executed": False,
            "blockers": [
                "aura_exact_residual_runtime_exception:FileNotFoundError",
                "[Errno 2] No such file or directory: "
                "'/workspace/adp_aura_exact_residual_provider_bundle/runtime_output/logs/"
                "train_shared_retained_scene.log'",
            ],
        },
    )
    second_teardown = _write(
        tmp_path / "second" / "teardown.json",
        {"status": "completed", "continuing_spend_from_this_run": False},
    )
    second_watchdog = _write(
        tmp_path / "second" / "watchdog.json",
        {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "final_inventory": {"live_resource_count": 0},
        },
    )
    second_cleanup = _write(
        tmp_path / "second" / "cleanup.json",
        {"status": "completed", "all_objects_absent": True},
    )
    bundle_receipt = _write(tmp_path / "bundle-receipt.json", {"fixture": "validator only"})
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_exact_residual_vast.validate_aura_exact_residual_bundle",
        lambda _path: {**bundle, "receipt_path": str(bundle_receipt)},
    )

    third = materialize_aura_exact_residual_paid_attempt_authority(
        bundle_receipt_path=bundle_receipt,
        previous_terminal_execution_result_path=second_execution,
        previous_runtime_result_path=second_runtime,
        previous_teardown_path=second_teardown,
        previous_watchdog_path=second_watchdog,
        previous_object_store_cleanup_path=second_cleanup,
        prior_provider_runtime_campaign_path=first_authority[
            "prior_provider_runtime_campaign"
        ]["path"],
        prior_manual_corrected_attempt_authority_path=first_path,
        authorization_reference="fixture-second-manual-corrected-aura-attempt",
        authorized_by="fixture-user",
        authorized_on="2026-08-12",
        corrective_blueprint_commit="f" * 40,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        output_path=tmp_path / "third-attempt-authority.json",
    )

    assert third["prior_manual_corrected_attempt_authority"] == _record(first_path)
    assert third["prior_goal_spend_usd"] == 0.295821
    assert validate_aura_exact_residual_paid_attempt_authority(
        third,
        prepared_bundle=bundle,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        allowed_active_instance_ids=[47373597],
    )["prior_goal_spend_usd"] == 0.295821

    third_execution = _write(
        tmp_path / "third" / "execution.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "retry_cap": 0,
            "raw_result_path": None,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "bundle_sha256": bundle["bundle_sha256"],
            "preflight_digest": bundle["preflight_digest"],
            "estimated_cost_usd": 0.552913,
        },
    )
    third_runtime = _write(
        tmp_path / "third" / "runtime.json",
        {
            "schema_version": "public_scene_aura_exact_residual_runtime_result.v1",
            "status": "blocked",
            "aura_inpainting_executed": False,
            "blockers": [
                "aura_exact_residual_runtime_exception:ValueError",
                "aura_exact_residual_runtime_native_point_cloud_missing",
            ],
        },
    )
    fourth = materialize_aura_exact_residual_paid_attempt_authority(
        bundle_receipt_path=bundle_receipt,
        previous_terminal_execution_result_path=third_execution,
        previous_runtime_result_path=third_runtime,
        previous_teardown_path=second_teardown,
        previous_watchdog_path=second_watchdog,
        previous_object_store_cleanup_path=second_cleanup,
        prior_provider_runtime_campaign_path=first_authority[
            "prior_provider_runtime_campaign"
        ]["path"],
        prior_manual_corrected_attempt_authority_path=tmp_path
        / "third-attempt-authority.json",
        authorization_reference="fixture-third-manual-corrected-aura-attempt",
        authorized_by="fixture-user",
        authorized_on="2026-08-12",
        corrective_blueprint_commit="f" * 40,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        output_path=tmp_path / "fourth-attempt-authority.json",
    )

    assert fourth["prior_goal_spend_usd"] == 0.848734
    assert validate_aura_exact_residual_paid_attempt_authority(
        fourth,
        prepared_bundle=bundle,
        max_hourly_rate_usd=3.0,
        hard_cap_usd=6.0,
        hard_ttl_seconds=7200,
        allowed_active_instance_ids=[47373597],
    )["prior_goal_spend_usd"] == 0.848734


def _allocator_args(
    tmp_path: Path, *, bundle_receipt: Path, attempt_authority: Path | None = None
) -> list[str]:
    """Canonical allocator arguments for the exact residual specialization."""

    args = [
        "gpu-canary",
        "--probe-kind",
        "adp-aurafusion360-exact-residual",
        "--provider",
        "vast",
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--expected-source-commit",
        "f" * 40,
        "--experimental-branch-diagnostic",
        "--adp-aura-exact-residual-bundle-receipt",
        str(bundle_receipt),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "3.0",
        "--adp-max-spend-usd",
        "6.0",
        "--adp-hard-ttl-seconds",
        "7200",
        "--adp-allowed-active-vast-instance-id",
        "47373597",
        "--execute",
    ]
    if attempt_authority is not None:
        args.extend(["--adp-aura-attempt-authority", str(attempt_authority)])
    return args


def test_canonical_allocator_requires_and_binds_manual_corrected_attempt_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only paid branch passes the exact authority to the provider adapter."""

    authority, bundle = _corrected_attempt_authority(tmp_path)
    receipt = _write(tmp_path / "bundle-receipt.json", {"fixture": "bound by validator"})
    authority_path = _write(tmp_path / "attempt-authority.json", authority)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "f" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(allocator, "validate_aura_exact_residual_bundle", lambda _path: bundle)
    granted = object()
    monkeypatch.setattr(allocator, "require_paid_resource_admission", lambda *_args, **_kwargs: granted)
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(allocator, "run_aura_exact_residual_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, bundle_receipt=receipt)) == 2
    blocked = json.loads((tmp_path / "adapter.json").read_text(encoding="utf-8"))
    assert "aura_exact_residual_paid_attempt_authority_missing" in blocked["blockers"]

    assert (
        allocator.main(
            _allocator_args(
                tmp_path, bundle_receipt=receipt, attempt_authority=authority_path
            )
        )
        == 0
    )
    assert observed["execute"] is True
    assert observed["paid_resource_admission_grant"] is granted
    assert observed["paid_attempt_authority"] == authority
    admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    binding = admission["allocation_binding"]
    assert binding["paid_attempt_authority_digest"] == authority["authorization_digest"]
    assert binding["paid_attempt_authority_file_sha256"] == _record(authority_path)["sha256"]


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
