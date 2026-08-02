from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.reconstruction_gpu_admission import (
    EXPECTED_RUNTIME_RESULT_SCHEMAS,
    PREFLIGHT_SCHEMA_VERSION,
    PROBE_KIND,
    REQUEST_SCHEMA_VERSION,
    build_reconstruction_gpu_canary_admission,
    build_reconstruction_gpu_canary_request,
    collect_reconstruction_vast_preflight,
)
from blueprint_pipeline.reconstruction_isaac_image_release import (
    build_reconstruction_isaac_image_release,
)


SHA = "a" * 40
D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
D4 = "sha256:" + "4" * 64
D5 = "sha256:" + "5" * 64
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _image_release(*, image=IMAGE, source_commit=SHA):
    return build_reconstruction_isaac_image_release(
        image_manifest={
            "schema_version": "isaac_worker_image_manifest_diagnostic.v2",
            "status": "completed",
            "resolved_digest_ref": image,
            "runnable_platform": "linux/amd64",
            "raw_secret_values_recorded": False,
            "worker_build_identity": {
                "status": "verified",
                "blockers": [],
                "source_commit": source_commit,
                "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
                "worker_image_family": "isaac-eval-worker",
                "isaac_sim_major_version": 6,
                "identity_source": "immutable_registry_image_config_environment",
            },
        },
        expected_source_commit=source_commit,
    )


def _request(**overrides):
    operation = overrides.get("operation", "worker_smoke")
    value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "worker_smoke",
        "capture_profile": "trainer_smoke_fixture",
        "source_commit_sha": SHA,
        "worker_image_digest": IMAGE,
        "worker_stack_manifest_digest": D1,
        "reconstruction_dataset_digest": D2,
        "frozen_split_digest": D3,
        "calibration_digest": D4,
        "deterministic_configuration_digest": D5,
        "operation_request_digest": D1,
        "operation_input_bundle_digest": D2,
        "expected_runtime_result_schema": EXPECTED_RUNTIME_RESULT_SCHEMAS.get(operation),
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 1800,
        "retry_cap": 1,
        "authority_id": "user-authorization-20260730",
        "proof_effect": "none",
    }
    if operation == "provider_nurec_isaac_canary":
        for key in (
            "reconstruction_dataset_digest",
            "frozen_split_digest",
            "calibration_digest",
        ):
            value.pop(key)
        value.update(
            external_import_receipt_digest=D2,
            provider_qualification_report_digest=D3,
            source_relationship_to_blueprint_raw_capture="none",
            external_derived_support_asset=True,
            blueprint_raw_capture_truth=False,
        )
    if operation == "external_scene_isaac_canary":
        for key in (
            "reconstruction_dataset_digest",
            "frozen_split_digest",
            "calibration_digest",
        ):
            value.pop(key)
        value.update(
            remote_processing_authorization_digest=D1,
            package_result_digest=D2,
            collision_candidate_digest=D3,
            scene_frame_binding_digest=D4,
            target_analysis_digest=D5,
            target_binding_digest=D1,
            placement_proposal_digest=D2,
            source_relationship_to_blueprint_raw_capture="none",
            external_derived_support_asset=True,
            blueprint_raw_capture_truth=False,
            source_video_available=False,
            source_video_required_for_candidate_execution=False,
            independent_metric_scale_proven=False,
            remote_upload_authorized=True,
            paid_compute_authorized=True,
        )
    value.update(overrides)
    return build_reconstruction_gpu_canary_request(value)


def _preflight(**overrides):
    value = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "verified",
        "provider": "vast",
        "observed_at_epoch": 1000.0,
        "provider_api_verified": True,
        "provider_inventory_verified_zero": True,
        "conflicting_owner_present": False,
        "watchdog": {"status": "armed", "independent_process": True},
        "single_gpu_available": True,
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 120 * 1024**3,
        "on_demand_price_usd_per_hour": 0.75,
    }
    value.update(overrides)
    return value


def _build(*, request=None, preflight=None, provider="vast", execute=False, **overrides):
    args = {
        "request": request or _request(),
        "preflight": preflight or _preflight(),
        "provider": provider,
        "expected_source_commit": SHA,
        "checkout_source_commit": SHA,
        "checkout_clean": True,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 1800,
        "retry_cap": 1,
        "authority_id": "user-authorization-20260730",
        "execute": execute,
        "observed_now_epoch": 1001.0,
    }
    args.update(overrides)
    return build_reconstruction_gpu_canary_admission(**args)


def test_vast_first_reconstruction_canary_dry_run_binds_all_proof_inputs():
    admission, bound = _build()
    assert admission["status"] == "dry_run_ready"
    assert admission["provider"] == "vast"
    assert admission["provider_mutations_performed"] == 0
    assert admission["paid_execution_started"] is False
    assert admission["allocation_success_is_scientific_success"] is False
    assert bound["reconstruction_dataset_digest"] == D2
    assert bound["frozen_split_digest"] == D3
    assert bound["operation_request_digest"] == D1
    assert bound["operation_input_bundle_digest"] == D2
    assert bound["expected_runtime_result_schema"] == "reconstruction_vast_worker_smoke_result.v1"
    assert bound["provider_mutation_authorized"] is False
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_gpu_canary.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(admission, schema)
    jsonschema.validate(bound, schema)


def test_request_builder_rejects_untyped_or_authority_mutating_operations():
    for override, code in (
        ({"operation": "shell"}, "reconstruction_gpu_operation_unsupported"),
        (
            {"candidate_may_read_hidden_heldout": True},
            "reconstruction_gpu_hidden_heldout_access_forbidden",
        ),
        (
            {"trainer_may_grade_heldout": True},
            "reconstruction_gpu_trainer_self_grading_forbidden",
        ),
        ({"proof_effect": "qualified"}, "reconstruction_gpu_request_proof_effect_invalid"),
    ):
        try:
            _request(**override)
        except ValueError as exc:
            assert code in str(exc)
        else:
            raise AssertionError(f"request builder accepted forbidden override: {override}")


def test_provider_request_forbids_fabricated_capture_bindings():
    request = _request(
        operation="provider_nurec_isaac_canary",
        capture_profile="public_provider_sample",
    )
    assert request["source_relationship_to_blueprint_raw_capture"] == "none"
    assert "reconstruction_dataset_digest" not in request
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_gpu_canary.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(request, schema)
    with pytest.raises(ValueError, match="reconstruction_gpu_provider_capture_binding_forbidden"):
        _request(
            operation="provider_nurec_isaac_canary",
            capture_profile="public_provider_sample",
            reconstruction_dataset_digest=D1,
        )


def test_external_scene_canary_allows_missing_video_but_not_fake_metric_scale():
    request = _request(
        operation="external_scene_isaac_canary",
        capture_profile="user_managed_provider_export",
    )
    assert request["source_video_available"] is False
    assert request["source_video_required_for_candidate_execution"] is False
    assert request["independent_metric_scale_proven"] is False
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_gpu_canary.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(request, schema)
    with pytest.raises(ValueError, match="independent_metric_scale_proven"):
        _request(
            operation="external_scene_isaac_canary",
            capture_profile="user_managed_provider_export",
            independent_metric_scale_proven=True,
        )


def test_preflight_collector_requires_global_zero_and_independent_watchdog():
    inventories = {
        "blueprint-reconstruction-": {
            "api_confirmed": True,
            "live_resource_count": 0,
        },
        "": {"api_confirmed": True, "live_resource_count": 0},
    }
    result = collect_reconstruction_vast_preflight(
        name_prefix="blueprint-reconstruction-",
        container_disk_bytes=120 * 1024**3,
        watchdog={"status": "armed", "independent_process": True},
        conflicting_owner_present=False,
        capacity_probe=lambda _request: {
            "status": "available",
            "selected_offer": {
                "gpu_name": "L40S",
                "gpu_ram_mb": 46_068,
                "hourly_rate_usd": 0.47,
            },
        },
        inventory_probe=lambda prefix: inventories[prefix],
        max_hourly_rate_usd=0.75,
        clock=lambda: 1000.0,
    )
    assert result["status"] == "verified"
    assert result["provider_inventory_verified_zero"] is True
    assert result["capacity_reserved"] is False
    assert result["provider_mutations_performed"] == 0
    assert result["gpu_memory_bytes"] == 46_068_000_000

    blocked = collect_reconstruction_vast_preflight(
        name_prefix="blueprint-reconstruction-",
        container_disk_bytes=10,
        watchdog={"status": "missing", "independent_process": False},
        conflicting_owner_present=True,
        capacity_probe=lambda _request: {"status": "blocked"},
        inventory_probe=lambda _prefix: {
            "api_confirmed": True,
            "live_resource_count": 1,
        },
        max_hourly_rate_usd=0.75,
        clock=lambda: 1000.0,
    )
    assert blocked["status"] == "blocked"
    assert "reconstruction_gpu_provider_inventory_not_zero" in blocked["blockers"]
    assert "reconstruction_gpu_conflicting_owner_present" in blocked["blockers"]
    assert "reconstruction_gpu_independent_watchdog_not_armed" in blocked["blockers"]


def test_reconstruction_canary_fails_closed_on_authority_provider_and_evidence_drift():
    request = _request()
    request["candidate_may_read_hidden_heldout"] = True
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    admission, _ = _build(
        request=request,
        provider="runpod",
        preflight=_preflight(provider_inventory_verified_zero=False),
        max_spend_usd=None,
        hard_ttl_seconds=None,
        retry_cap=None,
        authority_id=None,
    )
    assert admission["status"] == "blocked"
    assert "reconstruction_gpu_vast_first_required" in admission["blockers"]
    assert "reconstruction_gpu_hidden_heldout_access_forbidden" in admission["blockers"]
    assert "reconstruction_gpu_provider_inventory_not_zero" in admission["blockers"]
    assert "reconstruction_gpu_explicit_budget_missing" in admission["blockers"]
    assert admission["provider_mutations_performed"] == 0


def test_execute_stays_blocked_until_vast_execution_adapter_is_qualified():
    admission, bound = _build(execute=True)
    assert admission["status"] == "blocked"
    assert admission["blockers"] == ["reconstruction_vast_execution_adapter_not_qualified"]
    assert admission["legal_next_actions"] == ["qualify_vast_execution_adapter"]
    assert bound["provider_mutation_authorized"] is False


def test_scientific_canaries_use_only_their_qualified_typed_adapters():
    for operation in (
        "pose_canary",
        "trainer_canary",
        "isaac_canary",
        "provider_nurec_isaac_canary",
    ):
        request = _request(
            operation=operation,
            capture_profile=(
                "public_provider_sample"
                if operation == "provider_nurec_isaac_canary"
                else "trainer_smoke_fixture"
            ),
        )
        dry_run, dry_bound = _build(request=request)
        assert dry_run["status"] == "dry_run_ready"
        assert dry_run["operation"] == operation
        assert (
            dry_run["expected_runtime_result_schema"] == EXPECTED_RUNTIME_RESULT_SCHEMAS[operation]
        )
        assert dry_run["execution_adapter_qualified"] is False
        assert dry_run["legal_next_actions"] == [
            "invoke_canonical_gpu_canary_with_explicit_execute_authority"
        ]
        assert dry_bound["provider_mutation_authorized"] is False

        execute, execute_bound = _build(
            request=request,
            execute=True,
            execution_adapter_qualified=True,
            image_release=(
                _image_release()
                if operation in {"isaac_canary", "provider_nurec_isaac_canary"}
                else None
            ),
        )
        assert execute["status"] == "execute_ready"
        assert execute["blockers"] == []
        assert execute["execution_adapter_qualified"] is True
        assert execute_bound["provider_mutation_authorized"] is True


def test_isaac_execute_requires_exact_clean_image_release_binding():
    for operation in ("isaac_canary", "provider_nurec_isaac_canary"):
        request = _request(
            operation=operation,
            capture_profile=(
                "public_provider_sample"
                if operation == "provider_nurec_isaac_canary"
                else "trainer_smoke_fixture"
            ),
        )
        missing, missing_bound = _build(
            request=request,
            execute=True,
            execution_adapter_qualified=True,
        )
        assert "reconstruction_isaac_image_release_missing" in missing["blockers"]
        assert missing_bound["provider_mutation_authorized"] is False

        mismatch, mismatch_bound = _build(
            request=request,
            execute=True,
            execution_adapter_qualified=True,
            image_release=_image_release(
                image="registry.example/blueprint/isaac@sha256:" + "c" * 64
            ),
        )
        assert "reconstruction_isaac_image_release_digest_mismatch" in mismatch["blockers"]
        assert mismatch_bound["provider_mutation_authorized"] is False

        passed, passed_bound = _build(
            request=request,
            execute=True,
            execution_adapter_qualified=True,
            image_release=_image_release(),
        )
        assert passed["status"] == "execute_ready"
        assert passed["isaac_image_release_digest"] == _image_release()["image_release_digest"]
        assert passed_bound["isaac_image_release_digest"] == passed["isaac_image_release_digest"]


def test_operation_request_input_and_result_schema_are_immutable_admission_inputs():
    request = _request()
    request.update(
        operation_request_digest="not-a-digest",
        operation_input_bundle_digest="sha256:" + "z" * 64,
        expected_runtime_result_schema="reconstruction_training_result.v1",
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    admission, bound = _build(request=request)

    assert admission["status"] == "blocked"
    assert "reconstruction_gpu_operation_request_digest_invalid" in admission["blockers"]
    assert "reconstruction_gpu_operation_input_bundle_digest_invalid" in admission["blockers"]
    assert "reconstruction_gpu_expected_runtime_result_schema_invalid" in admission["blockers"]
    assert bound["provider_mutation_authorized"] is False


def test_reconstruction_canary_rejects_stale_preflight_and_underfunded_ttl():
    admission, _ = _build(
        preflight=_preflight(observed_at_epoch=1.0, on_demand_price_usd_per_hour=1.0),
        max_spend_usd=0.1,
        request=_request(max_spend_usd=0.1),
    )
    assert admission["status"] == "blocked"
    assert "reconstruction_gpu_preflight_stale_or_future" in admission["blockers"]
    assert "reconstruction_gpu_budget_below_worst_case_cost" in admission["blockers"]


def test_reconstruction_canary_rejects_digest_and_clean_checkout_drift():
    request = _request()
    request["request_digest"] = D1
    admission, _ = _build(
        request=request,
        checkout_source_commit="b" * 40,
        checkout_clean=False,
    )
    assert "reconstruction_gpu_request_digest_mismatch" in admission["blockers"]
    assert "reconstruction_gpu_checkout_source_commit_mismatch" in admission["blockers"]
    assert "reconstruction_gpu_checkout_not_clean" in admission["blockers"]


def test_allocator_routes_reconstruction_probe_without_provider_mutation(
    tmp_path: Path, monkeypatch, capsys
):
    request_path = tmp_path / "request.json"
    preflight_path = tmp_path / "preflight.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")
    preflight_path.write_text(json.dumps(_preflight(observed_at_epoch=1000.0)), encoding="utf-8")
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda _expected, **_kwargs: ([], SHA),
    )
    monkeypatch.setattr("blueprint_pipeline.reconstruction_gpu_admission.time.time", lambda: 1001.0)
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            PROBE_KIND,
            "--provider-launch-request",
            str(request_path),
            "--release-evidence",
            str(tmp_path / "unused-release.json"),
            "--model-cache-evidence",
            str(tmp_path / "unused-models.json"),
            "--preflight-bundle",
            str(preflight_path),
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--pod-name",
            "blueprint-reconstruction-smoke",
            "--expected-source-commit",
            SHA,
            "--reconstruction-max-spend-usd",
            "1.0",
            "--reconstruction-hard-ttl-seconds",
            "1800",
            "--reconstruction-retry-cap",
            "1",
            "--reconstruction-authority-id",
            "user-authorization-20260730",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}
    admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    adapter = json.loads((tmp_path / "adapter.json").read_text(encoding="utf-8"))
    assert admission["status"] == "dry_run_ready"
    assert adapter["provider_mutations_performed"] == 0
    assert adapter["cost_usd"] == 0.0


@pytest.mark.parametrize(
    "operation",
    ["isaac_canary", "provider_nurec_isaac_canary", "external_scene_isaac_canary"],
)
def test_allocator_routes_isaac_only_to_separate_vast_lifecycle(tmp_path, monkeypatch, operation):
    request_digest = D1
    verification_digest = D2
    bundle_digest = D3
    image = "registry.example/isaac@sha256:" + "f" * 64
    admission = {
        "status": "execute_ready",
        "operation": operation,
        "operation_request_digest": verification_digest,
        "operation_input_bundle_digest": bundle_digest,
        "worker_image_digest": image,
        "source_commit_sha": SHA,
        "blockers": [],
    }
    receipt = {
        "isaac_verification_request_digest": verification_digest,
        "bundle_digest": bundle_digest,
        "runtime_container_image_digest": image,
        "source_commit_sha": SHA,
    }
    bound = {"request_digest": request_digest}
    preflight = {"provider": "vast"}
    receipt_path = tmp_path / "isaac-receipt.json"
    bound_path = tmp_path / "bound.json"
    preflight_path = tmp_path / "preflight.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    bound_path.write_text(json.dumps(bound), encoding="utf-8")
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(allocator, "prepare_reconstruction_gpu_canary", lambda **_kwargs: admission)
    monkeypatch.setattr(
        allocator,
        "validate_isaac_verification_worker_bundle_receipt",
        lambda value: value,
    )
    monkeypatch.setattr(
        allocator,
        "read_sensitive_url_file",
        lambda _path, *, label: (
            f"https://objects.example/{label}",
            {"mode_is_0600": True},
        ),
    )
    monkeypatch.setattr(allocator, "get_render_provider", lambda _name: object())
    calls = []

    def fake_isaac(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": "reconstruction_isaac_vast_execution.v1",
            "status": "completed",
            "provider_mutations_performed": 2,
            "cost_usd": 0.1,
        }

    monkeypatch.setattr(allocator, "run_reconstruction_isaac_vast_operation", fake_isaac)
    monkeypatch.setattr(
        allocator,
        "run_reconstruction_vast_operation",
        lambda **_kwargs: pytest.fail("pose/trainer adapter must not receive Isaac"),
    )
    args = SimpleNamespace(
        reconstruction_refresh_preflight=False,
        provider_launch_request=str(tmp_path / "unused-request.json"),
        preflight_bundle=str(preflight_path),
        admission_out=str(tmp_path / "admission.json"),
        bound_request_out=str(bound_path),
        adapter_output=str(tmp_path / "adapter.json"),
        provider="vast",
        expected_source_commit=SHA,
        reconstruction_max_spend_usd=18.0,
        reconstruction_hard_ttl_seconds=3600,
        reconstruction_retry_cap=1,
        reconstruction_authority_id="user-authorized",
        execute=True,
        provider_output_put_url_file=str(tmp_path / "put-url"),
        provider_output_get_url_file=str(tmp_path / "get-url"),
        provider_bundle_url_file=str(tmp_path / "input-url"),
        reconstruction_operation_receipt_url_file=str(tmp_path / "receipt-url"),
        reconstruction_operation_bundle_receipt=str(receipt_path),
    )
    result = allocator._run_reconstruction_gpu_canary(args, checkout_commit=SHA)
    assert result["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["bundle_receipt"] == receipt
    assert calls[0]["job_dir"].name == "reconstruction_isaac_vast_operation"
    assert calls[0]["input_bundle_get_url"].endswith("provider_bundle_url")
