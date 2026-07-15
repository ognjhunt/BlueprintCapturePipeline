import json
import os
import time
import zipfile

import pytest

import blueprint_pipeline.groot_oscar_runpod_canary as canary_module
from blueprint_pipeline.groot_oscar_runpod_canary import (
    _finalize_adapter_allocation,
    _reserve_campaign_budget,
    _settle_zero_budget,
    bind_canary_request,
    prepare_canary_launch,
    refresh_runpod_preflight,
    run_canary,
    validate_strict_policy_smoke_output,
)


def test_strict_policy_smoke_binding_is_fixed_and_bounded() -> None:
    request = _request()
    shape = request["provider_request_shape"]
    shape["command"] = "echo caller-command"
    shape.setdefault("limits", {})["external_watchdog_ttl_seconds"] = 180
    admission = {
        "release_image_ref": DIGEST,
        "gpu_type_id": "NVIDIA A40",
        "model_cache_path": "/workspace/models",
        "model_manifest_digest": "sha256:" + "b" * 64,
        "network_volume_id": "volume-1",
        "data_center_id": "US-TX-3",
        "required_cuda_version": "12.6",
    }

    result = bind_canary_request(
        request=request,
        admission=admission,
        probe_kind="strict-policy-smoke",
    )

    assert result["status"] == "ready"
    bound = result["request"]
    bound_shape = bound["provider_request_shape"]
    assert bound["operation"] == "enqueue_runpod_strict_policy_smoke"
    assert bound_shape["operation"] == "enqueue_runpod_strict_policy_smoke"
    assert "command" not in bound_shape
    assert bound_shape["limits"]["hard_timeout_seconds"] == 300
    assert bound_shape["limits"]["external_watchdog_ttl_seconds"] == 360
    assert bound_shape["claim_boundary"]["strict_policy_smoke_only"] is True
    assert (
        bound_shape["claim_boundary"]["fresh_three_action_policy_smoke_required"]
        is True
    )


def test_strict_policy_smoke_output_requires_exact_three_action_proof(
    tmp_path,
) -> None:
    payload = {
        "schema_version": "groot_oscar_runpod_strict_policy_smoke.v1",
        "status": "completed",
        "requested_action_count": 3,
        "completed_action_count": 3,
        "fresh_learned_action_trace": [
            {"action_chunk": [float(index), 0.5]} for index in range(3)
        ],
        "model_execution_proven": True,
        "policy_action_model_command_ran": True,
        "physical_robot_control_performed": False,
        "raw_secret_values_recorded": False,
    }
    output_zip = tmp_path / "strict-smoke.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "groot_oscar_runpod_strict_policy_smoke.json",
            json.dumps(payload),
        )
    result = validate_strict_policy_smoke_output(
        output_zip=output_zip,
        evidence_out=tmp_path / "validation.json",
    )
    assert result["status"] == "passed"
    assert result["completed_action_count"] == 3
    assert result["model_execution_proven"] is True
    assert result["task_success_proven"] is False

    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "groot_oscar_runpod_strict_policy_smoke.json",
            json.dumps(payload | {"completed_action_count": 2}),
        )
        archive.writestr("unexpected.sh", "echo bypass")
    blocked = validate_strict_policy_smoke_output(
        output_zip=output_zip,
        evidence_out=tmp_path / "blocked-validation.json",
    )
    assert blocked["status"] == "blocked"
    assert "strict_policy_smoke_zip_inventory_invalid" in blocked["blockers"]


def _budget_config(tmp_path, **overrides):
    config = {
        "ledger_path": str(tmp_path / "campaign-budget.json"),
        "initial_spent_usd": 11.57,
        "initial_used_gpu_seconds": 11_619,
        "total_spend_cap_usd": 20.0,
        "combined_gpu_wall_cap_seconds": 16_800,
        "reservation_gpu_seconds": 1_200,
        "campaign_stage": "gpu_canary",
        "maximum_canary_reservation_gpu_seconds": 1_200,
        "future_campaign_allowance_gpu_seconds": 3_900,
        "maximum_future_campaign_allowance_gpu_seconds": 3_900,
        "maximum_combined_plan_gpu_seconds": 5_100,
        "reduced_canary_timeout_acknowledged": True,
        "max_hourly_rate_usd": 1.99,
        "minimum_reconciled_spend_usd": 11.57,
        "minimum_reconciled_gpu_seconds": 11_619,
    }
    config.update(overrides)
    return config


def test_original_1500_plus_3900_plan_exceeds_current_authority(tmp_path) -> None:
    result = _reserve_campaign_budget(
        _budget_config(
            tmp_path,
            reservation_gpu_seconds=1_500,
            maximum_canary_reservation_gpu_seconds=1_500,
            maximum_combined_plan_gpu_seconds=5_400,
        ),
        reservation_id="blueprint-canary-budget-blocked",
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["combined_gpu_plan_exceeds_campaign_wall_cap"]


def test_1200_canary_plus_3900_campaign_plan_fits_current_authority(tmp_path) -> None:
    result = _reserve_campaign_budget(
        _budget_config(tmp_path),
        reservation_id="blueprint-canary-reduced-ceiling",
    )
    assert result["status"] == "reserved"
    assert result["reservation"]["reserved_gpu_seconds"] == 1_200
    assert result["plan"] == {
        "campaign_stage": "gpu_canary",
        "canary_reservation_gpu_seconds": 1_200,
        "future_campaign_allowance_gpu_seconds": 3_900,
        "combined_plan_gpu_seconds": 5_100,
    }


def test_generic_5100_second_canary_reservation_is_rejected(tmp_path) -> None:
    result = _reserve_campaign_budget(
        _budget_config(tmp_path, reservation_gpu_seconds=5_100),
        reservation_id="blueprint-canary-not-combined-plan",
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["gpu_canary_stage_reservation_exceeds_plan"]


def test_reduced_canary_plan_requires_explicit_acknowledgement(tmp_path) -> None:
    result = _reserve_campaign_budget(
        _budget_config(tmp_path, reduced_canary_timeout_acknowledged=False),
        reservation_id="blueprint-canary-reduction-not-acknowledged",
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["gpu_canary_reduced_timeout_authorization_missing"]


def test_campaign_budget_rejects_understated_baseline(tmp_path) -> None:
    result = _reserve_campaign_budget(
        _budget_config(tmp_path, initial_used_gpu_seconds=11_618),
        reservation_id="blueprint-canary-understated",
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["gpu_canary_cumulative_baseline_understated"]


def test_confirmed_pre_provider_block_settles_zero(tmp_path) -> None:
    context = _reserve_campaign_budget(
        _budget_config(tmp_path, reservation_gpu_seconds=100),
        reservation_id="blueprint-canary-zero-settlement",
    )
    assert context["status"] == "reserved"
    settlement = _settle_zero_budget(context, outcome="test_no_mutation")
    assert settlement["status"] == "settled"
    assert settlement["charged_gpu_seconds"] == 0


DIGEST = "docker.io/example/release@sha256:" + "a" * 64
MODEL_VOLUME_WATCHDOG_STATE = "/tmp/model-volume/watchdog_state.json"
CANARY_WATCHDOG_OUT_DIR = "/tmp/canary-watchdog"


def _watchdog_argv(pid: int) -> tuple[str, ...]:
    if pid == os.getpid():
        return (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_model_volume",
            "watchdog",
            "--state",
            MODEL_VOLUME_WATCHDOG_STATE,
        )
    return (
        "python",
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_watchdog",
        "--out-dir",
        CANARY_WATCHDOG_OUT_DIR,
        "--pod-name-prefix",
        "blueprint-groot-oscar-canary-",
        "--deadline-epoch",
        "1900.0",
    )


def test_submitted_canary_without_pod_id_fails_closed(tmp_path) -> None:
    output = tmp_path / "adapter.json"

    result = _finalize_adapter_allocation(
        adapter={"status": "submitted", "runpod_response": {}},
        adapter_output=output,
        pod_name="blueprint-groot-oscar-canary-test",
        release_image_ref=DIGEST,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == ["runpod_canary_pod_id_missing"]
    assert result["provider_allocation_ambiguous"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    allocation = json.loads(
        (tmp_path / "warm_serve_pod.json").read_text(encoding="utf-8")
    )
    assert allocation["status"] == "allocation_ambiguous"
    assert allocation["pod_id"] is None


def _request() -> dict:
    return {
        "provider_request_shape": {
            "image": {"configured_image_ref": DIGEST},
            "gpu": {"preferred_gpu_type_id": "NVIDIA A40"},
        }
    }


def _preflight() -> dict:
    return {
        "status": "verified",
        "volume": {
            "provider": "runpod",
            "provider_api_verified": True,
            "id": "volume-1",
            "data_center_id": "US-TX-3",
            "size_bytes": 50 * 1024**3,
            "model_cache_path": "/workspace/models",
        },
        "runtime": {
            "provider": "runpod",
            "provider_api_verified": True,
            "data_center_id": "US-TX-3",
            "capacity_data_center_id": "US-TX-3",
            "capacity_allowed_cuda_versions": ["12.6"],
            "gpu_type_id": "NVIDIA A40",
            "capacity_confidence": "advisory",
            "single_gpu_available": True,
            "required_cuda_version": "12.6",
            "allowed_cuda_versions": ["12.6"],
            "on_demand_price_usd_per_hour": 0.44,
            "warm_worker_only": True,
            "provider_inventory_verified_zero": True,
        },
        "spend": {
            "paid_mutation_authorized": True,
            "max_spend_usd": 1.0,
            "hard_ttl_seconds": 900,
            "one_resource_limit": True,
            "independent_teardown_watchdog": True,
            "watchdog_armed_before_allocation": True,
            "watchdog_pid": 1,
            "watchdog_deadline_epoch": 1900.0,
            "watchdog_pod_name_prefix": "blueprint-groot-oscar-canary-",
            "watchdog_out_dir": CANARY_WATCHDOG_OUT_DIR,
        },
        "model_volume_watchdog_handoff": {
            "schema_version": "groot_oscar_model_volume_watchdog_handoff.v1",
            "status": "volume_ready_watchdog_retained",
            "volume_id": "volume-1",
            "preparation_pod_absence_confirmed": True,
            "volume_presence_confirmed": True,
            "teardown_owner": "independent_model_volume_watchdog",
            "watchdog_pid": os.getpid(),
            "watchdog_state_path": MODEL_VOLUME_WATCHDOG_STATE,
            "watchdog_deadline_epoch": 2800.0,
            "next_owner_must_arm_before_transfer": True,
        },
    }


def test_canary_preparation_binds_exact_admitted_tuple_into_request() -> None:
    result = prepare_canary_launch(
        request=_request(),
        release={
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        model_cache={
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "admitted"
    shape = result["bound_request"]["provider_request_shape"]
    assert shape["network_volume_id"] == "volume-1"
    assert shape["data_center_id"] == "US-TX-3"
    assert shape["allowed_cuda_versions"] == ["12.6"]
    assert shape["cache"]["paths"]["groot_oscar_models"] == "/workspace/models"
    assert shape["docker_entrypoint"] == [
        "/opt/blueprint/thin_release_entrypoint.sh"
    ]
    assert shape["environment"]["plaintext_env_values"][
        "BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST"
    ] == "sha256:" + "b" * 64


def test_canary_preparation_rejects_tag_or_different_digest() -> None:
    request = _request()
    request["provider_request_shape"]["image"]["configured_image_ref"] = "example/release:v1"
    result = prepare_canary_launch(
        request=request,
        release={
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        model_cache={
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "blocked"
    assert "runpod_request_release_image_differs_from_admission" in result["blockers"]


def test_canary_preparation_rejects_different_model_cache_path() -> None:
    request = _request()
    request["provider_request_shape"]["cache"] = {
        "paths": {"groot_oscar_models": "/workspace/other-cache"}
    }
    result = prepare_canary_launch(
        request=request,
        release={
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        model_cache={
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        preflight=_preflight(),
    )
    assert result["status"] == "blocked"
    assert "runpod_request_model_cache_path_differs_from_admission" in result["blockers"]


def test_execute_refresh_rechecks_every_mutable_provider_fact() -> None:
    observed: dict[str, object] = {}

    def volume_getter(volume_id: str):
        observed["volume_id"] = volume_id
        return 200, {"id": volume_id, "dataCenterId": "US-TX-3", "size": 50}

    def capacity_probe(request):
        observed["capacity_request"] = request
        return {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": "NVIDIA A40",
                    "capacity_confidence": "advisory",
                    "capacity_data_center_id": request["dataCenterIds"][0],
                    "capacity_allowed_cuda_versions": request[
                        "allowedCudaVersions"
                    ],
                    "available_gpu_counts": [1],
                    "single_gpu_offer_requested": True,
                    "single_gpu_offer_available": True,
                    "on_demand_price_usd_per_hour": 0.44,
                }
            ],
        }

    def inventory_probe(prefix: str | None):
        observed.setdefault("prefixes", []).append(prefix)
        return {"api_confirmed": True, "live_resource_count": 0}

    result = refresh_runpod_preflight(
        preflight=_preflight(),
        volume_getter=volume_getter,
        capacity_probe=capacity_probe,
        inventory_probe=inventory_probe,
        clock=lambda: 1000.0,
        process_argv_probe=_watchdog_argv,
    )
    assert result["status"] == "verified"
    assert result["observed_at_epoch"] == 1000.0
    assert observed["volume_id"] == "volume-1"
    assert observed["capacity_request"] == {
        "cloudType": "SECURE",
        "gpuTypeIds": ["NVIDIA A40"],
        "dataCenterIds": ["US-TX-3"],
        "allowedCudaVersions": ["12.6"],
        "requires_rtx": True,
    }
    assert observed["prefixes"] == ["blueprint-groot-oscar-canary-", None]


def test_execute_blocks_before_adapter_when_launch_refresh_fails(
    tmp_path, monkeypatch
) -> None:
    class Provider:
        def _key(self):
            return "test-key"

        def capacity_preflight(self, _request):
            return {"status": "blocked", "viable_gpu_types": []}

        def billable_inventory(self, *, name_prefix):
            del name_prefix
            return {"api_confirmed": False, "live_resource_count": 0}

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.get_render_provider",
        lambda _name: Provider(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary._runpod_call",
        lambda *args, **kwargs: (404, {}),
    )

    def adapter_must_not_run(**_kwargs):
        raise AssertionError("provider adapter reached after blocked launch refresh")

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.run_runpod_provider_adapter",
        adapter_must_not_run,
    )
    payloads = {
        "request.json": _request(),
        "release.json": {
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        "models.json": {
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        "preflight.json": _preflight(),
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    result = run_canary(
        provider_launch_request=tmp_path / "request.json",
        release_evidence=tmp_path / "release.json",
        model_cache_evidence=tmp_path / "models.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        pod_name="blueprint-groot-oscar-canary-test",
        execute=True,
    )
    assert result["status"] == "blocked"
    assert "runpod_preflight_bundle_not_verified" in result["blockers"]
    refresh = json.loads(
        (tmp_path / "runpod_preflight_launch_refresh.json").read_text(
            encoding="utf-8"
        )
    )
    assert refresh["status"] == "blocked"
    assert refresh["provider_mutations_performed"] == 0


def test_execute_reserves_then_accepts_handoff_before_adapter(
    tmp_path, monkeypatch
) -> None:
    order: list[str] = []
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    preflight = _preflight()
    preflight["spend"]["watchdog_out_dir"] = str(watchdog_dir)
    preflight["spend"]["hard_ttl_seconds"] = 1_200
    preflight["spend"]["watchdog_deadline_epoch"] = time.time() + 1_100
    preflight["model_volume_watchdog_handoff"]["provider_lane_handoff"] = {
        "binding": {"volume_id": "volume-1"}
    }

    class Provider:
        def _key(self):
            return "test-key"

        def capacity_preflight(self, _request):
            return {}

        def billable_inventory(self, *, name_prefix):
            del name_prefix
            return {}

    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.get_render_provider",
        lambda _name: Provider(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.refresh_runpod_preflight",
        lambda **_kwargs: order.append("refresh") or preflight,
    )
    real_reserve = canary_module._reserve_campaign_budget
    monkeypatch.setattr(
        canary_module,
        "_reserve_campaign_budget",
        lambda config, *, reservation_id: order.append("reserve")
        or real_reserve(config, reservation_id=reservation_id),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.accept_paid_provider_lane_lease_handoff",
        lambda *_args, **_kwargs: order.append("accept")
        or {"status": "accepted", "lease_path": str(tmp_path / "lease"), "owner_pid": 1},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.open_pending_teardown",
        lambda **_kwargs: order.append("pending") or {"path": str(tmp_path / "pending")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.bind_pending_teardown_instance",
        lambda *_args: order.append("bind") or {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.groot_oscar_runpod_canary.run_runpod_provider_adapter",
        lambda **_kwargs: order.append("adapter")
        or {"status": "submitted", "runpod_response": {"id": "pod-1"}},
    )
    payloads = {
        "request.json": _request(),
        "release.json": {
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        "models.json": {
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        "preflight.json": preflight,
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    result = run_canary(
        provider_launch_request=tmp_path / "request.json",
        release_evidence=tmp_path / "release.json",
        model_cache_evidence=tmp_path / "models.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        pod_name="blueprint-groot-oscar-canary-ordering",
        execute=True,
        campaign_budget=_budget_config(tmp_path),
    )
    assert result["status"] == "submitted"
    assert order == ["refresh", "reserve", "accept", "pending", "adapter", "bind"]
    receipt = json.loads(
        (watchdog_dir / "provider_lane_handoff_receipt.json").read_text()
    )
    assert receipt["campaign_budget"]["status"] == "reserved"
    contract = receipt["campaign_budget"]["watchdog_contract"]
    assert contract["hard_ttl_seconds"] == 1_200
    assert contract["reserved_gpu_seconds"] == 1_200
    assert 1 <= contract["watchdog_remaining_seconds_at_reservation"] <= 1_200


def test_handoff_rejection_blocks_adapter_and_settles_zero(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    preflight = _preflight()
    preflight["spend"]["watchdog_out_dir"] = str(watchdog_dir)
    preflight["spend"]["hard_ttl_seconds"] = 1_200
    preflight["spend"]["watchdog_deadline_epoch"] = time.time() + 1_100
    preflight["model_volume_watchdog_handoff"]["provider_lane_handoff"] = {
        "binding": {"volume_id": "volume-1"}
    }

    class Provider:
        def _key(self):
            return "test-key"

        def capacity_preflight(self, _request):
            return {}

        def billable_inventory(self, *, name_prefix):
            del name_prefix
            return {}

    monkeypatch.setattr(canary_module, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        canary_module,
        "refresh_runpod_preflight",
        lambda **_kwargs: order.append("refresh") or preflight,
    )
    real_reserve = canary_module._reserve_campaign_budget
    monkeypatch.setattr(
        canary_module,
        "_reserve_campaign_budget",
        lambda config, *, reservation_id: order.append("reserve")
        or real_reserve(config, reservation_id=reservation_id),
    )
    monkeypatch.setattr(
        canary_module,
        "accept_paid_provider_lane_lease_handoff",
        lambda *_args, **_kwargs: order.append("accept")
        or {"status": "blocked", "blockers": ["handoff_test_rejected"]},
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("provider boundary reached after rejected handoff")

    monkeypatch.setattr(canary_module, "open_pending_teardown", must_not_run)
    monkeypatch.setattr(canary_module, "run_runpod_provider_adapter", must_not_run)
    payloads = {
        "request.json": _request(),
        "release.json": {
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        "models.json": {
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        "preflight.json": preflight,
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    result = run_canary(
        provider_launch_request=tmp_path / "request.json",
        release_evidence=tmp_path / "release.json",
        model_cache_evidence=tmp_path / "models.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        pod_name="blueprint-groot-oscar-canary-rejected",
        execute=True,
        campaign_budget=_budget_config(tmp_path),
    )
    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert order == ["refresh", "reserve", "accept"]
    ledger = json.loads((tmp_path / "campaign-budget.json").read_text())
    assert ledger["reservations"][0]["status"] == "settled"
    assert ledger["reservations"][0]["charged_gpu_seconds"] == 0


@pytest.mark.parametrize(
    ("hard_ttl_seconds", "deadline_offset"),
    [(1_201, 100), (1_200, 1_300)],
    ids=["ttl_exceeds_reservation", "deadline_exceeds_reservation"],
)
def test_watchdog_contract_exceeding_reservation_blocks_before_handoff(
    tmp_path, monkeypatch, hard_ttl_seconds, deadline_offset
) -> None:
    preflight = _preflight()
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    preflight["spend"]["watchdog_out_dir"] = str(watchdog_dir)
    preflight["spend"]["hard_ttl_seconds"] = hard_ttl_seconds
    preflight["spend"]["watchdog_deadline_epoch"] = time.time() + deadline_offset
    preflight["model_volume_watchdog_handoff"]["provider_lane_handoff"] = {
        "binding": {"volume_id": "volume-1"}
    }

    class Provider:
        def _key(self):
            return "test-key"

        def capacity_preflight(self, _request):
            return {}

        def billable_inventory(self, *, name_prefix):
            del name_prefix
            return {}

    monkeypatch.setattr(canary_module, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        canary_module, "refresh_runpod_preflight", lambda **_kwargs: preflight
    )

    def must_not_accept(*_args, **_kwargs):
        raise AssertionError("handoff accepted with an under-reserved watchdog")

    monkeypatch.setattr(
        canary_module, "accept_paid_provider_lane_lease_handoff", must_not_accept
    )
    payloads = {
        "request.json": _request(),
        "release.json": {
            "resolved_digest_ref": DIGEST,
            "thin_release_contract_status": "passed",
            "runnable_platform": "linux/amd64",
            "required_cuda_version": "12.6",
            "required_cuda_version_source": "image_config_env:CUDA_VERSION",
        },
        "models.json": {
            "schema_version": "groot_oscar_external_model_cache_verification.v2",
            "status": "passed",
            "model_manifest_digest": "sha256:" + "b" * 64,
            "verified_size_bytes": 20 * 1024**3,
            "cache_root": "/workspace/models",
            "provider_volume_id": "volume-1",
            "checks": {"models_cached_offline": True},
        },
        "preflight.json": preflight,
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    result = run_canary(
        provider_launch_request=tmp_path / "request.json",
        release_evidence=tmp_path / "release.json",
        model_cache_evidence=tmp_path / "models.json",
        preflight_bundle=tmp_path / "preflight.json",
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        pod_name="blueprint-groot-oscar-canary-budget-contract",
        execute=True,
        campaign_budget=_budget_config(tmp_path),
    )
    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == ["gpu_canary_watchdog_exceeds_budget_reservation"]
    evidence = json.loads(
        (tmp_path / "campaign_budget_reservation.json").read_text()
    )
    assert evidence["watchdog_contract"]["hard_ttl_seconds"] == hard_ttl_seconds
    assert evidence["zero_settlement"]["status"] == "settled"


@pytest.mark.parametrize("raises", [False, True])
def test_pre_provider_recovery_persists_control_plane_open_receipt(
    tmp_path, monkeypatch, raises
) -> None:
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    budget = _reserve_campaign_budget(
        _budget_config(tmp_path), reservation_id=f"recovery-{raises}"
    )

    def restore(_acceptance):
        if raises:
            raise OSError("lease unavailable")
        return {"status": "refused_identity_mismatch"}

    monkeypatch.setattr(
        canary_module,
        "restore_paid_provider_lane_lease_to_retained_watchdog",
        restore,
    )
    if raises:
        monkeypatch.setattr(
            canary_module,
            "_settle_zero_budget",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("ledger busy")),
        )
    result = canary_module._recover_accepted_handoff_before_provider_mutation(
        acceptance={"status": "accepted", "owner_pid": 123},
        budget_context=budget,
        watchdog_out_dir=str(watchdog_dir),
        pod_name_prefix="blueprint-groot-oscar-canary-",
        outcome="test_pre_provider_failure",
    )
    assert result["status"] == "control_plane_open"
    assert result["provider_mutations_performed"] == 0
    assert result["recovery_receipt_written"] is True
    receipt_path = watchdog_dir / "provider_lane_handoff_receipt.json"
    assert receipt_path.stat().st_mode & 0o077 == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["pre_provider_mutation_confirmed_absent"] is True
    assert receipt["pre_provider_recovery"]["status"] == "control_plane_open"
