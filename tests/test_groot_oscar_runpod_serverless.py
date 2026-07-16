import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import groot_oscar_runpod_serverless as serverless
from blueprint_pipeline.groot_oscar_runpod_serverless import (
    CARRIER_CONTAINER_DISK_GIB,
    DEFAULT_RESERVATION_SECONDS,
    build_endpoint_payload,
    build_template_payload,
    collect_inventory,
    compute_startup_wall_timeout_seconds,
    validate_serverless_inputs,
)
from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
)


IMAGE = "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:" + "a" * 64
MODEL_DIGEST = "sha256:" + "b" * 64
CARRIER_IMAGE = "pytorch/pytorch@sha256:" + "d" * 64


def _release() -> dict:
    return {
        "status": "completed",
        "resolved_digest_ref": IMAGE,
        "source_commit": "c" * 40,
        "runnable_platform": "linux/amd64",
        "thin_release_contract": {
            "status": "passed",
            "models_externalized": True,
        },
        "serverless_worker_contract": {
            "status": "passed",
            "worker_source_packaged": True,
            "worker_command_packaged": True,
            "runpod_sdk_exactly_pinned": True,
        },
    }


def _model() -> dict:
    return {
        "schema_version": "groot_oscar_external_model_cache_verification.v2",
        "status": "passed",
        "model_manifest_digest": MODEL_DIGEST,
        "provider_volume_id": "volume-1",
    }


def _volume() -> dict:
    return {
        "id": "volume-1",
        "data_center_id": "EUR-IS-1",
        "provider_api_verified": True,
    }


def _inventory() -> dict:
    return {
        "api_confirmed": True,
        "matching_compute_count": 0,
        "matching_template_count": 0,
    }


def _carrier(data_center_id: str = "EUR-IS-1") -> dict:
    return {
        "schema_version": CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        "status": "verified",
        "carrier_image_ref": CARRIER_IMAGE,
        "network_volume": {
            "id": "volume-1",
            "data_center_id": data_center_id,
            "size_gib": 120,
        },
        "runtime_bundle": {
            "manifest_schema_version": RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "source_release_image_ref": IMAGE,
            "root": DEFAULT_RUNTIME_ROOT,
            "archive_path": DEFAULT_RUNTIME_ARCHIVE_PATH,
            "manifest_path": DEFAULT_RUNTIME_MANIFEST_PATH,
            "archive_sha256": "e" * 64,
            "manifest_sha256": "f" * 64,
        },
        "runtime_source_release": {
            "schema_version": RUNTIME_SOURCE_RELEASE_VERIFICATION_SCHEMA_VERSION,
            "status": "verified",
            "release_image_ref": IMAGE,
            "source_commit": "c" * 40,
            "thin_release_contract_sha256": "1" * 64,
            "models_externalized": True,
        },
        "model_cache": {
            "status": "verified",
            "root": DEFAULT_MODEL_CACHE_ROOT,
            "manifest_sha256": "b" * 64,
            "manifest_digest": MODEL_DIGEST,
        },
        "s3_transfer_verification": {
            "upload_completed": True,
            "full_redownload_sha256_verified": True,
            "provider_volume_id": "volume-1",
            "data_center_id": data_center_id,
        },
        "raw_secret_values_recorded": False,
    }


def test_template_is_private_queue_worker_with_serverless_cache_mapping() -> None:
    payload = build_template_payload(
        name="blueprint-groot-oscar-serverless-test",
        image_ref=IMAGE,
        source_commit="c" * 40,
        model_manifest_digest=MODEL_DIGEST,
    )

    assert payload["isServerless"] is True
    assert payload["isPublic"] is False
    assert payload["imageName"] == IMAGE
    assert payload["dockerEntrypoint"] == ["/opt/blueprint/thin_release_entrypoint.sh"]
    assert payload["dockerStartCmd"] == [
        "/opt/runpod-serverless-venv/bin/python",
        "-m",
        "blueprint_pipeline.groot_oscar_runpod_serverless_worker",
    ]
    assert payload["env"]["BLUEPRINT_GROOT_OSCAR_MODEL_CACHE"].startswith("/runpod-volume/")


def test_endpoint_uses_active_flashboot_a40_then_l40s_and_one_worker() -> None:
    payload = build_endpoint_payload(
        name="blueprint-groot-oscar-serverless-test",
        template_id="template-1",
        network_volume_id="volume-1",
        data_center_id="EUR-IS-1",
    )

    assert payload["flashboot"] is True
    assert payload["workersMin"] == payload["workersMax"] == 1
    assert payload["gpuTypeIds"] == ["NVIDIA A40", "NVIDIA L40S"]
    assert all("H100" not in gpu for gpu in payload["gpuTypeIds"])
    assert payload["networkVolumeId"] == "volume-1"
    assert payload["dataCenterIds"] == ["EUR-IS-1"]
    assert payload["executionTimeoutMs"] == 3_500_000


def test_template_bootstraps_verified_runtime_from_persistent_carrier_volume() -> None:
    payload = build_template_payload(
        name="blueprint-groot-oscar-serverless-carrier-test",
        image_ref=IMAGE,
        source_commit="c" * 40,
        model_manifest_digest=MODEL_DIGEST,
        carrier_volume_admission=_carrier("US-WA-1"),
    )

    assert payload["imageName"] == CARRIER_IMAGE
    assert payload["containerDiskInGb"] == CARRIER_CONTAINER_DISK_GIB
    assert payload["dockerEntrypoint"] == ["/bin/bash", "-lc"]
    assert "BLUEPRINT_RUNPOD_CARRIER_RUNTIME_BOOTSTRAP_STARTED" in payload["dockerStartCmd"][0]
    assert payload["env"]["BLUEPRINT_RUNTIME_ARCHIVE_PATH"].startswith("/runpod-volume/")
    assert payload["env"]["BLUEPRINT_GROOT_OSCAR_MODEL_CACHE"].startswith("/runpod-volume/")
    assert payload["env"]["BLUEPRINT_WORKER_IMAGE_DIGEST"] == IMAGE
    assert payload["env"]["BLUEPRINT_RUNTIME_CARRIER_IMAGE_DIGEST"] == CARRIER_IMAGE


def test_legacy_v2_carrier_uses_volume_bound_model_verification_digest() -> None:
    carrier = _carrier("US-WA-1")
    carrier["schema_version"] = LEGACY_CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION
    carrier["model_cache"].pop("manifest_digest")
    volume = _volume()
    volume["data_center_id"] = "US-WA-1"

    admission = validate_serverless_inputs(
        release=_release(),
        model_cache=_model(),
        volume=volume,
        provider_inventory=_inventory(),
        expected_source_commit="c" * 40,
        resource_name_prefix="blueprint-groot-oscar-serverless-legacy-",
        reservation_seconds=5_215,
        initial_spent_usd=15.875304422841,
        initial_gpu_seconds=15_785,
        max_hourly_rate_usd=1.75,
        gpu_type_ids=("NVIDIA RTX 6000 Ada Generation",),
        carrier_volume_admission=carrier,
    )
    payload = build_template_payload(
        name="blueprint-groot-oscar-serverless-legacy",
        image_ref=IMAGE,
        source_commit="c" * 40,
        model_manifest_digest=MODEL_DIGEST,
        carrier_volume_admission=carrier,
    )

    assert admission["status"] == "admitted"
    assert admission["model_manifest_digest"] == MODEL_DIGEST
    assert payload["env"]["BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST"] == MODEL_DIGEST


def test_template_rejects_non_hex_model_manifest_digest() -> None:
    with pytest.raises(ValueError, match="serverless_model_manifest_digest_invalid"):
        build_template_payload(
            name="blueprint-groot-oscar-serverless-invalid-digest",
            image_ref=IMAGE,
            source_commit="c" * 40,
            model_manifest_digest="sha256:" + "z" * 64,
        )


def test_admission_accepts_us_carrier_and_exact_residual_campaign_budget() -> None:
    volume = _volume()
    volume["data_center_id"] = "US-WA-1"
    result = validate_serverless_inputs(
        release=_release(),
        model_cache=_model(),
        volume=volume,
        provider_inventory=_inventory(),
        expected_source_commit="c" * 40,
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        reservation_seconds=5_215,
        initial_spent_usd=15.875304422841,
        initial_gpu_seconds=15_785,
        max_hourly_rate_usd=1.75,
        gpu_type_ids=("NVIDIA RTX 6000 Ada Generation",),
        carrier_volume_admission=_carrier("US-WA-1"),
    )

    assert result["status"] == "admitted"
    assert result["carrier_volume_verified"] is True
    assert result["maximum_startup_seconds"] == 1_295
    assert result["gpu_type_ids"] == ["NVIDIA RTX 6000 Ada Generation"]


def test_blocked_carrier_returns_admission_without_building_request_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = tmp_path / "release.json"
    model = tmp_path / "model.json"
    handoff = tmp_path / "handoff.json"
    carrier = tmp_path / "carrier.json"
    api_key = tmp_path / "runpod_api_key"
    release.write_text(json.dumps(_release()), encoding="utf-8")
    model.write_text(json.dumps(_model()), encoding="utf-8")
    handoff.write_text(
        json.dumps(
            {
                "provider_lane_handoff": {
                    "binding": {
                        "provider": "runpod",
                        "lane": "groot_oscar_model_volume",
                        "volume_id": "volume-1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    carrier.write_text(json.dumps({"status": "malformed"}), encoding="utf-8")
    api_key.write_text("private", encoding="utf-8")
    api_key.chmod(0o600)

    def fake_provider(method, path, _payload, *, key, timeout):
        del method, key, timeout
        if path == "/networkvolumes/volume-1":
            return 200, {"id": "volume-1", "dataCenterId": "EUR-IS-1"}
        return 200, []

    monkeypatch.setattr(serverless, "_runpod_call", fake_provider)
    monkeypatch.setattr(
        serverless,
        "validate_campaign_io_evidence",
        lambda *_args, **_kwargs: {"status": "passed", "blockers": []},
    )
    monkeypatch.setattr(
        serverless,
        "build_template_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked admission must not build a template")
        ),
    )

    result = serverless.run_active_worker(
        output_dir=tmp_path / "out",
        release_evidence=release,
        model_cache_evidence=model,
        watchdog_handoff_evidence=handoff,
        api_key_file=api_key,
        campaign_io_evidence=tmp_path / "campaign_io.json",
        runpod_s3_access_key_file="unused",
        runpod_s3_secret_key_file="unused",
        resource_name_prefix="blueprint-groot-oscar-serverless-blocked-",
        expected_source_commit="c" * 40,
        execute=False,
        campaign_budget_ledger=tmp_path / "budget.json",
        initial_spent_usd=15.875304422841,
        initial_gpu_seconds=15_785,
        carrier_volume_admission=carrier,
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    shapes = json.loads((tmp_path / "out" / "serverless_request_shapes.json").read_text())
    assert shapes == {
        "status": "blocked_before_request_shape",
        "template": None,
        "endpoint": None,
    }


def test_startup_timeout_preserves_strict_and_campaign_reserves() -> None:
    assert compute_startup_wall_timeout_seconds(deadline_epoch=5_000.0, now_epoch=992.0) == 88
    assert compute_startup_wall_timeout_seconds(deadline_epoch=10_000.0, now_epoch=1_000.0) == 1_200


def test_admission_preserves_full_campaign_and_exact_caps() -> None:
    result = validate_serverless_inputs(
        release=_release(),
        model_cache=_model(),
        volume=_volume(),
        provider_inventory=_inventory(),
        expected_source_commit="c" * 40,
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        reservation_seconds=DEFAULT_RESERVATION_SECONDS,
        initial_spent_usd=14.708611,
        initial_gpu_seconds=15_785,
        carrier_volume_admission=_carrier(),
    )

    assert result["status"] == "admitted"
    assert result["ordinary_runpod_pod_create_allowed"] is False
    assert result["workers_min"] == 1
    assert result["semantic_task_success_proven"] is False


def test_admission_blocks_unbounded_or_preexisting_compute() -> None:
    inventory = _inventory()
    inventory["matching_compute_count"] = 1
    result = validate_serverless_inputs(
        release=_release(),
        model_cache=_model(),
        volume=_volume(),
        provider_inventory=inventory,
        expected_source_commit="c" * 40,
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        reservation_seconds=DEFAULT_RESERVATION_SECONDS - 1,
        initial_spent_usd=14.708611,
        initial_gpu_seconds=15_785,
        carrier_volume_admission=_carrier(),
    )

    assert result["status"] == "blocked"
    assert "serverless_matching_compute_already_present" in result["blockers"]
    assert "serverless_campaign_reservation_must_equal_remaining_wall_cap" in result["blockers"]


def test_admission_blocks_release_from_another_source_head() -> None:
    result = validate_serverless_inputs(
        release=_release(),
        model_cache=_model(),
        volume=_volume(),
        provider_inventory=_inventory(),
        expected_source_commit="d" * 40,
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        reservation_seconds=DEFAULT_RESERVATION_SECONDS,
        initial_spent_usd=14.708611,
        initial_gpu_seconds=15_785,
        carrier_volume_admission=_carrier(),
    )

    assert result["status"] == "blocked"
    assert "serverless_release_source_commit_mismatch" in result["blockers"]


def test_admission_blocks_rate_above_selected_gpu_ceiling() -> None:
    result = validate_serverless_inputs(
        release=_release(),
        model_cache=_model(),
        volume=_volume(),
        provider_inventory=_inventory(),
        expected_source_commit="c" * 40,
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        reservation_seconds=DEFAULT_RESERVATION_SECONDS,
        initial_spent_usd=14.708611,
        initial_gpu_seconds=15_785,
        max_hourly_rate_usd=3.5,
        carrier_volume_admission=_carrier(),
    )

    assert result["status"] == "blocked"
    assert "serverless_campaign_hourly_rate_must_equal_gpu_ceiling" in result["blockers"]


def test_inventory_joins_endpoints_templates_and_pods(monkeypatch) -> None:
    def fake_call(method, path, body, *, key, timeout):
        assert method == "GET"
        values = {
            "/endpoints": [{"name": "blueprint-groot-oscar-serverless-test-endpoint"}],
            "/templates": [{"name": "blueprint-groot-oscar-serverless-test-template"}],
            "/pods": [{"name": "unrelated"}],
        }
        return 200, values[path]

    monkeypatch.setattr("blueprint_pipeline.groot_oscar_runpod_serverless._runpod_call", fake_call)
    result = collect_inventory(
        api_key="private",
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
    )

    assert result["api_confirmed"] is True
    assert result["matching_compute_count"] == 1
    assert result["matching_template_count"] == 1
    assert "private" not in str(result)


def test_active_worker_runs_all_phases_then_retrieves_and_tears_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = tmp_path / "release.json"
    model = tmp_path / "model.json"
    handoff = tmp_path / "handoff.json"
    carrier = tmp_path / "carrier.json"
    api_key = tmp_path / "runpod_api_key"
    release.write_text(json.dumps(_release()), encoding="utf-8")
    model.write_text(json.dumps(_model()), encoding="utf-8")
    carrier.write_text(json.dumps(_carrier()), encoding="utf-8")
    handoff.write_text(
        json.dumps(
            {
                "provider_lane_handoff": {
                    "binding": {
                        "provider": "runpod",
                        "lane": "groot_oscar_model_volume",
                        "volume_id": "volume-1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    api_key.write_text("private", encoding="utf-8")
    api_key.chmod(0o600)
    calls = []
    endpoint_create_succeeds = [True]

    def fake_provider(method, path, payload, *, key, timeout):
        del payload, timeout
        assert key == "private"
        calls.append((method, path))
        if path == "/networkvolumes/volume-1":
            return 200, {"id": "volume-1", "dataCenterId": "EUR-IS-1"}
        if method == "GET":
            return 200, []
        if path == "/templates":
            return 201, {"id": "template-1"}
        if path == "/endpoints":
            return (201, {"id": "endpoint-1"}) if endpoint_create_succeeds[0] else (503, {})
        raise AssertionError((method, path))

    monkeypatch.setattr(serverless, "_runpod_call", fake_provider)
    monkeypatch.setattr(
        serverless,
        "validate_campaign_io_evidence",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "blockers": [],
            "campaign_manifest_relative_path": ".blueprint-campaigns/test/input/campaign.json",
            "campaign_manifest_sha256": "f" * 64,
            "output_relative_path": ".blueprint-campaigns/test/output/results",
        },
    )
    monkeypatch.setattr(
        serverless,
        "stage_campaign_inputs",
        lambda *_args, **_kwargs: {"status": "completed", "uploaded_file_count": 6},
    )
    monkeypatch.setattr(
        serverless,
        "retrieve_campaign_outputs",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "transfer_status": "completed",
        },
    )
    monkeypatch.setattr(
        serverless,
        "cleanup_campaign_storage",
        lambda *_args, **_kwargs: {"status": "completed", "deleted_file_count": 25},
    )
    monkeypatch.setattr(
        serverless,
        "open_pending_teardown",
        lambda **_kwargs: {"path": str(tmp_path / "pending.json")},
    )
    monkeypatch.setattr(serverless, "bind_pending_teardown_instance", lambda *_args: None)
    monkeypatch.setattr(
        serverless,
        "_arm_watchdog",
        lambda **_kwargs: (
            SimpleNamespace(pid=12345),
            {"status": "armed", "pid": 12345},
        ),
    )
    monkeypatch.setattr(
        serverless,
        "accept_paid_provider_lane_lease_handoff",
        lambda *_args, **_kwargs: {"status": "accepted", "blockers": []},
    )
    submitted = []

    def fake_submit(*, operation, job_input=None, **_kwargs):
        submitted.append((operation, job_input))
        return 201, {"id": f"job-{operation}"}

    monkeypatch.setattr(serverless, "_submit_job", fake_submit)

    def fake_poll(*, operation_label, **_kwargs):
        outputs = {
            "startup": {
                "status": "completed",
                "runtime_present": True,
                "gpu_name": "NVIDIA A40",
            },
            "strict-policy-smoke": {
                "status": "completed",
                "completed_action_count": 3,
                "model_execution_proven": True,
                "runtime_worker_identity_sha256": "1" * 64,
            },
            "kitchen-campaign": {
                "status": "completed",
                "smoke_passed": True,
                "all_dynamic_episodes_completed": True,
                "runs": [
                    {"attempt_id": row}
                    for row in ("smoke", "episode_001", "episode_002", "episode_003")
                ],
                "semantic_task_success_by_attempt": {
                    "smoke": False,
                    "episode_001": False,
                    "episode_002": False,
                    "episode_003": False,
                },
            },
        }
        return {"status": "COMPLETED", "output": outputs[operation_label]}

    monkeypatch.setattr(serverless, "_poll_job", fake_poll)
    monkeypatch.setattr(
        serverless,
        "_request_teardown",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "provider_absence": {"billing_compute_stopped": True},
            "campaign_budget_settlement": {"status": "settled"},
        },
    )

    result = serverless.run_active_worker(
        output_dir=tmp_path / "out",
        release_evidence=release,
        model_cache_evidence=model,
        watchdog_handoff_evidence=handoff,
        api_key_file=api_key,
        campaign_io_evidence=tmp_path / "campaign_io.json",
        runpod_s3_access_key_file="unused",
        runpod_s3_secret_key_file="unused",
        resource_name_prefix="blueprint-groot-oscar-serverless-test-",
        expected_source_commit="c" * 40,
        execute=True,
        campaign_budget_ledger=tmp_path / "budget.json",
        initial_spent_usd=14.708611,
        initial_gpu_seconds=15_785,
        carrier_volume_admission=carrier,
    )

    assert result["status"] == "completed", result["blockers"]
    assert result["structural_campaign_completed"] is True
    assert result["semantic_task_success_proven"] is False
    assert submitted == [
        ("startup", None),
        ("strict-policy-smoke", None),
        (
            "kitchen-campaign",
            {
                "campaign_manifest_relative_path": ".blueprint-campaigns/test/input/campaign.json",
                "campaign_manifest_sha256": "f" * 64,
                "output_relative_path": ".blueprint-campaigns/test/output/results",
                "expected_runtime_worker_identity_sha256": "1" * 64,
            },
        ),
    ]
    state = json.loads((tmp_path / "out" / "watchdog_state.json").read_text())
    assert state["campaign_budget"]["max_hourly_rate_usd"] == 1.22
    assert isinstance(state["endpoint_allocated_at_epoch"], float)
    request_shapes = json.loads((tmp_path / "out" / "serverless_request_shapes.json").read_text())
    assert request_shapes["template"]["name"].startswith("blueprint-groot-oscar-serverless-test-")
    assert request_shapes["endpoint"]["name"].startswith("blueprint-groot-oscar-serverless-test-")
    assert calls.count(("POST", "/endpoints")) == 1

    endpoint_create_succeeds[0] = False
    failed = serverless.run_active_worker(
        output_dir=tmp_path / "endpoint-failed",
        release_evidence=release,
        model_cache_evidence=model,
        watchdog_handoff_evidence=handoff,
        api_key_file=api_key,
        campaign_io_evidence=tmp_path / "campaign_io.json",
        runpod_s3_access_key_file="unused",
        runpod_s3_secret_key_file="unused",
        resource_name_prefix="blueprint-groot-oscar-serverless-endpoint-failed-",
        expected_source_commit="c" * 40,
        execute=True,
        campaign_budget_ledger=tmp_path / "failed-budget.json",
        initial_spent_usd=14.708611,
        initial_gpu_seconds=15_785,
        carrier_volume_admission=carrier,
    )
    assert failed["status"] == "blocked"
    assert failed["blockers"] == ["serverless_endpoint_create_failed_or_ambiguous"]
    failed_state = json.loads((tmp_path / "endpoint-failed" / "watchdog_state.json").read_text())
    assert "endpoint_allocated_at_epoch" not in failed_state
    assert isinstance(failed_state["endpoint_create_requested_at_epoch"], float)


def test_model_volume_handoff_binding_requires_exact_volume_and_lane() -> None:
    exact = {
        "provider": "runpod",
        "lane": "groot_oscar_model_volume",
        "volume_id": "volume-1",
    }
    assert serverless.validate_model_volume_handoff_binding(exact, volume_id="volume-1") == []
    assert set(
        serverless.validate_model_volume_handoff_binding(
            {"provider": "other", "lane": "wrong", "volume_id": "volume-2"},
            volume_id="volume-1",
        )
    ) == {
        "serverless_model_volume_handoff_provider_mismatch",
        "serverless_model_volume_handoff_lane_mismatch",
        "serverless_model_volume_handoff_volume_mismatch",
    }
