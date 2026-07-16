from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import blueprint_pipeline.groot_oscar_runpod_persistent_carrier_campaign as campaign
from blueprint_pipeline.groot_oscar_runpod_persistent_carrier import (
    PERSISTENT_CARRIER_IMAGE_REF,
)
from blueprint_pipeline.groot_oscar_runpod_carrier_volume import (
    CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_RUNTIME_ARCHIVE_PATH,
    DEFAULT_RUNTIME_MANIFEST_PATH,
    DEFAULT_RUNTIME_ROOT,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
)


RELEASE_REF = "docker.io/blueprint/release@sha256:" + "1" * 64
CARRIER_REF = PERSISTENT_CARRIER_IMAGE_REF


def _carrier() -> dict:
    return {
        "schema_version": CARRIER_VOLUME_ADMISSION_SCHEMA_VERSION,
        "status": "verified",
        "carrier_image_ref": CARRIER_REF,
        "network_volume": {
            "id": "volume123",
            "data_center_id": "EUR-IS-1",
            "size_gib": 120,
        },
        "runtime_bundle": {
            "manifest_schema_version": RUNTIME_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "source_release_image_ref": RELEASE_REF,
            "root": DEFAULT_RUNTIME_ROOT,
            "archive_path": DEFAULT_RUNTIME_ARCHIVE_PATH,
            "manifest_path": DEFAULT_RUNTIME_MANIFEST_PATH,
            "archive_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
        },
        "model_cache": {
            "status": "verified",
            "root": DEFAULT_MODEL_CACHE_ROOT,
            "manifest_sha256": "5" * 64,
        },
        "s3_transfer_verification": {
            "upload_completed": True,
            "full_redownload_sha256_verified": True,
            "provider_volume_id": "volume123",
            "data_center_id": "EUR-IS-1",
        },
    }


def _release() -> dict:
    return {
        "resolved_digest_ref": RELEASE_REF,
        "thin_release_contract": {"status": "passed"},
        "runnable_platform": "linux/amd64",
        "required_cuda_version": "12.8",
        "required_cuda_version_source": (
            "image_config_env:BLUEPRINT_REQUIRED_CUDA_VERSION"
        ),
    }


def _model() -> dict:
    return {
        "schema_version": "groot_oscar_external_model_cache_verification.v2",
        "status": "passed",
        "model_manifest_digest": "sha256:" + "6" * 64,
        "cache_root": DEFAULT_MODEL_CACHE_ROOT,
        "provider_volume_id": "volume123",
        "verified_size_bytes": 10 * 1024**3,
        "checks": {"models_cached_offline": True},
    }


def _preflight() -> dict:
    return {
        "status": "verified",
        "volume": {
            "provider": "runpod",
            "provider_api_verified": True,
            "id": "volume123",
            "data_center_id": "EUR-IS-1",
            "size_bytes": 120 * 1024**3,
            "model_cache_path": DEFAULT_MODEL_CACHE_ROOT,
        },
        "runtime": {
            "provider": "runpod",
            "data_center_id": "EUR-IS-1",
            "capacity_data_center_id": "EUR-IS-1",
            "gpu_type_id": "NVIDIA A40",
            "provider_api_verified": True,
            "capacity_confidence": "advisory",
            "single_gpu_available": True,
            "required_cuda_version": "12.8",
            "allowed_cuda_versions": ["12.8"],
            "capacity_allowed_cuda_versions": ["12.8"],
            "warm_worker_only": True,
            "provider_inventory_verified_zero": True,
            "on_demand_price_usd_per_hour": 0.74,
        },
        "spend": {
            "paid_mutation_authorized": True,
            "max_spend_usd": 4.0,
            "hard_ttl_seconds": 18_600,
            "one_resource_limit": True,
            "independent_teardown_watchdog": True,
            "watchdog_armed_before_allocation": True,
        },
    }


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    preflight = _preflight()
    watchdog = tmp_path / "watchdog"
    preflight["spend"].update(
        {
            "watchdog_pod_name_prefix": "blueprint-persistent-",
            "watchdog_out_dir": str(watchdog.resolve()),
            "watchdog_deadline_epoch": time.time() + 18_599,
        }
    )
    preflight["model_volume_watchdog_handoff"] = {
        "provider_lane_handoff": {
            "status": "pending_canary_acceptance",
            "binding": {"volume_id": "volume123"},
        }
    }
    return {
        "provider_launch_request": _write(
            tmp_path / "request.json", {"provider_request_shape": {}}
        ),
        "release_evidence": _write(tmp_path / "release.json", _release()),
        "model_cache_evidence": _write(tmp_path / "model.json", _model()),
        "preflight_bundle": _write(tmp_path / "preflight.json", preflight),
        "carrier_volume_admission": _write(tmp_path / "carrier.json", _carrier()),
        "policy_observation_path": tmp_path / "observation.json",
        "persistent_job_dir": tmp_path / "job",
        "admission_out": tmp_path / "admission.json",
        "bound_request_out": tmp_path / "bound.json",
        "adapter_output": tmp_path / "result.json",
    }


def _budget(tmp_path: Path) -> dict:
    return {
        "ledger_path": str(tmp_path / "budget.json"),
        "initial_spent_usd": 14.557003,
        "initial_used_gpu_seconds": 15_624,
        "total_spend_cap_usd": 20.0,
        "combined_gpu_wall_cap_seconds": 36_000,
        "reservation_gpu_seconds": 18_600,
        "campaign_stage": "persistent_carrier_campaign",
        "maximum_canary_reservation_gpu_seconds": 18_600,
        "future_campaign_allowance_gpu_seconds": 0,
        "maximum_future_campaign_allowance_gpu_seconds": 0,
        "maximum_combined_plan_gpu_seconds": 18_600,
        "reduced_canary_timeout_acknowledged": True,
        "max_hourly_rate_usd": 0.74,
        "minimum_reconciled_spend_usd": 14.557003,
        "minimum_reconciled_gpu_seconds": 15_624,
    }


def test_persistent_campaign_dry_run_is_provider_pure(tmp_path: Path) -> None:
    result = campaign.run_persistent_carrier_campaign(
        **_inputs(tmp_path),
        pod_name="blueprint-persistent-dry",
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    bound = json.loads((tmp_path / "bound.json").read_text(encoding="utf-8"))
    assert bound["provider_request_shape"]["persistent_campaign"] == {
        "policy_call_count": 5,
        "learned_wam_generation_count": 4,
        "same_pod_required": True,
        "provider_output_replay_disallowed": True,
        "max_wait_seconds": 18_000,
    }


def test_persistent_campaign_executes_exact_loop_and_requires_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)

    class Provider:
        def _key(self) -> str:
            return "test-key"

        def capacity_preflight(self, _runtime: dict) -> dict:
            raise AssertionError("refresh is stubbed")

        def billable_inventory(self, *, name_prefix: str) -> dict:
            raise AssertionError(name_prefix)

    monkeypatch.setattr(campaign, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        campaign,
        "refresh_runpod_preflight",
        lambda **kwargs: dict(kwargs["preflight"]),
    )
    monkeypatch.setattr(
        campaign,
        "accept_paid_provider_lane_lease_handoff",
        lambda *args, **kwargs: {"status": "accepted", "lease": "test"},
    )
    monkeypatch.setattr(
        campaign,
        "restore_paid_provider_lane_lease_to_retained_watchdog",
        lambda _acceptance: {"status": "restored", "restored": True},
    )
    observed: dict = {}

    def fake_session(**kwargs):
        observed.update(kwargs)
        imported = Path(kwargs["job_dir"]) / "imported_persistent_session_output"
        for index in range(5):
            _write(
                imported / "policy_calls" / f"policy_call_{index:04d}.json",
                {
                    "status": "completed",
                    "action": {"action_chunk": [index, index + 0.5]},
                    "provider_output_replay_used": False,
                },
            )
        for index in range(1, 5):
            _write(
                imported / "wam_calls" / f"wam_call_{index:04d}.json",
                {"status": "completed", "structural_fallback_used": False},
            )
        media = imported / "review_media"
        media.mkdir(parents=True)
        for index in range(21):
            (media / f"media_{index:04d}.png").write_bytes(b"png")
        poll = (
            Path(kwargs["job_dir"])
            / "runpod_persistent_session_run/runpod_wam_async_poll_manifest.json"
        )
        _write(
            poll,
            {
                "teardown_performed": True,
                "continuing_spend_from_this_run": False,
            },
        )
        return {
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 5,
            "generated_next_observation_count": 4,
            "live_wam_generation_success_count": 4,
            "learned_wam_model_success_count": 4,
            "provider_output_replay_used": False,
            "provider_output_resume_used": False,
            "imported_provider_output_dir": str(imported),
        }, 0

    result = campaign.run_persistent_carrier_campaign(
        **inputs,
        pod_name="blueprint-persistent-exact",
        execute=True,
        campaign_budget=_budget(tmp_path),
        session_runner=fake_session,
    )

    assert result["status"] == "completed"
    assert result["gpu_teardown_verified"] is True
    assert result["semantic_task_success_proven"] is False
    assert result["persistent_carrier_output_audit"]["observed_counts"] == {
        "repeated_policy_calls_count": 5,
        "generated_next_observation_count": 4,
        "live_wam_generation_success_count": 4,
        "learned_wam_model_success_count": 4,
        "policy_artifact_count": 5,
        "distinct_action_count": 5,
        "wam_artifact_count": 4,
        "media_file_count": 21,
    }
    assert observed["loop_step_count"] == 5
    assert observed["max_wait_seconds"] == 18_000
    assert observed["use_live_wam"] is True
    assert observed["allow_structural_wam_fallback"] is False
    assert observed["pod_name"] == "blueprint-persistent-exact"
    assert observed["provider_lane_handoff_receipt_path"] == (
        tmp_path / "watchdog" / "provider_lane_handoff_receipt.json"
    )
    assert not observed["provider_lane_handoff_receipt_path"].exists()


def test_nonterminal_pod_leaves_watchdog_receipt_and_budget_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)

    class Provider:
        def _key(self) -> str:
            return "test-key"

        def capacity_preflight(self, _runtime: dict) -> dict:
            raise AssertionError("refresh is stubbed")

        def billable_inventory(self, *, name_prefix: str) -> dict:
            raise AssertionError(name_prefix)

    monkeypatch.setattr(campaign, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        campaign,
        "refresh_runpod_preflight",
        lambda **kwargs: dict(kwargs["preflight"]),
    )
    monkeypatch.setattr(
        campaign,
        "accept_paid_provider_lane_lease_handoff",
        lambda *args, **kwargs: {"status": "accepted", "lease": "test"},
    )
    restore_calls: list[dict] = []
    monkeypatch.setattr(
        campaign,
        "restore_paid_provider_lane_lease_to_retained_watchdog",
        lambda acceptance: restore_calls.append(dict(acceptance))
        or {"status": "restored", "restored": True},
    )

    def fake_session(**kwargs):
        receipt_path = Path(kwargs["provider_lane_handoff_receipt_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update(
            {
                "pre_provider_mutation_confirmed_absent": False,
                "pod_pending_teardown_record": str(tmp_path / "pending.json"),
                "pod_id": "pod-still-live",
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_path.chmod(0o600)
        poll = (
            Path(kwargs["job_dir"])
            / "runpod_persistent_session_run/runpod_wam_async_poll_manifest.json"
        )
        _write(
            poll,
            {
                "teardown_performed": False,
                "continuing_spend_from_this_run": True,
            },
        )
        return {"status": "blocked"}, 2

    result = campaign.run_persistent_carrier_campaign(
        **inputs,
        pod_name="blueprint-persistent-nonterminal",
        execute=True,
        campaign_budget=_budget(tmp_path),
        session_runner=fake_session,
    )

    receipt_path = tmp_path / "watchdog" / "provider_lane_handoff_receipt.json"
    retained = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["campaign_budget_settlement"]["status"] == (
        "watchdog_retains_open_reservation"
    )
    assert result["provider_lane_owner_return"]["status"] == (
        "watchdog_retains_control"
    )
    assert restore_calls == []
    assert retained["campaign_budget"]["status"] == "reserved"
    assert retained["pod_pending_teardown_record"] == str(tmp_path / "pending.json")
    budget_snapshot = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert budget_snapshot["reservations"][0]["status"] == "open"


def test_handoff_rejection_settles_zero_gpu_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)

    class Provider:
        def _key(self) -> str:
            return "test-key"

        def capacity_preflight(self, _runtime: dict) -> dict:
            raise AssertionError("refresh is stubbed")

        def billable_inventory(self, *, name_prefix: str) -> dict:
            raise AssertionError(name_prefix)

    monkeypatch.setattr(campaign, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        campaign,
        "refresh_runpod_preflight",
        lambda **kwargs: dict(kwargs["preflight"]),
    )
    monkeypatch.setattr(
        campaign,
        "accept_paid_provider_lane_lease_handoff",
        lambda *args, **kwargs: {"status": "blocked", "blockers": ["rejected"]},
    )

    result = campaign.run_persistent_carrier_campaign(
        **inputs,
        pod_name="blueprint-persistent-rejected",
        execute=True,
        campaign_budget=_budget(tmp_path),
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["campaign_budget_settlement"]["charged_gpu_seconds"] == 0
