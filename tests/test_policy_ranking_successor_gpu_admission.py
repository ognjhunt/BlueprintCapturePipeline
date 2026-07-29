from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import policy_ranking_successor_gpu_admission as admission
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "docs/experiments/policy_ranking_cosmos3_followup_20260728"
HISTORICAL_EXPERIMENT = ROOT / "docs/experiments/policy_ranking_successor_experiment_20260727"
BUNDLE_NAME = "cosmos3_followup_provider_bundle.zip"
BUNDLE_RECEIPT_NAME = "cosmos3_followup_provider_bundle_receipt.json"


def _load(name: str) -> dict[str, Any]:
    path = EXPERIMENT / name
    if not path.is_file():
        path = HISTORICAL_EXPERIMENT / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if name == "vast_compute_preflight.json":
        value["experiment_id"] = "policy_ranking_cosmos3_followup_20260728"
    return value


def _inspect_bundle(path: Path | None = None) -> dict[str, Any]:
    return admission.inspect_successor_bundle(
        path or EXPERIMENT / BUNDLE_NAME,
        receipt=_load(BUNDLE_RECEIPT_NAME),
        smoke_inventory=_load("smoke_request_inventory.json"),
    )


def _preflight_path(tmp_path: Path) -> Path:
    path = tmp_path / "vast_compute_preflight.json"
    path.write_text(json.dumps(_load("vast_compute_preflight.json")), encoding="utf-8")
    return path


def test_frozen_successor_bundle_passes_integrity_inspection() -> None:
    result = _inspect_bundle()

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["bundle_sha256"] == (
        "0e938e1674ff2efc043363ab9b7e2724ae2f9bc264e289895d7759a9eb8173fd"
    )
    assert result["manifest"]["qualification_canary_request_count"] == 2
    assert result["manifest"]["scientific_matrix_request_count"] == 10
    assert result["manifest"]["total_initial_generation_request_count"] == 12


def test_successor_gpu_admission_requires_explicit_authorization() -> None:
    result = admission.build_successor_gpu_admission(
        authorization={},
        environment=_load("environment_and_source_manifest.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=_load("vast_compute_preflight.json"),
        bundle_inspection=_inspect_bundle(),
        expected_source_commit="a" * 40,
        execute=False,
    )

    assert result["status"] == "blocked"
    assert "successor_compute_not_explicitly_authorized" in result["blockers"]
    assert result["provider_mutations_performed"] == 0


def test_successor_gpu_admission_accepts_only_frozen_rtx_envelope() -> None:
    preflight = _load("vast_compute_preflight.json")
    result = admission.build_successor_gpu_admission(
        authorization=_load("compute_authorization_allocation_2.json"),
        environment=_load("environment_and_source_manifest.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=preflight,
        bundle_inspection=_inspect_bundle(),
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "admitted"
    assert result["blockers"] == []
    assert result["limits"]["hard_cap_usd"] == 6.0
    assert result["limits"]["allowed_gpu_keywords"] == ["RTX PRO 6000"]
    assert result["shared_paid_lane_admission"]["status"] == "admitted"
    assert result["request_budget"]["total_initial_generation_request_count"] == 12


def test_successor_gpu_admission_accepts_signed_allocation_3_retry() -> None:
    preflight = _load("vast_compute_preflight.json")
    result = admission.build_successor_gpu_admission(
        authorization=_load("compute_authorization_allocation_3.json"),
        environment=_load("environment_and_source_manifest.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=preflight,
        bundle_inspection=_inspect_bundle(),
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "admitted"
    assert result["authorization"]["status"] == "accepted"
    assert result["blockers"] == []


def test_phase_b_profile_reuses_admission_with_its_own_frozen_limits() -> None:
    profile = admission.PHASE_B_PROFILE
    environment = _load("environment_and_source_manifest.json")
    environment["experiment_id"] = profile.experiment_id
    preflight = _load("vast_compute_preflight.json")
    preflight.update(
        {
            "schema_version": profile.preflight_schema,
            "experiment_id": profile.experiment_id,
            "status": "verified",
            "provider": "vast",
            "provider_inventory_verified_zero": True,
            "provider_mutations_performed": 0,
        }
    )
    authorization = {
        "schema_version": profile.authorization_schema,
        "experiment_id": profile.experiment_id,
        "authorization_id": profile.authorization_ids_by_allocation_index[1],
        "allocation_index": 1,
        "maximum_provider_allocations": 1,
        "single_use_consumption_required": True,
        "paid_mutation_authorized": True,
        "authorized_compute_cap_usd": profile.max_compute_cap_usd,
        "per_allocation_maximum_spend_required": True,
        "prior_cumulative_compute_cap_superseded": True,
        "goal_cost_authorization_amendment_sha256": (profile.cost_authorization_binding_sha256),
        "hard_ttl_seconds": profile.hard_ttl_seconds,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "automatic_spend_cutoff": True,
        "teardown_required": True,
        "provider_zero_verification_required": True,
        "physical_robot_endpoint_access_allowed": False,
    }
    result = admission.build_successor_gpu_admission(
        authorization=authorization,
        environment=environment,
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=preflight,
        bundle_inspection={"status": "passed", "blockers": [], "bundle_sha256": "a" * 64},
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
        profile=profile,
    )

    assert result["status"] == "admitted"
    assert result["experiment_id"] == profile.experiment_id
    assert result["limits"]["hard_cap_usd"] == 5.0
    assert result["limits"]["hard_ttl_seconds"] == 7_200
    assert result["request_budget"]["amendment_sha256"] is None


def test_phase_b_vast_preflight_explicitly_admits_blackwell_compute_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Provider:
        def capacity_preflight(self, request: dict[str, Any]) -> dict[str, Any]:
            captured.update(request)
            return {
                "status": "available",
                "viable_gpu_types": [
                    {
                        "ask_contract_id": 1,
                        "gpu_name": "RTX PRO 6000 WS",
                        "num_gpus": 1,
                        "gpu_ram_mb": 97_887,
                        "hourly_rate_usd": 0.88,
                        "reliability": 0.99,
                    }
                ],
            }

        def billable_inventory(self, *, name_prefix: str) -> dict[str, Any]:
            return {"api_confirmed": True, "live_resource_count": 0}

    monkeypatch.setattr(admission, "get_render_provider", lambda provider: Provider())
    result = admission.collect_successor_vast_preflight(name_prefix="phase-b-")

    assert result["status"] == "verified", result["blockers"]
    assert captured["min_compute_cap"] == 1200
    assert captured["max_compute_cap"] == 0
    assert captured["prefer_isaac_rt"] is False


def test_droid_reference_runpod_preflight_selects_compatible_secure_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Provider:
        def capacity_preflight(self, request: dict[str, Any]) -> dict[str, Any]:
            captured.update(request)
            return {
                "status": "available",
                "viable_gpu_types": [
                    {
                        "gpu_type_id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "display_name": "RTX PRO 6000 Blackwell",
                        "memory_in_gb": 96,
                        "cloud_type": "SECURE",
                        "capacity_confidence": "advisory",
                        "on_demand_price_usd_per_hour": 1.99,
                    }
                ],
            }

        def billable_inventory(self, *, name_prefix: str) -> dict[str, Any]:
            return {"api_confirmed": True, "live_resource_count": 0}

    monkeypatch.setattr(admission, "get_render_provider", lambda provider: Provider())

    result = admission.collect_successor_runpod_preflight(
        name_prefix="blueprint-groot-oscar-canary-droid-reference-"
    )

    assert result["status"] == "verified", result
    assert result["provider"] == "runpod"
    assert result["selected_offer"]["gpu_ram_mb"] == 96_000
    assert result["selected_offer"]["hourly_rate_usd"] == 1.99
    assert captured["cloudType"] == "SECURE"
    assert captured["requires_rtx"] is False


@pytest.mark.parametrize("allocation_index", [1, 2, 3, 4])
def test_droid_reference_admission_accepts_runpod_without_smoke_inventory(
    allocation_index: int,
) -> None:
    profile = admission.DROID_REFERENCE_PROFILE
    environment = {
        "experiment_id": profile.experiment_id,
        "upstream_source": {
            "cosmos": {"revision": profile.cosmos_revision},
            "cosmos_framework": {"revision": profile.cosmos_framework_revision},
            "vllm_omni": {
                "revision": profile.vllm_omni_revision,
                "runtime_image": admission.PUBLIC_IMAGE,
            },
        },
        "checkpoint": {
            "repository": admission.CHECKPOINT_REPOSITORY,
            "revision": admission.CHECKPOINT_REVISION,
            "remote_code_policy": "no_unpinned_remote_code_and_trust_remote_code_false",
        },
    }
    authorization = {
        "schema_version": profile.authorization_schema,
        "experiment_id": profile.experiment_id,
        "authorization_id": profile.authorization_ids_by_allocation_index[allocation_index],
        "allocation_index": allocation_index,
        "maximum_provider_allocations": 1,
        "single_use_consumption_required": True,
        "paid_mutation_authorized": True,
        "authorized_compute_cap_usd": profile.max_compute_cap_usd,
        "per_allocation_maximum_spend_required": True,
        "prior_cumulative_compute_cap_superseded": True,
        "goal_cost_authorization_amendment_sha256": (profile.cost_authorization_binding_sha256),
        "hard_ttl_seconds": profile.hard_ttl_seconds,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "automatic_spend_cutoff": True,
        "teardown_required": True,
        "provider_zero_verification_required": True,
        "physical_robot_endpoint_access_allowed": False,
    }
    observed = 1234.0
    preflight = {
        "schema_version": profile.preflight_schema,
        "experiment_id": profile.experiment_id,
        "status": "verified",
        "provider": "runpod",
        "provider_inventory_verified_zero": True,
        "provider_mutations_performed": 0,
        "observed_at_epoch": observed,
        "selected_offer": {
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_ram_mb": 96_000,
            "hourly_rate_usd": 1.99,
            "cloud_type": "SECURE",
            "capacity_confidence": "advisory",
        },
    }

    result = admission.build_successor_gpu_admission(
        authorization=authorization,
        environment=environment,
        smoke_inventory={},
        provider_preflight=preflight,
        bundle_inspection={"status": "passed", "blockers": [], "bundle_sha256": "a" * 64},
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=observed + 1,
        profile=profile,
    )

    assert result["status"] == "admitted", result["blockers"]
    assert result["provider"] == "runpod"
    assert result["smoke_inventory_validation"]["status"] == "not_applicable"
    assert result["shared_paid_lane_admission"]["resource_class"] == "runpod_wam_async"


def test_droid_reference_runpod_executor_binds_watchdog_public_model_and_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    watchdog_captured: dict[str, Any] = {}
    poll_captured: dict[str, Any] = {}
    process = object()
    monkeypatch.delenv(admission.RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV, raising=False)

    def fake_arm(**kwargs: Any):
        watchdog_captured.update(kwargs)
        return (
            {"status": "armed", "independent_process": True},
            process,
            tmp_path / "watchdog",
        )

    monkeypatch.setattr(admission, "_arm_runpod_successor_watchdog", fake_arm)

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        assert admission.os.environ[admission.RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV] == "7200"
        assert kwargs["pre_provider_mutation_hook"]()["status"] == "consumed"
        (tmp_path / "job").mkdir(exist_ok=True)
        (tmp_path / "job" / "runpod_wam_async_state.json").write_text(
            json.dumps({"pod_id": "pod-123", "created_at_epoch": 1000.0}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    monkeypatch.setattr(admission, "create_runpod_wam_async_run", fake_create)

    def fake_poll(**kwargs: Any) -> dict[str, Any]:
        poll_captured.update(kwargs)
        return {
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "blockers": [],
        }

    monkeypatch.setattr(admission, "poll_runpod_wam_async_run", fake_poll)

    class Provider:
        def billable_inventory(self, *, name_prefix: str) -> dict[str, Any]:
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "name_prefix": name_prefix,
            }

    monkeypatch.setattr(admission, "get_render_provider", lambda _provider: Provider())
    monkeypatch.setattr(
        admission,
        "_close_runpod_watchdog_after_provider_zero",
        lambda **_kwargs: {
            "status": "provider_terminal",
            "provider_absence_confirmed": True,
        },
    )
    monkeypatch.setattr(admission.time, "time", lambda: 1060.0)

    result = admission._run_successor_runpod(
        job_dir=tmp_path / "job",
        provider_bundle_path=tmp_path / "bundle.zip",
        public_base_url=None,
        token_file=None,
        secret_env_file=None,
        provider_bundle_url_file=tmp_path / "bundle-url",
        provider_output_put_url_file=tmp_path / "output-put-url",
        provider_output_get_url_file=tmp_path / "output-get-url",
        output_path=tmp_path / "output.zip",
        profile=admission.DROID_REFERENCE_PROFILE,
        selected_offer={
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "hourly_rate_usd": 1.99,
        },
        session_max_live_minutes=132,
        paid_resource_admission_grant=object(),
        pre_provider_mutation_hook=lambda: {"status": "consumed"},
    )

    assert result["status"] == "completed"
    assert result["provider_zero_verified"] is True
    assert result["estimated_gpu_cost_usd"] == pytest.approx(1.99 / 60.0)
    assert captured["gpu_type_ids"] == ("NVIDIA RTX PRO 6000 Blackwell Server Edition",)
    assert captured["forward_model_secret_env"] is False
    assert captured["cloud_type"] == "SECURE"
    assert captured["pre_provider_mutation_hook"] is not None
    assert admission.RUNPOD_WAM_TERMINAL_HOLD_SECONDS_ENV not in admission.os.environ
    assert watchdog_captured["deadline_epoch"] == 1060.0 + 7200
    assert poll_captured["max_wait_seconds"] == 7200


def test_successor_gpu_admission_rejects_mismatched_retry_identity() -> None:
    preflight = _load("vast_compute_preflight.json")
    authorization = _load("compute_authorization_allocation_3.json")
    authorization["allocation_index"] = 2
    result = admission.build_successor_gpu_admission(
        authorization=authorization,
        environment=_load("environment_and_source_manifest.json"),
        smoke_inventory=_load("smoke_request_inventory.json"),
        provider_preflight=preflight,
        bundle_inspection=_inspect_bundle(),
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "blocked"
    assert "successor_compute_authorization_id_invalid" in result["blockers"]


def test_successor_gpu_lane_passes_opaque_grant_and_hardware_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        consumption = kwargs["pre_provider_mutation_hook"]()
        if consumption["status"] != "consumed":
            return {
                "status": "blocked",
                "blockers": consumption.get("blockers") or [],
            }
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(admission, "run_vast_wam_authorized_runner", fake_runner)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", " YES ")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "1")
    monkeypatch.setattr(
        admission,
        "AUTHORIZATION_CONSUMPTION_ROOT",
        tmp_path / "authority-consumption",
    )
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps({"attempts": [{"actual_live_runtime_seconds_observed_by_adapter": 97.485577}]}),
        encoding="utf-8",
    )
    preflight = _load("vast_compute_preflight.json")
    monkeypatch.setattr(admission.time, "time", lambda: float(preflight["observed_at_epoch"]) + 1)
    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=budget,
        expected_source_commit="c" * 40,
        execute=True,
        provider_bundle_url_file=tmp_path / "bundle-url",
        provider_output_put_url_file=tmp_path / "output-put-url",
        provider_output_get_url_file=tmp_path / "output-get-url",
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "completed", result
    assert isinstance(captured["paid_resource_admission_grant"], PaidResourceAdmissionGrant)
    assert captured["hard_cap_usd"] == 6.0
    assert captured["target_spend_usd"] == 3.25
    assert captured["max_live_minutes"] == 180
    assert captured["session_max_live_minutes"] == 182
    assert captured["disk_gb"] == 250
    assert captured["min_gpu_ram_mb"] == 95_000
    assert captured["max_compute_cap"] == 0
    assert captured["gpu_selection_policy"]["allowed_gpu_keywords"] == ("RTX PRO 6000",)
    assert captured["require_independent_watchdog"] is True
    assert captured["provider_bundle_url_file"] == tmp_path / "bundle-url"
    assert captured["provider_output_put_url_file"] == tmp_path / "output-put-url"
    assert captured["provider_output_get_url_file"] == tmp_path / "output-get-url"
    assert result["authorization_consumption"]["status"] == "consumed"
    written_admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    assert written_admission["session_live_limit"]["prior_live_runtime_minutes_ceiling"] == 2
    assert written_admission["session_budget_preflight"]["status"] == "passed"

    second = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission-second.json",
        bound_request_out=tmp_path / "bound-second.json",
        adapter_output=tmp_path / "adapter-second.json",
        job_dir=tmp_path / "job-second",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output-second.zip",
        session_budget_ledger=budget,
        expected_source_commit="c" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )
    assert second["status"] == "blocked"
    assert second["blockers"] == ["successor_compute_authorization_already_consumed"]


def test_successor_lane_does_not_consume_authorization_when_staging_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption_root = tmp_path / "authority-consumption"
    monkeypatch.setattr(admission, "AUTHORIZATION_CONSUMPTION_ROOT", consumption_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        assert callable(kwargs["pre_provider_mutation_hook"])
        return {
            "status": "blocked",
            "blockers": ["provider_bundle_fetch_url_unreachable"],
            "provider_mutations_performed": 0,
        }

    monkeypatch.setattr(admission, "run_vast_wam_authorized_runner", fake_runner)
    budget = tmp_path / "budget.json"
    budget.write_text('{"attempts": []}', encoding="utf-8")
    preflight = _load("vast_compute_preflight.json")

    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://expired.example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=budget,
        expected_source_commit="d" * 40,
        execute=True,
        observed_now_epoch=float(preflight["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "blocked", result
    assert result["authorization_consumption"]["status"] == "not_consumed"
    assert result["blockers"] == ["provider_bundle_fetch_url_unreachable"]
    assert not consumption_root.exists()


def test_successor_lane_rechecks_preflight_age_before_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption_root = tmp_path / "authority-consumption"
    monkeypatch.setattr(admission, "AUTHORIZATION_CONSUMPTION_ROOT", consumption_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        consumption = kwargs["pre_provider_mutation_hook"]()
        return {
            "status": "blocked",
            "blockers": consumption.get("blockers") or [],
            "provider_mutations_performed": 0,
        }

    monkeypatch.setattr(admission, "run_vast_wam_authorized_runner", fake_runner)
    budget = tmp_path / "budget.json"
    budget.write_text('{"attempts": []}', encoding="utf-8")
    preflight = _load("vast_compute_preflight.json")
    observed = float(preflight["observed_at_epoch"])
    monkeypatch.setattr(
        admission.time,
        "time",
        lambda: observed + admission.MAX_PREFLIGHT_AGE_SECONDS + 1,
    )

    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=budget,
        expected_source_commit="e" * 40,
        execute=True,
        observed_now_epoch=observed + 1,
    )

    assert result["status"] == "blocked"
    assert result["authorization_consumption"]["status"] == "blocked"
    assert result["blockers"] == ["successor_vast_preflight_stale_or_future_at_provider_mutation"]
    assert not consumption_root.exists()


def test_authorization_publish_failure_leaves_no_consumed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption_root = tmp_path / "authority-consumption"
    monkeypatch.setattr(admission, "AUTHORIZATION_CONSUMPTION_ROOT", consumption_root)
    monkeypatch.setattr(
        admission.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link failed")),
    )

    result = admission._consume_authorization_once(
        _load("compute_authorization_allocation_2.json"),
        expected_source_commit="f" * 40,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["successor_authorization_consumption_write_failed"]
    assert list(consumption_root.iterdir()) == []


def test_successor_lane_checks_provider_env_before_consuming_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption_root = tmp_path / "authority-consumption"
    monkeypatch.setattr(
        admission,
        "AUTHORIZATION_CONSUMPTION_ROOT",
        consumption_root,
    )
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", raising=False)
    budget = tmp_path / "budget.json"
    budget.write_text('{"attempts": []}', encoding="utf-8")

    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=budget,
        expected_source_commit="f" * 40,
        execute=True,
        observed_now_epoch=float(_load("vast_compute_preflight.json")["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "blocked"
    assert result["authorization_consumed"] is False
    assert result["provider_mutations_performed"] == 0
    assert sorted(result["blockers"]) == [
        "missing_env_BLUEPRINT_ALLOW_VAST_API_CALLS",
        "missing_env_BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
    ]
    assert not consumption_root.exists()


def test_successor_lane_checks_session_budget_before_consuming_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption_root = tmp_path / "authority-consumption"
    monkeypatch.setattr(admission, "AUTHORIZATION_CONSUMPTION_ROOT", consumption_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setattr(
        admission,
        "run_vast_wam_authorized_runner",
        lambda **_kwargs: pytest.fail("runner must not start after a failed budget preflight"),
    )
    budget = tmp_path / "budget.json"
    budget.write_text("{bad json", encoding="utf-8")

    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=budget,
        expected_source_commit="f" * 40,
        execute=True,
        observed_now_epoch=float(_load("vast_compute_preflight.json")["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "blocked"
    assert "session_budget_ledger_parse_failed" in result["blockers"]
    assert not consumption_root.exists()


def test_successor_lane_requires_existing_session_budget_before_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumption_root = tmp_path / "authority-consumption"
    monkeypatch.setattr(admission, "AUTHORIZATION_CONSUMPTION_ROOT", consumption_root)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setattr(
        admission,
        "run_vast_wam_authorized_runner",
        lambda **_kwargs: pytest.fail("runner must not start without a session ledger"),
    )

    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=EXPERIMENT / "environment_and_source_manifest.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=tmp_path / "admission.json",
        bound_request_out=tmp_path / "bound.json",
        adapter_output=tmp_path / "adapter.json",
        job_dir=tmp_path / "job",
        public_base_url="https://example.test",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "output.zip",
        session_budget_ledger=tmp_path / "missing-budget.json",
        expected_source_commit="f" * 40,
        execute=True,
        observed_now_epoch=float(_load("vast_compute_preflight.json")["observed_at_epoch"]) + 1,
    )

    assert result["status"] == "blocked"
    assert "successor_session_budget_ledger_missing" in result["blockers"]
    assert not consumption_root.exists()


def test_successor_bundle_is_bound_to_receipt_and_embedded_inputs(
    tmp_path: Path,
) -> None:
    altered = tmp_path / "altered.zip"
    shutil.copyfile(EXPERIMENT / BUNDLE_NAME, altered)
    with zipfile.ZipFile(altered, "a") as archive:
        archive.writestr("provider_runtime/unregistered_marker.txt", "changed")

    result = _inspect_bundle(altered)

    assert result["status"] == "blocked"
    assert "successor_cosmos_provider_bundle_receipt_hash_mismatch" in result["blockers"]


def test_successor_bundle_requires_shared_crash_fallback_contract(
    tmp_path: Path,
) -> None:
    altered = tmp_path / "missing-crash-fallback.zip"
    with (
        zipfile.ZipFile(EXPERIMENT / BUNDLE_NAME) as source,
        zipfile.ZipFile(altered, "w") as target,
    ):
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "provider_runtime/run_wam_provider_runtime.sh":
                payload = payload.replace(b"write_missing_result", b"removed_fallback")
            target.writestr(info, payload)

    result = _inspect_bundle(altered)

    assert result["status"] == "blocked"
    assert "provider_entrypoint_missing_runtime_result_crash_fallback" in result["blockers"]


def test_successor_lane_writes_blocked_artifacts_for_unreadable_input(
    tmp_path: Path,
) -> None:
    admission_out = tmp_path / "admission.json"
    bound_out = tmp_path / "bound.json"
    adapter_out = tmp_path / "adapter.json"
    result = admission.run_successor_gpu_lane(
        authorization_path=EXPERIMENT / "compute_authorization_allocation_2.json",
        environment_path=tmp_path / "missing-environment.json",
        smoke_inventory_path=EXPERIMENT / "smoke_request_inventory.json",
        provider_preflight_path=_preflight_path(tmp_path),
        provider_bundle_path=EXPERIMENT / BUNDLE_NAME,
        provider_bundle_receipt_path=EXPERIMENT / BUNDLE_RECEIPT_NAME,
        admission_out=admission_out,
        bound_request_out=bound_out,
        adapter_output=adapter_out,
        job_dir=tmp_path / "job",
        public_base_url=None,
        token_file=None,
        secret_env_file=None,
        output_path=None,
        session_budget_ledger=None,
        expected_source_commit="e" * 40,
        execute=False,
    )

    assert result["status"] == "blocked"
    assert "successor_environment_unreadable" in result["blockers"]
    assert "successor_session_budget_ledger_missing" in result["blockers"]
    assert admission_out.is_file()
    assert bound_out.is_file()
    assert adapter_out.is_file()


def test_paid_resource_allocator_dispatches_successor_lane_only_through_probe_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda _commit, **_kwargs: ([], "d" * 40),
    )

    def fake_lane(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "dry_run_ready", "provider_mutations_performed": 0}

    monkeypatch.setattr(allocator, "run_successor_gpu_lane", fake_lane)
    code = allocator.main(
        [
            "gpu-canary",
            "--probe-kind",
            allocator.POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
            "--provider-launch-request",
            "authorization.json",
            "--release-evidence",
            "environment.json",
            "--model-cache-evidence",
            "inventory.json",
            "--preflight-bundle",
            "preflight.json",
            "--episode-bundle",
            "bundle.zip",
            "--successor-bundle-receipt",
            "receipt.json",
            "--provider-bundle-url-file",
            "bundle-url.txt",
            "--provider-output-put-url-file",
            "output-put-url.txt",
            "--provider-output-get-url-file",
            "output-get-url.txt",
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--pod-name",
            str(tmp_path / "job"),
            "--expected-source-commit",
            "d" * 40,
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}
    assert captured["execute"] is False
    assert captured["provider_bundle_path"] == "bundle.zip"
    assert captured["provider_bundle_receipt_path"] == "receipt.json"
    assert captured["provider_bundle_url_file"] == "bundle-url.txt"
    assert captured["provider_output_put_url_file"] == "output-put-url.txt"
    assert captured["provider_output_get_url_file"] == "output-get-url.txt"
    assert captured["expected_source_commit"] == "d" * 40
