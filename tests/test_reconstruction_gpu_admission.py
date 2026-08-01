from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_gpu_admission import (
    PREFLIGHT_SCHEMA_VERSION,
    PROBE_KIND,
    REQUEST_SCHEMA_VERSION,
    build_reconstruction_gpu_canary_admission,
    collect_reconstruction_vast_preflight,
)


SHA = "a" * 40
D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
D4 = "sha256:" + "4" * 64
D5 = "sha256:" + "5" * 64
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _request(**overrides):
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
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 1800,
        "retry_cap": 1,
        "authority_id": "user-authorization-20260730",
        "proof_effect": "none",
    }
    value.update(overrides)
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


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
    assert bound["provider_mutation_authorized"] is False
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_gpu_canary.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(admission, schema)


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
    request = _request(candidate_may_read_hidden_heldout=True)
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


def test_pose_and_trainer_canaries_cannot_masquerade_as_worker_smoke_execution():
    for operation in ("pose_canary", "trainer_canary"):
        request = _request(operation=operation)
        dry_run, dry_bound = _build(request=request)
        assert dry_run["status"] == "dry_run_ready"
        assert dry_run["operation"] == operation
        assert dry_run["execution_adapter_qualified"] is False
        assert dry_run["legal_next_actions"] == [
            "qualify_reconstruction_operation_execution_adapter"
        ]
        assert dry_bound["provider_mutation_authorized"] is False

        execute, execute_bound = _build(
            request=request,
            execute=True,
            execution_adapter_qualified=True,
        )
        assert execute["status"] == "blocked"
        assert execute["blockers"] == [
            "reconstruction_gpu_operation_execution_adapter_unavailable"
        ]
        assert execute["execution_adapter_qualified"] is False
        assert execute_bound["provider_mutation_authorized"] is False


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
