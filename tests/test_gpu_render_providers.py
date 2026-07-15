"""Hermetic tests for the provider-agnostic GPU render launch layer (no GPU spend, no net).

Covers: the neutral RenderLaunchSpec, the registry, per-provider request translation
(RunPod pod body vs Vast offer-search/create-instance), credential availability, the
fail-closed no-spend guards, and provider-parameterized teardown.
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.gpu_render_providers import (
    RenderLaunchSpec,
    RunPodRenderProvider,
    VastRenderProvider,
    get_render_provider,
    list_render_providers,
    validate_runpod_restart_storage_contract,
)
from blueprint_pipeline.cloud_vm_render_providers import AWSRenderProvider, GCPRenderProvider


def _spec(**over) -> RenderLaunchSpec:
    base = dict(
        name="blueprint-isaac-splat-render",
        image="img:tag",
        env={
            "BLUEPRINT_EVAL_MANIFEST_URI": "https://spaces.example/bundle.zip?sig=A",
            "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": "https://spaces.example/out.zip?sig=B",
            "CAMERAS_FILE": "cameras_canary.json",
        },
        bootstrap_argv=["-lc", "echo container_bash_started; run render"],
    )
    base.update(over)
    return RenderLaunchSpec(**base)


def _passing_prelaunch_guard() -> dict[str, object]:
    return {
        "schema_version": "isaac_particlefield_prelaunch_spend_guard.v1",
        "required_before_provider_launch": True,
        "can_launch": True,
        "blockers": [],
    }


def _guarded_runpod_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "imageName": "img:tag",
        "env": {},
        "dockerStartCmd": ["-lc", "run"],
        "prelaunch_spend_guard": _passing_prelaunch_guard(),
    }
    request.update(overrides)
    return request


def _with_prelaunch_guard(request: dict) -> dict:
    request["prelaunch_spend_guard"] = _passing_prelaunch_guard()
    return request


# ----------------------------- spec + registry -----------------------------

def test_render_launch_spec_bootstrap_script_is_last_argv() -> None:
    spec = _spec(bootstrap_argv=["-lc", "the-script-body"])
    assert spec.bootstrap_script == "the-script-body"
    assert spec.entrypoint == ["bash"]
    assert spec.container_disk_gb >= 120  # must hold the 10.7GB image + outputs


def test_registry_returns_known_providers_and_rejects_unknown() -> None:
    assert isinstance(get_render_provider("runpod"), RunPodRenderProvider)
    assert isinstance(get_render_provider("vast"), VastRenderProvider)
    assert isinstance(get_render_provider(None), RunPodRenderProvider)  # default
    assert isinstance(get_render_provider("VAST"), VastRenderProvider)  # case-insensitive
    assert isinstance(get_render_provider("gcp"), GCPRenderProvider)
    assert isinstance(get_render_provider("aws"), AWSRenderProvider)
    with pytest.raises(ValueError):
        get_render_provider("lambda-labs")


def test_list_render_providers_reports_both_with_availability() -> None:
    listed = list_render_providers()
    names = {p["provider"] for p in listed}
    assert names == {"runpod", "vast", "digitalocean", "gcp", "aws"}
    for entry in listed:
        assert "available" in entry  # bool reflecting credential presence


# ----------------------------- RunPod translation -----------------------------

def test_runpod_build_request_is_pod_body(tmp_path: Path) -> None:
    body = RunPodRenderProvider().build_request(_spec(), tmp_path)
    assert body["imageName"] == "img:tag"
    assert body["dockerEntrypoint"] == ["bash"]
    assert body["dockerStartCmd"] == ["-lc", "echo container_bash_started; run render"]
    assert "NVIDIA L40S" in body["gpuTypeIds"]
    assert body["containerDiskInGb"] >= 120
    assert body["env"]["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("sig=A")
    assert body["cloudType"] == "SECURE"
    assert body["max_hourly_rate_usd"] == pytest.approx(5.0)
    assert body["env"]["BLUEPRINT_RUNPOD_CONTAINER_DISK_EPHEMERAL"] == "1"
    assert body["env"]["BLUEPRINT_RESUMABLE_STATE_ROOT"].startswith("/workspace/")
    assert body["blueprintStorageContract"]["persistent_volume"].startswith(
        "survives_restart"
    )


def test_runpod_launch_strips_local_capability_filters_before_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[dict] = []

    def fake_call(method, path, body, **_kwargs):
        if method == "POST" and path == "/pods":
            observed.append(dict(body))
            return 400, {"error": "no capacity"}
        raise AssertionError((method, path))

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
    )
    monkeypatch.setattr(
        RunPodRenderProvider,
        "_key",
        lambda _self: "test-key",
    )
    monkeypatch.setattr(
        RunPodRenderProvider,
        "capacity_preflight",
        lambda _self, _request=None: {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": "NVIDIA L40S",
                    "on_demand_price_usd_per_hour": 0.99,
                }
            ],
        },
    )
    request = _guarded_runpod_request(
        gpuTypeIds=["NVIDIA L40S"],
        max_hourly_rate_usd=1.10,
        min_gpu_ram_mb=48_000,
        requires_rtx=True,
    )
    result = RunPodRenderProvider().launch(tmp_path, request, cold=True)

    assert result["allocation_created"] is False
    assert len(observed) == 1
    assert "min_gpu_ram_mb" not in observed[0]
    assert "requires_rtx" not in observed[0]


def test_runpod_restart_storage_contract_requires_volume_not_container_sentinel(
    tmp_path: Path,
) -> None:
    volume = tmp_path / "volume" / "sentinel"
    volume.parent.mkdir()
    volume.write_text("persistent", encoding="utf-8")
    result = validate_runpod_restart_storage_contract(
        container_disk_sentinel=tmp_path / "container" / "sentinel",
        volume_sentinel=volume,
    )
    assert result["status"] == "passed"
    container = tmp_path / "container" / "sentinel"
    container.parent.mkdir()
    container.write_text("stale", encoding="utf-8")
    assert (
        validate_runpod_restart_storage_contract(
            container_disk_sentinel=container,
            volume_sentinel=volume,
        )["status"]
        == "blocked"
    )


def test_runpod_capacity_preflight_requires_secure_rtx_stock(monkeypatch) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")

    def fake_graphql(query, *, key, timeout=60):
        assert key == "rp-key"
        assert "secureCloud: true" in query
        return 200, {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA L40S",
                        "displayName": "L40S",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "lowestPrice": {
                            "stockStatus": "Medium",
                            "uninterruptablePrice": 1.14,
                            "availableGpuCounts": None,
                        },
                    },
                    {
                        "id": "NVIDIA RTX A6000",
                        "displayName": "RTX A6000",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "lowestPrice": {
                            "stockStatus": "None",
                            "uninterruptablePrice": None,
                            "availableGpuCounts": [],
                        },
                    },
                ]
            }
        }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        fake_graphql,
    )
    result = RunPodRenderProvider().capacity_preflight(
        {"min_gpu_ram_mb": 48000, "requires_rtx": True}
    )

    assert result["status"] == "available"
    assert result["capacity_confidence"] == "advisory"
    assert [row["gpu_type_id"] for row in result["viable_gpu_types"]] == ["NVIDIA L40S"]
    assert result["viable_gpu_types"][0]["available_gpu_counts"] == []
    assert result["viable_gpu_types"][0]["single_gpu_offer_available"] is True
    a6000 = next(
        row for row in result["considered_gpu_types"]
        if row["gpu_type_id"] == "NVIDIA RTX A6000"
    )
    assert "single_gpu_stock_unavailable" in a6000["blockers"]


def test_runpod_capacity_preflight_scopes_price_and_stock_to_data_center(
    monkeypatch,
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")

    def fake_graphql(query, *, key, timeout=60):
        del timeout
        assert key == "rp-key"
        assert 'dataCenterId: "US-TX-3"' in query
        assert 'allowedCudaVersions: ["12.6"]' in query
        return 200, {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA A40",
                        "displayName": "A40",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "lowestPrice": {
                            "stockStatus": "High",
                            "uninterruptablePrice": 0.44,
                            "availableGpuCounts": [1],
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        fake_graphql,
    )
    result = RunPodRenderProvider().capacity_preflight(
        {
            "gpuTypeIds": ["NVIDIA A40"],
            "dataCenterIds": ["US-TX-3"],
            "allowedCudaVersions": ["12.6"],
            "requires_rtx": True,
        }
    )
    assert result["status"] == "available"
    assert result["requested_data_center_ids"] == ["US-TX-3"]
    assert result["requested_allowed_cuda_versions"] == ["12.6"]
    assert result["viable_gpu_types"][0]["capacity_data_center_id"] == "US-TX-3"
    assert result["viable_gpu_types"][0]["capacity_allowed_cuda_versions"] == [
        "12.6"
    ]


def test_runpod_capacity_preflight_supports_explicit_community_pool(monkeypatch) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")

    def fake_graphql(query, *, key, timeout=60):
        assert key == "rp-key"
        assert "secureCloud: false" in query
        return 200, {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA RTX 6000 Ada Generation",
                        "displayName": "RTX 6000 Ada",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "communityCloud": True,
                        "lowestPrice": {
                            "stockStatus": "Low",
                            "uninterruptablePrice": 0.74,
                            "availableGpuCounts": None,
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        fake_graphql,
    )
    result = RunPodRenderProvider().capacity_preflight(
        {
            "cloudType": "COMMUNITY",
            "gpuTypeIds": ["NVIDIA RTX 6000 Ada Generation"],
            "min_gpu_ram_mb": 46000,
            "requires_rtx": True,
        }
    )

    assert result["status"] == "available"
    assert result["cloud_type"] == "COMMUNITY"
    assert result["capacity_confidence"] == "advisory"
    assert result["viable_gpu_types"][0]["on_demand_price_usd_per_hour"] == 0.74


def test_runpod_capacity_preflight_reports_honest_confidence(monkeypatch) -> None:
    """The exact one-GPU offer is advisory; only create is authoritative."""
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")

    def fake_graphql(query, *, key, timeout=60):
        return 200, {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA A40",
                        "displayName": "A40",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "lowestPrice": {
                            # Attempt-021 shape: label says Medium, counts empty,
                            # create then failed with HTTP 500 no-resources.
                            "stockStatus": "Medium",
                            "uninterruptablePrice": 0.44,
                            "availableGpuCounts": [],
                        },
                    },
                    {
                        "id": "NVIDIA RTX A6000",
                        "displayName": "RTX A6000",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "lowestPrice": {
                            "stockStatus": "High",
                            "uninterruptablePrice": 0.49,
                            "availableGpuCounts": [1, 2],
                        },
                    },
                ]
            }
        }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        fake_graphql,
    )
    result = RunPodRenderProvider().capacity_preflight(
        {
            "gpuTypeIds": ["NVIDIA A40", "NVIDIA RTX A6000"],
            "min_gpu_ram_mb": 48000,
            "requires_rtx": True,
        }
    )

    assert result["reservation_proven"] is False
    assert result["authoritative_capacity_source"] == "provider_create_response"
    by_id = {row["gpu_type_id"]: row for row in result["considered_gpu_types"]}
    a40 = by_id["NVIDIA A40"]
    assert a40["catalog_reported_stock"] == "Medium"
    assert a40["single_gpu_count_known"] is False
    assert a40["single_gpu_offer_requested"] is True
    assert a40["single_gpu_offer_available"] is True
    assert a40["reservation_proven"] is False
    assert a40["capacity_confidence"] == "advisory"
    a6000 = by_id["NVIDIA RTX A6000"]
    assert a6000["single_gpu_count_known"] is True
    assert a6000["capacity_confidence"] == "advisory"
    # Overall confidence never exceeds advisory: the probe is not a reservation.
    assert result["capacity_confidence"] == "advisory"


def test_runpod_capacity_preflight_registers_rtx_pro_6000_blackwell(
    monkeypatch,
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")

    def fake_graphql(query, *, key, timeout=60):
        del timeout
        assert key == "rp-key"
        assert "gpuCount: 1" in query
        return 200, {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "displayName": "RTX PRO 6000",
                        "memoryInGb": 96,
                        "secureCloud": True,
                        "lowestPrice": {
                            "stockStatus": "Medium",
                            "uninterruptablePrice": 1.99,
                            "availableGpuCounts": None,
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        fake_graphql,
    )
    result = RunPodRenderProvider().capacity_preflight(
        {
            "gpuTypeIds": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
            "dataCenterIds": ["US-NC-2"],
            "allowedCudaVersions": ["12.8"],
            "min_gpu_ram_mb": 48000,
            "requires_rtx": True,
        }
    )

    row = result["viable_gpu_types"][0]
    assert result["capacity_confidence"] == "advisory"
    assert row["memory_in_gb"] == 96
    assert row["single_gpu_count_known"] is False
    assert row["single_gpu_offer_available"] is True


def test_runpod_exact_one_gpu_offer_requires_explicit_stock_label(monkeypatch) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")

    def fake_graphql(query, *, key, timeout=60):
        del query, timeout
        assert key == "rp-key"
        return 200, {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA A40",
                        "displayName": "A40",
                        "memoryInGb": 48,
                        "secureCloud": True,
                        "lowestPrice": {
                            "stockStatus": None,
                            "uninterruptablePrice": 0.44,
                            "availableGpuCounts": None,
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        fake_graphql,
    )
    result = RunPodRenderProvider().capacity_preflight(
        {
            "gpuTypeIds": ["NVIDIA A40"],
            "dataCenterIds": ["US-NC-2"],
            "allowedCudaVersions": ["12.8"],
            "requires_rtx": True,
        }
    )

    row = result["viable_gpu_types"][0]
    assert row["single_gpu_offer_requested"] is True
    assert row["single_gpu_offer_available"] is False
    assert row["capacity_confidence"] == "unknown"
    assert result["capacity_confidence"] == "unknown"


def test_runpod_create_capacity_failure_is_capacity_outcome_not_spend(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        lambda method, path, body, *, key, timeout=90: (
            500,
            {"error": "This machine does not have the resources to deploy your pod"},
        ),
    )
    request = {
        "imageName": "img@sha256:abc",
        "prelaunch_spend_guard": {
            "required_before_provider_launch": True,
            "can_launch": True,
        },
    }
    res = RunPodRenderProvider().launch(tmp_path, request, cold=True)
    assert res["status"] == "blocked"
    assert res["blockers"][0] == "runpod_secure_cloud_create_capacity_unavailable"
    assert res["capacity_outcome"] is True
    assert res["allocation_created"] is False
    assert res["spend_occurred"] is False


def test_runpod_capacity_preflight_fails_closed_on_query_error(monkeypatch) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_graphql_call",
        lambda query, *, key, timeout=60: (503, {"error": "redacted"}),
    )

    result = RunPodRenderProvider().capacity_preflight(
        {"min_gpu_ram_mb": 48000, "requires_rtx": True}
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["runpod_capacity_probe_failed"]
    assert result["raw_provider_response_recorded"] is False


def test_runpod_launch_fail_closed_without_key(tmp_path: Path, monkeypatch) -> None:
    # point secret lookups at an empty dir so no key is found and no network call happens
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = RunPodRenderProvider().launch(tmp_path, {"imageName": "x"}, cold=True)
    assert res["status"] == "blocked"
    assert "runpod_api_key_missing" in res["blockers"]


def test_runpod_launch_blocks_without_prelaunch_guard_before_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        return 201, {"id": "pod-should-not-start"}

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)

    res = RunPodRenderProvider().launch(
        tmp_path,
        {"imageName": "img:tag", "env": {}, "dockerStartCmd": ["-lc", "run"]},
        cold=True,
    )

    assert res["status"] == "blocked"
    assert "runpod_render_prelaunch_spend_guard_missing" in res["blockers"]
    assert calls == []


def test_runpod_launch_strips_internal_guard_fields_before_create(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    observed: list[dict] = []

    def fake_call(method, path, body, *, key, timeout=90):
        assert (method, path, key) == ("POST", "/pods", "rp-key")
        observed.append(dict(body))
        return 201, {"id": "pod-1"}

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        fake_call,
    )
    request = _guarded_runpod_request()
    request["pending_teardown_record"] = "/tmp/internal-record.json"
    result = RunPodRenderProvider().launch(tmp_path, request, cold=True)

    assert result["status"] == "launched"
    assert result["instance_id"] == "pod-1"
    assert "prelaunch_spend_guard" not in observed[0]
    assert "pending_teardown_record" not in observed[0]


def test_runpod_rechecks_and_filters_rate_cap_before_cold_create(
    tmp_path: Path, monkeypatch
) -> None:
    provider = RunPodRenderProvider()
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        provider,
        "capacity_preflight",
        lambda _request: {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": "NVIDIA A40",
                    "on_demand_price_usd_per_hour": 0.44,
                },
                {
                    "gpu_type_id": "NVIDIA L40S",
                    "on_demand_price_usd_per_hour": 0.79,
                },
            ],
        },
    )
    sent: dict[str, object] = {}

    def fake_call(method, path, body, *, key, timeout=90):
        assert (method, path) == ("POST", "/pods")
        sent.update(body)
        return 201, {"id": "pod-1"}

    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
    )
    request = _guarded_runpod_request(
        gpuTypeIds=["NVIDIA A40", "NVIDIA L40S"],
        max_hourly_rate_usd=0.5,
    )

    result = provider.launch(tmp_path, request, cold=True)

    assert result["status"] == "launched"
    assert sent["gpuTypeIds"] == ["NVIDIA A40"]
    assert "max_hourly_rate_usd" not in sent


def test_runpod_blocks_when_fresh_price_cannot_meet_rate_cap(
    tmp_path: Path, monkeypatch
) -> None:
    provider = RunPodRenderProvider()
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        provider,
        "capacity_preflight",
        lambda _request: {
            "status": "available",
            "viable_gpu_types": [
                {
                    "gpu_type_id": "NVIDIA L40S",
                    "on_demand_price_usd_per_hour": 0.79,
                }
            ],
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("create must remain uncalled")
        ),
    )
    request = _guarded_runpod_request(gpuTypeIds=["NVIDIA L40S"])
    request["prelaunch_spend_guard"]["max_hourly_rate_usd"] = 0.5

    result = provider.launch(tmp_path, request, cold=True)

    assert result["status"] == "blocked"
    assert result["allocation_created"] is False
    assert result["blockers"] == ["runpod_pre_mutation_rate_cap_unverified"]


def test_runpod_warm_rate_cap_blocks_before_start_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        assert (method, path) == ("GET", "/pods/warm-1")
        return 200, {
            "id": "warm-1",
            "desiredStatus": "STOPPED",
            "costPerHr": 0.79,
        }

    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
    )
    request = _guarded_runpod_request()
    request["prelaunch_spend_guard"]["max_hourly_rate_usd"] = 0.5

    result = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        request,
        cold=False,
        allow_cold_fallback=False,
    )

    assert result["status"] == "blocked"
    assert result["allocation_created"] is False
    assert result["blockers"] == [
        "runpod_warm_hourly_rate_exceeds_spend_cap"
    ]
    assert calls == [("GET", "/pods/warm-1")]


def test_runpod_launch_classifies_create_resource_500_as_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        lambda method, path, body, *, key, timeout=90: (
            500,
            {
                "error": (
                    '{"error":"create pod: This machine does not have the resources '
                    'to deploy your pod. Please try a different machine","status":500}'
                )
            },
        ),
    )

    result = RunPodRenderProvider().launch(
        tmp_path,
        _guarded_runpod_request(),
        cold=True,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "runpod_secure_cloud_create_capacity_unavailable",
        "no_pod_started",
    ]
    assert result["attempts"][0]["pod_id"] is None


def test_runpod_warm_start_rejection_is_recorded_before_cold_fallback(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        assert key == "rp-key"
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/start" and method == "POST":
            return 409, {"error": "pod is not startable from EXITED"}
        if path == "/pods" and method == "POST":
            return 201, {"id": "cold-1"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)
    res = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        _guarded_runpod_request(),
        cold=False,
    )

    assert res["status"] == "launched"
    assert res["instance_id"] == "cold-1"
    assert res["mode"] == "cold_create"
    assert res["attempts"][0] == {
        "pod_id": "warm-1",
        "get_status": 200,
        "desiredStatus": "EXITED",
        "update_status": 200,
        "start_status": 409,
        "start_error": "pod is not startable from EXITED",
    }
    assert res["attempts"][1]["cold_create_status"] == 201
    assert calls == [
        ("GET", "/pods/warm-1"),
        ("POST", "/pods/warm-1/update"),
        ("POST", "/pods/warm-1/start"),
        ("POST", "/pods"),
    ]


def test_runpod_warm_only_blocks_without_cold_create(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/start" and method == "POST":
            return 409, {"error": "pod is not startable from EXITED"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)
    res = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        _guarded_runpod_request(),
        cold=False,
        allow_cold_fallback=False,
    )

    assert res["status"] == "blocked"
    assert "warm_restart_failed_cold_fallback_disabled" in res["blockers"]
    assert res["attempts"][0]["start_status"] == 409
    assert ("POST", "/pods") not in calls


def test_runpod_warm_update_failure_does_not_start_stale_command(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "STOPPED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 400, {"error": "invalid update"}
        if path == "/pods" and method == "POST":
            return 201, {"id": "cold-1"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)
    res = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        _guarded_runpod_request(),
        cold=False,
    )

    assert res["status"] == "launched"
    assert res["instance_id"] == "cold-1"
    assert res["attempts"][0]["update_status"] == 400
    assert res["attempts"][0]["update_error"] == "invalid update"
    assert ("POST", "/pods/warm-1/start") not in calls


@pytest.mark.parametrize(
    ("ambiguous_path", "ambiguous_status", "expected_blocker"),
    [
        ("/pods/warm-1/update", 0, "runpod_warm_update_outcome_ambiguous"),
        ("/pods/warm-1/update", 500, "runpod_warm_update_outcome_ambiguous"),
        ("/pods/warm-1/start", 0, "runpod_warm_start_outcome_ambiguous"),
        ("/pods/warm-1/start", 408, "runpod_warm_start_outcome_ambiguous"),
        ("/pods/warm-1/start", 429, "runpod_warm_start_outcome_ambiguous"),
        ("/pods/warm-1/start", 500, "runpod_warm_start_outcome_ambiguous"),
    ],
)
def test_runpod_warm_mutation_lost_response_never_falls_back_to_cold_create(
    tmp_path: Path,
    monkeypatch,
    ambiguous_path: str,
    ambiguous_status: int,
    expected_blocker: str,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "STOPPED"}
        if path == ambiguous_path and method == "POST":
            return ambiguous_status, {"error": "ambiguous mutation"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 200, {"id": "warm-1"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
    )

    result = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path, _guarded_runpod_request(), cold=False
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == [expected_blocker]
    assert result["allocation_outcome_ambiguous"] is True
    assert "allocation_created" not in result
    assert ("POST", "/pods") not in calls


def test_runpod_generic_start_conflict_is_ambiguous_not_cold_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "STOPPED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 200, {"id": "warm-1"}
        if path == "/pods/warm-1/start" and method == "POST":
            return 409, {"error": "conflict"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
    )

    result = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path, _guarded_runpod_request(), cold=False
    )

    assert result["blockers"] == ["runpod_warm_start_outcome_ambiguous"]
    assert ("POST", "/pods") not in calls


def test_runpod_warm_start_accepts_any_successful_2xx_response(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_call(method, path, body, *, key, timeout=90):
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "STOPPED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 204, {}
        if path == "/pods/warm-1/start" and method == "POST":
            return 204, {}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", fake_call
    )

    result = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path, _guarded_runpod_request(), cold=False
    )

    assert result["status"] == "launched"
    assert result["instance_id"] == "warm-1"
    assert result["mode"] == "warm_restart"


def test_runpod_cold_create_lost_response_is_not_explicit_no_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        lambda method, path, body, *, key, timeout=90: (
            0,
            {"error": "TimeoutError"},
        ),
    )

    result = RunPodRenderProvider().launch(
        tmp_path, _guarded_runpod_request(), cold=True
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["runpod_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert "allocation_created" not in result
    assert result["spend_occurred"] is None


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (201, {}),
        (500, {"error": "internal server error"}),
    ],
)
def test_runpod_no_id_without_definitive_rejection_is_ambiguous(
    tmp_path: Path, monkeypatch, status: int, body: dict
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        lambda method, path, request, *, key, timeout=90: (status, body),
    )

    result = RunPodRenderProvider().launch(
        tmp_path, _guarded_runpod_request(), cold=True
    )

    assert result["blockers"] == ["runpod_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert "allocation_created" not in result


@pytest.mark.parametrize(
    "malformed_id",
    [True, 123, {"id": "pod-1"}, " pod-1", "pod/escape"],
)
def test_runpod_malformed_success_id_is_ambiguous(
    tmp_path: Path, monkeypatch, malformed_id: object
) -> None:
    monkeypatch.setattr(RunPodRenderProvider, "_key", lambda _self: "rp-key")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call",
        lambda *_args, **_kwargs: (201, {"id": malformed_id}),
    )

    result = RunPodRenderProvider().launch(
        tmp_path, _guarded_runpod_request(), cold=True
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["runpod_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert "instance_id" not in result
    assert not (tmp_path / "started_pod_id.txt").exists()


def test_runpod_teardown_404_is_already_gone_success(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        assert key == "rp-key"
        return 404, {"error": "pod not found"}

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)

    stop = RunPodRenderProvider().stop("pod-missing")
    terminate = RunPodRenderProvider().terminate("pod-missing")

    assert stop == {"status": "stopped", "http": 404, "already_gone": True}
    assert terminate == {"status": "terminated", "http": 404, "already_gone": True}
    assert calls == [
        ("POST", "/pods/pod-missing/stop"),
        ("DELETE", "/pods/pod-missing"),
    ]


def test_runpod_inspect_redacts_and_marks_pre_runtime(monkeypatch) -> None:
    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        assert method == "GET"
        assert path == "/pods/pod-1"
        assert body is None
        assert key == "rp-key"
        return 200, {
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "publicIp": "",
            "machineId": "machine-a",
            "costPerHr": 0.69,
            "createdAt": "2026-07-01 21:42:02.335 +0000 UTC",
            "lastStartedAt": "2026-07-01 21:42:02.33 +0000 UTC",
            "lastStatusChange": "Rented by User",
            "imageName": "img:tag",
        }

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)

    res = RunPodRenderProvider().inspect("pod-1")

    assert res["status"] == "observed"
    assert res["runtime_present"] is False
    assert res["public_ip_present"] is False
    assert res["machineId"] == "machine-a"
    assert res["raw_provider_response_recorded"] is False
    assert "env" not in res and "dockerStartCmd" not in res


# ----------------------------- Vast translation -----------------------------

def test_vast_build_request_offer_search_and_create(tmp_path: Path) -> None:
    req = VastRenderProvider().build_request(_spec(), tmp_path)
    # offer search filters to a single rentable on-demand GPU under the hourly rate
    sp = req["search_payload"]
    assert sp["type"] == "on-demand"
    assert sp["rentable"] == {"eq": True}
    assert sp["num_gpus"] == {"eq": 1}
    assert sp["dph_total"]["lte"] == pytest.approx(5.0)
    # create-instance body: args mode runs our bootstrap via bash, env carries the signed urls
    cp = req["create_payload"]
    assert cp["image"] == "img:tag"
    assert cp["disk"] == req["disk"] >= 120
    assert cp["runtype"] == "args"
    assert cp["target_state"] == "running"
    assert cp["args_str"].startswith("bash -lc")
    assert "container_bash_started" in cp["args_str"]
    assert cp["env"]["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith("sig=B")
    assert req["create_endpoint"] == "PUT /asks/{ask_contract_id}/"


def test_vast_launch_fail_closed_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = VastRenderProvider().launch(tmp_path, {"search_payload": {}}, cold=False)
    assert res["status"] == "blocked"
    assert "vast_api_key_missing" in res["blockers"]


def test_vast_launch_blocks_without_prelaunch_guard_before_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        calls.append((method, path))
        return 200, {"offers": []}

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    res = VastRenderProvider().launch(tmp_path, {"search_payload": {}}, cold=False)

    assert res["status"] == "blocked"
    assert "vast_render_prelaunch_spend_guard_missing" in res["blockers"]
    assert calls == []


def test_vast_stop_fail_closed_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = VastRenderProvider().stop("12345")
    assert res["status"] == "blocked"
    assert "vast_api_key_missing" in res["blockers"]


def test_vast_launch_writes_started_instance_id(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        calls.append((method, path))
        assert api_key == "vast-key"
        if method == "POST" and path == "/bundles/":
            return 200, {"offers": [{"id": "raw-offer"}]}
        if method == "PUT" and path == "/asks/ask-1/":
            return 200, {"new_contract": 12345}
        raise AssertionError((method, path, payload, timeout_seconds))

    offer = {
        "ask_contract_id": "ask-1",
        "gpu_name": "RTX 4090",
        "hourly_rate_usd": 0.44,
    }

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._offers_from_response", lambda _resp: [offer])
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._select_offer", lambda offers, **_kw: offers[0])

    req = _with_prelaunch_guard(VastRenderProvider().build_request(_spec(), tmp_path))
    res = VastRenderProvider().launch(tmp_path, req)

    assert res["status"] == "launched"
    assert res["instance_id"] == "12345"
    assert res["mode"] == "vast_on_demand"
    assert (tmp_path / "started_vast_instance_id.txt").read_text() == "12345"
    assert calls == [("POST", "/bundles/"), ("PUT", "/asks/ask-1/")]


def test_vast_terminate_delegates_to_destroy_instance_delete(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=30):
        calls.append((method, path))
        assert api_key == "vast-key"
        assert payload is None
        return 204, {}

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    res = VastRenderProvider().terminate("inst-123")

    assert res["status"] == "stopped"
    assert res["http"] == 204
    assert calls == [("DELETE", "/instances/inst-123/")]


def test_vast_teardown_404_is_already_gone_success(monkeypatch) -> None:
    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=30):
        assert method == "DELETE"
        assert path == "/instances/inst-missing/"
        assert api_key == "vast-key"
        raise urllib.error.HTTPError(
            url="https://console.vast.ai/api/v0/instances/inst-missing/",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    assert VastRenderProvider().stop("inst-missing") == {
        "status": "stopped",
        "http": 404,
        "already_gone": True,
    }


# ----------------------------- availability reflects secrets -----------------------------

def test_availability_reflects_secret_presence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    assert RunPodRenderProvider().available()["available"] is False
    assert VastRenderProvider().available()["available"] is False
    (tmp_path / "runpod_api_key").write_text("rp-key")
    (tmp_path / "vast_api_key").write_text("vast-key")
    assert RunPodRenderProvider().available()["available"] is True
    assert VastRenderProvider().available()["available"] is True


# ----------------------------- teardown is provider-parameterized -----------------------------

def test_watch_and_collect_tears_down_via_provider(tmp_path: Path) -> None:
    from blueprint_pipeline.isaac_particlefield_render_job import watch_and_collect

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.terminated: str | None = None

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()
    # max_seconds=0 -> skip the poll loop entirely (no network), go straight to teardown
    res = watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=0)
    assert fake.terminated == "inst-9"  # blocked/no-result pod is DELETED
    assert res["status"] == "blocked"  # nothing rendered
    assert res["teardown"]["status"] == "terminated"


def test_watch_and_collect_terminates_no_output_pod_even_when_preserve_requested(tmp_path: Path) -> None:
    from blueprint_pipeline.isaac_particlefield_render_job import watch_and_collect

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=0,
        preserve_instance=True,
    )

    assert res["status"] == "blocked"
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown"]["status"] == "terminated"
    assert res["teardown_reason"] == "timeout_without_runner_done_terminated"
    assert res["timed_out_without_runner_done"] is True


def test_watch_and_collect_stops_successful_pod_for_warm_reuse(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 0}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({"status": "completed"}))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "completed"
    assert fake.stopped == "inst-9"
    assert fake.terminated is None
    assert res["teardown_reason"] == "runner_done_preserved_for_warm_reuse"
    assert res["teardown"]["status"] == "stopped"


def test_watch_and_collect_terminates_digitalocean_runner_done(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "digitalocean"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 201}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 0}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({"status": "completed"}))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "completed"
    assert fake.terminated == "inst-9"
    assert fake.stopped is None
    assert res["teardown_reason"] == "runner_done_terminated_no_warm_reuse"
    assert res["teardown"]["status"] == "terminated"


def test_watch_and_collect_stops_blocked_runner_pod_for_warm_reuse(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 0}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({
            "status": "blocked",
            "blockers": ["placement_validation_failed"],
        }))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "blocked"
    assert fake.stopped == "inst-9"
    assert fake.terminated is None
    assert res["teardown"]["status"] == "stopped"


def test_watch_and_collect_terminates_blocked_startup_canary(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def inspect(self, instance_id: str) -> dict:
            return {
                "status": "observed",
                "http": 200,
                "instance_id": instance_id,
                "machineId": "machine-bad-driver",
            }

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 2}))
        zf.writestr(
            "isaac_g1_kitchen_parity_result.json",
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["isaac_sim_6_rtx_driver_unsupported"],
                    "image_startup_canary": True,
                }
            ),
        )
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=1,
        poll=1,
        preserve_instance=True,
        preserve_blocked_instance=False,
    )

    assert res["status"] == "blocked"
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown"]["status"] == "terminated"
    assert res["provider_snapshot_before_teardown"]["machineId"] == "machine-bad-driver"


def test_watch_and_collect_terminates_runner_timeout(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({
            "phase": "runner_timeout",
            "timeout_seconds": 840,
        }))
        zf.writestr("runner_console.log", "SimulationApp boot did not finish\n")
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "blocked"
    assert res["runner_timeout_observed"] is True
    assert res["timed_out_without_runner_done"] is False
    assert res["teardown_reason"] == "runner_timeout_terminated"
    assert fake.terminated == "inst-9"
    assert fake.stopped is None


def test_watch_and_collect_ignores_stale_result_before_runner_done(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "kitchen_fetching"}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({
            "status": "blocked",
            "blockers": ["stale_previous_run"],
        }))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(job.time, "time", lambda: next(clock, 2.0))
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "blocked"
    assert res["last_bootstrap"]["phase"] == "kitchen_fetching"
    assert res["runner_result"]["blockers"] == ["stale_previous_run"]
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown_reason"] == "timeout_without_runner_done_terminated"


def test_watch_and_collect_terminates_current_final_result_without_runner_done(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped"}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated"}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({
            "phase": "runner_starting",
            "launch_session_id": "launch-123",
        }))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({
            "status": "blocked",
            "scenarios_executed": 0,
            "blockers": ["isaac_runner_exception_before_scenario_outcome"],
        }))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    (job_dir / "launch_session_nonce.txt").write_text("launch-123")
    fake = _FakeProvider()

    res = job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=1,
        poll=1,
        preserve_instance=True,
    )

    assert res["status"] == "blocked"
    assert res["runner_result"]["scenarios_executed"] == 0
    assert res["timed_out_without_runner_done"] is False
    assert res["runner_done_observed"] is False
    assert res["final_result_without_runner_done"] is True
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown_reason"] == "final_result_without_runner_done_terminated"


def test_watch_and_collect_terminates_heartbeat_timeout_despite_preserve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped"}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated"}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_starting"}))
        zf.writestr("runner_console.log", "Isaac is still starting")
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(job.time, "time", lambda: next(clock, 2.0))
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=1,
        poll=1,
        preserve_instance=True,
    )

    assert res["status"] == "blocked"
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["last_bootstrap"]["phase"] == "runner_starting"
    assert res["timed_out_without_runner_done"] is True


def test_runpod_terminate_is_delete_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = RunPodRenderProvider().terminate("podabc")
    assert res["status"] == "blocked"  # no key -> no network, fail closed
    assert "runpod_api_key_missing" in res["blockers"]
    # terminate is distinct from stop (DELETE vs POST /stop) — both exist on the provider
    assert hasattr(RunPodRenderProvider(), "terminate") and hasattr(RunPodRenderProvider(), "stop")


def test_vast_launch_retries_next_offer_on_create_400(tmp_path: Path, monkeypatch) -> None:
    """A stale ask 400s at create; the launch must record the error body and
    fall through to the next candidate offer instead of blocking the race."""
    import io
    import urllib.error

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        if method == "POST" and path == "/bundles/":
            return 200, {"offers": ["raw"]}
        if method == "PUT" and path == "/asks/ask-stale/":
            raise urllib.error.HTTPError(
                "https://vast/asks/ask-stale/", 400, "Bad Request", None,
                io.BytesIO(b'{"success": false, "msg": "ask expired"}'))
        if method == "PUT" and path == "/asks/ask-fresh/":
            return 200, {"new_contract": 777}
        raise AssertionError((method, path))

    stale = {"ask_contract_id": "ask-stale", "gpu_name": "RTX 4090", "hourly_rate_usd": 0.4}
    fresh = {"ask_contract_id": "ask-fresh", "gpu_name": "RTX 4090", "hourly_rate_usd": 0.5}

    def fake_select(offers, **_kw):
        return offers[0] if offers else None

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._offers_from_response",
                        lambda _resp: [stale, fresh])
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._select_offer", fake_select)

    req = _with_prelaunch_guard(VastRenderProvider().build_request(_spec(), tmp_path))
    res = VastRenderProvider().launch(tmp_path, req)

    assert res["status"] == "launched"
    assert res["instance_id"] == "777"
    create_errors = [a for a in res.get("attempts", []) if a.get("create_http_status") == 400]
    assert create_errors and "ask expired" in str(create_errors[0].get("create_error_body"))


@pytest.mark.parametrize("failure_kind", ["timeout", "http_500", "success_without_id"])
def test_vast_ambiguous_create_never_tries_a_second_offer(
    monkeypatch, tmp_path: Path, failure_kind: str
) -> None:
    import io
    import urllib.error

    calls: list[str] = []

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        if method == "POST":
            return 200, {"offers": ["raw"]}
        calls.append(path)
        if failure_kind == "timeout":
            raise TimeoutError("response lost")
        if failure_kind == "http_500":
            raise urllib.error.HTTPError(
                "https://vast", 500, "error", None, io.BytesIO(b"server error")
            )
        return 200, {}

    offers = [
        {"ask_contract_id": "ask-1", "gpu_name": "RTX", "hourly_rate_usd": 0.4},
        {"ask_contract_id": "ask-2", "gpu_name": "RTX", "hourly_rate_usd": 0.5},
    ]
    monkeypatch.setattr(VastRenderProvider, "_key", lambda _self: "vast-key")
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json
    )
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._offers_from_response",
        lambda _response: offers,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._select_offer",
        lambda remaining, **_kwargs: remaining[0] if remaining else None,
    )

    provider = VastRenderProvider()
    result = provider.launch(
        tmp_path, _with_prelaunch_guard(provider.build_request(_spec(), tmp_path))
    )

    assert result["blockers"] == ["vast_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert len(calls) == 1
    assert "allocation_created" not in result


@pytest.mark.parametrize(
    "malformed_id",
    [True, {"id": 777}, " 777", "contract-777", 0, -1],
)
def test_vast_malformed_success_id_is_ambiguous(
    monkeypatch, tmp_path: Path, malformed_id: object
) -> None:
    calls: list[str] = []

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        if method == "POST":
            return 200, {"offers": ["raw"]}
        calls.append(path)
        return 200, {"new_contract": malformed_id}

    offers = [
        {"ask_contract_id": "ask-1", "gpu_name": "RTX", "hourly_rate_usd": 0.4},
        {"ask_contract_id": "ask-2", "gpu_name": "RTX", "hourly_rate_usd": 0.5},
    ]
    monkeypatch.setattr(VastRenderProvider, "_key", lambda _self: "vast-key")
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json
    )
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._offers_from_response",
        lambda _response: offers,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.vast_provider_adapter._select_offer",
        lambda remaining, **_kwargs: remaining[0] if remaining else None,
    )

    provider = VastRenderProvider()
    result = provider.launch(
        tmp_path, _with_prelaunch_guard(provider.build_request(_spec(), tmp_path))
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["vast_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert calls == ["/asks/ask-1/"]
    assert "instance_id" not in result
    assert not (tmp_path / "started_vast_instance_id.txt").exists()


def test_default_runpod_gpu_types_exclude_consumer_4090_pool(monkeypatch) -> None:
    """The GeForce 4090 pool produced ~10 dud nodes on 2026-07-02 (never-started
    containers, driver segfaults, wedged workers). Default to the datacenter RTX
    tier; BLUEPRINT_RUNPOD_GPU_TYPES re-adds types for deliberate experiments."""
    monkeypatch.delenv("BLUEPRINT_RUNPOD_GPU_TYPES", raising=False)
    spec = _spec()
    assert "NVIDIA GeForce RTX 4090" not in spec.gpu_types
    # Price-aware capability-gated priority (P1-1): cheapest RTX-capable first.
    assert spec.gpu_types[0] == "NVIDIA A40"
    assert spec.gpu_types[1] == "NVIDIA RTX A6000"
    assert set(spec.gpu_types) == {
        "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40",
        "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    }
    assert not any(("H100" in g or "H200" in g) for g in spec.gpu_types)
    assert all(("GeForce" not in g) for g in spec.gpu_types)

    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_GPU_TYPES",
        "NVIDIA GeForce RTX 4090, NVIDIA L40S",
    )
    spec2 = _spec()
    assert spec2.gpu_types == ("NVIDIA GeForce RTX 4090", "NVIDIA L40S")


# ----------------------------- DigitalOcean GPU Droplets -----------------------------

def test_digitalocean_provider_is_registered() -> None:
    from blueprint_pipeline.gpu_render_providers import DigitalOceanRenderProvider

    assert "digitalocean" in {p["provider"] for p in list_render_providers()}
    assert isinstance(get_render_provider("digitalocean"), DigitalOceanRenderProvider)


def test_digitalocean_build_request_wraps_worker_in_user_data(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline.gpu_render_providers import DigitalOceanRenderProvider

    monkeypatch.delenv("BLUEPRINT_DO_GPU_SIZE", raising=False)
    monkeypatch.delenv("BLUEPRINT_DO_GPU_REGION", raising=False)
    spec = _spec()
    body = DigitalOceanRenderProvider().build_request(spec, tmp_path)
    assert body["size"] == "gpu-6000adax1-48gb"   # RT cores + 48GB default
    assert body["region"] == "atl1"
    assert body["image"] == "gpu-h100x1-base"     # NVIDIA AI/ML-ready (drivers+docker)
    assert body["min_gpu_ram_mb"] == spec.min_gpu_ram_mb
    assert body["requires_rtx"] is True
    assert "max_hourly_rate_usd" not in body
    ud = body["user_data"]
    assert "set -x" not in ud
    assert "set -euo pipefail" in ud
    assert "mkdir -p /root/blueprint-workspace/out" in ud
    assert '"docker", "run", "-d"' in ud
    assert '"--gpus", "all"' in ud
    assert '"--user", "0:0"' in ud
    assert '"-v", "/root/blueprint-workspace:/workspace"' in ud
    assert '"--workdir", "/workspace"' in ud
    assert spec.image in ud
    # env + bootstrap ride base64 so presigned URLs / scripts survive shell quoting
    assert "base64 -d" in ud
    assert "$(cat /root/blueprint_run.sh)" not in ud
    assert "subprocess.check_call(cmd)" in ud
    assert "blueprint_argv_decoded.json" in ud
    assert body["tags"] == ["blueprint-isaac-render"]


def test_digitalocean_capacity_preflight_blocks_empty_gpu_region_lists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-6000adax1-48gb,gpu-l40sx1-48gb")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1,nyc2")
    calls: list[tuple[str, str]] = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        assert token == "t-redacted"
        if method == "GET" and path == "/sizes?per_page=200":
            return 200, {
                "sizes": [
                    {
                        "slug": "gpu-6000adax1-48gb",
                        "available": True,
                        "regions": [],
                        "memory": 65536,
                        "price_hourly": 1.57,
                    },
                    {
                        "slug": "gpu-l40sx1-48gb",
                        "available": True,
                        "regions": [],
                        "memory": 65536,
                        "price_hourly": 1.57,
                    },
                ]
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)

    res = G.DigitalOceanRenderProvider().capacity_preflight()

    assert res["status"] == "blocked"
    assert res["blockers"] == ["digitalocean_gpu_size_region_unavailable"]
    assert res["region_candidates"] == ["atl1", "nyc2"]
    assert [row["matching_regions"] for row in res["considered_size_regions"]] == [[], []]
    assert calls == [("GET", "/sizes?per_page=200")]


def test_digitalocean_capacity_preflight_rejects_h100_h200_for_rtx_render(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.delenv("BLUEPRINT_DO_GPU_SIZES", raising=False)
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1,nyc2")

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert token == "t-redacted"
        if method == "GET" and path == "/sizes?per_page=200":
            return 200, {
                "sizes": [
                    {
                        "slug": "gpu-6000adax1-48gb",
                        "available": True,
                        "regions": [],
                        "memory": 65536,
                        "price_hourly": 1.57,
                    },
                    {
                        "slug": "gpu-l40sx1-48gb",
                        "available": True,
                        "regions": [],
                        "memory": 65536,
                        "price_hourly": 1.57,
                    },
                    {
                        "slug": "gpu-h100x1-80gb",
                        "available": True,
                        "regions": ["nyc2"],
                        "memory": 245760,
                        "price_hourly": 3.39,
                    },
                    {
                        "slug": "gpu-h200x1-141gb",
                        "available": True,
                        "regions": ["nyc2"],
                        "memory": 245760,
                        "price_hourly": 3.44,
                    },
                ]
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)

    default_budget = G.DigitalOceanRenderProvider().capacity_preflight(
        {"min_gpu_ram_mb": 48000}
    )
    launcher_budget = G.DigitalOceanRenderProvider().capacity_preflight(
        {"min_gpu_ram_mb": 48000, "max_hourly_rate_usd": 3.5}
    )

    assert default_budget["status"] == "blocked"
    assert "digitalocean_gpu_size_region_unavailable" in default_budget["blockers"]
    assert {
        row["size"] for row in default_budget["budget_policy"]["rejected_size_candidates"]
    } == {"gpu-h100x1-80gb", "gpu-h200x1-141gb"}
    assert launcher_budget["status"] == "blocked"
    assert launcher_budget["blockers"] == ["digitalocean_gpu_size_region_unavailable"]
    rejected = launcher_budget["render_capability_policy"]["rejected_size_candidates"]
    assert {row["size"] for row in rejected} == {
        "gpu-h100x1-80gb",
        "gpu-h200x1-141gb",
    }
    assert all(row["reason"] == "rtx_render_capability_missing" for row in rejected)

    monkeypatch.setenv(
        "BLUEPRINT_DO_GPU_SIZES",
        "gpu-h100x1-80gb,gpu-h200x1-141gb",
    )
    rtx_only = G.DigitalOceanRenderProvider().capacity_preflight(
        {"min_gpu_ram_mb": 48000, "max_hourly_rate_usd": 3.5}
    )
    assert rtx_only["status"] == "blocked"
    assert rtx_only["blockers"] == ["digitalocean_gpu_size_not_rtx_capable"]

    compute_only = G.DigitalOceanRenderProvider().capacity_preflight(
        {
            "min_gpu_ram_mb": 48000,
            "max_hourly_rate_usd": 3.5,
            "requires_rtx": False,
        }
    )
    assert compute_only["status"] == "available"
    assert compute_only["viable_size_regions"][0]["size"] == "gpu-h100x1-80gb"


def test_digitalocean_capacity_preflight_reports_viable_size_region(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-6000adax1-48gb,gpu-l40sx1-48gb")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1,nyc2")

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert token == "t-redacted"
        if method == "GET" and path == "/sizes?per_page=200":
            return 200, {
                "sizes": [
                    {
                        "slug": "gpu-6000adax1-48gb",
                        "available": True,
                        "regions": ["nyc2"],
                        "memory": 65536,
                        "price_hourly": 1.57,
                    }
                ]
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)

    res = G.DigitalOceanRenderProvider().capacity_preflight()

    assert res["status"] == "available"
    assert res["blockers"] == []
    assert res["viable_size_regions"][0]["size"] == "gpu-6000adax1-48gb"
    assert res["viable_size_regions"][0]["matching_regions"] == ["nyc2"]


def test_digitalocean_capacity_preflight_filters_below_requested_gpu_ram(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-4000adax1-20gb,gpu-6000adax1-48gb")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1")

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert token == "t-redacted"
        if method == "GET" and path == "/sizes?per_page=200":
            return 200, {
                "sizes": [
                    {
                        "slug": "gpu-4000adax1-20gb",
                        "available": True,
                        "regions": ["atl1"],
                        "memory": 32768,
                        "price_hourly": 0.76,
                    },
                    {
                        "slug": "gpu-6000adax1-48gb",
                        "available": True,
                        "regions": ["atl1"],
                        "memory": 65536,
                        "price_hourly": 1.57,
                    },
                ]
            }
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)

    res = G.DigitalOceanRenderProvider().capacity_preflight({"min_gpu_ram_mb": 48000})

    assert res["status"] == "available"
    assert res["size_candidates"] == ["gpu-6000adax1-48gb"]
    assert res["viable_size_regions"][0]["gpu_ram_mb"] == 48000
    assert res["gpu_ram_policy"]["rejected_size_candidates"] == [
        {
            "size": "gpu-4000adax1-20gb",
            "gpu_ram_mb": 20000,
            "min_gpu_ram_mb": 48000,
            "reason": "below_min_gpu_ram",
        }
    ]


def test_digitalocean_launch_fail_closed_without_token(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline.gpu_render_providers import DigitalOceanRenderProvider

    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tmp_path / "missing"))
    p = DigitalOceanRenderProvider()
    res = p.launch(tmp_path, {"name": "x"})
    assert res["status"] == "blocked"
    assert "digitalocean_token_missing" in res["blockers"]
    assert p.available()["available"] is False


def test_digitalocean_launch_blocks_without_prelaunch_guard_before_provider_call(
    monkeypatch, tmp_path: Path
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        return 202, {"droplet": {"id": 4242, "status": "new"}}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path),
    )

    assert res["status"] == "blocked"
    assert "digitalocean_render_prelaunch_spend_guard_missing" in res["blockers"]
    assert calls == []


def test_digitalocean_launch_creates_droplet_and_writes_id(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.delenv("BLUEPRINT_DO_SSH_KEY_IDS", raising=False)
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS_FILE", str(tmp_path / "missing_do_ssh_keys"))
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        assert token == "t-redacted"
        if method == "GET" and path == "/account/keys?per_page=200":
            return 200, {"ssh_keys": [{"id": 98765, "name": "worker-key"}]}
        if method == "POST" and path == "/droplets":
            assert "min_gpu_ram_mb" not in body
            assert "max_hourly_rate_usd" not in body
            assert "prelaunch_spend_guard" not in body
            assert body["ssh_keys"] == [98765]
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)
    p = G.DigitalOceanRenderProvider()
    res = p.launch(tmp_path, _with_prelaunch_guard(p.build_request(_spec(), tmp_path)))
    assert res["status"] == "launched"
    assert res["instance_id"] == "4242"
    assert res["mode"] == "do_gpu_droplet"
    assert res["ssh_key_configuration"]["source"] == "account_keys_api_first_available"
    assert (tmp_path / "started_do_droplet_id.txt").read_text() == "4242"


@pytest.mark.parametrize("status", [0, 202, 500])
def test_digitalocean_ambiguous_create_never_tries_another_region(
    monkeypatch, tmp_path: Path, status: int
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    monkeypatch.setattr(G.DigitalOceanRenderProvider, "_token", lambda _self: "token")
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    calls: list[tuple[str, str]] = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        return status, {"error": "unknown create outcome"}

    monkeypatch.setattr(G, "_do_call", fake_call)
    provider = G.DigitalOceanRenderProvider()
    result = provider.launch(
        tmp_path, _with_prelaunch_guard(provider.build_request(_spec(), tmp_path))
    )

    assert result["blockers"] == ["digitalocean_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert calls == [("POST", "/droplets")]
    assert "allocation_created" not in result


@pytest.mark.parametrize(
    "malformed_id",
    [None, True, {"id": 4242}, " 4242", "droplet-4242", 0, -1],
)
def test_digitalocean_malformed_success_id_is_ambiguous(
    monkeypatch, tmp_path: Path, malformed_id: object
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    monkeypatch.setattr(G.DigitalOceanRenderProvider, "_token", lambda _self: "token")
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    calls: list[tuple[str, str]] = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        return 202, {"droplet": {"id": malformed_id, "status": "new"}}

    monkeypatch.setattr(G, "_do_call", fake_call)
    provider = G.DigitalOceanRenderProvider()
    result = provider.launch(
        tmp_path, _with_prelaunch_guard(provider.build_request(_spec(), tmp_path))
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["digitalocean_create_outcome_ambiguous"]
    assert result["allocation_outcome_ambiguous"] is True
    assert calls == [("POST", "/droplets")]
    assert "instance_id" not in result
    assert not (tmp_path / "started_do_droplet_id.txt").exists()


def test_digitalocean_launch_regenerates_user_data_after_nonce_injection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    created: dict = {}

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert token == "t-redacted"
        if method == "POST" and path == "/droplets":
            created["body"] = body
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)
    provider = G.DigitalOceanRenderProvider()
    request = _with_prelaunch_guard(provider.build_request(_spec(), tmp_path))
    request["env"]["BLUEPRINT_LAUNCH_SESSION_ID"] = "nonce-123"

    res = provider.launch(tmp_path, request)

    assert res["status"] == "launched"
    body = created["body"]
    assert "env" not in body
    assert "_blueprint_worker_image" not in body
    match = re.search(
        r"echo ([A-Za-z0-9+/=]+) \| base64 -d > /root/blueprint_worker.env",
        body["user_data"],
    )
    assert match is not None
    env_text = base64.b64decode(match.group(1)).decode()
    assert "BLUEPRINT_LAUNCH_SESSION_ID=nonce-123" in env_text


def test_digitalocean_launch_uses_configured_ssh_keys_without_account_lookup(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123, fingerprint-abc")
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        assert path != "/account/keys?per_page=200"
        if method == "POST" and path == "/droplets":
            assert body["ssh_keys"] == [123, "fingerprint-abc"]
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        _with_prelaunch_guard(
            G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path)
        ),
    )

    assert res["status"] == "launched"
    assert res["ssh_key_configuration"]["source"] == "BLUEPRINT_DO_SSH_KEY_IDS"
    assert calls == [("POST", "/droplets")]


def test_digitalocean_launch_retries_gpu_size_region_unavailable(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-6000adax1-48gb,gpu-l40sx1-48gb")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1,nyc2")
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert method == "POST"
        assert path == "/droplets"
        calls.append((body["size"], body["region"]))
        if body["size"] == "gpu-l40sx1-48gb" and body["region"] == "nyc2":
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        return 422, {
            "error": '{"id":"unprocessable_entity","message":"Size is not available in this region."}\n'
        }

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        _with_prelaunch_guard(
            G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path)
        ),
    )

    assert res["status"] == "launched"
    assert res["instance_id"] == "4242"
    assert calls == [
        ("gpu-6000adax1-48gb", "atl1"),
        ("gpu-6000adax1-48gb", "nyc2"),
        ("gpu-l40sx1-48gb", "atl1"),
        ("gpu-l40sx1-48gb", "nyc2"),
    ]
    assert res["attempts"][-1]["size"] == "gpu-l40sx1-48gb"
    assert res["attempts"][-1]["region"] == "nyc2"
    assert res["budget_policy"]["max_hourly_rate_usd"] == pytest.approx(1.75)


def test_digitalocean_launch_retries_skip_below_requested_gpu_ram(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-4000adax1-20gb,gpu-6000adax1-48gb")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1")
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert method == "POST"
        assert path == "/droplets"
        calls.append((body["size"], body["region"]))
        assert body["size"] != "gpu-4000adax1-20gb"
        assert "min_gpu_ram_mb" not in body
        assert "max_hourly_rate_usd" not in body
        assert "prelaunch_spend_guard" not in body
        return 202, {"droplet": {"id": 4242, "status": "new"}}

    monkeypatch.setattr(G, "_do_call", fake_call)
    provider = G.DigitalOceanRenderProvider()
    request = _with_prelaunch_guard(provider.build_request(_spec(), tmp_path))
    request["min_gpu_ram_mb"] = 48000

    res = provider.launch(tmp_path, request)

    assert res["status"] == "launched"
    assert calls == [("gpu-6000adax1-48gb", "atl1")]
    assert res["gpu_ram_policy"]["rejected_size_candidates"] == [
        {
            "size": "gpu-4000adax1-20gb",
            "gpu_ram_mb": 20000,
            "min_gpu_ram_mb": 48000,
            "reason": "below_min_gpu_ram",
        }
    ]


def test_digitalocean_launch_blocks_h200_without_hourly_budget_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-h200x1-141gb")
    monkeypatch.delenv("BLUEPRINT_DO_MAX_HOURLY_RATE_USD", raising=False)

    def fake_call(method, path, body=None, *, token, timeout=90):
        raise AssertionError("must not create an over-budget DigitalOcean droplet")

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        _with_prelaunch_guard(
            G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path)
        ),
    )

    assert res["status"] == "blocked"
    assert "digitalocean_gpu_size_over_budget" in res["blockers"]
    assert res["budget_policy"]["rejected_size_candidates"] == [
        {
            "size": "gpu-h200x1-141gb",
            "hourly_rate_usd": 3.44,
            "reason": "over_max_hourly_rate",
        }
    ]


def test_digitalocean_launch_blocks_without_ssh_key(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.delenv("BLUEPRINT_DO_SSH_KEY_IDS", raising=False)
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS_FILE", str(tmp_path / "missing_do_ssh_keys"))

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert method == "GET"
        assert path == "/account/keys?per_page=200"
        return 200, {"ssh_keys": []}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        _with_prelaunch_guard(
            G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path)
        ),
    )

    assert res["status"] == "blocked"
    assert "digitalocean_ssh_key_missing" in res["blockers"]
    assert res["ssh_key_configuration"]["raw_provider_response_recorded"] is False


def test_digitalocean_terminate_deletes_droplet(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        return 204, {}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().terminate("4242")
    assert res["status"] == "terminated"
    assert ("DELETE", "/droplets/4242") in calls


def test_digitalocean_stop_warns_droplets_bill_while_off(monkeypatch, tmp_path: Path) -> None:
    """Powered-off droplets still bill full price; stop() must say so instead of
    silently pretending it saved money."""
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))

    def fake_call(method, path, body=None, *, token, timeout=90):
        return 201, {"action": {"id": 1, "status": "in-progress"}}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().stop("4242")
    assert res["status"] == "stopped"
    assert "billing" in json.dumps(res).lower()


def test_runpod_billable_inventory_is_api_confirmed_and_prefix_scoped(monkeypatch) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    monkeypatch.setattr(G.RunPodRenderProvider, "_key", lambda self: "secret")
    monkeypatch.setattr(
        G,
        "_runpod_call",
        lambda *args, **kwargs: (
            200,
            [
                {"id": "pod-1", "name": "blueprint-isaac-g1-supervised-a", "desiredStatus": "RUNNING", "costPerHr": 0.49},
                {"id": "pod-2", "name": "unrelated", "desiredStatus": "RUNNING", "costPerHr": 1.0},
            ],
        ),
    )
    result = G.RunPodRenderProvider().billable_inventory(
        name_prefix="blueprint-isaac-g1-supervised"
    )
    assert result["api_confirmed"] is True
    assert result["live_resource_count"] == 1
    assert result["resources"][0]["instance_id"] == "pod-1"
    assert result["raw_provider_response_recorded"] is False


def test_runpod_billable_inventory_includes_explicit_legacy_warm_candidate(monkeypatch) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    monkeypatch.setattr(G.RunPodRenderProvider, "_key", lambda self: "secret")
    monkeypatch.setattr(
        G,
        "_runpod_call",
        lambda *args, **kwargs: (
            200,
            [
                {
                    "id": "legacy-warm-id",
                    "name": "old-name-outside-current-prefix",
                    "desiredStatus": "EXITED",
                    "costPerHr": 0.49,
                },
                {"id": "unrelated", "name": "unrelated"},
            ],
        ),
    )

    result = G.RunPodRenderProvider(
        warm_candidates=("legacy-warm-id",)
    ).billable_inventory(name_prefix="blueprint-isaac-g1")

    assert result["api_confirmed"] is True
    assert result["live_resource_count"] == 1
    assert result["resources"][0]["instance_id"] == "legacy-warm-id"
    assert result["explicit_warm_candidate_ids_checked"] == ["legacy-warm-id"]


def test_digitalocean_billable_inventory_counts_powered_off_resources(monkeypatch) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    monkeypatch.setattr(G.DigitalOceanRenderProvider, "_token", lambda self: "secret")
    monkeypatch.setattr(
        G,
        "_do_call",
        lambda *args, **kwargs: (
            200,
            {
                "droplets": [
                    {"id": 4, "name": "blueprint-isaac-g1-supervised-a", "status": "off", "size_slug": "gpu", "region": {"slug": "tor1"}},
                    {"id": 5, "name": "unrelated", "status": "active"},
                ]
            },
        ),
    )
    result = G.DigitalOceanRenderProvider().billable_inventory(
        name_prefix="blueprint-isaac-g1-supervised"
    )
    assert result["api_confirmed"] is True
    assert result["live_resource_count"] == 1
    assert result["resources"][0]["status"] == "off"
