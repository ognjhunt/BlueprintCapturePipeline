from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_runpod_storage_volume as storage
from blueprint_pipeline.groot_oscar_runpod_storage_volume import (
    build_storage_volume_admission,
    launch_detached,
    run_storage_model_volume,
)
from blueprint_pipeline.paid_resource_admission import (
    require_paid_resource_admission,
)


def _admission(**overrides):
    values = {
        "data_center_id": "US-WA-1",
        "volume_size_gib": 50,
        "storage_ttl_seconds": 14_400,
        "storage_hourly_rate_usd": 0.005,
        "max_storage_spend_usd": 0.05,
        "builder_ttl_seconds": 7200,
        "inventory_verified_zero": True,
        "credentials_verified": True,
        "source_clean": True,
        "local_staging_bytes": 2 * 1024**3,
        "paid_mutation_authorized": True,
        "watchdog_armed_before_allocation": True,
    }
    values.update(overrides)
    return build_storage_volume_admission(**values)


def test_storage_volume_admission_accepts_bounded_no_gpu_tuple() -> None:
    admission = _admission()
    assert admission["status"] == "admitted"
    grant = require_paid_resource_admission(
        admission,
        resource_class="model_volume",
        expected_schema_version=storage.SCHEMA_VERSION,
    )
    assert grant.resource_class == "model_volume"
    assert admission["limits"]["runpod_gpu_pod_limit"] == 0


def test_storage_volume_admission_rejects_one_hour_and_near_canary_deadlines() -> None:
    one_hour = _admission(storage_ttl_seconds=3600)
    assert "storage_model_volume_ttl_outside_guardrail" in one_hour["blockers"]
    assert (
        "storage_model_volume_ttl_does_not_cover_builder_and_canary"
        in one_hour["blockers"]
    )
    near = _admission(storage_ttl_seconds=10_000)
    assert "storage_model_volume_ttl_does_not_cover_builder_and_canary" in near["blockers"]


def test_storage_route_has_watchdog_lease_ledger_and_no_pod_create() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_runpod_storage_volume.py"
    ).read_text(encoding="utf-8")
    run = source[source.index("def run_storage_model_volume(") : source.index("def launch_detached(")]
    assert '"/pods"' not in run
    assert run.index("acquire_paid_provider_lane_lease(") < run.index(
        '"POST",\n            "/networkvolumes"'
    )
    assert run.index("open_pending_teardown(") < run.index(
        '"POST",\n            "/networkvolumes"'
    )
    assert run.index("bind_pending_teardown_instance(") > run.index(
        '"POST",\n            "/networkvolumes"'
    )
    assert run.index("_arm_watchdog(") < run.index("require_paid_resource_admission(")
    assert "build_runpod_network_volume_evidence(" in run
    assert "storage_model_volume_deadline_too_near_for_canary" in run


def _patch_preallocation(
    monkeypatch: pytest.MonkeyPatch, *, runpod_key: str = "runpod-key"
) -> None:
    class Provider:
        @staticmethod
        def _key() -> str:
            return runpod_key

    monkeypatch.setattr(storage, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        storage,
        "_source_identity",
        lambda _root: ("a" * 40, hashlib.sha256(b"").hexdigest(), False),
    )
    monkeypatch.setattr(storage, "_read_secret", lambda _path: "secret")
    monkeypatch.setattr(storage, "_read_private_secret", lambda _path: "secret")
    monkeypatch.setattr(
        storage,
        "_host_key_material",
        lambda _path: ("private", "public", "SHA256:" + "d" * 43),
    )
    monkeypatch.setattr(
        storage,
        "_live_profile",
        lambda **_kwargs: (
            {"status": "verified", "observed": {"price_hourly_usd": 0.16}},
            [],
        ),
    )
    monkeypatch.setattr(
        storage,
        "preflight_runpod_s3",
        lambda **_kwargs: {"status": "ready", "blockers": []},
    )
    monkeypatch.setattr(
        storage,
        "build_model_cache_wheelhouse",
        lambda **_kwargs: {
            "status": "ready",
            "wheelhouse_path": "wheels",
            "manifest_path": "manifest.json",
        },
    )
    monkeypatch.setattr(storage, "_matching_resources", lambda **_kwargs: ([], [], True))


def _inputs(tmp_path: Path) -> dict:
    builder = tmp_path / "builder.json"
    builder.write_text(
        json.dumps(
            {
                "provider": "digitalocean",
                "purpose": "model_cache_s3",
                "platform": "linux/amd64",
                "python_runtime_verified": True,
                "python_version": "3.12",
                "dependency_lock_verified": True,
                "dependency_wheelhouse_verified": True,
                "dns_resolution_verified": True,
                "outbound_https_verified": True,
                "s3_endpoint_host": "s3api-us-wa-1.runpod.io",
                "free_disk_bytes": 320 * 1024**3,
                "independent_teardown_watchdog": True,
                "ssh_host_key_sha256": "SHA256:" + "d" * 43,
                "ssh_host_key_independently_verified": True,
                "ssh_host_key_verification_method": "launch_bound_generated_host_key",
                "expected_source_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    spend = tmp_path / "spend.json"
    spend.write_text(
        json.dumps(
            {
                "paid_mutation_authorized": True,
                "max_spend_usd": 0.35,
                "hard_ttl_seconds": 7200,
                "one_resource_limit": True,
                "independent_teardown_watchdog": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": tmp_path / "out",
        "repo_root": tmp_path,
        "data_center_id": "US-WA-1",
        "volume_size_gib": 50,
        "storage_ttl_seconds": 14_400,
        "storage_hourly_rate_usd": 0.005,
        "max_storage_spend_usd": 0.05,
        "builder_evidence_path": builder,
        "builder_spend_path": spend,
        "digitalocean_token_file": tmp_path / "do-token",
        "hf_token_file": tmp_path / "hf-token",
        "runpod_s3_access_key_file": tmp_path / "s3-access",
        "runpod_s3_secret_key_file": tmp_path / "s3-secret",
        "login_private_key": tmp_path / "login-key",
        "host_private_key": tmp_path / "host-key",
        "ssh_key_id": 7,
        "region": "sfo3",
        "allow_paid": True,
    }


def test_builder_live_preflight_failure_blocks_before_runpod_volume_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preallocation(monkeypatch)
    monkeypatch.setattr(
        storage,
        "_live_profile",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("profile unavailable")),
    )
    monkeypatch.setattr(
        storage,
        "_runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider mutation reached")
        ),
    )
    result = run_storage_model_volume(**_inputs(tmp_path))
    assert result["status"] == "blocked_before_allocation"
    assert result["provider_mutation_attempted"] is False


def test_duplicate_lane_blocks_before_watchdog_or_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preallocation(monkeypatch)
    monkeypatch.setattr(
        storage,
        "acquire_paid_provider_lane_lease",
        lambda **_kwargs: {"status": "blocked", "blockers": ["already-owned"]},
    )
    monkeypatch.setattr(
        storage,
        "_arm_watchdog",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("watchdog started")),
    )
    monkeypatch.setattr(
        storage,
        "_runpod_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider mutation reached")
        ),
    )
    result = run_storage_model_volume(**_inputs(tmp_path))
    assert result["status"] == "blocked_before_allocation"
    assert result["blockers"] == ["already-owned"]


def test_storage_detached_launch_is_single_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 1234

    monkeypatch.setattr(storage.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    launched = launch_detached(output_dir=tmp_path, run_arguments=["--allow-paid"])
    assert launched["status"] == "supervisor_started"
    with pytest.raises(ValueError, match="already_has_supervisor"):
        launch_detached(output_dir=tmp_path, run_arguments=["--allow-paid"])
