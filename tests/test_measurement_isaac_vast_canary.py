from __future__ import annotations

import json
import os
import time
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.measurement_isaac_runtime_release import (
    RUNTIME_IMAGE,
    build_measurement_isaac_runtime_release,
)
from blueprint_pipeline.measurement_isaac_vast_bundle import RECEIPT_SCHEMA_VERSION
from blueprint_pipeline.measurement_isaac_vast_canary import (
    _bootstrap_script,
    _lightwheel_sink_bootstrap_script,
    _watchdog_valid,
    run_measurement_isaac_vast_canary,
)
from blueprint_pipeline.lightwheel_sink_isaac_bundle import (
    RECEIPT_SCHEMA_VERSION as LIGHTWHEEL_SINK_RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


SHA = "a" * 40
D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
INPUT_URL = "https://objects.example/input?signature=input-secret"
PUT_URL = "https://objects.example/output?signature=put-secret"
GET_URL = "https://objects.example/output?signature=get-secret"


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _receipt() -> dict:
    release = build_measurement_isaac_runtime_release()
    value = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_commit_sha": SHA,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": release["runtime_release_digest"],
        "bundle_manifest_digest": D2,
        "input_bundle_digest": D1,
        "input_bundle_size_bytes": 1000,
        "execution_request_digests": [D2, D3],
        "request_count": 2,
        "rtx_openusd_runtime_preflight_required": True,
        "rtx_renderer": "RayTracedLighting",
        "rtx_smoke_resolution": [64, 64],
        "rtx_required_output_kinds": ["rgb", "depth", "semantic_segmentation"],
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_development_input_bundle_only",
    }
    value["bundle_receipt_digest"] = canonical_digest(value, digest_field="bundle_receipt_digest")
    return value


def _bound_request() -> dict:
    release = build_measurement_isaac_runtime_release()
    value = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "measurement_isaac_canary",
        "source_commit_sha": SHA,
        "worker_image_digest": RUNTIME_IMAGE,
        "operation_input_bundle_digest": D1,
        "operation_request_digest": D2,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 60,
        "retry_cap": 0,
        "authority_id": "fixture-authority",
        "vast_preferred_gpu_keywords": ["L40"],
        "request_digest": D3,
        "bound_provider": "vast",
        "bound_preflight_digest": D2,
        "bound_checkout_source_commit": SHA,
        "bound_checkout_clean": True,
        "measurement_isaac_runtime_release_digest": release["runtime_release_digest"],
        "provider_mutation_authorized": True,
    }
    value["bound_request_digest"] = canonical_digest(value, digest_field="bound_request_digest")
    return value


def _preflight() -> dict:
    return {
        "provider": "vast",
        "watchdog": {
            "status": "armed",
            "independent_process": True,
            "pid": 123,
            "deadline_epoch": 2000,
            "name_prefix": "blueprint-measurement-isaac-",
        },
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 120 * 1024**3,
        "on_demand_price_usd_per_hour": 0.5,
    }


class _Provider:
    name = "vast"

    def __init__(self, *, initially_live: bool = False):
        self.initially_live = initially_live
        self.launched = False
        self.requests: list[dict] = []

    def billable_inventory(self, *, name_prefix: str) -> dict:
        return {
            "api_confirmed": True,
            "live_resource_count": 1 if self.initially_live or self.launched else 0,
            "resources": [],
        }

    def build_request(self, spec, job_dir):
        assert spec.image == RUNTIME_IMAGE
        assert spec.requires_rtx is True
        assert spec.gpu_count == 1
        assert spec.env["ACCEPT_EULA"] == "Y"
        assert spec.env["BLUEPRINT_MEASUREMENT_ISAAC_INPUT_GET_URL"] == INPUT_URL
        return {"create_payload": {"env": dict(spec.env)}}

    def launch(self, job_dir, request, **kwargs):
        self.requests.append(request)
        self.launched = True
        return {"status": "launched", "instance_id": "42"}

    def terminate(self, instance_id):
        self.launched = False
        return {"status": "terminated", "instance_id": instance_id}


def test_bootstrap_verifies_bundle_and_uses_exact_isaac_python() -> None:
    script = _bootstrap_script()
    assert 'mktemp -d "${TMPDIR:-/tmp}/blueprint-measurement-isaac.XXXXXX"' in script
    assert "/work/measurement_isaac" not in script
    assert "measurement_isaac_input_digest_mismatch" in script
    assert "measurement_isaac_input_member_unsafe" in script
    assert "measurement_isaac_source_digest_mismatch" in script
    assert script.count("/isaac-sim/python.sh") == 3
    assert "python3" not in script
    assert "BLUEPRINT_MEASUREMENT_ISAAC_OUTPUT_PUT_URL" in script
    assert 'headers={"Content-Type": "application/zip"}' in script
    assert 'headers={"Content-Type": "application/json"}' not in script


def test_lightwheel_bootstrap_runs_sink_worker_and_rehashes_every_member() -> None:
    script = _lightwheel_sink_bootstrap_script()
    assert "run_lightwheel_sink_isaac_bundle.py" in script
    assert "BLUEPRINT_LIGHTWHEEL_SINK_INPUT_BUNDLE_DIGEST" in script
    assert "lightwheel_sink_member_digest_mismatch" in script
    assert "source_files" in script and "asset_files" in script
    # download+verify, worker, fallback-result writer, output upload
    assert script.count("/isaac-sim/python.sh") == 4
    # A hung worker must be killed and a crashed worker must still ship evidence.
    assert "timeout --signal=TERM --kill-after=60 960" in script
    assert "worker_log_tail" in script
    assert "lightwheel_sink_worker_no_terminal_result" in script
    assert "host_driver" in script


def test_watchdog_validation_binds_live_process_and_exact_nonsymlink_evidence(
    tmp_path: Path,
) -> None:
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    watchdog = {
        "status": "armed",
        "independent_process": True,
        "pid": os.getpid(),
        "deadline_epoch": time.time() + 120,
        "provider": "vast",
        "pod_name_prefix": "blueprint-measurement-isaac-",
        "watchdog_out_dir": str(watchdog_dir),
    }
    evidence = watchdog_dir / "groot_oscar_runpod_canary_watchdog.json"
    evidence.write_text(json.dumps(watchdog), encoding="utf-8")
    assert _watchdog_valid(watchdog, now_epoch=time.time(), hard_ttl_seconds=60)

    symlink_root = tmp_path / "watchdog-link"
    symlink_root.symlink_to(watchdog_dir, target_is_directory=True)
    watchdog["watchdog_out_dir"] = str(symlink_root)
    evidence.write_text(json.dumps(watchdog), encoding="utf-8")
    assert not _watchdog_valid(watchdog, now_epoch=time.time(), hard_ttl_seconds=60)


def test_canary_records_exact_instance_for_watchdog_and_requests_close(
    tmp_path: Path, monkeypatch
) -> None:
    watchdog_dir = tmp_path / "watchdog"
    watchdog_dir.mkdir()
    preflight = _preflight()
    preflight["watchdog"]["watchdog_out_dir"] = str(watchdog_dir)
    provider = _Provider()
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_vast_canary."
        "validate_measurement_isaac_vast_runtime_result",
        lambda value, **_kwargs: dict(value),
    )
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_isaac_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=preflight,
        job_dir=tmp_path / "canary",
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: {
            "runtime_result_digest": D3,
            "status": "passed",
        },
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "completed"
    assert provider.requests[0]["preferred_gpu_keywords"] == ["L40"]
    started_id = watchdog_dir / "started_vast_instance_id.txt"
    assert started_id.read_text(encoding="utf-8") == "42"
    assert started_id.stat().st_mode & 0o777 == 0o600
    cancel = json.loads(
        (watchdog_dir / "groot_oscar_runpod_canary_watchdog_cancel.json").read_text(
            encoding="utf-8"
        )
    )
    assert cancel["instance_id"] == "42"
    assert cancel["provider_absence_confirmed"] is True


def test_default_fetcher_treats_missing_output_as_not_ready(monkeypatch) -> None:
    def missing(*_args, **_kwargs):
        raise urllib.error.HTTPError(GET_URL, 404, "Not Found", {}, None)

    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_vast_canary.safe_http_request",
        missing,
    )

    with pytest.raises(FileNotFoundError, match="measurement_isaac_output_http:404"):
        from blueprint_pipeline.measurement_isaac_vast_canary import (
            _default_result_fetcher,
        )

        _default_result_fetcher(GET_URL)


def test_canary_tears_down_and_persists_no_signed_urls(tmp_path: Path, monkeypatch) -> None:
    provider = _Provider()
    raw_result = {"runtime_result_digest": D3, "status": "passed"}
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_vast_canary."
        "validate_measurement_isaac_vast_runtime_result",
        lambda value, **_kwargs: dict(value),
    )
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_isaac_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: raw_result,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "completed"
    assert result["provider_zero_verified"] is True
    assert result["provider_mutations_performed"] == 2
    assert result["r7_admission_created"] is False
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    for secret in (INPUT_URL, PUT_URL, GET_URL):
        assert secret not in persisted
    assert not list((tmp_path / "leases").glob("*.lease.json"))


def test_canary_dispatches_external_sink_bundle_with_exact_environment(
    tmp_path: Path, monkeypatch
) -> None:
    release = build_measurement_isaac_runtime_release()
    receipt = {
        "schema_version": LIGHTWHEEL_SINK_RECEIPT_SCHEMA_VERSION,
        "source_commit_sha": SHA,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": release["runtime_release_digest"],
        "bundle_manifest_digest": D2,
        "input_bundle_digest": D1,
        "input_bundle_size_bytes": 1000,
        "source_model_digest": D1,
        "texture_manifest_digest": D2,
        "wrapper_digest": D3,
        "test_configuration_digest": D1,
        "asset_file_count": 6,
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_external_asset_development_input_only",
    }
    receipt["bundle_receipt_digest"] = canonical_digest(
        receipt, digest_field="bundle_receipt_digest"
    )
    request = _bound_request()
    request.pop("bound_request_digest")
    request["capture_profile"] = "external_generated_asset"
    request["bound_request_digest"] = canonical_digest(
        request, digest_field="bound_request_digest"
    )
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_isaac_vast_canary."
        "validate_lightwheel_sink_isaac_runtime_result",
        lambda value, **_kwargs: dict(value),
    )
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_isaac_vast_canary(
        bound_request=request,
        bundle_receipt=receipt,
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: {"runtime_result_digest": D3, "status": "passed"},
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda *_args: True,
    )
    assert result["status"] == "completed"
    env = provider.requests[0]["create_payload"]["env"]
    assert env["BLUEPRINT_LIGHTWHEEL_SINK_SOURCE_COMMIT"] == SHA
    assert env["BLUEPRINT_LIGHTWHEEL_SINK_INPUT_BUNDLE_DIGEST"] == D1
    assert result["claim_ceiling"] == "isaac_articulation_and_scripted_franka_contact_development"


def test_canary_records_fetch_failure_and_still_tears_down(tmp_path: Path) -> None:
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1002.0])

    def failed_fetch(_url: str):
        raise RuntimeError("provider response intentionally omitted")

    result = run_measurement_isaac_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=failed_fetch,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda *_args: True,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == [
        "measurement_isaac_output_fetch_failed:RuntimeError",
    ]
    assert result["provider_zero_verified"] is True
    assert provider.launched is False
    assert (tmp_path / "measurement_isaac_vast_execution.json").is_file()
    assert (tmp_path / "teardown_receipt.json").is_file()


def test_canary_refuses_nonzero_provider_and_missing_grant(tmp_path: Path) -> None:
    provider = _Provider(initially_live=True)
    with pytest.raises(Exception, match="provider_not_zero_before_launch"):
        run_measurement_isaac_vast_canary(
            bound_request=_bound_request(),
            bundle_receipt=_receipt(),
            preflight=_preflight(),
            job_dir=tmp_path / "nonzero",
            input_bundle_get_url=INPUT_URL,
            output_put_url=PUT_URL,
            output_get_url=GET_URL,
            provider=provider,
            paid_resource_admission_grant=_grant(),
            clock=lambda: 1000.0,
            watchdog_validator=lambda _watchdog, _now, _ttl: True,
        )
    assert provider.requests == []

    clean_provider = _Provider()
    with pytest.raises(Exception, match="paid_resource_admission_grant_missing"):
        run_measurement_isaac_vast_canary(
            bound_request=_bound_request(),
            bundle_receipt=_receipt(),
            preflight=_preflight(),
            job_dir=tmp_path / "missing-grant",
            input_bundle_get_url=INPUT_URL,
            output_put_url=PUT_URL,
            output_get_url=GET_URL,
            provider=clean_provider,
            paid_resource_admission_grant=None,
        )
    assert clean_provider.requests == []
