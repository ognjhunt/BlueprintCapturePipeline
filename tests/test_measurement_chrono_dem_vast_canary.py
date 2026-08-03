from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.measurement_chrono_dem_cuda_adapter import (
    EXPECTED_ENGINE_VERSION,
    EXPECTED_SOURCE_COMMIT,
)
from blueprint_pipeline.measurement_chrono_dem_runtime_release import (
    BUILD_CONFIGURATION,
    REQUIRED_DEBIAN_PACKAGES,
    RUNTIME_IMAGE,
    SOURCE_REPOSITORY,
    SOURCE_TAG_OBJECT,
    build_measurement_chrono_dem_runtime_release,
)
from blueprint_pipeline.measurement_chrono_dem_vast_bundle import RECEIPT_SCHEMA_VERSION
from blueprint_pipeline.measurement_chrono_dem_vast_canary import (
    FAILURE_RESULT_SCHEMA_VERSION,
    MeasurementChronoDemVastCanaryError,
    _bootstrap_script,
    run_measurement_chrono_dem_vast_canary,
    validate_measurement_chrono_dem_vast_failure_result,
    validate_measurement_chrono_dem_vast_runtime_result,
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
    release = build_measurement_chrono_dem_runtime_release()
    value = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_commit_sha": SHA,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": release["runtime_release_digest"],
        "chrono_source_commit": EXPECTED_SOURCE_COMMIT,
        "bundle_manifest_digest": D2,
        "input_bundle_digest": D1,
        "input_bundle_size_bytes": 1000,
        "execution_request_digests": [D2, D3],
        "request_count": 2,
        "required_backend": "cuda",
        "replay_count": 2,
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_development_input_bundle_only",
    }
    value["bundle_receipt_digest"] = canonical_digest(value, digest_field="bundle_receipt_digest")
    return value


def _bound_request() -> dict:
    release = build_measurement_chrono_dem_runtime_release()
    value = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "measurement_chrono_dem_canary",
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
        "measurement_chrono_dem_runtime_release_digest": release["runtime_release_digest"],
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
            "name_prefix": "blueprint-measurement-chrono-dem-",
        },
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 120 * 1024**3,
        "on_demand_price_usd_per_hour": 0.5,
    }


def _failure_result(*, stage: str = "probe_build") -> dict:
    receipt = _receipt()
    value = {
        "schema_version": FAILURE_RESULT_SCHEMA_VERSION,
        "status": "failed",
        "failure_stage": stage,
        "exit_code": 1,
        "log_excerpt": "compiler failed at an exact source location",
        "source_commit_sha": SHA,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": receipt["runtime_release_digest"],
        "input_bundle_digest": D1,
        "chrono_source_commit": EXPECTED_SOURCE_COMMIT,
        "raw_secret_values_recorded": False,
        "proof_effect": "provider_execution_failure_evidence_only",
        "claim_ceiling": "no_chrono_runtime_execution_evidence",
    }
    value["failure_result_digest"] = canonical_digest(
        value, digest_field="failure_result_digest"
    )
    return value


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
        assert spec.requires_rtx is False
        assert spec.gpu_count == 1
        assert spec.env["BLUEPRINT_MEASUREMENT_CHRONO_DEM_INPUT_GET_URL"] == INPUT_URL
        assert spec.env["BLUEPRINT_MEASUREMENT_CHRONO_DEM_SOURCE_UPSTREAM_COMMIT"] == (
            EXPECTED_SOURCE_COMMIT
        )
        return {"create_payload": {"env": dict(spec.env)}}

    def launch(self, job_dir, request, **kwargs):
        self.requests.append(request)
        self.launched = True
        return {"status": "launched", "instance_id": "42"}

    def inspect(self, instance_id):
        return {
            "status": "observed",
            "instance_id": instance_id,
            "actual_status": "running" if self.launched else "stopped",
            "api_confirmed": True,
            "provider_absence_confirmed": False,
        }

    def terminate(self, instance_id):
        self.launched = False
        return {"status": "terminated", "instance_id": instance_id}


def test_bootstrap_binds_exact_source_build_cuda_bundle_and_signed_upload() -> None:
    script = _bootstrap_script()
    assert "measurement_chrono_dem_input_digest_mismatch" in script
    assert "measurement_chrono_dem_input_member_unsafe" in script
    assert "measurement_chrono_dem_source_digest_mismatch" in script
    assert SOURCE_REPOSITORY in script
    assert SOURCE_TAG_OBJECT in script
    assert 'checkout --detach "$BLUEPRINT_MEASUREMENT_CHRONO_DEM_SOURCE_UPSTREAM_COMMIT"' in script
    assert (
        "apt-get install -y --no-install-recommends " + " ".join(REQUIRED_DEBIAN_PACKAGES) in script
    )
    assert script.index("apt-get install -y --no-install-recommends") < script.index(
        'python3 - "$archive" "$bundle"'
    )
    assert "-DCH_ENABLE_MODULE_DEM=" + BUILD_CONFIGURATION["CH_ENABLE_MODULE_DEM"] in script
    assert "-DCHRONO_CUDA_ARCHITECTURES=native" in script
    assert 'cmake --build "$chrono_build" --target install --parallel 2' in script
    assert script.count('headers={"Content-Type": "application/zip"}') == 2
    assert 'headers={"Content-Type": "application/json"}' not in script
    assert "upload_terminal_failure" in script
    assert 'failure_stage="chrono_build_install"' in script
    assert 'failure_stage="probe_build"' in script
    assert 're.sub(r"https?://\\\\S+", "<redacted-url>", excerpt)' in script


def test_probe_uses_self_contained_pi_constant_not_uninstalled_chrono_header() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/measurement_chrono_dem_cuda_probe.cpp"
    ).read_text(encoding="utf-8")
    assert '"chrono/core/ChConstants.h"' not in source
    assert "constexpr float kPi" in source


def test_runtime_validator_enforces_cuda_identity_and_claim_ceiling(monkeypatch) -> None:
    receipt = _receipt()
    bound = _bound_request()
    runtime = {
        "engine_version": EXPECTED_ENGINE_VERSION,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "chrono_dem_module_used": True,
        "cuda_available": True,
        "cuda_device_count": 1,
        "cpu_fallback_used": False,
        "deterministic_replay_match": True,
        "q_gran_qualification_created": False,
        "r5_evidence_created": False,
        "r6_decision_created": False,
        "r7_admission_created": False,
        "physical_success_established": False,
    }
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_chrono_dem_vast_canary."
        "validate_measurement_adapter_execution_bundle",
        lambda value: {"receipt": {"status": "completed", "runtime_observations": runtime}},
    )
    result = {
        "schema_version": "measurement_chrono_dem_cuda_vast_runtime_result.v1",
        "status": "passed",
        "source_commit_sha": SHA,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": receipt["runtime_release_digest"],
        "input_bundle_digest": D1,
        "bundle_manifest_digest": D2,
        "chrono_source_commit": EXPECTED_SOURCE_COMMIT,
        "execution_bundle_count": 2,
        "execution_bundles": [{}, {}],
        "aggregate_metrics": {
            "case_count": 2,
            "minimum_spread_ratio": 1.0,
            "maximum_spread_ratio": 1.2,
            "mean_ground_reaction_force_n": 3.0,
            "within_envelope_case_count": 2,
        },
        "blockers": [],
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
        "physical_material_characterization_included": False,
        "qualification_created": False,
        "r5_evidence": False,
        "r6_decision": False,
        "r7_admission": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "raw_secret_values_recorded": False,
        "proof_effect": "development_execution_only",
        "claim_ceiling": "chrono_dem_cuda_granular_development",
    }
    result["runtime_result_digest"] = canonical_digest(result, digest_field="runtime_result_digest")
    assert (
        validate_measurement_chrono_dem_vast_runtime_result(
            result, bound_request=bound, bundle_receipt=receipt
        )["status"]
        == "passed"
    )

    result["qualification_created"] = True
    result["runtime_result_digest"] = canonical_digest(result, digest_field="runtime_result_digest")
    with pytest.raises(MeasurementChronoDemVastCanaryError, match="qualification_created"):
        validate_measurement_chrono_dem_vast_runtime_result(
            result, bound_request=bound, bundle_receipt=receipt
        )


def test_failure_validator_preserves_exact_provider_failure_without_claim_upgrade() -> None:
    result = _failure_result()
    assert (
        validate_measurement_chrono_dem_vast_failure_result(
            result,
            bound_request=_bound_request(),
            bundle_receipt=_receipt(),
        )["failure_stage"]
        == "probe_build"
    )

    result["log_excerpt"] = "https://objects.example/output?X-Amz-Signature=secret"
    result["failure_result_digest"] = canonical_digest(
        result, digest_field="failure_result_digest"
    )
    with pytest.raises(MeasurementChronoDemVastCanaryError, match="log_excerpt_invalid"):
        validate_measurement_chrono_dem_vast_failure_result(
            result,
            bound_request=_bound_request(),
            bundle_receipt=_receipt(),
        )


def test_canary_launches_once_tears_down_and_preserves_no_signed_urls(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _Provider()
    raw_result = {"runtime_result_digest": D3, "status": "passed"}
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_chrono_dem_vast_canary."
        "validate_measurement_chrono_dem_vast_runtime_result",
        lambda value, **_kwargs: dict(value),
    )
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_chrono_dem_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path / "canary",
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
    assert result["qualification_created"] is False
    assert provider.launched is False
    assert len(provider.requests) == 1
    assert provider.requests[0]["prelaunch_spend_guard"]["retry_cap"] == 0
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "canary").rglob("*.json")
    )
    assert "signature=input-secret" not in persisted
    assert "signature=put-secret" not in persisted
    assert "signature=get-secret" not in persisted


def test_canary_blocks_before_launch_when_provider_is_not_zero(tmp_path: Path) -> None:
    provider = _Provider(initially_live=True)
    with pytest.raises(MeasurementChronoDemVastCanaryError, match="provider_not_zero"):
        run_measurement_chrono_dem_vast_canary(
            bound_request=_bound_request(),
            bundle_receipt=_receipt(),
            preflight=_preflight(),
            job_dir=tmp_path / "canary",
            input_bundle_get_url=INPUT_URL,
            output_put_url=PUT_URL,
            output_get_url=GET_URL,
            provider=provider,
            paid_resource_admission_grant=_grant(),
            watchdog_validator=lambda _watchdog, _now, _ttl: True,
        )
    assert provider.requests == []


def test_canary_times_out_once_then_tears_down_without_retry(tmp_path: Path) -> None:
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1061.0, 1062.0])
    result = run_measurement_chrono_dem_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path / "canary",
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: (_ for _ in ()).throw(FileNotFoundError("not-ready")),
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == ["measurement_chrono_dem_output_timeout"]
    assert result["provider_zero_verified"] is True
    assert result["provider_mutations_performed"] == 2
    assert len(provider.requests) == 1
    assert provider.launched is False


def test_canary_stops_polling_when_provider_is_terminal_without_output(tmp_path: Path) -> None:
    class _TerminalProvider(_Provider):
        def inspect(self, instance_id):
            return {
                "status": "absent",
                "instance_id": instance_id,
                "api_confirmed": True,
                "provider_absence_confirmed": True,
            }

    provider = _TerminalProvider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_chrono_dem_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path / "canary",
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: (_ for _ in ()).throw(FileNotFoundError("not-ready")),
        sleeper=lambda _seconds: pytest.fail("terminal provider must not sleep"),
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == [
        "measurement_chrono_dem_provider_terminal_without_output"
    ]
    assert result["provider_zero_verified"] is True
    assert result["provider_mutations_performed"] == 2
    assert len(provider.requests) == 1
    assert provider.launched is False


def test_canary_accepts_bounded_provider_failure_and_tears_down(tmp_path: Path) -> None:
    provider = _Provider()
    failure = _failure_result(stage="probe_build")
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_chrono_dem_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path / "canary",
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: failure,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == [
        "measurement_chrono_dem_provider_reported_failure:probe_build"
    ]
    assert result["provider_failure_result_digest"] == failure["failure_result_digest"]
    assert result["development_execution_completed"] is False
    assert result["proof_effect"] == "none"
    assert result["provider_zero_verified"] is True
    assert provider.launched is False


def test_canary_preserves_failed_teardown_and_provider_nonzero_evidence(tmp_path: Path) -> None:
    class _FailedTeardownProvider(_Provider):
        def terminate(self, instance_id):
            return {"status": "failed", "instance_id": instance_id}

    provider = _FailedTeardownProvider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_chrono_dem_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path / "canary",
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: {"runtime_result_digest": D3, "status": "passed"},
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )

    assert result["status"] == "failed"
    assert "measurement_chrono_dem_teardown_verification_failed" in result["blockers"]
    assert "measurement_chrono_dem_paid_lane_release_blocked" in result["blockers"]
    assert result["provider_zero_verified"] is False
    assert provider.launched is True
    assert (tmp_path / "canary" / "teardown_receipt.json").read_text(encoding="utf-8")
    assert (tmp_path / "canary" / "provider_zero_verification.json").read_text(encoding="utf-8")


def test_canary_converts_raised_terminate_into_terminal_evidence(tmp_path: Path) -> None:
    class _RaisedTeardownProvider(_Provider):
        def terminate(self, instance_id):
            raise RuntimeError("provider transport closed")

    provider = _RaisedTeardownProvider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_measurement_chrono_dem_vast_canary(
        bound_request=_bound_request(),
        bundle_receipt=_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path / "canary",
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: {"runtime_result_digest": D3, "status": "passed"},
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )

    assert result["status"] == "failed"
    assert "measurement_chrono_dem_provider_terminate_failed:RuntimeError" in result["blockers"]
    assert "measurement_chrono_dem_teardown_verification_failed" in result["blockers"]
    teardown = json.loads(
        (tmp_path / "canary" / "teardown_receipt.json").read_text(encoding="utf-8")
    )
    assert teardown["status"] == "FAIL"
    assert teardown["terminate_result"]["error_type"] == "RuntimeError"
    assert (tmp_path / "canary" / "provider_zero_verification.json").is_file()
