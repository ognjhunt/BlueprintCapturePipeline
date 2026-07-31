from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.reconstruction_vast_worker_smoke import (
    ReconstructionVastSmokeError,
    replay_reconstruction_vast_worker_smoke,
    run_reconstruction_vast_worker_smoke,
    validate_worker_smoke_result,
)
from blueprint_pipeline.reconstruction_worker_image_healthcheck import SCHEMA_VERSION


SHA = "a" * 40
D1 = "sha256:" + "1" * 64
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _receipt_schema():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/reconstruction_vast_worker_smoke.v1.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _bound_request():
    value = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "worker_smoke",
        "capture_profile": "trainer_smoke_fixture",
        "source_commit_sha": SHA,
        "worker_image_digest": IMAGE,
        "worker_stack_manifest_digest": D1,
        "reconstruction_dataset_digest": D1,
        "frozen_split_digest": D1,
        "calibration_digest": D1,
        "deterministic_configuration_digest": D1,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 60,
        "retry_cap": 0,
        "authority_id": "fixture-authority",
        "proof_effect": "none",
        "request_digest": D1,
        "bound_provider": "vast",
        "bound_preflight_digest": D1,
        "bound_checkout_source_commit": SHA,
        "bound_checkout_clean": True,
        "provider_mutation_authorized": True,
    }
    value["bound_request_digest"] = canonical_digest(value, digest_field="bound_request_digest")
    return value


def _preflight():
    return {
        "provider": "vast",
        "watchdog": {
            "status": "armed",
            "independent_process": True,
            "pid": 123,
            "deadline_epoch": 2000,
            "name_prefix": "blueprint-reconstruction-",
        },
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 120 * 1024**3,
        "on_demand_price_usd_per_hour": 0.5,
    }


def _runtime_result(*, passed=True):
    health = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-07-30T00:00:00Z",
        "status": "passed" if passed else "failed",
        "mode": "gpu_runtime",
        "checks": [{"check_id": "nvidia_runtime", "status": "passed"}],
        "blockers": [],
        "display_attached": False,
        "runtime_identity": {
            "worker_family": "blueprint-reconstruction-worker",
            "source_commit_sha": SHA,
            "container_image_digest": IMAGE,
        },
        "hidden_heldout_observations_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    health["healthcheck_digest"] = canonical_digest(health, digest_field="healthcheck_digest")
    value = {
        "schema_version": "reconstruction_vast_worker_smoke_result.v1",
        "status": "passed" if passed else "failed",
        "request_digest": D1,
        "worker_image_digest": IMAGE,
        "healthcheck": health,
        "hidden_heldout_observations_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    value["runtime_result_digest"] = canonical_digest(value, digest_field="runtime_result_digest")
    return value


class _Provider:
    name = "vast"

    def __init__(self, *, launch_status="launched", terminate_status="stopped", zero_after=True):
        self.launch_status = launch_status
        self.terminate_status = terminate_status
        self.zero_after = zero_after
        self.launched = False
        self.requests = []

    def billable_inventory(self, *, name_prefix):
        count = 0 if not self.launched or self.zero_after else 1
        return {
            "api_confirmed": True,
            "live_resource_count": count,
            "resources": [],
        }

    def build_request(self, spec, job_dir):
        assert spec.image == IMAGE
        assert spec.requires_rtx is False
        assert "BLUEPRINT_RECONSTRUCTION_SMOKE_OUTPUT_PUT_URL" in spec.env
        return {"create_payload": {"env": dict(spec.env)}}

    def launch(self, job_dir, request, **kwargs):
        self.requests.append(request)
        if self.launch_status == "ambiguous":
            return {
                "status": "blocked",
                "allocation_outcome_ambiguous": True,
                "blockers": ["ambiguous"],
            }
        if self.launch_status != "launched":
            return {"status": "blocked", "allocation_created": False}
        self.launched = True
        return {"status": "launched", "instance_id": "42"}

    def terminate(self, instance_id):
        if self.terminate_status == "stopped":
            self.launched = False
        return {"status": self.terminate_status, "instance_id": instance_id}


def test_smoke_result_validation_binds_image_gpu_and_no_proof():
    result = validate_worker_smoke_result(
        _runtime_result(),
        request_digest=D1,
        worker_image_digest=IMAGE,
        source_commit_sha=SHA,
    )
    assert result["proof_effect"] == "none"
    tampered = _runtime_result()
    tampered["healthcheck"]["runtime_identity"]["container_image_digest"] = (
        "registry.example/other@sha256:" + "c" * 64
    )
    tampered["healthcheck"]["healthcheck_digest"] = canonical_digest(
        tampered["healthcheck"], digest_field="healthcheck_digest"
    )
    tampered["runtime_result_digest"] = canonical_digest(
        tampered, digest_field="runtime_result_digest"
    )
    with pytest.raises(ReconstructionVastSmokeError, match="runtime_image_mismatch"):
        validate_worker_smoke_result(
            tampered,
            request_digest=D1,
            worker_image_digest=IMAGE,
            source_commit_sha=SHA,
        )

    wrong_sha = _runtime_result()
    wrong_sha["healthcheck"]["runtime_identity"]["source_commit_sha"] = "c" * 40
    wrong_sha["healthcheck"]["healthcheck_digest"] = canonical_digest(
        wrong_sha["healthcheck"], digest_field="healthcheck_digest"
    )
    wrong_sha["runtime_result_digest"] = canonical_digest(
        wrong_sha, digest_field="runtime_result_digest"
    )
    with pytest.raises(ReconstructionVastSmokeError, match="source_commit_mismatch"):
        validate_worker_smoke_result(
            wrong_sha,
            request_digest=D1,
            worker_image_digest=IMAGE,
            source_commit_sha=SHA,
        )


def test_one_instance_smoke_retrieves_output_and_proves_teardown_zero(tmp_path: Path):
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_reconstruction_vast_worker_smoke(
        bound_request=_bound_request(),
        preflight=_preflight(),
        job_dir=tmp_path,
        output_put_url="https://objects.example/upload?sig=secret",
        output_get_url="https://objects.example/download?sig=secret",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: _runtime_result(),
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "completed"
    assert result["instance_id"] == "42"
    assert result["provider_mutations_performed"] == 2
    assert result["provider_zero_verified"] is True
    assert result["scientific_qualification_inferred"] is False
    assert provider.requests[0]["prelaunch_spend_guard"]["retry_cap"] == 0
    assert (tmp_path / "teardown_receipt.json").is_file()
    assert (tmp_path / "provider_zero_verification.json").is_file()
    pending = list((tmp_path / "pending_teardowns").glob("*.json"))
    assert json.loads(pending[0].read_text(encoding="utf-8"))["status"] == "closed"
    assert not list((tmp_path / "leases").glob("*.lease.json"))
    replay = replay_reconstruction_vast_worker_smoke(
        job_dir=tmp_path, bound_request=_bound_request()
    )
    assert replay["status"] == "replay_verified"
    assert replay["live_provider_accessed"] is False
    validator = Draft202012Validator(_receipt_schema())
    validator.validate(_runtime_result())
    validator.validate(result)
    validator.validate(json.loads((tmp_path / "teardown_receipt.json").read_text()))
    validator.validate(
        json.loads((tmp_path / "provider_zero_verification.json").read_text())
    )
    validator.validate(replay)


def test_replay_rejects_tampered_execution_receipt(tmp_path: Path):
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1002.0])
    run_reconstruction_vast_worker_smoke(
        bound_request=_bound_request(),
        preflight=_preflight(),
        job_dir=tmp_path,
        output_put_url="https://objects.example/upload",
        output_get_url="https://objects.example/download",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: _runtime_result(),
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    path = tmp_path / "reconstruction_vast_worker_smoke_execution.json"
    execution = json.loads(path.read_text(encoding="utf-8"))
    execution["cost_usd"] = 999.0
    path.write_text(json.dumps(execution), encoding="utf-8")

    replay = replay_reconstruction_vast_worker_smoke(
        job_dir=tmp_path, bound_request=_bound_request()
    )
    assert replay["status"] == "replay_rejected"
    assert "reconstruction_replay_execution_digest_mismatch" in replay["blockers"]


def test_malformed_output_fails_science_but_still_tears_down(tmp_path: Path):
    provider = _Provider()
    malformed = _runtime_result()
    malformed["runtime_result_digest"] = D1
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_reconstruction_vast_worker_smoke(
        bound_request=_bound_request(),
        preflight=_preflight(),
        job_dir=tmp_path,
        output_put_url="https://objects.example/upload",
        output_get_url="https://objects.example/download",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: malformed,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "failed"
    assert "reconstruction_smoke_result_digest_mismatch" in result["blockers"]
    assert result["provider_zero_verified"] is True
    assert provider.launched is False


def test_ambiguous_create_is_failed_and_resolved_only_by_provider_zero(tmp_path: Path):
    provider = _Provider(launch_status="ambiguous")
    times = iter([1000.0, 1001.0])
    result = run_reconstruction_vast_worker_smoke(
        bound_request=_bound_request(),
        preflight=_preflight(),
        job_dir=tmp_path,
        output_put_url="https://objects.example/upload",
        output_get_url="https://objects.example/download",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "failed"
    assert result["provider_mutation_outcome_ambiguous"] is True
    assert result["provider_mutations_performed"] == 1
    assert result["provider_zero_verified"] is True
    pending = list((tmp_path / "pending_teardowns").glob("*.json"))
    assert json.loads(pending[0].read_text(encoding="utf-8"))["status"] == "cancelled_no_allocation"


def test_teardown_failure_keeps_pending_record_and_lane_lease(tmp_path: Path):
    provider = _Provider(terminate_status="stop_failed", zero_after=False)
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_reconstruction_vast_worker_smoke(
        bound_request=_bound_request(),
        preflight=_preflight(),
        job_dir=tmp_path,
        output_put_url="https://objects.example/upload",
        output_get_url="https://objects.example/download",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: _runtime_result(),
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "failed"
    assert "reconstruction_teardown_verification_failed" in result["blockers"]
    assert result["provider_zero_verified"] is False
    pending = list((tmp_path / "pending_teardowns").glob("*.json"))
    assert json.loads(pending[0].read_text(encoding="utf-8"))["status"] == "open"
    assert list((tmp_path / "leases").glob("*.lease.json"))


def test_missing_opaque_grant_refuses_before_provider_access(tmp_path: Path):
    provider = _Provider()
    with pytest.raises(Exception, match="paid_resource_admission_grant_missing"):
        run_reconstruction_vast_worker_smoke(
            bound_request=_bound_request(),
            preflight=_preflight(),
            job_dir=tmp_path,
            output_put_url="https://objects.example/upload",
            output_get_url="https://objects.example/download",
            provider=provider,
            paid_resource_admission_grant=None,
        )
    assert provider.requests == []
