from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import jsonschema

from blueprint_pipeline import reconstruction_vast_operation as vast_operation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.reconstruction_gpu_operation_output import (
    ReconstructionGpuOperationOutputError,
)
from blueprint_pipeline.reconstruction_vast_operation import (
    ReconstructionVastOperationError,
    replay_reconstruction_vast_operation,
    run_reconstruction_vast_operation,
)
from blueprint_pipeline.safe_outbound_http import SafeHttpFileTransfer


SHA = "a" * 40
D = ["sha256:" + str(index) * 64 for index in range(1, 8)]
IMAGE = "registry.example/reconstruction@sha256:" + "b" * 64


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _bound_request() -> dict:
    value = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "pose_canary",
        "capture_profile": "camera_360_native",
        "source_commit_sha": SHA,
        "worker_image_digest": IMAGE,
        "worker_stack_manifest_digest": D[0],
        "reconstruction_dataset_digest": D[3],
        "frozen_split_digest": D[4],
        "calibration_digest": D[5],
        "deterministic_configuration_digest": D[6],
        "operation_request_digest": D[1],
        "operation_input_bundle_digest": D[2],
        "expected_runtime_result_schema": "pose_estimation_result.v1",
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "max_spend_usd": 18.0,
        "hard_ttl_seconds": 3600,
        "retry_cap": 1,
        "authority_id": "user-authorized-18usd-60min-1retry",
        "proof_effect": "none",
        "request_digest": D[0],
        "bound_provider": "vast",
        "bound_preflight_digest": D[6],
        "bound_checkout_source_commit": SHA,
        "bound_checkout_clean": True,
        "provider_mutation_authorized": True,
    }
    value["bound_request_digest"] = canonical_digest(
        value, digest_field="bound_request_digest"
    )
    return value


def _bundle_receipt() -> dict:
    value = {
        "schema_version": "reconstruction_gpu_operation_bundle.v1",
        "status": "compiled",
        "operation": "pose_canary",
        "source_commit_sha": SHA,
        "worker_image_digest": IMAGE,
        "reconstruction_dataset_digest": D[3],
        "frozen_split_digest": D[4],
        "calibration_digest": D[5],
        "operation_request_digest": D[1],
        "operation_input_bundle_digest": D[2],
        "bundle_manifest_digest": D[6],
        "artifact_members": [
            {
                "archive_path": "inputs/frame.png",
                "digest": D[0],
                "bytes": 1,
            }
        ],
        "artifact_member_count": 1,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "candidate_operation_input_only",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _preflight() -> dict:
    return {
        "provider": "vast",
        "watchdog": {
            "status": "armed",
            "independent_process": True,
            "pid": 123,
            "deadline_epoch": 10_000,
            "name_prefix": "blueprint-reconstruction-",
        },
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 120 * 1024**3,
        "on_demand_price_usd_per_hour": 0.58,
    }


class _Provider:
    name = "vast"

    def __init__(self, *, terminate_status: str = "stopped", zero_after: bool = True):
        self.terminate_status = terminate_status
        self.zero_after = zero_after
        self.launched = False
        self.requests: list[dict] = []

    def billable_inventory(self, *, name_prefix):
        del name_prefix
        count = 1 if self.launched and not self.zero_after else 0
        return {
            "api_confirmed": True,
            "live_resource_count": count,
            "resources": [],
        }

    def build_request(self, spec, job_dir):
        del job_dir
        assert spec.image == IMAGE
        assert spec.env["BLUEPRINT_RECONSTRUCTION_OPERATION"] == "pose_canary"
        assert "INPUT_BUNDLE_GET_URL" in " ".join(spec.env)
        assert "reconstruction_gpu_operation_bootstrap" in spec.bootstrap_script
        return {"create_payload": {"env": dict(spec.env)}}

    def launch(self, job_dir, request, **kwargs):
        del job_dir, kwargs
        self.requests.append(request)
        self.launched = True
        return {"status": "launched", "instance_id": "42"}

    def terminate(self, instance_id):
        if self.terminate_status == "stopped":
            self.launched = False
        return {"status": self.terminate_status, "instance_id": instance_id}


def _validated_output() -> tuple[dict, dict]:
    output_digest = "sha256:" + hashlib.sha256(
        b"provider-operation-output"
    ).hexdigest()
    runtime = {
        "schema_version": "pose_estimation_result.v1",
        "status": "succeeded",
        "pose_estimation_result_digest": D[6],
    }
    receipt = {
        "schema_version": "reconstruction_gpu_operation_output_bundle.v1",
        "status": "validated",
        "operation": "pose_canary",
        "operation_request_digest": D[1],
        "operation_output_bundle_digest": output_digest,
        "proof_effect": "none",
    }
    receipt["output_bundle_receipt_digest"] = canonical_digest(
        receipt, digest_field="output_bundle_receipt_digest"
    )
    return receipt, runtime


def _fetcher(_url: str, destination: Path) -> SafeHttpFileTransfer:
    payload = b"provider-operation-output"
    destination.write_bytes(payload)
    return SafeHttpFileTransfer(
        status=200,
        transferred_bytes=len(payload),
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        host="objects.example",
    )


def _validator(**_kwargs):
    return _validated_output()


def test_operation_retrieves_validates_then_tears_down_and_replays_offline(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_reconstruction_vast_operation(
        bound_request=_bound_request(),
        bundle_receipt=_bundle_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url="https://objects.example/input?sig=secret",
        input_receipt_get_url="https://objects.example/receipt?sig=secret",
        output_bundle_put_url="https://objects.example/output-put?sig=secret",
        output_bundle_get_url="https://objects.example/output-get?sig=secret",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        output_fetcher=_fetcher,
        output_validator=_validator,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "completed"
    assert result["operation_result_status"] == "succeeded"
    assert result["output_retrieved_before_teardown"] is True
    assert result["provider_zero_verified"] is True
    assert result["scientific_qualification_inferred"] is False
    assert result["operation_scientific_success_inferred"] is False
    assert provider.launched is False
    assert provider.requests[0]["prelaunch_spend_guard"] == {
        "schema_version": "reconstruction_gpu_prelaunch_spend_guard.v1",
        "required_before_provider_launch": True,
        "can_launch": True,
        "blockers": [],
        "max_spend_usd": 18.0,
        "hard_ttl_seconds": 3600,
        "retry_cap": 1,
        "request_digest": D[0],
        "operation": "pose_canary",
    }
    assert "secret" not in json.dumps(result)
    teardown = json.loads((tmp_path / "teardown_receipt.json").read_text())
    assert teardown["output_retrieved_before_teardown"] is True
    replay = replay_reconstruction_vast_operation(
        job_dir=tmp_path,
        bound_request=_bound_request(),
        output_validator=_validator,
    )
    assert replay["status"] == "replay_verified"
    assert replay["live_provider_accessed"] is False
    assert replay["scientific_qualification_inferred"] is False
    schema = json.loads(
        Path("docs/schemas/reconstruction_vast_operation.v1.schema.json").read_text()
    )
    for artifact in (
        result,
        teardown,
        json.loads((tmp_path / "provider_zero_verification.json").read_text()),
        replay,
    ):
        jsonschema.validate(artifact, schema)


def test_canonical_operation_replay_uses_canonical_output_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _bound_request()
    request.update(
        {
            "operation": "trainer_canary",
            "execution_adapter_id": "canonical_splatfacto_vast_v1",
            "expected_runtime_result_schema": (
                "canonical_3dgs_vast_runtime_result.v1"
            ),
        }
    )
    request["bound_request_digest"] = canonical_digest(
        request, digest_field="bound_request_digest"
    )
    attempts = tmp_path / "retrieval_attempts"
    attempts.mkdir()
    output_path = attempts / "output_0001.zip"
    output_path.write_bytes(b"canonical-output")
    output_digest = "sha256:" + hashlib.sha256(output_path.read_bytes()).hexdigest()
    validated_receipt = {
        "schema_version": "canonical_3dgs_vast_output_bundle.v1",
        "operation_output_bundle_digest": output_digest,
        "proof_effect": "appearance_asset_candidate_only",
    }
    validated_receipt["output_bundle_receipt_digest"] = canonical_digest(
        validated_receipt, digest_field="output_bundle_receipt_digest"
    )
    validated_runtime = {
        "schema_version": "canonical_3dgs_vast_runtime_result.v1",
        "status": "succeeded",
    }
    execution = {
        "status": "completed",
        "bound_request_digest": request["bound_request_digest"],
        "canonical_allocator_admission_digest": D[6],
        "operation_output_bundle_digest": output_digest,
        "output_bundle_receipt_digest": validated_receipt[
            "output_bundle_receipt_digest"
        ],
        "output_retrieved_before_teardown": True,
        "scientific_qualification_inferred": False,
        "operation_result_status": "succeeded",
    }
    execution["execution_result_digest"] = canonical_digest(
        execution, digest_field="execution_result_digest"
    )
    teardown = {
        "status": "PASS",
        "output_retrieved_before_teardown": True,
        "provider_zero_verified": True,
    }
    teardown["teardown_receipt_digest"] = canonical_digest(
        teardown, digest_field="teardown_receipt_digest"
    )
    provider_zero = {
        "status": "PASS",
        "api_confirmed": True,
        "scoped_live_resource_count": 0,
        "global_live_resource_count": 0,
    }
    provider_zero["provider_zero_digest"] = canonical_digest(
        provider_zero, digest_field="provider_zero_digest"
    )
    for name, value in (
        ("reconstruction_vast_operation_execution.json", execution),
        ("teardown_receipt.json", teardown),
        ("provider_zero_verification.json", provider_zero),
        ("validated_output_bundle_receipt.json", validated_receipt),
        ("provider_runtime_result.json", validated_runtime),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    calls: list[dict] = []

    def canonical_validator(**kwargs):
        calls.append(kwargs)
        return validated_receipt, validated_runtime

    monkeypatch.setattr(
        vast_operation,
        "validate_canonical_3dgs_vast_output_bundle",
        canonical_validator,
    )

    def generic_validator(**_kwargs):
        raise AssertionError("generic validator must not handle canonical replay")

    replay = replay_reconstruction_vast_operation(
        job_dir=tmp_path,
        bound_request=request,
        output_validator=generic_validator,
    )

    assert replay["status"] == "replay_verified"
    assert len(calls) == 1
    assert calls[0]["expected_transport_bundle_digest"] == D[2]
    assert calls[0]["expected_reconstruction_dataset_digest"] == D[3]
    assert calls[0]["expected_allocator_admission_digest"] == D[6]


def test_repeated_identical_invalid_output_stops_and_preserves_failures(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    times = iter(float(value) for value in range(1000, 1010))

    def invalid(**_kwargs):
        raise ReconstructionGpuOperationOutputError(["malformed_output"])

    result = run_reconstruction_vast_operation(
        bound_request=_bound_request(),
        bundle_receipt=_bundle_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url="https://objects.example/input",
        input_receipt_get_url="https://objects.example/receipt",
        output_bundle_put_url="https://objects.example/output-put",
        output_bundle_get_url="https://objects.example/output-get",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        output_fetcher=_fetcher,
        output_validator=invalid,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "failed"
    assert "reconstruction_vast_operation_repeated_identical_blocker" in result[
        "blockers"
    ]
    assert result["fetch_attempts"] == 2
    assert result["output_retrieved_before_teardown"] is False
    assert result["provider_zero_verified"] is True
    assert len(list((tmp_path / "retrieval_attempts").glob("*.rejection.json"))) == 2
    assert provider.launched is False


def test_operation_refuses_bad_receipt_or_missing_grant_before_provider_access(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    bad = _bundle_receipt()
    bad["operation_request_digest"] = D[6]
    bad["receipt_digest"] = canonical_digest(bad, digest_field="receipt_digest")
    with pytest.raises(ReconstructionVastOperationError, match="request_digest_mismatch"):
        run_reconstruction_vast_operation(
            bound_request=_bound_request(),
            bundle_receipt=bad,
            preflight=_preflight(),
            job_dir=tmp_path / "bad",
            input_bundle_get_url="https://objects.example/input",
            input_receipt_get_url="https://objects.example/receipt",
            output_bundle_put_url="https://objects.example/output-put",
            output_bundle_get_url="https://objects.example/output-get",
            provider=provider,
            paid_resource_admission_grant=_grant(),
        )
    assert provider.requests == []

    with pytest.raises(ReconstructionVastOperationError, match="transport_url_invalid"):
        run_reconstruction_vast_operation(
            bound_request=_bound_request(),
            bundle_receipt=_bundle_receipt(),
            preflight=_preflight(),
            job_dir=tmp_path / "bad-url",
            input_bundle_get_url="http://objects.example/input",
            input_receipt_get_url="https://objects.example/receipt",
            output_bundle_put_url="https://objects.example/output-put",
            output_bundle_get_url="https://objects.example/output-get",
            provider=provider,
            paid_resource_admission_grant=_grant(),
        )
    assert provider.requests == []

    with pytest.raises(Exception, match="paid_resource_admission_grant_missing"):
        run_reconstruction_vast_operation(
            bound_request=_bound_request(),
            bundle_receipt=_bundle_receipt(),
            preflight=_preflight(),
            job_dir=tmp_path / "no-grant",
            input_bundle_get_url="https://objects.example/input",
            input_receipt_get_url="https://objects.example/receipt",
            output_bundle_put_url="https://objects.example/output-put",
            output_bundle_get_url="https://objects.example/output-get",
            provider=provider,
            paid_resource_admission_grant=None,
        )
    assert provider.requests == []


def test_teardown_failure_remains_terminal_blocker_and_keeps_lane_state(
    tmp_path: Path,
) -> None:
    provider = _Provider(terminate_status="stop_failed", zero_after=False)
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_reconstruction_vast_operation(
        bound_request=_bound_request(),
        bundle_receipt=_bundle_receipt(),
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url="https://objects.example/input",
        input_receipt_get_url="https://objects.example/receipt",
        output_bundle_put_url="https://objects.example/output-put",
        output_bundle_get_url="https://objects.example/output-get",
        provider=provider,
        paid_resource_admission_grant=_grant(),
        output_fetcher=_fetcher,
        output_validator=_validator,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "failed"
    assert "reconstruction_vast_operation_teardown_verification_failed" in result[
        "blockers"
    ]
    assert result["provider_zero_verified"] is False
    assert list((tmp_path / "leases").glob("*.lease.json"))
    pending = list((tmp_path / "pending_teardowns").glob("*.json"))
    assert json.loads(pending[0].read_text())["status"] == "open"
