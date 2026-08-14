from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.sam31_gpu_admission import CHECKPOINT_DIGEST, OPERATION
from blueprint_pipeline.sam31_source_track_canary_worker import RUNTIME_RESULT_SCHEMA_VERSION
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)
from blueprint_pipeline.sam31_vast_source_track_canary import (
    Sam31VastCanaryError,
    _bootstrap_script,
    _watchdog_valid,
    run_sam31_vast_source_track_canary,
    validate_sam31_runtime_result,
)


SHA = "a" * 40
D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
IMAGE = "registry.example/blueprint/sam31@sha256:" + "b" * 64
TOKEN = "hf_fixture_secret_value"
INPUT_URL = "https://objects.example/input?signature=input-secret"
PUT_URL = "https://objects.example/output?signature=put-secret"
GET_URL = "https://objects.example/output?signature=get-secret"


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class="gpu_render"),
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _bound_request() -> dict:
    value = {
        "schema_version": "semantic_sam31_gpu_canary_request.v1",
        "operation": OPERATION,
        "source_commit_sha": SHA,
        "worker_image_digest": IMAGE,
        "input_bundle_digest": D1,
        "source_track_run_request_digest": D2,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 60,
        "retry_cap": 0,
        "authority_id": "fixture-authority",
        "request_digest": D1,
        "bound_provider": "vast",
        "bound_preflight_digest": D1,
        "bound_checkout_source_commit": SHA,
        "bound_checkout_clean": True,
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
            "name_prefix": "blueprint-sam31-source-tracks-",
        },
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 80 * 1024**3,
        "on_demand_price_usd_per_hour": 0.5,
    }


def test_watchdog_validator_accepts_canonical_handoff_field_names(monkeypatch) -> None:
    monkeypatch.setattr("os.kill", lambda pid, signal: None)
    assert _watchdog_valid(
        {
            "status": "armed",
            "independent_process": True,
            "watchdog_pid": 123,
            "watchdog_deadline_epoch": 2000,
            "pod_name_prefix": "blueprint-sam31-source-tracks-bound-run-",
        },
        now_epoch=1000,
        hard_ttl_seconds=60,
    )


def _runtime_result() -> dict:
    normalized = {
        "schema_version": "semantic_source_track_import_result.v1",
        "status": "abstained",
        "bindings": {},
        "track_registry": [],
        "frame_masks": [],
        "blockers": [],
        "warnings": ["provider_returned_no_tracks"],
        "claim_ceiling": "no_source_tracks_detected",
    }
    normalized["result_digest"] = canonical_json_digest(normalized)
    value = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed",
        "request_digest": D1,
        "bound_request_digest": _bound_request()["bound_request_digest"],
        "worker_image_digest": IMAGE,
        "input_bundle_digest": D1,
        "source_track_run_request_digest": D2,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "runtime": {
            "torch_version": "2.10.0+cu128",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "fixture",
        },
        "stage_run_result": {
            "status": "abstained",
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
        "provider_result": {"tracks": []},
        "source_track_import_request": {},
        "normalized_source_tracks": normalized,
        "blockers": [],
        "source_frame_bytes_returned": False,
        "raw_secret_values_recorded": False,
        "network_access_during_inference": False,
        "directly_observed_object_fact": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "model_self_grading_permitted": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "source_bound_2d_binary_mask_tracks_only",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    value["runtime_result_digest"] = canonical_digest(value, digest_field="runtime_result_digest")
    return value


class _Provider:
    name = "vast"

    def __init__(self, *, initially_live: bool = False, terminate_status: str = "stopped"):
        self.initially_live = initially_live
        self.terminate_status = terminate_status
        self.launched = False
        self.requests: list[dict] = []

    def billable_inventory(self, *, name_prefix: str) -> dict:
        count = 1 if self.initially_live or self.launched else 0
        return {
            "api_confirmed": True,
            "live_resource_count": count,
            "resources": [],
        }

    def build_request(self, spec, job_dir):
        assert spec.name.startswith("blueprint-sam31-source-tracks-")
        assert spec.image == IMAGE
        assert spec.requires_rtx is False
        assert spec.env["HF_TOKEN"] == TOKEN
        assert spec.env["BLUEPRINT_SAM31_INPUT_BUNDLE_GET_URL"] == INPUT_URL
        assert spec.env["BLUEPRINT_SAM31_RUNTIME_DIGEST"] == "sha256:" + "b" * 64
        return {"create_payload": {"env": dict(spec.env)}}

    def launch(self, job_dir, request, **kwargs):
        self.requests.append(request)
        self.launched = True
        return {"status": "launched", "instance_id": "42"}

    def terminate(self, instance_id):
        if self.terminate_status == "stopped":
            self.launched = False
        return {"status": self.terminate_status, "instance_id": instance_id}


def test_bootstrap_fetches_exact_checkpoint_then_unsets_token() -> None:
    script = _bootstrap_script()
    assert 'repo_id="facebook/sam3.1"' in script
    assert 'filename="sam3.1_multiplex.pt"' in script
    assert "sam31_checkpoint_digest_mismatch" in script
    assert "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script


def test_runtime_result_validation_preserves_claim_ceiling() -> None:
    result = validate_sam31_runtime_result(_runtime_result(), bound_request=_bound_request())
    assert result["claim_ceiling"] == "source_bound_2d_binary_mask_tracks_only"
    tampered = _runtime_result()
    tampered["metric_box_ready"] = True
    tampered["runtime_result_digest"] = canonical_digest(
        tampered, digest_field="runtime_result_digest"
    )
    with pytest.raises(Sam31VastCanaryError, match="metric_box_ready_mismatch"):
        validate_sam31_runtime_result(tampered, bound_request=_bound_request())

    secret = _runtime_result()
    secret["provider_result"]["metadata"] = {"hf_token": "forbidden"}
    secret["runtime_result_digest"] = canonical_digest(secret, digest_field="runtime_result_digest")
    with pytest.raises(Sam31VastCanaryError, match="secret_field_forbidden"):
        validate_sam31_runtime_result(secret, bound_request=_bound_request())


def test_one_instance_canary_tears_down_and_persists_no_secrets(tmp_path: Path) -> None:
    provider = _Provider()
    times = iter([1000.0, 1001.0, 1002.0])
    result = run_sam31_vast_source_track_canary(
        bound_request=_bound_request(),
        preflight=_preflight(),
        job_dir=tmp_path,
        input_bundle_get_url=INPUT_URL,
        output_put_url=PUT_URL,
        output_get_url=GET_URL,
        hf_token=TOKEN,
        provider=provider,
        paid_resource_admission_grant=_grant(),
        result_fetcher=lambda _url: _runtime_result(),
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
        watchdog_validator=lambda _watchdog, _now, _ttl: True,
    )
    assert result["status"] == "completed"
    assert result["provider_zero_verified"] is True
    assert result["provider_mutations_performed"] == 2
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert Path(result["source_track_import_result_path"]).is_file()
    assert result["source_track_import_result_digest"] == _runtime_result()[
        "normalized_source_tracks"
    ]["result_digest"]
    assert provider.requests[0]["create_payload"]["env"]
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    for secret in (TOKEN, INPUT_URL, PUT_URL, GET_URL):
        assert secret not in persisted
    assert not list((tmp_path / "leases").glob("*.lease.json"))


def test_global_nonzero_refuses_before_launch(tmp_path: Path) -> None:
    provider = _Provider(initially_live=True)
    with pytest.raises(Sam31VastCanaryError, match="provider_not_zero_before_launch"):
        run_sam31_vast_source_track_canary(
            bound_request=_bound_request(),
            preflight=_preflight(),
            job_dir=tmp_path,
            input_bundle_get_url=INPUT_URL,
            output_put_url=PUT_URL,
            output_get_url=GET_URL,
            hf_token=TOKEN,
            provider=provider,
            paid_resource_admission_grant=_grant(),
            clock=lambda: 1000.0,
            watchdog_validator=lambda _watchdog, _now, _ttl: True,
        )
    assert provider.requests == []


def test_missing_grant_refuses_before_provider_access(tmp_path: Path) -> None:
    provider = _Provider()
    with pytest.raises(Exception, match="paid_resource_admission_grant_missing"):
        run_sam31_vast_source_track_canary(
            bound_request=_bound_request(),
            preflight=_preflight(),
            job_dir=tmp_path,
            input_bundle_get_url=INPUT_URL,
            output_put_url=PUT_URL,
            output_get_url=GET_URL,
            hf_token=TOKEN,
            provider=provider,
            paid_resource_admission_grant=None,
        )
    assert provider.requests == []
