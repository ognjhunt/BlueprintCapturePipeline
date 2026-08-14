from __future__ import annotations

import json
import os
import time
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
from blueprint_pipeline.task_evaluation_artifact_manifest import (
    PROVIDER_RUN_DIRNAME,
    TEARDOWN_MANIFEST_NAME,
)
from blueprint_pipeline.task_evaluation_live_profile import shared_control_surface


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
        "operation_request_digest": D1,
        "operation_input_bundle_digest": D1,
        "expected_runtime_result_schema": "reconstruction_vast_worker_smoke_result.v1",
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


def _completed_smoke(tmp_path: Path, **overrides):
    """One accepted smoke, run the way the canonical allocator runs it."""

    provider = overrides.pop("provider", None) or _Provider()
    times = iter([1000.0, 1001.0, 1002.0])
    return provider, run_reconstruction_vast_worker_smoke(
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
        **overrides,
    )


def test_a_completed_smoke_satisfies_the_shared_terminal_contract(tmp_path: Path):
    """The lane rented a GPU, passed the healthcheck and tore down -- and its
    launch profile still reported `allocator_terminal_artifact_missing:`,
    because the result named neither artifact. "All controls passed" was
    unreachable by construction.

    The contract is read from `shared_control_surface()` rather than restated
    here, so a field added there fails this lane instead of silently becoming
    the next thing a paid run discovers.
    """

    _provider, result = _completed_smoke(tmp_path)
    terminal = shared_control_surface()["terminal_contract"]

    assert result["status"] in terminal["success_statuses"]
    for field, expected in terminal["required_values"].items():
        assert result.get(field) == expected, f"terminal required_value unmet: {field}"
    for field in terminal["required_path_fields"]:
        named = str(result.get(field) or "").strip()
        assert named, f"terminal required_path_field never set: {field}"
        assert Path(named).is_file(), f"terminal required_path_field names no file: {field}"


def test_the_sealed_artifact_manifest_inventories_the_teardown_it_names(tmp_path: Path):
    """A manifest that names nothing is a path to nowhere dressed as evidence."""

    _provider, result = _completed_smoke(tmp_path)
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["blockers"] == []
    assert manifest["binding"]["allocator_lane"] == "reconstruction_worker_smoke"
    inventoried = {row["relative_path"] for row in manifest["files"]}
    # Built from the shared convention rather than a literal, so a lane that
    # renames its provider run out from under the sealer fails here.
    assert f"{PROVIDER_RUN_DIRNAME}/{TEARDOWN_MANIFEST_NAME}" in inventoried
    assert "reconstruction_vast_worker_smoke_execution.json" in inventoried
    assert Path(result["teardown_manifest_path"]) == (
        tmp_path / PROVIDER_RUN_DIRNAME / TEARDOWN_MANIFEST_NAME
    )


def test_the_teardown_manifest_is_bound_to_the_receipt_it_was_derived_from(tmp_path: Path):
    """The manifest restates the lane's own teardown receipt in the shape the
    shared contract reads. Binding it by digest is what stops the two drifting
    into a manifest that says zero while the receipt says otherwise."""

    _provider, result = _completed_smoke(tmp_path)
    manifest = json.loads(Path(result["teardown_manifest_path"]).read_text(encoding="utf-8"))
    receipt = json.loads((tmp_path / "teardown_receipt.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "vast_teardown_manifest.v1"
    assert manifest["continuing_spend_from_this_run"] is False
    assert manifest["vast_instance_ids"] == [42]
    assert manifest["source_teardown_receipt_digest"] == receipt["teardown_receipt_digest"]


def test_sealing_does_not_break_replay_from_the_recorded_execution_record(tmp_path: Path):
    """The sealed paths are where these bytes happen to live on this host, so
    they stay out of the digested execution record that replay verifies. A
    replay that re-derives the digest must still accept an untouched run."""

    _provider, result = _completed_smoke(tmp_path)
    recorded = json.loads(
        (tmp_path / "reconstruction_vast_worker_smoke_execution.json").read_text(
            encoding="utf-8"
        )
    )

    assert "artifact_manifest_path" not in recorded
    assert "teardown_manifest_path" not in recorded
    assert recorded["execution_result_digest"] == canonical_digest(
        recorded, digest_field="execution_result_digest"
    )
    assert result["execution_result_digest"] == recorded["execution_result_digest"]
    replay = replay_reconstruction_vast_worker_smoke(
        job_dir=tmp_path, bound_request=_bound_request()
    )
    assert replay["status"] == "replay_verified"


def test_a_run_that_left_spend_behind_says_so_in_the_field_the_profile_reads(
    tmp_path: Path,
):
    """`continuing_spend_from_this_run` is the one field standing between a
    leaked instance and a launch reported as a clean success."""

    provider = _Provider(terminate_status="stop_failed", zero_after=False)
    _provider, result = _completed_smoke(tmp_path, provider=provider)

    assert result["status"] == "failed"
    assert result["continuing_spend_from_this_run"] is True
    manifest = json.loads(Path(result["teardown_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["continuing_spend_from_this_run"] is True
    assert manifest["status"] == "blocked"


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


def test_worker_smoke_rejects_wrong_operation_result_contract_before_allocation(
    tmp_path: Path,
):
    request = _bound_request()
    request["expected_runtime_result_schema"] = "reconstruction_training_result.v1"
    request["bound_request_digest"] = canonical_digest(
        request, digest_field="bound_request_digest"
    )
    provider = _Provider()

    with pytest.raises(ReconstructionVastSmokeError, match="bound_request_not_executable"):
        run_reconstruction_vast_worker_smoke(
            bound_request=request,
            preflight=_preflight(),
            job_dir=tmp_path,
            output_put_url="https://objects.example/upload?sig=secret",
            output_get_url="https://objects.example/download?sig=secret",
            provider=provider,
            paid_resource_admission_grant=_grant(),
        )

    assert provider.requests == []


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


def test_the_canonical_allocator_writes_a_result_its_launch_profile_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The seam the defect actually lived at.

    A launch reads `{run_root}/allocator/result.json`, which is whatever the
    canonical allocator wrote -- so a lane that seals correctly and an allocator
    that drops the fields on the way out are indistinguishable from the profile's
    side. The allocator's own test stubs the lane out entirely and would keep
    passing either way, which is how a result no real lane could produce came to
    stand in for one.

    So this drives the real lane through the real allocator entrypoint, faking
    only the two things that would cost money or leave the host: the provider,
    and the HTTP fetch of the worker's envelope.
    """

    from argparse import Namespace

    from blueprint_pipeline import paid_resource_allocator as allocator

    args = Namespace(
        provider_launch_request=str(tmp_path / "request.json"),
        preflight_bundle=str(tmp_path / "preflight.json"),
        admission_out=str(tmp_path / "admission.json"),
        bound_request_out=str(tmp_path / "bound.json"),
        adapter_output=str(tmp_path / "allocator" / "result.json"),
        provider="vast",
        expected_source_commit=SHA,
        reconstruction_max_spend_usd=1.0,
        reconstruction_hard_ttl_seconds=300,
        reconstruction_retry_cap=0,
        reconstruction_authority_id="fixture-authority",
        provider_output_put_url_file=str(tmp_path / "put-url.txt"),
        provider_output_get_url_file=str(tmp_path / "get-url.txt"),
        execute=True,
    )
    preflight = _preflight()
    # The real watchdog validator runs here, so it needs a live process and a
    # deadline beyond this attempt's TTL rather than the fixture's constants.
    preflight["watchdog"]["pid"] = os.getpid()
    preflight["watchdog"]["deadline_epoch"] = time.time() + 100_000
    Path(args.preflight_bundle).write_text(json.dumps(preflight), encoding="utf-8")

    def fake_prepare(**kwargs):
        Path(str(kwargs["bound_request_out"])).write_text(
            json.dumps(_bound_request()), encoding="utf-8"
        )
        return {"status": "execute_ready", "blockers": [], "operation": "worker_smoke"}

    monkeypatch.setattr(allocator, "prepare_reconstruction_gpu_canary", fake_prepare)
    monkeypatch.setattr(
        "blueprint_pipeline.paid_resource_transport.read_sensitive_url_file",
        lambda _path, *, label: (f"https://objects.example/{label}", {"mode_is_0600": True}),
    )
    monkeypatch.setattr(allocator, "get_render_provider", lambda _name: _Provider())

    class _Response:
        status = 200
        body = json.dumps(_runtime_result()).encode("utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.reconstruction_vast_worker_smoke.safe_http_request",
        lambda *_args, **_kwargs: _Response(),
    )

    allocator._run_reconstruction_gpu_canary(args, checkout_commit=SHA)

    written = json.loads(Path(args.adapter_output).read_text(encoding="utf-8"))
    terminal = shared_control_surface()["terminal_contract"]
    assert written["status"] in terminal["success_statuses"]
    for field, expected in terminal["required_values"].items():
        assert written.get(field) == expected, f"terminal required_value unmet: {field}"
    for field in terminal["required_path_fields"]:
        named = str(written.get(field) or "").strip()
        assert named, f"terminal required_path_field never set: {field}"
        assert Path(named).is_file(), f"terminal required_path_field names no file: {field}"


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
