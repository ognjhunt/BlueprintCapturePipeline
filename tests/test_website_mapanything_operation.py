"""Exercise the actual bundle/worker/output contracts without model or provider calls."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_mapanything_operation as operation
from blueprint_pipeline import website_scene_geometry as geometry
from blueprint_pipeline.common import sha256_file, write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_gpu_operation_bundle import build_canary_request_from_operation_bundle
from blueprint_pipeline.reconstruction_gpu_operation_worker import execute_reconstruction_gpu_operation_bundle
from blueprint_pipeline.reconstruction_gpu_operation_output import (
    compile_reconstruction_gpu_operation_output_bundle, validate_reconstruction_gpu_operation_output_bundle,
)
from blueprint_pipeline.reconstruction_worker_contracts import ReconstructionWorkerContractError

SHA = "a" * 40
DIGEST = "sha256:" + "b" * 64
IMAGE = "registry.example/worker@" + DIGEST


@pytest.fixture
def packet(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    frames = []
    for index in range(2):
        image = root / f"frame-{index}.png"
        Image.new("RGB", (28, 14), (index * 20, 0, 0)).save(image)
        frames.append({"frame_id": str(index), "timestamp_seconds": float(index),
                       "source_image_path": image.name, "source_image_digest": "sha256:" + sha256_file(image),
                       "image_path": image.name, "image_digest": "sha256:" + sha256_file(image),
                       "width": 28, "height": 14})
    inputs = {"schema_version": geometry.INPUT_SCHEMA, "heldout_pixels_included": False, "frames": frames,
              "binding": {"model_id": geometry.MODEL_ID, "model_code_revision": geometry.MODEL_CODE_REVISION,
                          "source_video_digest": DIGEST, "adapter_version": 1}}
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    path = root / "geometry_inputs.json"
    write_json(path, inputs)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"hermetic model substitute")
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(operation, "MODEL_ROOT", model)
    monkeypatch.setattr(operation, "MODEL_CHECKPOINT_DIGEST", "sha256:" + sha256_file(model / "model.safetensors"))
    monkeypatch.setattr(operation, "MODEL_CONFIG_DIGEST", "sha256:" + sha256_file(model / "config.json"))
    monkeypatch.setattr(geometry, "_infer", lambda paths, **_kwargs: [{
        "depth_z": np.ones((1, 14, 28, 1)), "conf": np.ones((1, 14, 28)),
        "mask": np.ones((1, 14, 28, 1)), "intrinsics": np.array([[[20, 0, 14], [0, 20, 7], [0, 0, 1]]]),
        "camera_poses": np.eye(4)[None],
    } for _ in paths])
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("pipeline-1-py3-none-any.whl", "contracts-1-py3-none-any.whl", "mapanything-1-py3-none-any.whl",
                 "requirements.txt"):
        (runtime / name).write_bytes(b"hermetic runtime substitute")
    receipt = operation.compile_input_bundle(input_manifest=path, output_root=tmp_path / "bundles",
        source_commit_sha=SHA, worker_image_digest=IMAGE, remote_processing_authorization_digest=DIGEST,
        runtime_files=sorted(runtime.iterdir()))
    bundle = tmp_path / "bundles" / receipt["bundle_artifact_reference"]
    return path, receipt, bundle


def test_worker_round_trip_preserves_estimated_geometry_and_all_output_bytes(packet, tmp_path):
    path, receipt, bundle = packet
    materialized = tmp_path / "materialized"
    output = tmp_path / "output"
    result = execute_reconstruction_gpu_operation_bundle(bundle_path=bundle, bundle_receipt=receipt,
        materialization_root=materialized, output_root=output)
    request = json.loads((materialized / receipt["operation_input_bundle_digest"][7:] / "operation_request.json").read_text())
    output_bundle = tmp_path / "returned.zip"
    compiled = compile_reconstruction_gpu_operation_output_bundle(operation=operation.OPERATION,
        operation_request=request, runtime_result=result, operation_output_root=output, output_path=output_bundle)
    _, checked = validate_reconstruction_gpu_operation_output_bundle(bundle_path=output_bundle,
        expected_operation=operation.OPERATION, expected_operation_request_digest=request[operation.REQUEST_DIGEST],
        expected_worker_image_digest=IMAGE, expected_source_commit_sha=SHA)
    assert checked == result
    assert compiled["artifact_member_count"] == 7
    assert result["metric_measurement_proven"] is False
    assert result["physical_evidence"] is False
    assert result["geometry_input_digest"] == json.loads(path.read_text())["digest"]
    rebound = geometry.load_website_geometry_result(
        manifest_path=output / request[operation.REQUEST_DIGEST][7:23] / "source_geometry.json",
        inputs=json.loads(path.read_text()))
    assert len(rebound["frames"]) == 2


def test_model_mismatch_stops_before_inference(packet, tmp_path, monkeypatch):
    _, receipt, bundle = packet
    (operation.MODEL_ROOT / "model.safetensors").write_bytes(b"different model")
    monkeypatch.setattr(geometry, "_infer", lambda *_args, **_kwargs: pytest.fail("wrong model must not run"))
    with pytest.raises(ValueError, match="model_bytes_mismatch"):
        execute_reconstruction_gpu_operation_bundle(bundle_path=bundle, bundle_receipt=receipt,
            materialization_root=tmp_path / "materialized", output_root=tmp_path / "output")


def test_admission_uses_estimated_geometry_without_inventing_calibration(packet):
    _, receipt, _ = packet
    fields = {"schema_version": "reconstruction_gpu_canary_request.v1", "worker_stack_manifest_digest": DIGEST,
              "deterministic_configuration_digest": DIGEST, "max_spend_usd": 2, "hard_ttl_seconds": 3600,
              "retry_cap": 0, "authority_id": "hermetic-test-only", "proof_effect": "none",
              "candidate_may_read_hidden_heldout": False, "trainer_may_grade_heldout": False}
    request = build_canary_request_from_operation_bundle(request_fields=fields, operation_bundle=receipt)
    assert request["capture_profile"] == "website_monocular_video"
    assert request["calibration_digest"] is None
    assert request["remote_processing_authorization_digest"] == DIGEST
    with pytest.raises(ValueError, match="calibration_digest_mismatch"):
        build_canary_request_from_operation_bundle(request_fields={**fields, "calibration_digest": DIGEST},
                                                   operation_bundle=receipt)


def test_request_cannot_claim_measured_scale(packet, tmp_path):
    _, receipt, bundle = packet
    from blueprint_pipeline.reconstruction_gpu_operation_bundle import extract_reconstruction_gpu_operation_bundle
    extract_reconstruction_gpu_operation_bundle(bundle_path=bundle, bundle_receipt=receipt, output_root=tmp_path / "extracted")
    request_path = tmp_path / "extracted" / receipt["operation_input_bundle_digest"][7:] / "operation_request.json"
    request = json.loads(request_path.read_text())
    request["metric_measurement_proven"] = True
    with pytest.raises(ReconstructionWorkerContractError, match="metric_measurement_proven_invalid"):
        operation.build_request(request)


def test_bootstrap_installs_only_bundle_bound_runtime_files(packet, tmp_path, monkeypatch):
    from blueprint_pipeline import website_mapanything_bootstrap as bootstrap

    _, receipt, bundle = packet
    receipt_path = tmp_path / "receipt.json"
    write_json(receipt_path, receipt)
    source_paths = {"https://transport.example/receipt": receipt_path, "https://transport.example/bundle": bundle}
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda url, **_kwargs: source_paths[url].open("rb"))
    for name, value in {
        "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_GET_URL": "https://transport.example/receipt",
        "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_GET_URL": "https://transport.example/bundle",
        "BLUEPRINT_RECONSTRUCTION_INPUT_RECEIPT_FILE_DIGEST": "sha256:" + sha256_file(receipt_path),
        "BLUEPRINT_RECONSTRUCTION_INPUT_BUNDLE_DIGEST": receipt["operation_input_bundle_digest"],
        "BLUEPRINT_RECONSTRUCTION_OPERATION_REQUEST_DIGEST": receipt["operation_request_digest"],
        "BLUEPRINT_SOURCE_COMMIT": SHA, "BLUEPRINT_CONTAINER_IMAGE_DIGEST": IMAGE,
    }.items():
        monkeypatch.setenv(name, value)
    commands = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda args, **_kwargs: commands.append(args))
    bootstrap.install_runtime(tmp_path / "worker")
    # The deployed base is Ubuntu Python with an EXTERNALLY-MANAGED marker.
    assert all("--break-system-packages" in command for command in commands)
    assert len(commands) == 2
    assert "--require-hashes" in commands[0]
    assert len([arg for arg in commands[1] if arg.endswith(".whl")]) == 3
    assert all(Path(arg).is_file() for arg in commands[1] if arg.endswith(".whl"))
    monkeypatch.setenv("BLUEPRINT_SOURCE_COMMIT", "c" * 40)
    with pytest.raises(ValueError, match="receipt_binding_mismatch"):
        bootstrap.install_runtime(tmp_path / "wrong-worker")
    assert len(commands) == 2


def _controller_profile(tmp_path, monkeypatch):
    from blueprint_pipeline import website_geometry_dispatch as dispatch
    value = {"schema_version": "website_mapanything_runtime.v1", "source_commit": SHA,
             "worker_image_digest": IMAGE, "maximum_cost_usd": 2.0,
             "max_hourly_rate_usd": 1.5, "hard_ttl_seconds": 1800, "minimum_gpu_ram_mb": 80000,
             "runtime_files": [{"path": str(p), "digest": "sha256:" + sha256_file(p)}
                               for p in sorted((tmp_path / "runtime").iterdir())]}
    path = tmp_path / "profile.json"
    write_json(path, value)
    monkeypatch.setenv("BLUEPRINT_WEBSITE_MAPANYTHING_PROFILE", str(path))
    assert dispatch.load_profile(source_commit=SHA) == value
    return value


def test_controller_runtime_profile_rejects_stale_release_and_changed_wheels(packet, tmp_path, monkeypatch):
    from blueprint_pipeline import website_geometry_dispatch as dispatch
    profile = _controller_profile(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="runtime_release_mismatch"):
        dispatch.load_profile(source_commit="f" * 40)
    Path(profile["runtime_files"][0]["path"]).write_bytes(b"substituted code")
    with pytest.raises(ValueError, match="runtime_file_changed"):
        dispatch.load_profile(source_commit=SHA)


@pytest.mark.parametrize("uncertain", [False, True])
def test_controller_owns_geometry_funding_allocator_and_restart_without_second_rental(packet, tmp_path, monkeypatch, uncertain):
    from types import SimpleNamespace
    from blueprint_pipeline import website_geometry_dispatch as dispatch
    path, _, _ = packet
    _controller_profile(tmp_path, monkeypatch)
    events, closes = [], []
    task = {"context_digest": DIGEST}
    handle = object()
    monkeypatch.setattr(dispatch, "load_website_scene_sponsorship", lambda **kw: {
        "expires_at_epoch": 9_999_999_999, "authority_digest": DIGEST})
    def arm(**kwargs):
        events.append("watchdog")
        from blueprint_pipeline.vast_independent_watchdog_control import validate_independent_vast_watchdog_names
        validate_independent_vast_watchdog_names(pod_name_prefix=kwargs["pod_name_prefix"],
                                                resource_name_exact=kwargs["resource_name_exact"])
        return {"watchdog_pid": 123, "watchdog_deadline_epoch": 9_999_999_999,
                "watchdog_out_dir": str(tmp_path / "watchdog")}, handle
    monkeypatch.setattr(dispatch, "arm_independent_vast_watchdog", arm)
    monkeypatch.setattr(dispatch, "get_render_provider", lambda *_: SimpleNamespace(capacity_preflight=None, billable_inventory=None))
    def preflight(**kwargs):
        events.append("capacity")
        assert kwargs["minimum_gpu_ram_mb"] == 80000
        return {"status": "verified"}
    monkeypatch.setattr(dispatch, "collect_reconstruction_vast_preflight", preflight)
    def reserve(**kwargs):
        events.append("reserve")
        assert kwargs["resource_class"] == "gpu_render" and kwargs["provider"] == "vast"
        assert kwargs["maximum_cost_usd"] == 2 and kwargs["request_count"] == 1
        return {"expires_at_epoch": 9_999_999_999, "allocation_binding_digest": kwargs["binding_digest"]}, object()
    monkeypatch.setattr(dispatch, "reserve_website_preparation_spend", reserve)
    def stage(**kwargs):
        events.append("stage")
        assert "reserve" in events
        return {"status": "completed"}
    monkeypatch.setattr(dispatch, "stage_wam_provider_bundle_object_store", stage)
    def close(**kwargs):
        events.append("close")
        closes.append(kwargs)
    monkeypatch.setattr(dispatch, "close_independent_vast_watchdog", close)
    monkeypatch.setattr(dispatch, "cleanup_staged_wam_provider_objects", lambda **kw: events.append("cleanup"))
    def reuse(root, inputs):
        events.append("reuse")
        if uncertain:
            raise ValueError("website_mapanything_existing_attempt_requires_reconciliation")
        return {"status": "estimated", "input_digest": inputs["digest"]}
    monkeypatch.setattr(dispatch, "_reuse", reuse)
    def allocate(args, **kwargs):
        events.append("allocate")
        assert kwargs == {"checkout_commit": SHA}
        request = json.loads(Path(args.provider_launch_request).read_text())
        assert request["operation"] == "website_mapanything"
        assert request["retry_cap"] == 0 and request["max_spend_usd"] == 2
        assert args.execute is True and not hasattr(args, "experimental_branch_diagnostic")
        if uncertain:
            raise TimeoutError("controller lost allocation response")
        return {"status": "completed", "instance_id": 7, "provider_zero_verified": True}
    args = dict(input_manifest=path, output_root=tmp_path / "controller", task_context=task,
                source_commit=SHA, allocate=allocate)
    if uncertain:
        with pytest.raises(TimeoutError):
            dispatch.dispatch_geometry(**args)
        with pytest.raises(ValueError, match="requires_reconciliation"):
            dispatch.dispatch_geometry(**args)
        assert "cleanup" not in events
        assert closes[0]["provider_teardown_completed"] is False
        assert closes[0]["provider_allocation_impossible"] is False
    else:
        first = dispatch.dispatch_geometry(**args)
        assert dispatch.dispatch_geometry(**args) == first
        assert closes[0]["provider_teardown_completed"] is True
        assert events.count("cleanup") == 2
    assert events.count("allocate") == events.count("reserve") == 1
    assert events[:3] == ["watchdog", "capacity", "reserve"]
    with pytest.raises(ValueError, match="inputs_changed"):
        dispatch.dispatch_geometry(**{**args, "task_context": {"context_digest": "changed"}})
