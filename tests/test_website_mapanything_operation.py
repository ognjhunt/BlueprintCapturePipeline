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
