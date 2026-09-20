"""Website geometry on the existing admitted reconstruction worker (ADP-009B/day14).

This is one typed operation, not another allocator. Inputs contain only retained
candidate views; outputs remain estimates. The worker bootstrap fetches hash-pinned weights before inference.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping, Sequence

from .common import sha256_file
from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import ReconstructionWorkerContractError

OPERATION = "website_mapanything"
REQUEST_SCHEMA = "website_mapanything_request.v1"
RESULT_SCHEMA = "website_mapanything_result.v1"
REQUEST_DIGEST = "website_mapanything_request_digest"
RESULT_DIGEST = "website_mapanything_result_digest"
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
MODEL_CHECKPOINT_DIGEST = "sha256:fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"
MODEL_CONFIG_DIGEST = "sha256:65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5"
MODEL_ROOT = Path("/opt/mapanything/map-anything-apache")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReconstructionWorkerContractError(["website_mapanything_" + code])


def _seal(value: Mapping[str, Any], schema: str, field: str) -> dict[str, Any]:
    result = dict(value)
    _require(result.get("schema_version") == schema, "schema_invalid")
    digest = canonical_digest(result, digest_field=field)
    _require(result.get(field, digest) == digest, "digest_mismatch")
    result[field] = digest
    return result


def build_request(value: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("source_capture_digest", "reconstruction_dataset_digest", "train_heldout_split_digest",
                "geometry_input_digest", "remote_processing_authorization_digest"):
        _require(_DIGEST.fullmatch(str(value.get(key, ""))) is not None, key + "_invalid")
    _require(re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit_sha", ""))) is not None,
             "source_commit_invalid")
    _require(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", str(value.get("container_image_digest", ""))) is not None,
             "worker_image_invalid")
    for key, expected in {"model_revision": MODEL_REVISION, "model_checkpoint_digest": MODEL_CHECKPOINT_DIGEST,
                          "model_config_digest": MODEL_CONFIG_DIGEST, "calibration_digest": None,
                          "claim_ceiling": "development_only", "metric_measurement_proven": False,
                          "candidate_may_read_hidden_heldout": False}.items():
        _require(key in value and value[key] == expected, key + "_invalid")
    return _seal(value, REQUEST_SCHEMA, REQUEST_DIGEST)


def build_result(value: Mapping[str, Any]) -> dict[str, Any]:
    for key in (REQUEST_DIGEST, "geometry_manifest_digest", "source_capture_digest", "geometry_input_digest"):
        _require(_DIGEST.fullmatch(str(value.get(key, ""))) is not None, key + "_invalid")
    for key, expected in {"status": "succeeded", "heldout_labels_included": False,
                          "candidate_self_graded": False, "metric_measurement_proven": False,
                          "physical_evidence": False, "claim_ceiling": "development_only"}.items():
        _require(value.get(key) == expected, key + "_invalid")
    rows = value.get("output_digests")
    _require(isinstance(rows, list) and bool(rows), "outputs_missing")
    for row in rows:
        _require(isinstance(row, Mapping) and _DIGEST.fullmatch(str(row.get("digest", ""))) is not None,
                 "output_digest_invalid")
    _require(any(row.get("artifact_id") == "source_geometry.json" for row in rows), "manifest_missing")
    return _seal(value, RESULT_SCHEMA, RESULT_DIGEST)


def compile_input_bundle(*, input_manifest: Path, output_root: Path, source_commit_sha: str,
                         worker_image_digest: str, remote_processing_authorization_digest: str,
                         runtime_files: Sequence[Path] = ()) -> dict[str, Any]:
    from .reconstruction_gpu_operation_bundle import compile_reconstruction_gpu_operation_bundle
    from .website_scene_geometry import _bound_frames, INPUT_SCHEMA

    inputs = json.loads(input_manifest.read_text())
    _require(inputs.get("schema_version") == INPUT_SCHEMA and inputs.get("heldout_pixels_included") is False,
             "inputs_invalid")
    frames = _bound_frames(inputs, input_manifest.parent, geometry=False)
    runtime = []
    for source in runtime_files:
        _require(source.is_file() and not source.is_symlink(), "runtime_file_invalid")
        # Keep prior release inputs intact when a pre-allocation retry uses
        # a rebuilt wheel with the same distribution filename.
        source_digest = sha256_file(source)
        target = input_manifest.parent / "runtime" / source_digest / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _require(not target.is_symlink() and sha256_file(target) == source_digest, "runtime_file_conflict")
        else:
            shutil.copyfile(source, target)
        runtime.append((target.resolve(), "worker_wheel" if source.suffix == ".whl" else "worker_dependencies"))
    request = build_request({
        "schema_version": REQUEST_SCHEMA, "source_capture_digest": inputs["binding"]["source_video_digest"],
        "source_commit_sha": source_commit_sha, "container_image_digest": worker_image_digest,
        "reconstruction_dataset_digest": inputs["digest"], "geometry_input_digest": inputs["digest"],
        "train_heldout_split_digest": canonical_digest({"candidate_frames": [r["frame_id"] for r in frames],
                                                        "source_digest": inputs["binding"]["source_video_digest"]}),
        "calibration_digest": None, "model_revision": MODEL_REVISION,
        "model_checkpoint_digest": MODEL_CHECKPOINT_DIGEST, "model_config_digest": MODEL_CONFIG_DIGEST,
        "remote_processing_authorization_digest": remote_processing_authorization_digest,
        "claim_ceiling": "development_only", "metric_measurement_proven": False,
        "candidate_may_read_hidden_heldout": False,
        "worker_runtime_digest": canonical_digest({p.name: "sha256:" + sha256_file(p) for p, _ in runtime}),
    })
    paths = [(input_manifest.resolve(), "geometry_inputs")]
    paths.extend(runtime)
    paths.extend((Path(row[key]), "candidate_observation") for row in frames
                 for key in ("source_image_path", "image_path"))
    paths = list(dict.fromkeys(paths))
    bindings = [{"artifact_id": f"input-{index:04d}", "relative_path": str(path.relative_to(input_manifest.parent.resolve())),
                 "digest": "sha256:" + sha256_file(path), "role": role, "contains_hidden_heldout_pixels": False}
                for index, (path, role) in enumerate(paths)]
    return compile_reconstruction_gpu_operation_bundle(operation=OPERATION, operation_request=request,
        artifact_root=input_manifest.parent, artifact_bindings=bindings, output_root=output_root)


def execute(*, request: Mapping[str, Any], input_manifest: Path, output_root: Path) -> dict[str, Any]:
    from .website_scene_geometry import infer_website_geometry_inputs

    request = build_request(request)
    inputs = json.loads(input_manifest.read_text())
    _require(inputs.get("digest") == request["geometry_input_digest"]
             and inputs.get("binding", {}).get("source_video_digest") == request["source_capture_digest"],
             "input_binding_mismatch")
    for name, field in (("model.safetensors", "model_checkpoint_digest"), ("config.json", "model_config_digest")):
        path = MODEL_ROOT / name
        _require(path.is_file() and "sha256:" + sha256_file(path) == request[field], "model_bytes_mismatch")
    root = output_root / request[REQUEST_DIGEST][7:23]
    started = time.monotonic()
    infer_website_geometry_inputs(input_manifest=input_manifest, output_root=root, model_path=MODEL_ROOT)
    manifest = json.loads((root / "source_geometry.json").read_text())
    files = {"source_geometry.json"}
    files.update(row[key] for row in manifest["frames"] for key in ("source_image_path", "image_path", "geometry_path"))
    return build_result({
        "schema_version": RESULT_SCHEMA, "status": "succeeded", REQUEST_DIGEST: request[REQUEST_DIGEST],
        "geometry_manifest_digest": manifest["digest"], "source_capture_digest": request["source_capture_digest"],
        "geometry_input_digest": request["geometry_input_digest"], "duration_seconds": time.monotonic() - started,
        "output_digests": [{"artifact_id": name, "digest": "sha256:" + sha256_file(root / name)} for name in sorted(files)],
        "heldout_labels_included": False, "candidate_self_graded": False, "metric_measurement_proven": False,
        "physical_evidence": False, "claim_ceiling": "development_only",
    })
