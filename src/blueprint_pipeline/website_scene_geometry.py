"""Estimated source geometry for website task preparation, before pixel editing.

ADP-009B / public_scene_day_14: preserve source placement for a development
replica. Model-derived meters never establish measured scale or physical proof.
The existing decoder owns source frames; MapAnything owns their joint geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .camera_geometry_validation import validate_se3_matrix
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import LocalDecodedObservationAdapter, _sha256_file

MODEL_ID = "facebook/map-anything-apache"
MODEL_CODE_REVISION = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
INPUT_SCHEMA = "website_geometry_inputs.v1"
_FRAME_ARTIFACTS = (("source_image_path", "source_image_digest"),
                    ("image_path", "image_digest"), ("geometry_path", "geometry_digest"))


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _infer(paths: list[str], *, model_path: Path) -> Sequence[Mapping[str, Any]]:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    # Images are already resized. Keeping this size avoids a hidden center crop
    # that would invalidate the source-pixel-to-geometry transform.
    with Image.open(paths[0]) as image:
        size = image.size
    views = load_images(paths, resize_mode="fixed_size", size=size)
    if len(views) != len(paths):
        raise ValueError("mapanything_source_view_missing")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    with torch.inference_mode():
        return model.infer(views, memory_efficient_inference=True, use_amp=device == "cuda",
                           apply_mask=True, mask_edges=True, apply_confidence_mask=False)


def _prepare_image(source: Path, target: Path, rotation: float) -> dict[str, Any]:
    """Keep an exact source-to-model pixel transform, including phone rotation."""
    if not np.isfinite(rotation) or rotation % 90:
        raise ValueError("website_source_rotation_not_supported")
    with Image.open(source) as original:
        width, height = original.size
        image = original.convert("RGB").rotate(rotation, expand=True)
        upright_width, upright_height = image.size
        scale = 518 / max(image.size)
        size = tuple(max(14, round(dimension * scale / 14) * 14) for dimension in image.size)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.resize(size, Image.Resampling.LANCZOS).save(target)
    transforms = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
        180: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        270: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
    }
    sx, sy = size[0] / upright_width, size[1] / upright_height
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return {
        "image_path": str(target), "image_digest": _sha256_file(target),
        "source_width": width, "source_height": height,
        "width": size[0], "height": size[1], "display_rotation_degrees": rotation,
        "source_to_geometry_pixels": (resize @ np.array(transforms[int(rotation) % 360])).tolist(),
    }


def _write_prediction(prediction: Mapping[str, Any], frame: Mapping[str, Any], path: Path) -> dict[str, Any]:
    height, width = int(frame["height"]), int(frame["width"])
    depth = _array(prediction["depth_z"])
    confidence = _array(prediction["conf"])
    mask = _array(prediction["mask"])
    if depth.shape != (1, height, width, 1) or confidence.shape != (1, height, width) or mask.shape != depth.shape:
        raise ValueError("mapanything_geometry_shape_mismatch")
    depth, confidence, mask = depth[0, :, :, 0], confidence[0], mask[0, :, :, 0]
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("mapanything_validity_mask_invalid")
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence >= 0)
    if not valid.any():
        raise ValueError("mapanything_no_valid_geometry")
    intrinsics, poses = _array(prediction["intrinsics"]), _array(prediction["camera_poses"])
    if intrinsics.shape != (1, 3, 3) or poses.shape != (1, 4, 4):
        raise ValueError("mapanything_camera_shape_mismatch")
    k, pose = intrinsics[0], poses[0]
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("mapanything_intrinsics_invalid")
    if not validate_se3_matrix(pose.tolist(), field="world_from_camera")["valid"]:
        raise ValueError("mapanything_camera_pose_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth_m=np.where(valid, depth, 0).astype(np.float32),
                        confidence=np.where(valid, confidence, 0).astype(np.float32), valid_mask=valid)
    return {**frame, "geometry_path": str(path), "geometry_digest": _sha256_file(path),
            "intrinsics": k.tolist(), "world_from_camera": pose.tolist(),
            "camera_from_world": np.linalg.inv(pose).tolist(),
            "valid_pixel_fraction": float(valid.mean())}


def _bound_frames(document: Mapping[str, Any], root: Path, *, geometry: bool) -> list[dict[str, Any]]:
    if document.get("digest") != canonical_digest(document, digest_field="digest"):
        raise ValueError("website_source_geometry_digest_mismatch")
    rows = document.get("frames")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 16:
        raise ValueError("website_source_frames_invalid")
    frames = []
    for row in rows:
        frame = dict(row)
        for path_key, digest_key in _FRAME_ARTIFACTS if geometry else _FRAME_ARTIFACTS[:2]:
            path = Path(frame[path_key])
            path = path if path.is_absolute() else root / path
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("website_source_geometry_artifact_escape")
            if not path.is_file() or _sha256_file(path) != frame[digest_key]:
                raise ValueError("website_source_geometry_artifact_changed")
            frame[path_key] = str(path.resolve())
        frames.append(frame)
    if len({row["frame_id"] for row in frames}) != len(frames):
        raise ValueError("website_source_frame_identity_duplicate")
    return frames


def prepare_website_geometry_inputs(*, source_video: Path, output_root: Path, capture_id: str) -> dict[str, Any]:
    """Decode on CPU. The portable worker input directory excludes held-out frames."""
    output_root = output_root.resolve()
    manifest_path = output_root / "worker_inputs" / "geometry_inputs.json"
    binding = {"source_video_digest": _sha256_file(source_video), "model_id": MODEL_ID,
               "model_code_revision": MODEL_CODE_REVISION, "adapter_version": 1}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("binding") == binding and previous.get("schema_version") == INPUT_SCHEMA:
            _bound_frames(previous, manifest_path.parent, geometry=False)
            return previous
        raise ValueError("website_geometry_input_identity_conflict")
    decoded = LocalDecodedObservationAdapter().execute(
        intake_id=capture_id, capture_digest=binding["source_video_digest"],
        capture_authority_profile="monocular_video", capture_root=source_video.parent,
        video_relative_path=source_video.name, output_root=output_root / "source",
        rights_and_retention={"local_processing_authorized": True,
                              "provider_upload_authorized": False, "paid_compute_authorized": False},
        maximum_frames=16,
    )
    refs = decoded["asset_references"]
    candidate_path = output_root / "source" / refs["candidate_dataset_manifest"]["relative_path"]
    index_path = output_root / "source" / refs["decoded_observation_index"]["relative_path"]
    candidate, index = json.loads(candidate_path.read_text()), json.loads(index_path.read_text())
    if candidate.get("heldout_pixels_included") is not False or not candidate["frames"]:
        raise ValueError("website_source_frames_invalid")
    rotation = float(index["stream_metadata"].get("display_rotation_degrees") or 0)
    prepared = []
    for index, row in enumerate(candidate["frames"]):
        source = (candidate_path.parent / row["candidate_relative_path"]).resolve()
        if not source.is_relative_to(candidate_path.parent) or _sha256_file(source) != row["frame_digest"]:
            raise ValueError("website_source_frame_binding_invalid")
        retained = manifest_path.parent / "observations" / f"{index:06d}{source.suffix}"
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, retained)
        frame = {
            "frame_id": row["frame_id"], "timestamp_seconds": row["t_video_sec"],
            "source_image_path": str(retained), "source_image_digest": row["frame_digest"],
            **_prepare_image(source, manifest_path.parent / "model_inputs" / f"{index:06d}.png", rotation),
        }
        for key, _ in _FRAME_ARTIFACTS[:2]:
            frame[key] = str(Path(frame[key]).relative_to(manifest_path.parent))
        prepared.append(frame)
    result = {"schema_version": INPUT_SCHEMA, "binding": binding, "frames": prepared,
              "heldout_pixels_included": False, "claim_ceiling": "development_only"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(manifest_path, result)
    return result


def infer_website_geometry_inputs(*, input_manifest: Path, output_root: Path, model_path: Path) -> dict[str, Any]:
    """Worker entry point; allocation and model installation remain outside this stage."""
    weights, config = model_path / "model.safetensors", model_path / "config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError("mapanything_local_checkpoint_missing")
    inputs = json.loads(input_manifest.read_text())
    if (inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("heldout_pixels_included") is not False
            or inputs.get("binding", {}).get("model_code_revision") != MODEL_CODE_REVISION
            or inputs.get("binding", {}).get("model_id") != MODEL_ID):
        raise ValueError("website_geometry_inputs_invalid")
    prepared = _bound_frames(inputs, input_manifest.parent, geometry=False)
    binding = {**inputs["binding"], "checkpoint_digest": _sha256_file(weights),
               "model_config_digest": _sha256_file(config), "input_digest": inputs["digest"]}
    output_root = output_root.resolve()
    manifest_path = output_root / "source_geometry.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("binding") != binding or previous.get("status") != "estimated":
            raise ValueError("website_geometry_result_identity_conflict")
        return load_website_geometry_result(manifest_path=manifest_path, inputs=inputs)
    # Retain input bytes beside the output so the whole result can move hosts.
    for index, row in enumerate(prepared):
        for key, _ in _FRAME_ARTIFACTS[:2]:
            source = Path(row[key])
            target = output_root / key / f"{index:06d}{source.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            row[key] = str(target)
    predictions = _infer([row["image_path"] for row in prepared], model_path=model_path)
    if len(predictions) != len(prepared):
        raise ValueError("mapanything_prediction_count_mismatch")
    frames = [_write_prediction(prediction, row, output_root / "geometry" / f"{index:06d}.npz")
              for index, (row, prediction) in enumerate(zip(prepared, predictions, strict=True))]
    for frame in frames:
        for key, _ in _FRAME_ARTIFACTS:
            frame[key] = str(Path(frame[key]).relative_to(output_root))
    result = {
        "schema_version": "website_source_geometry.v1", "status": "estimated",
        "binding": binding, "claim_ceiling": "development_only",
        "coordinate_system": "opencv_camera_to_world", "unit": "estimated_meters",
        "scale_status": "model_estimated", "metric_measurement_proven": False,
        "physical_evidence": False, "confidence_is_calibrated_probability": False,
        "frames": frames,
    }
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(manifest_path, result)
    return load_website_geometry_result(manifest_path=manifest_path, inputs=inputs)


def load_website_geometry_result(*, manifest_path: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Rebind a returned worker directory to this host, checking its input identity."""
    result = json.loads(manifest_path.read_text())
    binding = result.get("binding", {})
    expected = {"schema_version": "website_source_geometry.v1", "status": "estimated",
                "claim_ceiling": "development_only", "metric_measurement_proven": False,
                "physical_evidence": False, "scale_status": "model_estimated",
                "unit": "estimated_meters", "coordinate_system": "opencv_camera_to_world",
                "confidence_is_calibrated_probability": False}
    if (any(result.get(key) != value for key, value in expected.items())
            or any(binding.get(key) != value for key, value in inputs["binding"].items())
            or binding.get("input_digest") != inputs["digest"]):
        raise ValueError("website_geometry_result_identity_conflict")
    frames = _bound_frames(result, manifest_path.parent, geometry=True)
    if len(frames) != len(inputs["frames"]):
        raise ValueError("website_geometry_result_frame_mismatch")
    for frame, source in zip(frames, inputs["frames"], strict=True):
        if any(frame.get(key) != value for key, value in source.items()
               if key not in {"source_image_path", "image_path"}):
            raise ValueError("website_geometry_result_frame_mismatch")
    result.update(frames=frames, manifest_path=str(manifest_path.resolve()))
    result["digest"] = canonical_digest(result, digest_field="digest")
    return result


def run_website_scene_geometry(*, source_video: Path, output_root: Path, capture_id: str,
                               task_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Local inference compatibility entry; CPU preparation is reusable by a worker."""
    inputs = prepare_website_geometry_inputs(source_video=source_video, output_root=output_root, capture_id=capture_id)
    if task_context is not None:
        from .paid_resource_allocator import run_sponsored_website_geometry
        if task_context.get("capture_id") != capture_id or task_context.get("confirmed") is not True:
            raise ValueError("website_mapanything_task_capture_mismatch")
        return run_sponsored_website_geometry(input_manifest=output_root / "worker_inputs/geometry_inputs.json",
                                               output_root=output_root, task_context=task_context)
    remote_result = os.getenv("BLUEPRINT_WEBSITE_GEOMETRY_RESULT")
    if remote_result:
        return load_website_geometry_result(manifest_path=Path(remote_result), inputs=inputs)
    model_path = Path(os.getenv("BLUEPRINT_MAPANYTHING_MODEL_PATH") or "/opt/mapanything/map-anything-apache")
    if not (model_path / "model.safetensors").is_file() or not (model_path / "config.json").is_file():
        raise ValueError("mapanything_local_checkpoint_missing")
    return infer_website_geometry_inputs(input_manifest=output_root / "worker_inputs" / "geometry_inputs.json",
                                         output_root=output_root, model_path=model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-video", type=Path, required=True)
    prepare.add_argument("--capture-id", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--input-manifest", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--model-path", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = prepare_website_geometry_inputs(**args) if command == "prepare" else infer_website_geometry_inputs(**args)
    print(json.dumps({"status": result.get("status", "prepared"), "frame_count": len(result["frames"]),
                      "digest": result["digest"]}))


if __name__ == "__main__":
    main()
