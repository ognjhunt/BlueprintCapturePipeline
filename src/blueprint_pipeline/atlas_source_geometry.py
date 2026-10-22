"""Atlas posing as website source geometry, in the shape MapAnything returns.

Beta journey step: concurrent scene preparation. An admitted Atlas
images2PosedRGBD run over the scene's own geometry inputs
(``website_worldlabs.prepare_atlas_pose_inputs``, bound to their digest)
becomes a ``website_source_geometry.v1`` result on the same uncropped
model-input grid MapAnything returns, so task masks lift onto it unchanged.

Atlas scales each input to cover its grid and centre-crops it
(docs/images-to-posed-images). Each model-input pixel takes the Atlas pixel
its centre falls in; pixels the crop removed are invalid. Intrinsics follow
the same map (Atlas puts pixel centres at +0.5, source geometry at integers),
cameras become OpenCV camera-to-world, and depth stays z-depth, read exactly
as ``atlas_depth_splat`` reads it.

Atlas depth is in unscaled world units, and every consumer of source geometry
reads metres. A result built without a scale is kept for comparison only and
``load_atlas_source_geometry`` refuses it; a scale names its basis and the
evidence it came from, and never proves a metric measurement. Nothing here
calls a provider, qualifies a scene, or routes a capture to Atlas.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .camera_geometry_validation import validate_se3_matrix
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

SCHEMA_VERSION = "website_source_geometry.v1"
BACKEND = "world_labs_atlas"
ADAPTER_VERSION = 1
POSE_RESULT = "atlas_posed_rgbd_result.json"
#: A floor that catches a wrong crop map, not the comparison's frozen threshold:
#: the returned RGB, averaged back onto the model grid, must match what was sent.
PIXEL_BINDING_MIN_NCC = 0.9
SCALE_BASES = ("measured_calibration", "reference_estimate")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_FRAME_FILES = (("source_image_path", "source_image_digest"), ("image_path", "image_digest"))
#: Atlas camera axes to OpenCV's (x right, y down, z forward).
_AXES = {"rub": np.diag([1.0, -1.0, -1.0]), "rdf": np.eye(3)}


def _scale_record(scale: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if scale is None:
        return None
    value = scale.get("meters_per_unit") if isinstance(scale, Mapping) else None
    if (not isinstance(scale, Mapping) or set(scale) != {"meters_per_unit", "basis", "evidence_digest",
                                                          "evidence_reference"}
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or scale["basis"] not in SCALE_BASES
            or not isinstance(scale["evidence_digest"], str) or not _SHA.fullmatch(scale["evidence_digest"])
            or not isinstance(scale["evidence_reference"], str) or not scale["evidence_reference"].strip()):
        raise ValueError("atlas_source_geometry_scale_invalid")
    return {**scale, "meters_per_unit": float(value)}


def crop_map(width: int, height: int, grid: tuple[int, int]) -> dict[str, Any]:
    """Where each model-input pixel centre lands on the Atlas grid after cover-and-centre-crop."""
    grid_width, grid_height = grid
    scale = max(grid_width / width, grid_height / height)
    offset_x, offset_y = (scale * width - grid_width) / 2, (scale * height - grid_height) / 2
    columns = np.floor(scale * (np.arange(width) + 0.5) - offset_x).astype(np.int64)
    rows = np.floor(scale * (np.arange(height) + 0.5) - offset_y).astype(np.int64)
    inside = ((rows >= 0) & (rows < grid_height))[:, None] & ((columns >= 0) & (columns < grid_width))[None, :]
    return {"scale": scale, "offset": (offset_x, offset_y), "inside": inside,
            "rows": np.clip(rows, 0, grid_height - 1), "columns": np.clip(columns, 0, grid_width - 1)}


def model_intrinsics(intrinsics: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[list[float]]:
    """Atlas grid intrinsics (pixel centres at +0.5) on the model grid (pixel centres at integers)."""
    scale, (offset_x, offset_y) = mapping["scale"], mapping["offset"]
    return [[intrinsics["fx"] / scale, 0.0, (intrinsics["cx"] + offset_x) / scale - 0.5],
            [0.0, intrinsics["fy"] / scale, (intrinsics["cy"] + offset_y) / scale - 0.5],
            [0.0, 0.0, 1.0]]


def opencv_world_from_camera(extrinsics: Mapping[str, Any], meters_per_unit: float = 1.0) -> np.ndarray:
    from .atlas_depth_splat import _rotation_matrix
    axes = _AXES.get(extrinsics.get("coordinateSystem", "rub"))
    if axes is None:
        raise ValueError("atlas_source_geometry_camera_axes_unsupported")
    pose = np.eye(4)
    pose[:3, :3] = _rotation_matrix(extrinsics["quaternion"]) @ axes
    pose[:3, 3] = np.asarray(extrinsics["position"], dtype=np.float64) * meters_per_unit
    if not validate_se3_matrix(pose.tolist(), field="world_from_camera")["valid"]:
        raise ValueError("atlas_source_geometry_camera_pose_invalid")
    return pose


def pixel_binding_ncc(model_rgb: np.ndarray, atlas_rgb: np.ndarray, mapping: Mapping[str, Any]) -> float:
    """Normalised cross-correlation of what was sent with what Atlas returned, area-averaged back."""
    scale, (offset_x, offset_y) = mapping["scale"], mapping["offset"]
    gray = atlas_rgb.astype(np.float64).mean(axis=-1)
    integral = np.pad(gray.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    height, width = model_rgb.shape[:2]

    def span(count: int, offset: float, limit: int) -> tuple[np.ndarray, np.ndarray]:
        edges = np.floor(scale * np.arange(count + 1) - offset).astype(np.int64)
        return np.clip(edges[:-1], 0, limit), np.clip(edges[1:], 0, limit)

    (r0, r1), (c0, c1) = span(height, offset_y, gray.shape[0]), span(width, offset_x, gray.shape[1])
    area = (r1 - r0)[:, None] * (c1 - c0)[None, :]
    total = (integral[r1][:, c1] - integral[r0][:, c1] - integral[r1][:, c0] + integral[r0][:, c0])
    keep = mapping["inside"] & (area > 0)
    if keep.sum() < 2:
        return 0.0
    returned = total[keep] / area[keep]
    sent = model_rgb.astype(np.float64).mean(axis=-1)[keep]
    returned, sent = returned - returned.mean(), sent - sent.mean()
    denominator = math.sqrt(float((returned ** 2).sum() * (sent ** 2).sum()))
    return float((returned * sent).sum() / denominator) if denominator > 0 else 0.0


def _confidence(path: Path) -> np.ndarray:
    import cv2
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.ndim != 2:
        raise ValueError("atlas_source_geometry_output_undecodable")
    scale = float(np.iinfo(value.dtype).max) if value.dtype.kind == "u" else 1.0
    return value.astype(np.float32) / scale


def _rgb(path: Path) -> np.ndarray:
    import cv2
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError("atlas_source_geometry_output_undecodable")
    return value[..., ::-1]


def atlas_source_geometry(*, geometry_root: Path, output_root: Path | None = None,
                          scale: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write ``source_geometry.json`` from a retained, admitted Atlas pose run over these inputs."""
    from .atlas_depth_splat import read_exr_depth
    from .website_atlas_pose import verify_atlas_pose_receipt
    from .website_scene_geometry import INPUT_SCHEMA, _bound_frames
    from .website_worldlabs import _atlas_posed_context

    geometry_root = geometry_root.resolve()
    output_root = (output_root or geometry_root / "atlas_source_geometry").resolve()
    input_manifest = geometry_root / "worker_inputs" / "geometry_inputs.json"
    inputs = json.loads(input_manifest.read_text())
    if inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("heldout_pixels_included") is not False:
        raise ValueError("atlas_source_geometry_inputs_invalid")
    prepared = _bound_frames(inputs, input_manifest.parent, geometry=False)
    pose = json.loads((geometry_root / POSE_RESULT).read_text())
    if (pose.get("schema_version") != "website_atlas_pose_result.v1"
            or pose.get("digest") != canonical_digest(pose, digest_field="digest")
            or pose.get("binding") != {"source_video_digest": inputs["binding"]["source_video_digest"],
                                       "geometry_input_digest": inputs["digest"]}):
        raise ValueError("atlas_source_geometry_pose_binding_mismatch")
    # Only an admitted run whose receipt still binds these exact frames and retained outputs.
    verify_atlas_pose_receipt(pose=pose, frames=prepared, output_root=geometry_root)
    _atlas_posed_context(operation=pose["operation"], expected_frame_count=len(prepared))
    record = _scale_record(scale)
    meters_per_unit = record["meters_per_unit"] if record else 1.0
    binding = {**inputs["binding"], "input_digest": inputs["digest"], "geometry_backend": BACKEND,
               "atlas_geometry_adapter_version": ADAPTER_VERSION, "atlas_pose_result_digest": pose["digest"],
               "atlas_operation_id": pose["operation"]["id"], "scale": record}
    manifest_path = output_root / "source_geometry.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("binding") != binding:
            raise ValueError("atlas_source_geometry_identity_conflict")
        return _rebound(previous, manifest_path)
    usage = pose["operation_receipt"]["input_usage"]["frames"]
    frames = []
    for index, (row, returned, used) in enumerate(zip(prepared, pose["operation"]["response"]["frames"], usage,
                                                      strict=True)):
        outputs = {output["role"]: geometry_root / output["path"] for output in used["outputs"]}
        if "confidence" not in outputs:
            raise ValueError("atlas_source_geometry_confidence_missing")
        camera = returned["camera"]
        grid = (int(camera["intrinsics"]["width"]), int(camera["intrinsics"]["height"]))
        depth = read_exr_depth(outputs["depth"])
        confidence, rgb = _confidence(outputs["confidence"]), _rgb(outputs["image"])
        if depth.shape != grid[::-1] or confidence.shape != grid[::-1] or rgb.shape[:2] != grid[::-1]:
            raise ValueError("atlas_source_geometry_grid_mismatch")
        with Image.open(row["image_path"]) as image:
            sent = np.asarray(image.convert("RGB"))
        if sent.shape[:2] != (int(row["height"]), int(row["width"])):
            raise ValueError("atlas_source_geometry_input_grid_mismatch")
        mapping = crop_map(int(row["width"]), int(row["height"]), grid)
        ncc = pixel_binding_ncc(sent, rgb, mapping)
        if ncc < PIXEL_BINDING_MIN_NCC:
            raise ValueError("atlas_source_geometry_pixel_binding_failed")
        sampled_depth = depth[mapping["rows"][:, None], mapping["columns"][None, :]]
        sampled_confidence = confidence[mapping["rows"][:, None], mapping["columns"][None, :]]
        with np.errstate(invalid="ignore"):
            valid = (mapping["inside"] & np.isfinite(sampled_depth) & (sampled_depth > 0)
                     & np.isfinite(sampled_confidence) & (sampled_confidence > 0))
        if not valid.any():
            raise ValueError("atlas_source_geometry_no_valid_depth")
        world_from_camera = opencv_world_from_camera(camera["extrinsics"], meters_per_unit)
        geometry_path = output_root / "geometry" / f"{index:06d}.npz"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(geometry_path,
                            depth_m=np.where(valid, sampled_depth * meters_per_unit, 0).astype(np.float32),
                            confidence=np.where(valid, sampled_confidence, 0).astype(np.float32), valid_mask=valid)
        frame = dict(row)
        # Retain input bytes beside the output so the whole result can move hosts.
        for key, _ in _FRAME_FILES:
            source = Path(row[key])
            target = output_root / key / f"{index:06d}{source.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            frame[key] = str(target.relative_to(output_root))
        frames.append({**frame, "geometry_path": str(geometry_path.relative_to(output_root)),
                       "geometry_digest": _sha256_file(geometry_path),
                       "intrinsics": model_intrinsics(camera["intrinsics"], mapping),
                       "world_from_camera": world_from_camera.tolist(),
                       "camera_from_world": np.linalg.inv(world_from_camera).tolist(),
                       "valid_pixel_fraction": float(valid.mean()),
                       "atlas_frame": {"grid": list(grid), "cover_scale": mapping["scale"],
                                       "crop_offset": list(mapping["offset"]), "pixel_binding_ncc": ncc,
                                       "coordinate_system": camera["extrinsics"].get("coordinateSystem", "rub")}})
    result = {"schema_version": SCHEMA_VERSION, "status": "estimated", "binding": binding,
              "geometry_backend": BACKEND, "claim_ceiling": "development_only",
              "coordinate_system": "opencv_camera_to_world",
              "unit": "estimated_meters" if record else "provider_world_units",
              "scale_status": "reference_scaled" if record else "unscaled_provider_world_units",
              "metric_measurement_proven": False, "physical_evidence": False,
              "confidence_is_calibrated_probability": False, "frames": frames}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(manifest_path, result)
    return _rebound(result, manifest_path)


def _rebound(result: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    from .website_scene_geometry import _bound_frames
    frames = _bound_frames(result, manifest_path.parent, geometry=True)
    value = {**result, "frames": frames, "manifest_path": str(manifest_path.resolve())}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def load_atlas_source_geometry(*, manifest_path: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Rebind a retained Atlas geometry to these inputs; only a scaled one is usable as metres."""
    result = json.loads(manifest_path.read_text())
    if result.get("scale_status") == "unscaled_provider_world_units":
        raise ValueError("atlas_source_geometry_unscaled")
    expected = {"schema_version": SCHEMA_VERSION, "status": "estimated", "geometry_backend": BACKEND,
                "claim_ceiling": "development_only", "coordinate_system": "opencv_camera_to_world",
                "unit": "estimated_meters", "scale_status": "reference_scaled",
                "metric_measurement_proven": False, "physical_evidence": False,
                "confidence_is_calibrated_probability": False}
    binding = result.get("binding", {})
    if (any(result.get(key) != value for key, value in expected.items())
            or any(binding.get(key) != value for key, value in inputs["binding"].items())
            or binding.get("input_digest") != inputs["digest"] or binding.get("geometry_backend") != BACKEND
            or _scale_record(binding.get("scale")) is None):
        raise ValueError("atlas_source_geometry_identity_conflict")
    value = _rebound(result, manifest_path)
    if len(value["frames"]) != len(inputs["frames"]):
        raise ValueError("atlas_source_geometry_frame_mismatch")
    for frame, source in zip(value["frames"], inputs["frames"], strict=True):
        if any(frame.get(key) != item for key, item in source.items() if key not in {"source_image_path", "image_path"}):
            raise ValueError("atlas_source_geometry_frame_mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--geometry-root", type=Path, required=True,
                        help="Directory holding worker_inputs/, atlas_pose_inputs.json and the admitted pose run")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scale", type=Path, help="JSON: meters_per_unit, basis, evidence_digest, evidence_reference")
    args = parser.parse_args()
    scale = json.loads(args.scale.read_text()) if args.scale else None
    result = atlas_source_geometry(geometry_root=args.geometry_root, output_root=args.output_root, scale=scale)
    print(json.dumps({"status": result["status"], "unit": result["unit"], "scale_status": result["scale_status"],
                      "frame_count": len(result["frames"]), "manifest_path": result["manifest_path"],
                      "digest": result["digest"]}))


if __name__ == "__main__":
    main()
