"""Deterministic ARKit depth back-projection into an observed surface mesh.

No camera convention, scale, pose, or missing depth is inferred here.  Every
frame binds exact depth/confidence bytes, calibrated depth intrinsics, an
explicit camera-ray convention, and a world-from-camera transform.  Triangles
are emitted only across retained high-confidence samples whose metric depth and
edge continuity satisfy the frozen request.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .camera_geometry_validation import validate_se3_matrix
from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA = "arkit_depth_surface_compilation_request.v1"
RESULT_SCHEMA = "arkit_depth_surface_compilation_result.v1"
OUTPUT_SURFACE_SCHEMA = "observed_surface_mesh.v1"
IMPLEMENTATION_VERSION = "1.0.0"
METHOD_ID = "blueprint.arkit_depth_observed_surface_backprojection"
CAMERA_CONVENTIONS = {
    "arkit_x_right_y_up_z_backward",
    "opencv_x_right_y_down_z_forward",
}
MAX_FRAME_COUNT = 10_000
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_PIXELS_PER_FRAME = 64 * 1024 * 1024

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ArkitDepthSurfaceCompilerError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _finite(value: Any, *, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _safe_input(root: Path, relative_path: Any, *, label: str) -> Path:
    text = str(relative_path or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ArkitDepthSurfaceCompilerError([f"{label}_relative_path_unsafe"])
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ArkitDepthSurfaceCompilerError([f"{label}_symlink_forbidden"])
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArkitDepthSurfaceCompilerError([f"{label}_missing"]) from exc
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise ArkitDepthSurfaceCompilerError([f"{label}_escape_or_not_file"])
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise ArkitDepthSurfaceCompilerError([f"{label}_oversized"])
    return resolved


def _prepare_output(output_root: str | Path, artifact_root: Path) -> Path:
    root = artifact_root.resolve(strict=True)
    output = Path(output_root)
    if output.is_symlink():
        raise ArkitDepthSurfaceCompilerError(["output_root_symlink_forbidden"])
    resolved = output.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ArkitDepthSurfaceCompilerError(["output_root_outside_artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or output.resolve(strict=True) != resolved:
        raise ArkitDepthSurfaceCompilerError(["output_root_symlink_forbidden"])
    return output


def _load_array(path: Path, *, encoding: str, label: str) -> np.ndarray:
    try:
        if encoding == "npy":
            array = np.load(path, allow_pickle=False)
        elif encoding in {"uint16_png", "uint8_png"}:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                array = np.asarray(image)
        else:
            raise ArkitDepthSurfaceCompilerError([f"{label}_encoding_unsupported"])
    except ArkitDepthSurfaceCompilerError:
        raise
    except Exception as exc:
        raise ArkitDepthSurfaceCompilerError([f"{label}_decode_failed"]) from exc
    array = np.asarray(array)
    if array.ndim != 2 or array.size == 0 or array.size > MAX_PIXELS_PER_FRAME:
        raise ArkitDepthSurfaceCompilerError([f"{label}_shape_invalid"])
    if encoding == "uint16_png" and array.dtype != np.uint16:
        raise ArkitDepthSurfaceCompilerError([f"{label}_dtype_invalid"])
    if encoding == "uint8_png" and array.dtype != np.uint8:
        raise ArkitDepthSurfaceCompilerError([f"{label}_dtype_invalid"])
    if encoding == "npy" and not np.issubdtype(array.dtype, np.number):
        raise ArkitDepthSurfaceCompilerError([f"{label}_dtype_invalid"])
    try:
        if not np.isfinite(array).all():
            raise ArkitDepthSurfaceCompilerError([f"{label}_nonfinite"])
    except TypeError as exc:
        raise ArkitDepthSurfaceCompilerError([f"{label}_dtype_invalid"]) from exc
    return array


def _intrinsics(value: Any, *, shape: tuple[int, int], frame_id: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ArkitDepthSurfaceCompilerError([f"depth_intrinsics_missing:{frame_id}"])
    width = value.get("width")
    height = value.get("height")
    fx = _finite(value.get("fx"), minimum=0.000001)
    fy = _finite(value.get("fy"), minimum=0.000001)
    cx = _finite(value.get("cx"))
    cy = _finite(value.get("cy"))
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or (height, width) != shape
        or fx is None
        or fy is None
        or cx is None
        or cy is None
        or not 0 <= cx < width
        or not 0 <= cy < height
    ):
        raise ArkitDepthSurfaceCompilerError([f"depth_intrinsics_invalid:{frame_id}"])
    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _camera_point(
    *, u: int, v: int, depth_m: float, intrinsics: Mapping[str, float], convention: str
) -> np.ndarray:
    x = (float(u) - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y_pixel = (float(v) - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    if convention == "arkit_x_right_y_up_z_backward":
        return np.asarray([x, -y_pixel, -depth_m, 1.0], dtype=np.float64)
    if convention == "opencv_x_right_y_down_z_forward":
        return np.asarray([x, y_pixel, depth_m, 1.0], dtype=np.float64)
    raise ArkitDepthSurfaceCompilerError(["camera_ray_convention_unsupported"])


def _stable_element_id(prefix: str, frame_id: str, *coordinates: int) -> str:
    payload = ":".join([frame_id, *(str(item) for item in coordinates)])
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _triangle_allowed(
    ids: tuple[str, str, str],
    *,
    points: Mapping[str, np.ndarray],
    depths: Mapping[str, float],
    maximum_edge_length_m: float,
    maximum_depth_discontinuity_m: float,
) -> bool:
    xyz = [points[item] for item in ids]
    if max(depths[item] for item in ids) - min(depths[item] for item in ids) > (
        maximum_depth_discontinuity_m
    ):
        return False
    return all(
        float(np.linalg.norm(xyz[left] - xyz[right])) <= maximum_edge_length_m
        for left, right in ((0, 1), (1, 2), (2, 0))
    )


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if path.is_file():
        if path.read_bytes() != payload:
            raise ArkitDepthSurfaceCompilerError(["output_artifact_immutable_conflict"])
        return digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def compile_arkit_depth_surface(
    *, source_artifact: Mapping[str, Any], artifact_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Back-project high-confidence captured depth without fusing or filling it."""

    request = json.loads(json.dumps(dict(source_artifact)))
    request_digest = request.get("arkit_depth_surface_compilation_request_digest")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or not isinstance(request_digest, str)
        or request_digest
        != canonical_digest(
            request, digest_field="arkit_depth_surface_compilation_request_digest"
        )
    ):
        raise ArkitDepthSurfaceCompilerError(["arkit_depth_surface_request_invalid"])
    if _COMMIT.fullmatch(str(request.get("source_commit_sha") or "")) is None:
        raise ArkitDepthSurfaceCompilerError(["source_commit_sha_invalid"])
    if request.get("capture_profile") != "iphone_arkit_lidar":
        raise ArkitDepthSurfaceCompilerError(["iphone_arkit_lidar_profile_required"])
    if request.get("camera_ray_convention") not in CAMERA_CONVENTIONS:
        raise ArkitDepthSurfaceCompilerError(["camera_ray_convention_unsupported"])
    if request.get("metric_scale_status") not in {
        "sensor_metric_unvalidated",
        "validated",
    }:
        raise ArkitDepthSurfaceCompilerError(["metric_scale_status_invalid"])
    if request.get("generated_fill_used") is not False:
        raise ArkitDepthSurfaceCompilerError(["generated_or_unseen_fill_forbidden"])
    if request.get("candidate_may_read_hidden_heldout") is not False:
        raise ArkitDepthSurfaceCompilerError(["hidden_heldout_access_forbidden"])
    stride = request.get("pixel_stride")
    accepted_values = request.get("accepted_confidence_values")
    edge_limit = _finite(request.get("maximum_edge_length_m"), minimum=0.000001)
    discontinuity_limit = _finite(
        request.get("maximum_depth_discontinuity_m"), minimum=0.0
    )
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or not 1 <= stride <= 64
        or not isinstance(accepted_values, list)
        or not accepted_values
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in accepted_values
        )
        or edge_limit is None
        or discontinuity_limit is None
    ):
        raise ArkitDepthSurfaceCompilerError(["surface_filter_configuration_invalid"])
    accepted_confidence = set(accepted_values)
    frames = request.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAME_COUNT:
        raise ArkitDepthSurfaceCompilerError(["depth_frames_invalid"])
    frame_ids = [str(row.get("frame_id") or "") for row in frames if isinstance(row, Mapping)]
    if len(frame_ids) != len(frames) or len(set(frame_ids)) != len(frame_ids) or any(
        _ID.fullmatch(item) is None for item in frame_ids
    ):
        raise ArkitDepthSurfaceCompilerError(["depth_frame_ids_invalid"])
    root = Path(artifact_root)
    original_digests = {
        str(row.get("digest") or "")
        for row in request.get("original_file_references") or []
        if isinstance(row, Mapping)
    }
    vertices: list[dict[str, Any]] = []
    faces: list[dict[str, Any]] = []
    accepted_pixel_count = 0
    rejected_pixel_count = 0
    discontinuity_rejected_triangle_count = 0
    regions_with_faces: set[str] = set()
    input_digests: list[dict[str, str]] = [
        {"artifact_id": "arkit_depth_surface_request", "digest": request_digest}
    ]
    for frame in sorted((dict(row) for row in frames), key=lambda row: str(row["frame_id"])):
        frame_id = str(frame["frame_id"])
        if frame.get("split") not in {"training", "validation"}:
            raise ArkitDepthSurfaceCompilerError([f"hidden_or_invalid_split_forbidden:{frame_id}"])
        region_id = str(frame.get("region_id") or "")
        if _ID.fullmatch(region_id) is None:
            raise ArkitDepthSurfaceCompilerError([f"depth_region_id_invalid:{frame_id}"])
        depth_binding = frame.get("depth_asset")
        confidence_binding = frame.get("confidence_asset")
        if not isinstance(depth_binding, Mapping) or not isinstance(confidence_binding, Mapping):
            raise ArkitDepthSurfaceCompilerError([f"depth_confidence_binding_missing:{frame_id}"])
        depth_path = _safe_input(root, depth_binding.get("relative_path"), label="depth_asset")
        confidence_path = _safe_input(
            root, confidence_binding.get("relative_path"), label="confidence_asset"
        )
        depth_digest = _sha256(depth_path)
        confidence_digest = _sha256(confidence_path)
        if depth_binding.get("digest") != depth_digest or confidence_binding.get(
            "digest"
        ) != confidence_digest:
            raise ArkitDepthSurfaceCompilerError([f"depth_confidence_digest_mismatch:{frame_id}"])
        if depth_digest not in original_digests or confidence_digest not in original_digests:
            raise ArkitDepthSurfaceCompilerError(
                [f"depth_confidence_provenance_binding_missing:{frame_id}"]
            )
        depth = _load_array(
            depth_path, encoding=str(depth_binding.get("encoding") or ""), label="depth_asset"
        )
        confidence = _load_array(
            confidence_path,
            encoding=str(confidence_binding.get("encoding") or ""),
            label="confidence_asset",
        )
        if depth.shape != confidence.shape:
            raise ArkitDepthSurfaceCompilerError([f"depth_confidence_shape_mismatch:{frame_id}"])
        scale = _finite(depth_binding.get("scale_to_meters"), minimum=0.000000001)
        if scale is None:
            raise ArkitDepthSurfaceCompilerError([f"depth_scale_to_meters_invalid:{frame_id}"])
        intrinsics = _intrinsics(frame.get("depth_intrinsics"), shape=depth.shape, frame_id=frame_id)
        pose_result = validate_se3_matrix(frame.get("T_world_camera"), field="T_world_camera")
        if pose_result.get("valid") is not True:
            raise ArkitDepthSurfaceCompilerError([f"world_from_camera_invalid:{frame_id}"])
        world_from_camera = np.asarray(pose_result["matrix"], dtype=np.float64)
        points: dict[str, np.ndarray] = {}
        depths: dict[str, float] = {}
        accepted_mask = np.isin(confidence, list(accepted_confidence)) & (depth > 0)
        for v in range(0, depth.shape[0], stride):
            for u in range(0, depth.shape[1], stride):
                if not bool(accepted_mask[v, u]):
                    rejected_pixel_count += 1
                    continue
                depth_m = float(depth[v, u]) * float(scale)
                if not math.isfinite(depth_m) or depth_m <= 0:
                    rejected_pixel_count += 1
                    continue
                camera = _camera_point(
                    u=u,
                    v=v,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    convention=str(request["camera_ray_convention"]),
                )
                world = world_from_camera @ camera
                if abs(float(world[3]) - 1.0) > 1e-8 or not np.isfinite(world[:3]).all():
                    raise ArkitDepthSurfaceCompilerError([f"backprojected_point_invalid:{frame_id}"])
                vertex_id = _stable_element_id("vertex", frame_id, u, v)
                points[vertex_id] = world[:3]
                depths[vertex_id] = depth_m
                vertices.append(
                    {
                        "vertex_id": vertex_id,
                        "position_m": [float(item) for item in world[:3]],
                        "confidence": 1.0,
                        "source_confidence_value": int(confidence[v, u]),
                        "region_id": region_id,
                        "source_observation_ids": [
                            _stable_element_id("depth-observation", frame_id, u, v)
                        ],
                        "source_pixel": {"frame_id": frame_id, "u": u, "v": v},
                        "generated": False,
                    }
                )
                accepted_pixel_count += 1
        for v in range(0, depth.shape[0] - stride, stride):
            for u in range(0, depth.shape[1] - stride, stride):
                corners = (
                    _stable_element_id("vertex", frame_id, u, v),
                    _stable_element_id("vertex", frame_id, u + stride, v),
                    _stable_element_id("vertex", frame_id, u, v + stride),
                    _stable_element_id("vertex", frame_id, u + stride, v + stride),
                )
                triangles = ((corners[0], corners[1], corners[3]), (corners[0], corners[3], corners[2]))
                for triangle_index, triangle in enumerate(triangles):
                    if any(item not in points for item in triangle):
                        continue
                    if not _triangle_allowed(
                        triangle,
                        points=points,
                        depths=depths,
                        maximum_edge_length_m=float(edge_limit),
                        maximum_depth_discontinuity_m=float(discontinuity_limit),
                    ):
                        discontinuity_rejected_triangle_count += 1
                        continue
                    face_id = _stable_element_id(
                        "face", frame_id, u, v, triangle_index
                    )
                    faces.append(
                        {
                            "face_id": face_id,
                            "vertex_ids": list(triangle),
                            "region_id": region_id,
                            "observed": True,
                            "generated": False,
                        }
                    )
                    regions_with_faces.add(region_id)
        input_digests.extend(
            [
                {"artifact_id": f"depth:{frame_id}", "digest": depth_digest},
                {"artifact_id": f"confidence:{frame_id}", "digest": confidence_digest},
            ]
        )
    if not vertices or not faces:
        raise ArkitDepthSurfaceCompilerError(["insufficient_high_confidence_surface"])
    vertices.sort(key=lambda row: row["vertex_id"])
    faces.sort(key=lambda row: row["face_id"])
    frame_declarations = request.get("declared_region_ids")
    unsupported_input = request.get("unsupported_region_ids")
    if (
        not isinstance(frame_declarations, list)
        or not isinstance(unsupported_input, list)
        or any(
            _ID.fullmatch(str(item)) is None
            for item in frame_declarations + unsupported_input
        )
    ):
        raise ArkitDepthSurfaceCompilerError(["region_ledger_invalid"])
    unsupported = sorted(
        set(str(item) for item in frame_declarations) - regions_with_faces
        | set(str(item) for item in unsupported_input)
    )
    surface = {
        "schema_version": OUTPUT_SURFACE_SCHEMA,
        "source_capture_digest": request["source_capture_digest"],
        "train_heldout_split_digest": request["train_heldout_split_digest"],
        "camera_calibration_binding": request["camera_calibration_binding"],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "camera_ray_convention": request["camera_ray_convention"],
        "metric_scale_status": request["metric_scale_status"],
        "vertices": vertices,
        "faces": faces,
        "observed_region_ids": sorted(regions_with_faces),
        "unsupported_region_ids": unsupported,
        "generated_fill_used": False,
        "unseen_or_rejected_depth_filled": False,
    }
    output = _prepare_output(output_root, root)
    surface_path = output / "arkit_observed_surface.json"
    surface_digest = _write_immutable_json(surface_path, surface)
    relative_path = surface_path.resolve().relative_to(root.resolve()).as_posix()
    result = {
        "schema_version": RESULT_SCHEMA,
        "stable_run_identity": request["stable_run_identity"],
        "source_capture_identity": request["source_capture_identity"],
        "source_capture_digest": request["source_capture_digest"],
        "original_file_references": request["original_file_references"],
        "producing_method": METHOD_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "container_image_digest": request.get("container_image_digest"),
        "source_commit_sha": request["source_commit_sha"],
        "deterministic_configuration_digest": request[
            "deterministic_configuration_digest"
        ],
        "input_digests": input_digests,
        "output_digests": [{"artifact_id": "arkit_observed_surface", "digest": surface_digest}],
        "train_heldout_split_digest": request["train_heldout_split_digest"],
        "camera_calibration_binding": request["camera_calibration_binding"],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "units": "meters",
        "metric_scale_status": request["metric_scale_status"],
        "provider_runtime_identity": {"provider": "local", "runtime": "python_numpy"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": request["authority_used"],
        "warnings": list(request.get("warnings") or []),
        "blockers": [],
        "parent_artifact_or_event": {"digest": request_digest},
        "timestamp": request["timestamp"],
        "status": "compiled_observed_surface_candidate",
        "surface_asset": {"relative_path": relative_path, "digest": surface_digest},
        "accepted_high_confidence_pixel_count": accepted_pixel_count,
        "rejected_or_missing_pixel_count": rejected_pixel_count,
        "emitted_vertex_count": len(vertices),
        "emitted_triangle_count": len(faces),
        "discontinuity_rejected_triangle_count": discontinuity_rejected_triangle_count,
        "observed_region_ids": sorted(regions_with_faces),
        "unsupported_region_ids": unsupported,
        "hidden_heldout_observations_accessed": False,
        "generated_fill_used": False,
        "raw_arkit_poses_modified": False,
        "proof_effect": "metric_reference_candidate_only",
        "claim_ceiling": "observed_arkit_depth_surface_candidate",
    }
    result["arkit_depth_surface_compilation_result_digest"] = canonical_digest(
        result, digest_field="arkit_depth_surface_compilation_result_digest"
    )
    return result


__all__ = [
    "ArkitDepthSurfaceCompilerError",
    "CAMERA_CONVENTIONS",
    "IMPLEMENTATION_VERSION",
    "METHOD_ID",
    "OUTPUT_SURFACE_SCHEMA",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "compile_arkit_depth_surface",
]
