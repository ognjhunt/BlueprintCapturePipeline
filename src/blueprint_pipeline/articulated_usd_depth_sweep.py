"""Rasterize actual articulated USD mesh depth over frozen camera/door cells.

This is a construction-time geometric measurement.  It reads mesh vertices
from the bound USD, applies the frozen asset placement and one articulated-link
rotation, and emits deterministic pinhole depth.  It does not assert native
simulator import, contact, physical equivalence, or policy readiness.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json


DEPTH_SWEEP_SCHEMA = "adp009b_articulated_usd_depth_sweep.v1"


class ArticulatedUsdDepthSweepError(ValueError):
    """Stable fail-closed articulated depth errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _matrix(value: Any, code: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0)
    ):
        raise ArticulatedUsdDepthSweepError([code])
    return matrix


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts.tolist():
        if count < 3:
            raise ArticulatedUsdDepthSweepError(["articulated_depth_face_invalid"])
        face = indices[offset : offset + count]
        triangles.extend(
            (int(face[0]), int(face[i]), int(face[i + 1]))
            for i in range(1, count - 1)
        )
        offset += count
    if offset != len(indices) or not triangles:
        raise ArticulatedUsdDepthSweepError(["articulated_depth_face_indices_invalid"])
    return np.asarray(triangles, dtype=np.int64)


def load_articulated_usd_triangles(
    usd_path: str | Path, *, moving_link_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return static and moving triangles in the USD stage's asset frame."""

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_openusd_runtime_missing"]
        ) from exc
    path = Path(usd_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArticulatedUsdDepthSweepError(["articulated_depth_usd_missing"])
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedUsdDepthSweepError(["articulated_depth_usd_unreadable"])
    if float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0 or str(
        UsdGeom.GetStageUpAxis(stage)
    ).upper() != "Z":
        raise ArticulatedUsdDepthSweepError(["articulated_depth_usd_frame_invalid"])
    moving = stage.GetPrimAtPath(moving_link_path)
    if not moving.IsValid():
        raise ArticulatedUsdDepthSweepError(["articulated_depth_moving_link_missing"])
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    groups: dict[str, list[np.ndarray]] = {"static": [], "moving": []}
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh or not prim.IsActive() or not prim.IsLoaded():
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ArticulatedUsdDepthSweepError(["articulated_depth_mesh_points_invalid"])
        faces = _triangulate(counts, indices)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
        asset_points = (transform @ homogeneous.T).T[:, :3]
        triangles = asset_points[faces]
        under_moving = prim.GetPath().HasPrefix(moving.GetPath())
        groups["moving" if under_moving else "static"].append(triangles)
    if not groups["static"] or not groups["moving"]:
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_static_or_moving_geometry_missing"]
        )
    return np.concatenate(groups["static"]), np.concatenate(groups["moving"])


def rotate_triangles_about_axis(
    triangles: np.ndarray,
    *,
    pivot: Sequence[float],
    axis: Sequence[float],
    angle_deg: float,
) -> np.ndarray:
    """Rotate triangles using deterministic Rodrigues axis-angle geometry."""

    values = np.asarray(triangles, dtype=np.float64)
    origin = np.asarray(pivot, dtype=np.float64)
    direction = np.asarray(axis, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[1:] != (3, 3)
        or origin.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(values).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(direction).all()
        or not math.isfinite(float(angle_deg))
        or np.linalg.norm(direction) <= 1e-12
    ):
        raise ArticulatedUsdDepthSweepError(["articulated_depth_rotation_invalid"])
    unit = direction / np.linalg.norm(direction)
    radians = math.radians(float(angle_deg))
    skew = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3) * math.cos(radians)
        + (1.0 - math.cos(radians)) * np.outer(unit, unit)
        + math.sin(radians) * skew
    )
    rotated = (rotation @ (values - origin).reshape((-1, 3)).T).T
    return rotated.reshape(values.shape) + origin


def _transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = triangles.reshape((-1, 3))
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3].reshape(triangles.shape)


def rasterize_triangle_depth(
    triangles_world: np.ndarray,
    *,
    T_world_camera_opencv: Sequence[Sequence[float]],
    intrinsics: Mapping[str, Any],
    resolution_scale: float = 1.0,
    near_m: float = 1e-4,
) -> np.ndarray:
    """Rasterize two-sided triangle depth with perspective-correct interpolation."""

    triangles = np.asarray(triangles_world, dtype=np.float64)
    camera_to_world = _matrix(
        T_world_camera_opencv, "articulated_depth_camera_transform_invalid"
    )
    try:
        width = int(round(int(intrinsics["width"]) * float(resolution_scale)))
        height = int(round(int(intrinsics["height"]) * float(resolution_scale)))
        fx = float(intrinsics["fx"]) * float(resolution_scale)
        fy = float(intrinsics["fy"]) * float(resolution_scale)
        cx = float(intrinsics["cx"]) * float(resolution_scale)
        cy = float(intrinsics["cy"]) * float(resolution_scale)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_camera_intrinsics_invalid"]
        ) from exc
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or not np.isfinite(triangles).all()
        or width <= 0
        or height <= 0
        or min(fx, fy) <= 0.0
        or not 0.0 < resolution_scale <= 1.0
        or near_m <= 0.0
    ):
        raise ArticulatedUsdDepthSweepError(["articulated_depth_raster_input_invalid"])
    world_to_camera = np.linalg.inv(camera_to_world)
    points = triangles.reshape((-1, 3))
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera = (world_to_camera @ homogeneous.T).T[:, :3].reshape(triangles.shape)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    for tri in camera:
        z = tri[:, 2]
        if np.any(z <= near_m):
            continue
        x = fx * tri[:, 0] / z + cx
        y = fy * tri[:, 1] / z + cy
        x_min = max(0, int(math.floor(float(x.min()))))
        x_max = min(width - 1, int(math.ceil(float(x.max()))))
        y_min = max(0, int(math.floor(float(y.min()))))
        y_max = min(height - 1, int(math.ceil(float(y.max()))))
        if x_min > x_max or y_min > y_max:
            continue
        denominator = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (
            y[0] - y[2]
        )
        if abs(float(denominator)) <= 1e-12:
            continue
        px, py = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float64),
            np.arange(y_min, y_max + 1, dtype=np.float64),
        )
        w0 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / denominator
        w1 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not np.any(inside):
            continue
        inverse_depth = w0 / z[0] + w1 / z[1] + w2 / z[2]
        values = np.full_like(inverse_depth, np.inf, dtype=np.float64)
        np.divide(1.0, inverse_depth, out=values, where=inside & (inverse_depth > 0.0))
        patch = depth[y_min : y_max + 1, x_min : x_max + 1]
        np.minimum(patch, values.astype(np.float32), out=patch)
    return depth


def materialize_articulated_usd_depth_sweep(
    *,
    usd_path: str | Path,
    cameras: Sequence[Mapping[str, Any]],
    door_angles_deg: Sequence[float],
    moving_link_path: str,
    hinge_origin_asset_m: Sequence[float],
    hinge_axis_asset: Sequence[float],
    T_world_asset: Sequence[Sequence[float]],
    output_root: str | Path,
    resolution_scale: float = 0.25,
) -> dict[str, Any]:
    """Materialize actual USD depth over the camera by articulation matrix."""

    usd = Path(usd_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArticulatedUsdDepthSweepError(["articulated_depth_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    asset_to_world = _matrix(T_world_asset, "articulated_depth_asset_transform_invalid")
    if (
        not cameras
        or not door_angles_deg
        or len({str(row.get("camera_id") or "") for row in cameras}) != len(cameras)
        or any(not str(row.get("camera_id") or "") for row in cameras)
        or any(not math.isfinite(float(angle)) for angle in door_angles_deg)
    ):
        raise ArticulatedUsdDepthSweepError(["articulated_depth_sweep_cells_invalid"])
    static_asset, moving_asset = load_articulated_usd_triangles(
        usd, moving_link_path=moving_link_path
    )
    static_world = _transform_triangles(static_asset, asset_to_world)
    moving_world_by_angle = {
        float(angle): _transform_triangles(
            rotate_triangles_about_axis(
                moving_asset,
                pivot=hinge_origin_asset_m,
                axis=hinge_axis_asset,
                angle_deg=float(angle),
            ),
            asset_to_world,
        )
        for angle in door_angles_deg
    }
    depths = []
    cells = []
    for camera in cameras:
        camera_id = str(camera["camera_id"])
        static_depth = rasterize_triangle_depth(
            static_world,
            T_world_camera_opencv=camera["T_world_camera_opencv"],
            intrinsics=camera["intrinsics"],
            resolution_scale=resolution_scale,
        )
        for angle in door_angles_deg:
            moving_depth = rasterize_triangle_depth(
                moving_world_by_angle[float(angle)],
                T_world_camera_opencv=camera["T_world_camera_opencv"],
                intrinsics=camera["intrinsics"],
                resolution_scale=resolution_scale,
            )
            depths.append(np.minimum(static_depth, moving_depth))
            cells.append(
                {
                    "camera_id": camera_id,
                    "commanded_door_angle_deg": float(angle),
                    "readback_door_angle_deg": float(angle),
                }
            )
    depth_array = np.stack(depths).astype(np.float32)
    # NPY is deliberately used instead of NPZ: ZIP member timestamps make an
    # otherwise identical compressed archive byte-nondeterministic.
    arrays_path = output / "replacement_depth_sweep.npy"
    np.save(arrays_path, depth_array, allow_pickle=False)
    manifest: dict[str, Any] = {
        "schema_version": DEPTH_SWEEP_SCHEMA,
        "status": "actual_usd_mesh_depth_rasterized",
        "actual_mesh_depth_rasterized": True,
        "caller_supplied_coverage_mask": False,
        "replacement_usd": {"path": str(usd), "sha256": _sha256(usd)},
        "moving_link_path": moving_link_path,
        "hinge_origin_asset_m": [float(value) for value in hinge_origin_asset_m],
        "hinge_axis_asset": [float(value) for value in hinge_axis_asset],
        "T_world_asset": asset_to_world.tolist(),
        "cells": cells,
        "camera_count": len(cameras),
        "door_state_count": len(door_angles_deg),
        "resolution_scale": float(resolution_scale),
        "depth_dimensions": [int(depth_array.shape[2]), int(depth_array.shape[1])],
        "finite_depth_pixel_count_by_cell": [
            int(np.isfinite(depth).sum()) for depth in depth_array
        ],
        "arrays": _record(arrays_path, output),
        "renderer": {
            "name": "blueprint_numpy_perspective_correct_triangle_depth.v1",
            "pixel_sample": "integer_opencv_pixel_centers",
            "two_sided": True,
            "near_plane_clipping": "triangles_crossing_near_plane_rejected",
        },
        "native_simulator_readback": False,
        "physical_equivalence_proven": False,
        "claim_ceiling": "construction_candidate_actual_usd_geometry_depth_only",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    (output / f"{DEPTH_SWEEP_SCHEMA}.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "ArticulatedUsdDepthSweepError",
    "DEPTH_SWEEP_SCHEMA",
    "load_articulated_usd_triangles",
    "materialize_articulated_usd_depth_sweep",
    "rasterize_triangle_depth",
    "rotate_triangles_about_axis",
]
