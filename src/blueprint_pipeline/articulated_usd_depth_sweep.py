"""Rasterize actual articulated USD mesh depth over frozen camera/door cells.

This is a construction-time geometric measurement.  It reads mesh vertices
from the bound USD, applies the frozen asset placement and one articulated-link
rotation, and emits deterministic pinhole depth.  It does not assert native
simulator import, contact, physical equivalence, or policy readiness.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_gaussian_excision_heldout import (
    derive_alpha_from_background_pair,
)


DEPTH_SWEEP_SCHEMA = "adp009b_articulated_usd_depth_sweep.v1"
SOURCE_COVERAGE_AUDIT_SCHEMA = "adp009b_source_layer_replacement_coverage_audit.v1"
REFERENCE_HYBRID_REVIEW_SCHEMA = "adp009b_reference_hybrid_review.v1"
TARGET_CORE_COVERAGE_AUDIT_SCHEMA = "articulated_excision_coverage.v1"


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


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatedUsdDepthSweepError([code]) from exc
    if not isinstance(value, dict):
        raise ArticulatedUsdDepthSweepError([code])
    return value


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


def _verified_render_rows(
    manifest_path: Path, *, expected_background: str
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = _read_object(
        manifest_path, "source_coverage_render_manifest_unreadable"
    )
    if (
        manifest.get("schema_version") != "sealed_camera_render_manifest.v1"
        or manifest.get("status") != "rendered_exact_cameras"
        or manifest.get("sealed_camera_render_manifest_digest")
        != canonical_digest(
            manifest, digest_field="sealed_camera_render_manifest_digest"
        )
        or (manifest.get("renderer_identity") or {}).get("background_rgb")
        != expected_background
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_render_manifest_invalid"]
        )
    rows = manifest.get("renders")
    if not isinstance(rows, list) or len(rows) != manifest.get("render_count"):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_render_manifest_invalid"]
        )
    by_camera: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_render_manifest_invalid"]
            )
        camera_id = str(row.get("camera_id") or "")
        relative = str(row.get("relative_path") or "")
        frame = (manifest_path.parent / relative).resolve()
        if (
            not camera_id
            or camera_id in by_camera
            or not frame.is_file()
            or frame.is_symlink()
            or _sha256(frame) != row.get("digest")
        ):
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_render_frame_changed"]
            )
        by_camera[camera_id] = row
    return manifest, by_camera


def conservative_max_pool_alpha(
    alpha: np.ndarray, *, output_height: int, output_width: int
) -> np.ndarray:
    """Downsample alpha without losing thin source-object contributions."""

    values = np.asarray(alpha, dtype=np.float32)
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or output_height <= 0
        or output_width <= 0
        or values.shape[0] % output_height
        or values.shape[1] % output_width
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_alpha_pool_invalid"]
        )
    factor_y = values.shape[0] // output_height
    factor_x = values.shape[1] // output_width
    return values.reshape(
        output_height, factor_y, output_width, factor_x
    ).max(axis=(1, 3))


def evaluate_source_alpha_coverage(
    source_alpha_by_camera: np.ndarray,
    replacement_depth_m: np.ndarray,
    *,
    cells: Sequence[Mapping[str, Any]],
    camera_ids: Sequence[str],
    significant_alpha_threshold: float = 1.0 / 255.0,
    coverage_margin_pixels: int = 1,
) -> list[dict[str, Any]]:
    """Measure visible source residue after conservative USD silhouette erosion."""

    alpha = np.asarray(source_alpha_by_camera, dtype=np.float32)
    depth = np.asarray(replacement_depth_m, dtype=np.float32)
    if (
        alpha.ndim != 3
        or depth.ndim != 3
        or depth.shape[1:] != alpha.shape[1:]
        or len(cells) != depth.shape[0]
        or len(camera_ids) != alpha.shape[0]
        or len(set(camera_ids)) != len(camera_ids)
        or not np.isfinite(alpha).all()
        or np.any(alpha < 0.0)
        or np.any(alpha > 1.0)
        or not 0.0 < significant_alpha_threshold <= 1.0
        or not isinstance(coverage_margin_pixels, int)
        or coverage_margin_pixels < 0
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_evaluation_input_invalid"]
        )
    camera_lookup = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    kernel_size = 2 * coverage_margin_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        camera_id = str(cell.get("camera_id") or "")
        if camera_id not in camera_lookup:
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_cell_camera_missing"]
            )
        source = alpha[camera_lookup[camera_id]]
        finite = np.isfinite(depth[cell_index]) & (depth[cell_index] > 0.0)
        covered = cv2.erode(
            finite.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        significant = source >= significant_alpha_threshold
        uncovered = ~covered
        alpha_sum = float(source.sum())
        residual_sum = float(source[uncovered].sum())
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (significant & uncovered).astype(np.uint8), 8
        )
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        rows.append(
            {
                "cell_index": cell_index,
                "camera_id": camera_id,
                "commanded_door_angle_deg": float(
                    cell["commanded_door_angle_deg"]
                ),
                "readback_door_angle_deg": float(cell["readback_door_angle_deg"]),
                "source_significant_pixel_count": int(significant.sum()),
                "uncovered_significant_pixel_count": int(
                    (significant & uncovered).sum()
                ),
                "largest_uncovered_component_pixels": largest,
                "source_alpha_sum": alpha_sum,
                "uncovered_alpha_sum": residual_sum,
                "uncovered_alpha_fraction": (
                    residual_sum / alpha_sum if alpha_sum > 0.0 else 0.0
                ),
                "replacement_covered_pixel_count_after_margin": int(covered.sum()),
            }
        )
    return rows


def materialize_source_layer_replacement_coverage_audit(
    *,
    black_render_manifest_path: str | Path,
    white_render_manifest_path: str | Path,
    depth_sweep_manifest_path: str | Path,
    output_root: str | Path,
    significant_alpha_threshold: float = 1.0 / 255.0,
    coverage_margin_pixels: int = 1,
) -> dict[str, Any]:
    """Audit a rendered source-object layer against actual USD sweep coverage."""

    black_path = Path(black_render_manifest_path).expanduser().resolve()
    white_path = Path(white_render_manifest_path).expanduser().resolve()
    depth_path = Path(depth_sweep_manifest_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArticulatedUsdDepthSweepError(["source_coverage_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    black_manifest, black_rows = _verified_render_rows(
        black_path, expected_background="#000000"
    )
    white_manifest, white_rows = _verified_render_rows(
        white_path, expected_background="#ffffff"
    )
    if (
        set(black_rows) != set(white_rows)
        or black_manifest.get("splat_digest") != white_manifest.get("splat_digest")
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_background_pair_mismatch"]
        )
    depth_manifest = _read_object(
        depth_path, "source_coverage_depth_manifest_unreadable"
    )
    if (
        depth_manifest.get("schema_version") != DEPTH_SWEEP_SCHEMA
        or depth_manifest.get("manifest_digest")
        != canonical_digest(depth_manifest, digest_field="manifest_digest")
        or depth_manifest.get("actual_mesh_depth_rasterized") is not True
        or depth_manifest.get("caller_supplied_coverage_mask") is not False
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_depth_manifest_invalid"]
        )
    arrays_record = depth_manifest.get("arrays") or {}
    depth_array_path = depth_path.parent / str(arrays_record.get("relative_path") or "")
    if (
        not depth_array_path.is_file()
        or depth_array_path.is_symlink()
        or depth_array_path.stat().st_size != arrays_record.get("size_bytes")
        or _sha256(depth_array_path) != arrays_record.get("sha256")
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_depth_array_changed"]
        )
    depth = np.load(depth_array_path, allow_pickle=False)
    cells = depth_manifest.get("cells")
    if not isinstance(cells, list) or depth.shape[0] != len(cells):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_depth_cells_invalid"]
        )
    camera_ids = list(black_rows)
    output_height, output_width = depth.shape[1:]
    source_alpha = []
    black_by_camera: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        black_frame = black_path.parent / str(black_rows[camera_id]["relative_path"])
        white_frame = white_path.parent / str(white_rows[camera_id]["relative_path"])
        black = cv2.imread(str(black_frame), cv2.IMREAD_COLOR)
        white = cv2.imread(str(white_frame), cv2.IMREAD_COLOR)
        if black is None or white is None:
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_render_frame_unreadable"]
            )
        black_by_camera[camera_id] = cv2.resize(
            black, (output_width, output_height), interpolation=cv2.INTER_AREA
        )
        source_alpha.append(
            conservative_max_pool_alpha(
                derive_alpha_from_background_pair(black, white),
                output_height=output_height,
                output_width=output_width,
            )
        )
    alpha_array = np.stack(source_alpha).astype(np.float32)
    rows = evaluate_source_alpha_coverage(
        alpha_array,
        depth,
        cells=cells,
        camera_ids=camera_ids,
        significant_alpha_threshold=significant_alpha_threshold,
        coverage_margin_pixels=coverage_margin_pixels,
    )
    alpha_path = output / "source_alpha_by_camera.npy"
    np.save(alpha_path, alpha_array, allow_pickle=False)
    review_root = output / "review_contact_sheets"
    review_root.mkdir()
    seam_root = output / "uncovered_source_support_masks"
    seam_root.mkdir()
    review_records = []
    seam_records = []
    kernel_size = 2 * coverage_margin_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for camera_index, camera_id in enumerate(camera_ids):
        cell_indices = [
            index
            for index, cell in enumerate(cells)
            if str(cell.get("camera_id") or "") == camera_id
        ]
        if not cell_indices:
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_cell_camera_missing"]
            )
        selected = sorted(
            {
                cell_indices[0],
                cell_indices[len(cell_indices) // 2],
                cell_indices[-1],
            }
        )
        source = alpha_array[camera_index]
        uncovered_union = np.zeros(source.shape, dtype=bool)
        for cell_index in cell_indices:
            finite = np.isfinite(depth[cell_index]) & (depth[cell_index] > 0.0)
            covered = cv2.erode(
                finite.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            uncovered_union |= (source >= significant_alpha_threshold) & ~covered
        seam_path = seam_root / f"{camera_id}.png"
        if not cv2.imwrite(str(seam_path), uncovered_union.astype(np.uint8) * 255):
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_seam_mask_write_failed"]
            )
        seam_records.append(
            {
                **_record(seam_path, output),
                "camera_id": camera_id,
                "pixel_count": int(uncovered_union.sum()),
                "derived_from_all_door_cells": len(cell_indices),
            }
        )
        base = black_by_camera[camera_id].astype(np.float32)
        base += (1.0 - source[..., None]) * 230.0
        panels = []
        for cell_index in selected:
            finite = np.isfinite(depth[cell_index]) & (depth[cell_index] > 0.0)
            covered = cv2.erode(
                finite.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            uncovered = (source >= significant_alpha_threshold) & ~covered
            panel = np.clip(base, 0.0, 255.0).astype(np.uint8)
            panel[uncovered] = (
                0.25 * panel[uncovered] + 0.75 * np.array([0, 0, 255])
            ).astype(np.uint8)
            angle = float(cells[cell_index]["commanded_door_angle_deg"])
            cv2.putText(
                panel,
                f"{camera_id}  door={angle:g}deg  red=uncovered",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{camera_id}  door={angle:g}deg  red=uncovered",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        sheet = np.concatenate(panels, axis=1)
        sheet_path = review_root / f"{camera_id}.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_contact_sheet_write_failed"]
            )
        review_records.append(_record(sheet_path, output))
    manifest: dict[str, Any] = {
        "schema_version": SOURCE_COVERAGE_AUDIT_SCHEMA,
        "status": "source_layer_coverage_measured",
        "source_layer_splat_digest": black_manifest.get("splat_digest"),
        "black_render_manifest": {
            "sha256": _sha256(black_path),
            "sealed_camera_render_manifest_digest": black_manifest.get(
                "sealed_camera_render_manifest_digest"
            ),
        },
        "white_render_manifest": {
            "sha256": _sha256(white_path),
            "sealed_camera_render_manifest_digest": white_manifest.get(
                "sealed_camera_render_manifest_digest"
            ),
        },
        "depth_sweep_manifest": {
            "sha256": _sha256(depth_path),
            "manifest_digest": depth_manifest["manifest_digest"],
        },
        "camera_ids": camera_ids,
        "significant_alpha_threshold": float(significant_alpha_threshold),
        "coverage_margin_pixels": coverage_margin_pixels,
        "source_alpha": _record(alpha_path, output),
        "review_contact_sheets": review_records,
        "uncovered_source_support_masks": seam_records,
        "uncovered_source_support_masks_are_inpainting_authority": False,
        "cells": rows,
        "summary": {
            "cell_count": len(rows),
            "worst_uncovered_significant_pixel_count": max(
                row["uncovered_significant_pixel_count"] for row in rows
            ),
            "worst_largest_uncovered_component_pixels": max(
                row["largest_uncovered_component_pixels"] for row in rows
            ),
            "worst_uncovered_alpha_fraction": max(
                row["uncovered_alpha_fraction"] for row in rows
            ),
        },
        "coverage_qualified": False,
        "claim_ceiling": "measured_source_layer_visibility_against_actual_usd_depth",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = output / f"{SOURCE_COVERAGE_AUDIT_SCHEMA}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def materialize_target_core_replacement_coverage_audit(
    *,
    target_core_mask_paths: Mapping[str, str | Path],
    depth_sweep_manifest_path: str | Path,
    output_root: str | Path,
    maximum_uncovered_fraction: float,
    coverage_margin_pixels: int = 0,
) -> dict[str, Any]:
    """Measure which frozen target-mask pixels the posed replacement does not hide.

    This is the coverage-conditioned counterpart to broad object-layer removal.
    It never guesses Gaussian ownership: it joins exact calibrated target masks
    to actual replacement-USD depth for every camera and articulated state.  A
    residual remains a narrow seam candidate, never implicit permission to
    delete more scene content or a claim that inpainting is unnecessary.
    """

    depth_path = Path(depth_sweep_manifest_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArticulatedUsdDepthSweepError(["target_core_coverage_output_not_empty"])
    if (
        isinstance(maximum_uncovered_fraction, bool)
        or not isinstance(maximum_uncovered_fraction, (int, float))
        or not math.isfinite(float(maximum_uncovered_fraction))
        or not 0.0 <= float(maximum_uncovered_fraction) < 1.0
    ):
        raise ArticulatedUsdDepthSweepError(
            ["target_core_coverage_fraction_threshold_invalid"]
        )
    if (
        isinstance(coverage_margin_pixels, bool)
        or not isinstance(coverage_margin_pixels, int)
        or coverage_margin_pixels < 0
    ):
        raise ArticulatedUsdDepthSweepError(
            ["target_core_coverage_margin_invalid"]
        )

    depth_manifest = _read_object(
        depth_path, "target_core_coverage_depth_manifest_unreadable"
    )
    if (
        depth_manifest.get("schema_version") != DEPTH_SWEEP_SCHEMA
        or depth_manifest.get("manifest_digest")
        != canonical_digest(depth_manifest, digest_field="manifest_digest")
        or depth_manifest.get("actual_mesh_depth_rasterized") is not True
        or depth_manifest.get("caller_supplied_coverage_mask") is not False
    ):
        raise ArticulatedUsdDepthSweepError(
            ["target_core_coverage_depth_manifest_invalid"]
        )
    depth_record = depth_manifest.get("arrays") or {}
    depth_array_path = depth_path.parent / str(depth_record.get("relative_path") or "")
    if (
        not depth_array_path.is_file()
        or depth_array_path.is_symlink()
        or depth_array_path.stat().st_size != depth_record.get("size_bytes")
        or _sha256(depth_array_path) != depth_record.get("sha256")
    ):
        raise ArticulatedUsdDepthSweepError(
            ["target_core_coverage_depth_array_changed"]
        )
    depth = np.load(depth_array_path, allow_pickle=False)
    cells = depth_manifest.get("cells")
    if depth.ndim != 3 or not isinstance(cells, list) or len(cells) != depth.shape[0]:
        raise ArticulatedUsdDepthSweepError(
            ["target_core_coverage_depth_cells_invalid"]
        )
    camera_ids = list(
        dict.fromkeys(str(cell.get("camera_id") or "") for cell in cells)
    )
    if (
        not camera_ids
        or any(not camera_id for camera_id in camera_ids)
        or set(target_core_mask_paths) != set(camera_ids)
    ):
        raise ArticulatedUsdDepthSweepError(
            ["target_core_coverage_camera_masks_mismatch"]
        )

    height, width = depth.shape[1:]
    masks: dict[str, np.ndarray] = {}
    mask_records: list[dict[str, Any]] = []
    for camera_id in camera_ids:
        mask_path = Path(target_core_mask_paths[camera_id]).expanduser().resolve()
        if not mask_path.is_file() or mask_path.is_symlink():
            raise ArticulatedUsdDepthSweepError(
                [f"target_core_coverage_mask_missing:{camera_id}"]
            )
        source = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None or not np.any(source > 0):
            raise ArticulatedUsdDepthSweepError(
                [f"target_core_coverage_mask_invalid:{camera_id}"]
            )
        # Any nonzero area contribution survives downsampling.  This is more
        # conservative than nearest-neighbour sampling at mask boundaries.
        mask = cv2.resize(
            source, (width, height), interpolation=cv2.INTER_AREA
        ) > 0
        masks[camera_id] = mask
        mask_records.append(
            {
                "camera_id": camera_id,
                "source_path": str(mask_path),
                "sha256": _sha256(mask_path),
                "source_dimensions": [int(source.shape[1]), int(source.shape[0])],
                "audit_dimensions": [width, height],
                "audit_pixel_count": int(mask.sum()),
            }
        )

    minimum_target_pixels = min(int(mask.sum()) for mask in masks.values())
    maximum_component_pixels = int(
        math.floor(float(maximum_uncovered_fraction) * minimum_target_pixels)
    )
    kernel = np.ones(
        (2 * coverage_margin_pixels + 1, 2 * coverage_margin_pixels + 1),
        dtype=np.uint8,
    )
    rows: list[dict[str, Any]] = []
    residual_by_camera = {
        camera_id: np.zeros((height, width), dtype=bool)
        for camera_id in camera_ids
    }
    for index, cell in enumerate(cells):
        camera_id = str(cell.get("camera_id") or "")
        try:
            commanded = float(cell["commanded_door_angle_deg"])
            readback = float(cell["readback_door_angle_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArticulatedUsdDepthSweepError(
                ["target_core_coverage_depth_cells_invalid"]
            ) from exc
        covered = np.isfinite(depth[index]) & (depth[index] > 0.0)
        if coverage_margin_pixels:
            covered = cv2.dilate(covered.astype(np.uint8), kernel).astype(bool)
        target = masks[camera_id]
        residual = target & ~covered
        residual_by_camera[camera_id] |= residual
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            residual.astype(np.uint8), 8
        )
        largest = (
            int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
        )
        residual_count = int(residual.sum())
        fraction = residual_count / int(target.sum())
        rows.append(
            {
                "cell_index": index,
                "camera_id": camera_id,
                "door_state_angle_degrees": commanded,
                "readback_door_state_angle_degrees": readback,
                "target_core_pixel_count": int(target.sum()),
                "replacement_covered_target_core_pixel_count": int(
                    (target & covered).sum()
                ),
                "residual_significant_pixels": residual_count,
                "residual_fraction_of_target_core": fraction,
                "residual_max_connected_component_pixels": largest,
                "residual_inside_target_core_mask": True,
                "outside_mask_changed_pixels": 0,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    seam_root = output / "residual_target_core_seam_masks"
    seam_root.mkdir()
    seam_records: list[dict[str, Any]] = []
    for camera_id in camera_ids:
        seam_path = seam_root / f"{camera_id}.png"
        if not cv2.imwrite(
            str(seam_path), residual_by_camera[camera_id].astype(np.uint8) * 255
        ):
            raise ArticulatedUsdDepthSweepError(
                ["target_core_coverage_seam_mask_write_failed"]
            )
        seam_records.append(
            {
                **_record(seam_path, output),
                "camera_id": camera_id,
                "pixel_count": int(residual_by_camera[camera_id].sum()),
                "derived_from_all_door_cells": sum(
                    str(cell.get("camera_id") or "") == camera_id for cell in cells
                ),
            }
        )

    worst_fraction = max(row["residual_fraction_of_target_core"] for row in rows)
    worst_component = max(
        row["residual_max_connected_component_pixels"] for row in rows
    )
    coverage_qualified = bool(
        worst_fraction <= float(maximum_uncovered_fraction)
        and worst_component <= maximum_component_pixels
    )
    door_states = list(
        dict.fromkeys(float(cell["commanded_door_angle_deg"]) for cell in cells)
    )
    manifest: dict[str, Any] = {
        "schema_version": TARGET_CORE_COVERAGE_AUDIT_SCHEMA,
        "status": "target_core_replacement_coverage_measured",
        "coverage_qualified": coverage_qualified,
        "depth_sweep_manifest": {
            "sha256": _sha256(depth_path),
            "manifest_digest": depth_manifest["manifest_digest"],
            "replacement_usd": depth_manifest.get("replacement_usd"),
        },
        "camera_ids": camera_ids,
        "door_state_angles_degrees": door_states,
        "target_core_masks": mask_records,
        "coverage_margin_pixels": coverage_margin_pixels,
        "maximum_uncovered_fraction": float(maximum_uncovered_fraction),
        "maximum_residual_connected_component_pixels": maximum_component_pixels,
        "maximum_protected_changed_pixels": 0,
        "cells": rows,
        "residual_target_core_seam_masks": seam_records,
        "summary": {
            "cell_count": len(rows),
            "worst_uncovered_target_core_fraction": worst_fraction,
            "worst_uncovered_target_core_pixel_count": max(
                row["residual_significant_pixels"] for row in rows
            ),
            "worst_residual_connected_component_pixels": worst_component,
        },
        "caller_asserted_coverage_accepted": False,
        "rendered_pixels_changed_by_audit": False,
        "residual_is_narrow_seam_candidate_not_inpainting_success": True,
        "claim_ceiling": "geometric_replacement_coverage_candidate_only",
        "receipt_digest": "",
    }
    manifest["receipt_digest"] = canonical_digest(
        manifest, digest_field="receipt_digest"
    )
    manifest_path = output / f"{TARGET_CORE_COVERAGE_AUDIT_SCHEMA}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def materialize_reference_hybrid_review(
    *,
    retained_scene_render_manifest_path: str | Path,
    depth_sweep_manifest_path: str | Path,
    output_root: str | Path,
    replacement_rgb: Sequence[int] = (184, 188, 194),
) -> dict[str, Any]:
    """Composite the actual USD silhouette over retained 3DGS review frames.

    The depth sweep owns the replacement geometry and articulation.  This
    helper deliberately uses a neutral, synthetic color instead of pretending
    to render USD materials.  Pixels outside the finite replacement depth mask
    are copied exactly from the downsampled retained-scene render.  The result
    is therefore useful for detecting coverage holes and source-object ghosts,
    but it is never a native Isaac/RTX or evaluation-authorized render.
    """

    scene_manifest_path = Path(
        retained_scene_render_manifest_path
    ).expanduser().resolve()
    depth_manifest_path = Path(depth_sweep_manifest_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArticulatedUsdDepthSweepError(["reference_hybrid_output_not_empty"])
    try:
        color = tuple(int(value) for value in replacement_rgb)
    except (TypeError, ValueError) as exc:
        raise ArticulatedUsdDepthSweepError(
            ["reference_hybrid_replacement_color_invalid"]
        ) from exc
    if len(color) != 3 or any(value < 0 or value > 255 for value in color):
        raise ArticulatedUsdDepthSweepError(
            ["reference_hybrid_replacement_color_invalid"]
        )

    scene_manifest, scene_rows = _verified_render_rows(
        scene_manifest_path,
        expected_background=str(
            (_read_object(
                scene_manifest_path,
                "reference_hybrid_scene_manifest_unreadable",
            ).get("renderer_identity") or {}).get("background_rgb")
        ),
    )
    depth_manifest = _read_object(
        depth_manifest_path, "reference_hybrid_depth_manifest_unreadable"
    )
    if (
        depth_manifest.get("schema_version") != DEPTH_SWEEP_SCHEMA
        or depth_manifest.get("manifest_digest")
        != canonical_digest(depth_manifest, digest_field="manifest_digest")
        or depth_manifest.get("actual_mesh_depth_rasterized") is not True
        or depth_manifest.get("caller_supplied_coverage_mask") is not False
    ):
        raise ArticulatedUsdDepthSweepError(
            ["reference_hybrid_depth_manifest_invalid"]
        )
    depth_record = depth_manifest.get("arrays") or {}
    depth_array_path = depth_manifest_path.parent / str(
        depth_record.get("relative_path") or ""
    )
    if (
        not depth_array_path.is_file()
        or depth_array_path.is_symlink()
        or depth_array_path.stat().st_size != depth_record.get("size_bytes")
        or _sha256(depth_array_path) != depth_record.get("sha256")
    ):
        raise ArticulatedUsdDepthSweepError(
            ["reference_hybrid_depth_array_changed"]
        )
    depth = np.load(depth_array_path, allow_pickle=False)
    cells = depth_manifest.get("cells")
    if (
        depth.ndim != 3
        or not isinstance(cells, list)
        or len(cells) != depth.shape[0]
    ):
        raise ArticulatedUsdDepthSweepError(
            ["reference_hybrid_depth_cells_invalid"]
        )
    camera_ids = list(dict.fromkeys(str(cell.get("camera_id") or "") for cell in cells))
    if not camera_ids or set(camera_ids) != set(scene_rows):
        raise ArticulatedUsdDepthSweepError(
            ["reference_hybrid_camera_join_invalid"]
        )

    output.mkdir(parents=True, exist_ok=True)
    frames_root = output / "frames"
    frames_root.mkdir()
    height, width = depth.shape[1:]
    retained_by_camera: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        row = scene_rows[camera_id]
        frame_path = scene_manifest_path.parent / str(row["relative_path"])
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_scene_frame_unreadable"]
            )
        retained_by_camera[camera_id] = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_AREA
        )

    frame_records: list[dict[str, Any]] = []
    contact_panels: dict[str, list[tuple[float, Path]]] = {
        camera_id: [] for camera_id in camera_ids
    }
    # OpenCV stores BGR.  The public contract remains RGB.
    base_bgr = np.asarray(color[::-1], dtype=np.float32)
    for index, cell in enumerate(cells):
        camera_id = str(cell.get("camera_id") or "")
        try:
            angle = float(cell["commanded_door_angle_deg"])
            readback = float(cell["readback_door_angle_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_depth_cells_invalid"]
            ) from exc
        if not math.isfinite(angle) or not math.isfinite(readback):
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_depth_cells_invalid"]
            )
        finite = np.isfinite(depth[index]) & (depth[index] > 0.0)
        frame = retained_by_camera[camera_id].copy()
        if np.any(finite):
            values = depth[index][finite].astype(np.float64)
            low, high = np.percentile(values, [2.0, 98.0])
            span = max(float(high - low), 1e-6)
            shade = np.clip(1.08 - 0.22 * (depth[index] - low) / span, 0.78, 1.08)
            shaded = np.clip(base_bgr[None, None, :] * shade[..., None], 0, 255)
            frame[finite] = shaded[finite].astype(np.uint8)
            boundary = cv2.morphologyEx(
                finite.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            ).astype(bool) & finite
            frame[boundary] = np.asarray([45, 210, 255], dtype=np.uint8)
        angle_token = f"{angle:07.3f}".replace("-", "m").replace(".", "p")
        frame_path = frames_root / f"{camera_id}__door_{angle_token}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_frame_write_failed"]
            )
        frame_records.append(
            {
                **_record(frame_path, output),
                "camera_id": camera_id,
                "commanded_door_angle_deg": angle,
                "readback_door_angle_deg": readback,
                "replacement_covered_pixel_count": int(finite.sum()),
            }
        )
        contact_panels[camera_id].append((angle, frame_path))

    sheets_root = output / "contact_sheets"
    sheets_root.mkdir()
    sheet_records: list[dict[str, Any]] = []
    for camera_id, candidates in contact_panels.items():
        selected_indices = sorted({0, len(candidates) // 2, len(candidates) - 1})
        panels = []
        for selected_index in selected_indices:
            angle, frame_path = candidates[selected_index]
            panel = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if panel is None:
                raise ArticulatedUsdDepthSweepError(
                    ["reference_hybrid_frame_unreadable"]
                )
            cv2.putText(
                panel,
                f"{camera_id}  door={angle:g}deg  neutral=USD silhouette",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (10, 10, 10),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{camera_id}  door={angle:g}deg  neutral=USD silhouette",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        sheet = np.concatenate(panels, axis=1)
        sheet_path = sheets_root / f"{camera_id}.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_contact_sheet_write_failed"]
            )
        sheet_records.append(_record(sheet_path, output))

    manifest: dict[str, Any] = {
        "schema_version": REFERENCE_HYBRID_REVIEW_SCHEMA,
        "status": "reference_hybrid_review_materialized",
        "retained_scene_render": {
            "sha256": _sha256(scene_manifest_path),
            "sealed_camera_render_manifest_digest": scene_manifest.get(
                "sealed_camera_render_manifest_digest"
            ),
            "splat_digest": scene_manifest.get("splat_digest"),
        },
        "depth_sweep": {
            "sha256": _sha256(depth_manifest_path),
            "manifest_digest": depth_manifest.get("manifest_digest"),
            "replacement_usd": depth_manifest.get("replacement_usd"),
        },
        "replacement_rgb": list(color),
        "dimensions": [width, height],
        "camera_ids": camera_ids,
        "cell_count": len(frame_records),
        "frames": frame_records,
        "contact_sheets": sheet_records,
        "pixels_outside_replacement_mask_copied_from_retained_scene": True,
        "actual_usd_geometry_silhouette_used": True,
        "usd_materials_rendered": False,
        "native_isaac_or_rtx_render": False,
        "evaluation_authorized_render": False,
        "claim_ceiling": "review_only_actual_usd_silhouette_over_reference_3dgs",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = output / f"{REFERENCE_HYBRID_REVIEW_SCHEMA}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "ArticulatedUsdDepthSweepError",
    "DEPTH_SWEEP_SCHEMA",
    "REFERENCE_HYBRID_REVIEW_SCHEMA",
    "SOURCE_COVERAGE_AUDIT_SCHEMA",
    "TARGET_CORE_COVERAGE_AUDIT_SCHEMA",
    "conservative_max_pool_alpha",
    "evaluate_source_alpha_coverage",
    "load_articulated_usd_triangles",
    "materialize_articulated_usd_depth_sweep",
    "materialize_reference_hybrid_review",
    "materialize_source_layer_replacement_coverage_audit",
    "materialize_target_core_replacement_coverage_audit",
    "rasterize_triangle_depth",
    "rotate_triangles_about_axis",
]
