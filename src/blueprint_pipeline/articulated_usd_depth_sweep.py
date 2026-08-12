"""Rasterize actual articulated USD mesh depth over frozen camera/door cells.

This is a construction-time geometric measurement.  It reads mesh vertices
from the bound USD, applies the frozen asset placement and one articulated-link
rotation, and emits deterministic pinhole depth.  It does not assert native
simulator import, contact, physical equivalence, or policy readiness.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .articulation_graph_contract import validate_articulation_graph
from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_gaussian_excision_heldout import (
    derive_alpha_from_background_pair,
)


DEPTH_SWEEP_SCHEMA = "adp009b_articulated_usd_depth_sweep.v1"
GENERAL_DEPTH_SWEEP_REQUEST_SCHEMA = "replacement_usd_depth_sweep_request.v2"
GENERAL_DEPTH_SWEEP_SCHEMA = "replacement_usd_depth_sweep.v2"
COMPOSED_DEPTH_SWEEP_SCHEMA = "public_scene_replacement_depth_composition.v1"
SCENE_STATE_ROLES = frozenset({"task_subject", "co_present_passive"})
SOURCE_COVERAGE_AUDIT_SCHEMA = "adp009b_source_layer_replacement_coverage_audit.v1"
REFERENCE_HYBRID_REVIEW_SCHEMA = "adp009b_reference_hybrid_review.v1"
TARGET_CORE_COVERAGE_AUDIT_SCHEMA = "articulated_excision_coverage.v1"
DELETED_SOURCE_LAYER_COVERAGE_STATUS = (
    "deleted_source_layer_replacement_coverage_qualified"
)
_EXACT_SOURCE_ALPHA_THRESHOLD = 1.0 / 255.0


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


def _is_sha256_digest(value: Any) -> bool:
    text = str(value or "")
    return bool(
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


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


def _qualified_depth_manifest(value: Mapping[str, Any]) -> bool:
    schema = value.get("schema_version")
    digest_field = "receipt_digest" if schema == COMPOSED_DEPTH_SWEEP_SCHEMA else "manifest_digest"
    return bool(
        schema in {
            DEPTH_SWEEP_SCHEMA,
            GENERAL_DEPTH_SWEEP_SCHEMA,
            COMPOSED_DEPTH_SWEEP_SCHEMA,
        }
        and value.get(digest_field)
        == canonical_digest(dict(value), digest_field=digest_field)
        and value.get("caller_supplied_coverage_mask") is False
        and (
            (
                schema == DEPTH_SWEEP_SCHEMA
                and value.get("actual_mesh_depth_rasterized") is True
            )
            or (
                schema == GENERAL_DEPTH_SWEEP_SCHEMA
                and value.get("actual_usd_geometry_depth_rasterized") is True
            )
            or (
                schema == COMPOSED_DEPTH_SWEEP_SCHEMA
                and value.get("actual_usd_geometry_depth_rasterized") is True
                and value.get("actual_composed_depth_rasterized") is True
            )
        )
    )


def _depth_manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-correct digest binding for a depth source."""

    if value.get("schema_version") == COMPOSED_DEPTH_SWEEP_SCHEMA:
        return {"receipt_digest": value["receipt_digest"]}
    return {"manifest_digest": value["manifest_digest"]}


def _cell_state_fields(cell: Mapping[str, Any]) -> dict[str, Any]:
    if "cell_id" in cell:
        return {
            "state_cell_id": str(cell["cell_id"]),
            "joint_positions": json.loads(json.dumps(cell.get("joint_positions") or {})),
            "T_world_asset": json.loads(json.dumps(cell.get("T_world_asset"))),
            "T_world_task_scoring": json.loads(
                json.dumps(cell.get("T_world_task_scoring"))
            ),
            "task_scoring_frame_id": str(cell.get("task_scoring_frame_id") or ""),
        }
    return {
        "commanded_door_angle_deg": float(cell["commanded_door_angle_deg"]),
        "readback_door_angle_deg": float(cell["readback_door_angle_deg"]),
    }


def _cell_label(cell: Mapping[str, Any]) -> str:
    if "cell_id" in cell:
        return f"state={cell['cell_id']}"
    return f"door={float(cell['commanded_door_angle_deg']):g}deg"


def _matrix(value: Any, code: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0)
    ):
        raise ArticulatedUsdDepthSweepError([code])
    return matrix


def _rigid_matrix(value: Any, code: str) -> np.ndarray:
    matrix = _matrix(value, code)
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8, rel_tol=0.0
    ):
        raise ArticulatedUsdDepthSweepError([code])
    return matrix


def _triangulate(
    counts: np.ndarray, indices: np.ndarray, *, point_count: int
) -> np.ndarray:
    if (
        counts.ndim != 1
        or indices.ndim != 1
        or not len(counts)
        or np.any(counts != 3)
    ):
        # Fan triangulation can cover pixels outside a concave polygon.  Mesh
        # topology must already be explicit triangles before it can support a
        # deletion-safety claim.
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_mesh_topology_not_explicit_triangles"]
        )
    if int(np.sum(counts)) != len(indices) or len(indices) % 3 != 0:
        raise ArticulatedUsdDepthSweepError(["articulated_depth_face_indices_invalid"])
    triangles = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    if (
        np.any(triangles < 0)
        or np.any(triangles >= int(point_count))
        or any(len(set(int(value) for value in row)) != 3 for row in triangles)
    ):
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_face_indices_invalid"]
        )
    return triangles


def _primitive_points_and_faces(prim: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic triangles in a supported Gprim's local frame."""

    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ArticulatedUsdDepthSweepError(
                ["articulated_depth_mesh_points_invalid"]
            )
        return points, _triangulate(counts, indices, point_count=len(points))
    if prim.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        half = size / 2.0
        points = np.asarray(
            list(itertools.product((-half, half), repeat=3)), dtype=np.float64
        )
        # The point ordering is binary xyz: 000, 001, 010, ...
        faces = np.asarray(
            [
                (0, 1, 3), (0, 3, 2),
                (4, 6, 7), (4, 7, 5),
                (0, 4, 5), (0, 5, 1),
                (2, 3, 7), (2, 7, 6),
                (0, 2, 6), (0, 6, 4),
                (1, 5, 7), (1, 7, 3),
            ],
            dtype=np.int64,
        )
        return points, faces
    if prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        axis = str(cylinder.GetAxisAttr().Get()).upper()
        if axis not in {"X", "Y", "Z"}:
            raise ArticulatedUsdDepthSweepError(
                ["articulated_depth_cylinder_axis_unsupported"]
            )
        radius = float(cylinder.GetRadiusAttr().Get())
        half = float(cylinder.GetHeightAttr().Get()) / 2.0
        segments = 64
        ring = [
            (radius * math.cos(2.0 * math.pi * index / segments),
             radius * math.sin(2.0 * math.pi * index / segments))
            for index in range(segments)
        ]
        def point(axis_value: float, first: float, second: float) -> tuple[float, float, float]:
            if axis == "X":
                return (axis_value, first, second)
            if axis == "Y":
                return (first, axis_value, second)
            return (first, second, axis_value)

        points = np.asarray(
            [point(-half, first, second) for first, second in ring]
            + [point(half, first, second) for first, second in ring]
            + [point(-half, 0.0, 0.0), point(half, 0.0, 0.0)],
            dtype=np.float64,
        )
        left_center = 2 * segments
        right_center = left_center + 1
        faces: list[tuple[int, int, int]] = []
        for index in range(segments):
            nxt = (index + 1) % segments
            left = index
            left_next = nxt
            right = segments + index
            right_next = segments + nxt
            faces.extend(
                [
                    (left, right, right_next),
                    (left, right_next, left_next),
                    (left_center, left_next, left),
                    (right_center, right, right_next),
                ]
            )
        return points, np.asarray(faces, dtype=np.int64)
    raise ArticulatedUsdDepthSweepError(
        [f"articulated_depth_geometry_type_unsupported:{prim.GetTypeName()}"]
    )


def _column_transform(matrix: Any) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).T


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3]


def _render_authorized_gprim(prim: Any) -> bool:
    """Admit only visible, render-purpose geometry proven fully opaque."""

    from pxr import UsdGeom, UsdShade

    imageable = UsdGeom.Imageable(prim)
    if (
        str(imageable.ComputeVisibility()).lower() == "invisible"
        or str(imageable.ComputePurpose()).lower() not in {"default", "render"}
    ):
        return False
    display_opacity = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
    if display_opacity and any(
        not math.isfinite(float(value))
        or not math.isclose(float(value), 1.0, abs_tol=1.0e-9, rel_tol=0.0)
        for value in display_opacity
    ):
        return False
    material, _relationship = UsdShade.MaterialBindingAPI(
        prim
    ).ComputeBoundMaterial()
    if not material or not material.GetPrim().IsValid():
        return True
    try:
        source, _source_name, _source_type = material.ComputeSurfaceSource()
        shader = UsdShade.Shader(source.GetPrim())
        shader_id = str(shader.GetIdAttr().Get() or "")
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if shader_id != "UsdPreviewSurface":
        return False
    opacity = shader.GetInput("opacity")
    if opacity and opacity.HasConnectedSource():
        return False
    opacity_value = opacity.Get() if opacity else None
    if opacity_value is not None and not math.isclose(
        float(opacity_value), 1.0, abs_tol=1.0e-9, rel_tol=0.0
    ):
        return False
    return True


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
    asset = stage.GetDefaultPrim()
    if not asset.IsValid():
        raise ArticulatedUsdDepthSweepError(["articulated_depth_asset_root_missing"])
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_from_asset = _column_transform(cache.GetLocalToWorldTransform(asset))
    asset_from_world = np.linalg.inv(world_from_asset)
    groups: dict[str, list[np.ndarray]] = {"static": [], "moving": []}
    for prim in stage.Traverse():
        if (
            not prim.IsActive()
            or not prim.IsLoaded()
            or not (
                prim.IsA(UsdGeom.Mesh)
                or prim.IsA(UsdGeom.Cube)
                or prim.IsA(UsdGeom.Cylinder)
            )
            or not _render_authorized_gprim(prim)
        ):
            continue
        points, faces = _primitive_points_and_faces(prim)
        world_from_prim = _column_transform(cache.GetLocalToWorldTransform(prim))
        asset_points = _transform_points(
            points, asset_from_world @ world_from_prim
        )
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


def seal_replacement_usd_depth_sweep_request(
    *,
    asset_id: str,
    task_kind: str,
    task_freeze_digest: str,
    replacement_usd_sha256: str,
    replacement_usd_size_bytes: int,
    articulation_graph: Mapping[str, Any],
    link_paths: Mapping[str, str],
    joint_paths: Mapping[str, str],
    task_scoring_frame: Mapping[str, Any],
    camera_contract_digest: str,
    cameras: Sequence[Mapping[str, Any]],
    joint_state_cells: Sequence[Mapping[str, Any]],
    asset_prim_path: str = "/Asset",
    resolution_scale: float = 0.25,
    scene_state_role: str = "task_subject",
) -> dict[str, Any]:
    """Seal the generic articulated/rigid replacement depth-cell contract."""

    graph = validate_articulation_graph(
        articulation_graph,
        require_target_joint=task_kind == "articulated_interaction",
    )
    value: dict[str, Any] = {
        "schema_version": GENERAL_DEPTH_SWEEP_REQUEST_SCHEMA,
        "asset_id": asset_id,
        "task_kind": task_kind,
        "task_freeze_digest": task_freeze_digest,
        "replacement_usd_sha256": replacement_usd_sha256,
        "replacement_usd_size_bytes": replacement_usd_size_bytes,
        "asset_prim_path": asset_prim_path,
        "articulation_graph": graph,
        "articulation_graph_digest": canonical_digest(graph),
        "link_paths": dict(link_paths),
        "joint_paths": dict(joint_paths),
        "task_scoring_frame": json.loads(json.dumps(task_scoring_frame)),
        "camera_contract_digest": camera_contract_digest,
        "cameras": [json.loads(json.dumps(row)) for row in cameras],
        "camera_rows_digest": canonical_digest({"cameras": list(cameras)}),
        "joint_state_cells": [
            json.loads(json.dumps(row)) for row in joint_state_cells
        ],
        "scene_state_role": scene_state_role,
        "geometry_visibility_policy": {
            "computed_visibility": "visible_only",
            "admitted_purposes": ["default", "render"],
            "opacity": "fully_opaque_only",
            "mesh_topology": "explicit_triangles_only",
        },
        "resolution_scale": float(resolution_scale),
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return validate_replacement_usd_depth_sweep_request(value)


def validate_replacement_usd_depth_sweep_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact cameras, object poses, and complete graph joint states."""

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_request_not_json"]
        ) from exc
    errors: list[str] = []
    task_kind = str(payload.get("task_kind") or "")
    if payload.get("schema_version") != GENERAL_DEPTH_SWEEP_REQUEST_SCHEMA:
        errors.append("replacement_depth_request_schema_invalid")
    if task_kind not in {"articulated_interaction", "rigid_object_manipulation"}:
        errors.append("replacement_depth_task_kind_invalid")
    scene_state_role = str(payload.get("scene_state_role") or "task_subject")
    if scene_state_role not in SCENE_STATE_ROLES:
        errors.append("replacement_depth_scene_state_role_invalid")
    asset_id = str(payload.get("asset_id") or "")
    if not asset_id or not asset_id.replace("_", "a").replace("-", "a").isalnum():
        errors.append("replacement_depth_asset_id_invalid")
    if not _is_sha256_digest(payload.get("task_freeze_digest")):
        errors.append("replacement_depth_task_freeze_digest_invalid")
    digest = str(payload.get("replacement_usd_sha256") or "")
    if not _is_sha256_digest(digest):
        errors.append("replacement_depth_usd_digest_invalid")
    size_bytes = payload.get("replacement_usd_size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        errors.append("replacement_depth_usd_size_invalid")
    asset_path = str(payload.get("asset_prim_path") or "")
    if not asset_path.startswith("/"):
        errors.append("replacement_depth_asset_prim_path_invalid")
    try:
        graph = validate_articulation_graph(
            payload.get("articulation_graph") or {},
            require_target_joint=task_kind == "articulated_interaction",
        )
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
        graph = {"links": [], "joints": []}
    if payload.get("articulation_graph_digest") != canonical_digest(graph):
        errors.append("replacement_depth_articulation_graph_digest_invalid")
    link_ids = {row["link_id"] for row in graph.get("links", [])}
    joint_ids = {row["joint_id"] for row in graph.get("joints", [])}
    link_paths = payload.get("link_paths")
    joint_paths = payload.get("joint_paths")
    if (
        not isinstance(link_paths, Mapping)
        or set(link_paths) != link_ids
        or len(set(str(path) for path in link_paths.values())) != len(link_ids)
        or any(not str(path).startswith(asset_path + "/") for path in link_paths.values())
    ):
        errors.append("replacement_depth_link_paths_invalid")
    if (
        not isinstance(joint_paths, Mapping)
        or set(joint_paths) != joint_ids
        or len(set(str(path) for path in joint_paths.values())) != len(joint_ids)
        or any(not str(path).startswith(asset_path + "/") for path in joint_paths.values())
    ):
        errors.append("replacement_depth_joint_paths_invalid")
    scoring_frame = payload.get("task_scoring_frame")
    if not isinstance(scoring_frame, Mapping) or not str(
        scoring_frame.get("frame_id") or ""
    ):
        errors.append("replacement_depth_task_scoring_frame_invalid")
    else:
        try:
            _rigid_matrix(
                scoring_frame.get("T_asset_task_scoring"),
                "replacement_depth_task_scoring_frame_invalid",
            )
        except ArticulatedUsdDepthSweepError:
            errors.append("replacement_depth_task_scoring_frame_invalid")
    cameras = payload.get("cameras")
    if payload.get("geometry_visibility_policy") != {
        "computed_visibility": "visible_only",
        "admitted_purposes": ["default", "render"],
        "opacity": "fully_opaque_only",
        "mesh_topology": "explicit_triangles_only",
    }:
        errors.append("replacement_depth_geometry_visibility_policy_invalid")
    if not _is_sha256_digest(payload.get("camera_contract_digest")):
        errors.append("replacement_depth_camera_contract_digest_invalid")
    if not isinstance(cameras, list) or not cameras:
        errors.append("replacement_depth_cameras_invalid")
        cameras = []
    camera_ids: list[str] = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping):
            errors.append(f"replacement_depth_camera_invalid:{index}")
            continue
        camera_id = str(camera.get("camera_id") or "")
        camera_ids.append(camera_id)
        try:
            _rigid_matrix(
                camera.get("T_world_camera_opencv"),
                f"replacement_depth_camera_transform_invalid:{index}",
            )
        except ArticulatedUsdDepthSweepError:
            errors.append(f"replacement_depth_camera_transform_invalid:{index}")
        try:
            intrinsics = camera["intrinsics"]
            numeric = [
                float(intrinsics[field]) for field in ("fx", "fy", "cx", "cy")
            ]
            dimensions = [int(intrinsics[field]) for field in ("width", "height")]
            if min(numeric[:2]) <= 0.0 or min(dimensions) <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"replacement_depth_camera_invalid:{index}")
    if any(not camera_id for camera_id in camera_ids) or len(camera_ids) != len(
        set(camera_ids)
    ):
        errors.append("replacement_depth_camera_ids_invalid")
    if payload.get("camera_rows_digest") != canonical_digest({"cameras": cameras}):
        errors.append("replacement_depth_camera_rows_digest_invalid")
    cells = payload.get("joint_state_cells")
    if not isinstance(cells, list) or len(cells) < 2:
        errors.append("replacement_depth_joint_state_cells_invalid")
        cells = []
    cell_ids: list[str] = []
    joint_by_id = {row["joint_id"]: row for row in graph.get("joints", [])}
    transforms: list[list[list[float]]] = []
    target_states: list[tuple[float, ...]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            errors.append(f"replacement_depth_joint_state_cell_invalid:{index}")
            continue
        cell_id = str(cell.get("cell_id") or "")
        cell_ids.append(cell_id)
        try:
            transform = _rigid_matrix(
                cell.get("T_world_task_scoring"),
                f"replacement_depth_cell_transform_invalid:{index}",
            )
            transforms.append(transform.tolist())
        except ArticulatedUsdDepthSweepError:
            errors.append(f"replacement_depth_cell_transform_invalid:{index}")
        positions = cell.get("joint_positions")
        if not isinstance(positions, Mapping) or set(positions) != joint_ids:
            errors.append(f"replacement_depth_cell_joint_set_invalid:{index}")
            continue
        normalized: dict[str, float] = {}
        for joint_id, joint in joint_by_id.items():
            try:
                position = float(positions[joint_id])
            except (TypeError, ValueError):
                errors.append(
                    f"replacement_depth_cell_joint_position_invalid:{index}:{joint_id}"
                )
                continue
            if not math.isfinite(position) or not (
                float(joint["limits"][0]) <= position <= float(joint["limits"][1])
            ):
                errors.append(
                    f"replacement_depth_cell_joint_position_invalid:{index}:{joint_id}"
                )
            normalized[joint_id] = position
            if joint["role"] == "locked" and abs(
                position - float(joint["reset_position"])
            ) > float(joint["reset_tolerance"]):
                errors.append(
                    f"replacement_depth_cell_locked_joint_changed:{index}:{joint_id}"
                )
        for joint_id, joint in joint_by_id.items():
            dependency = joint["dependency"]
            if dependency is None or joint_id not in normalized:
                continue
            driver = dependency["driver_joint_id"]
            if driver not in normalized:
                errors.append(
                    f"replacement_depth_cell_dependency_driver_invalid:{index}:{joint_id}"
                )
                continue
            expected = (
                float(dependency["multiplier"]) * normalized[driver]
                + float(dependency["offset"])
            )
            if abs(normalized[joint_id] - expected) > float(dependency["tolerance"]):
                errors.append(
                    f"replacement_depth_cell_dependency_invalid:{index}:{joint_id}"
                )
        target_states.append(
            tuple(
                normalized.get(joint_id, math.nan)
                for joint_id in sorted(joint_ids)
            )
        )
    if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(set(cell_ids)):
        errors.append("replacement_depth_cell_ids_invalid")
    if scene_state_role == "co_present_passive":
        # A passive replacement represents the other, co-present task object
        # at its reset state.  It must participate in every subject cell's
        # depth composition, but it cannot silently follow the subject's
        # motion or articulation sweep.
        if transforms and not all(
            np.allclose(transforms[0], transform, atol=1e-12, rtol=0.0)
            for transform in transforms[1:]
        ):
            errors.append("replacement_depth_passive_pose_not_reset")
        if target_states and not all(
            np.allclose(target_states[0], state, atol=1e-12, rtol=0.0, equal_nan=True)
            for state in target_states[1:]
        ):
            errors.append("replacement_depth_passive_joint_state_not_reset")
    elif task_kind == "rigid_object_manipulation":
        if transforms and all(
            np.allclose(transforms[0], transform, atol=1e-12, rtol=0.0)
            for transform in transforms[1:]
        ):
            errors.append("replacement_depth_rigid_pose_range_missing")
    else:
        target_ids = sorted(
            joint_id
            for joint_id, joint in joint_by_id.items()
            if joint["role"] == "target"
        )
        if target_ids and len(
            {
                tuple(state[sorted(joint_ids).index(joint_id)] for joint_id in target_ids)
                for state in target_states
            }
        ) < 2:
            errors.append("replacement_depth_articulated_target_sweep_missing")
    scale = payload.get("resolution_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or not 0.0 < float(scale) <= 1.0
    ):
        errors.append("replacement_depth_resolution_scale_invalid")
    expected_digest = canonical_digest(payload, digest_field="request_digest")
    if payload.get("request_digest") != expected_digest:
        errors.append("replacement_depth_request_digest_invalid")
    if errors:
        raise ArticulatedUsdDepthSweepError(errors)
    return payload


def load_usd_link_triangles(
    usd_path: str | Path,
    *,
    asset_prim_path: str,
    link_paths: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Load Mesh/Cube/Cylinder triangles in each rigid link's local frame."""

    try:
        from pxr import Sdf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_openusd_runtime_missing"]
        ) from exc
    path = Path(usd_path).expanduser().resolve()
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedUsdDepthSweepError(["articulated_depth_usd_unreadable"])
    if float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0 or str(
        UsdGeom.GetStageUpAxis(stage)
    ).upper() != "Z":
        raise ArticulatedUsdDepthSweepError(["articulated_depth_usd_frame_invalid"])
    asset = stage.GetPrimAtPath(asset_prim_path)
    if not asset.IsValid():
        raise ArticulatedUsdDepthSweepError(["articulated_depth_asset_root_missing"])
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_from_asset = _column_transform(cache.GetLocalToWorldTransform(asset))
    asset_from_world = np.linalg.inv(world_from_asset)
    link_prims = {
        link_id: stage.GetPrimAtPath(path_value)
        for link_id, path_value in link_paths.items()
    }
    if any(not prim.IsValid() for prim in link_prims.values()):
        raise ArticulatedUsdDepthSweepError(["replacement_depth_link_prim_missing"])
    rest = {
        link_id: asset_from_world
        @ _column_transform(cache.GetLocalToWorldTransform(prim))
        for link_id, prim in link_prims.items()
    }
    groups: dict[str, list[np.ndarray]] = {link_id: [] for link_id in link_paths}
    type_counts: dict[str, int] = {}
    sorted_links = sorted(
        link_paths.items(), key=lambda item: len(item[1]), reverse=True
    )
    for prim in stage.Traverse():
        if (
            not prim.IsActive()
            or not prim.IsLoaded()
            or not (
                prim.IsA(UsdGeom.Mesh)
                or prim.IsA(UsdGeom.Cube)
                or prim.IsA(UsdGeom.Cylinder)
            )
            or not _render_authorized_gprim(prim)
        ):
            continue
        owner = next(
            (
                link_id
                for link_id, link_path in sorted_links
                if prim.GetPath().HasPrefix(Sdf.Path(link_path))
            ),
            None,
        )
        if owner is None:
            continue
        points, faces = _primitive_points_and_faces(prim)
        world_from_prim = _column_transform(cache.GetLocalToWorldTransform(prim))
        link_from_world = np.linalg.inv(
            _column_transform(cache.GetLocalToWorldTransform(link_prims[owner]))
        )
        local_points = _transform_points(points, link_from_world @ world_from_prim)
        groups[owner].append(local_points[faces])
        type_name = str(prim.GetTypeName())
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    if any(not rows for rows in groups.values()):
        missing = sorted(link_id for link_id, rows in groups.items() if not rows)
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_link_geometry_missing:" + ",".join(missing)]
        )
    return (
        {link_id: np.concatenate(rows) for link_id, rows in groups.items()},
        rest,
        type_counts,
    )


def _pose_matrix(position: Any, orientation: Any) -> np.ndarray:
    x, y, z, w = (float(value) for value in orientation)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _joint_delta_matrix(
    joint_type: str, delta: float, axis_local: Sequence[float]
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    axis = np.asarray(axis_local, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_joint_axis_invalid"]
        )
    axis /= norm
    if joint_type in {"revolute", "continuous"}:
        cosine = math.cos(float(delta))
        sine = math.sin(float(delta))
        cross = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        result[:3, :3] = (
            np.eye(3, dtype=np.float64) * cosine
            + (1.0 - cosine) * np.outer(axis, axis)
            + sine * cross
        )
    elif joint_type == "prismatic":
        result[:3, 3] = axis * float(delta)
    return result


def _usd_joint_axis_local(prim: Any, joint_type: str) -> np.ndarray:
    from pxr import UsdPhysics

    if joint_type in {"revolute", "continuous"}:
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ArticulatedUsdDepthSweepError(
                ["replacement_depth_joint_type_mismatch"]
            )
        token = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "")
    elif joint_type == "prismatic":
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            raise ArticulatedUsdDepthSweepError(
                ["replacement_depth_joint_type_mismatch"]
            )
        token = str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get() or "")
    elif joint_type == "fixed":
        if not prim.IsA(UsdPhysics.FixedJoint):
            raise ArticulatedUsdDepthSweepError(
                ["replacement_depth_joint_type_mismatch"]
            )
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:  # pragma: no cover - articulation graph validation rejects this first
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_joint_type_mismatch"]
        )
    try:
        index = {"X": 0, "Y": 1, "Z": 2}[token.upper()]
    except KeyError as exc:
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_joint_axis_invalid"]
        ) from exc
    axis = np.zeros(3, dtype=np.float64)
    axis[index] = 1.0
    return axis


def _link_transforms_for_state(
    *,
    stage: Any,
    graph: Mapping[str, Any],
    link_paths: Mapping[str, str],
    joint_paths: Mapping[str, str],
    rest: Mapping[str, np.ndarray],
    joint_positions: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    from pxr import UsdPhysics

    root_id = next(row["link_id"] for row in graph["links"] if row["is_root"])
    current = {root_id: np.asarray(rest[root_id], dtype=np.float64)}
    remaining = list(graph["joints"])
    while remaining:
        progress = False
        for joint in list(remaining):
            parent = joint["parent_link_id"]
            child = joint["child_link_id"]
            if parent not in current:
                continue
            prim = stage.GetPrimAtPath(joint_paths[joint["joint_id"]])
            if not prim.IsValid():
                raise ArticulatedUsdDepthSweepError(
                    [f"replacement_depth_joint_prim_missing:{joint['joint_id']}"]
                )
            authored = UsdPhysics.Joint(prim)
            body0 = authored.GetBody0Rel().GetTargets()
            body1 = authored.GetBody1Rel().GetTargets()
            if (
                len(body0) != 1
                or len(body1) != 1
                or str(body0[0]) != link_paths[parent]
                or str(body1[0]) != link_paths[child]
            ):
                raise ArticulatedUsdDepthSweepError(
                    [f"replacement_depth_joint_body_binding_mismatch:{joint['joint_id']}"]
                )
            parent_frame = _pose_matrix(
                authored.GetLocalPos0Attr().Get(),
                _quat_from_gf(authored.GetLocalRot0Attr().Get()),
            )
            child_frame = _pose_matrix(
                authored.GetLocalPos1Attr().Get(),
                _quat_from_gf(authored.GetLocalRot1Attr().Get()),
            )
            axis_local = _usd_joint_axis_local(prim, joint["joint_type"])
            parent_joint_asset = rest[parent] @ parent_frame
            child_joint_asset = rest[child] @ child_frame
            reset_position = float(joint["reset_position"])
            reset_joint_asset = (
                parent_joint_asset
                @ _joint_delta_matrix(
                    joint["joint_type"], -reset_position, axis_local
                )
            )
            if not np.allclose(
                reset_joint_asset, child_joint_asset, atol=1.0e-5, rtol=0.0
            ):
                raise ArticulatedUsdDepthSweepError(
                    [f"replacement_depth_joint_reset_frame_mismatch:{joint['joint_id']}"]
                )
            if joint["joint_type"] != "fixed":
                expected_axis = np.asarray(joint["axis"], dtype=np.float64)
                expected_axis /= np.linalg.norm(expected_axis)
                for role, frame in (
                    ("parent", parent_joint_asset),
                    ("child", child_joint_asset),
                ):
                    authored_axis = frame[:3, :3] @ axis_local
                    if float(np.dot(authored_axis, expected_axis)) < 1.0 - 1.0e-6:
                        raise ArticulatedUsdDepthSweepError(
                            [
                                "replacement_depth_joint_axis_binding_mismatch:"
                                f"{joint['joint_id']}:{role}"
                            ]
                        )
            delta = float(joint_positions[joint["joint_id"]]) - float(
                joint["reset_position"]
            )
            relative_rest = np.linalg.inv(rest[parent]) @ rest[child]
            pivot_delta = (
                parent_frame
                @ _joint_delta_matrix(joint["joint_type"], -delta, axis_local)
                @ np.linalg.inv(parent_frame)
            )
            current[child] = current[parent] @ pivot_delta @ relative_rest
            remaining.remove(joint)
            progress = True
        if not progress:
            raise ArticulatedUsdDepthSweepError(
                ["replacement_depth_joint_graph_not_traversable"]
            )
    return current


def _quat_from_gf(value: Any) -> list[float]:
    imaginary = value.GetImaginary()
    return [
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(value.GetReal()),
    ]


def materialize_replacement_usd_depth_sweep(
    *,
    usd_path: str | Path,
    request: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Rasterize generic graph states or rigid pose cells from exact USD bytes."""

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover
        raise ArticulatedUsdDepthSweepError(
            ["articulated_depth_openusd_runtime_missing"]
        ) from exc
    admitted = validate_replacement_usd_depth_sweep_request(request)
    usd = Path(usd_path).expanduser().resolve()
    if (
        not usd.is_file()
        or usd.is_symlink()
        or usd.stat().st_size != admitted["replacement_usd_size_bytes"]
        or _sha256(usd) != admitted["replacement_usd_sha256"]
    ):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_usd_bytes_changed"]
        )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArticulatedUsdDepthSweepError(["articulated_depth_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(usd), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedUsdDepthSweepError(["articulated_depth_usd_unreadable"])
    local_triangles, rest, type_counts = load_usd_link_triangles(
        usd,
        asset_prim_path=admitted["asset_prim_path"],
        link_paths=admitted["link_paths"],
    )
    depths: list[np.ndarray] = []
    cells: list[dict[str, Any]] = []
    graph = admitted["articulation_graph"]
    for camera in admitted["cameras"]:
        for state in admitted["joint_state_cells"]:
            transforms = _link_transforms_for_state(
                stage=stage,
                graph=graph,
                link_paths=admitted["link_paths"],
                joint_paths=admitted["joint_paths"],
                rest=rest,
                joint_positions=state["joint_positions"],
            )
            asset_triangles = np.concatenate(
                [
                    _transform_triangles(local_triangles[link_id], transforms[link_id])
                    for link_id in sorted(local_triangles)
                ]
            )
            world_from_asset = _rigid_matrix(
                state["T_world_task_scoring"],
                "replacement_depth_cell_transform_invalid",
            ) @ np.linalg.inv(
                _rigid_matrix(
                    admitted["task_scoring_frame"]["T_asset_task_scoring"],
                    "replacement_depth_task_scoring_frame_invalid",
                )
            )
            world_triangles = _transform_triangles(
                asset_triangles, world_from_asset
            )
            depths.append(
                rasterize_triangle_depth(
                    world_triangles,
                    T_world_camera_opencv=camera["T_world_camera_opencv"],
                    intrinsics=camera["intrinsics"],
                    resolution_scale=float(admitted["resolution_scale"]),
                )
            )
            cells.append(
                {
                    "camera_id": camera["camera_id"],
                    "cell_id": state["cell_id"],
                    "task_scoring_frame_id": admitted["task_scoring_frame"]["frame_id"],
                    "T_world_task_scoring": state["T_world_task_scoring"],
                    "T_world_asset": world_from_asset.tolist(),
                    "joint_positions": state["joint_positions"],
                }
            )
    depth_array = np.stack(depths).astype(np.float32)
    arrays_path = output / "replacement_depth_sweep.npy"
    np.save(arrays_path, depth_array, allow_pickle=False)
    manifest: dict[str, Any] = {
        "schema_version": GENERAL_DEPTH_SWEEP_SCHEMA,
        "status": "actual_usd_geometry_depth_rasterized",
        "request_digest": admitted["request_digest"],
        "asset_id": admitted["asset_id"],
        "task_kind": admitted["task_kind"],
        "task_freeze_digest": admitted["task_freeze_digest"],
        "camera_contract_digest": admitted["camera_contract_digest"],
        "camera_rows_digest": admitted["camera_rows_digest"],
        "actual_usd_geometry_depth_rasterized": True,
        "actual_mesh_depth_rasterized": type_counts.get("Mesh", 0) > 0,
        "caller_supplied_coverage_mask": False,
        "replacement_usd": {
            "path": str(usd),
            "size_bytes": usd.stat().st_size,
            "sha256": _sha256(usd),
        },
        "asset_prim_path": admitted["asset_prim_path"],
        "task_scoring_frame": admitted["task_scoring_frame"],
        "scene_state_role": admitted.get("scene_state_role", "task_subject"),
        "geometry_visibility_policy": admitted["geometry_visibility_policy"],
        "asset_root_authored_transform_removed_before_placement": True,
        "T_world_asset_applied_exactly_once_per_cell": True,
        "articulation_graph_digest": admitted["articulation_graph_digest"],
        "geometry_type_counts": type_counts,
        "primitive_tessellation": {
            "Cube": "exact_triangular_faces",
            "Cylinder": "64_segment_inscribed_conservative_coverage",
        },
        "cells": cells,
        "camera_count": len(admitted["cameras"]),
        "state_cell_count": len(admitted["joint_state_cells"]),
        "resolution_scale": float(admitted["resolution_scale"]),
        "depth_dimensions": [int(depth_array.shape[2]), int(depth_array.shape[1])],
        "finite_depth_pixel_count_by_cell": [
            int(np.isfinite(depth).sum()) for depth in depth_array
        ],
        "arrays": _record(arrays_path, output),
        "renderer": {
            "name": "blueprint_numpy_perspective_correct_triangle_depth.v2",
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
    (output / f"{GENERAL_DEPTH_SWEEP_SCHEMA}.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def attest_legacy_default_subject_depth_sweep(
    *,
    source_manifest_path: str | Path,
    source_request_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Make a file-backed subject-role attestation for a legacy full-depth sweep.

    Versions of the generic depth sweep before co-present composition recorded
    ``task_subject`` only as the implicit request default.  This narrow bridge
    permits such a sweep to participate in a new shared-scene composition only
    after reopening its request, USD, manifest, and depth array.  It never
    re-rasterizes or changes the depth values: the copied array must be exactly
    the source file, and an explicit role is added only for the legacy default.
    """

    source_manifest_file = Path(source_manifest_path).expanduser().resolve()
    source_request_file = Path(source_request_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if (
        not source_manifest_file.is_file()
        or source_manifest_file.is_symlink()
        or not source_request_file.is_file()
        or source_request_file.is_symlink()
    ):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_source_missing"]
        )
    if output.exists() and any(output.iterdir()):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_output_not_empty"]
        )
    manifest = _read_object(
        source_manifest_file, "replacement_depth_legacy_subject_manifest_invalid"
    )
    request = _read_object(
        source_request_file, "replacement_depth_legacy_subject_request_invalid"
    )
    if "scene_state_role" in request or "scene_state_role" in manifest:
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_role_not_implicit"]
        )
    admitted = validate_replacement_usd_depth_sweep_request(request)
    if (
        manifest.get("schema_version") != GENERAL_DEPTH_SWEEP_SCHEMA
        or manifest.get("status") != "actual_usd_geometry_depth_rasterized"
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("request_digest") != admitted["request_digest"]
        or manifest.get("asset_id") != admitted["asset_id"]
        or manifest.get("task_kind") != admitted["task_kind"]
        or manifest.get("task_freeze_digest") != admitted["task_freeze_digest"]
        or manifest.get("camera_contract_digest")
        != admitted["camera_contract_digest"]
        or manifest.get("camera_rows_digest") != admitted["camera_rows_digest"]
        or manifest.get("resolution_scale") != admitted["resolution_scale"]
        or manifest.get("actual_usd_geometry_depth_rasterized") is not True
        or manifest.get("caller_supplied_coverage_mask") is not False
    ):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_manifest_invalid"]
        )
    usd_record = manifest.get("replacement_usd")
    usd = Path(str(usd_record.get("path") or "")).expanduser().resolve() if isinstance(usd_record, Mapping) else None
    if (
        usd is None
        or not usd.is_file()
        or usd.is_symlink()
        or usd.stat().st_size != admitted["replacement_usd_size_bytes"]
        or _sha256(usd) != admitted["replacement_usd_sha256"]
        or usd_record.get("size_bytes") != usd.stat().st_size
        or usd_record.get("sha256") != _sha256(usd)
    ):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_usd_changed"]
        )
    array_record = manifest.get("arrays")
    if not isinstance(array_record, Mapping):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_array_invalid"]
        )
    relative = str(array_record.get("relative_path") or "")
    array = (source_manifest_file.parent / relative).resolve()
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or not array.is_relative_to(source_manifest_file.parent)
        or not array.is_file()
        or array.is_symlink()
        or array.stat().st_size != array_record.get("size_bytes")
        or _sha256(array) != array_record.get("sha256")
    ):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_array_invalid"]
        )
    try:
        depth = np.asarray(np.load(array, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_array_invalid"]
        ) from exc
    if (
        depth.ndim != 3
        or depth.shape[0] != len(manifest.get("cells") or [])
        or manifest.get("depth_dimensions")
        != [int(depth.shape[2]), int(depth.shape[1])]
        or manifest.get("finite_depth_pixel_count_by_cell")
        != [int(np.isfinite(row).sum()) for row in depth]
        or np.isnan(depth).any()
        or np.isneginf(depth).any()
        or np.any(np.isfinite(depth) & (depth <= 0.0))
    ):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_array_invalid"]
        )
    output.mkdir(parents=True, exist_ok=True)
    destination_array = output / "replacement_depth_sweep.npy"
    try:
        os.link(array, destination_array)
    except OSError as exc:
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_array_hardlink_failed"]
        ) from exc
    if _sha256(destination_array) != _sha256(array):
        raise ArticulatedUsdDepthSweepError(
            ["replacement_depth_legacy_subject_array_copy_invalid"]
        )
    upgraded = json.loads(json.dumps(manifest, allow_nan=False))
    upgraded["scene_state_role"] = "task_subject"
    upgraded["arrays"] = _record(destination_array, output)
    upgraded["legacy_subject_role_attestation"] = {
        "source_manifest": {
            "path": str(source_manifest_file),
            "size_bytes": source_manifest_file.stat().st_size,
            "sha256": _sha256(source_manifest_file),
            "manifest_digest": manifest["manifest_digest"],
        },
        "source_request": {
            "path": str(source_request_file),
            "size_bytes": source_request_file.stat().st_size,
            "sha256": _sha256(source_request_file),
            "request_digest": admitted["request_digest"],
        },
        "source_role": "implicit_legacy_default_task_subject",
        "depth_array_hardlinked_byte_exact": True,
    }
    upgraded["manifest_digest"] = canonical_digest(
        upgraded, digest_field="manifest_digest"
    )
    (output / f"{GENERAL_DEPTH_SWEEP_SCHEMA}.json").write_text(
        canonical_json(upgraded) + "\n", encoding="utf-8"
    )
    return upgraded


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
    """840796-compatible adapter for a legacy one-moving-link door sweep."""

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
    from pxr import Usd, UsdGeom

    inspection_stage = Usd.Stage.Open(str(usd), load=Usd.Stage.LoadAll)
    actual_mesh_depth = bool(
        inspection_stage
        and any(prim.IsA(UsdGeom.Mesh) for prim in inspection_stage.Traverse())
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
        "actual_usd_geometry_depth_rasterized": True,
        "actual_mesh_depth_rasterized": actual_mesh_depth,
        "caller_supplied_coverage_mask": False,
        "replacement_usd": {"path": str(usd), "sha256": _sha256(usd)},
        "moving_link_path": moving_link_path,
        "hinge_origin_asset_m": [float(value) for value in hinge_origin_asset_m],
        "hinge_axis_asset": [float(value) for value in hinge_axis_asset],
        "T_world_asset": asset_to_world.tolist(),
        "asset_root_authored_transform_removed_before_placement": True,
        "T_world_asset_applied_exactly_once": True,
        "legacy_single_moving_link_compatibility_adapter": True,
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


def _full_resolution_residual_masks_are_inpainting_authority(
    *,
    black_manifest: Mapping[str, Any],
    white_manifest: Mapping[str, Any],
    black_rows: Mapping[str, Mapping[str, Any]],
    white_rows: Mapping[str, Mapping[str, Any]],
    depth_manifest: Mapping[str, Any],
    source_frames_match_depth: bool,
    output_width: int,
    output_height: int,
    coverage_margin_pixels: int,
) -> bool:
    """Return whether residual masks can constrain, but not complete, an edit.

    A coarse source-layer audit is useful for deciding that a replacement does
    *not* cover a deletion, but it must never become an inpainting mask.  This
    narrow promotion requires the actual calibrated source-frame dimensions,
    a full-resolution replacement-depth sweep, matched black/white method
    inputs, and a conservative erosion margin.  It authorizes only a later
    backend's inside-mask edit boundary; it never claims an inpainting result.
    """

    admitted_classes = {"method_input", "evaluation_authorized"}
    expected_dimensions = {"width": output_width, "height": output_height}
    black_settings = (
        black_manifest.get("render_settings")
        if isinstance(black_manifest.get("render_settings"), Mapping)
        else {}
    )
    white_settings = (
        white_manifest.get("render_settings")
        if isinstance(white_manifest.get("render_settings"), Mapping)
        else {}
    )
    black_calibration = (
        black_manifest.get("calibrated_camera_file")
        if isinstance(black_manifest.get("calibrated_camera_file"), Mapping)
        else {}
    )
    white_calibration = (
        white_manifest.get("calibrated_camera_file")
        if isinstance(white_manifest.get("calibrated_camera_file"), Mapping)
        else {}
    )
    if (
        black_manifest.get("authorization_class") not in admitted_classes
        or white_manifest.get("authorization_class") not in admitted_classes
        or black_manifest.get("authorization_class")
        != white_manifest.get("authorization_class")
        or black_manifest.get("splat_digest") != white_manifest.get("splat_digest")
        or black_settings.get("dimensions") != expected_dimensions
        or white_settings.get("dimensions") != expected_dimensions
        or black_calibration.get("binding") != "caller_file_exact_match"
        or white_calibration.get("binding") != "caller_file_exact_match"
        or set(black_rows) != set(white_rows)
        or any(
            row.get("width") != output_width or row.get("height") != output_height
            for row in [*black_rows.values(), *white_rows.values()]
        )
        or depth_manifest.get("resolution_scale") != 1.0
        or isinstance(coverage_margin_pixels, bool)
        or not isinstance(coverage_margin_pixels, int)
        or coverage_margin_pixels < 1
        or not source_frames_match_depth
    ):
        return False
    return True


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
                **_cell_state_fields(cell),
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
    if not _qualified_depth_manifest(depth_manifest):
        raise ArticulatedUsdDepthSweepError(
            ["source_coverage_depth_manifest_invalid"]
        )
    if depth_manifest.get("schema_version") == COMPOSED_DEPTH_SWEEP_SCHEMA:
        # A composition receipt is not a caller assertion: reopen every
        # constituent USD sweep and recompute the nearest-depth array before
        # using it as the source-coverage boundary.
        from .public_scene_replacement_depth_composition import (
            ReplacementDepthCompositionError,
            validate_replacement_depth_composition,
        )

        try:
            validate_replacement_depth_composition(
                depth_manifest, receipt_path=depth_path
            )
        except ReplacementDepthCompositionError as exc:
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_composed_depth_receipt_invalid", *exc.codes]
            ) from exc
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
    source_frames_match_depth = True
    for camera_id in camera_ids:
        black_frame = black_path.parent / str(black_rows[camera_id]["relative_path"])
        white_frame = white_path.parent / str(white_rows[camera_id]["relative_path"])
        black = cv2.imread(str(black_frame), cv2.IMREAD_COLOR)
        white = cv2.imread(str(white_frame), cv2.IMREAD_COLOR)
        if black is None or white is None:
            raise ArticulatedUsdDepthSweepError(
                ["source_coverage_render_frame_unreadable"]
            )
        if (
            black.shape[:2] != (output_height, output_width)
            or white.shape[:2] != (output_height, output_width)
        ):
            source_frames_match_depth = False
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
    inpainting_mask_authority = _full_resolution_residual_masks_are_inpainting_authority(
        black_manifest=black_manifest,
        white_manifest=white_manifest,
        black_rows=black_rows,
        white_rows=white_rows,
        depth_manifest=depth_manifest,
        source_frames_match_depth=source_frames_match_depth,
        output_width=output_width,
        output_height=output_height,
        coverage_margin_pixels=coverage_margin_pixels,
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
                "derived_from_all_state_cells": len(cell_indices),
                **(
                    {"derived_from_all_door_cells": len(cell_indices)}
                    if depth_manifest.get("schema_version") == DEPTH_SWEEP_SCHEMA
                    else {}
                ),
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
            label = _cell_label(cells[cell_index])
            cv2.putText(
                panel,
                f"{camera_id}  {label}  red=uncovered",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{camera_id}  {label}  red=uncovered",
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
            **_depth_manifest_identity(depth_manifest),
        },
        "camera_ids": camera_ids,
        "significant_alpha_threshold": float(significant_alpha_threshold),
        "coverage_margin_pixels": coverage_margin_pixels,
        "source_alpha": _record(alpha_path, output),
        "review_contact_sheets": review_records,
        "uncovered_source_support_masks": seam_records,
        "uncovered_source_support_masks_are_inpainting_authority": (
            inpainting_mask_authority
        ),
        "inpainting_mask_eligibility": {
            "full_resolution_source_frames": source_frames_match_depth,
            "full_resolution_replacement_depth": depth_manifest.get(
                "resolution_scale"
            )
            == 1.0,
            "calibrated_method_input_pair": inpainting_mask_authority,
            "authorizes_only": (
                "future_exact_mask_contained_multi_view_edit_input"
                if inpainting_mask_authority
                else None
            ),
            "inpainting_result_qualified": False,
        },
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
    if depth_manifest.get("schema_version") == COMPOSED_DEPTH_SWEEP_SCHEMA:
        manifest.update(
            {
                "task_id": depth_manifest["task_id"],
                "task_freeze_digest": depth_manifest["task_freeze_digest"],
                "replacement_asset_id": depth_manifest["scored_task_asset_id"],
                "co_present_replacement_asset_ids": depth_manifest[
                    "replacement_asset_ids"
                ],
                "replacement_depth_composition": {
                    "path": str(depth_path),
                    "size_bytes": depth_path.stat().st_size,
                    "sha256": _sha256(depth_path),
                    "receipt_digest": depth_manifest["receipt_digest"],
                },
            }
        )
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = output / f"{SOURCE_COVERAGE_AUDIT_SCHEMA}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def materialize_deleted_source_layer_replacement_coverage_qualification(
    *,
    source_layer_coverage_audit_path: str | Path,
    depth_sweep_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Promote an exact zero-residue source-layer audit into joinable coverage.

    A target-core silhouette audit is not enough to delete ambiguous source
    Gaussians: the exact *deleted* source layer must be hidden by the
    replacement in every frozen camera/state cell.  This is intentionally the
    strict, no-inpainting branch.  Any nonzero source residue remains a typed
    seam/inpainting blocker rather than becoming permission to delete more
    records.
    """

    audit_path = Path(source_layer_coverage_audit_path).expanduser().resolve()
    depth_path = Path(depth_sweep_manifest_path).expanduser().resolve()
    audit = _read_object(audit_path, "source_layer_coverage_qualification_audit_unreadable")
    if (
        audit.get("schema_version") != SOURCE_COVERAGE_AUDIT_SCHEMA
        or audit.get("status") != "source_layer_coverage_measured"
        or audit.get("manifest_digest")
        != canonical_digest(audit, digest_field="manifest_digest")
        or not _is_sha256_digest(audit.get("source_layer_splat_digest"))
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_audit_invalid"]
        )
    threshold = audit.get("significant_alpha_threshold")
    margin = audit.get("coverage_margin_pixels")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isclose(
            float(threshold), _EXACT_SOURCE_ALPHA_THRESHOLD, abs_tol=1e-12, rel_tol=0.0
        )
        or isinstance(margin, bool)
        or not isinstance(margin, int)
        or margin < 1
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_policy_not_conservative"]
        )
    depth = _read_object(depth_path, "source_layer_coverage_qualification_depth_unreadable")
    if not _qualified_depth_manifest(depth):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_depth_invalid"]
        )
    bound_depth = audit.get("depth_sweep_manifest")
    if (
        not isinstance(bound_depth, Mapping)
        or bound_depth.get("sha256") != _sha256(depth_path)
        or any(
            bound_depth.get(key) != value
            for key, value in _depth_manifest_identity(depth).items()
        )
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_depth_join_mismatch"]
        )
    cells = depth.get("cells")
    audit_cells = audit.get("cells")
    camera_ids = audit.get("camera_ids")
    if (
        not isinstance(cells, list)
        or not cells
        or not isinstance(audit_cells, list)
        or len(audit_cells) != len(cells)
        or not isinstance(camera_ids, list)
        or not camera_ids
        or any(not isinstance(camera_id, str) or not camera_id for camera_id in camera_ids)
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_cells_invalid"]
        )
    expected_camera_ids = list(
        dict.fromkeys(str(cell.get("camera_id") or "") for cell in cells)
    )
    if (
        any(not camera_id for camera_id in expected_camera_ids)
        or camera_ids != expected_camera_ids
    ):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_camera_join_mismatch"]
        )

    normalized_cells: list[dict[str, Any]] = []
    for index, (audit_cell, depth_cell) in enumerate(zip(audit_cells, cells, strict=True)):
        if not isinstance(audit_cell, Mapping) or not isinstance(depth_cell, Mapping):
            raise ArticulatedUsdDepthSweepError(
                ["source_layer_coverage_qualification_cells_invalid"]
            )
        state_fields = _cell_state_fields(depth_cell)
        if (
            audit_cell.get("cell_index") != index
            or audit_cell.get("camera_id") != depth_cell.get("camera_id")
            or any(audit_cell.get(key) != value for key, value in state_fields.items())
        ):
            raise ArticulatedUsdDepthSweepError(
                ["source_layer_coverage_qualification_cell_join_mismatch"]
            )
        numeric = {
            key: audit_cell.get(key)
            for key in (
                "uncovered_significant_pixel_count",
                "largest_uncovered_component_pixels",
                "uncovered_alpha_sum",
                "uncovered_alpha_fraction",
            )
        }
        if (
            isinstance(numeric["uncovered_significant_pixel_count"], bool)
            or not isinstance(numeric["uncovered_significant_pixel_count"], int)
            or numeric["uncovered_significant_pixel_count"] < 0
            or isinstance(numeric["largest_uncovered_component_pixels"], bool)
            or not isinstance(numeric["largest_uncovered_component_pixels"], int)
            or numeric["largest_uncovered_component_pixels"] < 0
            or any(
                isinstance(numeric[key], bool)
                or not isinstance(numeric[key], (int, float))
                or not math.isfinite(float(numeric[key]))
                or float(numeric[key]) < 0.0
                for key in ("uncovered_alpha_sum", "uncovered_alpha_fraction")
            )
        ):
            raise ArticulatedUsdDepthSweepError(
                ["source_layer_coverage_qualification_metrics_invalid"]
            )
        if (
            numeric["uncovered_significant_pixel_count"] != 0
            or numeric["largest_uncovered_component_pixels"] != 0
            or float(numeric["uncovered_alpha_sum"]) > 1e-12
            or float(numeric["uncovered_alpha_fraction"]) > 1e-12
        ):
            raise ArticulatedUsdDepthSweepError(
                ["source_layer_coverage_qualification_source_residue_observed"]
            )
        normalized_cells.append(
            {
                "camera_id": str(depth_cell["camera_id"]),
                **state_fields,
                "residual_significant_pixels": 0,
                "residual_max_connected_component_pixels": 0,
                # There is no residual to authorize.  The true value is
                # therefore vacuously contained and can never broaden a seam.
                "residual_inside_target_core_mask": True,
                "outside_mask_changed_pixels": 0,
                "source_residual_alpha_fraction": 0.0,
            }
        )

    state_cell_ids = list(
        dict.fromkeys(
            str(cell.get("cell_id") or "") for cell in cells if cell.get("cell_id")
        )
    )
    door_angles = (
        list(
            dict.fromkeys(
                float(cell["commanded_door_angle_deg"])
                for cell in cells
                if "commanded_door_angle_deg" in cell
            )
        )
        if depth.get("schema_version") == DEPTH_SWEEP_SCHEMA
        else []
    )
    if bool(state_cell_ids) == bool(door_angles):
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_state_binding_invalid"]
        )
    receipt: dict[str, Any] = {
        "schema_version": TARGET_CORE_COVERAGE_AUDIT_SCHEMA,
        "status": DELETED_SOURCE_LAYER_COVERAGE_STATUS,
        "coverage_scope": "deleted_source_layer",
        "coverage_qualified": True,
        "source_layer_coverage_audit": {
            "path": str(audit_path),
            "size_bytes": audit_path.stat().st_size,
            "sha256": _sha256(audit_path),
            "manifest_digest": audit["manifest_digest"],
        },
        "source_layer_splat_digest": audit.get("source_layer_splat_digest"),
        "depth_sweep_manifest": {
            "path": str(depth_path),
            "sha256": _sha256(depth_path),
            **_depth_manifest_identity(depth),
            "replacement_usd": depth.get("replacement_usd"),
        },
        "camera_ids": expected_camera_ids,
        "state_cell_ids": state_cell_ids,
        **({"door_state_angles_degrees": door_angles} if door_angles else {}),
        "cells": normalized_cells,
        "significant_alpha_threshold": float(threshold),
        "coverage_margin_pixels": margin,
        "maximum_residual_connected_component_pixels": 0,
        "maximum_protected_changed_pixels": 0,
        "all_deleted_source_contribution_occluded": True,
        "caller_asserted_coverage_accepted": False,
        "rendered_pixels_changed_by_audit": False,
        "residual_is_narrow_seam_candidate_not_inpainting_success": False,
        "claim_ceiling": "actual_usd_depth_zero_residue_visibility_qualification_not_native_or_physical",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ArticulatedUsdDepthSweepError(
            ["source_layer_coverage_qualification_output_exists"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


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
    if not _qualified_depth_manifest(depth_manifest):
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
            state_fields = _cell_state_fields(cell)
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
                **state_fields,
                **(
                    {
                        "door_state_angle_degrees": state_fields[
                            "commanded_door_angle_deg"
                        ],
                        "readback_door_state_angle_degrees": state_fields[
                            "readback_door_angle_deg"
                        ],
                    }
                    if "commanded_door_angle_deg" in state_fields
                    else {}
                ),
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
                "derived_from_all_state_cells": sum(
                    str(cell.get("camera_id") or "") == camera_id for cell in cells
                ),
                **(
                    {
                        "derived_from_all_door_cells": sum(
                            str(cell.get("camera_id") or "") == camera_id
                            for cell in cells
                        )
                    }
                    if depth_manifest.get("schema_version") == DEPTH_SWEEP_SCHEMA
                    else {}
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
    door_states = (
        list(
            dict.fromkeys(
                float(cell["commanded_door_angle_deg"]) for cell in cells
            )
        )
        if depth_manifest.get("schema_version") == DEPTH_SWEEP_SCHEMA
        else []
    )
    state_ids = list(
        dict.fromkeys(str(cell.get("cell_id") or "") for cell in cells if cell.get("cell_id"))
    )
    manifest: dict[str, Any] = {
        "schema_version": TARGET_CORE_COVERAGE_AUDIT_SCHEMA,
        "status": "target_core_replacement_coverage_measured",
        "coverage_qualified": coverage_qualified,
        "depth_sweep_manifest": {
            "sha256": _sha256(depth_path),
            **_depth_manifest_identity(depth_manifest),
            "replacement_usd": depth_manifest.get("replacement_usd"),
        },
        "camera_ids": camera_ids,
        "state_cell_ids": state_ids,
        **(
            {"door_state_angles_degrees": door_states}
            if depth_manifest.get("schema_version") == DEPTH_SWEEP_SCHEMA
            else {}
        ),
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
    if not _qualified_depth_manifest(depth_manifest):
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
    contact_panels: dict[str, list[tuple[str, Path]]] = {
        camera_id: [] for camera_id in camera_ids
    }
    # OpenCV stores BGR.  The public contract remains RGB.
    base_bgr = np.asarray(color[::-1], dtype=np.float32)
    for index, cell in enumerate(cells):
        camera_id = str(cell.get("camera_id") or "")
        try:
            state_fields = _cell_state_fields(cell)
            label = _cell_label(cell)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_depth_cells_invalid"]
            ) from exc
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
        state_token = str(cell.get("cell_id") or "")
        if not state_token:
            angle = float(cell["commanded_door_angle_deg"])
            state_token = f"door_{angle:07.3f}".replace("-", "m").replace(".", "p")
        frame_path = frames_root / f"{camera_id}__{state_token}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise ArticulatedUsdDepthSweepError(
                ["reference_hybrid_frame_write_failed"]
            )
        frame_records.append(
            {
                **_record(frame_path, output),
                "camera_id": camera_id,
                **state_fields,
                "replacement_covered_pixel_count": int(finite.sum()),
            }
        )
        contact_panels[camera_id].append((label, frame_path))

    sheets_root = output / "contact_sheets"
    sheets_root.mkdir()
    sheet_records: list[dict[str, Any]] = []
    for camera_id, candidates in contact_panels.items():
        selected_indices = sorted({0, len(candidates) // 2, len(candidates) - 1})
        panels = []
        for selected_index in selected_indices:
            label, frame_path = candidates[selected_index]
            panel = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if panel is None:
                raise ArticulatedUsdDepthSweepError(
                    ["reference_hybrid_frame_unreadable"]
                )
            cv2.putText(
                panel,
                f"{camera_id}  {label}  neutral=USD silhouette",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (10, 10, 10),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{camera_id}  {label}  neutral=USD silhouette",
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
            **_depth_manifest_identity(depth_manifest),
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
    "attest_legacy_default_subject_depth_sweep",
    "DEPTH_SWEEP_SCHEMA",
    "GENERAL_DEPTH_SWEEP_REQUEST_SCHEMA",
    "GENERAL_DEPTH_SWEEP_SCHEMA",
    "DELETED_SOURCE_LAYER_COVERAGE_STATUS",
    "REFERENCE_HYBRID_REVIEW_SCHEMA",
    "SOURCE_COVERAGE_AUDIT_SCHEMA",
    "TARGET_CORE_COVERAGE_AUDIT_SCHEMA",
    "conservative_max_pool_alpha",
    "evaluate_source_alpha_coverage",
    "load_articulated_usd_triangles",
    "load_usd_link_triangles",
    "materialize_articulated_usd_depth_sweep",
    "materialize_replacement_usd_depth_sweep",
    "materialize_reference_hybrid_review",
    "materialize_deleted_source_layer_replacement_coverage_qualification",
    "materialize_source_layer_replacement_coverage_audit",
    "materialize_target_core_replacement_coverage_audit",
    "rasterize_triangle_depth",
    "rotate_triangles_about_axis",
    "seal_replacement_usd_depth_sweep_request",
    "validate_replacement_usd_depth_sweep_request",
]
