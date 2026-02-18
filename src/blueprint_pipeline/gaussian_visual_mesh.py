"""Robust visual mesh generation from Gaussian point-cloud artifacts.

This module is intentionally best-effort and dependency-aware:
- If Open3D is available, it reconstructs a triangle mesh from the Gaussian PLY.
- If dependencies are missing, callers receive a non-fatal "ok=false" report.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_gaussian_visual_mesh(*, gaussian_ply: Path, output_glb: Path, target_faces: int) -> Dict[str, Any]:
    """Build a viewer-friendly mesh from Gaussian PLY.

    Note: current implementation reconstructs from Gaussian samples and writes
    vertex-colored GLB. This is a robust fallback path for generic viewers.
    """

    try:
        import numpy as np
        import open3d as o3d
    except Exception as exc:
        return {
            "ok": False,
            "method": "gaussian_tsdf",
            "reason": f"open3d_unavailable:{exc}",
        }

    if not gaussian_ply.is_file() or gaussian_ply.stat().st_size <= 0:
        return {
            "ok": False,
            "method": "gaussian_tsdf",
            "reason": f"missing_gaussian_ply:{gaussian_ply}",
        }

    try:
        pcd = o3d.io.read_point_cloud(str(gaussian_ply))
        points_before = len(pcd.points)
        if points_before <= 0:
            return {
                "ok": False,
                "method": "gaussian_tsdf",
                "reason": "gaussian_pointcloud_empty",
            }

        max_points = max(100000, _env_int("GAUSSIAN_TSDF_MAX_POINTS", 900000))
        if points_before > max_points:
            ratio = max(0.05, min(1.0, float(max_points) / float(points_before)))
            pcd = pcd.random_down_sample(ratio)

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=64)
            )
            pcd.orient_normals_consistent_tangent_plane(50)

        depth = max(7, min(12, _env_int("GAUSSIAN_TSDF_POISSON_DEPTH", 10)))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=0,
            scale=1.1,
            linear_fit=False,
        )
        if len(mesh.triangles) <= 0:
            return {
                "ok": False,
                "method": "gaussian_tsdf",
                "reason": "poisson_empty_mesh",
            }

        keep_quantile = min(0.2, max(0.0, _env_float("GAUSSIAN_TSDF_DENSITY_QUANTILE", 0.02)))
        if keep_quantile > 0.0:
            dens = np.asarray(densities)
            threshold = np.quantile(dens, keep_quantile)
            mesh.remove_vertices_by_mask(dens < threshold)

        if target_faces > 0 and len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_faces)

        # Transfer nearest-neighbor color from point cloud to mesh vertices.
        if pcd.has_colors() and len(pcd.colors) > 0 and len(mesh.vertices) > 0:
            tree = o3d.geometry.KDTreeFlann(pcd)
            pcd_colors = np.asarray(pcd.colors)
            verts = np.asarray(mesh.vertices)
            colors = np.zeros((len(verts), 3), dtype=np.float64)
            for i, v in enumerate(verts):
                _, idx, _ = tree.search_knn_vector_3d(v, 1)
                colors[i] = pcd_colors[idx[0]]
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        output_glb.parent.mkdir(parents=True, exist_ok=True)
        wrote = o3d.io.write_triangle_mesh(str(output_glb), mesh, write_vertex_colors=True)
        if not wrote:
            return {
                "ok": False,
                "method": "gaussian_tsdf",
                "reason": f"write_failed:{output_glb}",
            }

        return {
            "ok": True,
            "method": "gaussian_tsdf_open3d",
            "path": str(output_glb),
            "faces": int(len(mesh.triangles)),
            "points_input": int(points_before),
            "points_used": int(len(pcd.points)),
            "target_faces": int(target_faces),
        }
    except Exception as exc:
        return {
            "ok": False,
            "method": "gaussian_tsdf",
            "reason": f"gaussian_mesh_failed:{exc}",
        }
