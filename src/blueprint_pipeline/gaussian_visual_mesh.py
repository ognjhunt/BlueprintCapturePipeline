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


def _apply_open3d_thread_overrides() -> None:
    thread_count = max(0, _env_int("OPEN3D_CPU_THREADS", 0))
    if thread_count <= 0:
        return
    value = str(thread_count)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value


def build_gaussian_visual_mesh(*, gaussian_ply: Path, output_glb: Path, target_faces: int) -> Dict[str, Any]:
    """Build a viewer-friendly mesh from Gaussian PLY.

    Quality strategy:
    - Voxel downsampling (not random) to preserve spatial coverage.
    - Keep up to 2M points for room-scale detail.
    - Poisson depth 12 for high-resolution reconstruction.
    - Weighted KNN color transfer (K=5) for smooth vertex colors.
    """

    _apply_open3d_thread_overrides()
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

        max_points = max(100000, _env_int("GAUSSIAN_TSDF_MAX_POINTS", 2000000))
        if points_before > max_points:
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            volume = float(extent[0] * extent[1] * extent[2])
            voxel_size = max(0.001, (volume / max_points) ** (1.0 / 3.0))
            pcd = pcd.voxel_down_sample(voxel_size)

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=64)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        depth = max(7, min(13, _env_int("GAUSSIAN_TSDF_POISSON_DEPTH", 12)))
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

        # Weighted KNN color transfer (K=5) for smooth vertex colors.
        knn_k = max(1, _env_int("GAUSSIAN_TSDF_COLOR_KNN", 5))
        if pcd.has_colors() and len(pcd.colors) > 0 and len(mesh.vertices) > 0:
            tree = o3d.geometry.KDTreeFlann(pcd)
            pcd_colors = np.asarray(pcd.colors)
            verts = np.asarray(mesh.vertices)
            colors = np.zeros((len(verts), 3), dtype=np.float64)
            for i, v in enumerate(verts):
                _, idx, dist = tree.search_knn_vector_3d(v, knn_k)
                if knn_k == 1 or len(idx) <= 1:
                    colors[i] = pcd_colors[idx[0]]
                else:
                    dist_arr = np.asarray(dist, dtype=np.float64)
                    weights = 1.0 / np.maximum(dist_arr, 1e-12)
                    weights /= weights.sum()
                    colors[i] = (pcd_colors[idx] * weights[:, None]).sum(axis=0)
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
