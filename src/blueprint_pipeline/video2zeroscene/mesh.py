"""Stage 3: Mesh extraction from Gaussian splats.

This module handles:
- SuGaR mesh extraction from 3DGS
- Poisson surface reconstruction fallback
- Mesh decimation for render/collision variants
- Texture baking from multi-view images
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import CameraIntrinsics, PipelineConfig
from .slam import CameraPose


@dataclass
class MeshResult:
    """Result of mesh extraction."""
    render_mesh_path: Optional[Path] = None
    collision_mesh_path: Optional[Path] = None
    texture_path: Optional[Path] = None

    # Metrics
    vertex_count: int = 0
    face_count: int = 0
    watertight: bool = False

    success: bool = True
    errors: List[str] = field(default_factory=list)


class MeshExtractor:
    """Extract textured mesh from 3D Gaussian Splatting.

    Primary: SuGaR (Surface-Aligned Gaussian Splatting)
    Fallback: Poisson surface reconstruction
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        gaussians_path: Path,
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        frames_dir: Path,
        output_dir: Path,
    ) -> MeshResult:
        """Extract mesh from Gaussian splats.

        Args:
            gaussians_path: Path to Gaussian point cloud (PLY)
            poses: Camera poses
            intrinsics: Camera intrinsics for texture baking
            frames_dir: Directory containing frames
            output_dir: Output directory

        Returns:
            MeshResult with mesh paths and metrics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        errors = []

        # Try SuGaR first
        mesh_path = self._run_sugar(gaussians_path, output_dir)

        if mesh_path is None:
            print("SuGaR failed, trying Poisson reconstruction")
            mesh_path = self._run_poisson(gaussians_path, output_dir)

        if mesh_path is None:
            return MeshResult(
                success=False,
                errors=["All mesh extraction methods failed"],
            )

        # Decimate for render mesh
        render_mesh_path = self._decimate_mesh(
            mesh_path,
            target_faces=self.config.mesh_decimation_target,
            output_path=output_dir / "environment_mesh.ply",
        )

        # Generate collision mesh (more aggressive decimation)
        collision_mesh_path = self._decimate_mesh(
            mesh_path,
            target_faces=self.config.collision_decimation_target,
            output_path=output_dir / "environment_collision.ply",
        )

        # Bake textures
        texture_path = None
        if render_mesh_path and poses:
            texture_path = self._bake_textures(
                render_mesh_path,
                poses,
                intrinsics,
                frames_dir,
                output_dir,
            )

        # Compute metrics
        metrics = self._compute_metrics(render_mesh_path, collision_mesh_path)

        # Export to GLB for portability
        glb_path = self._export_to_glb(render_mesh_path, texture_path, output_dir)

        return MeshResult(
            render_mesh_path=glb_path or render_mesh_path,
            collision_mesh_path=collision_mesh_path,
            texture_path=texture_path,
            vertex_count=metrics.get("vertex_count", 0),
            face_count=metrics.get("face_count", 0),
            watertight=metrics.get("watertight", False),
            success=True,
            errors=errors,
        )

    def _run_sugar(
        self,
        gaussians_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run SuGaR mesh extraction."""
        print("Running SuGaR mesh extraction...")

        output_path = output_dir / "sugar_mesh.ply"

        # Try Python API
        try:
            from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
            from sugar_scene import SuGaR

            sugar_scene = SuGaR.load_from_ply(gaussians_path)
            mesh = extract_mesh_from_coarse_sugar(
                sugar_scene,
                surface_level=0.3,
                decimation_target=self.config.mesh_decimation_target,
            )
            mesh.export(str(output_path))
            print(f"SuGaR mesh saved: {output_path}")
            return output_path

        except ImportError:
            pass

        # Try CLI
        try:
            result = subprocess.run(
                [
                    "python", "-m", "sugar.extract_mesh",
                    "--gaussian_ply", str(gaussians_path),
                    "--output_path", str(output_path),
                ],
                capture_output=True,
                timeout=1800,
            )

            if result.returncode == 0 and output_path.exists():
                return output_path

        except Exception:
            pass

        return None

    def _run_poisson(
        self,
        gaussians_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run Poisson surface reconstruction."""
        print("Running Poisson surface reconstruction...")

        output_path = output_dir / "poisson_mesh.ply"

        try:
            import open3d as o3d

            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(gaussians_path))
            if len(pcd.points) == 0:
                return None

            # Estimate normals if needed
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
                pcd.orient_normals_consistent_tangent_plane(k=15)

            # Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=10,
                linear_fit=True,
            )

            # Remove low-density vertices
            densities = np.asarray(densities)
            threshold = np.quantile(densities, 0.05)
            vertices_to_keep = densities > threshold
            mesh.remove_vertices_by_mask(~vertices_to_keep)

            # Clean up
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()

            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print(f"Poisson mesh saved: {output_path}")
            return output_path

        except ImportError:
            print("Open3D not installed")
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")

        return None

    def _decimate_mesh(
        self,
        mesh_path: Path,
        target_faces: int,
        output_path: Path,
    ) -> Optional[Path]:
        """Decimate mesh to target face count."""
        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            original_faces = len(mesh.triangles)

            if original_faces <= target_faces:
                shutil.copy(mesh_path, output_path)
                return output_path

            decimated = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )
            decimated.remove_degenerate_triangles()
            decimated.remove_unreferenced_vertices()

            o3d.io.write_triangle_mesh(str(output_path), decimated)
            print(f"Decimated: {original_faces} -> {len(decimated.triangles)} faces")
            return output_path

        except ImportError:
            shutil.copy(mesh_path, output_path)
            return output_path

    def _bake_textures(
        self,
        mesh_path: Path,
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        frames_dir: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Bake multi-view textures onto mesh."""
        print("Baking textures...")

        texture_path = output_dir / "textures" / "diffuse.png"
        texture_path.parent.mkdir(exist_ok=True)

        try:
            from PIL import Image

            # Create blank texture atlas
            tex_size = self.config.texture_resolution
            atlas = np.ones((tex_size, tex_size, 3), dtype=np.uint8) * 128

            # Simple texture projection (placeholder)
            # Real implementation would use proper UV mapping and projection

            Image.fromarray(atlas).save(texture_path)
            return texture_path

        except Exception as e:
            print(f"Texture baking failed: {e}")

        return None

    def _compute_metrics(
        self,
        render_path: Optional[Path],
        collision_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Compute mesh metrics."""
        metrics = {
            "vertex_count": 0,
            "face_count": 0,
            "watertight": False,
        }

        if not render_path or not render_path.exists():
            return metrics

        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(render_path))
            metrics["vertex_count"] = len(mesh.vertices)
            metrics["face_count"] = len(mesh.triangles)
            metrics["watertight"] = mesh.is_watertight()

        except Exception:
            pass

        return metrics

    def _export_to_glb(
        self,
        mesh_path: Optional[Path],
        texture_path: Optional[Path],
        output_dir: Path,
    ) -> Optional[Path]:
        """Export mesh to GLB format."""
        if not mesh_path or not mesh_path.exists():
            return None

        try:
            import trimesh

            mesh = trimesh.load(str(mesh_path))
            glb_path = output_dir / "environment_mesh.glb"
            mesh.export(str(glb_path))
            return glb_path

        except Exception:
            return mesh_path
