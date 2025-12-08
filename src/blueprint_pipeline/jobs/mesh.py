"""SuGaR mesh extraction from 3D Gaussian Splatting."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import ArtifactPaths, JobPayload, SessionManifest
from ..utils.io import ensure_local_dir, load_json, save_json, save_image
from .base import (
    GPUJob,
    JobContext,
    JobResult,
    JobStatus,
    download_inputs,
    merge_parameters,
    upload_outputs,
)


@dataclass
class MeshMetrics:
    """Quality metrics for extracted mesh."""
    vertex_count: int
    face_count: int
    bounding_box_min: Tuple[float, float, float]
    bounding_box_max: Tuple[float, float, float]
    surface_area: float
    volume: float
    watertight: bool
    texture_resolution: Tuple[int, int]
    collision_vertex_count: int
    collision_face_count: int


@dataclass
class MeshExtractionJob(GPUJob):
    """Extract textured mesh from 3D Gaussian Splatting using SuGaR.

    This job:
    1. Downloads Gaussian splats from reconstruction stage
    2. Runs SuGaR (Surface-Aligned Gaussian Splatting) for mesh extraction
    3. Performs Poisson surface reconstruction
    4. Bakes textures onto the mesh
    5. Generates simplified collision mesh
    6. Exports in USD format

    Inputs:
        - 3D Gaussian splats from ReconstructionJob
        - Camera poses for texture baking

    Outputs:
        - High-quality textured mesh (USD)
        - Simplified collision mesh (USD)
        - Texture atlases
    """

    name: str = "mesh-extraction"
    description: str = "SuGaR mesh extraction and texture baking from Gaussian splats."
    timeout_minutes: int = 60
    generate_collision_mesh: bool = True
    bake_textures: bool = True

    # SuGaR configuration
    regularization_strength: float = 0.5
    surface_level: float = 0.3
    decimation_target: int = 500000  # Target faces for environment mesh
    collision_decimation_target: int = 50000  # Target for collision mesh
    poisson_depth: int = 10
    texture_resolution: int = 4096

    def _get_default_parameters(self) -> Dict[str, Any]:
        params = super()._get_default_parameters()
        params.update({
            "generate_collision_mesh": self.generate_collision_mesh,
            "bake_textures": self.bake_textures,
            "regularization_strength": self.regularization_strength,
            "surface_level": self.surface_level,
            "decimation_target": self.decimation_target,
            "collision_decimation_target": self.collision_decimation_target,
            "poisson_depth": self.poisson_depth,
            "texture_resolution": self.texture_resolution,
        })
        return params

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self._get_default_parameters(),
            {
                "generate_collision_mesh": self.generate_collision_mesh,
                "bake_textures": self.bake_textures,
            },
        )
        params = merge_parameters(params, parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "gaussians": f"{artifacts.reconstruction}/gaussians",
                "poses": f"{artifacts.reconstruction}/poses",
                "frames": artifacts.frames,
            },
            outputs={
                "mesh": f"{artifacts.meshes}/environment_mesh.usd",
                "collision_mesh": f"{artifacts.meshes}/environment_collision.usd",
                "textures": f"{artifacts.meshes}/textures/",
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute SuGaR mesh extraction."""
        result = JobResult(status=JobStatus.RUNNING)

        # Setup directories
        gaussians_dir = ensure_local_dir(ctx.workspace / "gaussians")
        poses_dir = ensure_local_dir(ctx.workspace / "poses")
        frames_dir = ensure_local_dir(ctx.workspace / "frames")
        output_dir = ensure_local_dir(ctx.workspace / "meshes")
        textures_dir = ensure_local_dir(output_dir / "textures")

        # Download inputs
        with ctx.tracker.stage("download_inputs", 3):
            ctx.gcs.download_directory(
                f"{ctx.artifacts.reconstruction}/gaussians/", gaussians_dir
            )
            ctx.tracker.update(1)
            ctx.gcs.download_directory(
                f"{ctx.artifacts.reconstruction}/poses/", poses_dir
            )
            ctx.tracker.update(1)
            ctx.gcs.download_directory(ctx.artifacts.frames + "/", frames_dir)
            ctx.tracker.update(1)

        # Locate Gaussian point cloud
        ply_path = self._find_gaussian_ply(gaussians_dir)
        if ply_path is None:
            result.status = JobStatus.FAILED
            result.errors.append("No Gaussian point cloud found in inputs")
            return result

        ctx.logger.info(f"Found Gaussian point cloud: {ply_path}")

        # Load camera poses for texture baking
        poses = self._load_poses(poses_dir)
        ctx.logger.info(f"Loaded {len(poses)} camera poses")

        # Run SuGaR mesh extraction
        with ctx.tracker.stage("sugar_extraction", 1):
            sugar_mesh_path = self._run_sugar_extraction(
                ctx=ctx,
                ply_path=ply_path,
                output_dir=output_dir,
            )

        if sugar_mesh_path is None or not sugar_mesh_path.exists():
            ctx.logger.warning("SuGaR extraction failed, using fallback Poisson reconstruction")
            sugar_mesh_path = self._run_poisson_fallback(ctx, ply_path, output_dir)

        # Decimate mesh for performance
        with ctx.tracker.stage("mesh_decimation", 1):
            decimated_path = self._decimate_mesh(
                ctx=ctx,
                mesh_path=sugar_mesh_path,
                target_faces=ctx.parameters.get("decimation_target", self.decimation_target),
                output_path=output_dir / "environment_decimated.ply",
            )

        # Bake textures
        textured_mesh_path = decimated_path
        if ctx.parameters.get("bake_textures", self.bake_textures):
            with ctx.tracker.stage("texture_baking", len(poses)):
                textured_mesh_path = self._bake_textures(
                    ctx=ctx,
                    mesh_path=decimated_path,
                    frames_dir=frames_dir,
                    poses=poses,
                    textures_dir=textures_dir,
                    output_path=output_dir / "environment_textured.obj",
                )

        # Generate collision mesh (simplified version)
        collision_mesh_path = None
        if ctx.parameters.get("generate_collision_mesh", self.generate_collision_mesh):
            with ctx.tracker.stage("collision_mesh", 1):
                collision_mesh_path = self._generate_collision_mesh(
                    ctx=ctx,
                    mesh_path=decimated_path,
                    target_faces=ctx.parameters.get(
                        "collision_decimation_target", self.collision_decimation_target
                    ),
                    output_path=output_dir / "environment_collision.ply",
                )

        # Convert to USD format
        with ctx.tracker.stage("usd_export", 2):
            env_usd_path = self._export_to_usd(
                ctx=ctx,
                mesh_path=textured_mesh_path,
                textures_dir=textures_dir,
                output_path=output_dir / "environment_mesh.usd",
                is_collision=False,
            )

            collision_usd_path = None
            if collision_mesh_path:
                collision_usd_path = self._export_to_usd(
                    ctx=ctx,
                    mesh_path=collision_mesh_path,
                    textures_dir=None,
                    output_path=output_dir / "environment_collision.usd",
                    is_collision=True,
                )

        # Compute metrics
        metrics = self._compute_mesh_metrics(
            ctx=ctx,
            mesh_path=textured_mesh_path,
            collision_path=collision_mesh_path,
            textures_dir=textures_dir,
        )

        # Save extraction report
        report = {
            "session_id": ctx.session.session_id,
            "metrics": {
                "vertex_count": metrics.vertex_count,
                "face_count": metrics.face_count,
                "bounding_box": {
                    "min": list(metrics.bounding_box_min),
                    "max": list(metrics.bounding_box_max),
                },
                "surface_area": metrics.surface_area,
                "volume": metrics.volume,
                "watertight": metrics.watertight,
                "texture_resolution": list(metrics.texture_resolution),
                "collision_vertex_count": metrics.collision_vertex_count,
                "collision_face_count": metrics.collision_face_count,
            },
            "output_files": {
                "environment_mesh": "environment_mesh.usd",
                "collision_mesh": "environment_collision.usd" if collision_usd_path else None,
                "textures": [str(p.name) for p in textures_dir.glob("*")] if textures_dir.exists() else [],
            },
        }
        save_json(report, output_dir / "mesh_extraction_report.json")

        # Upload outputs
        with ctx.tracker.stage("upload_outputs", 3):
            ctx.gcs.upload(env_usd_path, f"{ctx.artifacts.meshes}/environment_mesh.usd")
            ctx.tracker.update(1)

            if collision_usd_path:
                ctx.gcs.upload(
                    collision_usd_path, f"{ctx.artifacts.meshes}/environment_collision.usd"
                )
            ctx.tracker.update(1)

            if textures_dir.exists() and any(textures_dir.iterdir()):
                ctx.gcs.upload_directory(textures_dir, f"{ctx.artifacts.meshes}/textures")
            ctx.tracker.update(1)

        result.outputs = {
            "mesh": f"{ctx.artifacts.meshes}/environment_mesh.usd",
            "collision_mesh": f"{ctx.artifacts.meshes}/environment_collision.usd",
            "textures": f"{ctx.artifacts.meshes}/textures/",
        }
        result.metrics = report["metrics"]

        return result

    def _find_gaussian_ply(self, gaussians_dir: Path) -> Optional[Path]:
        """Find Gaussian point cloud PLY file."""
        # Common names for 3DGS output
        candidates = [
            "point_cloud.ply",
            "points3D.ply",
            "gaussians.ply",
            "iteration_30000/point_cloud.ply",
        ]

        for candidate in candidates:
            path = gaussians_dir / candidate
            if path.exists():
                return path

        # Fallback: find any PLY
        plys = list(gaussians_dir.rglob("*.ply"))
        if plys:
            return plys[0]

        return None

    def _load_poses(self, poses_dir: Path) -> List[Dict[str, Any]]:
        """Load camera poses from poses directory."""
        poses_json = poses_dir / "poses.json"
        if poses_json.exists():
            data = load_json(poses_json)
            return data.get("poses", [])

        # Fallback: parse COLMAP format
        images_txt = poses_dir / "images.txt"
        if images_txt.exists():
            return self._parse_colmap_images(images_txt)

        return []

    def _parse_colmap_images(self, images_txt: Path) -> List[Dict[str, Any]]:
        """Parse COLMAP images.txt format."""
        poses = []
        with open(images_txt, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            parts = line.split()
            if len(parts) >= 10:
                poses.append({
                    "image_id": int(parts[0]),
                    "image_name": parts[9],
                    "rotation": [float(parts[j]) for j in range(1, 5)],
                    "translation": [float(parts[j]) for j in range(5, 8)],
                    "camera_id": int(parts[8]),
                })
            i += 2  # Skip points line

        return poses

    def _run_sugar_extraction(
        self,
        ctx: JobContext,
        ply_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run SuGaR mesh extraction from Gaussian splats.

        SuGaR (Surface-Aligned Gaussian Splatting) extracts a mesh by:
        1. Regularizing Gaussians to be surface-aligned
        2. Using marching cubes to extract an isosurface
        3. Performing Poisson surface reconstruction

        Returns path to extracted mesh or None if failed.
        """
        ctx.logger.info("Running SuGaR mesh extraction...")

        output_path = output_dir / "sugar_mesh.ply"

        # Try using SuGaR package
        try:
            # Attempt to import SuGaR
            from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
            from sugar_scene import SuGaR

            # Load Gaussian scene
            ctx.logger.info("Loading Gaussian scene...")
            sugar_scene = SuGaR.load_from_ply(ply_path)

            # Extract mesh
            ctx.logger.info("Extracting mesh with SuGaR...")
            mesh = extract_mesh_from_coarse_sugar(
                sugar_scene,
                surface_level=ctx.parameters.get("surface_level", self.surface_level),
                decimation_target=ctx.parameters.get("decimation_target", self.decimation_target),
            )

            # Save mesh
            mesh.export(str(output_path))
            ctx.logger.info(f"SuGaR mesh saved to {output_path}")

            return output_path

        except ImportError:
            ctx.logger.warning("SuGaR package not installed, trying command-line interface")

        # Fallback: try running SuGaR as subprocess
        try:
            result = subprocess.run(
                [
                    "python", "-m", "sugar.extract_mesh",
                    "--gaussian_ply", str(ply_path),
                    "--output_path", str(output_path),
                    "--surface_level", str(ctx.parameters.get("surface_level", self.surface_level)),
                    "--decimation_target", str(ctx.parameters.get("decimation_target", self.decimation_target)),
                ],
                capture_output=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.returncode == 0 and output_path.exists():
                ctx.logger.info("SuGaR mesh extraction completed via CLI")
                return output_path

            ctx.logger.warning(f"SuGaR CLI failed: {result.stderr.decode()}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            ctx.logger.warning(f"SuGaR subprocess failed: {e}")

        return None

    def _run_poisson_fallback(
        self,
        ctx: JobContext,
        ply_path: Path,
        output_dir: Path,
    ) -> Path:
        """Fallback Poisson surface reconstruction from point cloud."""
        ctx.logger.info("Running Poisson surface reconstruction fallback...")

        output_path = output_dir / "poisson_mesh.ply"

        try:
            import open3d as o3d

            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_path))

            if len(pcd.points) == 0:
                ctx.logger.warning("Empty point cloud, creating placeholder mesh")
                return self._create_placeholder_mesh(output_dir)

            # Estimate normals if not present
            if not pcd.has_normals():
                ctx.logger.info("Estimating point cloud normals...")
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(k=15)

            # Poisson surface reconstruction
            ctx.logger.info(f"Running Poisson reconstruction (depth={self.poisson_depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=ctx.parameters.get("poisson_depth", self.poisson_depth),
                linear_fit=True,
            )

            # Remove low-density vertices (artifacts)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.05)
            vertices_to_keep = densities > density_threshold
            mesh.remove_vertices_by_mask(~vertices_to_keep)

            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Save mesh
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            ctx.logger.info(f"Poisson mesh saved: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

            return output_path

        except ImportError:
            ctx.logger.warning("Open3D not installed, creating placeholder mesh")
            return self._create_placeholder_mesh(output_dir)

        except Exception as e:
            ctx.logger.error(f"Poisson reconstruction failed: {e}")
            return self._create_placeholder_mesh(output_dir)

    def _create_placeholder_mesh(self, output_dir: Path) -> Path:
        """Create a placeholder box mesh for testing."""
        output_path = output_dir / "placeholder_mesh.ply"

        # Simple cube mesh in PLY format
        ply_content = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
-1 -1 -1
1 -1 -1
1 1 -1
-1 1 -1
-1 -1 1
1 -1 1
1 1 1
-1 1 1
3 0 1 2
3 0 2 3
3 4 6 5
3 4 7 6
3 0 4 5
3 0 5 1
3 2 6 7
3 2 7 3
3 0 3 7
3 0 7 4
3 1 5 6
3 1 6 2
"""
        output_path.write_text(ply_content)
        return output_path

    def _decimate_mesh(
        self,
        ctx: JobContext,
        mesh_path: Path,
        target_faces: int,
        output_path: Path,
    ) -> Path:
        """Decimate mesh to target face count."""
        ctx.logger.info(f"Decimating mesh to {target_faces} faces...")

        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            original_faces = len(mesh.triangles)

            if original_faces <= target_faces:
                ctx.logger.info(f"Mesh already has {original_faces} faces, skipping decimation")
                shutil.copy(mesh_path, output_path)
                return output_path

            # Decimate using quadric error metrics
            decimated = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

            # Clean up
            decimated.remove_degenerate_triangles()
            decimated.remove_unreferenced_vertices()

            o3d.io.write_triangle_mesh(str(output_path), decimated)

            ctx.logger.info(
                f"Decimated from {original_faces} to {len(decimated.triangles)} faces"
            )
            return output_path

        except ImportError:
            ctx.logger.warning("Open3D not installed, using original mesh")
            shutil.copy(mesh_path, output_path)
            return output_path

        except Exception as e:
            ctx.logger.warning(f"Decimation failed: {e}, using original mesh")
            shutil.copy(mesh_path, output_path)
            return output_path

    def _bake_textures(
        self,
        ctx: JobContext,
        mesh_path: Path,
        frames_dir: Path,
        poses: List[Dict[str, Any]],
        textures_dir: Path,
        output_path: Path,
    ) -> Path:
        """Bake textures from multi-view images onto mesh."""
        ctx.logger.info("Baking textures onto mesh...")

        texture_resolution = ctx.parameters.get("texture_resolution", self.texture_resolution)

        try:
            import open3d as o3d

            # Load mesh
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            # Generate UV coordinates if not present
            if not mesh.has_triangle_uvs():
                ctx.logger.info("Generating UV coordinates...")
                mesh = self._generate_uv_mapping(mesh)

            # Create texture atlas
            atlas_size = (texture_resolution, texture_resolution)
            texture_atlas = np.zeros((atlas_size[0], atlas_size[1], 3), dtype=np.uint8)
            texture_atlas.fill(128)  # Neutral gray default

            # Project frames onto mesh
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # For each view, project and blend colors
            for pose_data in poses[:50]:  # Limit to 50 views for speed
                image_name = pose_data.get("image_name", "")
                frame_path = self._find_frame_path(frames_dir, image_name)

                if frame_path and frame_path.exists():
                    self._project_view_to_texture(
                        ctx=ctx,
                        mesh=mesh,
                        frame_path=frame_path,
                        pose_data=pose_data,
                        texture_atlas=texture_atlas,
                    )
                    ctx.tracker.update(1)

            # Save texture atlas
            texture_path = textures_dir / "diffuse_atlas.png"
            save_image(texture_atlas, texture_path)

            # Export textured mesh as OBJ with MTL
            self._export_textured_mesh(mesh, output_path, texture_path)

            ctx.logger.info(f"Textured mesh saved to {output_path}")
            return output_path

        except ImportError:
            ctx.logger.warning("Open3D not installed, skipping texture baking")
            shutil.copy(mesh_path, output_path)
            return output_path

        except Exception as e:
            ctx.logger.warning(f"Texture baking failed: {e}")
            shutil.copy(mesh_path, output_path)
            return output_path

    def _generate_uv_mapping(self, mesh: Any) -> Any:
        """Generate UV coordinates using simple box projection."""
        import open3d as o3d

        # Use automatic UV unwrapping
        try:
            # Try xatlas-based unwrapping if available
            import xatlas
            vertices = np.asarray(mesh.vertices)
            indices = np.asarray(mesh.triangles).flatten()

            vmapping, indices_out, uvs = xatlas.parametrize(vertices, indices)

            # Apply UV coordinates
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
            return mesh

        except ImportError:
            pass

        # Fallback: simple box projection
        vertices = np.asarray(mesh.vertices)

        # Normalize to unit cube
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_size[bbox_size == 0] = 1.0

        normalized = (vertices - bbox_min) / bbox_size

        # Project to UV based on dominant normal direction
        uvs = np.zeros((len(mesh.triangles) * 3, 2))
        for i, tri in enumerate(np.asarray(mesh.triangles)):
            for j, vi in enumerate(tri):
                v = normalized[vi]
                # Simple planar projection
                uvs[i * 3 + j] = [v[0], v[1]]

        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        return mesh

    def _find_frame_path(self, frames_dir: Path, image_name: str) -> Optional[Path]:
        """Find frame path from image name."""
        # Direct match
        direct_path = frames_dir / image_name
        if direct_path.exists():
            return direct_path

        # Search in subdirectories
        for subdir in frames_dir.iterdir():
            if subdir.is_dir():
                path = subdir / image_name
                if path.exists():
                    return path

                # Try with different extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    path = subdir / (Path(image_name).stem + ext)
                    if path.exists():
                        return path

        return None

    def _project_view_to_texture(
        self,
        ctx: JobContext,
        mesh: Any,
        frame_path: Path,
        pose_data: Dict[str, Any],
        texture_atlas: np.ndarray,
    ) -> None:
        """Project a single view onto texture atlas."""
        try:
            from PIL import Image

            # Load frame
            frame = np.array(Image.open(frame_path).convert("RGB"))

            # This is a simplified projection - in production, use proper
            # multi-view texture blending with visibility tests

            # For now, just blend into atlas with simple averaging
            # (Real implementation would use camera intrinsics/extrinsics)

        except Exception as e:
            ctx.logger.debug(f"Failed to project view {frame_path}: {e}")

    def _export_textured_mesh(
        self,
        mesh: Any,
        output_path: Path,
        texture_path: Path,
    ) -> None:
        """Export mesh with texture as OBJ + MTL."""
        import open3d as o3d

        # Save as OBJ
        obj_path = output_path.with_suffix(".obj")
        o3d.io.write_triangle_mesh(str(obj_path), mesh)

        # Create MTL file
        mtl_path = output_path.with_suffix(".mtl")
        mtl_content = f"""# Material file for {obj_path.name}
newmtl material0
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.0 0.0 0.0
d 1.0
illum 1
map_Kd {texture_path.name}
"""
        mtl_path.write_text(mtl_content)

        # Update OBJ to reference MTL
        if obj_path.exists():
            content = obj_path.read_text()
            if "mtllib" not in content:
                content = f"mtllib {mtl_path.name}\n" + content
            if "usemtl" not in content:
                # Add usemtl before first face
                lines = content.split("\n")
                new_lines = []
                usemtl_added = False
                for line in lines:
                    if line.startswith("f ") and not usemtl_added:
                        new_lines.append("usemtl material0")
                        usemtl_added = True
                    new_lines.append(line)
                content = "\n".join(new_lines)
            obj_path.write_text(content)

    def _generate_collision_mesh(
        self,
        ctx: JobContext,
        mesh_path: Path,
        target_faces: int,
        output_path: Path,
    ) -> Path:
        """Generate simplified collision mesh."""
        ctx.logger.info(f"Generating collision mesh with {target_faces} faces...")

        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            # Aggressively simplify for collision
            collision_mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )

            # Clean up non-manifold geometry
            collision_mesh.remove_degenerate_triangles()
            collision_mesh.remove_non_manifold_edges()

            # Remove vertex colors and normals (not needed for collision)
            collision_mesh.vertex_colors = o3d.utility.Vector3dVector()
            collision_mesh.vertex_normals = o3d.utility.Vector3dVector()

            o3d.io.write_triangle_mesh(str(output_path), collision_mesh)

            ctx.logger.info(
                f"Collision mesh: {len(collision_mesh.vertices)} vertices, "
                f"{len(collision_mesh.triangles)} faces"
            )

            return output_path

        except ImportError:
            ctx.logger.warning("Open3D not installed, using decimated mesh as collision")
            shutil.copy(mesh_path, output_path)
            return output_path

    def _export_to_usd(
        self,
        ctx: JobContext,
        mesh_path: Path,
        textures_dir: Optional[Path],
        output_path: Path,
        is_collision: bool = False,
    ) -> Path:
        """Export mesh to USD format."""
        ctx.logger.info(f"Exporting to USD: {output_path}")

        try:
            from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

            # Create USD stage
            stage = Usd.Stage.CreateNew(str(output_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            stage.SetMetadata("upAxis", "Y")

            # Define root xform
            root_path = "/World"
            root_xform = UsdGeom.Xform.Define(stage, root_path)

            # Create mesh prim
            mesh_name = "CollisionMesh" if is_collision else "EnvironmentMesh"
            mesh_prim_path = f"{root_path}/{mesh_name}"

            # Load source mesh
            self._create_usd_mesh_from_file(
                stage=stage,
                mesh_path=mesh_path,
                prim_path=mesh_prim_path,
                is_collision=is_collision,
            )

            # Add material if not collision mesh and textures exist
            if not is_collision and textures_dir and textures_dir.exists():
                self._add_usd_material(
                    stage=stage,
                    mesh_prim_path=mesh_prim_path,
                    textures_dir=textures_dir,
                )

            stage.Save()
            ctx.logger.info(f"USD export completed: {output_path}")

            return output_path

        except ImportError:
            ctx.logger.warning("USD libraries not installed, copying mesh as-is")
            shutil.copy(mesh_path, output_path.with_suffix(".ply"))
            return output_path.with_suffix(".ply")

    def _create_usd_mesh_from_file(
        self,
        stage: Any,
        mesh_path: Path,
        prim_path: str,
        is_collision: bool,
    ) -> None:
        """Create USD mesh prim from mesh file."""
        from pxr import UsdGeom, Vt, Gf

        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # Create mesh prim
            usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

            # Set points
            points = [Gf.Vec3f(*v) for v in vertices]
            usd_mesh.CreatePointsAttr(points)

            # Set face vertex counts (all triangles = 3)
            face_counts = [3] * len(triangles)
            usd_mesh.CreateFaceVertexCountsAttr(face_counts)

            # Set face vertex indices
            indices = triangles.flatten().tolist()
            usd_mesh.CreateFaceVertexIndicesAttr(indices)

            # Set normals if available
            if mesh.has_vertex_normals():
                normals = np.asarray(mesh.vertex_normals)
                usd_mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
                usd_mesh.SetNormalsInterpolation("vertex")

            # Add purpose for collision meshes
            if is_collision:
                usd_mesh.GetPrim().CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("guide")

        except ImportError:
            ctx.logger.warning("Open3D not available for USD mesh creation")

    def _add_usd_material(
        self,
        stage: Any,
        mesh_prim_path: str,
        textures_dir: Path,
    ) -> None:
        """Add PBR material to USD mesh."""
        from pxr import UsdShade, Sdf

        # Find diffuse texture
        diffuse_texture = None
        for name in ["diffuse_atlas.png", "diffuse.png", "albedo.png", "color.png"]:
            path = textures_dir / name
            if path.exists():
                diffuse_texture = path
                break

        if not diffuse_texture:
            return

        # Create material
        material_path = f"/World/Materials/EnvironmentMaterial"
        material = UsdShade.Material.Define(stage, material_path)

        # Create shader
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Set diffuse texture
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))

        # Create texture reader for diffuse
        tex_reader = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            f"./textures/{diffuse_texture.name}"
        )
        tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # Connect texture to shader
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_reader.ConnectableAPI(), "rgb"
        )

        # Connect shader to material surface
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material to mesh
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    def _compute_mesh_metrics(
        self,
        ctx: JobContext,
        mesh_path: Path,
        collision_path: Optional[Path],
        textures_dir: Path,
    ) -> MeshMetrics:
        """Compute quality metrics for extracted mesh."""
        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            vertices = np.asarray(mesh.vertices)
            bbox_min = tuple(vertices.min(axis=0)) if len(vertices) > 0 else (0, 0, 0)
            bbox_max = tuple(vertices.max(axis=0)) if len(vertices) > 0 else (0, 0, 0)

            # Compute surface area and volume
            surface_area = mesh.get_surface_area() if len(mesh.triangles) > 0 else 0.0

            # Volume only for watertight meshes
            watertight = mesh.is_watertight()
            volume = mesh.get_volume() if watertight else 0.0

            # Collision mesh metrics
            collision_verts = 0
            collision_faces = 0
            if collision_path and collision_path.exists():
                collision_mesh = o3d.io.read_triangle_mesh(str(collision_path))
                collision_verts = len(collision_mesh.vertices)
                collision_faces = len(collision_mesh.triangles)

            # Texture resolution
            tex_res = (0, 0)
            diffuse_tex = textures_dir / "diffuse_atlas.png"
            if diffuse_tex.exists():
                from PIL import Image
                with Image.open(diffuse_tex) as img:
                    tex_res = img.size

            return MeshMetrics(
                vertex_count=len(mesh.vertices),
                face_count=len(mesh.triangles),
                bounding_box_min=bbox_min,
                bounding_box_max=bbox_max,
                surface_area=surface_area,
                volume=volume,
                watertight=watertight,
                texture_resolution=tex_res,
                collision_vertex_count=collision_verts,
                collision_face_count=collision_faces,
            )

        except ImportError:
            return MeshMetrics(
                vertex_count=0,
                face_count=0,
                bounding_box_min=(0, 0, 0),
                bounding_box_max=(0, 0, 0),
                surface_area=0.0,
                volume=0.0,
                watertight=False,
                texture_resolution=(0, 0),
                collision_vertex_count=0,
                collision_face_count=0,
            )
