"""ZeroScene export for BlueprintPipeline handoff.

This module creates the ZeroScene-compatible folder structure that
BlueprintPipeline's zeroscene_adapter_job.py expects.

Structure:
    zeroscene/
        scene_info.json
        objects/obj_i/{mesh.glb, pose.json, bounds.json, material.json}
        background/mesh.glb
        camera/intrinsics.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interfaces import (
    CameraIntrinsics,
    CaptureManifest,
    ObjectAssetBundle,
    PipelineConfig,
    ZeroSceneBundle,
)
from .slam import CameraPose


@dataclass
class ExportResult:
    """Result of ZeroScene export."""
    bundle_path: Path
    scene_info_path: Path
    object_count: int
    success: bool = True
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ZeroSceneExporter:
    """Export pipeline results to ZeroScene format for BlueprintPipeline.

    This creates the folder structure expected by zeroscene_adapter_job.py
    in the BlueprintPipeline repository.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def export(
        self,
        manifest: CaptureManifest,
        background_mesh_path: Optional[Path],
        collision_mesh_path: Optional[Path],
        gaussians_path: Optional[Path],
        objects: List[ObjectAssetBundle],
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        output_dir: Path,
        scale_factor: float = 1.0,
    ) -> ExportResult:
        """Export to ZeroScene format.

        Args:
            manifest: Capture manifest
            background_mesh_path: Path to environment mesh
            collision_mesh_path: Path to collision mesh
            gaussians_path: Path to 3D Gaussians PLY (for DWM compatibility)
            objects: List of object asset bundles
            poses: Camera poses
            intrinsics: Camera intrinsics
            output_dir: Output directory
            scale_factor: Scale factor applied to scene

        Returns:
            ExportResult with bundle path and metadata
        """
        bundle_dir = output_dir / "zeroscene"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        errors = []

        # Create scene_info.json
        scene_info = self._create_scene_info(
            manifest,
            len(objects),
            background_mesh_path is not None,
            gaussians_path is not None and gaussians_path.exists(),
            scale_factor
        )
        scene_info_path = bundle_dir / "scene_info.json"
        scene_info_path.write_text(json.dumps(scene_info, indent=2))

        # Export objects
        objects_dir = bundle_dir / "objects"
        objects_dir.mkdir(exist_ok=True)

        for i, obj in enumerate(objects):
            obj_dir = objects_dir / f"obj_{i:04d}"
            obj_dir.mkdir(exist_ok=True)

            # Copy/link mesh
            mesh_exported = self._export_object_mesh(obj, obj_dir)
            if not mesh_exported:
                errors.append(f"Failed to export mesh for {obj.asset_id}")

            # Write pose.json
            pose_data = {
                "position": list(obj.position),
                "rotation": list(obj.rotation),
                "scale": list(obj.scale),
            }
            (obj_dir / "pose.json").write_text(json.dumps(pose_data, indent=2))

            # Write bounds.json
            bounds_data = {
                "min": list(obj.bounds_min),
                "max": list(obj.bounds_max),
            }
            (obj_dir / "bounds.json").write_text(json.dumps(bounds_data, indent=2))

            # Write material.json
            material_data = {
                "label": obj.concept_label,
                "tier": obj.tier.value,
                "source": obj.source,
                "quality_score": obj.quality_score,
            }
            (obj_dir / "material.json").write_text(json.dumps(material_data, indent=2))

        # Export background
        background_dir = bundle_dir / "background"
        background_dir.mkdir(exist_ok=True)

        if background_mesh_path and background_mesh_path.exists():
            self._copy_mesh(background_mesh_path, background_dir / "mesh.glb")

        if collision_mesh_path and collision_mesh_path.exists():
            self._copy_mesh(collision_mesh_path, background_dir / "collision.glb")

        # Export 3D Gaussians for DWM compatibility
        if gaussians_path and gaussians_path.exists():
            import shutil
            shutil.copy(gaussians_path, background_dir / "gaussians.ply")

        # Write background info
        bg_info = {
            "has_mesh": background_mesh_path is not None,
            "has_collision": collision_mesh_path is not None,
            "has_gaussians": gaussians_path is not None and gaussians_path.exists(),
            "gaussians_format": "3dgs_ply" if gaussians_path and gaussians_path.exists() else None,
        }
        (background_dir / "info.json").write_text(json.dumps(bg_info, indent=2))

        # Export camera info
        camera_dir = bundle_dir / "camera"
        camera_dir.mkdir(exist_ok=True)

        if intrinsics:
            intrinsics_data = {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "width": intrinsics.width,
                "height": intrinsics.height,
            }
            (camera_dir / "intrinsics.json").write_text(
                json.dumps(intrinsics_data, indent=2)
            )

        if poses:
            trajectory = [
                {
                    "frame_id": p.frame_id,
                    "rotation": list(p.rotation),
                    "translation": list(p.translation),
                    "timestamp": p.timestamp,
                }
                for p in poses
            ]
            (camera_dir / "trajectory.json").write_text(
                json.dumps(trajectory, indent=2)
            )

        # Write completion marker
        (bundle_dir / ".complete").touch()

        # Write BlueprintPipeline handoff marker
        handoff_info = {
            "source": "BlueprintCapturePipeline",
            "capture_id": manifest.capture_id,
            "format": "zeroscene",
            "version": "1.0",
            "ready_for_processing": True,
        }
        (bundle_dir / "handoff.json").write_text(json.dumps(handoff_info, indent=2))

        return ExportResult(
            bundle_path=bundle_dir,
            scene_info_path=scene_info_path,
            object_count=len(objects),
            success=len(errors) == 0,
            errors=errors if errors else None,
        )

    def _create_scene_info(
        self,
        manifest: CaptureManifest,
        object_count: int,
        has_background: bool,
        has_gaussians: bool,
        scale_factor: float,
    ) -> Dict[str, Any]:
        """Create scene_info.json content."""
        return {
            "capture_id": manifest.capture_id,
            "capture_timestamp": manifest.capture_timestamp,
            "device": {
                "platform": manifest.device_platform,
                "model": manifest.device_model,
            },
            "sensor": {
                "type": manifest.sensor_type.value,
                "has_depth": manifest.has_depth,
                "has_imu": manifest.has_imu,
            },
            "scale_factor": scale_factor,
            "up_axis": "Y",
            "meters_per_unit": 1.0,
            "object_count": object_count,
            "has_background": has_background,
            "has_gaussians": has_gaussians,
            "gaussians_format": "3dgs_ply" if has_gaussians else None,
            "dwm_compatible": has_gaussians,  # DWM requires raw Gaussians
            "resolution": list(manifest.resolution),
            "total_frames": manifest.total_frames,
            "duration_seconds": manifest.estimated_duration_seconds,
        }

    def _export_object_mesh(
        self,
        obj: ObjectAssetBundle,
        obj_dir: Path,
    ) -> bool:
        """Export object mesh to object directory."""
        mesh_path = Path(obj.mesh_path)

        if not mesh_path.exists():
            return False

        # Determine target filename
        if mesh_path.suffix == ".glb":
            target = obj_dir / "mesh.glb"
        elif mesh_path.suffix == ".gltf":
            target = obj_dir / "mesh.gltf"
        elif mesh_path.suffix == ".usd":
            target = obj_dir / "mesh.usd"
        else:
            # Try to convert to GLB
            target = obj_dir / "mesh.glb"
            try:
                import trimesh
                mesh = trimesh.load(str(mesh_path))
                mesh.export(str(target))
                return True
            except Exception:
                # Fall back to copying original
                target = obj_dir / f"mesh{mesh_path.suffix}"

        shutil.copy(mesh_path, target)
        return True

    def _copy_mesh(self, src: Path, dst: Path) -> bool:
        """Copy mesh file, converting if necessary."""
        if not src.exists():
            return False

        if src.suffix == dst.suffix:
            shutil.copy(src, dst)
            return True

        # Try conversion
        try:
            import trimesh
            mesh = trimesh.load(str(src))
            mesh.export(str(dst))
            return True
        except Exception:
            # Copy as-is with original extension
            shutil.copy(src, dst.with_suffix(src.suffix))
            return True


def create_zeroscene_bundle(
    capture_id: str,
    output_path: Path,
    background_mesh: Optional[str] = None,
    collision_mesh: Optional[str] = None,
    gaussians_path: Optional[str] = None,
    objects: Optional[List[ObjectAssetBundle]] = None,
    intrinsics: Optional[CameraIntrinsics] = None,
    poses: Optional[List[CameraPose]] = None,
    scale_factor: float = 1.0,
) -> ZeroSceneBundle:
    """Convenience function to create a ZeroSceneBundle.

    Args:
        capture_id: Unique capture identifier
        output_path: Output directory path
        background_mesh: Path to background mesh
        collision_mesh: Path to collision mesh
        gaussians_path: Path to 3D Gaussians PLY (for DWM compatibility)
        objects: List of object asset bundles
        intrinsics: Camera intrinsics
        poses: Camera trajectory
        scale_factor: Scale factor

    Returns:
        ZeroSceneBundle ready for writing
    """
    camera_trajectory = []
    if poses:
        camera_trajectory = [
            {
                "frame_id": p.frame_id,
                "rotation": list(p.rotation),
                "translation": list(p.translation),
                "timestamp": p.timestamp,
            }
            for p in poses
        ]

    return ZeroSceneBundle(
        capture_id=capture_id,
        output_path=output_path,
        scene_info={
            "scale_factor": scale_factor,
        },
        background_mesh_path=background_mesh,
        background_collision_path=collision_mesh,
        objects=objects or [],
        intrinsics=intrinsics,
        camera_trajectory=camera_trajectory,
        scale_factor=scale_factor,
    )
