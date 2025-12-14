"""Stage 7: Tiered object assetization.

This module implements three tiers of object asset generation:
- Tier 1: Multi-view reconstruction (when coverage is good)
- Tier 2: Proxy geometry (box/capsule/convex hull)
- Tier 3: Asset replacement/retrieval (LiteReality-style)

Plus AI fallback:
- Hunyuan3D generation for low-coverage objects
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import (
    AssetizationTier,
    CameraIntrinsics,
    ObjectAssetBundle,
    ObjectProposal,
    PipelineConfig,
    TrackInfo,
)
from .slam import CameraPose


@dataclass
class AssetizationResult:
    """Result of object assetization."""
    assets: List[ObjectAssetBundle]
    success: bool = True
    errors: List[str] = field(default_factory=list)


class ObjectAssetizer:
    """Generate simulatable object assets from proposals.

    Implements tiered strategy:
    - Tier 1: Reconstruct from multi-view when we have good coverage
    - Tier 2: Generate proxy geometry for fast/robust simulation
    - Tier 3: Replace with curated assets (future)
    - Fallback: AI generation with Hunyuan3D
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        proposals: List[ObjectProposal],
        tracks: List[TrackInfo],
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
    ) -> AssetizationResult:
        """Generate assets for all object proposals.

        Args:
            proposals: 3D object proposals from lifting
            tracks: Original 2D tracks (for image crops)
            poses: Camera poses
            intrinsics: Camera intrinsics
            frames_dir: Directory containing frames
            masks_dir: Directory containing masks
            output_dir: Output directory for assets

        Returns:
            AssetizationResult with generated assets
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        assets = []
        errors = []

        # Build track lookup
        track_by_id = {t.track_id: t for t in tracks}

        for proposal in proposals:
            track = track_by_id.get(proposal.track_id)
            if not track:
                continue

            print(f"Assetizing {proposal.proposal_id} "
                  f"(tier: {proposal.recommended_tier.value})")

            obj_dir = output_dir / proposal.proposal_id
            obj_dir.mkdir(exist_ok=True)

            asset = None
            tier = proposal.recommended_tier

            # Try recommended tier first
            if tier == AssetizationTier.TIER_1_RECONSTRUCT:
                asset = self._tier1_reconstruct(
                    proposal, track, poses, intrinsics,
                    frames_dir, masks_dir, obj_dir
                )

            if asset is None and tier in (
                AssetizationTier.TIER_1_RECONSTRUCT,
                AssetizationTier.TIER_2_PROXY
            ):
                # Fall back to proxy
                asset = self._tier2_proxy(proposal, obj_dir)

            if asset is None and self.config.enable_hunyuan3d:
                # Fall back to AI generation
                asset = self._hunyuan3d_generate(
                    proposal, track, frames_dir, masks_dir, obj_dir
                )

            if asset is None:
                # Last resort: minimal proxy
                asset = self._minimal_proxy(proposal, obj_dir)

            if asset:
                assets.append(asset)

        # Save asset manifest
        self._save_manifest(assets, output_dir)

        return AssetizationResult(assets=assets, errors=errors)

    def _tier1_reconstruct(
        self,
        proposal: ObjectProposal,
        track: TrackInfo,
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
    ) -> Optional[ObjectAssetBundle]:
        """Tier 1: Reconstruct object from multi-view images."""
        print(f"  Attempting Tier 1 reconstruction...")

        # Extract crops for this object
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

        crops = self._extract_crops(track, frames_dir, masks_dir, crops_dir)
        if len(crops) < self.config.min_object_views:
            print(f"  Not enough crops ({len(crops)}), skipping reconstruction")
            return None

        # Try 3DGS-based object reconstruction
        mesh_path = self._run_object_3dgs(crops_dir, output_dir)

        if mesh_path is None:
            # Try photogrammetry fallback
            mesh_path = self._run_photogrammetry(crops_dir, output_dir)

        if mesh_path is None:
            return None

        # Create collision mesh
        collision_path = self._generate_collision(mesh_path, output_dir)

        # Export to GLB
        glb_path = self._export_to_glb(mesh_path, output_dir / f"{proposal.proposal_id}.glb")

        return ObjectAssetBundle(
            asset_id=proposal.proposal_id,
            proposal_id=proposal.proposal_id,
            concept_label=proposal.concept_label,
            mesh_path=str(glb_path),
            collision_path=str(collision_path) if collision_path else None,
            tier=AssetizationTier.TIER_1_RECONSTRUCT,
            source="reconstruction",
            position=proposal.position,
            rotation=proposal.rotation,
            bounds_min=tuple(
                np.array(proposal.obb_center) - np.array(proposal.obb_extents) / 2
            ),
            bounds_max=tuple(
                np.array(proposal.obb_center) + np.array(proposal.obb_extents) / 2
            ),
            quality_score=proposal.coverage_score,
        )

    def _tier2_proxy(
        self,
        proposal: ObjectProposal,
        output_dir: Path,
    ) -> Optional[ObjectAssetBundle]:
        """Tier 2: Generate proxy geometry (box, capsule, or convex hull)."""
        print(f"  Generating Tier 2 proxy geometry...")

        # Use oriented bounding box as proxy
        extents = np.array(proposal.obb_extents)
        extents = np.maximum(extents, 0.01)  # Minimum size

        # Create box mesh
        mesh_path = output_dir / f"{proposal.proposal_id}_proxy.obj"
        self._create_box_mesh(extents, mesh_path)

        # Export to GLB
        glb_path = output_dir / f"{proposal.proposal_id}.glb"
        self._export_to_glb(mesh_path, glb_path)

        return ObjectAssetBundle(
            asset_id=proposal.proposal_id,
            proposal_id=proposal.proposal_id,
            concept_label=proposal.concept_label,
            mesh_path=str(glb_path),
            collision_path=str(mesh_path),  # Box is its own collision
            tier=AssetizationTier.TIER_2_PROXY,
            source="proxy",
            position=proposal.position,
            rotation=proposal.rotation,
            bounds_min=tuple(-extents / 2),
            bounds_max=tuple(extents / 2),
            quality_score=0.5,
        )

    def _hunyuan3d_generate(
        self,
        proposal: ObjectProposal,
        track: TrackInfo,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
    ) -> Optional[ObjectAssetBundle]:
        """Generate object using Hunyuan3D from best crop image."""
        print(f"  Attempting Hunyuan3D generation...")

        # Get best crop
        crops_dir = output_dir / "crops"
        if not crops_dir.exists():
            crops_dir.mkdir(exist_ok=True)
            self._extract_crops(track, frames_dir, masks_dir, crops_dir)

        crop_paths = sorted(crops_dir.glob("*.png"))
        if not crop_paths:
            return None

        # Use middle crop (often best angle)
        best_crop = crop_paths[len(crop_paths) // 2]

        # Run Hunyuan3D
        mesh_path = self._run_hunyuan3d(best_crop, output_dir)

        if mesh_path is None:
            return None

        # Export to GLB
        glb_path = output_dir / f"{proposal.proposal_id}.glb"
        self._export_to_glb(mesh_path, glb_path)

        # Generate collision from mesh
        collision_path = self._generate_collision(mesh_path, output_dir)

        return ObjectAssetBundle(
            asset_id=proposal.proposal_id,
            proposal_id=proposal.proposal_id,
            concept_label=proposal.concept_label,
            mesh_path=str(glb_path),
            collision_path=str(collision_path) if collision_path else None,
            tier=AssetizationTier.TIER_2_PROXY,  # Generated is still tier 2
            source="hunyuan3d",
            position=proposal.position,
            rotation=proposal.rotation,
            bounds_min=tuple(
                np.array(proposal.obb_center) - np.array(proposal.obb_extents) / 2
            ),
            bounds_max=tuple(
                np.array(proposal.obb_center) + np.array(proposal.obb_extents) / 2
            ),
            quality_score=0.6,
        )

    def _minimal_proxy(
        self,
        proposal: ObjectProposal,
        output_dir: Path,
    ) -> ObjectAssetBundle:
        """Create minimal placeholder proxy."""
        print(f"  Creating minimal proxy...")

        extents = np.array(proposal.obb_extents)
        extents = np.maximum(extents, 0.05)

        mesh_path = output_dir / f"{proposal.proposal_id}_minimal.obj"
        self._create_box_mesh(extents, mesh_path)

        return ObjectAssetBundle(
            asset_id=proposal.proposal_id,
            proposal_id=proposal.proposal_id,
            concept_label=proposal.concept_label,
            mesh_path=str(mesh_path),
            collision_path=str(mesh_path),
            tier=AssetizationTier.TIER_2_PROXY,
            source="minimal_proxy",
            position=proposal.position,
            rotation=proposal.rotation,
            bounds_min=tuple(-extents / 2),
            bounds_max=tuple(extents / 2),
            quality_score=0.3,
        )

    def _extract_crops(
        self,
        track: TrackInfo,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
    ) -> List[Path]:
        """Extract masked crops for an object."""
        try:
            from PIL import Image
        except ImportError:
            return []

        crops = []

        for i, (frame_id, bbox, mask_path) in enumerate(
            zip(track.frame_ids, track.bboxes, track.mask_paths)
        ):
            # Find frame
            frame_path = None
            for ext in [".png", ".jpg"]:
                candidate = frames_dir.parent / f"{frame_id}{ext}"
                if candidate.exists():
                    frame_path = candidate
                    break

            if not frame_path:
                # Search in subdirectories
                for subdir in frames_dir.rglob("*"):
                    if subdir.is_file() and frame_id in subdir.stem:
                        frame_path = subdir
                        break

            if not frame_path or not frame_path.exists():
                continue

            try:
                frame = np.array(Image.open(frame_path).convert("RGB"))

                # Apply mask if available
                mask_file = masks_dir / mask_path
                if mask_file.exists():
                    mask = np.array(Image.open(mask_file).convert("L"))
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = np.array(
                            Image.fromarray(mask).resize(
                                (frame.shape[1], frame.shape[0])
                            )
                        )
                    mask_3d = np.expand_dims(mask > 127, axis=-1)
                    frame = frame * mask_3d + (1 - mask_3d) * 255
                    frame = frame.astype(np.uint8)

                # Crop to bbox with padding
                x, y, w, h = bbox
                pad = int(max(w, h) * 0.15)
                x1 = max(0, int(x) - pad)
                y1 = max(0, int(y) - pad)
                x2 = min(frame.shape[1], int(x + w) + pad)
                y2 = min(frame.shape[0], int(y + h) + pad)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_path = output_dir / f"crop_{i:04d}.png"
                    Image.fromarray(crop).save(crop_path)
                    crops.append(crop_path)

            except Exception as e:
                print(f"  Failed to extract crop for {frame_id}: {e}")

        return crops

    def _run_object_3dgs(
        self,
        crops_dir: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run object-level 3DGS reconstruction."""
        # Placeholder - would integrate with instant-ngp or similar
        return None

    def _run_photogrammetry(
        self,
        crops_dir: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run photogrammetry reconstruction."""
        try:
            colmap_dir = output_dir / "colmap"
            colmap_dir.mkdir(exist_ok=True)

            # Feature extraction
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(colmap_dir / "db.db"),
                "--image_path", str(crops_dir),
            ], capture_output=True, timeout=300)

            # Matching
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(colmap_dir / "db.db"),
            ], capture_output=True, timeout=300)

            # Reconstruction
            sparse_dir = colmap_dir / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(colmap_dir / "db.db"),
                "--image_path", str(crops_dir),
                "--output_path", str(sparse_dir),
            ], capture_output=True, timeout=600)

            # Dense reconstruction
            dense_dir = colmap_dir / "dense"
            dense_dir.mkdir(exist_ok=True)
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(crops_dir),
                "--input_path", str(sparse_dir / "0"),
                "--output_path", str(dense_dir),
            ], capture_output=True, timeout=300)

            subprocess.run([
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense_dir),
            ], capture_output=True, timeout=600)

            mesh_path = output_dir / "mesh.ply"
            subprocess.run([
                "colmap", "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(mesh_path),
            ], capture_output=True, timeout=300)

            if mesh_path.exists():
                return mesh_path

        except Exception:
            pass

        return None

    def _run_hunyuan3d(
        self,
        image_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Run Hunyuan3D generation."""
        try:
            from hunyuan3d import Hunyuan3D2

            model = Hunyuan3D2()
            output_path = output_dir / "hunyuan_output.glb"

            model.generate(
                image_path=str(image_path),
                output_path=str(output_path),
            )

            if output_path.exists():
                return output_path

        except ImportError:
            pass

        # Try CLI
        try:
            output_path = output_dir / "hunyuan_output.glb"
            subprocess.run([
                "python", "-m", "hunyuan3d.generate",
                "--input", str(image_path),
                "--output", str(output_path),
            ], capture_output=True, timeout=600)

            if output_path.exists():
                return output_path

        except Exception:
            pass

        return None

    def _create_box_mesh(
        self,
        extents: np.ndarray,
        output_path: Path,
    ) -> None:
        """Create a box mesh in OBJ format."""
        dx, dy, dz = extents / 2

        vertices = [
            (-dx, -dy, -dz),
            (dx, -dy, -dz),
            (dx, dy, -dz),
            (-dx, dy, -dz),
            (-dx, -dy, dz),
            (dx, -dy, dz),
            (dx, dy, dz),
            (-dx, dy, dz),
        ]

        faces = [
            (1, 2, 3, 4),  # Back
            (5, 8, 7, 6),  # Front
            (1, 5, 6, 2),  # Bottom
            (3, 7, 8, 4),  # Top
            (1, 4, 8, 5),  # Left
            (2, 6, 7, 3),  # Right
        ]

        with open(output_path, "w") as f:
            f.write("# Box proxy mesh\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")

    def _generate_collision(
        self,
        mesh_path: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Generate simplified collision mesh."""
        try:
            import trimesh

            mesh = trimesh.load(str(mesh_path))

            # Simplify to convex hull
            if hasattr(mesh, "convex_hull"):
                collision_mesh = mesh.convex_hull
            else:
                collision_mesh = mesh

            collision_path = output_dir / "collision.obj"
            collision_mesh.export(str(collision_path))
            return collision_path

        except Exception:
            return None

    def _export_to_glb(
        self,
        mesh_path: Path,
        output_path: Path,
    ) -> Path:
        """Export mesh to GLB format."""
        try:
            import trimesh

            mesh = trimesh.load(str(mesh_path))
            mesh.export(str(output_path))
            return output_path

        except Exception:
            # Just copy if conversion fails
            if mesh_path.suffix == ".glb":
                shutil.copy(mesh_path, output_path)
            else:
                shutil.copy(mesh_path, output_path.with_suffix(mesh_path.suffix))
            return output_path

    def _save_manifest(
        self,
        assets: List[ObjectAssetBundle],
        output_dir: Path,
    ) -> None:
        """Save asset manifest."""
        manifest = [
            {
                "asset_id": a.asset_id,
                "proposal_id": a.proposal_id,
                "concept": a.concept_label,
                "mesh_path": a.mesh_path,
                "collision_path": a.collision_path,
                "tier": a.tier.value,
                "source": a.source,
                "position": list(a.position),
                "rotation": list(a.rotation),
                "bounds_min": list(a.bounds_min),
                "bounds_max": list(a.bounds_max),
                "quality_score": a.quality_score,
            }
            for a in assets
        ]
        (output_dir / "assets_manifest.json").write_text(
            json.dumps({"assets": manifest}, indent=2)
        )
