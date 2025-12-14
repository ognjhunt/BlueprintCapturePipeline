"""Stage 6: Lift 2D tracks into 3D object proposals.

This module handles:
- Projecting 2D masks into 3D using camera poses
- Computing oriented bounding boxes (OBB) for each object
- Estimating support surfaces (floor, table, shelf)
- Computing confidence scores based on coverage and consistency
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import (
    AssetizationTier,
    CameraIntrinsics,
    ObjectProposal,
    PipelineConfig,
    TrackInfo,
)
from .slam import CameraPose


@dataclass
class LiftResult:
    """Result of 2D-to-3D lifting."""
    proposals: List[ObjectProposal]
    success: bool = True
    errors: List[str] = field(default_factory=list)


class ObjectLifter:
    """Lift 2D object tracks into 3D proposals.

    Uses camera poses and intrinsics to:
    1. Cast rays through masked pixels
    2. Intersect with environment mesh (or estimate depth)
    3. Accumulate 3D points per tracked instance
    4. Compute OBB, centroid, and support surface
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        tracks: List[TrackInfo],
        poses: List[CameraPose],
        intrinsics: Optional[CameraIntrinsics],
        frames_dir: Path,
        masks_dir: Path,
        mesh_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> LiftResult:
        """Lift 2D tracks into 3D object proposals.

        Args:
            tracks: Object tracks from SAM3
            poses: Camera poses from SLAM
            intrinsics: Camera intrinsics
            frames_dir: Directory containing frames
            masks_dir: Directory containing masks
            mesh_path: Optional environment mesh for ray intersection
            output_dir: Optional output directory for proposals

        Returns:
            LiftResult with object proposals
        """
        proposals = []

        # Build pose lookup by frame_id
        pose_by_frame = {p.frame_id: p for p in poses}
        pose_by_image = {p.image_name: p for p in poses}

        # Filter to static tracks only
        static_tracks = [t for t in tracks if not t.is_dynamic]

        for track in static_tracks:
            # Collect 3D points for this track
            points_3d = []

            for frame_id, bbox, mask_path in zip(
                track.frame_ids, track.bboxes, track.mask_paths
            ):
                # Find corresponding pose
                pose = pose_by_frame.get(frame_id)
                if pose is None:
                    # Try matching by image name
                    for p in poses:
                        if frame_id in p.image_name or p.image_name in frame_id:
                            pose = p
                            break

                if pose is None:
                    continue

                # Get mask center as 2D point
                x, y, w, h = bbox
                center_2d = (x + w / 2, y + h / 2)

                # Project to 3D
                point_3d = self._project_point_to_3d(
                    point_2d=center_2d,
                    pose=pose,
                    intrinsics=intrinsics,
                    estimated_depth=2.0,  # Default depth assumption
                )

                if point_3d is not None:
                    points_3d.append(point_3d)

            if len(points_3d) < self.config.min_object_views:
                continue

            points_3d = np.array(points_3d)

            # Compute centroid and OBB
            centroid = points_3d.mean(axis=0)
            obb_axes, obb_extents = self._compute_obb(points_3d)

            # Estimate support surface
            support_surface = self._estimate_support_surface(centroid, obb_extents)

            # Compute coverage and viewpoint diversity
            coverage = self._compute_coverage(track, poses)
            diversity = self._compute_viewpoint_diversity(track, poses, pose_by_frame)

            # Recommend assetization tier
            tier = self._recommend_tier(coverage, diversity, len(track.frame_ids))

            proposals.append(ObjectProposal(
                proposal_id=f"prop_{track.track_id}",
                track_id=track.track_id,
                concept_label=track.concept_label,
                obb_center=tuple(centroid),
                obb_axes=obb_axes.tolist(),
                obb_extents=tuple(obb_extents),
                position=tuple(centroid),
                rotation=(1.0, 0.0, 0.0, 0.0),  # Identity for now
                support_surface=support_surface,
                support_height=float(centroid[1]),  # Y-up assumed
                confidence=float(min(coverage, diversity)),
                coverage_score=coverage,
                reprojection_consistency=0.8,  # Placeholder
                num_observations=len(track.frame_ids),
                recommended_tier=tier,
            ))

        # Save proposals if output_dir provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_proposals(proposals, output_dir)

        return LiftResult(proposals=proposals)

    def _project_point_to_3d(
        self,
        point_2d: Tuple[float, float],
        pose: CameraPose,
        intrinsics: Optional[CameraIntrinsics],
        estimated_depth: float,
    ) -> Optional[np.ndarray]:
        """Project 2D point to 3D using camera pose and estimated depth."""
        u, v = point_2d

        # Get intrinsics
        if intrinsics:
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.cx, intrinsics.cy
        else:
            # Default intrinsics
            fx, fy = 1500, 1500
            cx, cy = 960, 540

        # Compute ray direction in camera coordinates
        x_cam = (u - cx) / fx * estimated_depth
        y_cam = (v - cy) / fy * estimated_depth
        z_cam = estimated_depth

        point_cam = np.array([x_cam, y_cam, z_cam])

        # Convert pose to rotation matrix
        R = self._quaternion_to_rotation(pose.rotation)
        t = np.array(pose.translation)

        # COLMAP uses world-to-camera convention
        # Camera center: -R^T @ t
        # Point in world: R^T @ (p_cam - t) ... not quite right
        # Actually for COLMAP: p_world = R^T @ p_cam + camera_center
        camera_center = -R.T @ t
        point_world = R.T @ point_cam + camera_center

        return point_world

    def _quaternion_to_rotation(
        self,
        quat: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

    def _compute_obb(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute oriented bounding box for point cloud.

        Returns:
            (axes, extents) where axes is 3x3 rotation and extents is (dx, dy, dz)
        """
        if len(points) < 3:
            return np.eye(3), np.array([0.1, 0.1, 0.1])

        # Simple approach: use PCA for orientation
        centroid = points.mean(axis=0)
        centered = points - centroid

        try:
            # SVD for principal axes
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            axes = Vt.T  # Columns are principal axes

            # Project to get extents
            projected = centered @ axes
            extents = projected.max(axis=0) - projected.min(axis=0)
            extents = np.maximum(extents, 0.01)  # Minimum size

        except np.linalg.LinAlgError:
            axes = np.eye(3)
            extents = np.array([0.1, 0.1, 0.1])

        return axes, extents

    def _estimate_support_surface(
        self,
        centroid: np.ndarray,
        extents: np.ndarray,
    ) -> str:
        """Estimate what surface the object is resting on."""
        # Simple heuristics based on height (Y-up assumed)
        height = centroid[1]
        obj_height = extents[1] if len(extents) > 1 else 0.1

        bottom = height - obj_height / 2

        if bottom < 0.05:
            return "floor"
        elif bottom < 0.5:
            return "floor"  # Low object
        elif bottom < 1.0:
            return "table"
        elif bottom < 1.8:
            return "shelf"
        else:
            return "wall"

    def _compute_coverage(
        self,
        track: TrackInfo,
        poses: List[CameraPose],
    ) -> float:
        """Compute what fraction of frames this object is visible in."""
        if not poses:
            return 0.0
        return min(1.0, len(track.frame_ids) / len(poses))

    def _compute_viewpoint_diversity(
        self,
        track: TrackInfo,
        poses: List[CameraPose],
        pose_by_frame: Dict[str, CameraPose],
    ) -> float:
        """Compute angular diversity of viewpoints observing this object."""
        if len(track.frame_ids) < 2:
            return 0.0

        # Get camera positions for frames where object is visible
        positions = []
        for frame_id in track.frame_ids:
            pose = pose_by_frame.get(frame_id)
            if pose:
                R = self._quaternion_to_rotation(pose.rotation)
                t = np.array(pose.translation)
                camera_center = -R.T @ t
                positions.append(camera_center)

        if len(positions) < 2:
            return 0.0

        positions = np.array(positions)
        centroid = positions.mean(axis=0)
        directions = positions - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1
        directions = directions / norms

        # Compute pairwise angles
        angles = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                dot = np.clip(np.dot(directions[i], directions[j]), -1, 1)
                angle = np.arccos(dot)
                angles.append(angle)

        if angles:
            return min(1.0, np.mean(angles) / np.pi)
        return 0.0

    def _recommend_tier(
        self,
        coverage: float,
        diversity: float,
        num_views: int,
    ) -> AssetizationTier:
        """Recommend assetization tier based on coverage and diversity."""
        if (coverage >= self.config.tier1_coverage_threshold and
            diversity >= self.config.tier1_diversity_threshold and
            num_views >= self.config.min_object_views * 2):
            return AssetizationTier.TIER_1_RECONSTRUCT

        if num_views >= self.config.min_object_views:
            return AssetizationTier.TIER_2_PROXY

        return AssetizationTier.TIER_2_PROXY

    def _save_proposals(
        self,
        proposals: List[ObjectProposal],
        output_dir: Path,
    ) -> None:
        """Save proposals to JSON."""
        proposals_data = [
            {
                "proposal_id": p.proposal_id,
                "track_id": p.track_id,
                "concept": p.concept_label,
                "position": list(p.position),
                "rotation": list(p.rotation),
                "obb_center": list(p.obb_center),
                "obb_extents": list(p.obb_extents),
                "support_surface": p.support_surface,
                "confidence": p.confidence,
                "coverage_score": p.coverage_score,
                "num_observations": p.num_observations,
                "recommended_tier": p.recommended_tier.value,
            }
            for p in proposals
        ]
        (output_dir / "proposals.json").write_text(
            json.dumps({"proposals": proposals_data}, indent=2)
        )
