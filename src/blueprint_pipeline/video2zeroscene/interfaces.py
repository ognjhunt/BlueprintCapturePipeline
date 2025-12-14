"""Core interfaces and data models for the video2zeroscene pipeline.

This module defines the contract between pipeline stages and the ZeroScene
output format that BlueprintPipeline expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SensorType(Enum):
    """Sensor modality for capture."""
    RGB_ONLY = "rgb_only"           # Meta glasses, generic RGB camera
    RGB_DEPTH = "rgb_depth"         # iPhone LiDAR, Azure Kinect, RealSense
    VISUAL_INERTIAL = "visual_inertial"  # RGB + synchronized IMU


class SLAMBackend(Enum):
    """SLAM backend selection based on sensor type."""
    WILDGS_SLAM = "wildgs_slam"     # Default for RGB-only (handles dynamics)
    SPLATMAP = "splatmap"           # Alternative for RGB-only (geometry focus)
    SPLATAM = "splatam"             # For RGB-D captures
    VIGS_SLAM = "vigs_slam"         # For visual-inertial captures
    ARKIT_DIRECT = "arkit_direct"   # Direct ARKit pose import (iOS)
    COLMAP_FALLBACK = "colmap"      # Fallback SfM + 3DGS


class AssetizationTier(Enum):
    """Tier strategy for object asset generation."""
    TIER_1_RECONSTRUCT = "reconstruct"  # Multi-view reconstruction
    TIER_2_PROXY = "proxy"              # Proxy geometry (box/hull)
    TIER_3_REPLACE = "replace"          # Asset replacement/retrieval


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: Optional[List[float]] = None
    camera_model: str = "PINHOLE"


@dataclass
class FrameMetadata:
    """Metadata for a single extracted frame."""
    frame_id: str
    timestamp_seconds: float
    source_clip: str
    frame_index_in_clip: int
    file_path: str
    width: int
    height: int
    # Quality metrics from keyframe selection
    blur_score: float = 0.0           # Variance of Laplacian
    exposure_score: float = 0.0       # Histogram-based exposure check
    parallax_score: float = 0.0       # Optical flow magnitude from previous
    is_keyframe: bool = True


@dataclass
class ScaleAnchorObservation:
    """Observation of a scale anchor in a frame."""
    anchor_type: str  # "aruco_board", "apriltag", "tape_measure", "known_object"
    frame_id: str
    size_meters: float
    pixel_size: float
    confidence: float
    detection_data: Optional[Dict[str, Any]] = None


@dataclass
class CaptureManifest:
    """Complete capture session metadata.

    This is the canonical input format for the pipeline, created during
    Stage 0 (Ingest).
    """
    capture_id: str
    capture_timestamp: str  # ISO 8601

    # Device info
    device_platform: str  # "ios", "meta_glasses", "android", "generic"
    device_model: Optional[str] = None
    dat_sdk_version: Optional[str] = None

    # Sensor configuration
    sensor_type: SensorType = SensorType.RGB_ONLY
    has_depth: bool = False
    has_imu: bool = False
    has_arkit_poses: bool = False

    # Camera
    intrinsics: Optional[CameraIntrinsics] = None

    # Video clips
    clips: List[Dict[str, Any]] = field(default_factory=list)

    # Scale anchors
    scale_anchors: List[ScaleAnchorObservation] = field(default_factory=list)

    # Optional IMU/depth paths
    imu_data_path: Optional[str] = None
    depth_frames_path: Optional[str] = None
    arkit_poses_path: Optional[str] = None

    # Computed during ingest
    total_frames: int = 0
    estimated_duration_seconds: float = 0.0
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0

    # User notes
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "capture_id": self.capture_id,
            "capture_timestamp": self.capture_timestamp,
            "device": {
                "platform": self.device_platform,
                "model": self.device_model,
                "dat_sdk_version": self.dat_sdk_version,
            },
            "sensor": {
                "type": self.sensor_type.value,
                "has_depth": self.has_depth,
                "has_imu": self.has_imu,
                "has_arkit_poses": self.has_arkit_poses,
            },
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
            } if self.intrinsics else None,
            "clips": self.clips,
            "scale_anchors": [
                {
                    "type": sa.anchor_type,
                    "frame_id": sa.frame_id,
                    "size_meters": sa.size_meters,
                    "confidence": sa.confidence,
                }
                for sa in self.scale_anchors
            ],
            "total_frames": self.total_frames,
            "duration_seconds": self.estimated_duration_seconds,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptureManifest":
        """Deserialize from dictionary."""
        device = data.get("device", {})
        sensor = data.get("sensor", {})
        intr = data.get("intrinsics")

        intrinsics = None
        if intr:
            intrinsics = CameraIntrinsics(
                fx=intr["fx"],
                fy=intr["fy"],
                cx=intr["cx"],
                cy=intr["cy"],
                width=intr["width"],
                height=intr["height"],
            )

        scale_anchors = [
            ScaleAnchorObservation(
                anchor_type=sa["type"],
                frame_id=sa["frame_id"],
                size_meters=sa["size_meters"],
                pixel_size=0,
                confidence=sa.get("confidence", 0.5),
            )
            for sa in data.get("scale_anchors", [])
        ]

        return cls(
            capture_id=data["capture_id"],
            capture_timestamp=data["capture_timestamp"],
            device_platform=device.get("platform", "generic"),
            device_model=device.get("model"),
            dat_sdk_version=device.get("dat_sdk_version"),
            sensor_type=SensorType(sensor.get("type", "rgb_only")),
            has_depth=sensor.get("has_depth", False),
            has_imu=sensor.get("has_imu", False),
            has_arkit_poses=sensor.get("has_arkit_poses", False),
            intrinsics=intrinsics,
            clips=data.get("clips", []),
            scale_anchors=scale_anchors,
            total_frames=data.get("total_frames", 0),
            estimated_duration_seconds=data.get("duration_seconds", 0.0),
            resolution=tuple(data.get("resolution", [1920, 1080])),
            fps=data.get("fps", 30.0),
            notes=data.get("notes"),
        )


@dataclass
class Submap:
    """A submap representing a chunk of the environment.

    For large spaces (grocery stores), we split into submaps to handle:
    - Memory constraints
    - Drift accumulation
    - Loop closure
    """
    submap_id: str
    start_frame_index: int
    end_frame_index: int
    start_timestamp: float
    end_timestamp: float
    keyframe_ids: List[str]

    # Submap-local reconstruction results
    poses_path: Optional[str] = None
    gaussians_path: Optional[str] = None
    mesh_path: Optional[str] = None

    # Transform to global coordinates (after alignment)
    global_transform: Optional[List[float]] = None  # 4x4 matrix flattened

    # Quality metrics
    registration_rate: float = 0.0
    reprojection_error: float = 0.0
    loop_closure_detected: bool = False


@dataclass
class TrackInfo:
    """Tracked object instance across frames (from SAM3).

    This represents a 2D object track before lifting to 3D.
    """
    track_id: str
    concept_label: str  # SAM3 concept (e.g., "chair", "table", "shelf")

    # Per-frame detections
    frame_ids: List[str]
    bboxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    mask_paths: List[str]
    confidences: List[float]

    # Track properties
    is_dynamic: bool = False  # Person, hand, pet, etc.
    first_frame_index: int = 0
    last_frame_index: int = 0
    total_observations: int = 0

    # Computed after lifting
    coverage_score: float = 0.0
    viewpoint_diversity: float = 0.0


@dataclass
class ObjectProposal:
    """3D object proposal lifted from 2D tracks.

    This is the intermediate representation before assetization.
    """
    proposal_id: str
    track_id: str
    concept_label: str

    # 3D bounding box (oriented bounding box)
    obb_center: Tuple[float, float, float]
    obb_axes: List[List[float]]  # 3x3 rotation matrix
    obb_extents: Tuple[float, float, float]

    # World transform
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)

    # Support surface
    support_surface: str  # "floor", "table", "shelf", "wall", "ceiling"
    support_height: float = 0.0

    # Quality metrics
    confidence: float = 0.0
    coverage_score: float = 0.0
    reprojection_consistency: float = 0.0
    num_observations: int = 0

    # Assetization recommendation
    recommended_tier: AssetizationTier = AssetizationTier.TIER_2_PROXY


@dataclass
class ObjectAssetBundle:
    """Generated object asset ready for scene composition."""
    asset_id: str
    proposal_id: str
    concept_label: str

    # Asset paths
    mesh_path: str  # GLB/USD path
    texture_path: Optional[str] = None
    collision_path: Optional[str] = None

    # Generation metadata
    tier: AssetizationTier = AssetizationTier.TIER_2_PROXY
    source: str = "proxy"  # "reconstruction", "proxy", "retrieval", "generation"

    # Placement in scene
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Bounds
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Quality
    quality_score: float = 0.5


@dataclass
class ZeroSceneBundle:
    """ZeroScene-compatible output bundle for BlueprintPipeline handoff.

    This follows the ZeroScene folder structure expected by
    BlueprintPipeline's zeroscene_adapter_job.py.

    Structure:
        zeroscene/
            scene_info.json
            objects/
                obj_i/
                    mesh.glb
                    pose.json
                    bounds.json
                    material.json
            background/
                mesh.glb
                collision.glb
            camera/
                intrinsics.json
                trajectory.json
    """
    capture_id: str
    output_path: Path

    # Scene metadata
    scene_info: Dict[str, Any] = field(default_factory=dict)

    # Background environment
    background_mesh_path: Optional[str] = None
    background_collision_path: Optional[str] = None

    # Objects
    objects: List[ObjectAssetBundle] = field(default_factory=list)

    # Camera
    intrinsics: Optional[CameraIntrinsics] = None
    camera_trajectory: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    scale_factor: float = 1.0
    up_axis: str = "Y"
    meters_per_unit: float = 1.0

    # Completion marker
    is_complete: bool = False

    def write(self) -> Path:
        """Write the ZeroScene bundle to disk."""
        bundle_path = self.output_path / "zeroscene"
        bundle_path.mkdir(parents=True, exist_ok=True)

        # Write scene_info.json
        scene_info = {
            "capture_id": self.capture_id,
            "scale_factor": self.scale_factor,
            "up_axis": self.up_axis,
            "meters_per_unit": self.meters_per_unit,
            "object_count": len(self.objects),
            "has_background": self.background_mesh_path is not None,
            **self.scene_info,
        }
        (bundle_path / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

        # Write objects
        objects_dir = bundle_path / "objects"
        objects_dir.mkdir(exist_ok=True)

        for i, obj in enumerate(self.objects):
            obj_dir = objects_dir / f"obj_{i:04d}"
            obj_dir.mkdir(exist_ok=True)

            # Write pose.json
            pose = {
                "position": list(obj.position),
                "rotation": list(obj.rotation),
                "scale": list(obj.scale),
            }
            (obj_dir / "pose.json").write_text(json.dumps(pose, indent=2))

            # Write bounds.json
            bounds = {
                "min": list(obj.bounds_min),
                "max": list(obj.bounds_max),
            }
            (obj_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))

            # Write material.json (placeholder)
            material = {
                "label": obj.concept_label,
                "tier": obj.tier.value,
                "source": obj.source,
            }
            (obj_dir / "material.json").write_text(json.dumps(material, indent=2))

        # Write background info
        background_dir = bundle_path / "background"
        background_dir.mkdir(exist_ok=True)

        if self.background_mesh_path:
            bg_info = {
                "mesh_path": self.background_mesh_path,
                "collision_path": self.background_collision_path,
            }
            (background_dir / "info.json").write_text(json.dumps(bg_info, indent=2))

        # Write camera info
        camera_dir = bundle_path / "camera"
        camera_dir.mkdir(exist_ok=True)

        if self.intrinsics:
            intrinsics = {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
            }
            (camera_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

        if self.camera_trajectory:
            (camera_dir / "trajectory.json").write_text(
                json.dumps(self.camera_trajectory, indent=2)
            )

        # Write completion marker
        self.is_complete = True
        (bundle_path / ".complete").touch()

        return bundle_path


@dataclass
class PipelineConfig:
    """Configuration for the video2zeroscene pipeline."""

    # SLAM configuration
    slam_backend: Optional[SLAMBackend] = None  # Auto-select based on sensors
    force_slam_backend: bool = False

    # Keyframe selection
    target_fps: float = 2.0  # Keyframes per second
    min_parallax_threshold: float = 0.1
    blur_threshold: float = 100.0  # Variance of Laplacian

    # Submap chunking (for large spaces)
    enable_submapping: bool = True
    submap_keyframe_count: int = 100
    submap_overlap_frames: int = 10

    # Scale calibration
    enforce_scale_anchor: bool = True
    scale_anchor_types: List[str] = field(
        default_factory=lambda: ["aruco_board", "apriltag", "known_object"]
    )

    # SAM3 segmentation
    sam3_concepts: List[str] = field(
        default_factory=lambda: [
            # Dynamic (for masking)
            "person", "hand", "pet",
            # Static objects (for inventory)
            "chair", "table", "desk", "sofa", "bed",
            "shelf", "cabinet", "drawer", "door",
            "lamp", "plant", "monitor", "keyboard",
            "bottle", "cup", "book", "box",
        ]
    )
    sam3_dynamic_concepts: List[str] = field(
        default_factory=lambda: ["person", "hand", "pet"]
    )

    # Object proposals
    min_object_area: int = 500  # Minimum mask area in pixels
    min_object_views: int = 3   # Minimum views for proposal
    max_objects: int = 100

    # Assetization
    tier1_coverage_threshold: float = 0.6  # Coverage needed for reconstruction
    tier1_diversity_threshold: float = 0.3  # Viewpoint diversity threshold
    enable_hunyuan3d: bool = True
    enable_asset_retrieval: bool = False  # LiteReality-style replacement

    # Mesh extraction
    mesh_decimation_target: int = 500000
    collision_decimation_target: int = 50000
    texture_resolution: int = 4096

    # Output
    output_format: str = "zeroscene"  # "zeroscene" or "blueprint_direct"

    def select_slam_backend(self, manifest: CaptureManifest) -> SLAMBackend:
        """Auto-select SLAM backend based on available sensors."""
        if self.force_slam_backend and self.slam_backend:
            return self.slam_backend

        # If ARKit poses available, use them directly
        if manifest.has_arkit_poses:
            return SLAMBackend.ARKIT_DIRECT

        # RGB-D captures
        if manifest.has_depth:
            return SLAMBackend.SPLATAM

        # Visual-inertial captures
        if manifest.has_imu:
            return SLAMBackend.VIGS_SLAM

        # Default: RGB-only with WildGS-SLAM
        return SLAMBackend.WILDGS_SLAM
