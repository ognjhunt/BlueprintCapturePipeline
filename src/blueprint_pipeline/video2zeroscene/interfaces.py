"""Core interfaces and data models for the BlueprintCapture pipeline.

This module defines the contract between pipeline stages for Phase 3: Capture.
The output format is designed for handoff to BlueprintPipeline for DWM processing.
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

    # Transform to global coordinates (after alignment)
    global_transform: Optional[List[float]] = None  # 4x4 matrix flattened

    # Quality metrics
    registration_rate: float = 0.0
    reprojection_error: float = 0.0
    loop_closure_detected: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the BlueprintCapture pipeline."""

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

    # Output
    output_format: str = "gaussian"  # Output format for DWM handoff

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
