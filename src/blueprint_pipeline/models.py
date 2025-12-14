"""Core data models for the BlueprintCapturePipeline.

This module defines the canonical data structures used throughout the pipeline,
including the recommended GCS storage layout for captures.

Storage Layout (recommended):
    gs://bucket/captures/{capture_id}/
        raw/
            video.mp4
            metadata.json
            arkit/              (iOS only)
                poses.jsonl
                intrinsics.json
        stage0_ingest/
            capture_manifest.json
            frame_index.json
        stage1_frames/
            frames/*.png
            keyframes.json
        stage2_slam/
            poses/
                poses.json
                images.txt (COLMAP format)
            gaussians/
                point_cloud.ply
            reports/
                reprojection.json
        stage3_mesh/
            environment_mesh.glb
            environment_collision.glb
            textures/
        stage4_tracks/
            masks/
            tracks.json
            annotations.json (COCO format)
        stage5_proposals/
            proposals.json
        stage6_assets/
            {object_id}/
                mesh.glb
                collision.obj
        zeroscene/              <- Handoff to BlueprintPipeline
            scene_info.json
            objects/
            background/
            camera/
            .complete
        blueprint/              <- Output from BlueprintPipeline
            scene.usdc
            reports/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SensorType(Enum):
    """Sensor modality for capture."""
    RGB_ONLY = "rgb_only"           # Meta glasses, generic RGB camera
    RGB_DEPTH = "rgb_depth"         # iPhone LiDAR, Azure Kinect, RealSense
    VISUAL_INERTIAL = "visual_inertial"  # RGB + synchronized IMU


@dataclass
class ScaleAnchor:
    """Observations that allow us to recover metric scale from monocular capture."""

    anchor_type: str  # e.g., "aruco_board", "apriltag", "tape_measure", "known_object"
    size_meters: float
    notes: Optional[str] = None
    frame_id: Optional[str] = None
    confidence: float = 0.5


@dataclass
class Clip:
    """Video clip reference."""
    uri: str
    fps: Optional[float] = None
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class SessionManifest:
    """Capture session metadata.

    This is the legacy format maintained for backward compatibility.
    New code should prefer CaptureManifest from video2zeroscene.interfaces.
    """
    session_id: str
    capture_start: str
    device: Dict[str, str]
    scale_anchors: List[ScaleAnchor]
    clips: List[Clip]
    user_notes: Optional[str] = None

    # Extended fields for sensor-conditional pipeline
    sensor_type: SensorType = SensorType.RGB_ONLY
    has_depth: bool = False
    has_imu: bool = False
    has_arkit_poses: bool = False
    arkit_data_uri: Optional[str] = None
    depth_data_uri: Optional[str] = None
    imu_data_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "capture_start": self.capture_start,
            "device": self.device,
            "scale_anchors": [
                {"anchor_type": sa.anchor_type, "size_meters": sa.size_meters, "notes": sa.notes}
                for sa in self.scale_anchors
            ],
            "clips": [
                {"uri": c.uri, "fps": c.fps, "notes": c.notes}
                for c in self.clips
            ],
            "user_notes": self.user_notes,
            "sensor_type": self.sensor_type.value,
            "has_depth": self.has_depth,
            "has_imu": self.has_imu,
            "has_arkit_poses": self.has_arkit_poses,
        }


@dataclass
class ArtifactPaths:
    """Logical paths in GCS for artifacts produced by each stage.

    Follows the recommended storage layout for video2zeroscene pipeline.
    """

    session_root: str

    # Stage directories (new layout)
    raw: str = ""
    stage0_ingest: str = ""
    stage1_frames: str = ""
    stage2_slam: str = ""
    stage3_mesh: str = ""
    stage4_tracks: str = ""
    stage5_proposals: str = ""
    stage6_assets: str = ""
    zeroscene: str = ""
    blueprint: str = ""

    # Legacy paths (for backward compatibility)
    frames: str = ""
    masks: str = ""
    reconstruction: str = ""
    meshes: str = ""
    objects: str = ""
    reports: str = ""

    def __post_init__(self):
        """Initialize paths based on session_root."""
        if not self.raw:
            self.raw = f"{self.session_root}/raw"
        if not self.stage0_ingest:
            self.stage0_ingest = f"{self.session_root}/stage0_ingest"
        if not self.stage1_frames:
            self.stage1_frames = f"{self.session_root}/stage1_frames"
        if not self.stage2_slam:
            self.stage2_slam = f"{self.session_root}/stage2_slam"
        if not self.stage3_mesh:
            self.stage3_mesh = f"{self.session_root}/stage3_mesh"
        if not self.stage4_tracks:
            self.stage4_tracks = f"{self.session_root}/stage4_tracks"
        if not self.stage5_proposals:
            self.stage5_proposals = f"{self.session_root}/stage5_proposals"
        if not self.stage6_assets:
            self.stage6_assets = f"{self.session_root}/stage6_assets"
        if not self.zeroscene:
            self.zeroscene = f"{self.session_root}/zeroscene"
        if not self.blueprint:
            self.blueprint = f"{self.session_root}/blueprint"

        # Legacy path mappings
        if not self.frames:
            self.frames = f"{self.stage1_frames}/frames"
        if not self.masks:
            self.masks = f"{self.stage4_tracks}/masks"
        if not self.reconstruction:
            self.reconstruction = self.stage2_slam
        if not self.meshes:
            self.meshes = self.stage3_mesh
        if not self.objects:
            self.objects = self.stage6_assets
        if not self.reports:
            self.reports = f"{self.session_root}/reports"


def create_artifact_paths(bucket: str, capture_id: str) -> ArtifactPaths:
    """Create ArtifactPaths for a capture.

    Args:
        bucket: GCS bucket name
        capture_id: Unique capture identifier

    Returns:
        ArtifactPaths with all paths initialized
    """
    session_root = f"gs://{bucket}/captures/{capture_id}"
    return ArtifactPaths(session_root=session_root)


@dataclass
class JobPayload:
    """A serializable payload to hand to Cloud Run Jobs."""

    job_name: str
    session_id: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    parameters: Dict[str, object] = field(default_factory=dict)

    # Extended fields for video2zeroscene pipeline
    sensor_type: Optional[str] = None
    slam_backend: Optional[str] = None
    pipeline_mode: str = "video2zeroscene"  # or "legacy"

    def as_json(self) -> Dict[str, object]:
        return {
            "job_name": self.job_name,
            "session_id": self.session_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "sensor_type": self.sensor_type,
            "slam_backend": self.slam_backend,
            "pipeline_mode": self.pipeline_mode,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "JobPayload":
        """Deserialize from dictionary."""
        return cls(
            job_name=data["job_name"],
            session_id=data["session_id"],
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            parameters=data.get("parameters", {}),
            sensor_type=data.get("sensor_type"),
            slam_backend=data.get("slam_backend"),
            pipeline_mode=data.get("pipeline_mode", "video2zeroscene"),
        )


@dataclass
class PipelineConfig:
    """Configuration for the video2zeroscene pipeline.

    This can be serialized and passed between Cloud Run jobs.
    """

    # SLAM configuration
    slam_backend: Optional[str] = None  # Auto-select based on sensors
    force_slam_backend: bool = False

    # Keyframe selection
    target_fps: float = 2.0
    min_parallax_threshold: float = 0.1
    blur_threshold: float = 100.0

    # Submap chunking
    enable_submapping: bool = True
    submap_keyframe_count: int = 100
    submap_overlap_frames: int = 10

    # Scale calibration
    enforce_scale_anchor: bool = True

    # SAM3 segmentation
    sam3_concepts: List[str] = field(default_factory=lambda: [
        "person", "hand", "pet",
        "chair", "table", "desk", "sofa", "bed",
        "shelf", "cabinet", "drawer", "door",
        "lamp", "plant", "monitor", "keyboard",
    ])
    sam3_dynamic_concepts: List[str] = field(default_factory=lambda: [
        "person", "hand", "pet"
    ])

    # Object proposals
    min_object_area: int = 500
    min_object_views: int = 3
    max_objects: int = 100

    # Assetization
    tier1_coverage_threshold: float = 0.6
    tier1_diversity_threshold: float = 0.3
    enable_hunyuan3d: bool = True

    # Mesh extraction
    mesh_decimation_target: int = 500000
    collision_decimation_target: int = 50000
    texture_resolution: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "slam_backend": self.slam_backend,
            "force_slam_backend": self.force_slam_backend,
            "target_fps": self.target_fps,
            "min_parallax_threshold": self.min_parallax_threshold,
            "blur_threshold": self.blur_threshold,
            "enable_submapping": self.enable_submapping,
            "submap_keyframe_count": self.submap_keyframe_count,
            "submap_overlap_frames": self.submap_overlap_frames,
            "enforce_scale_anchor": self.enforce_scale_anchor,
            "sam3_concepts": self.sam3_concepts,
            "sam3_dynamic_concepts": self.sam3_dynamic_concepts,
            "min_object_area": self.min_object_area,
            "min_object_views": self.min_object_views,
            "max_objects": self.max_objects,
            "tier1_coverage_threshold": self.tier1_coverage_threshold,
            "tier1_diversity_threshold": self.tier1_diversity_threshold,
            "enable_hunyuan3d": self.enable_hunyuan3d,
            "mesh_decimation_target": self.mesh_decimation_target,
            "collision_decimation_target": self.collision_decimation_target,
            "texture_resolution": self.texture_resolution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
