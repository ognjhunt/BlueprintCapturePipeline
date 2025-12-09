"""iOS manifest parsing and conversion utilities.

This module handles conversion between the iOS BlueprintCapture app's manifest
format and the pipeline's SessionManifest format.

iOS Manifest Structure (from CaptureUploadService):
    {
        "scene_id": "",
        "video_uri": "",
        "device_model": "iPhone 15 Pro",
        "os_version": "17.2",
        "fps_source": 30.0,
        "width": 1920,
        "height": 1440,
        "capture_start_epoch_ms": 1702137045123,
        "has_lidar": true,
        "scale_hint_m_per_unit": 1.0,
        "intended_space_type": "home",
        "object_point_cloud_index": "arkit/objects/index.json",
        "object_point_cloud_count": 5,
        "exposure_samples": [...]
    }

iOS Upload Structure:
    gs://bucket/scenes/{scene_id}/{source}/{timestamp}-{uuid}/raw/
    ├── walkthrough.mov          # Main video file
    ├── motion.jsonl             # IMU data (60 Hz)
    ├── manifest.json            # Capture metadata
    └── arkit/                   # ARKit data (if LiDAR available)
        ├── frames.jsonl         # Frame timestamps & transforms
        ├── poses.jsonl          # Camera pose per frame
        ├── intrinsics.json      # Camera intrinsics (once)
        ├── depth/               # Depth maps
        │   ├── 000001.png
        │   ├── smoothed-000001.png
        ├── confidence/          # Confidence maps
        │   ├── 000001.png
        ├── meshes/              # Mesh anchors (OBJ format)
        └── objects/             # Point clouds
            ├── index.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Clip, ScaleAnchor, SessionManifest
from .utils.gcs import GCSClient, GCSPath
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IOSManifest:
    """Parsed iOS capture manifest."""

    scene_id: str
    video_uri: str
    device_model: str
    os_version: str
    fps_source: float
    width: int
    height: int
    capture_start_epoch_ms: int
    has_lidar: bool
    scale_hint_m_per_unit: float
    intended_space_type: str
    exposure_samples: List[Dict[str, Any]] = field(default_factory=list)
    object_point_cloud_index: Optional[str] = None
    object_point_cloud_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IOSManifest":
        """Create IOSManifest from dictionary."""
        return cls(
            scene_id=data.get("scene_id", ""),
            video_uri=data.get("video_uri", ""),
            device_model=data.get("device_model", "iPhone"),
            os_version=data.get("os_version", "unknown"),
            fps_source=float(data.get("fps_source", 30.0)),
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            capture_start_epoch_ms=int(data.get("capture_start_epoch_ms", 0)),
            has_lidar=bool(data.get("has_lidar", False)),
            scale_hint_m_per_unit=float(data.get("scale_hint_m_per_unit", 1.0)),
            intended_space_type=data.get("intended_space_type", "unknown"),
            exposure_samples=data.get("exposure_samples", []),
            object_point_cloud_index=data.get("object_point_cloud_index"),
            object_point_cloud_count=int(data.get("object_point_cloud_count", 0)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "IOSManifest":
        """Create IOSManifest from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: Path) -> "IOSManifest":
        """Create IOSManifest from local file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_gcs(cls, gcs_uri: str, gcs_client: Optional[GCSClient] = None) -> "IOSManifest":
        """Create IOSManifest from GCS URI."""
        client = gcs_client or GCSClient()
        parsed = GCSPath.from_uri(gcs_uri)

        from google.cloud import storage
        bucket = client._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)
        content = blob.download_as_text()

        return cls.from_json(content)

    @property
    def capture_start_datetime(self) -> datetime:
        """Get capture start as datetime."""
        if self.capture_start_epoch_ms:
            return datetime.utcfromtimestamp(self.capture_start_epoch_ms / 1000)
        return datetime.utcnow()

    @property
    def capture_start_iso(self) -> str:
        """Get capture start as ISO format string."""
        return self.capture_start_datetime.isoformat() + "Z"


@dataclass
class IOSUploadInfo:
    """Information about an iOS upload in GCS."""

    bucket: str
    scene_id: str
    source: str  # "iphone" or "glasses"
    capture_folder: str  # "{timestamp}-{uuid}"
    raw_prefix: str  # Full path to raw/ folder

    # File availability
    has_video: bool = False
    has_manifest: bool = False
    has_motion: bool = False
    has_arkit_frames: bool = False
    has_arkit_poses: bool = False
    has_arkit_intrinsics: bool = False
    has_arkit_depth: bool = False
    has_arkit_confidence: bool = False
    has_arkit_meshes: bool = False
    has_arkit_objects: bool = False

    # File URIs (set when detected)
    video_uri: Optional[str] = None
    manifest_uri: Optional[str] = None
    motion_uri: Optional[str] = None
    arkit_frames_uri: Optional[str] = None
    arkit_poses_uri: Optional[str] = None
    arkit_intrinsics_uri: Optional[str] = None
    depth_uris: List[str] = field(default_factory=list)
    confidence_uris: List[str] = field(default_factory=list)
    mesh_uris: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if upload has minimum required files."""
        return self.has_video and self.has_manifest

    @property
    def has_arkit_data(self) -> bool:
        """Check if ARKit data is available."""
        return self.has_arkit_frames or self.has_arkit_poses or self.has_arkit_depth


def discover_ios_upload(
    bucket_name: str,
    raw_prefix: str,
    gcs_client: Optional[GCSClient] = None,
) -> IOSUploadInfo:
    """Discover files in an iOS upload directory.

    Args:
        bucket_name: GCS bucket name
        raw_prefix: Path to the raw/ directory (without gs:// prefix)
        gcs_client: Optional GCS client instance

    Returns:
        IOSUploadInfo with detected files
    """
    client = gcs_client or GCSClient()

    # Parse the raw_prefix to extract metadata
    # Expected format: scenes/{scene_id}/{source}/{capture_folder}/raw
    parts = raw_prefix.split("/")
    if len(parts) >= 5 and parts[0] == "scenes" and parts[-1] == "raw":
        scene_id = parts[1]
        source = parts[2]
        capture_folder = parts[3]
    else:
        # Fallback parsing
        scene_id = parts[1] if len(parts) > 1 else "unknown"
        source = parts[2] if len(parts) > 2 else "iphone"
        capture_folder = parts[3] if len(parts) > 3 else "unknown"

    info = IOSUploadInfo(
        bucket=bucket_name,
        scene_id=scene_id,
        source=source,
        capture_folder=capture_folder,
        raw_prefix=raw_prefix,
    )

    # List all blobs under the prefix
    gcs_prefix = f"gs://{bucket_name}/{raw_prefix}"

    try:
        for blob_uri in client.list_blobs(gcs_prefix):
            relative_path = blob_uri.replace(f"{gcs_prefix}/", "")

            # Check for known files
            if relative_path == "walkthrough.mov" or relative_path.endswith(".mov"):
                info.has_video = True
                info.video_uri = blob_uri
            elif relative_path == "manifest.json":
                info.has_manifest = True
                info.manifest_uri = blob_uri
            elif relative_path == "motion.jsonl":
                info.has_motion = True
                info.motion_uri = blob_uri
            elif relative_path == "arkit/frames.jsonl":
                info.has_arkit_frames = True
                info.arkit_frames_uri = blob_uri
            elif relative_path == "arkit/poses.jsonl":
                info.has_arkit_poses = True
                info.arkit_poses_uri = blob_uri
            elif relative_path == "arkit/intrinsics.json":
                info.has_arkit_intrinsics = True
                info.arkit_intrinsics_uri = blob_uri
            elif relative_path.startswith("arkit/depth/"):
                info.has_arkit_depth = True
                info.depth_uris.append(blob_uri)
            elif relative_path.startswith("arkit/confidence/"):
                info.has_arkit_confidence = True
                info.confidence_uris.append(blob_uri)
            elif relative_path.startswith("arkit/meshes/"):
                info.has_arkit_meshes = True
                info.mesh_uris.append(blob_uri)
            elif relative_path.startswith("arkit/objects/"):
                info.has_arkit_objects = True

    except Exception as e:
        logger.error(f"Error discovering iOS upload: {e}")

    return info


def convert_ios_to_session(
    ios_manifest: IOSManifest,
    upload_info: IOSUploadInfo,
) -> SessionManifest:
    """Convert iOS manifest and upload info to pipeline SessionManifest.

    Args:
        ios_manifest: Parsed iOS manifest
        upload_info: Upload discovery info

    Returns:
        SessionManifest suitable for pipeline processing
    """
    # Build device info
    device = {
        "platform": "iOS",
        "model": ios_manifest.device_model,
        "os_version": ios_manifest.os_version,
        "resolution": f"{ios_manifest.width}x{ios_manifest.height}",
        "fps": ios_manifest.fps_source,
        "has_lidar": ios_manifest.has_lidar,
        "capture_source": upload_info.source,
    }

    # Build scale anchors
    scale_anchors: List[ScaleAnchor] = []

    # iOS scale hint
    if ios_manifest.scale_hint_m_per_unit and ios_manifest.scale_hint_m_per_unit != 1.0:
        scale_anchors.append(ScaleAnchor(
            anchor_type="ios_scale_hint",
            size_meters=ios_manifest.scale_hint_m_per_unit,
            notes="Scale hint from iOS ARKit session",
        ))

    # ARKit provides metric scale via intrinsics
    if upload_info.has_arkit_intrinsics:
        scale_anchors.append(ScaleAnchor(
            anchor_type="arkit_intrinsics",
            size_meters=1.0,
            notes="ARKit camera intrinsics - metric scale available",
        ))

    # ARKit poses provide world-scale tracking
    if upload_info.has_arkit_poses:
        scale_anchors.append(ScaleAnchor(
            anchor_type="arkit_world_tracking",
            size_meters=1.0,
            notes="ARKit world tracking poses - metric scale",
        ))

    # Build clips
    clips: List[Clip] = []

    if upload_info.video_uri:
        clips.append(Clip(
            uri=upload_info.video_uri,
            fps=ios_manifest.fps_source,
            notes=f"Main capture - {ios_manifest.intended_space_type} space",
        ))

    # Create user notes with useful context
    user_notes_parts = [
        f"iOS {upload_info.source} capture",
        f"Device: {ios_manifest.device_model}",
        f"Space type: {ios_manifest.intended_space_type}",
    ]

    if upload_info.has_arkit_data:
        user_notes_parts.append("ARKit data available")
    if upload_info.has_motion:
        user_notes_parts.append("IMU motion data available")
    if ios_manifest.has_lidar:
        user_notes_parts.append("LiDAR depth available")

    user_notes = " | ".join(user_notes_parts)

    return SessionManifest(
        session_id=upload_info.scene_id,
        capture_start=ios_manifest.capture_start_iso,
        device=device,
        scale_anchors=scale_anchors,
        clips=clips,
        user_notes=user_notes,
    )


@dataclass
class ExtendedSessionData:
    """Extended session data including iOS-specific artifacts.

    This wraps SessionManifest with additional references to iOS-specific
    data that can enhance pipeline processing.
    """

    manifest: SessionManifest
    upload_info: IOSUploadInfo
    ios_manifest: IOSManifest

    # Convenience accessors
    @property
    def session_id(self) -> str:
        return self.manifest.session_id

    @property
    def has_arkit_poses(self) -> bool:
        """Check if ARKit poses are available (can skip SLAM)."""
        return self.upload_info.has_arkit_poses

    @property
    def has_arkit_depth(self) -> bool:
        """Check if ARKit depth maps are available."""
        return self.upload_info.has_arkit_depth

    @property
    def has_motion_data(self) -> bool:
        """Check if IMU motion data is available."""
        return self.upload_info.has_motion

    @property
    def has_lidar(self) -> bool:
        """Check if LiDAR data is available."""
        return self.ios_manifest.has_lidar

    def get_arkit_poses_uri(self) -> Optional[str]:
        """Get URI to ARKit poses file."""
        return self.upload_info.arkit_poses_uri

    def get_arkit_intrinsics_uri(self) -> Optional[str]:
        """Get URI to ARKit camera intrinsics."""
        return self.upload_info.arkit_intrinsics_uri

    def get_depth_uris(self) -> List[str]:
        """Get URIs to depth map files."""
        return self.upload_info.depth_uris

    def get_motion_uri(self) -> Optional[str]:
        """Get URI to motion.jsonl file."""
        return self.upload_info.motion_uri


def load_extended_session(
    bucket_name: str,
    raw_prefix: str,
    gcs_client: Optional[GCSClient] = None,
) -> ExtendedSessionData:
    """Load extended session data from an iOS upload.

    This is the primary entry point for loading iOS captures into the pipeline.

    Args:
        bucket_name: GCS bucket name
        raw_prefix: Path to raw/ directory

    Returns:
        ExtendedSessionData with all available information
    """
    client = gcs_client or GCSClient()

    # Discover upload contents
    upload_info = discover_ios_upload(bucket_name, raw_prefix, client)

    if not upload_info.is_complete:
        raise ValueError(
            f"Incomplete iOS upload at gs://{bucket_name}/{raw_prefix}. "
            f"Has video: {upload_info.has_video}, Has manifest: {upload_info.has_manifest}"
        )

    # Load iOS manifest
    ios_manifest = IOSManifest.from_gcs(upload_info.manifest_uri, client)

    # Convert to session manifest
    session_manifest = convert_ios_to_session(ios_manifest, upload_info)

    return ExtendedSessionData(
        manifest=session_manifest,
        upload_info=upload_info,
        ios_manifest=ios_manifest,
    )


def parse_ios_raw_prefix_from_trigger(
    bucket_name: str,
    object_name: str,
) -> Optional[Tuple[str, str]]:
    """Parse bucket and raw_prefix from a GCS trigger event.

    Args:
        bucket_name: GCS bucket from event
        object_name: Object path from event

    Returns:
        Tuple of (bucket_name, raw_prefix) or None if not a valid iOS upload
    """
    import re

    # Match: scenes/{scene_id}/{source}/{capture_folder}/raw/{filename}
    pattern = r'^(scenes/[^/]+/[^/]+/[^/]+/raw)/.*$'
    match = re.match(pattern, object_name)

    if match:
        raw_prefix = match.group(1)
        return (bucket_name, raw_prefix)

    return None
