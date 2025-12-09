"""ARKit data loader for iOS captures.

This module handles loading and parsing ARKit data from iOS captures:
- Camera poses from poses.jsonl
- Camera intrinsics from intrinsics.json
- Depth maps from depth/*.png
- Mesh anchors from meshes/*.obj
- Motion/IMU data from motion.jsonl

When ARKit poses are available, they can replace COLMAP/SLAM reconstruction
since ARKit provides metric-scale camera tracking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .utils.gcs import GCSClient, GCSPath
from .utils.logging import get_logger
from .jobs.reconstruction import CameraPose

logger = get_logger(__name__)


@dataclass
class ARKitIntrinsics:
    """Camera intrinsics from ARKit."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ARKitIntrinsics":
        return cls(
            fx=float(data.get("fx", data.get("focalLengthX", 0))),
            fy=float(data.get("fy", data.get("focalLengthY", 0))),
            cx=float(data.get("cx", data.get("principalPointX", 0))),
            cy=float(data.get("cy", data.get("principalPointY", 0))),
            width=int(data.get("width", data.get("imageWidth", 1920))),
            height=int(data.get("height", data.get("imageHeight", 1080))),
        )

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


@dataclass
class ARKitPose:
    """Single camera pose from ARKit."""
    frame_index: int
    timestamp: float
    transform: np.ndarray  # 4x4 camera-to-world transform

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ARKitPose":
        """Parse from poses.jsonl entry."""
        transform = np.array(data.get("transform", np.eye(4).tolist()))
        if transform.shape != (4, 4):
            # Flatten and reshape if needed
            transform = np.array(transform).reshape(4, 4)

        return cls(
            frame_index=int(data.get("frameIndex", 0)),
            timestamp=float(data.get("timestamp", 0)),
            transform=transform,
        )

    def to_colmap_pose(self, image_name: str) -> CameraPose:
        """Convert to COLMAP-style pose (world-to-camera).

        COLMAP uses world-to-camera convention, while ARKit provides
        camera-to-world transforms.
        """
        # Invert transform: camera-to-world -> world-to-camera
        world_to_camera = np.linalg.inv(self.transform)

        # Extract rotation and translation
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]

        # Convert rotation matrix to quaternion (w, x, y, z)
        quat = rotation_matrix_to_quaternion(rotation)

        return CameraPose(
            image_id=self.frame_index,
            image_name=image_name,
            qvec=tuple(quat),
            tvec=tuple(translation),
            camera_id=1,
        )


@dataclass
class ARKitFrame:
    """Frame metadata from ARKit."""
    frame_index: int
    timestamp: float
    camera_transform: np.ndarray  # 4x4 matrix
    image_resolution: Tuple[int, int]  # (width, height)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ARKitFrame":
        """Parse from frames.jsonl entry."""
        transform = np.array(data.get("cameraTransform", np.eye(4).tolist()))
        if transform.shape != (4, 4):
            transform = np.array(transform).reshape(4, 4)

        resolution = data.get("imageResolution", {"width": 1920, "height": 1080})
        if isinstance(resolution, dict):
            res_tuple = (int(resolution.get("width", 1920)), int(resolution.get("height", 1080)))
        else:
            res_tuple = (1920, 1080)

        return cls(
            frame_index=int(data.get("frameIndex", 0)),
            timestamp=float(data.get("timestamp", 0)),
            camera_transform=transform,
            image_resolution=res_tuple,
        )


@dataclass
class MotionSample:
    """IMU motion sample from iOS."""
    timestamp: float
    wall_time: str
    attitude_quaternion: Tuple[float, float, float, float]  # (x, y, z, w)
    rotation_rate: Tuple[float, float, float]  # (x, y, z) rad/s
    gravity: Tuple[float, float, float]  # (x, y, z) g
    user_acceleration: Tuple[float, float, float]  # (x, y, z) g

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionSample":
        """Parse from motion.jsonl entry."""
        attitude = data.get("attitude", {})
        quat = attitude.get("quaternion", {"x": 0, "y": 0, "z": 0, "w": 1})

        rotation = data.get("rotationRate", {"x": 0, "y": 0, "z": 0})
        gravity = data.get("gravity", {"x": 0, "y": -1, "z": 0})
        accel = data.get("userAcceleration", {"x": 0, "y": 0, "z": 0})

        return cls(
            timestamp=float(data.get("timestamp", 0)),
            wall_time=data.get("wallTime", ""),
            attitude_quaternion=(
                float(quat.get("x", 0)),
                float(quat.get("y", 0)),
                float(quat.get("z", 0)),
                float(quat.get("w", 1)),
            ),
            rotation_rate=(
                float(rotation.get("x", 0)),
                float(rotation.get("y", 0)),
                float(rotation.get("z", 0)),
            ),
            gravity=(
                float(gravity.get("x", 0)),
                float(gravity.get("y", 0)),
                float(gravity.get("z", 0)),
            ),
            user_acceleration=(
                float(accel.get("x", 0)),
                float(accel.get("y", 0)),
                float(accel.get("z", 0)),
            ),
        )


@dataclass
class ARKitData:
    """Complete ARKit data for a capture session."""
    intrinsics: Optional[ARKitIntrinsics] = None
    poses: List[ARKitPose] = field(default_factory=list)
    frames: List[ARKitFrame] = field(default_factory=list)
    depth_paths: List[str] = field(default_factory=list)
    confidence_paths: List[str] = field(default_factory=list)
    mesh_paths: List[str] = field(default_factory=list)
    motion_samples: List[MotionSample] = field(default_factory=list)

    @property
    def has_poses(self) -> bool:
        return len(self.poses) > 0

    @property
    def has_depth(self) -> bool:
        return len(self.depth_paths) > 0

    @property
    def has_motion(self) -> bool:
        return len(self.motion_samples) > 0

    @property
    def has_meshes(self) -> bool:
        return len(self.mesh_paths) > 0

    def get_pose_for_timestamp(self, timestamp: float) -> Optional[ARKitPose]:
        """Find closest pose for a given timestamp."""
        if not self.poses:
            return None

        closest = min(self.poses, key=lambda p: abs(p.timestamp - timestamp))
        return closest

    def to_colmap_poses(self, image_names: Optional[List[str]] = None) -> List[CameraPose]:
        """Convert all ARKit poses to COLMAP format.

        Args:
            image_names: Optional list of image names to match with poses.
                         If not provided, uses frame indices as names.
        """
        colmap_poses = []

        for i, arkit_pose in enumerate(self.poses):
            if image_names and i < len(image_names):
                name = image_names[i]
            else:
                name = f"frame_{arkit_pose.frame_index:06d}.png"

            colmap_poses.append(arkit_pose.to_colmap_pose(name))

        return colmap_poses


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Uses the Shepperd method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON object per line)."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_arkit_intrinsics(intrinsics_path: Path) -> Optional[ARKitIntrinsics]:
    """Load camera intrinsics from intrinsics.json."""
    if not intrinsics_path.exists():
        return None

    try:
        with open(intrinsics_path, "r") as f:
            data = json.load(f)
        return ARKitIntrinsics.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load ARKit intrinsics: {e}")
        return None


def load_arkit_poses(poses_path: Path) -> List[ARKitPose]:
    """Load camera poses from poses.jsonl."""
    if not poses_path.exists():
        return []

    try:
        entries = load_jsonl(poses_path)
        return [ARKitPose.from_dict(e) for e in entries]
    except Exception as e:
        logger.warning(f"Failed to load ARKit poses: {e}")
        return []


def load_arkit_frames(frames_path: Path) -> List[ARKitFrame]:
    """Load frame metadata from frames.jsonl."""
    if not frames_path.exists():
        return []

    try:
        entries = load_jsonl(frames_path)
        return [ARKitFrame.from_dict(e) for e in entries]
    except Exception as e:
        logger.warning(f"Failed to load ARKit frames: {e}")
        return []


def load_motion_data(motion_path: Path) -> List[MotionSample]:
    """Load IMU motion data from motion.jsonl."""
    if not motion_path.exists():
        return []

    try:
        entries = load_jsonl(motion_path)
        return [MotionSample.from_dict(e) for e in entries]
    except Exception as e:
        logger.warning(f"Failed to load motion data: {e}")
        return []


def load_arkit_data_from_directory(arkit_dir: Path, raw_dir: Optional[Path] = None) -> ARKitData:
    """Load all ARKit data from a directory.

    Expected structure:
        arkit_dir/
        ├── frames.jsonl
        ├── poses.jsonl
        ├── intrinsics.json
        ├── depth/
        │   ├── 000001.png
        │   └── smoothed-000001.png
        ├── confidence/
        │   └── 000001.png
        └── meshes/
            └── *.obj

    Args:
        arkit_dir: Path to arkit/ directory
        raw_dir: Optional path to raw/ directory (for motion.jsonl)
    """
    data = ARKitData()

    # Load intrinsics
    data.intrinsics = load_arkit_intrinsics(arkit_dir / "intrinsics.json")

    # Load poses
    data.poses = load_arkit_poses(arkit_dir / "poses.jsonl")

    # Load frames
    data.frames = load_arkit_frames(arkit_dir / "frames.jsonl")

    # Discover depth maps
    depth_dir = arkit_dir / "depth"
    if depth_dir.exists():
        data.depth_paths = sorted([str(p) for p in depth_dir.glob("*.png")])

    # Discover confidence maps
    confidence_dir = arkit_dir / "confidence"
    if confidence_dir.exists():
        data.confidence_paths = sorted([str(p) for p in confidence_dir.glob("*.png")])

    # Discover meshes
    mesh_dir = arkit_dir / "meshes"
    if mesh_dir.exists():
        data.mesh_paths = sorted([str(p) for p in mesh_dir.glob("*.obj")])

    # Load motion data from raw directory
    if raw_dir:
        motion_path = raw_dir / "motion.jsonl"
        data.motion_samples = load_motion_data(motion_path)

    logger.info(
        f"Loaded ARKit data: {len(data.poses)} poses, {len(data.depth_paths)} depth maps, "
        f"{len(data.mesh_paths)} meshes, {len(data.motion_samples)} motion samples"
    )

    return data


def load_arkit_data_from_gcs(
    bucket_name: str,
    raw_prefix: str,
    local_dir: Path,
    gcs_client: Optional[GCSClient] = None,
) -> ARKitData:
    """Download and load ARKit data from GCS.

    Args:
        bucket_name: GCS bucket name
        raw_prefix: Path prefix to raw/ directory
        local_dir: Local directory to download to
        gcs_client: Optional GCS client

    Returns:
        ARKitData with loaded data
    """
    client = gcs_client or GCSClient()

    # Create local directories
    arkit_local = local_dir / "arkit"
    arkit_local.mkdir(parents=True, exist_ok=True)
    raw_local = local_dir / "raw"
    raw_local.mkdir(parents=True, exist_ok=True)

    gcs_prefix = f"gs://{bucket_name}/{raw_prefix}"

    # Download ARKit files
    arkit_prefix = f"{gcs_prefix}/arkit"
    try:
        client.download_directory(arkit_prefix + "/", arkit_local)
        logger.info(f"Downloaded ARKit data from {arkit_prefix}")
    except Exception as e:
        logger.warning(f"Failed to download ARKit data: {e}")

    # Download motion.jsonl
    motion_uri = f"{gcs_prefix}/motion.jsonl"
    try:
        client.download(motion_uri, raw_local / "motion.jsonl")
        logger.info("Downloaded motion.jsonl")
    except Exception as e:
        logger.debug(f"motion.jsonl not available: {e}")

    return load_arkit_data_from_directory(arkit_local, raw_local)


def can_skip_slam(arkit_data: ARKitData) -> bool:
    """Determine if we can skip SLAM/COLMAP because ARKit poses are sufficient.

    Criteria:
    - Have at least 10 poses
    - Have camera intrinsics
    - Poses are well-distributed (not all identical)
    """
    if not arkit_data.has_poses:
        return False

    if len(arkit_data.poses) < 10:
        return False

    if arkit_data.intrinsics is None:
        return False

    # Check pose diversity (not all identical)
    if len(arkit_data.poses) > 1:
        first_pos = arkit_data.poses[0].transform[:3, 3]
        last_pos = arkit_data.poses[-1].transform[:3, 3]
        distance = np.linalg.norm(last_pos - first_pos)

        # Camera should have moved at least 10cm
        if distance < 0.1:
            logger.warning("ARKit poses show minimal camera movement")
            return False

    return True


def write_colmap_from_arkit(
    arkit_data: ARKitData,
    output_dir: Path,
    image_names: Optional[List[str]] = None,
) -> None:
    """Write COLMAP-format files from ARKit data.

    Creates:
    - cameras.txt: Camera intrinsics
    - images.txt: Camera poses
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write cameras.txt
    cameras_path = output_dir / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        if arkit_data.intrinsics:
            intr = arkit_data.intrinsics
            f.write(f"1 PINHOLE {intr.width} {intr.height} {intr.fx} {intr.fy} {intr.cx} {intr.cy}\n")
        else:
            # Default intrinsics
            f.write("1 PINHOLE 1920 1080 1500 1500 960 540\n")

    # Write images.txt
    colmap_poses = arkit_data.to_colmap_poses(image_names)
    images_path = output_dir / "images.txt"

    with open(images_path, "w") as f:
        f.write("# Image list with image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name\n")
        f.write(f"# Number of images: {len(colmap_poses)}\n")

        for pose in colmap_poses:
            qw, qx, qy, qz = pose.qvec
            tx, ty, tz = pose.tvec
            f.write(f"{pose.image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {pose.camera_id} {pose.image_name}\n")
            f.write("\n")  # Empty points line

    # Write points3D.txt (empty initially, can be populated from depth maps)
    points_path = output_dir / "points3D.txt"
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")

    logger.info(f"Wrote COLMAP files to {output_dir}")
