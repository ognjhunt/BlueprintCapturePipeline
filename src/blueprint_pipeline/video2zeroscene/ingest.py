"""Stage 0: Video ingestion and CaptureManifest creation.

This module handles:
- Video normalization and metadata extraction
- Frame extraction with quality filtering
- Keyframe selection based on blur, exposure, and parallax
- Scale anchor detection (ArUco, AprilTag)
- CaptureManifest creation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import (
    CameraIntrinsics,
    CaptureManifest,
    FrameMetadata,
    PipelineConfig,
    ScaleAnchorObservation,
    SensorType,
)


@dataclass
class IngestResult:
    """Result of video ingestion."""
    manifest: CaptureManifest
    frames: List[FrameMetadata]
    keyframes: List[FrameMetadata]
    scale_observations: List[ScaleAnchorObservation]
    frames_dir: Path
    success: bool = True
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class VideoIngestor:
    """Ingest video captures and create normalized CaptureManifest."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def ingest(
        self,
        capture_id: str,
        video_paths: List[Path],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        arkit_data_path: Optional[Path] = None,
        depth_path: Optional[Path] = None,
        imu_path: Optional[Path] = None,
    ) -> IngestResult:
        """Ingest video capture and create CaptureManifest.

        Args:
            capture_id: Unique identifier for this capture
            video_paths: Paths to video files (MP4/MOV)
            output_dir: Directory to write frames and manifest
            metadata: Optional device/capture metadata
            arkit_data_path: Optional path to ARKit poses
            depth_path: Optional path to depth frames
            imu_path: Optional path to IMU data

        Returns:
            IngestResult with manifest, frames, and keyframes
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        errors = []
        metadata = metadata or {}

        # Detect sensor type
        sensor_type = self._detect_sensor_type(
            metadata, arkit_data_path, depth_path, imu_path
        )

        # Extract intrinsics if available
        intrinsics = self._extract_intrinsics(metadata, arkit_data_path)

        # Extract frames from all clips
        all_frames = []
        all_clips = []

        for video_path in video_paths:
            try:
                frames, clip_info = self._extract_frames(
                    video_path=video_path,
                    output_dir=frames_dir / video_path.stem,
                    target_fps=self.config.target_fps * 2,  # Extract at 2x for keyframe selection
                )
                all_frames.extend(frames)
                all_clips.append(clip_info)
            except Exception as e:
                errors.append(f"Failed to extract frames from {video_path}: {e}")

        if not all_frames:
            return IngestResult(
                manifest=CaptureManifest(
                    capture_id=capture_id,
                    capture_timestamp=datetime.now().isoformat(),
                    device_platform="unknown",
                ),
                frames=[],
                keyframes=[],
                scale_observations=[],
                frames_dir=frames_dir,
                success=False,
                errors=errors or ["No frames extracted"],
            )

        # Compute quality scores
        self._compute_quality_scores(all_frames, frames_dir)

        # Select keyframes
        keyframes = self._select_keyframes(all_frames)

        # Detect scale anchors in keyframes
        scale_observations = self._detect_scale_anchors(keyframes, frames_dir)

        # Determine resolution and fps from first frame
        resolution = (all_frames[0].width, all_frames[0].height) if all_frames else (1920, 1080)
        fps = metadata.get("fps", 30.0)

        # Create manifest
        manifest = CaptureManifest(
            capture_id=capture_id,
            capture_timestamp=datetime.now().isoformat(),
            device_platform=metadata.get("platform", "generic"),
            device_model=metadata.get("model"),
            dat_sdk_version=metadata.get("dat_sdk_version"),
            sensor_type=sensor_type,
            has_depth=depth_path is not None,
            has_imu=imu_path is not None,
            has_arkit_poses=arkit_data_path is not None,
            intrinsics=intrinsics,
            clips=[{
                "uri": str(p),
                "fps": clip.get("fps"),
                "duration": clip.get("duration"),
                "frame_count": clip.get("frame_count"),
            } for p, clip in zip(video_paths, all_clips)],
            scale_anchors=scale_observations,
            imu_data_path=str(imu_path) if imu_path else None,
            depth_frames_path=str(depth_path) if depth_path else None,
            arkit_poses_path=str(arkit_data_path) if arkit_data_path else None,
            total_frames=len(all_frames),
            estimated_duration_seconds=all_frames[-1].timestamp_seconds if all_frames else 0,
            resolution=resolution,
            fps=fps,
            notes=metadata.get("notes"),
        )

        # Save manifest
        manifest_path = output_dir / "capture_manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        # Save frame index
        frame_index = {
            "capture_id": capture_id,
            "total_frames": len(all_frames),
            "keyframe_count": len(keyframes),
            "frames": [self._frame_to_dict(f) for f in all_frames],
            "keyframe_ids": [kf.frame_id for kf in keyframes],
        }
        (output_dir / "frame_index.json").write_text(json.dumps(frame_index, indent=2))

        return IngestResult(
            manifest=manifest,
            frames=all_frames,
            keyframes=keyframes,
            scale_observations=scale_observations,
            frames_dir=frames_dir,
            success=True,
            errors=errors if errors else None,
        )

    def _detect_sensor_type(
        self,
        metadata: Dict[str, Any],
        arkit_path: Optional[Path],
        depth_path: Optional[Path],
        imu_path: Optional[Path],
    ) -> SensorType:
        """Detect sensor modality from available data."""
        if depth_path:
            return SensorType.RGB_DEPTH
        if imu_path:
            return SensorType.VISUAL_INERTIAL
        return SensorType.RGB_ONLY

    def _extract_intrinsics(
        self,
        metadata: Dict[str, Any],
        arkit_path: Optional[Path],
    ) -> Optional[CameraIntrinsics]:
        """Extract camera intrinsics from metadata or ARKit."""
        # Try ARKit intrinsics first
        if arkit_path and arkit_path.exists():
            intrinsics_file = arkit_path / "intrinsics.json"
            if intrinsics_file.exists():
                try:
                    data = json.loads(intrinsics_file.read_text())
                    return CameraIntrinsics(
                        fx=data["fx"],
                        fy=data["fy"],
                        cx=data["cx"],
                        cy=data["cy"],
                        width=data["width"],
                        height=data["height"],
                    )
                except Exception:
                    pass

        # Try metadata
        if "intrinsics" in metadata:
            intr = metadata["intrinsics"]
            return CameraIntrinsics(
                fx=intr.get("fx", 1500),
                fy=intr.get("fy", 1500),
                cx=intr.get("cx", 960),
                cy=intr.get("cy", 540),
                width=intr.get("width", 1920),
                height=intr.get("height", 1080),
            )

        return None

    def _extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        target_fps: float,
    ) -> Tuple[List[FrameMetadata], Dict[str, Any]]:
        """Extract frames from video at target FPS."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for frame extraction")

        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / source_fps

        frame_interval = max(1, int(source_fps / target_fps))

        clip_info = {
            "fps": source_fps,
            "duration": duration,
            "frame_count": total_frames,
            "width": width,
            "height": height,
        }

        frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB and save
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_id = f"{video_path.stem}_{extracted_count:06d}"
                frame_filename = f"{frame_id}.png"
                frame_path = output_dir / frame_filename

                cv2.imwrite(str(frame_path), frame)

                timestamp = frame_idx / source_fps

                frames.append(FrameMetadata(
                    frame_id=frame_id,
                    timestamp_seconds=timestamp,
                    source_clip=str(video_path),
                    frame_index_in_clip=frame_idx,
                    file_path=str(frame_path.relative_to(output_dir.parent.parent)),
                    width=width,
                    height=height,
                ))

                extracted_count += 1

            frame_idx += 1

        cap.release()
        return frames, clip_info

    def _compute_quality_scores(
        self,
        frames: List[FrameMetadata],
        frames_dir: Path,
    ) -> None:
        """Compute quality scores for keyframe selection."""
        try:
            import cv2
        except ImportError:
            return

        prev_frame = None
        prev_gray = None

        for i, frame_meta in enumerate(frames):
            frame_path = frames_dir.parent / frame_meta.file_path
            if not frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur score (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            frame_meta.blur_score = float(laplacian.var())

            # Exposure score (histogram spread)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            # Penalize clipped highlights/shadows
            clipped = hist[0] + hist[255]
            frame_meta.exposure_score = 1.0 - min(1.0, clipped * 10)

            # Parallax score (optical flow from previous frame)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                frame_meta.parallax_score = float(np.median(magnitude))
            else:
                frame_meta.parallax_score = 0.0

            prev_gray = gray

    def _select_keyframes(
        self,
        frames: List[FrameMetadata],
    ) -> List[FrameMetadata]:
        """Select keyframes based on quality scores."""
        if not frames:
            return []

        keyframes = []

        for i, frame in enumerate(frames):
            # Skip very blurry frames
            if frame.blur_score < self.config.blur_threshold:
                continue

            # Skip poorly exposed frames
            if frame.exposure_score < 0.3:
                continue

            # Ensure minimum parallax from last keyframe
            if keyframes:
                # Check parallax threshold
                if frame.parallax_score < self.config.min_parallax_threshold:
                    # Still include if enough time has passed
                    time_since_last = (
                        frame.timestamp_seconds - keyframes[-1].timestamp_seconds
                    )
                    if time_since_last < 1.0 / self.config.target_fps:
                        continue

            frame.is_keyframe = True
            keyframes.append(frame)

        return keyframes

    def _detect_scale_anchors(
        self,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
    ) -> List[ScaleAnchorObservation]:
        """Detect scale anchors (ArUco/AprilTag) in keyframes."""
        observations = []

        if "aruco_board" not in self.config.scale_anchor_types:
            return observations

        try:
            import cv2
            from cv2 import aruco
        except ImportError:
            return observations

        # Set up ArUco detector
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector_params = aruco.DetectorParameters()

        for frame in keyframes[:50]:  # Limit to first 50 keyframes
            frame_path = frames_dir.parent / frame.file_path
            if not frame_path.exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(
                gray, aruco_dict, parameters=detector_params
            )

            if ids is not None:
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    # Compute marker size in pixels
                    pts = marker_corners[0]
                    edge_lengths = [
                        np.linalg.norm(pts[i] - pts[(i + 1) % 4])
                        for i in range(4)
                    ]
                    pixel_size = float(np.mean(edge_lengths))

                    # Default ArUco marker size assumption
                    marker_size_meters = 0.05  # 5cm default

                    observations.append(ScaleAnchorObservation(
                        anchor_type="aruco_board",
                        frame_id=frame.frame_id,
                        size_meters=marker_size_meters,
                        pixel_size=pixel_size,
                        confidence=0.9,
                        detection_data={
                            "marker_id": int(marker_id),
                            "corners": pts.tolist(),
                        },
                    ))

        return observations

    def _frame_to_dict(self, frame: FrameMetadata) -> Dict[str, Any]:
        """Convert FrameMetadata to dictionary."""
        return {
            "frame_id": frame.frame_id,
            "timestamp_seconds": frame.timestamp_seconds,
            "source_clip": frame.source_clip,
            "frame_index_in_clip": frame.frame_index_in_clip,
            "file_path": frame.file_path,
            "width": frame.width,
            "height": frame.height,
            "blur_score": frame.blur_score,
            "exposure_score": frame.exposure_score,
            "parallax_score": frame.parallax_score,
            "is_keyframe": frame.is_keyframe,
        }
