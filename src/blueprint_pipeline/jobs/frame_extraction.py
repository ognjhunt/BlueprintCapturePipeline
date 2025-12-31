"""Frame extraction job - extract keyframes from video clips."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..models import ArtifactPaths, Clip, JobPayload, SessionManifest
from ..utils.io import ensure_local_dir, FrameWriter, save_image, save_json
from .base import (
    BaseJob,
    JobContext,
    JobResult,
    JobStatus,
    download_inputs,
    merge_parameters,
    upload_outputs,
)


@dataclass
class FrameInfo:
    """Metadata for an extracted frame."""
    frame_id: str
    source_clip: str
    timestamp_seconds: float
    frame_index: int
    width: int
    height: int
    file_path: str


@dataclass
class FrameExtractionJob(BaseJob):
    """Extract frames from video clips for 3D Gaussian reconstruction.

    This job:
    1. Downloads video clips from GCS
    2. Extracts frames at configured FPS
    3. Optionally generates simple dynamic object masks for SLAM

    Inputs:
        - Video clips (MP4/MOV files from GCS)

    Outputs:
        - Extracted frames (PNG images)
        - Frame index JSON with metadata
        - Optional: simple dynamic masks (using YOLO/optical flow)
    """

    name: str = "frame-extraction"
    description: str = "Decode video clips and extract keyframes for 3DGS reconstruction."
    timeout_minutes: int = 30
    target_fps: float = 4.0
    enable_dynamic_masking: bool = False  # Optional simple masking for SLAM

    def _get_default_parameters(self) -> Dict[str, Any]:
        return {
            "target_fps": self.target_fps,
            "enable_dynamic_masking": self.enable_dynamic_masking,
            "max_dimension": 1920,  # Max frame dimension (resize if larger)
        }

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(self._get_default_parameters(), parameters)
        return JobPayload(
            job_name=self.name,
            session_id=session.session_id,
            inputs={
                "clips": ",".join([clip.uri for clip in session.clips]),
            },
            outputs={
                "frames": artifacts.frames,
                "masks": artifacts.masks,
            },
            parameters=params,
        )

    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute frame extraction."""
        result = JobResult(status=JobStatus.RUNNING)

        # Setup output directories
        frames_dir = ensure_local_dir(ctx.workspace / "frames")
        masks_dir = ensure_local_dir(ctx.workspace / "masks")
        clips_dir = ensure_local_dir(ctx.workspace / "clips")

        # Download video clips
        clips = ctx.session.clips
        clip_paths = self._download_clips(ctx, clips, clips_dir)

        # Extract frames from each clip
        all_frames: List[FrameInfo] = []
        target_fps = ctx.parameters.get("target_fps", self.target_fps)
        max_dimension = ctx.parameters.get("max_dimension", 1920)

        with ctx.tracker.stage("extract_frames", len(clip_paths)):
            for clip, clip_path in zip(clips, clip_paths):
                ctx.logger.info(f"Processing clip: {clip_path.name}")

                frames = self._extract_frames_from_clip(
                    ctx=ctx,
                    clip=clip,
                    clip_path=clip_path,
                    output_dir=frames_dir,
                    target_fps=target_fps,
                    max_dimension=max_dimension,
                )
                all_frames.extend(frames)
                ctx.tracker.update(1)

        ctx.logger.info(f"Extracted {len(all_frames)} frames from {len(clips)} clips")
        ctx.tracker.log_metric("total_frames", len(all_frames))

        # Optionally generate simple dynamic masks
        if ctx.parameters.get("enable_dynamic_masking", self.enable_dynamic_masking):
            self._generate_simple_masks(ctx, all_frames, frames_dir, masks_dir)

        # Save frame index
        frame_index = {
            "session_id": ctx.session.session_id,
            "total_frames": len(all_frames),
            "fps": target_fps,
            "frames": [self._frame_to_dict(f) for f in all_frames],
        }
        frame_index_path = frames_dir / "frame_index.json"
        save_json(frame_index, frame_index_path)

        # Upload outputs
        outputs_to_upload = {"frames": frames_dir}
        output_destinations = {"frames": ctx.artifacts.frames}

        # Only upload masks if masking was enabled and masks exist
        if (masks_dir / "dynamic_masks").exists():
            outputs_to_upload["masks"] = masks_dir
            output_destinations["masks"] = ctx.artifacts.masks

        uploaded = upload_outputs(ctx, outputs_to_upload, output_destinations)

        result.outputs = uploaded
        result.artifacts_uploaded = len(all_frames) + 1  # +1 for frame_index.json
        result.status = JobStatus.COMPLETED
        return result

    def _download_clips(
        self,
        ctx: JobContext,
        clips: List[Clip],
        output_dir: Path,
    ) -> List[Path]:
        """Download video clips from GCS."""
        clip_paths = []

        with ctx.tracker.stage("download_clips", len(clips)):
            for i, clip in enumerate(clips):
                ctx.logger.info(f"Downloading clip {i+1}/{len(clips)}: {clip.uri}")

                # Extract filename from URI
                filename = clip.uri.split("/")[-1]
                local_path = output_dir / filename

                ctx.gcs.download(clip.uri, local_path)
                clip_paths.append(local_path)
                ctx.tracker.update(1)

        return clip_paths

    def _extract_frames_from_clip(
        self,
        ctx: JobContext,
        clip: Clip,
        clip_path: Path,
        output_dir: Path,
        target_fps: float,
        max_dimension: int = 1920,
    ) -> List[FrameInfo]:
        """Extract frames from a single video clip.

        Uses OpenCV for video decoding.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required for video processing")

        frames: List[FrameInfo] = []
        cap = cv2.VideoCapture(str(clip_path))

        if not cap.isOpened():
            ctx.logger.error(f"Failed to open video: {clip_path}")
            return frames

        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ctx.logger.info(
            f"Video: {width}x{height}, {source_fps:.1f}fps, {total_frames} frames"
        )

        # Calculate frame skip interval
        frame_interval = max(1, int(source_fps / target_fps))

        # Calculate resize if needed
        scale = 1.0
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            ctx.logger.info(f"Resizing frames to {new_width}x{new_height}")
        else:
            new_width, new_height = width, height

        clip_name = clip_path.stem
        frame_writer = FrameWriter(
            output_dir=output_dir / clip_name,
            prefix="frame",
            extension="png",
        )

        frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Resize if needed
                if scale < 1.0:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save frame
                frame_path = frame_writer.write(frame_rgb, index=extracted_count)

                # Create frame info
                timestamp = frame_idx / source_fps
                frame_id = f"{clip_name}_{extracted_count:06d}"

                frames.append(FrameInfo(
                    frame_id=frame_id,
                    source_clip=clip.uri,
                    timestamp_seconds=timestamp,
                    frame_index=extracted_count,
                    width=new_width,
                    height=new_height,
                    file_path=str(frame_path.relative_to(output_dir)),
                ))

                extracted_count += 1

            frame_idx += 1

        cap.release()
        ctx.logger.info(f"Extracted {extracted_count} frames from {clip_name}")
        return frames

    def _generate_simple_masks(
        self,
        ctx: JobContext,
        frames: List[FrameInfo],
        frames_dir: Path,
        masks_dir: Path,
    ) -> None:
        """Generate simple dynamic object masks using YOLO/optical flow.

        This is optional and used to help SLAM ignore dynamic objects.
        """
        try:
            from ..masking.dynamic_mask import DynamicMaskGenerator, MaskConfig

            config = MaskConfig(
                backend="auto",  # Will try YOLO, cascade, optical_flow
                classes_to_mask=["person", "car", "dog", "cat"],
            )

            generator = DynamicMaskGenerator(config)
            if not generator.initialize():
                ctx.logger.info("No masking backend available, skipping dynamic masks")
                return

            output_dir = ensure_local_dir(masks_dir / "dynamic_masks")

            # Get frame paths
            frame_paths = [frames_dir / f.file_path for f in frames]

            # Generate masks
            ctx.logger.info(f"Generating dynamic masks for {len(frame_paths)} frames")
            masks = generator.generate_masks_for_sequence(
                image_paths=frame_paths,
                output_dir=output_dir,
                progress_callback=lambda cur, total: ctx.tracker.update(1) if cur % 10 == 0 else None,
            )

            ctx.logger.info(f"Generated {len(masks)} dynamic masks")

        except ImportError as e:
            ctx.logger.warning(f"Masking module not available: {e}")
        except Exception as e:
            ctx.logger.warning(f"Failed to generate dynamic masks: {e}")

    def _frame_to_dict(self, frame: FrameInfo) -> Dict[str, Any]:
        """Convert FrameInfo to dictionary for serialization."""
        return {
            "frame_id": frame.frame_id,
            "source_clip": frame.source_clip,
            "timestamp_seconds": frame.timestamp_seconds,
            "frame_index": frame.frame_index,
            "width": frame.width,
            "height": frame.height,
            "file_path": frame.file_path,
        }
