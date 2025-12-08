"""Frame extraction and SAM3 video segmentation job."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..models import ArtifactPaths, Clip, JobPayload, SessionManifest
from ..utils.io import ensure_local_dir, FrameWriter, save_image, save_json
from .base import (
    GPUJob,
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
class MaskInfo:
    """Metadata for a segmentation mask."""
    mask_id: str
    frame_id: str
    category: str
    is_dynamic: bool
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    file_path: str


@dataclass
class FrameExtractionJob(GPUJob):
    """Extract frames from video clips and generate SAM3 segmentation masks.

    This job:
    1. Downloads video clips from GCS
    2. Extracts frames at configured FPS
    3. Runs SAM3-video for instance segmentation
    4. Tags dynamic objects (people, hands) for masking during reconstruction
    5. Outputs frames and masks in COCO format

    Inputs:
        - Video clips (MP4/MOV files from GCS)

    Outputs:
        - Extracted frames (PNG images)
        - Instance masks (PNG masks + COCO JSON annotations)
    """

    name: str = "frame-extraction"
    description: str = "Decode video clips, extract frames, and run SAM 3 masking."
    timeout_minutes: int = 45
    target_fps: float = 4.0
    mask_model: str = "sam3-video"
    include_dynamic_masks: bool = True

    # Dynamic object categories to flag for masking during reconstruction
    dynamic_categories: List[str] = field(
        default_factory=lambda: ["person", "hand", "pet", "moving_object"]
    )

    def _get_default_parameters(self) -> Dict[str, Any]:
        params = super()._get_default_parameters()
        params.update({
            "target_fps": self.target_fps,
            "mask_model": self.mask_model,
            "include_dynamic_masks": self.include_dynamic_masks,
        })
        return params

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        params = merge_parameters(
            self._get_default_parameters(),
            {
                "target_fps": self.target_fps,
                "mask_model": self.mask_model,
                "include_dynamic_masks": self.include_dynamic_masks,
            },
        )
        params = merge_parameters(params, parameters)
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
        """Execute frame extraction and segmentation."""
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
        all_masks: List[MaskInfo] = []

        target_fps = ctx.parameters.get("target_fps", self.target_fps)

        with ctx.tracker.stage("extract_frames", len(clip_paths)):
            for clip, clip_path in zip(clips, clip_paths):
                ctx.logger.info(f"Processing clip: {clip_path.name}")

                frames = self._extract_frames_from_clip(
                    ctx=ctx,
                    clip=clip,
                    clip_path=clip_path,
                    output_dir=frames_dir,
                    target_fps=target_fps,
                )
                all_frames.extend(frames)
                ctx.tracker.update(1)

        ctx.logger.info(f"Extracted {len(all_frames)} frames from {len(clips)} clips")
        ctx.tracker.log_metric("total_frames", len(all_frames))

        # Run SAM3 segmentation
        if ctx.parameters.get("include_dynamic_masks", self.include_dynamic_masks):
            with ctx.tracker.stage("sam3_segmentation", len(all_frames)):
                all_masks = self._run_sam3_segmentation(
                    ctx=ctx,
                    frames=all_frames,
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                )

        ctx.tracker.log_metric("total_masks", len(all_masks))

        # Generate COCO-format annotations
        annotations = self._generate_coco_annotations(
            ctx=ctx,
            frames=all_frames,
            masks=all_masks,
        )
        annotations_path = masks_dir / "annotations.json"
        save_json(annotations, annotations_path)

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
        uploaded = upload_outputs(
            ctx,
            {"frames": frames_dir, "masks": masks_dir},
            {"frames": ctx.artifacts.frames, "masks": ctx.artifacts.masks},
        )

        result.outputs = uploaded
        result.artifacts_uploaded = len(all_frames) + len(all_masks) + 2  # +2 for JSONs
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
                    width=width,
                    height=height,
                    file_path=str(frame_path.relative_to(output_dir)),
                ))

                extracted_count += 1

            frame_idx += 1

        cap.release()
        ctx.logger.info(f"Extracted {extracted_count} frames from {clip_name}")
        return frames

    def _run_sam3_segmentation(
        self,
        ctx: JobContext,
        frames: List[FrameInfo],
        frames_dir: Path,
        masks_dir: Path,
    ) -> List[MaskInfo]:
        """Run SAM3 video segmentation on extracted frames.

        This creates instance masks for each frame, tracking objects across frames.
        """
        all_masks: List[MaskInfo] = []

        # Load SAM3 model
        model = self._load_sam3_model(ctx)
        if model is None:
            ctx.logger.warning("SAM3 model not available - skipping segmentation")
            return all_masks

        # Group frames by clip for video-based segmentation
        frames_by_clip: Dict[str, List[FrameInfo]] = {}
        for frame in frames:
            clip_name = frame.frame_id.rsplit("_", 1)[0]
            if clip_name not in frames_by_clip:
                frames_by_clip[clip_name] = []
            frames_by_clip[clip_name].append(frame)

        for clip_name, clip_frames in frames_by_clip.items():
            ctx.logger.info(f"Segmenting {len(clip_frames)} frames from {clip_name}")

            clip_masks_dir = ensure_local_dir(masks_dir / clip_name)

            # Process frames in batches for video tracking
            masks = self._segment_video_frames(
                ctx=ctx,
                model=model,
                frames=clip_frames,
                frames_dir=frames_dir,
                masks_dir=clip_masks_dir,
            )
            all_masks.extend(masks)

            ctx.tracker.update(len(clip_frames))

        return all_masks

    def _load_sam3_model(self, ctx: JobContext) -> Optional[Any]:
        """Load SAM3 video model.

        Returns None if model is not available (for testing without GPU).
        """
        model_name = ctx.parameters.get("mask_model", self.mask_model)
        ctx.logger.info(f"Loading segmentation model: {model_name}")

        try:
            # Try to load SAM 2/3 from segment-anything-2
            from sam2.build_sam import build_sam2_video_predictor

            # Model checkpoint path - configurable
            checkpoint = ctx.parameters.get(
                "sam_checkpoint",
                "segment-anything-2/checkpoints/sam2_hiera_large.pt"
            )
            config = ctx.parameters.get(
                "sam_config",
                "sam2_hiera_l.yaml"
            )

            predictor = build_sam2_video_predictor(config, checkpoint)
            ctx.logger.info("SAM2 video predictor loaded successfully")
            return predictor

        except ImportError:
            ctx.logger.warning(
                "SAM2 not installed. Install with: pip install segment-anything-2"
            )
        except Exception as e:
            ctx.logger.warning(f"Failed to load SAM2 model: {e}")

        # Fallback: try ultralytics SAM
        try:
            from ultralytics import SAM

            model = SAM("sam2_l.pt")
            ctx.logger.info("Ultralytics SAM loaded as fallback")
            return model
        except ImportError:
            ctx.logger.warning("Ultralytics not available")
        except Exception as e:
            ctx.logger.warning(f"Failed to load ultralytics SAM: {e}")

        return None

    def _segment_video_frames(
        self,
        ctx: JobContext,
        model: Any,
        frames: List[FrameInfo],
        frames_dir: Path,
        masks_dir: Path,
    ) -> List[MaskInfo]:
        """Segment a sequence of video frames with object tracking."""
        masks: List[MaskInfo] = []

        # Determine model type and call appropriate method
        model_type = type(model).__name__

        if hasattr(model, "init_state"):
            # SAM2 video predictor style
            masks = self._segment_with_sam2_video(
                ctx, model, frames, frames_dir, masks_dir
            )
        elif hasattr(model, "predict"):
            # Ultralytics style
            masks = self._segment_with_ultralytics(
                ctx, model, frames, frames_dir, masks_dir
            )
        else:
            # Fallback: per-frame segmentation without tracking
            masks = self._segment_frames_individually(
                ctx, model, frames, frames_dir, masks_dir
            )

        return masks

    def _segment_with_sam2_video(
        self,
        ctx: JobContext,
        predictor: Any,
        frames: List[FrameInfo],
        frames_dir: Path,
        masks_dir: Path,
    ) -> List[MaskInfo]:
        """Segment frames using SAM2 video predictor with tracking."""
        masks: List[MaskInfo] = []

        try:
            import torch
            from PIL import Image

            # Initialize video state
            frame_paths = [frames_dir / f.file_path for f in frames]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = predictor.init_state(video_path=str(frame_paths[0].parent))

                # Auto-generate prompts for first frame (grid points)
                # In production, this could use object detection for better prompts
                first_frame = np.array(Image.open(frame_paths[0]))
                h, w = first_frame.shape[:2]

                # Add grid points as prompts
                grid_size = 4
                points = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        x = int((i + 0.5) * w / grid_size)
                        y = int((j + 0.5) * h / grid_size)
                        points.append([x, y])

                # Add prompts to predictor
                for obj_id, point in enumerate(points):
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        state,
                        frame_idx=0,
                        obj_id=obj_id,
                        points=np.array([point]),
                        labels=np.array([1]),
                    )

                # Propagate through video
                for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                    if frame_idx >= len(frames):
                        break

                    frame_info = frames[frame_idx]

                    for obj_idx, obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[obj_idx] > 0).cpu().numpy().squeeze()

                        if mask.sum() < 100:  # Skip very small masks
                            continue

                        # Classify as dynamic or static
                        is_dynamic = self._classify_dynamic(mask, frame_info)

                        # Save mask
                        mask_filename = f"{frame_info.frame_id}_obj{obj_id:03d}.png"
                        mask_path = masks_dir / mask_filename
                        save_image((mask * 255).astype(np.uint8), mask_path)

                        # Compute bounding box
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            bbox = (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())
                        else:
                            bbox = (0, 0, 0, 0)

                        masks.append(MaskInfo(
                            mask_id=f"{frame_info.frame_id}_obj{obj_id:03d}",
                            frame_id=frame_info.frame_id,
                            category="object",
                            is_dynamic=is_dynamic,
                            confidence=0.9,  # SAM2 doesn't provide confidence
                            bbox=bbox,
                            area=int(mask.sum()),
                            file_path=mask_filename,
                        ))

        except Exception as e:
            ctx.logger.error(f"SAM2 video segmentation failed: {e}")

        return masks

    def _segment_with_ultralytics(
        self,
        ctx: JobContext,
        model: Any,
        frames: List[FrameInfo],
        frames_dir: Path,
        masks_dir: Path,
    ) -> List[MaskInfo]:
        """Segment frames using Ultralytics SAM."""
        masks: List[MaskInfo] = []

        for frame_info in frames:
            frame_path = frames_dir / frame_info.file_path

            try:
                results = model.predict(str(frame_path), verbose=False)

                for result in results:
                    if result.masks is None:
                        continue

                    for idx, (mask_data, box) in enumerate(
                        zip(result.masks.data, result.boxes)
                    ):
                        mask = mask_data.cpu().numpy().astype(np.uint8) * 255

                        if mask.sum() < 100:
                            continue

                        # Get class if available
                        cls_id = int(box.cls) if hasattr(box, "cls") else 0
                        category = result.names.get(cls_id, "object") if hasattr(result, "names") else "object"

                        is_dynamic = category in self.dynamic_categories

                        # Save mask
                        mask_filename = f"{frame_info.frame_id}_mask{idx:03d}.png"
                        mask_path = masks_dir / mask_filename
                        save_image(mask, mask_path)

                        # Bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                        masks.append(MaskInfo(
                            mask_id=f"{frame_info.frame_id}_mask{idx:03d}",
                            frame_id=frame_info.frame_id,
                            category=category,
                            is_dynamic=is_dynamic,
                            confidence=float(box.conf) if hasattr(box, "conf") else 0.9,
                            bbox=bbox,
                            area=int(mask.sum() / 255),
                            file_path=mask_filename,
                        ))

            except Exception as e:
                ctx.logger.warning(f"Failed to segment frame {frame_info.frame_id}: {e}")

        return masks

    def _segment_frames_individually(
        self,
        ctx: JobContext,
        model: Any,
        frames: List[FrameInfo],
        frames_dir: Path,
        masks_dir: Path,
    ) -> List[MaskInfo]:
        """Fallback: segment each frame independently."""
        ctx.logger.info("Using fallback individual frame segmentation")
        # Placeholder for generic segmentation
        return []

    def _classify_dynamic(self, mask: np.ndarray, frame_info: FrameInfo) -> bool:
        """Classify whether a mask likely represents a dynamic object.

        Uses heuristics based on mask size, position, and shape.
        """
        h, w = mask.shape
        area = mask.sum()

        # Very large masks are likely background
        if area > (h * w * 0.5):
            return False

        # Masks in center-bottom (likely person) are dynamic
        ys, xs = np.where(mask)
        if len(ys) > 0:
            center_y = ys.mean() / h
            if center_y > 0.6:  # Lower part of frame
                return True

        return False

    def _generate_coco_annotations(
        self,
        ctx: JobContext,
        frames: List[FrameInfo],
        masks: List[MaskInfo],
    ) -> Dict[str, Any]:
        """Generate COCO-format annotations JSON."""
        images = []
        annotations = []
        categories = {}

        for frame in frames:
            images.append({
                "id": frame.frame_id,
                "file_name": frame.file_path,
                "width": frame.width,
                "height": frame.height,
            })

        for mask in masks:
            if mask.category not in categories:
                categories[mask.category] = len(categories) + 1

            annotations.append({
                "id": mask.mask_id,
                "image_id": mask.frame_id,
                "category_id": categories[mask.category],
                "segmentation": {"mask_file": mask.file_path},
                "bbox": list(mask.bbox),
                "area": mask.area,
                "iscrowd": 0,
                "is_dynamic": mask.is_dynamic,
                "confidence": mask.confidence,
            })

        return {
            "info": {
                "description": f"SAM3 segmentation for session {ctx.session.session_id}",
                "version": "1.0",
            },
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": cat_id, "name": cat_name}
                for cat_name, cat_id in categories.items()
            ],
        }

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
