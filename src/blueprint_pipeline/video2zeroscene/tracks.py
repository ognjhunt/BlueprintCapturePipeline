"""Stage 5: SAM3 concept segmentation and video tracking.

This module handles:
- SAM3 concept segmentation with open-vocabulary prompts
- Video tracking for consistent instance IDs
- Dynamic object classification (for SLAM masking)
- Object inventory generation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces import (
    CaptureManifest,
    FrameMetadata,
    PipelineConfig,
    TrackInfo,
)


@dataclass
class SAM3Detection:
    """Single SAM3 detection in a frame."""
    detection_id: str
    frame_id: str
    concept: str
    mask_path: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    confidence: float
    is_dynamic: bool


@dataclass
class TracksResult:
    """Result of SAM3 tracking."""
    tracks: List[TrackInfo]
    dynamic_mask_paths: Dict[str, Path]  # frame_id -> combined dynamic mask
    detections_per_frame: Dict[str, List[SAM3Detection]]
    success: bool = True
    errors: List[str] = field(default_factory=list)


class SAM3Tracker:
    """SAM3-based concept segmentation and video tracking.

    Uses SAM3's open-vocabulary concept segmentation capabilities to:
    1. Segment all instances of specified concepts (e.g., "chair", "table")
    2. Track instances across video frames
    3. Generate masks for dynamic objects (for SLAM exclusion)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.predictor = None
        self.processor = None

    def run(
        self,
        manifest: CaptureManifest,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
    ) -> TracksResult:
        """Run SAM3 concept segmentation and tracking.

        Args:
            manifest: Capture manifest
            keyframes: Selected keyframes for processing
            frames_dir: Directory containing extracted frames
            output_dir: Directory to write masks and tracking data

        Returns:
            TracksResult with tracks, dynamic masks, and per-frame detections
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Load SAM3 model
        model_loaded = self._load_sam3_model()
        if not model_loaded:
            print("SAM3 not available, using fallback segmentation")
            return self._run_fallback(keyframes, frames_dir, output_dir)

        # Run concept segmentation
        all_detections: Dict[str, List[SAM3Detection]] = {}
        dynamic_masks: Dict[str, Path] = {}

        # Process frames
        for i, kf in enumerate(keyframes):
            frame_path = frames_dir.parent / kf.file_path
            if not frame_path.exists():
                continue

            print(f"Processing frame {i+1}/{len(keyframes)}: {kf.frame_id}")

            # Run concept segmentation
            frame_detections = self._segment_frame(
                frame_path=frame_path,
                frame_id=kf.frame_id,
                masks_dir=masks_dir,
            )
            all_detections[kf.frame_id] = frame_detections

            # Create combined dynamic mask
            dynamic_mask_path = self._create_dynamic_mask(
                frame_detections, masks_dir, kf.frame_id
            )
            if dynamic_mask_path:
                dynamic_masks[kf.frame_id] = dynamic_mask_path

        # Build tracks from detections
        tracks = self._build_tracks(all_detections, keyframes)

        # Save tracking data
        self._save_tracking_data(tracks, all_detections, output_dir)

        return TracksResult(
            tracks=tracks,
            dynamic_mask_paths=dynamic_masks,
            detections_per_frame=all_detections,
        )

    def _load_sam3_model(self) -> bool:
        """Load SAM3 model."""
        try:
            from sam3.model_builder import build_sam3_video_predictor
            from sam3.model.sam3_image_processor import Sam3Processor

            self.predictor = build_sam3_video_predictor()
            self.processor = Sam3Processor()
            print("SAM3 model loaded successfully")
            return True

        except ImportError:
            print("SAM3 not installed, trying SAM2 fallback")

        # Try SAM2 as fallback
        try:
            from sam2.build_sam import build_sam2_video_predictor

            self.predictor = build_sam2_video_predictor(
                config_file="sam2_hiera_l.yaml",
                ckpt_path=None,  # Use default
            )
            print("SAM2 loaded as fallback")
            return True

        except ImportError:
            print("Neither SAM3 nor SAM2 available")

        return False

    def _segment_frame(
        self,
        frame_path: Path,
        frame_id: str,
        masks_dir: Path,
    ) -> List[SAM3Detection]:
        """Segment a single frame using SAM3 concept prompts."""
        detections = []

        try:
            from PIL import Image
            import torch
        except ImportError:
            return detections

        # Load image
        image = np.array(Image.open(frame_path).convert("RGB"))
        h, w = image.shape[:2]

        # Check if we have SAM3-style concept segmentation
        if hasattr(self.predictor, "handle_request"):
            # SAM3 API
            detections = self._segment_with_sam3(
                image, frame_id, masks_dir, self.config.sam3_concepts
            )
        elif hasattr(self.predictor, "set_image"):
            # SAM2-style API with point prompts
            detections = self._segment_with_sam2(
                image, frame_id, masks_dir
            )
        else:
            # Generic fallback
            detections = self._segment_with_grid(image, frame_id, masks_dir)

        return detections

    def _segment_with_sam3(
        self,
        image: np.ndarray,
        frame_id: str,
        masks_dir: Path,
        concepts: List[str],
    ) -> List[SAM3Detection]:
        """Segment using SAM3 concept prompts."""
        detections = []

        try:
            # SAM3 session-based API
            from PIL import Image
            import tempfile

            # Save image temporarily for SAM3
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                Image.fromarray(image).save(f.name)
                temp_path = f.name

            # Start session
            response = self.predictor.handle_request({
                "type": "start_session",
                "resource_path": temp_path,
            })
            session_id = response.get("session_id")

            if not session_id:
                return detections

            # Segment each concept
            for concept in concepts:
                try:
                    response = self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": concept,
                    })

                    masks = response.get("masks", [])
                    for idx, mask_data in enumerate(masks):
                        mask = np.array(mask_data, dtype=np.uint8) * 255
                        if mask.sum() < self.config.min_object_area:
                            continue

                        # Save mask
                        mask_filename = f"{frame_id}_{concept}_{idx:03d}.png"
                        mask_path = masks_dir / mask_filename
                        Image.fromarray(mask).save(mask_path)

                        # Compute bbox
                        ys, xs = np.where(mask > 127)
                        if len(xs) == 0:
                            continue
                        bbox = (int(xs.min()), int(ys.min()),
                               int(xs.max() - xs.min()), int(ys.max() - ys.min()))

                        is_dynamic = concept in self.config.sam3_dynamic_concepts

                        detections.append(SAM3Detection(
                            detection_id=f"{frame_id}_{concept}_{idx}",
                            frame_id=frame_id,
                            concept=concept,
                            mask_path=mask_filename,
                            bbox=bbox,
                            area=int(mask.sum() / 255),
                            confidence=0.9,
                            is_dynamic=is_dynamic,
                        ))

                except Exception as e:
                    print(f"Failed to segment concept '{concept}': {e}")

            # End session
            self.predictor.handle_request({
                "type": "end_session",
                "session_id": session_id,
            })

        except Exception as e:
            print(f"SAM3 segmentation failed: {e}")

        return detections

    def _segment_with_sam2(
        self,
        image: np.ndarray,
        frame_id: str,
        masks_dir: Path,
    ) -> List[SAM3Detection]:
        """Segment using SAM2 with automatic point prompts."""
        detections = []

        try:
            from PIL import Image
            import torch

            h, w = image.shape[:2]

            # Set image
            self.predictor.set_image(image)

            # Generate grid points for automatic segmentation
            grid_size = 8
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int((i + 0.5) * w / grid_size)
                    y = int((j + 0.5) * h / grid_size)
                    points.append([x, y])

            points = np.array(points)
            labels = np.ones(len(points))

            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )

            for idx, (mask, score) in enumerate(zip(masks, scores)):
                mask_uint8 = (mask * 255).astype(np.uint8)
                if mask_uint8.sum() < self.config.min_object_area:
                    continue

                # Save mask
                mask_filename = f"{frame_id}_obj_{idx:03d}.png"
                mask_path = masks_dir / mask_filename
                Image.fromarray(mask_uint8).save(mask_path)

                # Compute bbox
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                bbox = (int(xs.min()), int(ys.min()),
                       int(xs.max() - xs.min()), int(ys.max() - ys.min()))

                # Classify as dynamic based on position (bottom-center typically person)
                center_y = (ys.min() + ys.max()) / 2 / h
                is_dynamic = center_y > 0.7

                detections.append(SAM3Detection(
                    detection_id=f"{frame_id}_obj_{idx}",
                    frame_id=frame_id,
                    concept="object",
                    mask_path=mask_filename,
                    bbox=bbox,
                    area=int(mask_uint8.sum() / 255),
                    confidence=float(score),
                    is_dynamic=is_dynamic,
                ))

        except Exception as e:
            print(f"SAM2 segmentation failed: {e}")

        return detections

    def _segment_with_grid(
        self,
        image: np.ndarray,
        frame_id: str,
        masks_dir: Path,
    ) -> List[SAM3Detection]:
        """Fallback grid-based segmentation without SAM."""
        # Placeholder - returns empty for now
        return []

    def _create_dynamic_mask(
        self,
        detections: List[SAM3Detection],
        masks_dir: Path,
        frame_id: str,
    ) -> Optional[Path]:
        """Create combined mask of all dynamic objects."""
        dynamic_detections = [d for d in detections if d.is_dynamic]
        if not dynamic_detections:
            return None

        try:
            from PIL import Image

            combined_mask = None

            for det in dynamic_detections:
                mask_path = masks_dir / det.mask_path
                if not mask_path.exists():
                    continue

                mask = np.array(Image.open(mask_path).convert("L"))
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = np.maximum(combined_mask, mask)

            if combined_mask is not None:
                output_path = masks_dir / f"{frame_id}_dynamic.png"
                Image.fromarray(combined_mask).save(output_path)
                return output_path

        except Exception as e:
            print(f"Failed to create dynamic mask: {e}")

        return None

    def _build_tracks(
        self,
        detections_per_frame: Dict[str, List[SAM3Detection]],
        keyframes: List[FrameMetadata],
    ) -> List[TrackInfo]:
        """Build object tracks from per-frame detections.

        Uses simple IoU-based tracking to link detections across frames.
        """
        tracks: Dict[str, TrackInfo] = {}
        next_track_id = 0

        # Sort frames by timestamp
        sorted_frames = sorted(keyframes, key=lambda kf: kf.timestamp_seconds)

        for kf in sorted_frames:
            frame_detections = detections_per_frame.get(kf.frame_id, [])

            for det in frame_detections:
                # Try to match with existing track
                best_track_id = None
                best_iou = 0.3  # Minimum IoU threshold

                for track_id, track in tracks.items():
                    if track.concept_label != det.concept:
                        continue

                    # Check IoU with last detection in track
                    if track.bboxes:
                        iou = self._compute_iou(track.bboxes[-1], det.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_track_id = track_id

                if best_track_id:
                    # Add to existing track
                    track = tracks[best_track_id]
                    track.frame_ids.append(det.frame_id)
                    track.bboxes.append(det.bbox)
                    track.mask_paths.append(det.mask_path)
                    track.confidences.append(det.confidence)
                    track.last_frame_index = sorted_frames.index(kf)
                    track.total_observations += 1
                else:
                    # Create new track
                    track_id = f"track_{next_track_id:04d}"
                    next_track_id += 1

                    tracks[track_id] = TrackInfo(
                        track_id=track_id,
                        concept_label=det.concept,
                        frame_ids=[det.frame_id],
                        bboxes=[det.bbox],
                        mask_paths=[det.mask_path],
                        confidences=[det.confidence],
                        is_dynamic=det.is_dynamic,
                        first_frame_index=sorted_frames.index(kf),
                        last_frame_index=sorted_frames.index(kf),
                        total_observations=1,
                    )

        # Filter short tracks
        min_views = self.config.min_object_views
        tracks = {
            tid: track for tid, track in tracks.items()
            if track.total_observations >= min_views or track.is_dynamic
        }

        return list(tracks.values())

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x1, y1, x2, y2) format
        box1 = (x1, y1, x1 + w1, y1 + h1)
        box2 = (x2, y2, x2 + w2, y2 + h2)

        # Compute intersection
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _save_tracking_data(
        self,
        tracks: List[TrackInfo],
        detections: Dict[str, List[SAM3Detection]],
        output_dir: Path,
    ) -> None:
        """Save tracking data to JSON."""
        # Save tracks
        tracks_data = [
            {
                "track_id": t.track_id,
                "concept": t.concept_label,
                "frame_ids": t.frame_ids,
                "bboxes": [list(b) for b in t.bboxes],
                "mask_paths": t.mask_paths,
                "confidences": t.confidences,
                "is_dynamic": t.is_dynamic,
                "first_frame": t.first_frame_index,
                "last_frame": t.last_frame_index,
                "total_observations": t.total_observations,
            }
            for t in tracks
        ]
        (output_dir / "tracks.json").write_text(
            json.dumps({"tracks": tracks_data}, indent=2)
        )

        # Save COCO-format annotations
        annotations = self._generate_coco_annotations(detections)
        (output_dir / "annotations.json").write_text(
            json.dumps(annotations, indent=2)
        )

    def _generate_coco_annotations(
        self,
        detections: Dict[str, List[SAM3Detection]],
    ) -> Dict[str, Any]:
        """Generate COCO-format annotations."""
        images = []
        annotations = []
        categories = {}

        ann_id = 0
        for frame_id, frame_dets in detections.items():
            images.append({
                "id": frame_id,
                "file_name": f"{frame_id}.png",
            })

            for det in frame_dets:
                if det.concept not in categories:
                    categories[det.concept] = len(categories) + 1

                annotations.append({
                    "id": ann_id,
                    "image_id": frame_id,
                    "category_id": categories[det.concept],
                    "segmentation": {"mask_file": det.mask_path},
                    "bbox": list(det.bbox),
                    "area": det.area,
                    "is_dynamic": det.is_dynamic,
                    "confidence": det.confidence,
                })
                ann_id += 1

        return {
            "info": {"description": "SAM3 segmentation", "version": "1.0"},
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": cat_id, "name": cat_name}
                for cat_name, cat_id in categories.items()
            ],
        }

    def _run_fallback(
        self,
        keyframes: List[FrameMetadata],
        frames_dir: Path,
        output_dir: Path,
    ) -> TracksResult:
        """Fallback when SAM3 is not available."""
        print("Using fallback (no segmentation)")
        return TracksResult(
            tracks=[],
            dynamic_mask_paths={},
            detections_per_frame={},
            success=True,
            errors=["SAM3 not available, segmentation skipped"],
        )
