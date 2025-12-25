"""Dynamic object masking using Segment Anything (SAM) and other detection methods.

This module provides multiple approaches for detecting and masking dynamic objects:
1. SAM (Segment Anything Model) - Most accurate, requires model weights
2. YOLO-based detection - Fast object detection
3. Optical flow based - Detects motion without ML models
4. Simple person detector - OpenCV cascade classifier fallback
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check available backends
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class MaskConfig:
    """Configuration for dynamic mask generation."""
    # Backend selection
    backend: str = "auto"  # auto, sam, yolo, optical_flow, cascade

    # SAM settings
    sam_model_type: str = "vit_b"  # vit_h, vit_l, vit_b
    sam_checkpoint: Optional[str] = None

    # Detection settings
    classes_to_mask: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "bird", "horse"
    ])

    # Mask processing
    dilate_kernel_size: int = 15
    min_area_ratio: float = 0.001  # Minimum mask area as ratio of image
    temporal_smoothing: bool = True
    temporal_window: int = 3

    # Output
    save_visualizations: bool = False


class DynamicMaskGenerator:
    """Generator for dynamic object masks."""

    def __init__(self, config: MaskConfig = None):
        self.config = config or MaskConfig()
        self._backend = None
        self._sam_predictor = None
        self._yolo_model = None
        self._cascade_classifier = None

    def initialize(self) -> bool:
        """Initialize the mask generator backend."""
        backend = self.config.backend

        if backend == "auto":
            # Try backends in order of preference
            if self._try_init_sam():
                self._backend = "sam"
                return True
            if self._try_init_yolo():
                self._backend = "yolo"
                return True
            if self._try_init_cascade():
                self._backend = "cascade"
                return True
            if self._try_init_optical_flow():
                self._backend = "optical_flow"
                return True
            logger.warning("No masking backend available, masks will be empty")
            self._backend = "none"
            return False

        elif backend == "sam":
            if self._try_init_sam():
                self._backend = "sam"
                return True
            raise RuntimeError("SAM backend requested but not available")

        elif backend == "yolo":
            if self._try_init_yolo():
                self._backend = "yolo"
                return True
            raise RuntimeError("YOLO backend requested but not available")

        elif backend == "cascade":
            if self._try_init_cascade():
                self._backend = "cascade"
                return True
            raise RuntimeError("Cascade backend requested but not available")

        elif backend == "optical_flow":
            if self._try_init_optical_flow():
                self._backend = "optical_flow"
                return True
            raise RuntimeError("Optical flow backend requested but not available")

        return False

    def _try_init_sam(self) -> bool:
        """Try to initialize Segment Anything Model."""
        if not TORCH_AVAILABLE:
            return False

        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Look for checkpoint
            checkpoint = self.config.sam_checkpoint
            if not checkpoint:
                # Try common locations
                for path in [
                    Path.home() / ".cache" / "sam" / f"sam_{self.config.sam_model_type}.pth",
                    Path("models") / f"sam_{self.config.sam_model_type}.pth",
                ]:
                    if path.exists():
                        checkpoint = str(path)
                        break

            if not checkpoint or not Path(checkpoint).exists():
                logger.info("SAM checkpoint not found")
                return False

            # Load model
            sam = sam_model_registry[self.config.sam_model_type](checkpoint=checkpoint)
            if torch.cuda.is_available():
                sam.cuda()

            self._sam_predictor = SamPredictor(sam)
            logger.info(f"Initialized SAM with {self.config.sam_model_type}")
            return True

        except ImportError:
            logger.debug("segment_anything package not installed")
            return False
        except Exception as e:
            logger.debug(f"SAM initialization failed: {e}")
            return False

    def _try_init_yolo(self) -> bool:
        """Try to initialize YOLO for object detection."""
        if not TORCH_AVAILABLE:
            return False

        try:
            from ultralytics import YOLO

            # Use YOLOv8 for detection
            self._yolo_model = YOLO("yolov8n.pt")  # Nano model for speed
            logger.info("Initialized YOLO for object detection")
            return True

        except ImportError:
            logger.debug("ultralytics package not installed")
            return False
        except Exception as e:
            logger.debug(f"YOLO initialization failed: {e}")
            return False

    def _try_init_cascade(self) -> bool:
        """Try to initialize OpenCV cascade classifier."""
        if not CV2_AVAILABLE:
            return False

        try:
            # Load pre-trained cascade for people detection
            cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
            self._cascade_classifier = cv2.CascadeClassifier(cascade_path)

            if self._cascade_classifier.empty():
                return False

            logger.info("Initialized OpenCV cascade classifier")
            return True

        except Exception as e:
            logger.debug(f"Cascade initialization failed: {e}")
            return False

    def _try_init_optical_flow(self) -> bool:
        """Check if optical flow is available."""
        return CV2_AVAILABLE

    def generate_mask(
        self,
        image: Union[np.ndarray, Path, str],
        prev_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate dynamic object mask for a single image.

        Args:
            image: Input image (array, path, or string path)
            prev_image: Previous frame for motion-based detection

        Returns:
            Binary mask where 255 = dynamic region to mask out
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            if CV2_AVAILABLE:
                img = cv2.imread(str(image))
                if img is None:
                    return np.zeros((480, 640), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif PIL_AVAILABLE:
                img = np.array(Image.open(image).convert("RGB"))
            else:
                return np.zeros((480, 640), dtype=np.uint8)
        else:
            img = image

        h, w = img.shape[:2]

        # Generate mask based on backend
        if self._backend == "sam":
            mask = self._generate_sam_mask(img)
        elif self._backend == "yolo":
            mask = self._generate_yolo_mask(img)
        elif self._backend == "cascade":
            mask = self._generate_cascade_mask(img)
        elif self._backend == "optical_flow" and prev_image is not None:
            mask = self._generate_optical_flow_mask(img, prev_image)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Post-process mask
        mask = self._postprocess_mask(mask)

        return mask

    def _generate_sam_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate mask using Segment Anything Model."""
        if self._sam_predictor is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        self._sam_predictor.set_image(image)

        # Use automatic mask generation for people
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Generate point prompts in a grid
        grid_size = 32
        points = []
        for y in range(grid_size // 2, h, grid_size):
            for x in range(grid_size // 2, w, grid_size):
                points.append([x, y])

        if not points:
            return mask

        points = np.array(points)
        labels = np.ones(len(points))  # All foreground

        try:
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            # Use highest scoring mask
            best_mask = masks[np.argmax(scores)]
            mask = (best_mask * 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"SAM prediction failed: {e}")

        return mask

    def _generate_yolo_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate mask using YOLO object detection."""
        if self._yolo_model is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        try:
            # Run detection
            results = self._yolo_model(image, verbose=False)

            for result in results:
                boxes = result.boxes

                for i, box in enumerate(boxes):
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self._yolo_model.names[class_id]

                    # Check if this class should be masked
                    if class_name.lower() in [c.lower() for c in self.config.classes_to_mask]:
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Fill mask region
                        mask[y1:y2, x1:x2] = 255

                        # If segmentation is available, use it
                        if hasattr(result, 'masks') and result.masks is not None:
                            seg_mask = result.masks[i].data.cpu().numpy()[0]
                            seg_mask = cv2.resize(
                                (seg_mask * 255).astype(np.uint8),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST
                            )
                            mask = np.maximum(mask, seg_mask)

        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")

        return mask

    def _generate_cascade_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate mask using OpenCV cascade classifier."""
        if self._cascade_classifier is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        try:
            # Detect people
            bodies = self._cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            for (x, y, bw, bh) in bodies:
                # Expand bounding box slightly
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + bw + pad)
                y2 = min(h, y + bh + pad)

                mask[y1:y2, x1:x2] = 255

        except Exception as e:
            logger.warning(f"Cascade detection failed: {e}")

        return mask

    def _generate_optical_flow_mask(
        self,
        image: np.ndarray,
        prev_image: np.ndarray,
    ) -> np.ndarray:
        """Generate mask using optical flow motion detection."""
        if not CV2_AVAILABLE:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Calculate flow magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Threshold to find moving regions
            # Use adaptive threshold based on mean motion
            mean_mag = np.mean(mag)
            threshold = max(mean_mag * 3, 2.0)  # At least 2 pixels of motion

            motion_mask = (mag > threshold).astype(np.uint8) * 255
            mask = motion_mask

        except Exception as e:
            logger.warning(f"Optical flow failed: {e}")

        return mask

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process mask with morphological operations."""
        if not CV2_AVAILABLE:
            return mask

        # Dilate to expand mask regions
        kernel_size = self.config.dilate_kernel_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove small regions
        h, w = mask.shape
        min_area = int(h * w * self.config.min_area_ratio)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        return filtered_mask

    def generate_masks_for_sequence(
        self,
        image_paths: List[Path],
        output_dir: Path,
        progress_callback: callable = None,
    ) -> Dict[str, Path]:
        """Generate masks for a sequence of images.

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save masks
            progress_callback: Optional callback(current, total)

        Returns:
            Dict mapping frame_id to mask path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize if needed
        if self._backend is None:
            self.initialize()

        masks = {}
        prev_image = None

        for i, img_path in enumerate(image_paths):
            frame_id = img_path.stem

            # Load image
            if CV2_AVAILABLE:
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif PIL_AVAILABLE:
                image = np.array(Image.open(img_path).convert("RGB"))
            else:
                continue

            if image is None:
                continue

            # Generate mask
            mask = self.generate_mask(image, prev_image)

            # Temporal smoothing
            if self.config.temporal_smoothing and i > 0:
                mask = self._temporal_smooth(mask, masks, image_paths, i)

            # Save mask
            mask_path = output_dir / f"{frame_id}_mask.png"
            if CV2_AVAILABLE:
                cv2.imwrite(str(mask_path), mask)
            elif PIL_AVAILABLE:
                Image.fromarray(mask).save(mask_path)

            masks[frame_id] = mask_path

            # Save visualization if requested
            if self.config.save_visualizations:
                self._save_visualization(image, mask, output_dir / f"{frame_id}_vis.png")

            prev_image = image

            if progress_callback:
                progress_callback(i + 1, len(image_paths))

        logger.info(f"Generated {len(masks)} dynamic masks")
        return masks

    def _temporal_smooth(
        self,
        current_mask: np.ndarray,
        masks_dict: Dict[str, Path],
        image_paths: List[Path],
        current_idx: int,
    ) -> np.ndarray:
        """Apply temporal smoothing to mask."""
        if not CV2_AVAILABLE:
            return current_mask

        window = self.config.temporal_window
        start_idx = max(0, current_idx - window)

        accumulated = current_mask.astype(np.float32)
        count = 1

        for i in range(start_idx, current_idx):
            frame_id = image_paths[i].stem
            if frame_id in masks_dict:
                prev_mask = cv2.imread(str(masks_dict[frame_id]), cv2.IMREAD_GRAYSCALE)
                if prev_mask is not None:
                    accumulated += prev_mask.astype(np.float32)
                    count += 1

        # Average and threshold
        averaged = accumulated / count
        smoothed = (averaged > 127).astype(np.uint8) * 255

        return smoothed

    def _save_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save visualization of image with mask overlay."""
        if not CV2_AVAILABLE:
            return

        # Create colored overlay
        vis = image.copy()
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 0] = 255  # Red channel

        # Blend where mask is active
        alpha = 0.4
        mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0
        vis = (vis * (1 - mask_3ch * alpha) + red_overlay * mask_3ch * alpha).astype(np.uint8)

        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def generate_dynamic_masks(
    image_paths: List[Path],
    output_dir: Path,
    config: MaskConfig = None,
    progress_callback: callable = None,
) -> Dict[str, Path]:
    """Convenience function to generate masks for a sequence.

    Args:
        image_paths: List of image file paths
        output_dir: Directory to save masks
        config: Mask generation configuration
        progress_callback: Optional callback(current, total)

    Returns:
        Dict mapping frame_id to mask path
    """
    generator = DynamicMaskGenerator(config)
    return generator.generate_masks_for_sequence(
        image_paths, output_dir, progress_callback
    )
