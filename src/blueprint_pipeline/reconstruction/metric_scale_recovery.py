"""Metric Scale Recovery for RGB-Only Captures.

This module provides marker-free metric scale estimation using:
1. Foundation depth models (Depth Pro, Metric3D v2, UniDepth)
2. Semantic scale anchors (doors, cars, people, furniture)
3. Person height priors with pose estimation

This eliminates the need for ArUco/AprilTag markers for RGB-only captures
from Meta Glasses, iOS (without ARKit), and Android devices.

References:
- Depth Pro: https://github.com/apple/ml-depth-pro (ICLR 2025)
- Metric3D v2: https://github.com/YvanYin/Metric3D (TPAMI 2024)
- UniDepth: https://github.com/lpiccinelli-eth/UniDepth (CVPR 2024)
- MoGe-2: https://arxiv.org/abs/2507.02546 (July 2025)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class MetricDepthModel(Enum):
    """Available metric depth foundation models."""
    DEPTH_PRO = "depth_pro"           # Apple's Depth Pro (ICLR 2025) - fastest
    METRIC3D_V2 = "metric3d_v2"       # Metric3D v2 (TPAMI 2024) - proven SLAM integration
    UNIDEPTH_V2 = "unidepth_v2"       # UniDepth v2 - with uncertainty
    MOGE2 = "moge2"                   # MoGe-2 (July 2025) - latest
    AUTO = "auto"                     # Auto-select best available


# Known object sizes in meters (used as semantic scale anchors)
SEMANTIC_SCALE_ANCHORS = {
    # Doors (highly reliable - standardized sizes)
    "door": {
        "height": 2.03,        # Standard door height (80 inches)
        "width": 0.91,         # Standard door width (36 inches)
        "confidence": 0.95,
    },
    "door_residential": {
        "height": 2.03,
        "width": 0.81,         # 32 inch common
        "confidence": 0.90,
    },
    "door_commercial": {
        "height": 2.13,        # 84 inches
        "width": 0.91,
        "confidence": 0.90,
    },
    # People (good anchor when standing upright)
    "person": {
        "height": 1.70,        # Global average adult height
        "confidence": 0.70,    # Lower confidence due to variation
    },
    "person_standing": {
        "height": 1.70,
        "confidence": 0.75,
    },
    # Vehicles (very reliable in parking/street scenes)
    "car": {
        "height": 1.45,        # Average sedan height
        "length": 4.5,         # Average sedan length
        "width": 1.8,          # Average sedan width
        "confidence": 0.85,
    },
    "suv": {
        "height": 1.75,
        "length": 4.8,
        "width": 1.9,
        "confidence": 0.85,
    },
    # Furniture (indoor scenes)
    "chair": {
        "seat_height": 0.45,   # Standard seat height
        "confidence": 0.70,
    },
    "dining_table": {
        "height": 0.76,        # Standard table height (30 inches)
        "confidence": 0.75,
    },
    "desk": {
        "height": 0.74,        # Standard desk height (29 inches)
        "confidence": 0.75,
    },
    "couch": {
        "seat_height": 0.45,
        "back_height": 0.85,
        "confidence": 0.70,
    },
    # Retail/Commercial
    "shopping_cart": {
        "height": 1.0,
        "length": 1.0,
        "confidence": 0.80,
    },
    "refrigerator_commercial": {
        "height": 2.0,
        "width": 0.75,
        "confidence": 0.85,
    },
    # Stairs (very reliable)
    "stair_step": {
        "height": 0.18,        # Standard step height (7 inches)
        "depth": 0.28,         # Standard step depth (11 inches)
        "confidence": 0.90,
    },
    # Road markings (outdoor)
    "lane_width": {
        "width": 3.7,          # US standard lane width (12 feet)
        "confidence": 0.85,
    },
    "crosswalk_stripe": {
        "width": 0.15,         # Standard stripe width (6 inches)
        "confidence": 0.80,
    },
}


@dataclass
class MetricScaleConfig:
    """Configuration for metric scale recovery.

    The system uses a multi-modal approach:
    1. Primary: Foundation depth models (most accurate)
    2. Secondary: Semantic object detection (reliable anchors)
    3. Tertiary: Person height priors (always available fallback)

    All methods are fused with confidence weighting.
    """

    # Model selection
    depth_model: MetricDepthModel = MetricDepthModel.AUTO
    device: str = "cuda"
    dtype: str = "float16"

    # Depth model parameters
    depth_model_batch_size: int = 4  # Process multiple frames
    depth_confidence_threshold: float = 0.7  # Min confidence for depth

    # Semantic anchor detection
    enable_semantic_anchors: bool = True
    semantic_detector: str = "yolo"  # "yolo" or "detic" or "grounding_dino"
    semantic_confidence_threshold: float = 0.5
    min_anchor_observations: int = 3  # Min detections to use anchor

    # Person height prior
    enable_person_prior: bool = True
    person_height_mean: float = 1.70  # meters (global average)
    person_height_std: float = 0.10   # standard deviation
    use_pose_estimation: bool = True  # Use MediaPipe/YOLO for better height

    # Scale fusion
    fusion_method: str = "weighted_median"  # "mean", "median", "weighted_median"
    min_scale_confidence: float = 0.5  # Min confidence to report scale

    # Output
    save_debug_visualizations: bool = False


@dataclass
class ScaleEstimate:
    """A single scale estimate from any source."""
    scale_factor: float       # Multiply SLAM units by this to get meters
    confidence: float         # 0-1 confidence in this estimate
    source: str               # "depth_model", "semantic_door", "person_height", etc.
    frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScaleResult:
    """Result of metric scale recovery."""
    scale_factor: float = 1.0
    confidence: float = 0.0
    is_metric: bool = False  # True if we achieved metric scale

    # Individual estimates
    estimates: List[ScaleEstimate] = field(default_factory=list)

    # Source breakdown
    depth_model_scale: Optional[float] = None
    depth_model_confidence: float = 0.0
    semantic_scale: Optional[float] = None
    semantic_confidence: float = 0.0
    person_prior_scale: Optional[float] = None
    person_prior_confidence: float = 0.0

    # Diagnostics
    frames_processed: int = 0
    anchors_detected: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale_factor": self.scale_factor,
            "confidence": self.confidence,
            "is_metric": self.is_metric,
            "depth_model_scale": self.depth_model_scale,
            "semantic_scale": self.semantic_scale,
            "person_prior_scale": self.person_prior_scale,
            "frames_processed": self.frames_processed,
            "anchors_detected": self.anchors_detected,
            "warnings": self.warnings,
        }


# =============================================================================
# Foundation Depth Models
# =============================================================================

class BaseDepthModel(ABC):
    """Base class for metric depth estimation models."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict metric depth from an image.

        Args:
            image: RGB image as numpy array [H, W, 3]

        Returns:
            Tuple of (depth_map [H, W], confidence)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this model is available."""
        pass


class DepthProModel(BaseDepthModel):
    """Apple's Depth Pro model (ICLR 2025).

    Key features:
    - Zero-shot metric depth (no camera metadata needed)
    - 0.3s inference for 2.25MP images
    - State-of-the-art boundary accuracy

    Install: pip install depth-pro
    Ref: https://github.com/apple/ml-depth-pro
    """

    def __init__(self, device: str = "cuda", dtype: str = "float16"):
        self.device = device
        self.dtype = dtype
        self._model = None
        self._transform = None

    def is_available(self) -> bool:
        try:
            import depth_pro
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        import depth_pro

        logger.info("Loading Depth Pro model...")
        self._model, self._transform = depth_pro.create_model_and_transforms(
            device=self.device,
            precision=torch.float16 if self.dtype == "float16" else torch.float32,
        )
        self._model.eval()
        logger.info("Depth Pro model loaded")

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        import torch
        from PIL import Image

        self._load_model()

        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Transform and predict
        image_tensor = self._transform(pil_image).to(self.device)

        with torch.no_grad():
            prediction = self._model.infer(image_tensor)

        depth = prediction["depth"].cpu().numpy()

        # Depth Pro provides focal length estimate which indicates confidence
        focallength_px = prediction.get("focallength_px", None)
        confidence = 0.9 if focallength_px is not None else 0.8

        return depth, confidence


class Metric3Dv2Model(BaseDepthModel):
    """Metric3D v2 model (TPAMI 2024).

    Key features:
    - Zero-shot metric depth and surface normals
    - Canonical camera space for cross-camera generalization
    - Proven SLAM integration

    Install: pip install torch torchvision
    Ref: https://github.com/YvanYin/Metric3D
    """

    def __init__(self, device: str = "cuda", dtype: str = "float16",
                 backbone: str = "vit_large"):
        self.device = device
        self.dtype = dtype
        self.backbone = backbone  # vit_small, vit_large, vit_giant2
        self._model = None

    def is_available(self) -> bool:
        try:
            import torch
            # Check if we can load via torch hub
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return

        import torch

        logger.info(f"Loading Metric3D v2 ({self.backbone}) model...")

        # Load via torch hub
        model_name = f"metric3d_{self.backbone}"
        self._model = torch.hub.load(
            'yvanyin/metric3d',
            model_name,
            pretrain=True,
            trust_repo=True,
        )
        self._model.to(self.device)
        self._model.eval()

        logger.info("Metric3D v2 model loaded")

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        import torch
        import torch.nn.functional as F

        self._load_model()

        # Prepare image
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image - mean) / std

        # To tensor [1, 3, H, W]
        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)

        # Resize to model input size
        h, w = tensor.shape[2:]
        input_size = (616, 1064)  # Metric3D v2 default
        tensor_resized = F.interpolate(tensor, size=input_size, mode='bilinear')

        with torch.no_grad():
            pred_depth, confidence, _ = self._model.inference({'input': tensor_resized})

        # Resize back
        pred_depth = F.interpolate(
            pred_depth.unsqueeze(1), size=(h, w), mode='bilinear'
        ).squeeze().cpu().numpy()

        conf_value = confidence.mean().item() if confidence is not None else 0.85

        return pred_depth, conf_value


class UniDepthModel(BaseDepthModel):
    """UniDepth v2 model (CVPR 2024).

    Key features:
    - Universal metric depth across domains
    - Uncertainty estimation for downstream tasks
    - Edge-guided loss for sharp boundaries

    Ref: https://github.com/lpiccinelli-eth/UniDepth
    """

    def __init__(self, device: str = "cuda", dtype: str = "float16"):
        self.device = device
        self.dtype = dtype
        self._model = None

    def is_available(self) -> bool:
        try:
            from unidepth.models import UniDepthV2
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return

        from unidepth.models import UniDepthV2
        import torch

        logger.info("Loading UniDepth v2 model...")
        self._model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        self._model.to(self.device)
        self._model.eval()
        logger.info("UniDepth v2 model loaded")

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        import torch
        from PIL import Image

        self._load_model()

        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        with torch.no_grad():
            predictions = self._model.infer(pil_image)

        depth = predictions["depth"].cpu().numpy()
        confidence = 1.0 - predictions.get("uncertainty", torch.zeros(1)).mean().item()

        return depth, max(0.5, confidence)


def get_depth_model(
    model_type: MetricDepthModel,
    device: str = "cuda",
    dtype: str = "float16",
) -> Optional[BaseDepthModel]:
    """Get the best available depth model.

    Args:
        model_type: Requested model type or AUTO for best available
        device: Device to run on
        dtype: Data type for inference

    Returns:
        Depth model instance or None if none available
    """
    # Priority order for AUTO selection
    priority = [
        (MetricDepthModel.DEPTH_PRO, DepthProModel),
        (MetricDepthModel.METRIC3D_V2, Metric3Dv2Model),
        (MetricDepthModel.UNIDEPTH_V2, UniDepthModel),
    ]

    if model_type != MetricDepthModel.AUTO:
        # Try to get the specific requested model
        for mtype, mclass in priority:
            if mtype == model_type:
                model = mclass(device=device, dtype=dtype)
                if model.is_available():
                    return model
                else:
                    logger.warning(f"{model_type.value} not available")
                    break

    # AUTO: try each model in priority order
    for mtype, mclass in priority:
        model = mclass(device=device, dtype=dtype)
        if model.is_available():
            logger.info(f"Using {mtype.value} for metric depth estimation")
            return model

    logger.warning("No metric depth model available")
    return None


# =============================================================================
# Semantic Scale Anchor Detection
# =============================================================================

class SemanticAnchorDetector:
    """Detect objects with known sizes for scale estimation.

    Uses YOLO or other object detectors to find common objects
    (doors, cars, people, furniture) and uses their known sizes
    as scale references.
    """

    def __init__(self, config: MetricScaleConfig):
        self.config = config
        self._detector = None
        self._pose_estimator = None

    def _load_detector(self):
        if self._detector is not None:
            return

        try:
            from ultralytics import YOLO

            # Use YOLOv8 for detection
            self._detector = YOLO("yolov8l.pt")  # Large model for accuracy
            logger.info("YOLO detector loaded")
        except ImportError:
            logger.warning("ultralytics not available, semantic detection disabled")

    def _load_pose_estimator(self):
        if self._pose_estimator is not None or not self.config.use_pose_estimation:
            return

        try:
            from ultralytics import YOLO
            self._pose_estimator = YOLO("yolov8l-pose.pt")
            logger.info("YOLO pose estimator loaded")
        except ImportError:
            logger.warning("Pose estimation not available")

    def detect_anchors(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[ScaleEstimate]:
        """Detect semantic scale anchors in an image.

        Args:
            image: RGB image [H, W, 3]
            depth_map: Optional metric depth map [H, W]

        Returns:
            List of scale estimates from detected objects
        """
        self._load_detector()
        if self._detector is None:
            return []

        estimates = []

        # Run YOLO detection
        results = self._detector(image, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < self.config.semantic_confidence_threshold:
                    continue

                # Get class name
                cls_name = result.names[cls_id]

                # Check if this is a known anchor
                estimate = self._process_detection(
                    cls_name, box, image.shape, depth_map, conf
                )

                if estimate is not None:
                    estimates.append(estimate)

        # Also try person height estimation with pose
        if self.config.enable_person_prior:
            person_estimates = self._estimate_person_heights(image, depth_map)
            estimates.extend(person_estimates)

        return estimates

    def _process_detection(
        self,
        cls_name: str,
        box: Any,
        image_shape: Tuple[int, ...],
        depth_map: Optional[np.ndarray],
        detection_conf: float,
    ) -> Optional[ScaleEstimate]:
        """Process a detection and compute scale estimate."""
        H, W = image_shape[:2]

        # Get bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_h = y2 - y1
        box_w = x2 - x1

        # Map YOLO class to our anchor database
        anchor_key = self._map_class_to_anchor(cls_name)
        if anchor_key is None:
            return None

        anchor_info = SEMANTIC_SCALE_ANCHORS.get(anchor_key)
        if anchor_info is None:
            return None

        # Get expected metric size
        if "height" in anchor_info:
            metric_size = anchor_info["height"]
            pixel_size = box_h
        elif "width" in anchor_info:
            metric_size = anchor_info["width"]
            pixel_size = box_w
        else:
            return None

        # If we have depth, compute scale directly
        if depth_map is not None:
            # Get median depth in the bounding box
            box_depth = depth_map[int(y1):int(y2), int(x1):int(x2)]
            if box_depth.size > 0:
                median_depth = np.median(box_depth[box_depth > 0])
                if median_depth > 0:
                    # Object apparent size: pixel_size * depth / focal_length
                    # For now, use a simplified estimate
                    scale_factor = metric_size / (pixel_size / H * median_depth * 2)

                    confidence = (
                        anchor_info["confidence"] *
                        detection_conf *
                        0.9  # Depth-based confidence boost
                    )

                    return ScaleEstimate(
                        scale_factor=scale_factor,
                        confidence=confidence,
                        source=f"semantic_{anchor_key}_depth",
                        metadata={
                            "class": cls_name,
                            "anchor": anchor_key,
                            "metric_size": metric_size,
                            "pixel_size": pixel_size,
                            "depth": float(median_depth),
                        }
                    )

        # Without depth, we can't compute absolute scale from detection alone
        # But we can record the observation for relative comparison
        return ScaleEstimate(
            scale_factor=1.0,  # Unknown without depth
            confidence=0.0,   # Can't determine without depth
            source=f"semantic_{anchor_key}_nodepth",
            metadata={
                "class": cls_name,
                "anchor": anchor_key,
                "metric_size": metric_size,
                "pixel_size": pixel_size,
                "needs_depth": True,
            }
        )

    def _map_class_to_anchor(self, cls_name: str) -> Optional[str]:
        """Map YOLO class name to our anchor database key."""
        cls_lower = cls_name.lower()

        mapping = {
            "person": "person",
            "car": "car",
            "truck": "car",
            "bus": "car",
            "chair": "chair",
            "couch": "couch",
            "sofa": "couch",
            "dining table": "dining_table",
            "desk": "desk",
            "refrigerator": "refrigerator_commercial",
        }

        return mapping.get(cls_lower)

    def _estimate_person_heights(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> List[ScaleEstimate]:
        """Estimate scale from detected people using pose estimation."""
        if not self.config.use_pose_estimation:
            return []

        self._load_pose_estimator()
        if self._pose_estimator is None:
            return []

        estimates = []
        H, W = image.shape[:2]

        results = self._pose_estimator(image, verbose=False)

        for result in results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

            for person_idx, person_kpts in enumerate(keypoints):
                # Get head and feet positions
                # COCO keypoints: 0=nose, 15=left_ankle, 16=right_ankle
                nose = person_kpts[0] if len(person_kpts) > 0 else None
                left_ankle = person_kpts[15] if len(person_kpts) > 15 else None
                right_ankle = person_kpts[16] if len(person_kpts) > 16 else None

                # Need at least nose and one ankle
                if nose is None or (left_ankle is None and right_ankle is None):
                    continue

                # Check confidence
                if confidences is not None:
                    nose_conf = confidences[person_idx][0]
                    ankle_conf = max(
                        confidences[person_idx][15] if len(confidences[person_idx]) > 15 else 0,
                        confidences[person_idx][16] if len(confidences[person_idx]) > 16 else 0,
                    )
                    if nose_conf < 0.5 or ankle_conf < 0.5:
                        continue
                else:
                    nose_conf = ankle_conf = 0.7

                # Compute pixel height
                ankle = left_ankle if left_ankle is not None else right_ankle
                if ankle is None or np.any(ankle == 0):
                    continue

                pixel_height = abs(ankle[1] - nose[1])

                # Add ~10% for head above nose
                pixel_height *= 1.10

                if pixel_height < H * 0.1:  # Too small, likely noise
                    continue

                # If we have depth, compute scale
                if depth_map is not None:
                    # Sample depth at torso area
                    torso_y = int((nose[1] + ankle[1]) / 2)
                    torso_x = int(nose[0])

                    if 0 <= torso_y < H and 0 <= torso_x < W:
                        person_depth = depth_map[
                            max(0, torso_y-20):min(H, torso_y+20),
                            max(0, torso_x-20):min(W, torso_x+20)
                        ]
                        median_depth = np.median(person_depth[person_depth > 0])

                        if median_depth > 0:
                            # Scale = expected_height / observed_height_in_meters
                            # observed_height ≈ pixel_height * depth / focal_length_approx
                            focal_approx = H * 0.8  # Rough approximation
                            observed_height = pixel_height * median_depth / focal_approx
                            scale_factor = self.config.person_height_mean / observed_height

                            confidence = (
                                SEMANTIC_SCALE_ANCHORS["person"]["confidence"] *
                                min(nose_conf, ankle_conf) *
                                0.85
                            )

                            estimates.append(ScaleEstimate(
                                scale_factor=scale_factor,
                                confidence=confidence,
                                source="person_pose_depth",
                                metadata={
                                    "pixel_height": pixel_height,
                                    "depth": float(median_depth),
                                    "expected_height": self.config.person_height_mean,
                                }
                            ))

        return estimates


# =============================================================================
# Scale Fusion
# =============================================================================

class MetricScaleFusion:
    """Fuse multiple scale estimates into a single robust estimate."""

    def __init__(self, config: MetricScaleConfig):
        self.config = config

    def fuse(self, estimates: List[ScaleEstimate]) -> MetricScaleResult:
        """Fuse multiple scale estimates.

        Args:
            estimates: List of scale estimates from various sources

        Returns:
            Fused metric scale result
        """
        result = MetricScaleResult()
        result.estimates = estimates

        if not estimates:
            result.warnings.append("No scale estimates available")
            return result

        # Filter valid estimates (positive scale, non-zero confidence)
        valid_estimates = [
            e for e in estimates
            if e.scale_factor > 0 and e.confidence > 0
        ]

        if not valid_estimates:
            result.warnings.append("No valid scale estimates after filtering")
            return result

        # Group by source type
        depth_estimates = [e for e in valid_estimates if "depth_model" in e.source]
        semantic_estimates = [e for e in valid_estimates if "semantic" in e.source]
        person_estimates = [e for e in valid_estimates if "person" in e.source]

        # Compute per-source scales
        if depth_estimates:
            result.depth_model_scale, result.depth_model_confidence = self._aggregate(depth_estimates)

        if semantic_estimates:
            result.semantic_scale, result.semantic_confidence = self._aggregate(semantic_estimates)

        if person_estimates:
            result.person_prior_scale, result.person_prior_confidence = self._aggregate(person_estimates)

        # Record anchor detections
        for e in valid_estimates:
            source = e.source.split("_")[1] if "_" in e.source else e.source
            result.anchors_detected[source] = result.anchors_detected.get(source, 0) + 1

        # Final fusion
        if self.config.fusion_method == "weighted_median":
            result.scale_factor, result.confidence = self._weighted_median(valid_estimates)
        elif self.config.fusion_method == "median":
            result.scale_factor = np.median([e.scale_factor for e in valid_estimates])
            result.confidence = np.mean([e.confidence for e in valid_estimates])
        else:  # mean
            result.scale_factor, result.confidence = self._aggregate(valid_estimates)

        # Mark as metric if confidence exceeds threshold
        result.is_metric = result.confidence >= self.config.min_scale_confidence

        if not result.is_metric:
            result.warnings.append(
                f"Scale confidence {result.confidence:.2f} below threshold "
                f"{self.config.min_scale_confidence}"
            )

        return result

    def _aggregate(self, estimates: List[ScaleEstimate]) -> Tuple[float, float]:
        """Compute weighted average of estimates."""
        if not estimates:
            return 1.0, 0.0

        total_weight = sum(e.confidence for e in estimates)
        if total_weight == 0:
            return 1.0, 0.0

        weighted_scale = sum(e.scale_factor * e.confidence for e in estimates) / total_weight
        avg_confidence = total_weight / len(estimates)

        return weighted_scale, avg_confidence

    def _weighted_median(self, estimates: List[ScaleEstimate]) -> Tuple[float, float]:
        """Compute weighted median of estimates (more robust to outliers)."""
        if not estimates:
            return 1.0, 0.0

        # Sort by scale factor
        sorted_estimates = sorted(estimates, key=lambda e: e.scale_factor)

        total_weight = sum(e.confidence for e in sorted_estimates)
        if total_weight == 0:
            return 1.0, 0.0

        # Find weighted median
        cumsum = 0
        for e in sorted_estimates:
            cumsum += e.confidence
            if cumsum >= total_weight / 2:
                avg_confidence = total_weight / len(estimates)
                return e.scale_factor, avg_confidence

        # Fallback
        return sorted_estimates[-1].scale_factor, total_weight / len(estimates)


# =============================================================================
# Main Pipeline
# =============================================================================

class MetricScaleRecovery:
    """Main class for metric scale recovery from RGB-only captures.

    This integrates:
    1. Foundation depth models for metric depth estimation
    2. Semantic anchor detection for known object sizes
    3. Person height priors with pose estimation

    Usage:
        config = MetricScaleConfig()
        recovery = MetricScaleRecovery(config)
        result = recovery.estimate_scale(frames, slam_poses, intrinsics)
    """

    def __init__(self, config: Optional[MetricScaleConfig] = None):
        self.config = config or MetricScaleConfig()
        self._depth_model = None
        self._anchor_detector = None
        self._fusion = MetricScaleFusion(self.config)

    def _get_depth_model(self) -> Optional[BaseDepthModel]:
        if self._depth_model is None:
            self._depth_model = get_depth_model(
                self.config.depth_model,
                self.config.device,
                self.config.dtype,
            )
        return self._depth_model

    def _get_anchor_detector(self) -> SemanticAnchorDetector:
        if self._anchor_detector is None:
            self._anchor_detector = SemanticAnchorDetector(self.config)
        return self._anchor_detector

    def estimate_scale(
        self,
        frames: List[Dict[str, Any]],
        slam_poses: Optional[List[Any]] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        output_dir: Optional[Path] = None,
    ) -> MetricScaleResult:
        """Estimate metric scale from RGB frames.

        Args:
            frames: List of frame dicts with 'image' (np.ndarray) and 'frame_id'
            slam_poses: Optional SLAM poses for scale comparison
            intrinsics: Optional camera intrinsics {fx, fy, cx, cy}
            output_dir: Optional output directory for debug visualizations

        Returns:
            MetricScaleResult with fused scale estimate
        """
        all_estimates = []
        frames_processed = 0

        depth_model = self._get_depth_model()
        anchor_detector = self._get_anchor_detector() if self.config.enable_semantic_anchors else None

        # Sample frames for processing (don't need every frame)
        sample_indices = self._select_sample_frames(len(frames))

        for idx in sample_indices:
            if idx >= len(frames):
                continue

            frame = frames[idx]
            image = frame.get("image")
            frame_id = frame.get("frame_id", str(idx))

            if image is None:
                continue

            # Ensure numpy array
            if not isinstance(image, np.ndarray):
                try:
                    import torch
                    if isinstance(image, torch.Tensor):
                        image = image.cpu().numpy()
                        if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
                            image = image.transpose(1, 2, 0)
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                except:
                    continue

            frames_processed += 1
            depth_map = None

            # 1. Get metric depth from foundation model
            if depth_model is not None:
                try:
                    depth_map, depth_conf = depth_model.predict(image)

                    # Compute scale from depth statistics
                    # For indoor scenes, median depth is often 2-5m
                    # For outdoor, it's larger
                    valid_depths = depth_map[depth_map > 0]
                    if len(valid_depths) > 0:
                        median_depth = np.median(valid_depths)

                        # This gives us absolute metric information
                        all_estimates.append(ScaleEstimate(
                            scale_factor=1.0,  # Depth is already metric
                            confidence=depth_conf,
                            source="depth_model",
                            frame_id=frame_id,
                            metadata={
                                "median_depth": float(median_depth),
                                "depth_range": [float(valid_depths.min()), float(valid_depths.max())],
                            }
                        ))
                except Exception as e:
                    logger.warning(f"Depth model failed on frame {frame_id}: {e}")

            # 2. Detect semantic anchors
            if anchor_detector is not None:
                try:
                    anchor_estimates = anchor_detector.detect_anchors(image, depth_map)
                    for est in anchor_estimates:
                        est.frame_id = frame_id
                    all_estimates.extend(anchor_estimates)
                except Exception as e:
                    logger.warning(f"Anchor detection failed on frame {frame_id}: {e}")

        # 3. If we have SLAM poses and depth, compute scale alignment
        if slam_poses and depth_model is not None:
            pose_estimates = self._compute_pose_depth_scale(
                frames, slam_poses, intrinsics
            )
            all_estimates.extend(pose_estimates)

        # Fuse all estimates
        result = self._fusion.fuse(all_estimates)
        result.frames_processed = frames_processed

        logger.info(
            f"Metric scale recovery: scale={result.scale_factor:.4f}, "
            f"confidence={result.confidence:.2f}, is_metric={result.is_metric}"
        )

        return result

    def _select_sample_frames(self, total_frames: int) -> List[int]:
        """Select a subset of frames for processing."""
        if total_frames <= 10:
            return list(range(total_frames))

        # Sample ~10 frames evenly distributed
        step = total_frames // 10
        return [i * step for i in range(10)]

    def _compute_pose_depth_scale(
        self,
        frames: List[Dict[str, Any]],
        slam_poses: List[Any],
        intrinsics: Optional[Dict[str, float]],
    ) -> List[ScaleEstimate]:
        """Compute scale by comparing SLAM trajectory to metric depth.

        This aligns the SLAM scale to the metric depth predictions.
        """
        estimates = []
        depth_model = self._get_depth_model()

        if depth_model is None or not slam_poses or len(slam_poses) < 2:
            return estimates

        # Get depth model prediction for a few frames
        depth_predictions = {}

        for i, pose in enumerate(slam_poses[:5]):
            if i >= len(frames):
                break

            frame = frames[i]
            image = frame.get("image")
            if image is None:
                continue

            try:
                depth_map, conf = depth_model.predict(image)
                depth_predictions[i] = (depth_map, conf)
            except:
                continue

        # Compare SLAM baseline to metric depth baseline
        # This is a simplified scale alignment
        if len(depth_predictions) >= 2:
            indices = sorted(depth_predictions.keys())

            for i in range(len(indices) - 1):
                idx1, idx2 = indices[i], indices[i+1]

                if idx1 >= len(slam_poses) or idx2 >= len(slam_poses):
                    continue

                # Get SLAM translation distance
                pose1, pose2 = slam_poses[idx1], slam_poses[idx2]
                t1 = np.array(pose1.translation if hasattr(pose1, 'translation') else pose1['translation'])
                t2 = np.array(pose2.translation if hasattr(pose2, 'translation') else pose2['translation'])
                slam_dist = np.linalg.norm(t2 - t1)

                if slam_dist < 0.001:
                    continue

                # Get metric depth change (rough approximation)
                depth1, conf1 = depth_predictions[idx1]
                depth2, conf2 = depth_predictions[idx2]

                # Use center depth as proxy for distance
                h, w = depth1.shape[:2]
                center_depth1 = depth1[h//2-50:h//2+50, w//2-50:w//2+50].mean()
                center_depth2 = depth2[h//2-50:h//2+50, w//2-50:w//2+50].mean()

                # This is a rough scale estimate
                # More sophisticated: use full depth alignment
                if abs(center_depth1 - center_depth2) > 0.01:
                    # Scale factor to convert SLAM to metric
                    # Note: This is simplified; full solution would use ICP or similar
                    scale_estimate = 1.0 / slam_dist * min(center_depth1, center_depth2) * 0.1

                    estimates.append(ScaleEstimate(
                        scale_factor=scale_estimate,
                        confidence=min(conf1, conf2) * 0.6,  # Lower confidence for this method
                        source="pose_depth_alignment",
                        metadata={
                            "slam_dist": float(slam_dist),
                            "depth_change": float(abs(center_depth2 - center_depth1)),
                        }
                    ))

        return estimates

    def estimate_scale_from_images(
        self,
        image_paths: List[Path],
        output_dir: Optional[Path] = None,
    ) -> MetricScaleResult:
        """Convenience method to estimate scale from image file paths.

        Args:
            image_paths: List of paths to image files
            output_dir: Optional output directory

        Returns:
            MetricScaleResult
        """
        from PIL import Image

        frames = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                frames.append({
                    "image": np.array(img),
                    "frame_id": path.stem,
                })
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

        return self.estimate_scale(frames, output_dir=output_dir)


# =============================================================================
# Integration Helper
# =============================================================================

def recover_metric_scale(
    frames: List[Dict[str, Any]],
    slam_poses: Optional[List[Any]] = None,
    intrinsics: Optional[Dict[str, float]] = None,
    config: Optional[MetricScaleConfig] = None,
) -> MetricScaleResult:
    """Convenience function to recover metric scale.

    This is the main entry point for the metric scale recovery system.

    Args:
        frames: List of frame dicts with 'image' and optionally 'frame_id'
        slam_poses: Optional list of SLAM poses
        intrinsics: Optional camera intrinsics
        config: Optional configuration

    Returns:
        MetricScaleResult with scale factor and confidence

    Example:
        >>> frames = [{"image": img1}, {"image": img2}, ...]
        >>> result = recover_metric_scale(frames)
        >>> if result.is_metric:
        ...     scaled_poses = apply_scale(poses, result.scale_factor)
    """
    recovery = MetricScaleRecovery(config)
    return recovery.estimate_scale(frames, slam_poses, intrinsics)
