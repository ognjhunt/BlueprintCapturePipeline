#!/usr/bin/env python3
"""SAM3 object detection for the BlueprintCapture swap pipeline.

Replaces ARKit's on-device object detection with Meta's SAM3 (Segment
Anything Model 3) running server-side on a GPU VM.  Produces the
``object_point_cloud_index.json`` consumed by the swap orchestrator's
candidate-selection stage.

The script:
  1. Extracts frames from the capture video (or reads pre-extracted frames)
  2. Runs SAM3 text-prompted detection across multiple sampled frames
  3. Merges per-frame detections into unique scene-level objects
  4. Estimates bounding boxes from 2D detections + COLMAP camera info
  5. Writes ``object_point_cloud_index.json`` and optional per-object masks

Usage:
  python3 sam3_detect.py \
      --frames-dir /workspace/test_scene/images \
      --output /workspace/test_scene/raw/arkit/objects/index.json \
      --environment warehouse \
      --colmap-sparse /workspace/test_scene/colmap/sparse/0
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Swap-policy keyword lists used for text prompts
_DETECTION_PROMPTS: Dict[str, List[str]] = {
    "default": [
        "shelf", "door", "cabinet", "drawer", "box", "container",
        "tote", "bin", "crate", "chair", "table", "desk",
        "refrigerator", "microwave", "oven", "dishwasher",
        "bottle", "cup", "mug", "tool",
    ],
    "warehouse": [
        "shelf", "box", "tote", "bin", "crate", "container",
        "pallet", "carton", "package", "door", "cart",
        "forklift", "rack", "barrel", "drum",
    ],
    "kitchen": [
        "cabinet", "drawer", "refrigerator", "fridge", "microwave",
        "oven", "dishwasher", "door", "mug", "cup", "bowl",
        "plate", "pot", "pan", "bottle",
    ],
}

# Objects that are structural and should be excluded from swap candidates
_STRUCTURAL_LABELS = {"wall", "floor", "ceiling", "window", "stairs"}

# Minimum detection confidence to include
_MIN_CONFIDENCE = 0.45

# How many frames to sample for detection
_DEFAULT_SAMPLE_FRAMES = 8

# IoU threshold for merging detections across frames
_MERGE_IOU_THRESHOLD = 0.35


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Tracking/association settings
_TRACKING_MODE_DEFAULT = (os.getenv("SAM3_TRACKING_MODE", "full_video") or "full_video").strip().lower()
_TRACK_MAX_FRAME_GAP = max(1, _env_int("SAM3_TRACK_MAX_FRAME_GAP", 3))
_TRACK_MIN_ASSOC_SCORE = max(0.0, min(1.0, _env_float("SAM3_TRACK_MIN_ASSOC_SCORE", 0.28)))
_MAX_REFERENCE_CROPS = max(1, _env_int("SAM3_MAX_REFERENCE_CROPS", 12))

# DA3 model source (prefer local snapshot to avoid runtime downloads)
_DA3_MODEL_ID = os.getenv("DA3_MODEL_ID", "depth-anything/DA3Metric-Large")
_DA3_MODEL_PATH = Path(os.getenv("DA3_MODEL_PATH", "/opt/da3/weights/metric_large"))
_DA3_MODEL_NAME = os.getenv("DA3_MODEL_NAME", "da3metric-large")


def _log(msg: str) -> None:
    print(f"[sam3-detect] {msg}", flush=True)


def _save_object_crop(
    image_path: Path,
    mask_np: np.ndarray,
    box: List[float],
    crops_dir: Path,
    label: str,
    frame_idx: int,
    det_idx: int,
) -> Optional[Path]:
    """Save a masked RGBA crop of the detected object.

    Extracts the bounding-box region with 5% padding, applies the
    segmentation mask as an alpha channel (transparent background),
    and resizes to max 512×512 preserving aspect ratio.

    Returns a path relative to the object index directory (for portability),
    or None on failure.
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Bounding box with 5% padding
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        pad_x = bw * 0.05
        pad_y = bh * 0.05
        x1 = max(0, int(box[0] - pad_x))
        y1 = max(0, int(box[1] - pad_y))
        x2 = min(w, int(box[2] + pad_x))
        y2 = min(h, int(box[3] + pad_y))

        crop_rgb = img.crop((x1, y1, x2, y2))

        # Build alpha from segmentation mask
        mask_full = (mask_np.astype(np.uint8) * 255)
        mask_crop = mask_full[y1:y2, x1:x2]
        alpha = Image.fromarray(mask_crop, mode="L")

        # Combine into RGBA
        crop_rgba = crop_rgb.copy().convert("RGBA")
        crop_rgba.putalpha(alpha)

        # Resize to max 512×512 preserving aspect ratio
        max_dim = 512
        cw, ch = crop_rgba.size
        if max(cw, ch) > max_dim:
            scale = max_dim / max(cw, ch)
            crop_rgba = crop_rgba.resize(
                (int(cw * scale), int(ch * scale)),
                Image.LANCZOS,
            )

        # Save
        crops_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace("/", "_").replace(" ", "_")
        filename = f"{safe_label}_{frame_idx:03d}_{det_idx:03d}.png"
        out_path = crops_dir / filename
        crop_rgba.save(out_path, "PNG")
        return Path(crops_dir.name) / filename

    except Exception as exc:
        _log(f"    Crop save failed for {label}: {exc}")
        return None


def _load_sam3():
    """Load SAM3 model and processor."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=_MIN_CONFIDENCE)
    _log(f"SAM3 loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return processor


def _load_da3():
    """Load Depth Anything v3 metric depth model."""
    from depth_anything_3.api import DepthAnything3

    if _DA3_MODEL_PATH.exists():
        model_source = str(_DA3_MODEL_PATH)
        _log(f"Loading DA3 from local path: {model_source} (model_name={_DA3_MODEL_NAME})")
        model = DepthAnything3.from_pretrained(model_source, model_name=_DA3_MODEL_NAME)
    else:
        model_source = _DA3_MODEL_ID
        _log(f"Loading DA3 from hub id: {model_source} (model_name={_DA3_MODEL_NAME})")
        model = DepthAnything3.from_pretrained(model_source, model_name=_DA3_MODEL_NAME)
    model = model.to(device=torch.device("cuda"))
    model.eval()
    _log(f"DA3-Metric loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model


def _get_metric_depth(da3_model, image, focal_px: float) -> np.ndarray:
    """Get per-pixel metric depth in meters from DA3.

    Args:
        da3_model: Loaded DA3 model
        image: PIL Image
        focal_px: Focal length in pixels (from COLMAP or estimate)

    Returns:
        depth_map: numpy array (H, W) in meters
    """
    with torch.no_grad():
        pred = da3_model.inference([image])
        raw_depth = pred.depth[0]  # shape (proc_H, proc_W)

        # Convert to metric depth: metric_depth = focal * raw / 300.0
        metric_depth = focal_px * raw_depth / 300.0

        # Resize to original image dimensions
        from PIL import Image as PILImage
        w, h = image.size
        if metric_depth.shape != (h, w):
            depth_img = PILImage.fromarray(metric_depth)
            depth_img = depth_img.resize((w, h), PILImage.BILINEAR)
            metric_depth = np.array(depth_img)

    return metric_depth


def _sample_frame_paths(frames_dir: Path, n_samples: int) -> List[Path]:
    """Select evenly-spaced frames from the directory."""
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No image files in {frames_dir}")

    if len(frames) <= n_samples:
        return frames

    indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    return [frames[i] for i in indices]


def _resolve_sampling_settings(
    *,
    environment: str,
    total_frames: int,
    requested_n_frames: int,
    requested_min_frame_detections: int,
) -> tuple[int, int]:
    """Resolve sampling/filter defaults for robust multi-frame detection."""
    env = environment.strip().lower()
    if env == "warehouse":
        auto_n_frames = 12
        auto_min_detections = 2
    elif env == "kitchen":
        auto_n_frames = 10
        auto_min_detections = 2
    else:
        auto_n_frames = _DEFAULT_SAMPLE_FRAMES
        auto_min_detections = 2

    if total_frames > 0:
        auto_n_frames = max(auto_n_frames, min(32, max(8, total_frames // 10)))

    n_frames = requested_n_frames if requested_n_frames > 0 else auto_n_frames
    min_frame_detections = (
        requested_min_frame_detections
        if requested_min_frame_detections > 0
        else auto_min_detections
    )

    if total_frames > 0:
        n_frames = max(1, min(n_frames, total_frames))
    min_frame_detections = max(1, min_frame_detections)
    return n_frames, min_frame_detections


def _detect_objects_in_frame(
    processor,
    image_path: Path,
    prompts: List[str],
    depth_map: Optional[np.ndarray] = None,
    focal_px: float = 1000.0,
    crops_dir: Optional[Path] = None,
    frame_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Run SAM3 detection on a single frame for all prompts.

    If depth_map is provided (from DA3), computes accurate 3D bounding
    boxes by masking the metric depth with SAM3 segmentation masks.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    state = processor.set_image(img)
    detections = []

    for prompt in prompts:
        processor.reset_all_prompts(state)
        result = processor.set_text_prompt(state=state, prompt=prompt)

        masks = result.get("masks")
        scores = result.get("scores")
        boxes = result.get("boxes")

        if masks is None or scores is None:
            continue

        n = masks.shape[0] if hasattr(masks, "shape") and len(masks.shape) >= 1 else 0
        for i in range(n):
            score = float(scores[i])
            if score < _MIN_CONFIDENCE:
                continue

            box = boxes[i].tolist() if boxes is not None else [0, 0, w, h]
            mask_np = masks[i].squeeze().cpu().numpy().astype(bool) if masks is not None else None

            # Compute mask centroid if available
            if mask_np is not None and mask_np.any():
                ys, xs = np.where(mask_np)
                cx, cy = float(xs.mean()), float(ys.mean())
                mask_area = int(mask_np.sum())
            else:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                mask_area = int((box[2] - box[0]) * (box[3] - box[1]))

            det = {
                "label": prompt,
                "score": score,
                "box": box,  # [x1, y1, x2, y2]
                "centroid_px": [cx, cy],
                "mask_area_px": mask_area,
                "image_size": [w, h],
                "frame_path": str(image_path),
                "frame_idx": int(frame_idx),
            }

            # If we have depth, compute 3D extent from mask + depth
            if depth_map is not None and mask_np is not None and mask_np.any():
                # Resize mask to depth map size if needed
                mask_for_depth = mask_np
                if mask_np.shape != depth_map.shape:
                    from PIL import Image as PILImg
                    mask_pil = PILImg.fromarray(mask_np.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize(
                        (depth_map.shape[1], depth_map.shape[0]),
                        PILImg.NEAREST,
                    )
                    mask_for_depth = np.array(mask_pil) > 127

                object_depths = depth_map[mask_for_depth]
                if len(object_depths) > 0:
                    median_depth = float(np.median(object_depths))
                    depth_range = float(np.percentile(object_depths, 90) -
                                       np.percentile(object_depths, 10))

                    # Convert 2D extent to 3D using depth + focal length
                    box_w_px = box[2] - box[0]
                    box_h_px = box[3] - box[1]
                    width_m = box_w_px * median_depth / focal_px
                    height_m = box_h_px * median_depth / focal_px
                    depth_m = max(depth_range, min(width_m, height_m) * 0.3)

                    # 3D center from 2D centroid + depth
                    cx_3d = (cx - w / 2) * median_depth / focal_px
                    cy_3d = (h / 2 - cy) * median_depth / focal_px
                    cz_3d = median_depth

                    det["depth_3d"] = {
                        "center": [round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4)],
                        "extents": [
                            round(max(0.02, width_m), 4),
                            round(max(0.02, height_m), 4),
                            round(max(0.02, depth_m), 4),
                        ],
                        "median_depth_m": round(median_depth, 4),
                        "depth_range_m": round(depth_range, 4),
                    }

            # Save masked RGBA crop if crops_dir is set
            if crops_dir is not None and mask_np is not None and mask_np.any():
                crop_path = _save_object_crop(
                    image_path, mask_np, box, crops_dir,
                    prompt, frame_idx, len(detections),
                )
                if crop_path is not None:
                    det["crop_path"] = crop_path.as_posix()

            detections.append(det)

    return detections


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _box_area(box: List[float]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _detections_match(det_a: Dict[str, Any], det_b: Dict[str, Any]) -> bool:
    """Cross-frame association using IoU + depth + centroid/scale cues."""
    iou = _box_iou(det_a["box"], det_b["box"])
    if iou >= _MERGE_IOU_THRESHOLD:
        return True

    area_a = _box_area(det_a["box"])
    area_b = _box_area(det_b["box"])
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    if area_ratio > 4.0:
        return False

    centroid_a = det_a.get("centroid_px", [0.0, 0.0])
    centroid_b = det_b.get("centroid_px", [0.0, 0.0])
    dx = float(centroid_a[0]) - float(centroid_b[0])
    dy = float(centroid_a[1]) - float(centroid_b[1])
    center_dist = float(np.hypot(dx, dy))
    diag_a = float(np.hypot(det_a["box"][2] - det_a["box"][0], det_a["box"][3] - det_a["box"][1]))
    diag_b = float(np.hypot(det_b["box"][2] - det_b["box"][0], det_b["box"][3] - det_b["box"][1]))
    center_dist_norm = center_dist / max(1.0, diag_a, diag_b)

    depth_a = det_a.get("depth_3d") if isinstance(det_a.get("depth_3d"), dict) else None
    depth_b = det_b.get("depth_3d") if isinstance(det_b.get("depth_3d"), dict) else None
    if depth_a is not None and depth_b is not None:
        center_a = np.array(depth_a.get("center", [0.0, 0.0, 0.0]), dtype=float)
        center_b = np.array(depth_b.get("center", [0.0, 0.0, 0.0]), dtype=float)
        ext_a = np.array(depth_a.get("extents", [0.2, 0.2, 0.2]), dtype=float)
        ext_b = np.array(depth_b.get("extents", [0.2, 0.2, 0.2]), dtype=float)

        dist_3d = float(np.linalg.norm(center_a - center_b))
        size_ref = max(float(np.max(ext_a)), float(np.max(ext_b)), 0.25)
        depth_gap = abs(float(center_a[2] - center_b[2]))

        if dist_3d <= max(0.6, 1.25 * size_ref) and center_dist_norm <= 1.2:
            return True
        if iou >= 0.15 and depth_gap <= max(0.8, 1.5 * size_ref):
            return True
        return False

    return iou >= 0.2 or (iou >= 0.1 and center_dist_norm <= 0.5 and area_ratio <= 2.5)


def _suppress_frame_duplicates(frame_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply per-frame duplicate suppression per label."""
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in frame_detections:
        by_label[str(det.get("label") or "object")].append(det)

    kept: List[Dict[str, Any]] = []
    for _, label_dets in by_label.items():
        selected: List[Dict[str, Any]] = []
        for det in sorted(label_dets, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            duplicate = False
            for prior in selected:
                iou = _box_iou(det["box"], prior["box"])
                if iou >= 0.65:
                    duplicate = True
                    break
                c0 = det.get("centroid_px", [0.0, 0.0])
                c1 = prior.get("centroid_px", [0.0, 0.0])
                d = float(np.hypot(float(c0[0]) - float(c1[0]), float(c0[1]) - float(c1[1])))
                diag = max(
                    1.0,
                    float(np.hypot(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1])),
                    float(np.hypot(prior["box"][2] - prior["box"][0], prior["box"][3] - prior["box"][1])),
                )
                if d / diag <= 0.2:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
        kept.extend(selected)
    return kept


def _track_association_score(track: Dict[str, Any], det: Dict[str, Any]) -> float:
    """Return [0,1] association score for assigning ``det`` to ``track``."""
    last = track["last_det"]
    frame_gap = int(det.get("frame_idx", 0)) - int(track.get("last_frame_idx", -1))
    if frame_gap <= 0 or frame_gap > _TRACK_MAX_FRAME_GAP:
        return -1.0

    iou = _box_iou(last["box"], det["box"])
    area_last = _box_area(last["box"])
    area_det = _box_area(det["box"])
    area_ratio = max(area_last, area_det) / max(1.0, min(area_last, area_det))
    if area_ratio > 6.0:
        return -1.0

    c0 = last.get("centroid_px", [0.0, 0.0])
    c1 = det.get("centroid_px", [0.0, 0.0])
    center_dist = float(np.hypot(float(c0[0]) - float(c1[0]), float(c0[1]) - float(c1[1])))
    diag_last = float(np.hypot(last["box"][2] - last["box"][0], last["box"][3] - last["box"][1]))
    diag_det = float(np.hypot(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]))
    center_dist_norm = center_dist / max(1.0, diag_last, diag_det)
    max_center_norm = 1.1 + (0.4 * float(max(0, frame_gap - 1)))
    center_score = max(0.0, 1.0 - (center_dist_norm / max(1e-6, max_center_norm)))

    size_score = max(0.0, 1.0 - min(6.0, area_ratio) / 6.0)
    depth_score = 0.0

    depth_a = last.get("depth_3d") if isinstance(last.get("depth_3d"), dict) else None
    depth_b = det.get("depth_3d") if isinstance(det.get("depth_3d"), dict) else None
    if depth_a is not None and depth_b is not None:
        center_a = np.array(depth_a.get("center", [0.0, 0.0, 0.0]), dtype=float)
        center_b = np.array(depth_b.get("center", [0.0, 0.0, 0.0]), dtype=float)
        ext_a = np.array(depth_a.get("extents", [0.2, 0.2, 0.2]), dtype=float)
        ext_b = np.array(depth_b.get("extents", [0.2, 0.2, 0.2]), dtype=float)

        # Use horizontal size as primary scale reference (vertical can be noisy).
        size_ref = max(
            float(max(ext_a[0], ext_a[2])),
            float(max(ext_b[0], ext_b[2])),
            0.25,
        )
        dist_3d = float(np.linalg.norm(center_a - center_b))
        depth_gap = abs(float(center_a[2] - center_b[2]))
        max_dist = max(1.1, (2.1 * size_ref) + (0.35 * float(max(0, frame_gap - 1))))
        if dist_3d > max_dist or depth_gap > max(0.9, 1.5 * size_ref):
            return -1.0
        depth_score = max(0.0, 1.0 - (dist_3d / max_dist))
    elif center_dist_norm > max_center_norm and iou < 0.05:
        return -1.0

    score = (0.45 * iou) + (0.25 * center_score) + (0.2 * depth_score) + (0.1 * size_score)
    if _detections_match(last, det):
        score = max(score, 0.33)
    score -= min(0.2, 0.06 * float(max(0, frame_gap - 1)))
    return float(max(0.0, min(1.0, score)))


def _track_label_detections(label_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build temporal tracks for detections of a single label."""
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for det in label_dets:
        by_frame[int(det.get("frame_idx", 0))].append(det)

    tracks: List[Dict[str, Any]] = []
    for frame_idx in sorted(by_frame.keys()):
        frame_dets = _suppress_frame_duplicates(by_frame[frame_idx])
        used_track_ids: set[int] = set()
        for det in sorted(frame_dets, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            best_track_idx = -1
            best_score = _TRACK_MIN_ASSOC_SCORE
            for track_idx, track in enumerate(tracks):
                if track_idx in used_track_ids:
                    continue
                score = _track_association_score(track, det)
                if score > best_score:
                    best_score = score
                    best_track_idx = track_idx

            if best_track_idx >= 0:
                track = tracks[best_track_idx]
                track["detections"].append(det)
                track["last_det"] = det
                track["last_frame_idx"] = frame_idx
                track["frame_indices"].add(frame_idx)
                used_track_ids.add(best_track_idx)
            else:
                tracks.append(
                    {
                        "detections": [det],
                        "last_det": det,
                        "last_frame_idx": frame_idx,
                        "frame_indices": {frame_idx},
                    }
                )
                used_track_ids.add(len(tracks) - 1)

    return tracks


def _track_prototype(track: Dict[str, Any]) -> Dict[str, Any]:
    dets = track.get("detections", [])
    centers = []
    extents = []
    centroids = []
    areas = []
    for det in dets:
        centroids.append(det.get("centroid_px", [0.0, 0.0]))
        areas.append(_box_area(det["box"]))
        depth = det.get("depth_3d") if isinstance(det.get("depth_3d"), dict) else None
        if depth is not None:
            centers.append(depth.get("center", [0.0, 0.0, 0.0]))
            extents.append(depth.get("extents", [0.2, 0.2, 0.2]))

    out: Dict[str, Any] = {
        "frame_indices": set(track.get("frame_indices", set())),
        "centroid_mean": [0.0, 0.0],
        "area_median": 0.0,
    }
    if centroids:
        c = np.array(centroids, dtype=float)
        out["centroid_mean"] = [float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))]
    if areas:
        out["area_median"] = float(np.median(np.array(areas, dtype=float)))
    if centers and extents:
        out["depth_center"] = np.median(np.array(centers, dtype=float), axis=0)
        out["depth_extents"] = np.median(np.array(extents, dtype=float), axis=0)
    return out


def _tracks_mergeable(track_a: Dict[str, Any], track_b: Dict[str, Any]) -> bool:
    proto_a = _track_prototype(track_a)
    proto_b = _track_prototype(track_b)
    frames_a = proto_a.get("frame_indices", set())
    frames_b = proto_b.get("frame_indices", set())
    overlap = set(frames_a).intersection(set(frames_b))
    if overlap:
        # If both tracks are present in same frame(s), treat as distinct objects.
        return False

    depth_center_a = proto_a.get("depth_center")
    depth_center_b = proto_b.get("depth_center")
    depth_ext_a = proto_a.get("depth_extents")
    depth_ext_b = proto_b.get("depth_extents")
    if depth_center_a is not None and depth_center_b is not None:
        center_a = np.array(depth_center_a, dtype=float)
        center_b = np.array(depth_center_b, dtype=float)
        ext_a = np.array(depth_ext_a, dtype=float) if depth_ext_a is not None else np.array([0.2, 0.2, 0.2])
        ext_b = np.array(depth_ext_b, dtype=float) if depth_ext_b is not None else np.array([0.2, 0.2, 0.2])
        size_ref = max(float(max(ext_a[0], ext_a[2])), float(max(ext_b[0], ext_b[2])), 0.3)
        dist_3d = float(np.linalg.norm(center_a - center_b))
        depth_gap = abs(float(center_a[2] - center_b[2]))
        ext_ratio = max(float(np.max(ext_a)), float(np.max(ext_b))) / max(0.05, min(float(np.max(ext_a)), float(np.max(ext_b))))
        return (
            dist_3d <= max(1.2, 1.8 * size_ref)
            and depth_gap <= max(1.0, 1.4 * size_ref)
            and ext_ratio <= 3.5
        )

    c_a = np.array(proto_a.get("centroid_mean", [0.0, 0.0]), dtype=float)
    c_b = np.array(proto_b.get("centroid_mean", [0.0, 0.0]), dtype=float)
    center_dist = float(np.linalg.norm(c_a - c_b))
    area_a = float(proto_a.get("area_median", 0.0))
    area_b = float(proto_b.get("area_median", 0.0))
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    return center_dist <= 80.0 and area_ratio <= 3.0


def _merge_tracklets(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge non-overlapping short track fragments likely from same object."""
    merged = [dict(track) for track in tracks]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            if changed:
                break
            for j in range(i + 1, len(merged)):
                if not _tracks_mergeable(merged[i], merged[j]):
                    continue
                merged[i]["detections"].extend(merged[j]["detections"])
                merged[i]["detections"].sort(key=lambda det: int(det.get("frame_idx", 0)))
                merged[i]["frame_indices"] = set(int(det.get("frame_idx", 0)) for det in merged[i]["detections"])
                merged[i]["last_det"] = merged[i]["detections"][-1]
                merged[i]["last_frame_idx"] = int(merged[i]["last_det"].get("frame_idx", 0))
                del merged[j]
                changed = True
                break
    return merged


def _reference_quality(det: Dict[str, Any]) -> float:
    score = float(det.get("score", 0.0))
    area = max(1.0, _box_area(det["box"]))
    image_size = det.get("image_size", [1, 1])
    image_area = max(1.0, float(image_size[0]) * float(image_size[1]))
    coverage = min(1.0, area / image_area)

    cx, cy = det.get("centroid_px", [0.0, 0.0])
    center_x = float(image_size[0]) * 0.5
    center_y = float(image_size[1]) * 0.5
    dist = float(np.hypot(float(cx) - center_x, float(cy) - center_y))
    center_norm = dist / max(1.0, float(np.hypot(center_x, center_y)))
    center_score = max(0.0, 1.0 - center_norm)
    return (0.65 * score) + (0.25 * np.sqrt(coverage)) + (0.10 * center_score)


def _select_reference_crops(cluster: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    crop_dets = [det for det in cluster if isinstance(det.get("crop_path"), str) and str(det.get("crop_path")).strip()]
    if not crop_dets:
        return None, []

    ranked = sorted(crop_dets, key=_reference_quality, reverse=True)
    seen: set[str] = set()
    ordered: List[str] = []
    for det in ranked:
        crop = str(det.get("crop_path")).strip()
        if crop and crop not in seen:
            seen.add(crop)
            ordered.append(crop)
        if len(ordered) >= _MAX_REFERENCE_CROPS:
            break
    if not ordered:
        return None, []
    return ordered[0], ordered


def _cluster_to_object(
    *,
    label: str,
    cluster_idx: int,
    cluster: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scores = [d["score"] for d in cluster]
    boxes = [d["box"] for d in cluster]
    centroids = [d["centroid_px"] for d in cluster]
    n_frames = len(set(int(d.get("frame_idx", 0)) for d in cluster))
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))

    mean_box = [float(np.mean([b[i] for b in boxes])) for i in range(4)]
    mean_centroid = [float(np.mean([c[i] for c in centroids])) for i in range(2)]
    img_w, img_h = cluster[0]["image_size"]

    depth_3d_list = [d["depth_3d"] for d in cluster if "depth_3d" in d]
    has_depth = len(depth_3d_list) > 0
    if has_depth:
        centers = np.array([d["center"] for d in depth_3d_list])
        extents_arr = np.array([d["extents"] for d in depth_3d_list])
        cx_3d = float(np.median(centers[:, 0]))
        cy_3d = float(np.median(centers[:, 1]))
        cz_3d = float(np.median(centers[:, 2]))
        width_m = float(np.median(extents_arr[:, 0]))
        height_m = float(np.median(extents_arr[:, 1]))
        depth_m = float(np.median(extents_arr[:, 2]))
        refinement_source = "da3_metric_depth"
    else:
        box_w = mean_box[2] - mean_box[0]
        box_h = mean_box[3] - mean_box[1]
        scene_depth_est = 3.0
        scale = scene_depth_est / max(img_w, img_h)
        width_m = box_w * scale
        height_m = box_h * scale
        depth_m = min(width_m, height_m) * 0.6
        cx_3d = (mean_centroid[0] / img_w - 0.5) * scene_depth_est
        cy_3d = (0.5 - mean_centroid[1] / img_h) * scene_depth_est
        cz_3d = scene_depth_est * 0.5
        refinement_source = "heuristic_2d"

    best_crop, all_crops = _select_reference_crops(cluster)
    frame_indices = sorted(set(int(det.get("frame_idx", 0)) for det in cluster))
    frame_paths = [str(det.get("frame_path")) for det in sorted(cluster, key=lambda det: int(det.get("frame_idx", 0)))]
    unique_frame_paths = list(dict.fromkeys(frame_paths))

    obj_entry = {
        "id": f"{label}_{cluster_idx + 1}",
        "label": label,
        "confidence": round(max_score, 3),
        "mean_confidence": round(mean_score, 3),
        "n_frame_detections": n_frames,
        "n_total_detections": len(cluster),
        "frame_indices": frame_indices,
        "frame_paths": unique_frame_paths,
        "boundingBox": {
            "center": [round(cx_3d, 4), round(cy_3d, 4), round(cz_3d, 4)],
            "extents": [
                round(max(0.02, width_m), 4),
                round(max(0.02, height_m), 4),
                round(max(0.02, depth_m), 4),
            ],
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "orientationQuaternion": [1, 0, 0, 0],
        },
        "mean_box_px": [round(v, 1) for v in mean_box],
        "mean_centroid_px": [round(v, 1) for v in mean_centroid],
        "detection_source": "sam3",
        "refinement": refinement_source,
    }
    if best_crop is not None:
        obj_entry["reference_crop"] = best_crop
    if all_crops:
        obj_entry["all_crops"] = all_crops
    return obj_entry


def _merge_detections(
    all_detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Temporal association over the sampled sequence into scene-level objects."""
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in all_detections:
        by_label[det["label"]].append(det)

    merged_objects: List[Dict[str, Any]] = []

    for label, dets in by_label.items():
        if label.lower() in _STRUCTURAL_LABELS:
            continue

        tracks = _track_label_detections(dets)
        tracks = _merge_tracklets(tracks)
        tracks.sort(
            key=lambda track: (
                max(float(det.get("score", 0.0)) for det in track["detections"]),
                len(track["frame_indices"]),
            ),
            reverse=True,
        )
        for cluster_idx, track in enumerate(tracks):
            merged_objects.append(
                _cluster_to_object(
                    label=label,
                    cluster_idx=cluster_idx,
                    cluster=track["detections"],
                )
            )

    # Sort by confidence descending
    merged_objects.sort(key=lambda x: x["confidence"], reverse=True)
    return merged_objects


def _read_colmap_cameras(cameras_bin: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP cameras.bin file."""
    import struct as st

    cameras = {}
    with open(cameras_bin, "rb") as f:
        n_cameras = st.unpack("<Q", f.read(8))[0]
        for _ in range(n_cameras):
            cam_id = st.unpack("<i", f.read(4))[0]
            model_id = st.unpack("<i", f.read(4))[0]
            width = st.unpack("<Q", f.read(8))[0]
            height = st.unpack("<Q", f.read(8))[0]

            # Number of params per model: SIMPLE_PINHOLE=3, PINHOLE=4
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
            params = st.unpack(f"<{n_params}d", f.read(8 * n_params))

            cam = {"id": cam_id, "model_id": model_id, "width": width,
                   "height": height, "params": params}
            if model_id == 1:  # PINHOLE: fx, fy, cx, cy
                cam["fx"], cam["fy"], cam["cx"], cam["cy"] = params
            elif model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
                cam["fx"] = cam["fy"] = params[0]
                cam["cx"], cam["cy"] = params[1], params[2]
            cameras[cam_id] = cam
    return cameras


def _read_colmap_images(images_bin: Path) -> List[Dict[str, Any]]:
    """Read COLMAP images.bin file (camera poses per image)."""
    import struct as st

    images = []
    with open(images_bin, "rb") as f:
        n_images = st.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            image_id = st.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = st.unpack("<4d", f.read(32))
            tx, ty, tz = st.unpack("<3d", f.read(24))
            camera_id = st.unpack("<i", f.read(4))[0]

            # Read image name (null-terminated)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)

            # Skip 2D points
            n_points = st.unpack("<Q", f.read(8))[0]
            f.read(n_points * 24)  # each: x(8) + y(8) + point3D_id(8)

            # Convert quaternion to rotation matrix
            r = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qz**2)],
            ])
            t = np.array([tx, ty, tz])
            # Camera center in world coordinates: C = -R^T * t
            center = -r.T @ t

            images.append({
                "id": image_id, "name": name, "camera_id": camera_id,
                "R": r, "t": t, "center": center,
            })
    return images


def _load_gaussian_ply(ply_path: Path) -> np.ndarray:
    """Load XYZ coordinates from a Gaussian splat PLY file."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(ply_path))
        v = ply["vertex"]
        return np.column_stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])])
    except ImportError:
        _log("plyfile not available, trying numpy-based PLY reader")
        # Simple ASCII/binary PLY reader fallback
        import struct as st
        xyz = []
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            header_str = header.decode("ascii", errors="ignore")
            n_vertices = 0
            for line in header_str.split("\n"):
                if line.startswith("element vertex"):
                    n_vertices = int(line.split()[-1])
            # Read binary little-endian floats (assuming x,y,z are first 3 floats)
            for _ in range(min(n_vertices, 500000)):
                data = f.read(4 * 3)
                if len(data) < 12:
                    break
                x, y, z = st.unpack("<3f", data)
                xyz.append([x, y, z])
                # Skip remaining properties per vertex
                remaining = f.read(max(0, 62 * 4 - 12))  # approximate
        return np.array(xyz) if xyz else np.zeros((0, 3))


def _project_points_to_image(
    points_3d: np.ndarray, R: np.ndarray, t: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates. Returns (uv, mask)."""
    # Transform to camera coordinates: p_cam = R * p_world + t
    p_cam = (R @ points_3d.T).T + t
    z = p_cam[:, 2]

    # Only keep points in front of camera
    valid = z > 0.1
    u = np.full(len(points_3d), -1.0)
    v = np.full(len(points_3d), -1.0)
    u[valid] = fx * p_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * p_cam[valid, 1] / z[valid] + cy

    # Check image bounds
    in_bounds = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.column_stack([u, v]), in_bounds


def _refine_with_colmap(
    objects: List[Dict[str, Any]],
    colmap_sparse_dir: Optional[Path],
    gaussian_ply_path: Optional[Path] = None,
    all_detections: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Refine 3D bounding boxes using COLMAP cameras + Gaussian PLY.

    Back-projects SAM3 2D masks through COLMAP camera poses into the
    3D Gaussian point cloud to extract accurate oriented bounding boxes.
    """
    if colmap_sparse_dir is None or not colmap_sparse_dir.exists():
        _log("No COLMAP sparse dir provided, using heuristic 3D estimates")
        return objects

    cameras_bin = colmap_sparse_dir / "cameras.bin"
    images_bin = colmap_sparse_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        _log("COLMAP cameras.bin or images.bin not found, using heuristic estimates")
        return objects

    try:
        cameras = _read_colmap_cameras(cameras_bin)
        images = _read_colmap_images(images_bin)
        _log(f"Loaded {len(cameras)} cameras, {len(images)} images from COLMAP")

        if not cameras or not images:
            return objects

        # Get the first camera's intrinsics (single camera assumption)
        cam = list(cameras.values())[0]
        fx = cam.get("fx", 1000)
        fy = cam.get("fy", 1000)
        cx = cam.get("cx", cam["width"] / 2)
        cy = cam.get("cy", cam["height"] / 2)
        img_w, img_h = cam["width"], cam["height"]
        _log(f"Camera: {img_w}x{img_h}, fx={fx:.1f}, fy={fy:.1f}")

        # Build image name → pose lookup
        name_to_pose = {img["name"]: img for img in images}

        # Load Gaussian point cloud if available
        points_3d = None
        if gaussian_ply_path and gaussian_ply_path.exists():
            _log(f"Loading Gaussian PLY for back-projection: {gaussian_ply_path}")
            points_3d = _load_gaussian_ply(gaussian_ply_path)
            _log(f"  Loaded {len(points_3d)} Gaussians")

            # Subsample for performance if very large
            if len(points_3d) > 200000:
                indices = np.random.choice(len(points_3d), 200000, replace=False)
                points_3d = points_3d[indices]
                _log(f"  Subsampled to {len(points_3d)} points")

        # For each object, find its 3D extent by back-projecting its 2D
        # bounding box through all cameras that see it, and intersecting
        # with the Gaussian point cloud
        for obj in objects:
            box_px = obj.get("mean_box_px", [0, 0, 100, 100])
            det_img_w, det_img_h = 1, 1
            # Get detection image size for coordinate scaling
            if obj.get("n_total_detections", 0) > 0:
                # mean_box_px is in detection image coordinates (original frame)
                # Need to scale to COLMAP image coordinates
                pass

            # Scale 2D box from detection resolution to COLMAP resolution
            # Detection was done on original frames, COLMAP uses undistorted frames
            # which may have different resolution
            orig_w = obj.get("_det_img_w", img_w)
            orig_h = obj.get("_det_img_h", img_h)
            scale_x = img_w / max(1, orig_w) if orig_w != img_w else 1.0
            scale_y = img_h / max(1, orig_h) if orig_h != img_h else 1.0

            x1 = box_px[0] * scale_x
            y1 = box_px[1] * scale_y
            x2 = box_px[2] * scale_x
            y2 = box_px[3] * scale_y

            # Skip Gaussian back-projection if DA3 already gave us metric 3D
            if obj.get("refinement") == "da3_metric_depth":
                _log(f"  {obj['id']}: using DA3 metric depth (skipping Gaussian backprojection)")
                continue

            if points_3d is not None and len(points_3d) > 0:
                # For each COLMAP image, project all 3D points and check
                # which fall inside the 2D bounding box. Take the union.
                selected_mask = np.zeros(len(points_3d), dtype=bool)
                n_views = 0

                # Use a tighter box (shrink by 10%) to avoid selecting background
                pad_x = (x2 - x1) * 0.1
                pad_y = (y2 - y1) * 0.1
                x1_tight = x1 + pad_x
                y1_tight = y1 + pad_y
                x2_tight = x2 - pad_x
                y2_tight = y2 - pad_y

                for img_pose in images[:20]:  # Limit to 20 views for speed
                    uv, in_bounds = _project_points_to_image(
                        points_3d, img_pose["R"], img_pose["t"],
                        fx, fy, cx, cy, img_w, img_h,
                    )
                    # Points that project inside the tightened 2D bounding box
                    in_box = (
                        in_bounds &
                        (uv[:, 0] >= x1_tight) & (uv[:, 0] <= x2_tight) &
                        (uv[:, 1] >= y1_tight) & (uv[:, 1] <= y2_tight)
                    )
                    if in_box.any():
                        selected_mask |= in_box
                        n_views += 1

                n_selected = int(selected_mask.sum())
                if n_selected >= 10:
                    obj_points = points_3d[selected_mask]

                    # Compute OBB from selected points
                    center = obj_points.mean(axis=0)
                    # Use PCA for oriented bounding box
                    cov = np.cov(obj_points.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Sort by eigenvalue descending
                    order = eigenvalues.argsort()[::-1]
                    axes = eigenvectors[:, order].T

                    # Project onto principal axes for extents
                    local = (obj_points - center) @ axes.T
                    extents = local.max(axis=0) - local.min(axis=0)

                    obj["boundingBox"] = {
                        "center": [round(float(c), 4) for c in center],
                        "extents": [round(max(0.02, float(e)), 4) for e in extents],
                        "axes": [[round(float(v), 6) for v in ax] for ax in axes],
                        "orientationQuaternion": [1, 0, 0, 0],  # TODO: compute from axes
                    }
                    obj["n_gaussian_points"] = n_selected
                    obj["n_views_matched"] = n_views
                    obj["refinement"] = "gaussian_backprojection"

                    bb = obj["boundingBox"]
                    _log(f"  {obj['id']}: {n_selected} Gaussians from {n_views} views → "
                         f"{bb['extents'][0]:.3f}x{bb['extents'][1]:.3f}x{bb['extents'][2]:.3f}m")
                    continue

            # Fallback: use focal length for depth estimation
            box_w_px = x2 - x1
            box_h_px = y2 - y1
            if box_w_px > 10:
                # Use median camera distance as reference
                cam_centers = np.array([img["center"] for img in images])
                scene_center = cam_centers.mean(axis=0)
                median_dist = float(np.median(np.linalg.norm(cam_centers - scene_center, axis=1)))

                # Estimate real-world size from pixel size + focal length + estimated depth
                est_depth = median_dist * 0.8
                width_m = box_w_px * est_depth / fx
                height_m = box_h_px * est_depth / fy
                depth_m = min(width_m, height_m) * 0.6

                cx_3d = (((x1 + x2) / 2) - cx) * est_depth / fx
                cy_3d = (cy - ((y1 + y2) / 2)) * est_depth / fy

                obj["boundingBox"]["center"] = [round(cx_3d, 4), round(cy_3d, 4), round(est_depth, 4)]
                obj["boundingBox"]["extents"] = [
                    round(max(0.05, width_m), 4),
                    round(max(0.05, height_m), 4),
                    round(max(0.05, depth_m), 4),
                ]
                obj["refinement"] = "focal_length_estimate"

        _log(f"Refined {len(objects)} objects with COLMAP + Gaussian data")
    except Exception as e:
        _log(f"COLMAP refinement failed: {e}, using heuristic estimates")
        import traceback
        traceback.print_exc()

    return objects


def run_sam3_detection(
    *,
    frames_dir: Path,
    output_path: Path,
    environment: str = "default",
    colmap_sparse_dir: Optional[Path] = None,
    gaussian_ply_path: Optional[Path] = None,
    n_sample_frames: int = _DEFAULT_SAMPLE_FRAMES,
    min_frame_detections: int = 2,
    save_crops: bool = True,
) -> Dict[str, Any]:
    """Run full SAM3 detection pipeline and write object index.

    When ``gaussian_ply_path`` is provided (the 3DGRUT Gaussian splat PLY),
    SAM3 2D masks are back-projected through COLMAP cameras into the 3D
    point cloud to produce accurate real-world bounding boxes (position +
    width/height/depth in meters).
    """

    _log(f"Environment: {environment}")
    _log(f"Frames dir: {frames_dir}")
    _log(f"Output: {output_path}")
    if gaussian_ply_path:
        _log(f"Gaussian PLY: {gaussian_ply_path}")

    # Select prompts for this environment
    prompts = _DETECTION_PROMPTS.get(environment, _DETECTION_PROMPTS["default"])
    _log(f"Detection prompts ({len(prompts)}): {', '.join(prompts)}")

    all_frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    n_sample_frames, min_frame_detections = _resolve_sampling_settings(
        environment=environment,
        total_frames=len(all_frames),
        requested_n_frames=n_sample_frames,
        requested_min_frame_detections=min_frame_detections,
    )
    _log(
        f"Sampling settings: total_frames={len(all_frames)} "
        f"n_frames={n_sample_frames} min_frame_detections={min_frame_detections}"
    )

    tracking_mode = (os.getenv("SAM3_TRACKING_MODE", _TRACKING_MODE_DEFAULT) or _TRACKING_MODE_DEFAULT).strip().lower()
    if tracking_mode not in {"full_video", "sampled"}:
        tracking_mode = _TRACKING_MODE_DEFAULT if _TRACKING_MODE_DEFAULT in {"full_video", "sampled"} else "full_video"

    if tracking_mode == "full_video":
        frame_paths = all_frames
        _log(f"Tracking mode: full_video (using all {len(frame_paths)} frames)")
    else:
        frame_paths = _sample_frame_paths(frames_dir, n_sample_frames)
        _log(f"Tracking mode: sampled (using {len(frame_paths)} frames)")

    # Load SAM3
    processor = _load_sam3()

    # Load DA3 for metric depth (optional but recommended)
    da3_model = None
    focal_px = 1000.0  # default, overridden by COLMAP if available
    if colmap_sparse_dir and (colmap_sparse_dir / "cameras.bin").exists():
        try:
            cams = _read_colmap_cameras(colmap_sparse_dir / "cameras.bin")
            if cams:
                cam = list(cams.values())[0]
                focal_px = cam.get("fx", 1000.0)
                _log(f"COLMAP focal length: {focal_px:.1f}px")
        except Exception as e:
            _log(f"Could not read COLMAP cameras: {e}")

    try:
        da3_model = _load_da3()
        _log("DA3 metric depth enabled - will compute accurate 3D bounding boxes")
    except Exception as e:
        _log(f"DA3 not available ({e}), using heuristic 3D estimates")

    # Set up crops directory for reference image extraction
    crops_dir = None
    if save_crops:
        crops_dir = output_path.parent / "object_crops"
        _log(f"Object crops will be saved to: {crops_dir}")

    # Run detection on each frame
    all_detections: List[Dict[str, Any]] = []
    for i, frame_path in enumerate(frame_paths):
        _log(f"  Frame {i+1}/{len(frame_paths)}: {frame_path.name}")

        # Get depth map for this frame
        depth_map = None
        if da3_model is not None:
            try:
                from PIL import Image
                img_for_depth = Image.open(frame_path).convert("RGB")
                depth_map = _get_metric_depth(da3_model, img_for_depth, focal_px)
            except Exception as e:
                _log(f"    DA3 depth failed: {e}")

        dets = _detect_objects_in_frame(
            processor, frame_path, prompts,
            depth_map=depth_map, focal_px=focal_px,
            crops_dir=crops_dir, frame_idx=i,
        )
        n_with_depth = sum(1 for d in dets if "depth_3d" in d)
        _log(f"    {len(dets)} detections ({n_with_depth} with metric depth)")
        all_detections.extend(dets)

    _log(f"Total raw detections: {len(all_detections)}")

    # Free DA3 memory before merge step
    if da3_model is not None:
        del da3_model
        torch.cuda.empty_cache()

    # Merge across frames
    objects = _merge_detections(all_detections)
    _log(f"Merged into {len(objects)} unique objects")

    # Filter: require detection in multiple frames for robustness
    if min_frame_detections > 1:
        before = len(objects)
        objects = [
            obj for obj in objects
            if obj["n_frame_detections"] >= min_frame_detections
        ]
        _log(f"After multi-frame filter (>={min_frame_detections}): {len(objects)} objects (removed {before - len(objects)})")

    # Refine with COLMAP + Gaussian PLY for accurate 3D bounding boxes
    objects = _refine_with_colmap(objects, colmap_sparse_dir, gaussian_ply_path, all_detections)

    # Report
    n_with_crops = sum(1 for obj in objects if "reference_crop" in obj)
    _log(f"\nDetected objects ({n_with_crops}/{len(objects)} with reference crops):")
    for obj in objects:
        bb = obj["boundingBox"]
        crop_tag = " [crop]" if "reference_crop" in obj else ""
        _log(f"  {obj['id']:20s}  conf={obj['confidence']:.2f}  "
             f"frames={obj['n_frame_detections']}  "
             f"size={bb['extents'][0]:.2f}x{bb['extents'][1]:.2f}x{bb['extents'][2]:.2f}m"
             f"{crop_tag}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_payload = {
        "schema_version": "v1",
        "detection_source": "sam3",
        "environment": environment,
        "tracking_mode": tracking_mode,
        "track_max_frame_gap": _TRACK_MAX_FRAME_GAP,
        "track_min_assoc_score": _TRACK_MIN_ASSOC_SCORE,
        "n_frames_sampled": len(frame_paths),
        "n_raw_detections": len(all_detections),
        "prompts_used": prompts,
        "objects": objects,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2)

    _log(f"\nWrote {len(objects)} objects to {output_path}")

    # Free GPU memory
    del processor
    torch.cuda.empty_cache()

    return index_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAM3 object detection for swap pipeline"
    )
    parser.add_argument("--frames-dir", required=True,
                        help="Directory with extracted video frames")
    parser.add_argument("--output", required=True,
                        help="Output path for object_point_cloud_index.json")
    parser.add_argument("--environment", default="default",
                        choices=list(_DETECTION_PROMPTS.keys()),
                        help="Environment type for prompt selection")
    parser.add_argument("--colmap-sparse", default=None,
                        help="Path to COLMAP sparse/0/ for 3D refinement")
    parser.add_argument("--gaussian-ply", default=None,
                        help="Path to 3DGRUT export_last.ply for accurate 3D back-projection")
    parser.add_argument("--n-frames", type=int, default=0,
                        help="Number of frames to sample (0=auto)")
    parser.add_argument("--min-frame-detections", type=int, default=0,
                        help="Minimum frames an object must appear in (0=auto)")
    parser.add_argument("--no-crops", action="store_true",
                        help="Disable saving per-object reference crops")
    args = parser.parse_args()

    result = run_sam3_detection(
        frames_dir=Path(args.frames_dir),
        output_path=Path(args.output),
        environment=args.environment,
        colmap_sparse_dir=Path(args.colmap_sparse) if args.colmap_sparse else None,
        gaussian_ply_path=Path(args.gaussian_ply) if args.gaussian_ply else None,
        n_sample_frames=args.n_frames,
        min_frame_detections=args.min_frame_detections,
        save_crops=not args.no_crops,
    )

    n_objects = len(result.get("objects", []))
    return 0 if n_objects > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
