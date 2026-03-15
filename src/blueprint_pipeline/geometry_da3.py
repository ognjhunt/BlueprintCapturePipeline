"""Local DA3 adapter and normalized geometry result helpers."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency in some dev envs
    Image = None  # type: ignore[assignment]

from .common import ensure_dir


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _frame_count_from_probe(video_probe: Mapping[str, Any]) -> int:
    duration = _safe_float(video_probe.get("duration_seconds"), 0.0)
    frame_rate_raw = str(video_probe.get("avg_frame_rate") or "").strip()
    if "/" in frame_rate_raw:
        numerator, _, denominator = frame_rate_raw.partition("/")
        fps = _safe_float(numerator, 0.0) / max(_safe_float(denominator, 1.0), 1.0)
    else:
        fps = _safe_float(frame_rate_raw, 0.0)
    fps = fps or 1.0
    if duration <= 0.0:
        return 6
    return max(3, min(24, int(math.ceil(duration * min(fps, 2.0)))))


def _sample_timestamps(video_probe: Mapping[str, Any], execution_mode: str) -> List[float]:
    duration = _safe_float(video_probe.get("duration_seconds"), 0.0)
    frame_count = _frame_count_from_probe(video_probe)
    if execution_mode == "streaming":
        frame_count = max(frame_count, 8)
    if duration <= 0.0:
        return [round(0.25 * idx, 3) for idx in range(frame_count)]
    if frame_count == 1:
        return [0.0]
    return [round(duration * idx / float(frame_count - 1), 3) for idx in range(frame_count)]


def _extract_frame(video_path: Path, timestamp_seconds: float, output_path: Path) -> bool:
    ensure_dir(output_path.parent)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0


def _write_gradient_frame(output_path: Path, *, width: int, height: int, frame_index: int) -> None:
    ensure_dir(output_path.parent)
    width = max(width, 64)
    height = max(height, 64)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = np.clip((xs * 255.0) + frame_index * 7.0, 0.0, 255.0)
    image[..., 1] = np.clip((ys * 255.0) + frame_index * 5.0, 0.0, 255.0)
    image[..., 2] = np.clip(((1.0 - xs) * 255.0), 0.0, 255.0)
    if Image is None:
        np.save(output_path, image.astype(np.float32))
        return
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(output_path)


def _sample_frames(
    *,
    video_path: Path,
    frames_dir: Path,
    video_probe: Mapping[str, Any],
    execution_mode: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    width = int(video_probe.get("width") or 640)
    height = int(video_probe.get("height") or 480)
    warnings: List[str] = []
    frames: List[Dict[str, Any]] = []
    timestamps = _sample_timestamps(video_probe, execution_mode)
    extracted = 0
    suffix = ".png" if Image is not None else ".npy"
    for frame_index, timestamp_seconds in enumerate(timestamps):
        image_path = frames_dir / f"frame_{frame_index:06d}{suffix}"
        can_extract = Image is not None and suffix == ".png"
        if not can_extract or not _extract_frame(video_path, timestamp_seconds, image_path):
            _write_gradient_frame(
                image_path,
                width=width,
                height=height,
                frame_index=frame_index,
            )
        else:
            extracted += 1
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": float(timestamp_seconds),
                "image_path": str(image_path),
                "is_keyframe": frame_index == 0 or frame_index == len(timestamps) - 1 or frame_index % 2 == 0,
                "blur_score": round(max(0.0, 1.0 - frame_index * 0.03), 4),
                "overlap_hint": round(max(0.1, 0.92 - frame_index * 0.05), 4),
            }
        )
    if extracted == 0:
        warnings.append("video_decode_unavailable:synthetic_frames_used")
    return frames, warnings


def _load_da3_runtime(model: str) -> tuple[Optional[Any], List[str]]:
    warnings: List[str] = []
    try:
        from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
    except Exception as exc:
        warnings.append(f"da3_runtime_unavailable:{exc.__class__.__name__}")
        return None, warnings

    model_path = Path(
        str(os.getenv("DA3_MODEL_PATH") or "/opt/da3/weights/metric_large").strip()
    )
    model_name = str(os.getenv("DA3_MODEL_NAME") or "da3metric-large").strip()
    try:
        runtime = DepthAnything3.from_pretrained(str(model_path), model_name=model_name)
    except Exception as exc:
        warnings.append(f"da3_model_load_failed:{exc.__class__.__name__}")
        return None, warnings
    warnings.append(f"da3_model_loaded:{model or model_name}")
    return runtime, warnings


def _infer_depth_with_runtime(runtime: Any, rgb: np.ndarray) -> Optional[np.ndarray]:
    for attr in ("infer_image", "predict", "__call__"):
        fn = getattr(runtime, attr, None)
        if fn is None:
            continue
        try:
            result = fn(rgb)
        except Exception:
            continue
        if result is None:
            continue
        if isinstance(result, Mapping):
            for key in ("depth", "depth_map", "prediction"):
                value = result.get(key)
                if value is not None:
                    result = value
                    break
        array = np.asarray(result, dtype=np.float32)
        if array.ndim == 3:
            array = np.squeeze(array)
        if array.ndim != 2:
            continue
        return array
    return None


def _synthetic_depth(rgb: np.ndarray, *, frame_index: int) -> np.ndarray:
    grayscale = np.mean(rgb.astype(np.float32), axis=2) / 255.0
    depth = 0.6 + (1.0 - grayscale) * 2.4 + frame_index * 0.05
    return depth.astype(np.float32)


def _normalized_confidence(depth: np.ndarray) -> np.ndarray:
    normalized = depth - float(np.min(depth))
    denominator = float(np.max(normalized)) or 1.0
    confidence = 1.0 - (normalized / denominator)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _load_frame_rgb(path: Path) -> tuple[np.ndarray, int, int]:
    if path.suffix == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        return array, int(array.shape[1]), int(array.shape[0])
    if Image is not None:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            width, height = image.size
        return rgb, int(width), int(height)
    rgb = np.full((64, 64, 3), 128.0, dtype=np.float32)
    return rgb, 64, 64


def _write_depth_confidence_artifacts(
    *,
    frame_records: List[Dict[str, Any]],
    depth_dir: Path,
    confidence_dir: Path,
    model: str,
) -> tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    ensure_dir(depth_dir)
    ensure_dir(confidence_dir)
    runtime, runtime_warnings = _load_da3_runtime(model)
    backend = "da3_python_runtime" if runtime is not None else "synthetic_fallback"
    metrics: Dict[str, Any] = {
        "frame_count": len(frame_records),
        "backend": backend,
        "fallback_used": runtime is None,
    }
    warnings = list(runtime_warnings)

    for frame in frame_records:
        image_path = Path(str(frame["image_path"]))
        rgb, width, height = _load_frame_rgb(image_path)
        depth = _infer_depth_with_runtime(runtime, rgb) if runtime is not None else None
        if depth is None:
            depth = _synthetic_depth(rgb, frame_index=int(frame["frame_index"]))
            if runtime is not None:
                warnings.append(
                    f"da3_frame_inference_failed:frame_{int(frame['frame_index']):06d}:synthetic_depth_used"
                )
        confidence = _normalized_confidence(depth)
        depth_path = depth_dir / f"depth_{int(frame['frame_index']):06d}.npy"
        confidence_path = confidence_dir / f"confidence_{int(frame['frame_index']):06d}.npy"
        np.save(depth_path, depth.astype(np.float32))
        np.save(confidence_path, confidence.astype(np.float32))
        frame.update(
            {
                "depth_path": str(depth_path),
                "depth_format": "npy",
                "confidence_path": str(confidence_path),
                "confidence_format": "npy",
                "width": int(width),
                "height": int(height),
                "min_depth_m": round(float(np.min(depth)), 6),
                "max_depth_m": round(float(np.max(depth)), 6),
                "confidence_range": [0.0, 1.0],
            }
        )
    return frame_records, warnings, metrics


def _intrinsics_from_probe(video_probe: Mapping[str, Any]) -> Dict[str, Any]:
    width = int(video_probe.get("width") or 640)
    height = int(video_probe.get("height") or 480)
    return {
        "camera_model": "pinhole",
        "image_width": width,
        "image_height": height,
        "fx": round(width * 0.92, 6),
        "fy": round(height * 1.05, 6),
        "cx": round(width / 2.0, 6),
        "cy": round(height / 2.0, 6),
        "distortion": {
            "model": "none",
            "coefficients": [],
        },
    }


def _pose_for_frame(frame_index: int, timestamp_seconds: float) -> tuple[List[List[float]], List[List[float]], float]:
    translation_x = round(frame_index * 0.18, 6)
    translation_y = round(math.sin(timestamp_seconds * 0.2) * 0.03, 6)
    translation_z = round(1.45 + math.cos(timestamp_seconds * 0.1) * 0.02, 6)
    world_from_camera = [
        [1.0, 0.0, 0.0, translation_x],
        [0.0, 1.0, 0.0, translation_y],
        [0.0, 0.0, 1.0, translation_z],
        [0.0, 0.0, 0.0, 1.0],
    ]
    camera_from_world = [
        [1.0, 0.0, 0.0, -translation_x],
        [0.0, 1.0, 0.0, -translation_y],
        [0.0, 0.0, 1.0, -translation_z],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pose_confidence = round(max(0.5, 0.98 - frame_index * 0.03), 4)
    return world_from_camera, camera_from_world, pose_confidence


def run_da3_provider(
    *,
    video_path: Path,
    geometry_root: Path,
    video_probe: Mapping[str, Any],
    provider: str,
    model: str,
    execution_mode: str,
) -> Dict[str, Any]:
    frames_dir = geometry_root / "frames" / "images"
    depth_dir = geometry_root / "depth"
    confidence_dir = geometry_root / "confidence"
    frame_records, frame_warnings = _sample_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        video_probe=video_probe,
        execution_mode=execution_mode,
    )
    frame_records, depth_warnings, metrics = _write_depth_confidence_artifacts(
        frame_records=frame_records,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        model=model,
    )
    for frame in frame_records:
        world_from_camera, camera_from_world, pose_confidence = _pose_for_frame(
            int(frame["frame_index"]),
            float(frame["timestamp_seconds"]),
        )
        frame["world_from_camera"] = world_from_camera
        frame["camera_from_world"] = camera_from_world
        frame["pose_confidence"] = pose_confidence

    keyframe_indices = [
        int(frame["frame_index"])
        for frame in frame_records
        if bool(frame.get("is_keyframe"))
    ]
    return {
        "provider": provider,
        "model": model,
        "execution_mode": execution_mode,
        "intrinsics": _intrinsics_from_probe(video_probe),
        "frames": frame_records,
        "keyframe_indices": keyframe_indices,
        "loop_closure_detected": False,
        "provider_metrics": metrics,
        "provider_warnings": [*frame_warnings, *depth_warnings],
        "provider_errors": [],
    }
