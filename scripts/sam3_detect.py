#!/usr/bin/env python3
"""Optional SAM3 detection entrypoint for object-index builds.

This narrowed helper is intentionally conservative:

- it validates local video input
- it reports explicit runtime blockers when SAM3 is unavailable
- it only attempts detection when both the `sam3` package and weights are present

SAM3 is optional for the supported single-VM path. YOLO-World remains the
required supported backend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse


_PROMPT_BANKS: Dict[str, List[str]] = {
    "default": ["door", "cabinet", "drawer", "shelf", "table", "chair", "desk", "box", "container"],
    "warehouse": ["rack", "shelf", "pallet", "tote", "bin", "box", "package", "cart", "door", "container"],
    "kitchen": ["cabinet", "drawer", "refrigerator", "fridge", "oven", "dishwasher", "sink", "mug", "bottle"],
    "bedroom": ["bed", "nightstand", "dresser", "wardrobe", "closet", "door", "desk", "chair", "lamp", "box"],
    "office": ["desk", "chair", "monitor", "keyboard", "mouse", "laptop", "printer", "cabinet", "shelf"],
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _validate_local_video_path(raw_video_path: str) -> Path:
    parsed = urlparse((raw_video_path or "").strip())
    if parsed.scheme:
        raise ValueError("--video must be a local filesystem path, not a URI")
    video_path = Path(raw_video_path).expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"--video does not exist or is not a file: {video_path}")
    if video_path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("--video must point to a .mp4 or .mov file")
    return video_path


def _resolve_prompts(environment: str) -> List[str]:
    raw = str(os.getenv("SAM3_DETECTION_PROMPTS") or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = [item.strip() for item in raw.split(",")]
        if isinstance(payload, list):
            prompts = [str(item).strip() for item in payload if str(item).strip()]
            if prompts:
                return prompts
    return list(_PROMPT_BANKS.get(environment, _PROMPT_BANKS["default"]))


def _extract_frames(video_path: Path, *, n_frames: int) -> List[Path]:
    n_frames = max(1, n_frames)
    with tempfile.TemporaryDirectory(prefix="sam3_detect_frames_") as tmp_dir:
        temp_root = Path(tmp_dir)
        frame_paths: List[Path] = []
        for index in range(n_frames):
            frame_path = temp_root / f"frame_{index:06d}.png"
            timestamp = f"{float(index):.3f}"
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    timestamp,
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0 and frame_path.is_file():
                persisted = video_path.parent / "object_index_artifacts" / "sam3_frames" / frame_path.name
                persisted.parent.mkdir(parents=True, exist_ok=True)
                persisted.write_bytes(frame_path.read_bytes())
                frame_paths.append(persisted)
        return frame_paths


def _sam3_ready_reason() -> str | None:
    if not _module_available("torch"):
        return "torch_not_installed"
    if not _module_available("sam3"):
        return "sam3_not_installed"
    weights_path = Path(os.getenv("SAM3_WEIGHTS_PATH", "/opt/sam3_weights/sam3.pt"))
    if not weights_path.is_file():
        return f"sam3_weights_missing:{weights_path}"
    return None


def _detection_payload(frame_paths: Iterable[Path], prompts: List[str]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for frame_index, frame_path in enumerate(frame_paths):
        for prompt in prompts[:1]:
            detections.append(
                {
                    "frame_index": frame_index,
                    "label": prompt,
                    "score": 0.0,
                    "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                    "source_prompt": prompt,
                    "frame_path": str(frame_path),
                }
            )
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional SAM3 object detection helper")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--tracking-backend", default="image_model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--environment", default="auto")
    parser.add_argument("--colmap-sparse", default=None)
    parser.add_argument("--gaussian-ply", default=None)
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--min-frame-detections", type=int, default=1)
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--save-instance-masks", action="store_true")
    parser.add_argument("--instance-masks-dir", default=None)
    parser.add_argument("--force-full-video-masks", action="store_true")
    parser.add_argument("--scene-semantics", action="store_true")
    parser.add_argument("--dimension-completion-mode", default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.video:
        payload = {
            "backend_status": "skipped",
            "reason": "video_required",
            "objects": [],
            "detections": [],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    try:
        video_path = _validate_local_video_path(args.video)
    except ValueError as exc:
        payload = {
            "backend_status": "failed",
            "reason": str(exc),
            "objects": [],
            "detections": [],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    runtime_blocker = _sam3_ready_reason()
    prompts = _resolve_prompts(str(args.environment or "default").strip().lower() or "default")
    frame_paths = _extract_frames(video_path, n_frames=max(1, int(args.n_frames or 8)))
    if runtime_blocker:
        payload = {
            "backend_status": "skipped",
            "reason": runtime_blocker,
            "objects": [],
            "detections": [],
            "metadata": {
                "video_path": str(video_path),
                "tracking_backend": str(args.tracking_backend),
                "sampled_frame_count": len(frame_paths),
                "prompts": prompts,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1

    payload = {
        "backend_status": "skipped",
        "reason": "sam3_runtime_available_but_detection_not_enabled_without_project-specific tracker",
        "objects": [],
        "detections": [],
        "metadata": {
            "video_path": str(video_path),
            "tracking_backend": str(args.tracking_backend),
            "sampled_frame_count": len(frame_paths),
            "prompts": prompts,
            "note": "SAM3 is optional for this narrowed path; YOLO-World remains the required supported backend.",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
