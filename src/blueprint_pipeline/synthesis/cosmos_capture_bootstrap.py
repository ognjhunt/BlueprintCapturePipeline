"""Shared bootstrap helpers for Cosmos evaluation/export from staged capture assets."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional

from ..common import ensure_dir, read_json_any, resolve_gs_uri_to_path


def _optional_existing_path(raw_value: Any) -> Optional[Path]:
    text = str(raw_value or "").strip()
    if not text or text.startswith(("gs://", "http://", "https://")):
        return None
    path = Path(text).expanduser().resolve()
    return path if path.exists() else None


def _existing_path_from_candidates(*candidates: Optional[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def _resolved_gs_path(context, uri: str) -> Optional[Path]:
    text = str(uri or "").strip()
    if not text.startswith("gs://"):
        return None
    path = resolve_gs_uri_to_path(text, context.storage_root)
    return path if path.exists() else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _read_pose_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    return [row for row in rows if row.get("T_world_camera") or row.get("transform")]


def _ffprobe_total_frames(video_path: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return None
        raw_value = streams[0].get("nb_frames")
        if raw_value in (None, "N/A", ""):
            return None
        return max(1, int(raw_value))
    except Exception:
        return None


def _extract_frame_ffmpeg(*, video_path: Path, frame_index: int, frame_path: Path) -> bool:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"select=eq(n\\,{max(0, int(frame_index))})",
                "-vframes",
                "1",
                str(frame_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return False
    return frame_path.is_file()


def resolve_video_bootstrap_sources(
    *,
    context,
    conditioning_bundle: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    conditioning = dict(conditioning_bundle or {})
    arkit = dict(conditioning.get("arkit") or {}) if isinstance(conditioning.get("arkit"), Mapping) else {}
    local_paths = (
        dict(conditioning.get("local_paths") or {})
        if isinstance(conditioning.get("local_paths"), Mapping)
        else {}
    )

    raw_video_uri = str(conditioning.get("raw_video_uri") or "").strip()
    poses_uri = str(arkit.get("poses_uri") or conditioning.get("arkit_poses_uri") or "").strip()
    intrinsics_uri = str(arkit.get("intrinsics_uri") or conditioning.get("arkit_intrinsics_uri") or "").strip()

    video_path = _existing_path_from_candidates(
        _optional_existing_path(local_paths.get("raw_video_path")),
        _resolved_gs_path(context, raw_video_uri),
        (context.raw_root / "walkthrough.mov").resolve() if context.raw_root.exists() else None,
        (context.raw_root / "walkthrough.mp4").resolve() if context.raw_root.exists() else None,
    )
    poses_path = _existing_path_from_candidates(
        _optional_existing_path(local_paths.get("arkit_poses_path")),
        _resolved_gs_path(context, poses_uri),
        (context.raw_root / "arkit" / "poses.jsonl").resolve() if context.raw_root.exists() else None,
    )
    intrinsics_path = _existing_path_from_candidates(
        _optional_existing_path(local_paths.get("arkit_intrinsics_path")),
        _resolved_gs_path(context, intrinsics_uri),
        (context.raw_root / "arkit" / "intrinsics.json").resolve() if context.raw_root.exists() else None,
    )

    if video_path is None or poses_path is None or intrinsics_path is None:
        return {}

    if raw_video_uri and poses_uri and intrinsics_uri:
        origin = "conditioning_bundle"
    else:
        origin = "raw_capture_assets"

    return {
        "origin": origin,
        "video_path": str(video_path),
        "poses_path": str(poses_path),
        "intrinsics_path": str(intrinsics_path),
        "source_video_uri": raw_video_uri or str(video_path),
        "poses_uri": poses_uri or str(poses_path),
        "intrinsics_uri": intrinsics_uri or str(intrinsics_path),
    }


def extract_video_bootstrap_records(
    *,
    bootstrap_sources: Mapping[str, Any],
    export_root: Path,
    max_frames: int,
) -> List[Dict[str, Any]]:
    video_path = Path(str(bootstrap_sources.get("video_path") or "")).expanduser().resolve()
    poses_path = Path(str(bootstrap_sources.get("poses_path") or "")).expanduser().resolve()
    intrinsics_path = Path(str(bootstrap_sources.get("intrinsics_path") or "")).expanduser().resolve()
    if not video_path.is_file() or not poses_path.is_file() or not intrinsics_path.is_file():
        return []

    pose_rows = _read_pose_rows(poses_path)
    intrinsics_payload = read_json_any(intrinsics_path)
    intrinsics = dict(intrinsics_payload) if isinstance(intrinsics_payload, Mapping) else {}
    if not pose_rows:
        return []

    cv2 = None
    capture = None
    total_frames = _ffprobe_total_frames(video_path) or len(pose_rows)
    try:
        import cv2 as _cv2  # type: ignore[import]

        capture = _cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            cv2 = _cv2
            total_frames = max(1, int(capture.get(_cv2.CAP_PROP_FRAME_COUNT) or total_frames))
        else:
            capture.release()
            capture = None
    except ImportError:
        capture = None

    output_dir = export_root / "video_bootstrap_frames"
    ensure_dir(output_dir)

    target_count = min(max_frames, len(pose_rows))
    if target_count < 2:
        if capture is not None:
            capture.release()
        return []

    pose_indices = [
        round(index * (len(pose_rows) - 1) / float(max(1, target_count - 1)))
        for index in range(target_count)
    ]
    records: List[Dict[str, Any]] = []
    for export_index, pose_index in enumerate(sorted(dict.fromkeys(pose_indices))):
        pose_row = pose_rows[pose_index]
        frame_index = round(pose_index * (total_frames - 1) / float(max(1, len(pose_rows) - 1)))
        frame_path = output_dir / f"frame_{export_index:04d}.jpg"
        if capture is not None and cv2 is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            if not cv2.imwrite(str(frame_path), frame):
                continue
        elif not _extract_frame_ffmpeg(video_path=video_path, frame_index=int(frame_index), frame_path=frame_path):
            continue
        records.append(
            {
                "frame_id": f"video_bootstrap_{export_index:04d}",
                "frame_index": int(frame_index),
                "frame_uri": str(frame_path.resolve()),
                "embedding_uri": None,
                "included_in_index": True,
                "t_capture_sec": pose_row.get("t_capture_sec", pose_row.get("t_device_sec", float(pose_index))),
                "T_world_camera": pose_row.get("T_world_camera") or pose_row.get("transform"),
                "intrinsics": intrinsics,
                "anchor_observations": [],
                "retrieval_signals": {
                    "anchor_observation_count": 0,
                    "route_anchor_density": 0.0,
                    "checkpoint_proximity_sec": None,
                    "capture_confidence": 0.65,
                    "geometry_grounding_quality": 0.5,
                },
                "source_mode": "video_bootstrap",
                "bootstrap_origin": str(bootstrap_sources.get("origin") or "unknown"),
                "source_video_uri": str(bootstrap_sources.get("source_video_uri") or str(video_path)),
            }
        )
    if capture is not None:
        capture.release()
    return records
