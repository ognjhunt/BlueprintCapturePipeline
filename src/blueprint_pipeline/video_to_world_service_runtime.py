"""Runtime helpers for the dedicated video_to_world GPU service."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .common import ensure_dir, ensure_local_uri_path, is_gs_uri, parse_gs_uri, resolve_gs_uri_to_path


def _gcs_root() -> Path:
    return Path(str(os.getenv("GCS_ROOT") or "/mnt/gcs").strip() or "/mnt/gcs")


def _string(value: Any) -> str:
    return str(value or "").strip()


def _timeout_seconds() -> int:
    raw = _string(os.getenv("VIDEO_TO_WORLD_COMMAND_TIMEOUT_SECONDS") or "7200")
    try:
        value = int(raw)
    except ValueError:
        value = 7200
    return max(60, value)


def _command_template() -> str:
    return _string(
        os.getenv("VIDEO_TO_WORLD_COMMAND_TEMPLATE")
        or "cd ${VIDEO_TO_WORLD_REPO_DIR:-/opt/video_to_world} && python run_reconstruction.py --config.input-video {INPUT_VIDEO} --config.scene-root {SCENE_ROOT} --config.mode fast"
    )


def _materialize_file(uri: str, path_hint: str, working_dir: Path) -> Path:
    hint = Path(path_hint) if path_hint else None
    if hint and hint.is_file():
        return hint
    if not uri:
        raise FileNotFoundError("missing_input_video")
    if is_gs_uri(uri):
        return ensure_local_uri_path(uri, gcs_root=_gcs_root(), scratch_dir=working_dir)
    candidate = Path(uri)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"input_not_found:{uri}")


def _copy_to_uri(source: Path, destination: str) -> str:
    if not destination:
        return ""
    if is_gs_uri(destination):
        mounted = resolve_gs_uri_to_path(destination, _gcs_root())
        ensure_dir(mounted.parent)
        shutil.copyfile(source, mounted)
        return destination
    target = Path(destination)
    ensure_dir(target.parent)
    shutil.copyfile(source, target)
    return str(target)


def _run_template(template: str, substitutions: Mapping[str, str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    rendered = template
    for key, value in substitutions.items():
        rendered = rendered.replace("{" + key + "}", value)
    return subprocess.run(
        rendered,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _write_npy(path: Path, array: np.ndarray) -> str:
    ensure_dir(path.parent)
    np.save(path, array)
    return str(path)


def _matrix44_from_w2c(matrix_3x4: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :4] = matrix_3x4
    return np.linalg.inv(m)


def _normalize_npz_outputs(*, scene_root: Path, geometry_root: Path) -> Dict[str, Any]:
    npz_path = scene_root / "exports" / "npz" / "results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"video_to_world_results_missing:{npz_path}")
    data = np.load(npz_path)
    depths = np.asarray(data["depth"])
    confs = np.asarray(data["conf"])
    extrinsics = np.asarray(data["extrinsics"])
    intrinsics = np.asarray(data["intrinsics"])
    images = np.asarray(data["image"])

    frames_dir = geometry_root / "frames" / "images"
    depth_dir = geometry_root / "depth"
    confidence_dir = geometry_root / "confidence"
    ensure_dir(frames_dir)
    ensure_dir(depth_dir)
    ensure_dir(confidence_dir)

    h = int(images.shape[1]) if images.ndim >= 3 else 0
    w = int(images.shape[2]) if images.ndim >= 3 else 0
    intr0 = intrinsics[0]
    intr_payload = {
        "camera_model": "pinhole",
        "image_width": w,
        "image_height": h,
        "fx": float(intr0[0, 0]),
        "fy": float(intr0[1, 1]),
        "cx": float(intr0[0, 2]),
        "cy": float(intr0[1, 2]),
        "distortion": {"model": "none", "coefficients": []},
    }

    frames: List[Dict[str, Any]] = []
    for idx in range(int(images.shape[0])):
        image_path = Path(_write_npy(frames_dir / f"frame_{idx:06d}.npy", images[idx]))
        depth_path = Path(_write_npy(depth_dir / f"depth_{idx:06d}.npy", depths[idx]))
        confidence_path = Path(_write_npy(confidence_dir / f"confidence_{idx:06d}.npy", confs[idx]))
        world_from_camera = _matrix44_from_w2c(np.asarray(extrinsics[idx], dtype=np.float64))
        camera_from_world = np.linalg.inv(world_from_camera)
        frames.append(
            {
                "frame_index": idx,
                "frame_id": str(idx).zfill(6),
                "timestamp_seconds": float(idx),
                "image_path": str(image_path),
                "is_keyframe": True,
                "blur_score": 0.0,
                "overlap_hint": 0.9,
                "world_from_camera": world_from_camera.tolist(),
                "camera_from_world": camera_from_world.tolist(),
                "pose_confidence": 0.9,
                "depth_path": str(depth_path),
                "confidence_path": str(confidence_path),
                "depth_format": "npy",
                "confidence_format": "npy",
                "width": w,
                "height": h,
                "min_depth_m": float(np.min(depths[idx])),
                "max_depth_m": float(np.max(depths[idx])),
                "confidence_range": [float(np.min(confs[idx])), float(np.max(confs[idx]))],
            }
        )

    result: Dict[str, Any] = {
        "status": "succeeded",
        "intrinsics": intr_payload,
        "frames": frames,
        "provider_metrics": {"backend": "video_to_world_npz"},
        "provider_warnings": [],
        "provider_errors": [],
        "loop_closure_detected": False,
    }
    for candidate in (
        scene_root / "frame_to_model_icp_50_2_offset0" / "after_global_optimization" / "aligned_points.ply",
        scene_root / "frame_to_model_icp_50_2_offset0" / "after_non_rigid_icp" / "aligned_points.ply",
    ):
        if candidate.is_file():
            result["canonical_pointcloud_source_path"] = str(candidate)
            break
    return result


def execute_video_to_world_request(body: Mapping[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="video_to_world_runner_") as tmp_dir:
        tmp = Path(tmp_dir)
        input_video = _materialize_file(
            _string(body.get("input_video_uri")),
            _string(body.get("input_video_path")),
            tmp,
        )
        geometry_root = Path(_string(body.get("geometry_root_path")) or str(tmp / "geometry"))
        ensure_dir(geometry_root)
        scene_root = geometry_root / "internal" / "video_to_world"
        ensure_dir(scene_root)
        result_json = tmp / "video_to_world_result.json"
        template = _command_template()
        if not template:
            return {"status": "failed", "reason": "video_to_world_command_not_configured"}
        proc = _run_template(
            template,
            {
                "INPUT_VIDEO": str(input_video),
                "GEOMETRY_ROOT": str(geometry_root),
                "SCENE_ROOT": str(scene_root),
                "RESULT_JSON": str(result_json),
                "DYNAMIC_MASK_MANIFEST": _string(body.get("dynamic_mask_manifest_path")),
            },
            _timeout_seconds(),
        )
        payload: Dict[str, Any]
        if result_json.is_file():
            try:
                payload = json.loads(result_json.read_text(encoding="utf-8"))
            except Exception:
                return {
                    "status": "failed",
                    "reason": "video_to_world_result_invalid_json",
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                }
        else:
            try:
                payload = _normalize_npz_outputs(scene_root=scene_root, geometry_root=geometry_root)
            except Exception:
                return {
                    "status": "failed",
                    "reason": f"video_to_world_command_failed:{proc.returncode}",
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                }
        if not isinstance(payload, dict):
            return {"status": "failed", "reason": "video_to_world_result_invalid_payload"}
        return payload
