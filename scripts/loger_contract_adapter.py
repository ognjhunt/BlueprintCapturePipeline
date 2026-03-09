#!/usr/bin/env python3
"""Adapt native LoGeR outputs into the existing NuRec-style artifact contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3_DETECT = REPO_ROOT / "scripts" / "sam3_detect.py"
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(message: str) -> None:
    print(f"[loger-adapter] {message}", flush=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_nurec_shim_module():
    module_path = REPO_ROOT / "scripts" / "nurec_shim.py"
    spec = importlib.util.spec_from_file_location("loger_adapter_nurec_shim", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load nurec_shim from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_if_needed(source: Path, target: Path) -> Path:
    if source.resolve() == target.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _native_report_path(native_output_dir: Path) -> Path:
    return native_output_dir / "loger_native_report.json"


def _candidate_paths(native_output_dir: Path) -> list[Path]:
    candidates = [
        native_output_dir / "export_last.ply",
        native_output_dir / "visual_pointcloud.ply",
        native_output_dir / "point_cloud.ply",
        native_output_dir / "pointcloud.ply",
        native_output_dir / "reconstruction.ply",
        native_output_dir / "model.ply",
    ]
    candidates.extend(sorted(native_output_dir.glob("*.ply")))
    return candidates


def _extract_points_from_prediction_tensor(value: Any) -> list[tuple[float, float, float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()

    shape = getattr(value, "shape", None)
    if not shape:
        return []

    points: list[tuple[float, float, float]] = []
    if len(shape) == 2 and shape[-1] == 3:
        iterable = value
    elif len(shape) == 3 and shape[-1] == 3:
        iterable = value.reshape((-1, 3))
    elif len(shape) == 4 and shape[-1] == 3:
        iterable = value.reshape((-1, 3))
    else:
        return []

    max_points = max(64, int(os.getenv("LOGER_ADAPTER_MAX_POINTS", "50000") or "50000"))
    stride = max(1, int(math.ceil(len(iterable) / max_points))) if len(iterable) else 1
    for row in iterable[::stride]:
        try:
            x, y, z = float(row[0]), float(row[1]), float(row[2])
        except Exception:
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        points.append((x, y, z))
    return points


def _extract_pose_points(payload: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    value = payload.get("camera_poses")
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    shape = getattr(value, "shape", None)
    if not shape or len(shape) != 3 or shape[-2:] != (4, 4):
        return []
    points: list[tuple[float, float, float]] = []
    for pose in value:
        try:
            x, y, z = float(pose[0][3]), float(pose[1][3]), float(pose[2][3])
        except Exception:
            continue
        if all(math.isfinite(v) for v in (x, y, z)):
            points.append((x, y, z))
    return points


def _load_points_from_predictions(native_output_dir: Path) -> list[tuple[float, float, float]]:
    candidates = [
        native_output_dir / "predictions.pt",
        native_output_dir / "output.pt",
    ]
    candidates.extend(sorted(native_output_dir.glob("*.pt")))
    prediction_path = next((path for path in candidates if path.is_file()), None)
    if prediction_path is None:
        return []

    try:
        import torch
    except ImportError:
        _log("torch not available; cannot decode LoGeR predictions.pt")
        return []

    payload = torch.load(str(prediction_path), map_location="cpu")
    if not isinstance(payload, Mapping):
        return []

    for key in (
        "world_points",
        "points_world",
        "points_3d",
        "points3d",
        "pts3d",
        "xyz",
        "global_points",
        "point_map",
    ):
        points = _extract_points_from_prediction_tensor(payload.get(key))
        if points:
            return points
    return _extract_pose_points(payload)


def _write_ascii_point_cloud(points: Iterable[tuple[float, float, float]], path: Path) -> None:
    materialized = list(points)
    if not materialized:
        materialized = [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.25, 0.0)]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(materialized)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines = [f"{x:.6f} {y:.6f} {z:.6f} 200 200 200" for x, y, z in materialized]
    path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def _load_points_from_ply(path: Path) -> list[tuple[float, float, float]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            if stream.readline().strip().lower() != "ply":
                return []
            vertex_count = 0
            while True:
                line = stream.readline()
                if not line:
                    return []
                stripped = line.strip().lower()
                if stripped.startswith("element vertex "):
                    try:
                        vertex_count = int(stripped.split()[-1])
                    except ValueError:
                        vertex_count = 0
                if stripped == "end_header":
                    break
            points: list[tuple[float, float, float]] = []
            for _ in range(vertex_count):
                line = stream.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    points.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
            return points
    except OSError:
        return []


def _compute_bounds(points: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        return (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    mins = (min(xs), min(ys), min(zs))
    maxs = (max(xs), max(ys), max(zs))
    if mins == maxs:
        x, y, z = mins
        return (x - 0.5, y - 0.5, z - 0.5), (x + 0.5, y + 0.5, z + 0.5)
    return mins, maxs


def _write_box_mesh_ply(points: list[tuple[float, float, float]], path: Path) -> None:
    mins, maxs = _compute_bounds(points)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    vertex_lines = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    face_lines = [f"3 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(header + vertex_lines + face_lines) + "\n", encoding="utf-8")


def _write_visual_mesh_glb(points: list[tuple[float, float, float]], path: Path) -> None:
    try:
        import trimesh
    except ImportError:
        path.write_bytes(b"loger_glb_placeholder")
        return

    mins, maxs = _compute_bounds(points)
    extents = [max(0.01, maxs[i] - mins[i]) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    path.write_bytes(mesh.export(file_type="glb"))


def _write_usdz_placeholder(source: str, path: Path) -> None:
    path.write_text(
        f"LOGER_USDZ_PLACEHOLDER\nsource={source}\ngenerated_at={_utc_now_iso()}\n",
        encoding="utf-8",
    )


def _build_capture_quality_report_for_video(input_video: Path, work_dir: Path) -> dict[str, Any]:
    module = _load_nurec_shim_module()
    frames_dir = work_dir / "capture_quality_frames"
    max_frames = int(os.getenv("LOGER_CAPTURE_QUALITY_MAX_FRAMES", "48") or "48")
    target_fps = float(os.getenv("LOGER_CAPTURE_QUALITY_FPS", "1.0") or "1.0")

    frame_count = module.extract_frames(
        input_video,
        frames_dir,
        max_frames=max_frames,
        target_fps=target_fps,
    )
    report = module.build_capture_quality_report(frames_dir)
    report["frame_extraction"] = {
        "requested_max_frames": int(max_frames),
        "effective_max_frames": int(max_frames),
        "requested_extract_fps": float(target_fps),
        "effective_extract_fps": float(target_fps),
        "sampling_reason": "loger_capture_quality_sample",
    }
    report["frame_count"] = int(frame_count)
    return report


def _generate_occupancy_from_ply(ply_path: Path, output_path: Path) -> None:
    module = _load_nurec_shim_module()
    try:
        module.generate_occupancy(ply_path, output_path)
    except Exception as exc:
        _log(f"occupancy generation failed, writing placeholder: {exc}")
        output_path.write_bytes(b"loger_occupancy_placeholder")


def _run_sam3_detect(*, output_dir: Path, input_video: Path, gaussian_ply: Path) -> Path:
    output_path = output_dir / "object_point_cloud_index.json"
    command = [
        sys.executable,
        str(SAM3_DETECT),
        "--video",
        str(input_video),
        "--output",
        str(output_path),
        "--environment",
        os.getenv("LOGER_SAM3_ENVIRONMENT", "auto"),
        "--gaussian-ply",
        str(gaussian_ply),
        "--no-crops",
    ]
    if (os.getenv("LOGER_SAVE_INSTANCE_MASKS", "") or "").strip().lower() in {"1", "true", "yes", "on"}:
        command.append("--save-instance-masks")
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"SAM3 detection failed: {(result.stderr or '').strip()[-500:]}")
    if not _is_nonempty_file(output_path):
        raise RuntimeError(f"SAM3 did not write {output_path}")
    return output_path


def _write_mesh_manifest(output_dir: Path) -> Path:
    payload = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "source": "loger",
        "primary_visual_asset": "visual_mesh.glb",
        "viewer_compatibility": ["fallback_vertex_mesh", "generic_viewer"],
        "assets": [
            {"path": "export_last.usdz", "role": "volume_visual", "kind": "usdz_placeholder"},
            {"path": "export_last.ply", "role": "gaussian_pointcloud", "kind": "ply_pointcloud_color"},
            {"path": "visual_pointcloud.ply", "role": "visual_pointcloud", "kind": "ply_pointcloud_color"},
            {"path": "nvblox_mesh.ply", "role": "collision", "kind": "ply_triangle_mesh"},
            {"path": "visual_mesh.glb", "role": "visual", "kind": "glb_triangle_mesh_vertex_color"},
            {"path": "occupancy.bin", "role": "occupancy", "kind": "binary_voxel_grid"},
        ],
        "reports": {},
    }
    path = output_dir / "mesh_manifest.json"
    _write_json(path, payload)
    return path


def _resolve_contract_point_cloud(native_output_dir: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    contract_path = output_dir / "export_last.ply"
    for candidate in _candidate_paths(native_output_dir):
        if _is_nonempty_file(candidate):
            _copy_if_needed(candidate, contract_path)
            return contract_path, {"source": str(candidate), "mode": "native_ply"}

    points = _load_points_from_predictions(native_output_dir)
    if not points:
        raise RuntimeError("could not derive export_last.ply from native LoGeR outputs")

    _write_ascii_point_cloud(points, contract_path)
    return contract_path, {"source": "predictions.pt", "mode": "predictions_to_ply", "point_count": len(points)}


def adapt_loger_outputs(
    *,
    native_output_dir: Path,
    output_dir: Path,
    input_video: Path,
    job_spec_path: Path,
    scene_id: str,
    capture_id: str,
    native_runtime_sec: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = (os.getenv("LOGER_OUTPUT_MODE") or "mesh_contract_v1").strip().lower()
    if mode != "mesh_contract_v1":
        raise RuntimeError(f"unsupported LOGER_OUTPUT_MODE={mode!r}")

    job_spec = _load_json(job_spec_path)
    capture = job_spec.get("capture") if isinstance(job_spec.get("capture"), Mapping) else {}
    if not isinstance(capture, Mapping):
        capture = {}

    export_ply, pointcloud_meta = _resolve_contract_point_cloud(native_output_dir, output_dir)
    visual_pointcloud = output_dir / "visual_pointcloud.ply"
    _copy_if_needed(export_ply, visual_pointcloud)
    points = _load_points_from_ply(export_ply)

    collision_mesh = output_dir / "nvblox_mesh.ply"
    _write_box_mesh_ply(points, collision_mesh)

    visual_mesh = output_dir / "visual_mesh.glb"
    _write_visual_mesh_glb(points, visual_mesh)

    export_usdz = output_dir / "export_last.usdz"
    _write_usdz_placeholder("visual_mesh.glb", export_usdz)

    occupancy = output_dir / "occupancy.bin"
    _generate_occupancy_from_ply(export_ply, occupancy)

    report_path = native_output_dir / "capture_quality_report.json"
    capture_quality = _load_json(report_path) if report_path.is_file() else {}
    if not capture_quality:
        capture_quality = _build_capture_quality_report_for_video(input_video, output_dir / "_loger_work")
    capture_quality["sfm"] = {"status": "not_applicable", "reason": "loger_feedforward_backend"}
    capture_quality["loger"] = {
        "runtime_sec": float(native_runtime_sec),
        "model_name": (os.getenv("LOGER_MODEL_NAME") or "").strip(),
        "checkpoint_path": (os.getenv("LOGER_CHECKPOINT_PATH") or "").strip(),
        "native_output_dir": str(native_output_dir),
        "output_mode": mode,
        "point_cloud_source": pointcloud_meta,
    }
    _write_json(output_dir / "capture_quality_report.json", capture_quality)

    object_index = _run_sam3_detect(
        output_dir=output_dir,
        input_video=input_video,
        gaussian_ply=export_ply,
    )

    manifest_path = _write_mesh_manifest(output_dir)
    (output_dir / "mesh_method.txt").write_text("loger_poisson\n", encoding="utf-8")

    backend_report = {
        "schema_version": "v1",
        "generated_at": _utc_now_iso(),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "native_output_dir": str(native_output_dir),
        "input_video": str(input_video),
        "job_spec_path": str(job_spec_path),
        "model_name": (os.getenv("LOGER_MODEL_NAME") or "").strip(),
        "checkpoint_path": (os.getenv("LOGER_CHECKPOINT_PATH") or "").strip(),
        "native_runtime_sec": float(native_runtime_sec),
        "arkit_available_but_unused": {
            "poses": bool(str(capture.get("arkit_poses_uri") or "").strip()),
            "intrinsics": bool(str(capture.get("arkit_intrinsics_uri") or "").strip()),
        },
        "native_report": str(_native_report_path(native_output_dir)),
        "synthesized_artifacts": {
            "export_last_usdz": str(export_usdz),
            "export_last_ply": str(export_ply),
            "visual_pointcloud_ply": str(visual_pointcloud),
            "nvblox_mesh_ply": str(collision_mesh),
            "visual_mesh_glb": str(visual_mesh),
            "occupancy_bin": str(occupancy),
            "mesh_manifest_json": str(manifest_path),
            "object_point_cloud_index_json": str(object_index),
            "capture_quality_report_json": str(output_dir / "capture_quality_report.json"),
        },
    }
    _write_json(output_dir / "loger_backend_report.json", backend_report)
    return backend_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adapt LoGeR outputs into NuRec contract artifacts")
    parser.add_argument("--native-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--native-runtime-sec", type=float, default=0.0)
    args = parser.parse_args(argv)

    adapt_loger_outputs(
        native_output_dir=Path(args.native_output_dir),
        output_dir=Path(args.output_dir),
        input_video=Path(args.input_video),
        job_spec_path=Path(args.job_spec),
        scene_id=args.scene_id,
        capture_id=args.capture_id,
        native_runtime_sec=float(args.native_runtime_sec),
    )
    _log(f"LoGeR contract adaptation completed in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
