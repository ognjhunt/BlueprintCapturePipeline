#!/usr/bin/env python3
"""Write the canonical Stage 1 reconstruction job spec."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


NORMALIZED_CONTRACT = [
    "export_last.usdz",
    "nvblox_mesh.ply",
    "visual_mesh.glb",
    "mesh_manifest.json",
    "occupancy.bin",
    "object_point_cloud_index.json",
    "capture_quality_report.json",
]


def _existing_file(path: str) -> str | None:
    value = (path or "").strip()
    if not value:
        return None
    candidate = Path(value)
    return str(candidate) if candidate.exists() else None


def build_job_spec(args: argparse.Namespace) -> dict[str, object]:
    capture: dict[str, object] = {
        "raw_video_path": str(Path(args.input_video)),
        "raw_video_uri": args.raw_video_uri or "",
    }
    optional_fields = {
        "arkit_poses_path": args.arkit_poses_path,
        "arkit_intrinsics_path": args.arkit_intrinsics_path,
        "arkit_depth_dir": args.arkit_depth_dir,
        "arkit_confidence_dir": args.arkit_confidence_dir,
        "scene_memory_conditioning_bundle_path": args.scene_memory_conditioning_bundle_path,
        "advanced_geometry_bundle_path": args.advanced_geometry_bundle_path,
    }
    for key, raw in optional_fields.items():
        value = _existing_file(raw)
        if value is not None:
            capture[key] = value

    if str(args.requested_backend).strip().lower() == "gen3c":
        has_geometry = bool(capture.get("advanced_geometry_bundle_path"))
        has_camera_bundle = all(
            bool(capture.get(key))
            for key in ("arkit_poses_path", "arkit_intrinsics_path", "arkit_depth_dir")
        )
        if not has_geometry and not has_camera_bundle:
            raise ValueError(
                "GEN3C requires arkit poses + intrinsics + depth, or an advanced geometry bundle"
            )

    return {
        "schema_version": "v1",
        "contract_version": "stage1_world_model_v1",
        "scene_id": args.scene_id,
        "capture_id": args.capture_id,
        "requested_backend": args.requested_backend,
        "capture": capture,
        "outputs": {
            "output_dir": str(Path(args.output_dir)),
            "compare_report_path": str(Path(args.compare_report_path)),
            "normalized_contract": list(NORMALIZED_CONTRACT),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Stage 1 reconstruction job spec")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--requested-backend", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--raw-video-uri", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-report-path", required=True)
    parser.add_argument("--arkit-poses-path", default="")
    parser.add_argument("--arkit-intrinsics-path", default="")
    parser.add_argument("--arkit-depth-dir", default="")
    parser.add_argument("--arkit-confidence-dir", default="")
    parser.add_argument("--scene-memory-conditioning-bundle-path", default="")
    parser.add_argument("--advanced-geometry-bundle-path", default="")
    args = parser.parse_args(argv)

    try:
        payload = build_job_spec(args)
    except ValueError as exc:
        parser.error(str(exc))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
