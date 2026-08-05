#!/usr/bin/env python3
"""Render every frozen Inpaint360 camera before starting a paid training stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize(rows: list[dict[str, Any]], *, expected_count: int) -> dict[str, Any]:
    blockers: list[str] = []
    if len(rows) != expected_count:
        blockers.append("inpaint360_camera_rasterizer_view_count_mismatch")
    failed = [str(row.get("image_name")) for row in rows if row.get("status") != "rendered"]
    if failed:
        blockers.append("inpaint360_camera_rasterizer_view_failed")
    return {
        "schema_version": "adp_inpaint360_camera_rasterizer_preflight.v1",
        "status": "accepted" if not blockers else "blocked",
        "expected_camera_count": expected_count,
        "observed_camera_count": len(rows),
        "failed_camera_names": failed,
        "views": rows,
        "blockers": blockers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-camera-count", type=int, required=True)
    args = parser.parse_args(argv)
    scene_info = json.loads(
        (Path(args.source_path) / args.object_path / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    args.num_classes = int(scene_info["num_classes"])
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    cameras = [
        *(("train", camera) for camera in scene.getTrainCameras()),
        *(("test", camera) for camera in scene.getTestCameras()),
    ]
    rows: list[dict[str, Any]] = []
    for split, camera in cameras:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row: dict[str, Any] = {
            "split": split,
            "image_name": str(camera.image_name),
            "width": int(camera.image_width),
            "height": int(camera.image_height),
        }
        try:
            with torch.no_grad():
                rendered = render(camera, gaussians, pipe, background)
                torch.cuda.synchronize()
            radii = rendered["radii"]
            row.update(
                {
                    "status": "rendered",
                    "visible_gaussian_count": int((radii > 0).sum().item()),
                    "maximum_projected_radius_px": float(radii.max().item()),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            message = " ".join(str(exc).split())
            row.update(
                {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error_summary": message[:1000],
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
        finally:
            if "rendered" in locals():
                del rendered
            torch.cuda.empty_cache()
        rows.append(row)
    receipt = _summarize(rows, expected_count=args.expected_camera_count)
    receipt.update(
        {
            "resolution_argument": int(args.resolution),
            "source_path": str(Path(args.source_path).resolve()),
            "model_path": str(Path(args.model_path).resolve()),
            "vanilla_3dgs_path": str(Path(args.vanilla_3dgs_path).resolve()),
            "raw_secret_values_recorded": False,
        }
    )
    _write_json(Path(args.receipt), receipt)
    return 0 if receipt["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
