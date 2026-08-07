#!/usr/bin/env python3
"""Render sealed Aura 2D surflets at exact registered cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image


RESULT_NAME = "adp009d_aura_native_live_camera_result.json"
SOURCE_COMMIT = "f23b26c44ba84608306ba952510533ebf4c7877d"
SOURCE_TREE = "cc8447c66448b29bb4d39fec29c031df63d4b179"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array_artifact(path: Path, value: np.ndarray, root: Path) -> dict[str, Any]:
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _render_camera(*, source: Path, ply: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    import torch
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov

    calibration = config["calibration"]
    intrinsic = np.asarray(calibration["intrinsic_matrix"], dtype=np.float64)
    c2w = np.asarray(calibration["world_from_camera"], dtype=np.float64)
    width, height = (int(value) for value in calibration["resolution"])
    if (
        intrinsic.shape != (3, 3)
        or c2w.shape != (4, 4)
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(c2w).all()
        or abs(intrinsic[0, 2] - width / 2.0) > 1.0e-6
        or abs(intrinsic[1, 2] - height / 2.0) > 1.0e-6
    ):
        raise ValueError("aura_native_runtime_camera_invalid")
    w2c = np.linalg.inv(c2w)
    rotation_for_camera = w2c[:3, :3].T
    translation = w2c[:3, 3]
    dummy = torch.zeros((3, height, width), dtype=torch.float32, device="cpu")
    camera = Camera(
        colmap_id=0,
        R=rotation_for_camera,
        T=translation,
        FoVx=focal2fov(float(intrinsic[0, 0]), width),
        FoVy=focal2fov(float(intrinsic[1, 1]), height),
        image=dummy,
        gt_alpha_mask=None,
        image_name=str(config["camera_id"]),
        uid=0,
        data_device="cpu",
    )
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply))
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        depth_ratio=0.0,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    with torch.inference_mode():
        rendered = render(camera, gaussians, pipe, background)
    torch.cuda.synchronize()
    rgb_float = rendered["render"].permute(1, 2, 0).detach().cpu().numpy()
    rgb = (np.clip(np.nan_to_num(rgb_float), 0.0, 1.0) * 255.0).astype(np.uint8)
    depth = rendered["surf_depth"][0].detach().cpu().numpy().astype(np.float32)
    alpha = rendered["rend_alpha"][0].detach().cpu().numpy().astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output / "rgb.png", format="PNG")
    artifacts = [
        {
            "path": (output / "rgb.png").relative_to(output.parent).as_posix(),
            "sha256": _sha256(output / "rgb.png"),
            "size_bytes": (output / "rgb.png").stat().st_size,
            "dtype": "uint8",
            "shape": list(rgb.shape),
        },
        _array_artifact(output / "rgb.npy", rgb, output.parent),
        _array_artifact(output / "depth_m.npy", depth, output.parent),
        _array_artifact(output / "alpha.npy", alpha, output.parent),
    ]
    finite_depth = np.isfinite(depth) & (depth > 0) & (alpha > 0)
    valid = bool(rgb.std() > 0 and finite_depth.any() and np.isfinite(alpha).all())
    return {
        "camera_id": config["camera_id"],
        "valid": valid,
        "calibration": calibration,
        "calibration_digest": config["calibration_digest"],
        "native_reference_sha256": config["native_reference_sha256"],
        "rgb_pixel_std": float(rgb.std()),
        "positive_finite_depth_count": int(finite_depth.sum()),
        "minimum_positive_depth_m": (
            float(depth[finite_depth].min()) if finite_depth.any() else None
        ),
        "maximum_positive_depth_m": (
            float(depth[finite_depth].max()) if finite_depth.any() else None
        ),
        "alpha_positive_count": int((alpha > 0).sum()),
        "artifacts": artifacts,
    }


def run(*, runtime_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (runtime_dir / "adp009d_aura_native_provider_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source = runtime_dir / "AuraFusion360_official"
    ply = runtime_dir / "aura_sealed.ply"
    blockers: list[str] = []
    if (
        manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("source_tree") != SOURCE_TREE
        or _sha256(ply) != manifest.get("aura_ply_sha256")
        or manifest.get("depth_output") != "surf_depth_expected_camera_z_m"
        or manifest.get("depth_ratio") != 0.0
    ):
        blockers.append("aura_native_runtime_manifest_invalid")
    rows = []
    for binding in manifest.get("camera_configs", []):
        camera_id = str(binding.get("camera_id") or "")
        config_path = runtime_dir / "camera_configs" / f"{camera_id}.json"
        if not config_path.is_file() or _sha256(config_path) != binding.get(
            "configuration_sha256"
        ):
            blockers.append(f"aura_native_camera_config_changed:{camera_id}")
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        try:
            row = _render_camera(
                source=source,
                ply=ply,
                config=config,
                output=output_dir / camera_id,
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"aura_native_render_exception:{camera_id}:{type(exc).__name__}")
            continue
        if not row["valid"]:
            blockers.append(f"aura_native_render_invalid:{camera_id}")
        rows.append(row)
    return {
        "schema_version": "adp009d_aura_native_live_camera_result.v1",
        "status": (
            "completed"
            if not blockers
            and len(rows) == len(manifest.get("camera_configs", []))
            and len(rows) >= 2
            else "blocked"
        ),
        "blockers": sorted(set(blockers)),
        "implementation_commit": manifest.get("implementation_commit"),
        "input_digest": manifest.get("input_digest"),
        "source_repository": manifest.get("source_repository"),
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_submodules": manifest.get("source_submodules"),
        "source_modified": False,
        "aura_ply_sha256": manifest.get("aura_ply_sha256"),
        "source_probe_manifest_digest": manifest.get("source_probe_manifest_digest"),
        "aura_native_render_manifest_digest": manifest.get(
            "aura_native_render_manifest_digest"
        ),
        "camera_rows": rows,
        "depth_output": "surf_depth_expected_camera_z_m",
        "depth_ratio": 0.0,
        "metric_scene_units": "meters",
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "proof_boundary": (
            "Standalone official Aura exact-camera RGB/depth conformance probe; "
            "no Isaac composition or policy observation admitted yet."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            runtime_dir=args.runtime_dir.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema_version": "adp009d_aura_native_live_camera_result.v1",
            "status": "blocked",
            "blockers": [f"aura_native_provider_runner_exception:{type(exc).__name__}"],
            "error": str(exc),
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "provider_zero_required_after_return": True,
        }
    _write(args.output_dir / RESULT_NAME, result)
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
