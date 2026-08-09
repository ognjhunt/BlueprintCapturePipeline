"""Render an accepted provider splat at exact cameras with a pinned renderer.

This is the evaluator-side bridge between an imported provider splat (plus its
candidate-frame alignment) and independent held-out evaluation.  The module
maps requested cameras into the provider frame with the recorded similarity
transform, drives the repository's reference Spark renderer with exact
OpenCV-pose/intrinsics camera specs, and emits a digest-bound render manifest.

It renders whatever cameras the caller supplies: candidate cameras for the
pre-evaluation sanity loop, evaluator-owned hidden cameras during sealed
evaluation.  The module itself never opens hidden pixels and cannot upgrade
appearance results into metric, collision, Isaac, task, physical, or
deployment claims.  Renders are reference-renderer output, not Isaac RTX.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json


RENDER_MANIFEST_SCHEMA_VERSION = "sealed_camera_render_manifest.v1"
RENDER_HARNESS_REL = "tools/splat_render/render_splat.mjs"
RENDER_ENTRY_REL = "tools/splat_render/src/render_entry.mjs"
RENDERED_BY = "reference_spark_renderer_exact_camera"
PROJECTION_PIXEL_CONVENTION = "colmap_pixel_center_half_offset"


class SealedCameraRenderError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def transform_camera_into_provider_frame(
    *,
    camera_to_world_candidate: Sequence[Sequence[float]],
    alignment: Mapping[str, Any],
) -> list[list[float]]:
    """Map a candidate-frame OpenCV camera pose into the provider frame.

    The alignment maps provider -> candidate as ``x_c = s R x_p + t``; the
    inverse pose keeps the camera seeing the provider-frame scene exactly as
    the candidate-frame camera sees the candidate-frame scene (pinhole
    projection is similarity-invariant, so intrinsics stay unchanged).
    """

    matrix = np.asarray(camera_to_world_candidate, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SealedCameraRenderError(["render_camera_pose_invalid"])
    scale = float(alignment["estimated_scale_factor"])
    rotation = np.asarray(alignment["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or scale <= 0.0:
        raise SealedCameraRenderError(["render_alignment_invalid"])
    provider_rotation = rotation.T @ matrix[:3, :3]
    provider_position = rotation.T @ (matrix[:3, 3] - translation) / scale
    provider = np.eye(4)
    provider[:3, :3] = provider_rotation
    provider[:3, 3] = provider_position
    return [[float(value) for value in row] for row in provider]


def render_splat_at_exact_cameras(
    *,
    splat_path: str | Path,
    cameras: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    provider_splat_import_receipt_digest: str,
    alignment_digest: str,
    camera_set_label: str,
    repo_root: str | Path | None = None,
    node: str = "node",
    graphics_backend: str = "swiftshader",
    background_rgb: int = 0x0B0B10,
    warmup_ms: int = 2500,
    settle_frames: int = 6,
    settle_ms: int = 100,
    render_timeout: int = 3600,
) -> dict[str, Any]:
    """Render the exact splat file at exact cameras and return a digest-bound manifest.

    ``cameras`` rows: ``{"camera_id", "T_world_camera_provider_frame" (4x4),
    "intrinsics": {fx, fy, cx, cy, width, height}}``.  All cameras must share
    one image size because the renderer canvas is sized once.
    """

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    harness = root / RENDER_HARNESS_REL
    entry = root / RENDER_ENTRY_REL
    splat = Path(splat_path)
    output = Path(output_dir)
    errors: list[str] = []
    if (
        isinstance(background_rgb, bool)
        or not isinstance(background_rgb, int)
        or not 0 <= background_rgb <= 0xFFFFFF
    ):
        errors.append("render_background_rgb_invalid")
    if shutil.which(node) is None:
        errors.append("render_node_runtime_unavailable")
    if not harness.is_file() or not entry.is_file():
        errors.append("render_harness_unavailable")
    if splat.is_symlink() or not splat.is_file():
        errors.append("render_splat_missing_or_symlink")
    if not cameras:
        errors.append("render_cameras_missing")
    if not str(camera_set_label or "").strip():
        errors.append("render_camera_set_label_missing")
    sizes = set()
    camera_specs = []
    seen_ids: set[str] = set()
    for row in cameras:
        camera_id = str(row.get("camera_id") or "")
        if not camera_id or camera_id in seen_ids or "/" in camera_id or ".." in camera_id:
            errors.append("render_camera_id_invalid_or_duplicate")
            continue
        seen_ids.add(camera_id)
        intrinsics = row.get("intrinsics")
        matrix = np.asarray(row.get("T_world_camera_provider_frame"), dtype=np.float64)
        if (
            not isinstance(intrinsics, Mapping)
            or matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
        ):
            errors.append("render_camera_row_invalid")
            continue
        try:
            width, height = int(intrinsics["width"]), int(intrinsics["height"])
            spec_intrinsics = {
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "width": width,
                "height": height,
            }
            for key in ("near", "far"):
                if intrinsics.get(key) is not None:
                    spec_intrinsics[key] = float(intrinsics[key])
        except (KeyError, TypeError, ValueError):
            errors.append("render_camera_intrinsics_invalid")
            continue
        sizes.add((width, height))
        camera_specs.append(
            {
                "id": camera_id,
                "spec": {
                    "pose": {"T_world_camera_opencv": matrix.tolist()},
                    "intrinsics": spec_intrinsics,
                },
            }
        )
    if len(sizes) > 1:
        errors.append("render_mixed_image_sizes_unsupported")
    if errors:
        raise SealedCameraRenderError(errors)
    width, height = next(iter(sizes))
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    cameras_json = output / "exact_cameras.json"
    cameras_json.write_text(json.dumps(camera_specs), encoding="utf-8")
    command = [
        node,
        str(harness),
        "--splat",
        str(splat),
        "--out",
        str(frames_dir),
        "--cameras",
        str(cameras_json),
        "--width",
        str(width),
        "--height",
        str(height),
        "--warmup-ms",
        str(warmup_ms),
        "--settle-frames",
        str(settle_frames),
        "--settle-ms",
        str(settle_ms),
        "--graphics-backend",
        str(graphics_backend),
        "--bg",
        f"0x{background_rgb:06x}",
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=render_timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise SealedCameraRenderError(["render_harness_timeout"]) from exc
    harness_output: dict[str, Any] = {}
    stdout = (process.stdout or "").strip()
    if stdout:
        try:
            harness_output = json.loads(stdout[stdout.index("{") :])
        except (ValueError, json.JSONDecodeError):
            harness_output = {}
    if process.returncode != 0 or harness_output.get("status") != "completed":
        raise SealedCameraRenderError(
            [
                "render_harness_failed",
                *[
                    f"render_blocker:{blocker}"
                    for blocker in harness_output.get("blockers", [])
                ],
            ]
        )
    node_version = subprocess.run(
        [node, "--version"], capture_output=True, text=True
    ).stdout.strip()
    rendered_rows = []
    for spec in camera_specs:
        frame_path = frames_dir / f"{spec['id']}.png"
        if not frame_path.is_file():
            raise SealedCameraRenderError([f"render_frame_missing:{spec['id']}"])
        with Image.open(frame_path) as image:
            if image.size != (width, height):
                raise SealedCameraRenderError([f"render_frame_size_mismatch:{spec['id']}"])
            pixels = np.asarray(image.convert("RGB"))
        if int(pixels.std()) == 0:
            raise SealedCameraRenderError([f"render_frame_blank:{spec['id']}"])
        rendered_rows.append(
            {
                "camera_id": spec["id"],
                "relative_path": f"frames/{spec['id']}.png",
                "digest": _sha256_file(frame_path),
                "width": width,
                "height": height,
                "pixel_std": round(float(pixels.std()), 4),
            }
        )
    manifest = {
        "schema_version": RENDER_MANIFEST_SCHEMA_VERSION,
        "status": "rendered_exact_cameras",
        "rendered_by": RENDERED_BY,
        "camera_set_label": str(camera_set_label),
        "provider_splat_import_receipt_digest": str(provider_splat_import_receipt_digest),
        "provider_reconstruction_alignment_digest": str(alignment_digest),
        "splat_digest": _sha256_file(splat),
        "projection_pixel_convention": PROJECTION_PIXEL_CONVENTION,
        "renderer_identity": {
            "harness_digest": _sha256_file(harness),
            "render_entry_digest": _sha256_file(entry),
            "node_version": node_version,
            "graphics_backend": str(graphics_backend),
            "background_rgb": f"#{background_rgb:06x}",
            "warmup_ms": warmup_ms,
            "settle_frames": settle_frames,
            "settle_ms": settle_ms,
        },
        "renders": rendered_rows,
        "render_count": len(rendered_rows),
        "rendered_by_isaac_rtx": False,
        "hidden_pixels_read_by_renderer": False,
        "proof_effect": "reference_render_for_independent_evaluation_only",
        "claim_ceiling": "appearance_reconstruction_candidate",
    }
    manifest["sealed_camera_render_manifest_digest"] = canonical_digest(
        manifest, digest_field="sealed_camera_render_manifest_digest"
    )
    (output / "sealed_camera_render_manifest.v1.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "PROJECTION_PIXEL_CONVENTION",
    "RENDER_MANIFEST_SCHEMA_VERSION",
    "SealedCameraRenderError",
    "render_splat_at_exact_cameras",
    "transform_camera_into_provider_frame",
]
