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
import re
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
LEGACY_AUTHORIZATION_CLASS = "legacy_unqualified"
LEGACY_RENDER_PURPOSE = "legacy_reference_render"
EVALUATION_AUTHORIZATION_CLASS = "evaluation_authorized"
SUPPORTED_AUTHORIZATION_CLASSES = frozenset(
    {
        LEGACY_AUTHORIZATION_CLASS,
        "reconnaissance_preview",
        "method_input",
        EVALUATION_AUTHORIZATION_CLASS,
        "review_only",
    }
)
QUALIFIED_AUTHORIZATION_CLASSES = frozenset(
    {"method_input", EVALUATION_AUTHORIZATION_CLASS, "review_only"}
)
SUPPORTED_COLOR_SPACE = "srgb"
SUPPORTED_ALPHA_MODE = "opaque_rgb"
SUPPORTED_EXPOSURE_MODE = "renderer_default_unmodified"


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


def _standard_ply_vertex_count(path: Path) -> int | None:
    """Return the declared standard-PLY vertex count without decoding pixels."""

    if path.suffix.lower() != ".ply":
        return None
    with path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    end = header.find(b"end_header")
    if end < 0:
        return None
    match = re.search(rb"(?m)^element vertex ([0-9]+)\r?$", header[:end])
    return int(match.group(1)) if match else None


def _renderer_source_identity(root: Path, *, node_version: str) -> dict[str, Any]:
    package_manifest = root / "tools/splat_render/package.json"
    package_lock = root / "tools/splat_render/package-lock.json"
    version: str | None = None
    dependency_versions: dict[str, str] = {}
    if package_manifest.is_file():
        try:
            parsed = json.loads(package_manifest.read_text(encoding="utf-8"))
            version = str(parsed.get("version") or "").strip() or None
        except (OSError, json.JSONDecodeError):
            version = None
    if package_lock.is_file():
        try:
            lock = json.loads(package_lock.read_text(encoding="utf-8"))
            packages = lock.get("packages", {})
            if isinstance(packages, Mapping):
                for package_name in (
                    "@sparkjsdev/spark",
                    "playwright",
                    "three",
                ):
                    row = packages.get(f"node_modules/{package_name}")
                    if isinstance(row, Mapping) and row.get("version"):
                        dependency_versions[package_name] = str(row["version"])
        except (OSError, json.JSONDecodeError):
            dependency_versions = {}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    exact_revision = revision.stdout.strip() if revision.returncode == 0 else None
    renderer_files = [
        RENDER_HARNESS_REL,
        RENDER_ENTRY_REL,
        "tools/splat_render/package.json",
        "tools/splat_render/package-lock.json",
    ]
    cleanliness = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *renderer_files],
        cwd=root,
        capture_output=True,
    )
    return {
        "repository_revision": exact_revision,
        "repository_renderer_files_clean": cleanliness.returncode == 0,
        "package_name": "blueprint-splat-render",
        "package_version": version,
        "dependency_versions": dependency_versions,
        "package_manifest_digest": (
            _sha256_file(package_manifest) if package_manifest.is_file() else None
        ),
        "package_lock_digest": _sha256_file(package_lock) if package_lock.is_file() else None,
        "node_version": node_version or None,
    }


def _camera_specs_from_calibration_file(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedCameraRenderError(["render_calibrated_camera_file_invalid"]) from exc
    rows = value.get("cameras") if isinstance(value, Mapping) else value
    if not isinstance(rows, list):
        raise SealedCameraRenderError(["render_calibrated_camera_file_invalid"])
    specs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SealedCameraRenderError(["render_calibrated_camera_file_invalid"])
        if "id" in row and isinstance(row.get("spec"), Mapping):
            spec = row["spec"]
            pose = spec.get("pose") if isinstance(spec, Mapping) else None
            intrinsics = spec.get("intrinsics") if isinstance(spec, Mapping) else None
            matrix = pose.get("T_world_camera_opencv") if isinstance(pose, Mapping) else None
            camera_id = row.get("id")
        else:
            intrinsics = row.get("intrinsics")
            matrix = row.get("T_world_camera_provider_frame")
            camera_id = row.get("camera_id")
        if not isinstance(intrinsics, Mapping):
            raise SealedCameraRenderError(["render_calibrated_camera_file_invalid"])
        try:
            normalized_intrinsics = {
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "width": int(intrinsics["width"]),
                "height": int(intrinsics["height"]),
            }
            for key in ("near", "far"):
                if intrinsics.get(key) is not None:
                    normalized_intrinsics[key] = float(intrinsics[key])
            pose_matrix = np.asarray(matrix, dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise SealedCameraRenderError(["render_calibrated_camera_file_invalid"]) from exc
        if pose_matrix.shape != (4, 4) or not np.isfinite(pose_matrix).all():
            raise SealedCameraRenderError(["render_calibrated_camera_file_invalid"])
        specs.append(
            {
                "id": str(camera_id or ""),
                "spec": {
                    "pose": {"T_world_camera_opencv": pose_matrix.tolist()},
                    "intrinsics": normalized_intrinsics,
                },
            }
        )
    return specs


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
    calibrated_camera_file: str | Path | None = None,
    retained_gaussian_count: int | None = None,
    source_splat_digest: str | None = None,
    purpose: str | None = None,
    authorization_class: str = LEGACY_AUTHORIZATION_CLASS,
    supersampling: int = 1,
    color_space: str = SUPPORTED_COLOR_SPACE,
    alpha_mode: str = SUPPORTED_ALPHA_MODE,
    exposure_mode: str = SUPPORTED_EXPOSURE_MODE,
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
    one image size because the renderer canvas is sized once.  A render becomes
    ``evaluation_authorized`` only when an exact calibration JSON containing
    the same normalized rows and a nonempty purpose are supplied; legacy calls
    remain explicitly unqualified.  The current renderer supports one sample
    per output pixel, opaque sRGB output, and its unmodified exposure path.
    """

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    harness = root / RENDER_HARNESS_REL
    entry = root / RENDER_ENTRY_REL
    splat = Path(splat_path)
    output = Path(output_dir)
    errors: list[str] = []
    exact_splat_digest = _sha256_file(splat) if splat.is_file() else None
    declared_vertex_count = _standard_ply_vertex_count(splat) if splat.is_file() else None
    if source_splat_digest is not None and source_splat_digest != exact_splat_digest:
        errors.append("render_source_splat_digest_mismatch")
    if retained_gaussian_count is not None and (
        isinstance(retained_gaussian_count, bool) or retained_gaussian_count <= 0
    ):
        errors.append("render_retained_gaussian_count_invalid")
    if (
        retained_gaussian_count is not None
        and declared_vertex_count is not None
        and retained_gaussian_count != declared_vertex_count
    ):
        errors.append("render_retained_gaussian_count_mismatch")
    exact_retained_count = retained_gaussian_count or declared_vertex_count
    if authorization_class not in SUPPORTED_AUTHORIZATION_CLASSES:
        errors.append("render_authorization_class_invalid")
    if purpose is not None and not str(purpose).strip():
        errors.append("render_purpose_invalid")
    if isinstance(supersampling, bool) or supersampling != 1:
        errors.append("render_supersampling_unsupported")
    if color_space != SUPPORTED_COLOR_SPACE:
        errors.append("render_color_space_unsupported")
    if alpha_mode != SUPPORTED_ALPHA_MODE:
        errors.append("render_alpha_mode_unsupported")
    if exposure_mode != SUPPORTED_EXPOSURE_MODE:
        errors.append("render_exposure_mode_unsupported")
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
    camera_calibration_path = (
        Path(calibrated_camera_file) if calibrated_camera_file is not None else None
    )
    if authorization_class in QUALIFIED_AUTHORIZATION_CLASSES:
        if purpose is None:
            errors.append("render_evaluation_purpose_missing")
        if camera_calibration_path is None:
            errors.append("render_evaluation_calibrated_camera_file_missing")
        if exact_retained_count is None:
            errors.append("render_evaluation_retained_gaussian_count_missing")
    if camera_calibration_path is not None:
        if camera_calibration_path.is_symlink() or not camera_calibration_path.is_file():
            errors.append("render_calibrated_camera_file_missing_or_symlink")
        elif camera_specs:
            try:
                file_specs = _camera_specs_from_calibration_file(camera_calibration_path)
            except SealedCameraRenderError as exc:
                errors.extend(exc.codes)
            else:
                if canonical_json(file_specs) != canonical_json(camera_specs):
                    errors.append("render_calibrated_camera_file_mismatch")
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
    renderer_source_identity = _renderer_source_identity(root, node_version=node_version)
    if authorization_class in QUALIFIED_AUTHORIZATION_CLASSES and (
        not renderer_source_identity["repository_revision"]
        or not renderer_source_identity["repository_renderer_files_clean"]
        or not renderer_source_identity["package_version"]
        or not renderer_source_identity["package_lock_digest"]
        or not renderer_source_identity["dependency_versions"].get("@sparkjsdev/spark")
    ):
        raise SealedCameraRenderError(["render_evaluation_renderer_identity_incomplete"])
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
        "purpose": str(purpose or LEGACY_RENDER_PURPOSE),
        "authorization_class": str(authorization_class),
        "provider_splat_import_receipt_digest": str(provider_splat_import_receipt_digest),
        "provider_reconstruction_alignment_digest": str(alignment_digest),
        "splat_digest": exact_splat_digest,
        "source_splat": {
            "digest": exact_splat_digest,
            "retained_gaussian_count": exact_retained_count,
            "retained_count_source": (
                "verified_standard_ply_header"
                if declared_vertex_count is not None
                else "caller_bound"
                if retained_gaussian_count is not None
                else "unavailable"
            ),
        },
        "calibrated_camera_file": {
            "digest": (
                _sha256_file(camera_calibration_path)
                if camera_calibration_path is not None
                else _sha256_file(cameras_json)
            ),
            "binding": (
                "caller_file_exact_match"
                if camera_calibration_path is not None
                else "runtime_rows_materialized_legacy_unqualified"
            ),
            "camera_count": len(camera_specs),
        },
        "calibrated_cameras": camera_specs,
        "projection_pixel_convention": PROJECTION_PIXEL_CONVENTION,
        "render_settings": {
            "dimensions": {"width": width, "height": height},
            "supersampling": supersampling,
            "color_space": color_space,
            "alpha_mode": alpha_mode,
            "background_rgb": f"#{background_rgb:06x}",
            "exposure": {"mode": exposure_mode, "ev": None},
        },
        "renderer_identity": {
            "harness_digest": _sha256_file(harness),
            "render_entry_digest": _sha256_file(entry),
            **renderer_source_identity,
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
    "EVALUATION_AUTHORIZATION_CLASS",
    "LEGACY_AUTHORIZATION_CLASS",
    "LEGACY_RENDER_PURPOSE",
    "PROJECTION_PIXEL_CONVENTION",
    "QUALIFIED_AUTHORIZATION_CLASSES",
    "RENDER_MANIFEST_SCHEMA_VERSION",
    "SealedCameraRenderError",
    "SUPPORTED_AUTHORIZATION_CLASSES",
    "render_splat_at_exact_cameras",
    "transform_camera_into_provider_frame",
]
