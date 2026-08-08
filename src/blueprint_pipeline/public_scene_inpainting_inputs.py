"""Materialize render-derived ADP-009B inputs from an admitted InteriorGS scene.

InteriorGS does not publish the original capture photographs. This module makes
the substitute explicit: decode the publisher splat, render frozen translated
virtual cameras, and derive conservative masks from the publisher target OBB
and the Gaussians inside it. The output is an input packet, not an inpainting
result, object removal, or SimReady replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import (
    SplatData,
    convert_to_standard_ply,
    find_splat_transform_cli,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)

REQUEST_SCHEMA = "adp009b_interiorgs_edit_input_request.v1"
RECEIPT_SCHEMA = "adp009b_interiorgs_edit_input_receipt.v1"
RENDER_HARNESS_REL = "tools/splat_render/render_splat.mjs"
RENDER_ENTRY_REL = "tools/splat_render/src/render_entry.mjs"


class PublicSceneInpaintingInputError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_under(path: Path, roots: Sequence[Path], *, code: str) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PublicSceneInpaintingInputError([code])
    return resolved


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSceneInpaintingInputError([code]) from exc
    if not isinstance(value, dict):
        raise PublicSceneInpaintingInputError([code])
    return value


def build_public_scene_inpainting_input_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and digest the frozen, non-outcome camera/mask request."""

    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PublicSceneInpaintingInputError(["edit_input_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("edit_input_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1" or request.get("adp_item") != "ADP-009B":
        errors.append("edit_input_program_identity_invalid")
    if request.get("frozen_before_render") is not True:
        errors.append("edit_input_not_frozen_before_render")
    if request.get("method_outcomes_observed_before_freeze") is not False:
        errors.append("edit_input_method_outcome_leakage")
    if any(key in request for key in ("status", "admitted", "method_succeeded")):
        errors.append("edit_input_caller_asserted_outcome_forbidden")
    scene = request.get("scene")
    if not isinstance(scene, Mapping):
        errors.append("edit_input_scene_missing")
    else:
        for key in (
            "publisher_scene_id",
            "target_instance_id",
            "target_semantic_label",
            "component_manifest_path",
            "component_receipt_path",
        ):
            if not str(scene.get(key) or "").strip():
                errors.append(f"edit_input_scene_{key}_missing")
    rendering = request.get("rendering")
    if not isinstance(rendering, Mapping):
        errors.append("edit_input_rendering_missing")
    else:
        if rendering.get("renderer") != "reference_spark_renderer_exact_camera":
            errors.append("edit_input_renderer_invalid")
        if rendering.get("graphics_backend") not in {"swiftshader", "metal"}:
            errors.append("edit_input_graphics_backend_invalid")
        for key in ("width", "height"):
            item = rendering.get(key)
            if not isinstance(item, int) or isinstance(item, bool) or not 1024 <= item <= 4096:
                errors.append(f"edit_input_{key}_invalid")
        fov = rendering.get("vertical_fov_deg")
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not 25 <= float(fov) <= 90:
            errors.append("edit_input_fov_invalid")
    policy = request.get("camera_policy")
    if not isinstance(policy, Mapping):
        errors.append("edit_input_camera_policy_missing")
    else:
        if policy.get("generator") != "translated_target_coverage_v1":
            errors.append("edit_input_camera_generator_invalid")
        if policy.get("orbit_only_forbidden") is not True:
            errors.append("edit_input_orbit_only_must_be_forbidden")
        views = policy.get("views") if isinstance(policy.get("views"), list) else []
        if not 6 <= len(views) <= 16:
            errors.append("edit_input_camera_count_invalid")
        ids: set[str] = set()
        radii: list[float] = []
        heights: list[float] = []
        for row in views:
            if not isinstance(row, Mapping):
                errors.append("edit_input_camera_row_invalid")
                continue
            camera_id = str(row.get("camera_id") or "")
            if not camera_id or camera_id in ids or "/" in camera_id or ".." in camera_id:
                errors.append("edit_input_camera_id_invalid_or_duplicate")
            ids.add(camera_id)
            for field in ("position_offset_m", "target_offset_m"):
                vector = row.get(field)
                if (
                    not isinstance(vector, list)
                    or len(vector) != 3
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not math.isfinite(float(item))
                        for item in vector
                    )
                ):
                    errors.append(f"edit_input_camera_{field}_invalid")
            offset = row.get("position_offset_m")
            if isinstance(offset, list) and len(offset) == 3:
                radii.append(math.sqrt(sum(float(item) ** 2 for item in offset)))
                heights.append(float(offset[2]))
        if len({round(radius, 3) for radius in radii}) < 3 or len({round(z, 3) for z in heights}) < 3:
            errors.append("edit_input_camera_translation_baselines_insufficient")
    mask = request.get("mask_policy")
    if not isinstance(mask, Mapping):
        errors.append("edit_input_mask_policy_missing")
    else:
        if mask.get("authority") != "publisher_target_obb_plus_contained_gaussians":
            errors.append("edit_input_mask_authority_invalid")
        if mask.get("minimum_contained_gaussians", 0) < 16:
            errors.append("edit_input_minimum_gaussians_invalid")
        dilation = mask.get("dilation_pixels")
        if not isinstance(dilation, int) or isinstance(dilation, bool) or not 0 <= dilation <= 64:
            errors.append("edit_input_mask_dilation_invalid")
        maximum_fraction = mask.get("maximum_image_fraction", 0.2)
        if (
            isinstance(maximum_fraction, bool)
            or not isinstance(maximum_fraction, (int, float))
            or not 0.01 <= float(maximum_fraction) <= 0.85
        ):
            errors.append("edit_input_mask_maximum_image_fraction_invalid")
        contribution_threshold = mask.get("visual_contribution_threshold_8bit", 8)
        if (
            isinstance(contribution_threshold, bool)
            or not isinstance(contribution_threshold, int)
            or not 1 <= contribution_threshold <= 64
        ):
            errors.append("edit_input_visual_contribution_threshold_invalid")
        minimum_visible_fraction = mask.get("minimum_visible_target_fraction", 0.01)
        if (
            isinstance(minimum_visible_fraction, bool)
            or not isinstance(minimum_visible_fraction, (int, float))
            or not 0.001 <= float(minimum_visible_fraction) <= 0.9
        ):
            errors.append("edit_input_minimum_visible_target_fraction_invalid")
    if errors:
        raise PublicSceneInpaintingInputError(errors)
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        raise PublicSceneInpaintingInputError(["edit_input_request_digest_mismatch"])
    request["request_digest"] = expected
    return request


def _publisher_obb(labels: Any, instance_id: str, semantic_label: str) -> np.ndarray:
    if not isinstance(labels, list):
        raise PublicSceneInpaintingInputError(["edit_input_labels_not_list"])
    rows = [row for row in labels if isinstance(row, Mapping) and str(row.get("ins_id")) == instance_id]
    if len(rows) != 1:
        raise PublicSceneInpaintingInputError(["edit_input_target_identity_not_unique"])
    row = rows[0]
    if str(row.get("label") or "").strip().lower() != semantic_label.strip().lower():
        raise PublicSceneInpaintingInputError(["edit_input_target_semantic_label_mismatch"])
    corners = row.get("bounding_box")
    try:
        points = np.asarray([[item["x"], item["y"], item["z"]] for item in corners], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicSceneInpaintingInputError(["edit_input_target_obb_invalid"]) from exc
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise PublicSceneInpaintingInputError(["edit_input_target_obb_invalid"])
    return points


def _obb_basis(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tolerance = max(float(np.ptp(corners, axis=0).max()) * 1e-5, 1e-8)
    for origin in corners:
        vectors = [point - origin for point in corners if not np.array_equal(point, origin)]
        vectors.sort(key=lambda vector: (float(np.linalg.norm(vector)), *vector.tolist()))
        for candidates in itertools.combinations(vectors, 3):
            basis = np.column_stack(candidates)
            if abs(float(np.linalg.det(basis))) <= tolerance**3:
                continue
            generated = [
                origin + basis @ np.asarray(bits, dtype=np.float64)
                for bits in itertools.product((0, 1), repeat=3)
            ]
            if all(float(np.min(np.linalg.norm(corners - point, axis=1))) <= tolerance for point in generated):
                return origin, basis
    raise PublicSceneInpaintingInputError(["edit_input_target_obb_basis_unresolved"])


def _inside_obb(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    origin, basis = _obb_basis(corners)
    coordinates = np.linalg.solve(basis, (points - origin).T).T
    return np.isfinite(coordinates).all(axis=1) & np.all(
        (coordinates >= -1e-5) & (coordinates <= 1.0 + 1e-5), axis=1
    )


def _look_at_opencv(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-6:
        raise PublicSceneInpaintingInputError(["edit_input_camera_lookat_degenerate"])
    forward /= forward_norm
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-6:
        raise PublicSceneInpaintingInputError(["edit_input_camera_up_degenerate"])
    right /= right_norm
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = np.cross(forward, right)
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


def _camera_rows(request: Mapping[str, Any], target_center: np.ndarray) -> list[dict[str, Any]]:
    rendering = request["rendering"]
    width, height = int(rendering["width"]), int(rendering["height"])
    focal = 0.5 * height / math.tan(math.radians(float(rendering["vertical_fov_deg"])) / 2.0)
    rows = []
    for view in request["camera_policy"]["views"]:
        position = target_center + np.asarray(view["position_offset_m"], dtype=np.float64)
        target = target_center + np.asarray(view["target_offset_m"], dtype=np.float64)
        rows.append(
            {
                "camera_id": str(view["camera_id"]),
                "T_world_camera_opencv": _look_at_opencv(position, target).tolist(),
                "intrinsics": {
                    "model": "PINHOLE",
                    "fx": focal,
                    "fy": focal,
                    "cx": width / 2.0,
                    "cy": height / 2.0,
                    "width": width,
                    "height": height,
                },
            }
        )
    return rows


def _render_harness(
    *,
    splat: Path,
    cameras: Sequence[Mapping[str, Any]],
    output: Path,
    repo_root: Path,
    graphics_backend: str,
    warmup_ms: int,
    settle_frames: int,
    settle_ms: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    harness = repo_root / RENDER_HARNESS_REL
    if not harness.is_file() or shutil.which("node") is None:
        raise PublicSceneInpaintingInputError(["edit_input_render_runtime_missing"])
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "id": row["camera_id"],
            "spec": {
                "pose": {"T_world_camera_opencv": row["T_world_camera_opencv"]},
                "intrinsics": row["intrinsics"],
            },
        }
        for row in cameras
    ]
    camera_path = output.parent / f"{output.name}_cameras.json"
    camera_path.write_text(canonical_json(specs) + "\n", encoding="utf-8")
    width = int(cameras[0]["intrinsics"]["width"])
    height = int(cameras[0]["intrinsics"]["height"])
    command = [
        "node", str(harness), "--splat", str(splat), "--out", str(output),
        "--cameras", str(camera_path), "--width", str(width), "--height", str(height),
        "--bg", "0x000000", "--graphics-backend", graphics_backend,
        "--warmup-ms", str(warmup_ms), "--settle-frames", str(settle_frames),
        "--settle-ms", str(settle_ms),
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise PublicSceneInpaintingInputError(["edit_input_render_timeout"]) from exc
    stdout = process.stdout or ""
    try:
        result = json.loads(stdout[stdout.index("{") :])
    except (ValueError, json.JSONDecodeError) as exc:
        raise PublicSceneInpaintingInputError(["edit_input_render_result_invalid"]) from exc
    if process.returncode != 0 or result.get("status") != "completed":
        raise PublicSceneInpaintingInputError(
            ["edit_input_render_failed", *[f"render:{item}" for item in result.get("blockers", [])]]
        )
    return {"command": command, "result": result}


def _project_obb(corners: np.ndarray, camera: Mapping[str, Any]) -> list[tuple[float, float]]:
    pose = np.asarray(camera["T_world_camera_opencv"], dtype=np.float64)
    camera_points = (pose[:3, :3].T @ (corners - pose[:3, 3]).T).T
    if np.any(camera_points[:, 2] <= 1e-4):
        raise PublicSceneInpaintingInputError(["edit_input_target_obb_behind_camera"])
    intrinsics = camera["intrinsics"]
    u = float(intrinsics["fx"]) * camera_points[:, 0] / camera_points[:, 2] + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * camera_points[:, 1] / camera_points[:, 2] + float(intrinsics["cy"])
    points = sorted({(float(x), float(y)) for x, y in zip(u, v, strict=True)})

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_identity(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise PublicSceneInpaintingInputError(["edit_input_repository_identity_unavailable"]) from exc

    dirty = run("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise PublicSceneInpaintingInputError(["edit_input_repository_tracked_files_dirty"])
    return {
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "tracked_files_clean": True,
    }


def materialize_public_scene_inpainting_inputs(
    *, request_path: str | Path, repo_root: str | Path, data_root: str | Path,
    output_root: str | Path, receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize and receipt one real render-derived InteriorGS input packet."""

    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    output = _require_under(Path(output_root), (data,), code="edit_input_output_outside_data_root")
    retained_receipt = (
        _require_under(
            Path(receipt_output), (repo,), code="edit_input_receipt_output_outside_repo_root"
        )
        if receipt_output is not None
        else None
    )
    repository = _git_identity(repo)
    request_file = _require_under(Path(request_path), (repo,), code="edit_input_request_outside_repo_root")
    request = build_public_scene_inpainting_input_request(
        _read_object(request_file, code="edit_input_request_invalid")
    )
    scene = request["scene"]
    manifest_path = _require_under(
        repo / str(scene["component_manifest_path"]), (repo,),
        code="edit_input_component_manifest_outside_repo",
    )
    receipt_path = _require_under(
        repo / str(scene["component_receipt_path"]), (repo,),
        code="edit_input_component_receipt_outside_repo",
    )
    manifest = _read_object(manifest_path, code="edit_input_component_manifest_invalid")
    component_receipt = _read_object(receipt_path, code="edit_input_component_receipt_invalid")
    if canonical_digest(manifest, digest_field="manifest_digest") != manifest.get("manifest_digest"):
        raise PublicSceneInpaintingInputError(["edit_input_component_manifest_digest_mismatch"])
    if canonical_digest(component_receipt, digest_field="receipt_digest") != component_receipt.get("receipt_digest"):
        raise PublicSceneInpaintingInputError(["edit_input_component_receipt_digest_mismatch"])
    if (
        component_receipt.get("status") != "admitted"
        or component_receipt.get("component_manifest_digest") != manifest.get("manifest_digest")
    ):
        raise PublicSceneInpaintingInputError(["edit_input_scene_component_not_admitted"])
    mapping = manifest.get("scene_mapping") or {}
    target_binding = manifest.get("target_binding") or {}
    if str(mapping.get("publisher_scene_id")) != str(scene["publisher_scene_id"]):
        raise PublicSceneInpaintingInputError(["edit_input_scene_id_mismatch"])
    if (
        str(target_binding.get("interiorgs_instance_id")) != str(scene["target_instance_id"])
        or str(target_binding.get("semantic_label")).lower()
        != str(scene["target_semantic_label"]).lower()
    ):
        raise PublicSceneInpaintingInputError(["edit_input_target_binding_mismatch"])
    artifacts = {record["role"]: record for record in manifest.get("materialized_artifacts", [])}
    try:
        source_records = {
            "splat": artifacts["appearance_3dgs"],
            "labels": artifacts["semantic_metadata"],
            "structure": artifacts["scene_structure"],
        }
    except KeyError as exc:
        raise PublicSceneInpaintingInputError(["edit_input_scene_artifacts_missing"]) from exc
    observed_sources = []
    resolved: dict[str, Path] = {}
    for name, record in source_records.items():
        path = _require_under(
            data / str(record["external_relative_path"]), (data,),
            code="edit_input_source_outside_data_root",
        )
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256")
        ):
            raise PublicSceneInpaintingInputError([f"edit_input_{name}_bytes_changed"])
        resolved[name] = path
        observed_sources.append(
            {"role": name, "relative_path": path.relative_to(data).as_posix(), **_record(path, data)}
        )
    labels = json.loads(resolved["labels"].read_text(encoding="utf-8"))
    corners = _publisher_obb(
        labels, str(scene["target_instance_id"]), str(scene["target_semantic_label"])
    )
    output.mkdir(parents=True, exist_ok=True)
    standard_ply = output / "scene_standard.ply"
    decode_cli = find_splat_transform_cli(repo)
    decode_command = (
        ["node", str(decode_cli), "-w", "-q", str(resolved["splat"]), str(standard_ply)]
        if decode_cli is not None
        else None
    )
    conversion = convert_to_standard_ply(
        resolved["splat"], standard_ply, repo_root=repo, timeout_seconds=1800
    )
    if conversion.get("status") != "completed":
        raise PublicSceneInpaintingInputError(
            ["edit_input_splat_decode_failed", *conversion.get("blockers", [])]
        )
    splat = read_standard_3dgs_ply(standard_ply)
    selected = _inside_obb(np.asarray(splat.xyz, dtype=np.float64), corners)
    target_count = int(selected.sum())
    if target_count < int(request["mask_policy"]["minimum_contained_gaussians"]):
        raise PublicSceneInpaintingInputError(["edit_input_target_gaussian_support_insufficient"])
    target_splat = SplatData(
        count=target_count,
        xyz=splat.xyz[selected].copy(),
        opacity=np.full(target_count, 12.0, dtype=np.float32),
        f_dc=np.full((target_count, 3), np.float32(0.5 / 0.28209479177387814), dtype=np.float32),
        scales=splat.scales[selected].copy(),
        quats=splat.quats[selected].copy(),
        properties=splat.properties,
        sh_rest=None,
    )
    target_ply = write_standard_3dgs_ply(target_splat, output / "target_obb_gaussians.ply")
    retained = ~selected
    background_splat = SplatData(
        count=int(retained.sum()),
        xyz=splat.xyz[retained].copy(),
        opacity=splat.opacity[retained].copy(),
        f_dc=splat.f_dc[retained].copy(),
        scales=splat.scales[retained].copy(),
        quats=splat.quats[retained].copy(),
        properties=splat.properties,
        sh_rest=splat.sh_rest[retained].copy() if splat.sh_rest is not None else None,
    )
    background_ply = write_standard_3dgs_ply(
        background_splat, output / "scene_without_target_obb_gaussians.ply"
    )
    target_center = corners.mean(axis=0)
    cameras = _camera_rows(request, target_center)
    camera_file = output / "cameras.v1.json"
    camera_file.write_text(canonical_json(cameras) + "\n", encoding="utf-8")
    rendering = request["rendering"]
    common = {
        "cameras": cameras, "repo_root": repo,
        "graphics_backend": str(rendering["graphics_backend"]),
        "warmup_ms": int(rendering["warmup_ms"]),
        "settle_frames": int(rendering["settle_frames"]),
        "settle_ms": int(rendering["settle_ms"]),
        "timeout_seconds": int(rendering["timeout_seconds"]),
    }
    rgb_run = _render_harness(splat=standard_ply, output=output / "images", **common)
    support_run = _render_harness(splat=target_ply, output=output / "target_support", **common)
    background_run = _render_harness(
        splat=background_ply, output=output / "scene_without_target", **common
    )
    width, height = int(rendering["width"]), int(rendering["height"])
    dilation = int(request["mask_policy"]["dilation_pixels"])
    mask_rows = []
    image_rows = []
    for camera in cameras:
        camera_id = camera["camera_id"]
        rgb = output / "images" / f"{camera_id}.png"
        support = output / "target_support" / f"{camera_id}.png"
        background = output / "scene_without_target" / f"{camera_id}.png"
        if not rgb.is_file() or not support.is_file() or not background.is_file():
            raise PublicSceneInpaintingInputError([f"edit_input_render_missing:{camera_id}"])
        rgb_pixels = np.asarray(Image.open(rgb).convert("RGB"))
        support_pixels = np.asarray(Image.open(support).convert("RGB"))
        background_pixels = np.asarray(Image.open(background).convert("RGB"))
        if (
            rgb_pixels.shape[:2] != (height, width)
            or support_pixels.shape[:2] != (height, width)
            or background_pixels.shape[:2] != (height, width)
        ):
            raise PublicSceneInpaintingInputError([f"edit_input_render_size_mismatch:{camera_id}"])
        if float(rgb_pixels.std()) < 1.0:
            raise PublicSceneInpaintingInputError([f"edit_input_rgb_blank:{camera_id}"])
        support_mask = Image.fromarray(
            (np.max(support_pixels, axis=2) >= int(request["mask_policy"]["support_threshold_8bit"]))
            .astype(np.uint8) * 255, mode="L",
        )
        if dilation:
            support_mask = support_mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
        obb_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(obb_mask).polygon(_project_obb(corners, camera), fill=255)
        final = Image.fromarray(
            np.maximum(np.asarray(obb_mask), np.asarray(support_mask)).astype(np.uint8), mode="L"
        )
        mask_path = output / "masks" / f"{camera_id}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(mask_path, format="PNG", optimize=False)
        final_pixels = np.asarray(final) > 0
        support_binary = np.asarray(support_mask) > 0
        coverage = float(final_pixels.mean())
        maximum_fraction = float(request["mask_policy"].get("maximum_image_fraction", 0.2))
        if not 0.00001 < coverage < maximum_fraction or int(support_binary.sum()) == 0:
            raise PublicSceneInpaintingInputError([f"edit_input_mask_invalid:{camera_id}"])
        support_inside = float((support_binary & final_pixels).sum() / support_binary.sum())
        if support_inside < float(request["mask_policy"]["minimum_support_inside_final_fraction"]):
            raise PublicSceneInpaintingInputError([f"edit_input_mask_support_mismatch:{camera_id}"])
        contribution = np.max(
            np.abs(rgb_pixels.astype(np.int16) - background_pixels.astype(np.int16)), axis=2
        ) >= int(request["mask_policy"].get("visual_contribution_threshold_8bit", 8))
        visible_pixels = int((contribution & final_pixels).sum())
        visible_fraction = float(visible_pixels / final_pixels.sum())
        if visible_fraction < float(
            request["mask_policy"].get("minimum_visible_target_fraction", 0.01)
        ):
            raise PublicSceneInpaintingInputError(
                [f"edit_input_target_occluded_or_unrenderable:{camera_id}"]
            )
        image_rows.append({"camera_id": camera_id, **_record(rgb, output)})
        mask_rows.append(
            {"camera_id": camera_id, **_record(mask_path, output),
             "masked_pixel_count": int(final_pixels.sum()), "image_fraction": round(coverage, 9),
             "gaussian_support_inside_fraction": round(support_inside, 9),
             "visible_target_contribution_pixel_count": visible_pixels,
             "visible_target_contribution_fraction": round(visible_fraction, 9),
             "scene_without_target_render": _record(background, output)}
        )
    renderer = {
        "name": "reference_spark_renderer_exact_camera",
        "harness_sha256": _sha256(repo / RENDER_HARNESS_REL),
        "entry_sha256": _sha256(repo / RENDER_ENTRY_REL),
        "node_version": subprocess.run(
            ["node", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "graphics_backend": rendering["graphics_backend"], "width": width, "height": height,
        "warmup_ms": rendering["warmup_ms"], "settle_frames": rendering["settle_frames"],
        "settle_ms": rendering["settle_ms"],
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "render_derived_input_packet_materialized",
        "program_id": "arm-decision-proof-v1", "adp_item": "ADP-009B",
        "repository": repository,
        "request_digest": request["request_digest"],
        "scene_component_manifest_digest": manifest["manifest_digest"],
        "scene_component_receipt_digest": component_receipt["receipt_digest"],
        "scene": {
            "publisher_scene_id": str(scene["publisher_scene_id"]),
            "target_instance_id": str(scene["target_instance_id"]),
            "target_semantic_label": str(scene["target_semantic_label"]),
            "target_obb_corners_m": corners.tolist(), "target_gaussian_count": target_count,
            "scene_gaussian_count": int(splat.count),
        },
        "source_artifacts": observed_sources,
        "derived_artifacts": {
            "standard_splat": _record(standard_ply, output),
            "target_gaussian_support": _record(target_ply, output),
            "scene_without_target_obb_gaussians": _record(background_ply, output),
            "cameras": _record(camera_file, output), "images": image_rows, "masks": mask_rows,
        },
        "camera_policy": {
            "generator": "translated_target_coverage_v1", "orbit_only": False,
            "camera_count": len(cameras),
            "radii_m": [
                round(float(np.linalg.norm(np.asarray(row["T_world_camera_opencv"])[:3, 3] - target_center)), 6)
                for row in cameras
            ],
        },
        "mask_policy": {
            "authority": request["mask_policy"]["authority"],
            "dilation_pixels": dilation,
            "maximum_image_fraction": float(
                request["mask_policy"].get("maximum_image_fraction", 0.2)
            ),
            "visual_contribution_threshold_8bit": int(
                request["mask_policy"].get("visual_contribution_threshold_8bit", 8)
            ),
            "minimum_visible_target_fraction": float(
                request["mask_policy"].get("minimum_visible_target_fraction", 0.01)
            ),
        },
        "renderer": renderer,
        "executed_commands": {
            "decode": conversion.get("command") or decode_command, "rgb_render": rgb_run["command"],
            "target_support_render": support_run["command"],
            "scene_without_target_render": background_run["command"],
        },
        "method_execution": {
            "inpaint360gs_executed": False, "infusion_executed": False,
            "aurafusion360_executed": False,
        },
        "proof_boundaries": {
            "uses_original_capture_frames": False, "uses_rendered_scene_consistent_rgb": True,
            "hidden_background_truth_available": False,
            "source_target_obb_visual_contribution_measured": True,
            "source_object_removed_from_appearance": False, "source_collider_removed": False,
            "simready_replacement_inserted": False, "inpainting_result": False,
        },
        "smallest_next_blocker": "method_native_interiorgs_adapter_and_unchanged_author_runtime_required",
        "claim_ceiling": "synthetic_public_scene_inpainting_input_candidate",
        "replay_command": (
            "python -m blueprint_pipeline.public_scene_inpainting_inputs "
            f"--request {request_file.relative_to(repo).as_posix()} --repo-root . "
            f"--data-root {data} --output-root {output}"
        ),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (output / "adp009b_interiorgs_edit_input_receipt.v1.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    if retained_receipt is not None:
        retained_receipt.parent.mkdir(parents=True, exist_ok=True)
        retained_receipt.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args(argv)
    receipt = materialize_public_scene_inpainting_inputs(
        request_path=args.request, repo_root=args.repo_root, data_root=args.data_root,
        output_root=args.output_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


__all__ = [
    "PublicSceneInpaintingInputError", "RECEIPT_SCHEMA", "REQUEST_SCHEMA",
    "build_public_scene_inpainting_input_request", "materialize_public_scene_inpainting_inputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
