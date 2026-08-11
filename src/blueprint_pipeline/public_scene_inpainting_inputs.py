"""Materialize render-derived inputs from an admitted InteriorGS scene.

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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import validate_scene_freeze, validate_task_freeze
from .gaussian_splat_decode import (
    SplatData,
    convert_to_standard_ply,
    find_splat_transform_cli,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)
from .sealed_camera_render import (
    SealedCameraRenderError,
    render_splat_at_exact_cameras,
)

REQUEST_SCHEMA = "adp009b_interiorgs_edit_input_request.v1"
RECEIPT_SCHEMA = "adp009b_interiorgs_edit_input_receipt.v1"
REQUEST_SCHEMA_V2 = "public_scene_interiorgs_edit_input_request.v2"
RECEIPT_SCHEMA_V2 = "public_scene_interiorgs_edit_input_receipt.v2"
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


def _semantic_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def build_public_scene_inpainting_input_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and digest the frozen, non-outcome camera/mask request."""

    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PublicSceneInpaintingInputError(["edit_input_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    schema = request.get("schema_version")
    if schema not in {REQUEST_SCHEMA, REQUEST_SCHEMA_V2}:
        errors.append("edit_input_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1" or not str(
        request.get("adp_item") or ""
    ).startswith("ADP-"):
        errors.append("edit_input_program_identity_invalid")
    if schema == REQUEST_SCHEMA and request.get("adp_item") != "ADP-009B":
        errors.append("edit_input_legacy_program_identity_invalid")
    if request.get("frozen_before_render") is not True:
        errors.append("edit_input_not_frozen_before_render")
    if request.get("method_outcomes_observed_before_freeze") is not False:
        errors.append("edit_input_method_outcome_leakage")
    if any(key in request for key in ("status", "admitted", "method_succeeded")):
        errors.append("edit_input_caller_asserted_outcome_forbidden")
    scene = request.get("scene")
    if not isinstance(scene, Mapping):
        errors.append("edit_input_scene_missing")
    elif schema == REQUEST_SCHEMA:
        for key in (
            "publisher_scene_id",
            "target_instance_id",
            "target_semantic_label",
            "component_manifest_path",
            "component_receipt_path",
        ):
            if not str(scene.get(key) or "").strip():
                errors.append(f"edit_input_scene_{key}_missing")
    else:
        if scene.get("source_adapter") != "dual_task_freeze_and_standard_splat_v1":
            errors.append("edit_input_scene_source_adapter_invalid")
        for key in (
            "scene_freeze_path",
            "task_freeze_path",
            "standard_splat_conversion_receipt_path",
            "standard_splat_path",
            "labels_path",
            "structure_path",
            "registered_frame_receipt_path",
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
    if _semantic_key(row.get("label")) != _semantic_key(semantic_label):
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


def _verified_dual_task_scene_source(
    *, scene: Mapping[str, Any], repo: Path, data: Path
) -> dict[str, Any]:
    """Normalize the dual-task source adapter into one verified render source.

    The adapter accepts only paths.  Every scientific identity is derived from
    opened, digest-verified freezes and receipts; callers cannot assert scene,
    target, registration, or conversion qualification fields directly.
    """

    scene_freeze_path = _require_under(
        repo / str(scene["scene_freeze_path"]),
        (repo,),
        code="edit_input_scene_freeze_outside_repo",
    )
    task_freeze_path = _require_under(
        repo / str(scene["task_freeze_path"]),
        (repo,),
        code="edit_input_task_freeze_outside_repo",
    )
    try:
        scene_freeze = validate_scene_freeze(
            _read_object(scene_freeze_path, code="edit_input_scene_freeze_invalid")
        )
        task_freeze = validate_task_freeze(
            _read_object(task_freeze_path, code="edit_input_task_freeze_invalid")
        )
    except ValueError as exc:
        raise PublicSceneInpaintingInputError(
            ["edit_input_dual_task_freeze_invalid"]
        ) from exc
    if task_freeze["scene_freeze_digest"] != scene_freeze["scene_freeze_digest"]:
        raise PublicSceneInpaintingInputError(["edit_input_task_scene_freeze_mismatch"])

    conversion_receipt_path = _require_under(
        repo / str(scene["standard_splat_conversion_receipt_path"]),
        (repo,),
        code="edit_input_conversion_receipt_outside_repo",
    )
    conversion_receipt = _read_object(
        conversion_receipt_path, code="edit_input_conversion_receipt_invalid"
    )
    if (
        canonical_digest(conversion_receipt, digest_field="receipt_digest")
        != conversion_receipt.get("receipt_digest")
        or conversion_receipt.get("schema_version")
        != "standard_splat_conversion_receipt.v1"
        or conversion_receipt.get("status")
        != "standard_splat_conversion_materialized"
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_conversion_receipt_not_qualified"]
        )
    appearance_source = scene_freeze["source_components"]["interiorgs"]
    conversion_source = conversion_receipt.get("source") or {}
    if (
        conversion_source.get("sha256") != appearance_source.get("sha256")
        or conversion_source.get("size_bytes") != appearance_source.get("size_bytes")
        or conversion_receipt.get("raw_source_uploaded") is not False
        or conversion_receipt.get("gaussian_ownership_claimed") is not False
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_conversion_source_join_invalid"]
        )
    standard_ply = _require_under(
        data / str(scene["standard_splat_path"]),
        (data,),
        code="edit_input_standard_splat_outside_data_root",
    )
    output_record = conversion_receipt.get("output") or {}
    if (
        not standard_ply.is_file()
        or standard_ply.is_symlink()
        or standard_ply.stat().st_size != output_record.get("size_bytes")
        or _sha256(standard_ply) != output_record.get("sha256")
        or output_record.get("standard_3dgs_schema_validated") is not True
        or output_record.get("gaussian_count_preserved") is not True
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_standard_splat_bytes_changed"]
        )

    support_records = appearance_source.get("supporting_files") or {}
    resolved_support: dict[str, Path] = {}
    for role, request_key in (("labels", "labels_path"), ("structure", "structure_path")):
        record = support_records.get(role) or {}
        path = _require_under(
            data / str(scene[request_key]),
            (data,),
            code=f"edit_input_{role}_outside_data_root",
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise PublicSceneInpaintingInputError(
                [f"edit_input_{role}_bytes_changed"]
            )
        resolved_support[role] = path

    frame_path = _require_under(
        data / str(scene["registered_frame_receipt_path"]),
        (data,),
        code="edit_input_registered_frame_outside_data_root",
    )
    frame = _read_object(frame_path, code="edit_input_registered_frame_invalid")
    if (
        canonical_digest(frame, digest_field="receipt_digest")
        != frame.get("receipt_digest")
        or frame.get("schema_version")
        != "interiorgs_sage_shared_frame_candidate.v1"
        or frame.get("source_digests", {}).get("interiorgs_labels")
        != support_records.get("labels", {}).get("sha256")
        or frame.get("source_digests", {}).get("sage_collision_usd")
        != scene_freeze["source_components"]["sage_collision"].get("sha256")
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_registered_frame_join_invalid"]
        )
    target = task_freeze["source_object"]
    collider_path = task_freeze["removal_plan"]["source_collider_prim_path"]
    correspondences = [
        row
        for row in frame.get("correspondences", [])
        if isinstance(row, Mapping)
        and str(row.get("interiorgs_instance_id")) == str(target["instance_id"])
    ]
    if (
        len(correspondences) != 1
        or _semantic_key(correspondences[0].get("semantic_label"))
        != _semantic_key(target["semantic_label"])
        or correspondences[0].get("sage_prim_path") != collider_path
        or correspondences[0].get("identity_receipt_digest")
        != target.get("collision_identity_receipt_digest")
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_registered_target_join_invalid"]
        )
    labels = json.loads(resolved_support["labels"].read_text(encoding="utf-8"))
    corners = _publisher_obb(
        labels, str(target["instance_id"]), str(target["semantic_label"])
    )
    observed_bounds = target["observed_bounds_world_m"]
    if not (
        np.allclose(corners.min(axis=0), observed_bounds["minimum"], atol=1e-6)
        and np.allclose(corners.max(axis=0), observed_bounds["maximum"], atol=1e-6)
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_target_bounds_freeze_mismatch"]
        )
    return {
        "scene_id": str(scene_freeze["selected_scene_id"]),
        "task_id": str(task_freeze["task_id"]),
        "target_instance_id": str(target["instance_id"]),
        "target_semantic_label": str(target["semantic_label"]),
        "scene_freeze_digest": str(scene_freeze["scene_freeze_digest"]),
        "task_freeze_digest": str(task_freeze["task_freeze_digest"]),
        "mask_set_id": str(task_freeze["removal_plan"]["mask_set_id"]),
        "removal_id": str(task_freeze["removal_plan"]["removal_id"]),
        "standard_ply": standard_ply,
        "standard_splat_digest": str(output_record["sha256"]),
        "gaussian_count": int(output_record["gaussian_count"]),
        "conversion_receipt_digest": str(conversion_receipt["receipt_digest"]),
        "registered_frame_receipt_digest": str(frame["receipt_digest"]),
        "registered_frame_status": str(frame.get("shared_frame_status") or "unavailable"),
        "corners": corners,
        "source_artifacts": [
            {
                "role": "standard_splat",
                "relative_path": standard_ply.relative_to(data).as_posix(),
                **_record(standard_ply, data),
            },
            *[
                {
                    "role": role,
                    "relative_path": path.relative_to(data).as_posix(),
                    **_record(path, data),
                }
                for role, path in resolved_support.items()
            ],
        ],
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
    output.mkdir(parents=True, exist_ok=True)
    source_adapter = str(scene.get("source_adapter") or "legacy_component_v1")
    source_identity: dict[str, Any]
    manifest: dict[str, Any] | None = None
    component_receipt: dict[str, Any] | None = None
    if source_adapter == "dual_task_freeze_and_standard_splat_v1":
        source_identity = _verified_dual_task_scene_source(
            scene=scene, repo=repo, data=data
        )
        standard_ply = source_identity["standard_ply"]
        observed_sources = source_identity["source_artifacts"]
        corners = source_identity["corners"]
        conversion = {
            "status": "reused_qualified_standard_splat",
            "command": [
                "reuse-standard-splat-conversion-receipt",
                source_identity["conversion_receipt_digest"],
            ],
        }
        decode_command = None
    else:
        manifest_path = _require_under(
            repo / str(scene["component_manifest_path"]), (repo,),
            code="edit_input_component_manifest_outside_repo",
        )
        receipt_path = _require_under(
            repo / str(scene["component_receipt_path"]), (repo,),
            code="edit_input_component_receipt_outside_repo",
        )
        manifest = _read_object(
            manifest_path, code="edit_input_component_manifest_invalid"
        )
        component_receipt = _read_object(
            receipt_path, code="edit_input_component_receipt_invalid"
        )
        if canonical_digest(
            manifest, digest_field="manifest_digest"
        ) != manifest.get("manifest_digest"):
            raise PublicSceneInpaintingInputError(
                ["edit_input_component_manifest_digest_mismatch"]
            )
        if canonical_digest(
            component_receipt, digest_field="receipt_digest"
        ) != component_receipt.get("receipt_digest"):
            raise PublicSceneInpaintingInputError(
                ["edit_input_component_receipt_digest_mismatch"]
            )
        if (
            component_receipt.get("status") != "admitted"
            or component_receipt.get("component_manifest_digest")
            != manifest.get("manifest_digest")
        ):
            raise PublicSceneInpaintingInputError(
                ["edit_input_scene_component_not_admitted"]
            )
        mapping = manifest.get("scene_mapping") or {}
        target_binding = manifest.get("target_binding") or {}
        if str(mapping.get("publisher_scene_id")) != str(
            scene["publisher_scene_id"]
        ):
            raise PublicSceneInpaintingInputError(["edit_input_scene_id_mismatch"])
        if (
            str(target_binding.get("interiorgs_instance_id"))
            != str(scene["target_instance_id"])
            or _semantic_key(target_binding.get("semantic_label"))
            != _semantic_key(scene["target_semantic_label"])
        ):
            raise PublicSceneInpaintingInputError(
                ["edit_input_target_binding_mismatch"]
            )
        artifacts = {
            record["role"]: record
            for record in manifest.get("materialized_artifacts", [])
        }
        try:
            source_records = {
                "splat": artifacts["appearance_3dgs"],
                "labels": artifacts["semantic_metadata"],
                "structure": artifacts["scene_structure"],
            }
        except KeyError as exc:
            raise PublicSceneInpaintingInputError(
                ["edit_input_scene_artifacts_missing"]
            ) from exc
        observed_sources = []
        resolved: dict[str, Path] = {}
        for name, record in source_records.items():
            path = _require_under(
                data / str(record["external_relative_path"]),
                (data,),
                code="edit_input_source_outside_data_root",
            )
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record.get("size_bytes")
                or _sha256(path) != record.get("sha256")
            ):
                raise PublicSceneInpaintingInputError(
                    [f"edit_input_{name}_bytes_changed"]
                )
            resolved[name] = path
            observed_sources.append(
                {
                    "role": name,
                    "relative_path": path.relative_to(data).as_posix(),
                    **_record(path, data),
                }
            )
        labels = json.loads(resolved["labels"].read_text(encoding="utf-8"))
        corners = _publisher_obb(
            labels,
            str(scene["target_instance_id"]),
            str(scene["target_semantic_label"]),
        )
        standard_ply = output / "scene_standard.ply"
        decode_cli = find_splat_transform_cli(repo)
        decode_command = (
            [
                "node",
                str(decode_cli),
                "-w",
                "-q",
                str(resolved["splat"]),
                str(standard_ply),
            ]
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
        source_identity = {
            "scene_id": str(scene["publisher_scene_id"]),
            "task_id": None,
            "target_instance_id": str(scene["target_instance_id"]),
            "target_semantic_label": str(scene["target_semantic_label"]),
            "scene_freeze_digest": None,
            "task_freeze_digest": None,
            "mask_set_id": None,
            "removal_id": None,
            "conversion_receipt_digest": None,
            "registered_frame_receipt_digest": None,
            "registered_frame_status": "legacy_component_admission",
        }
    splat = read_standard_3dgs_ply(standard_ply)
    if source_adapter == "dual_task_freeze_and_standard_splat_v1" and (
        splat.count != source_identity["gaussian_count"]
    ):
        raise PublicSceneInpaintingInputError(
            ["edit_input_standard_splat_count_mismatch"]
        )
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
    sealed_cameras = [
        {
            "camera_id": row["camera_id"],
            "T_world_camera_provider_frame": row["T_world_camera_opencv"],
            "intrinsics": row["intrinsics"],
        }
        for row in cameras
    ]
    camera_file = output / "cameras.v1.json"
    camera_file.write_text(
        canonical_json(
            sealed_cameras
            if source_adapter == "dual_task_freeze_and_standard_splat_v1"
            else cameras
        )
        + "\n",
        encoding="utf-8",
    )
    rendering = request["rendering"]
    common = {
        "cameras": cameras, "repo_root": repo,
        "graphics_backend": str(rendering["graphics_backend"]),
        "warmup_ms": int(rendering["warmup_ms"]),
        "settle_frames": int(rendering["settle_frames"]),
        "settle_ms": int(rendering["settle_ms"]),
        "timeout_seconds": int(rendering["timeout_seconds"]),
    }
    sealed_render_manifests: dict[str, Any] = {}
    if source_adapter == "dual_task_freeze_and_standard_splat_v1":
        render_inputs = (
            ("images", standard_ply, int(splat.count), "complete source appearance"),
            (
                "target_support",
                target_ply,
                target_count,
                "candidate target OBB Gaussian support",
            ),
            (
                "scene_without_target",
                background_ply,
                int(retained.sum()),
                "source appearance without candidate target OBB Gaussians",
            ),
        )
        try:
            for label, splat_path, retained_count, purpose_label in render_inputs:
                manifest_row = render_splat_at_exact_cameras(
                    splat_path=splat_path,
                    cameras=sealed_cameras,
                    output_dir=output / label,
                    provider_splat_import_receipt_digest=source_identity[
                        "conversion_receipt_digest"
                    ],
                    alignment_digest=source_identity[
                        "registered_frame_receipt_digest"
                    ],
                    camera_set_label=(
                        f"{source_identity['task_id']}:removal_input:{label}"
                    ),
                    calibrated_camera_file=camera_file,
                    retained_gaussian_count=retained_count,
                    source_splat_digest=_sha256(splat_path),
                    purpose=(
                        f"{source_identity['task_id']} calibrated removal analysis: "
                        f"{purpose_label}"
                    ),
                    authorization_class="method_input",
                    supersampling=int(rendering.get("supersampling", 1)),
                    color_space=str(rendering.get("color_space", "srgb")),
                    alpha_mode=str(rendering.get("alpha_mode", "opaque_rgb")),
                    exposure_mode=str(
                        rendering.get("exposure_mode", "renderer_default_unmodified")
                    ),
                    repo_root=repo,
                    graphics_backend=str(rendering["graphics_backend"]),
                    background_rgb=int(rendering.get("background_rgb", 0)),
                    warmup_ms=int(rendering["warmup_ms"]),
                    settle_frames=int(rendering["settle_frames"]),
                    settle_ms=int(rendering["settle_ms"]),
                    render_timeout=int(rendering["timeout_seconds"]),
                )
                sealed_render_manifests[label] = manifest_row
        except SealedCameraRenderError as exc:
            raise PublicSceneInpaintingInputError(
                ["edit_input_authorized_render_failed", *exc.codes]
            ) from exc
        rgb_run = {
            "command": [
                "sealed-camera-render",
                sealed_render_manifests["images"][
                    "sealed_camera_render_manifest_digest"
                ],
            ]
        }
        support_run = {
            "command": [
                "sealed-camera-render",
                sealed_render_manifests["target_support"][
                    "sealed_camera_render_manifest_digest"
                ],
            ]
        }
        background_run = {
            "command": [
                "sealed-camera-render",
                sealed_render_manifests["scene_without_target"][
                    "sealed_camera_render_manifest_digest"
                ],
            ]
        }
        render_frame_subdir = "frames"
    else:
        rgb_run = _render_harness(
            splat=standard_ply, output=output / "images", **common
        )
        support_run = _render_harness(
            splat=target_ply, output=output / "target_support", **common
        )
        background_run = _render_harness(
            splat=background_ply, output=output / "scene_without_target", **common
        )
        render_frame_subdir = ""
    width, height = int(rendering["width"]), int(rendering["height"])
    dilation = int(request["mask_policy"]["dilation_pixels"])
    mask_rows = []
    image_rows = []
    for camera in cameras:
        camera_id = camera["camera_id"]
        rgb = output / "images" / render_frame_subdir / f"{camera_id}.png"
        support = output / "target_support" / render_frame_subdir / f"{camera_id}.png"
        background = (
            output / "scene_without_target" / render_frame_subdir / f"{camera_id}.png"
        )
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
    if source_adapter == "dual_task_freeze_and_standard_splat_v1":
        renderer = {
            "name": "reference_spark_renderer_exact_camera",
            "authorization_class": "method_input",
            "purpose_bound": True,
            "render_manifest_digests": {
                label: row["sealed_camera_render_manifest_digest"]
                for label, row in sealed_render_manifests.items()
            },
            "render_settings": sealed_render_manifests["images"]["render_settings"],
            "renderer_identity": sealed_render_manifests["images"][
                "renderer_identity"
            ],
        }
        standard_splat_record = next(
            row for row in observed_sources if row["role"] == "standard_splat"
        )
        source_admission = {
            "adapter": source_adapter,
            "scene_freeze_digest": source_identity["scene_freeze_digest"],
            "task_freeze_digest": source_identity["task_freeze_digest"],
            "standard_splat_conversion_receipt_digest": source_identity[
                "conversion_receipt_digest"
            ],
            "registered_frame_receipt_digest": source_identity[
                "registered_frame_receipt_digest"
            ],
            "registered_frame_status": source_identity["registered_frame_status"],
        }
    else:
        renderer = {
            "name": "reference_spark_renderer_exact_camera",
            "authorization_class": "legacy_unqualified",
            "harness_sha256": _sha256(repo / RENDER_HARNESS_REL),
            "entry_sha256": _sha256(repo / RENDER_ENTRY_REL),
            "node_version": subprocess.run(
                ["node", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "graphics_backend": rendering["graphics_backend"],
            "width": width,
            "height": height,
            "warmup_ms": rendering["warmup_ms"],
            "settle_frames": rendering["settle_frames"],
            "settle_ms": rendering["settle_ms"],
        }
        standard_splat_record = _record(standard_ply, output)
        source_admission = {
            "adapter": source_adapter,
            "scene_component_manifest_digest": manifest["manifest_digest"],
            "scene_component_receipt_digest": component_receipt["receipt_digest"],
        }
    receipt = {
        "schema_version": (
            RECEIPT_SCHEMA_V2
            if source_adapter == "dual_task_freeze_and_standard_splat_v1"
            else RECEIPT_SCHEMA
        ),
        "status": "render_derived_input_packet_materialized",
        "program_id": "arm-decision-proof-v1",
        "adp_item": request["adp_item"],
        "repository": repository,
        "request_digest": request["request_digest"],
        "source_admission": source_admission,
        "scene": {
            "publisher_scene_id": source_identity["scene_id"],
            "task_id": source_identity["task_id"],
            "target_instance_id": source_identity["target_instance_id"],
            "target_semantic_label": source_identity["target_semantic_label"],
            "mask_set_id": source_identity["mask_set_id"],
            "removal_id": source_identity["removal_id"],
            "target_obb_corners_m": corners.tolist(), "target_gaussian_count": target_count,
            "scene_gaussian_count": int(splat.count),
        },
        "source_artifacts": observed_sources,
        "derived_artifacts": {
            "standard_splat": standard_splat_record,
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
        "camera_pose_contract": {
            "schema_version": "public_scene_inpainting_camera_pose_contract.v1",
            "camera_file_pose_field": (
                "T_world_camera_provider_frame"
                if source_adapter == "dual_task_freeze_and_standard_splat_v1"
                else "T_world_camera_opencv"
            ),
            "semantic_pose_field": "T_world_camera_opencv",
            "camera_coordinate_convention": "opencv_x_right_y_down_z_forward",
            "provider_frame_aliases_opencv": (
                source_adapter == "dual_task_freeze_and_standard_splat_v1"
            ),
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
            "mask_is_calibrated_candidate_not_owned_gaussian_classification": True,
            "gaussian_ownership_qualified": False,
        },
        "smallest_next_blocker": (
            "independent_gaussian_contribution_ownership_and_replacement_depth_coverage"
            if source_adapter == "dual_task_freeze_and_standard_splat_v1"
            else "method_native_interiorgs_adapter_and_unchanged_author_runtime_required"
        ),
        "claim_ceiling": "synthetic_public_scene_inpainting_input_candidate",
        "replay_command": (
            "python -m blueprint_pipeline.public_scene_inpainting_inputs "
            f"--request {request_file.relative_to(repo).as_posix()} --repo-root . "
            f"--data-root {data} --output-root {output}"
        ),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output_receipt_name = (
        "public_scene_interiorgs_edit_input_receipt.v2.json"
        if source_adapter == "dual_task_freeze_and_standard_splat_v1"
        else "adp009b_interiorgs_edit_input_receipt.v1.json"
    )
    (output / output_receipt_name).write_text(
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
