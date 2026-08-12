"""Materialize full projected repair support for broad Gaussian excision.

The broad cutout is intentionally not a cleaned-scene claim.  It may remove
protected background Gaussians together with a source object.  This module
turns that collateral deletion into explicit repair support by projecting the
exact shared deleted layer into every calibrated camera.  Any later editor
must remain byte-exact outside the materialized support.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .public_scene_gaussian_excision_heldout import (
    derive_alpha_from_background_pair,
)


PACKET_SCHEMA = "public_scene_broad_repair_support_packet.v1"
RELOCATION_SCHEMA = "adp009d_retained_scene_gpu_render_output_relocation.v1"
RESULT_SCHEMA = "adp009d_retained_scene_gpu_render_result.v1"
RENDER_SCHEMA = "sealed_camera_render_manifest.v1"
SUPPORTED_CANDIDATE_SCHEMAS = frozenset(
    {
        "adp009b_direct_evidence_expansion_set.v1",
        "adp009b_ownership_coverage_cutout_set.v1",
    }
)


class BroadRepairSupportError(ValueError):
    """Stable fail-closed errors for broad repair-support materialization."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BroadRepairSupportError([code]) from exc
    if not isinstance(value, dict):
        raise BroadRepairSupportError([code])
    return value


def _file(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise BroadRepairSupportError([code])
    return path


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        record["path"] = str(path)
    else:
        record["relative_path"] = path.relative_to(root).as_posix()
    return record


def _verified_record(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise BroadRepairSupportError([code])
    path = _file(str(value.get("path") or ""), code=code)
    if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256(path):
        raise BroadRepairSupportError([code])
    return path


def _verified_relative_record(root: Path, value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise BroadRepairSupportError([code])
    relative = str(value.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise BroadRepairSupportError([code])
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise BroadRepairSupportError([code])
    if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256(path):
        raise BroadRepairSupportError([code])
    return path


def _validate_candidate(path: Path) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    candidate = _read(path, code="broad_repair_candidate_unreadable")
    if (
        candidate.get("schema_version") not in SUPPORTED_CANDIDATE_SCHEMAS
        or candidate.get("receipt_digest")
        != canonical_digest(candidate, digest_field="receipt_digest")
        or (candidate.get("claim_boundary") or {}).get("candidate_derived_layers_only")
        is not True
    ):
        raise BroadRepairSupportError(["broad_repair_candidate_invalid"])
    tasks = candidate.get("task_candidates")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_REPLACEMENT_OBJECTS:
        raise BroadRepairSupportError(["broad_repair_task_count_invalid"])
    task_ids = {str(row.get("task_id") or "") for row in tasks if isinstance(row, Mapping)}
    if len(task_ids) != len(tasks) or "" in task_ids:
        raise BroadRepairSupportError(["broad_repair_task_ids_invalid"])
    union = candidate.get("shared_scene_union")
    outputs = union.get("outputs") if isinstance(union, Mapping) else None
    counts = union.get("counts") if isinstance(union, Mapping) else None
    if not isinstance(outputs, Mapping) or not isinstance(counts, Mapping):
        raise BroadRepairSupportError(["broad_repair_shared_union_invalid"])
    deleted = _verified_relative_record(
        path.parent,
        outputs.get("deleted_source_gaussians"),
        code="broad_repair_deleted_splat_invalid",
    )
    retained = _verified_relative_record(
        path.parent,
        outputs.get("retained_scene_gaussians"),
        code="broad_repair_retained_splat_invalid",
    )
    if (
        not isinstance(counts.get("deleted_total"), int)
        or not isinstance(counts.get("retained_total"), int)
        or int(counts["deleted_total"]) <= 0
        or int(counts["retained_total"]) <= 0
    ):
        raise BroadRepairSupportError(["broad_repair_shared_counts_invalid"])
    return candidate, task_ids, {
        "deleted_digest": _sha256(deleted),
        "retained_digest": _sha256(retained),
        "deleted_count": counts["deleted_total"],
        "retained_count": counts["retained_total"],
    }


def _validate_result(path: Path, *, candidate: Mapping[str, Any], union: Mapping[str, Any]) -> dict[str, Any]:
    result = _read(path, code="broad_repair_renderer_result_unreadable")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "completed"
        or result.get("released_renderer_executed") is not True
        or result.get("gpu_runtime_started") is not True
        or result.get("paid_inference_performed") is not False
        or result.get("provider_mutations_performed") != 0
        or result.get("candidate_set_digest") != candidate.get("receipt_digest")
        or result.get("shared_deleted_source_layer_digest") != union["deleted_digest"]
        or result.get("shared_retained_scene_digest") != union["retained_digest"]
        or result.get("blockers") != []
    ):
        raise BroadRepairSupportError(["broad_repair_renderer_result_invalid"])
    return result


def _validate_relocation(
    path: Path, *, result_path: Path, result: Mapping[str, Any]
) -> dict[tuple[str, str, str], Path]:
    receipt = _read(path, code="broad_repair_relocation_unreadable")
    if (
        receipt.get("schema_version") != RELOCATION_SCHEMA
        or receipt.get("status") != "extracted_manifest_paths_verified"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise BroadRepairSupportError(["broad_repair_relocation_invalid"])
    provider_result = receipt.get("provider_result")
    if not isinstance(provider_result, Mapping):
        raise BroadRepairSupportError(["broad_repair_relocation_result_invalid"])
    if (
        Path(str(provider_result.get("path") or "")).resolve() != result_path
        or provider_result.get("size_bytes") != result_path.stat().st_size
        or provider_result.get("sha256") != _sha256(result_path)
    ):
        raise BroadRepairSupportError(["broad_repair_relocation_result_invalid"])
    expected = {
        (
            str(row.get("task_id") or ""),
            str(row.get("layer") or ""),
            str(row.get("background_rgb") or ""),
        )
        for row in result.get("render_manifests") or []
        if isinstance(row, Mapping)
    }
    manifests: dict[tuple[str, str, str], Path] = {}
    for row in receipt.get("render_manifests") or []:
        if not isinstance(row, Mapping):
            raise BroadRepairSupportError(["broad_repair_relocation_manifest_invalid"])
        key = (
            str(row.get("task_id") or ""),
            str(row.get("layer") or ""),
            str(row.get("background_rgb") or ""),
        )
        local = row.get("local_manifest")
        manifest = _verified_record(local, code="broad_repair_relocation_manifest_invalid")
        observed = _read(manifest, code="broad_repair_render_manifest_unreadable")
        if not isinstance(local, Mapping) or local.get("manifest_digest") != observed.get(
            "sealed_camera_render_manifest_digest"
        ):
            raise BroadRepairSupportError(["broad_repair_relocation_manifest_invalid"])
        if key in manifests:
            raise BroadRepairSupportError(["broad_repair_relocation_manifest_duplicate"])
        manifests[key] = manifest
    if set(manifests) != expected:
        raise BroadRepairSupportError(["broad_repair_relocation_manifest_set_invalid"])
    return manifests


def _manifest_frames(
    path: Path,
    *,
    task_id: str,
    layer: str,
    background: str,
    candidate_digest: str,
    splat_digest: str,
    splat_count: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest = _read(path, code="broad_repair_render_manifest_unreadable")
    source = manifest.get("source_splat")
    settings = manifest.get("render_settings")
    if (
        manifest.get("schema_version") != RENDER_SCHEMA
        or manifest.get("status") != "rendered_exact_cameras"
        or manifest.get("sealed_camera_render_manifest_digest")
        != canonical_digest(manifest, digest_field="sealed_camera_render_manifest_digest")
        or manifest.get("authorization_class") not in {"method_input", "evaluation_authorized"}
        or manifest.get("rendered_by_gpu") is not True
        or manifest.get("candidate_set_digest") != candidate_digest
        or manifest.get("splat_digest") != splat_digest
        or not isinstance(source, Mapping)
        or source.get("digest") != splat_digest
        or source.get("retained_gaussian_count") != splat_count
        or not isinstance(settings, Mapping)
        or settings.get("background_rgb") != background
    ):
        raise BroadRepairSupportError(["broad_repair_render_manifest_invalid"])
    dimensions = settings.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise BroadRepairSupportError(["broad_repair_render_dimensions_invalid"])
    width, height = dimensions.get("width"), dimensions.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise BroadRepairSupportError(["broad_repair_render_dimensions_invalid"])
    cameras = manifest.get("calibrated_cameras")
    rows = manifest.get("renders")
    if not isinstance(cameras, list) or not isinstance(rows, list):
        raise BroadRepairSupportError(["broad_repair_render_rows_invalid"])
    camera_ids = [str(row.get("id") or "") for row in cameras if isinstance(row, Mapping)]
    if len(camera_ids) != len(cameras) or len(set(camera_ids)) != len(camera_ids):
        raise BroadRepairSupportError(["broad_repair_render_cameras_invalid"])
    frames: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise BroadRepairSupportError(["broad_repair_render_frame_invalid"])
        camera_id = str(row.get("camera_id") or "")
        relative = str(row.get("relative_path") or "")
        frame = (path.parent / relative).resolve()
        if (
            camera_id not in camera_ids
            or camera_id in frames
            or path.parent.resolve() not in frame.parents
            or not frame.is_file()
            or frame.is_symlink()
            or row.get("size_bytes") != frame.stat().st_size
            or row.get("digest") != _sha256(frame)
            or row.get("width") != width
            or row.get("height") != height
        ):
            raise BroadRepairSupportError(["broad_repair_render_frame_invalid"])
        with Image.open(frame) as image:
            if image.size != (width, height):
                raise BroadRepairSupportError(["broad_repair_render_frame_invalid"])
        frames[camera_id] = frame
    if set(frames) != set(camera_ids):
        raise BroadRepairSupportError(["broad_repair_render_frame_set_invalid"])
    return frames, {
        **_record(path),
        "task_id": task_id,
        "layer": layer,
        "background_rgb": background,
        "sealed_camera_render_manifest_digest": manifest[
            "sealed_camera_render_manifest_digest"
        ],
        "camera_ids": camera_ids,
        "dimensions": {"width": width, "height": height},
    }


def _bbox(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return None
    return {
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max_inclusive": int(xs.max()),
        "y_max_inclusive": int(ys.max()),
    }


def _save_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def materialize_broad_repair_support(
    *,
    candidate_set_path: str | Path,
    renderer_result_path: str | Path,
    output_relocation_receipt_path: str | Path,
    output_root: str | Path,
    repair_support_dilation_pixels: int = 2,
) -> dict[str, Any]:
    """Derive detectable-alpha and repair-support masks for one to five tasks."""

    if (
        isinstance(repair_support_dilation_pixels, bool)
        or not isinstance(repair_support_dilation_pixels, int)
        or not 0 <= repair_support_dilation_pixels <= 16
    ):
        raise BroadRepairSupportError(["broad_repair_dilation_invalid"])
    candidate_path = _file(candidate_set_path, code="broad_repair_candidate_missing")
    result_path = _file(renderer_result_path, code="broad_repair_renderer_result_missing")
    relocation_path = _file(
        output_relocation_receipt_path,
        code="broad_repair_relocation_missing",
    )
    candidate, task_ids, union = _validate_candidate(candidate_path)
    result = _validate_result(result_path, candidate=candidate, union=union)
    manifest_paths = _validate_relocation(
        relocation_path,
        result_path=result_path,
        result=result,
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise BroadRepairSupportError(["broad_repair_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    lanes: list[dict[str, Any]] = []
    kernel_size = 2 * repair_support_dilation_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    for task_id in sorted(task_ids):
        keys = {
            "black": (task_id, "shared_deleted_source_layer", "#000000"),
            "white": (task_id, "shared_deleted_source_layer", "#ffffff"),
            "retained": (task_id, "shared_retained_scene", "#000000"),
        }
        try:
            black_path, white_path, retained_path = (
                manifest_paths[keys["black"]],
                manifest_paths[keys["white"]],
                manifest_paths[keys["retained"]],
            )
        except KeyError as exc:
            raise BroadRepairSupportError(["broad_repair_task_render_triplet_missing"]) from exc
        black, black_record = _manifest_frames(
            black_path,
            task_id=task_id,
            layer="shared_deleted_source_layer",
            background="#000000",
            candidate_digest=str(candidate["receipt_digest"]),
            splat_digest=str(union["deleted_digest"]),
            splat_count=int(union["deleted_count"]),
        )
        white, white_record = _manifest_frames(
            white_path,
            task_id=task_id,
            layer="shared_deleted_source_layer",
            background="#ffffff",
            candidate_digest=str(candidate["receipt_digest"]),
            splat_digest=str(union["deleted_digest"]),
            splat_count=int(union["deleted_count"]),
        )
        retained, retained_record = _manifest_frames(
            retained_path,
            task_id=task_id,
            layer="shared_retained_scene",
            background="#000000",
            candidate_digest=str(candidate["receipt_digest"]),
            splat_digest=str(union["retained_digest"]),
            splat_count=int(union["retained_count"]),
        )
        if not (set(black) == set(white) == set(retained)):
            raise BroadRepairSupportError(["broad_repair_camera_set_mismatch"])
        lane_root = output / "tasks" / task_id
        frames: list[dict[str, Any]] = []
        alpha_stack: list[np.ndarray] = []
        for camera_id in black_record["camera_ids"]:
            black_rgb = np.asarray(Image.open(black[camera_id]).convert("RGB"))
            white_rgb = np.asarray(Image.open(white[camera_id]).convert("RGB"))
            retained_rgb = np.asarray(Image.open(retained[camera_id]).convert("RGB"))
            if black_rgb.shape != white_rgb.shape or black_rgb.shape != retained_rgb.shape:
                raise BroadRepairSupportError(["broad_repair_frame_shape_mismatch"])
            alpha = derive_alpha_from_background_pair(black_rgb, white_rgb)
            detectable = alpha > 0.0
            repair = cv2.dilate(detectable.astype(np.uint8), kernel, iterations=1).astype(bool)
            if not np.all(repair[detectable]):
                raise BroadRepairSupportError(["broad_repair_support_not_conservative"])
            alpha_stack.append(alpha.astype(np.float32))
            raw_path = lane_root / "detectable_deleted_support_masks" / f"{camera_id}.png"
            repair_path = lane_root / "repair_support_masks" / f"{camera_id}.png"
            overlay_path = lane_root / "review_overlays" / f"{camera_id}.png"
            _save_png(raw_path, detectable.astype(np.uint8) * 255)
            _save_png(repair_path, repair.astype(np.uint8) * 255)
            overlay = retained_rgb.copy()
            overlay[repair] = np.rint(
                0.35 * overlay[repair].astype(np.float32)
                + 0.65 * np.array([255.0, 0.0, 255.0], dtype=np.float32)
            ).astype(np.uint8)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay, mode="RGB").save(overlay_path)
            support_alpha = alpha[detectable]
            frames.append(
                {
                    "camera_id": camera_id,
                    "deleted_black_frame": _record(black[camera_id]),
                    "deleted_white_frame": _record(white[camera_id]),
                    "retained_scene_frame": _record(retained[camera_id]),
                    "detectable_deleted_support_mask": _record(raw_path, root=output),
                    "repair_support_mask": _record(repair_path, root=output),
                    "review_overlay": _record(overlay_path, root=output),
                    "detectable_alpha_pixel_count": int(detectable.sum()),
                    "repair_support_pixel_count": int(repair.sum()),
                    "repair_support_fraction": float(repair.mean()),
                    "detectable_support_bbox": _bbox(detectable),
                    "repair_support_bbox": _bbox(repair),
                    "maximum_detectable_alpha": (
                        float(support_alpha.max()) if support_alpha.size else 0.0
                    ),
                }
            )
        alpha_path = lane_root / "deleted_source_alpha_by_camera.npy"
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(alpha_path, np.stack(alpha_stack).astype(np.float32), allow_pickle=False)
        lanes.append(
            {
                "task_id": task_id,
                "camera_count": len(frames),
                "camera_ids": list(black_record["camera_ids"]),
                "deleted_source_black_manifest": black_record,
                "deleted_source_white_manifest": white_record,
                "retained_scene_manifest": retained_record,
                "deleted_source_alpha_by_camera": _record(alpha_path, root=output),
                "frames": frames,
            }
        )
    if {lane["task_id"] for lane in lanes} != task_ids:
        raise BroadRepairSupportError(["broad_repair_materialized_task_set_mismatch"])
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA,
        "status": "full_deleted_projection_repair_support_materialized",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "candidate_set": {
            **_record(candidate_path),
            "schema_version": candidate["schema_version"],
            "receipt_digest": candidate["receipt_digest"],
        },
        "renderer_result": _record(result_path),
        "output_relocation_receipt": {
            **_record(relocation_path),
            "receipt_digest": _read(
                relocation_path,
                code="broad_repair_relocation_unreadable",
            )["receipt_digest"],
        },
        "shared_deleted_source_layer": {
            "sha256": union["deleted_digest"],
            "gaussian_count": union["deleted_count"],
        },
        "shared_retained_scene": {
            "sha256": union["retained_digest"],
            "gaussian_count": union["retained_count"],
        },
        "alpha_recovery_method": "paired_identical_rgb_over_black_and_white_median.v1",
        "detectable_support_rule": "derived_alpha_strictly_greater_than_zero_after_uint8_render",
        "repair_support_dilation_pixels": repair_support_dilation_pixels,
        "task_lanes": lanes,
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "claim_boundary": {
            "broad_cutout_is_truthful_cleaned_scene": False,
            "all_detectable_deleted_projection_inside_repair_support": True,
            "repair_support_authorizes_future_generated_background_only": True,
            "outside_repair_support_change_authorized": False,
            "outside_repair_support_invariance_measured": False,
            "multiview_repair_consistency_measured": False,
            "repaired_pixels_are_observed_physical_evidence": False,
            "learned_policy_or_simulator_output_is_physical_evidence": False,
        },
        "blockers": [
            "rights_admitted_multiview_repair_not_executed",
            "outside_repair_support_invariance_not_measured",
            "multiview_repair_consistency_not_measured",
            "repaired_views_not_distilled_to_3dgs",
        ],
        "receipt_digest": "",
    }
    packet["receipt_digest"] = canonical_digest(packet, digest_field="receipt_digest")
    (output / "public_scene_broad_repair_support_packet.v1.json").write_text(
        canonical_json(packet) + "\n",
        encoding="utf-8",
    )
    return packet


__all__ = [
    "BroadRepairSupportError",
    "PACKET_SCHEMA",
    "materialize_broad_repair_support",
]
