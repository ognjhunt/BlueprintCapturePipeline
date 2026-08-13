"""Seal paired-target ArtiFixer3D renders inside their declared support.

The representation learned by ArtiFixer3D may change the full rendered frame.
This module turns that raw representation review into a bounded appearance
candidate: the exact object mask uses the generated render, the already
preregistered transition halo blends generated and original pixels, and every
pixel outside that support is copied exactly from the original observation.

It performs no model execution and makes no physical, collision, or policy
claim.  The same interface accepts one to five co-present task objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


SCHEMA_VERSION = "public_scene_artifixer3d_final_composite.v1"
DUAL_INPUT_SCHEMA = "public_scene_artifixer3d_dual_target_inputs.v1"
RAW_RESULT_SCHEMA = "public_scene_artifixer3d_raw_result.v1"
TRANSITION_MORPHOLOGY = "euclidean_disk_inclusive_radius_constant_zero_border"


class ArtiFixer3DFinalCompositeError(ValueError):
    """Stable fail-closed error codes for final appearance composition."""


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
        raise ArtiFixer3DFinalCompositeError(code) from exc
    if not isinstance(value, dict):
        raise ArtiFixer3DFinalCompositeError(code)
    return value


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        record["path"] = str(path)
    else:
        record["relative_path"] = path.relative_to(root).as_posix()
    return record


def _bound_absolute(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise ArtiFixer3DFinalCompositeError(code)
    path = Path(str(value.get("path") or "")).expanduser()
    if path.is_symlink():
        raise ArtiFixer3DFinalCompositeError(code)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtiFixer3DFinalCompositeError(code) from exc
    if (
        not resolved.is_file()
        or resolved.stat().st_size != value.get("size_bytes")
        or _sha256(resolved) != value.get("sha256")
    ):
        raise ArtiFixer3DFinalCompositeError(code)
    return resolved


def _image(path: Path, *, mode: str, code: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert(mode), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ArtiFixer3DFinalCompositeError(code) from exc


def _alpha(exact: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact-core alpha plus a linear, declared exterior transition."""

    core = exact > 0
    if not np.any(core) or set(exact.tobytes()) - {0, 255}:
        raise ArtiFixer3DFinalCompositeError("artifixer3d_final_exact_mask_invalid")
    if radius == 0:
        return core.astype(np.float32), core
    distance = distance_transform_edt(~core)
    support = core | (distance <= radius)
    alpha = np.zeros(core.shape, dtype=np.float32)
    alpha[core] = 1.0
    transition = support & ~core
    # The outermost admitted pixel still receives a positive generated weight;
    # the first pixel outside support is copied exactly from the original.
    alpha[transition] = (radius + 1.0 - distance[transition]) / (radius + 1.0)
    return alpha, support


def materialize_artifixer3d_final_composite(
    *,
    dual_input_receipt_paths: Sequence[str | Path],
    raw_result_paths: Sequence[str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    """Compose and seal one to five task results with exact outside invariance."""

    if not dual_input_receipt_paths or len(dual_input_receipt_paths) != len(raw_result_paths):
        raise ArtiFixer3DFinalCompositeError("artifixer3d_final_input_pairing_invalid")
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArtiFixer3DFinalCompositeError("artifixer3d_final_output_not_empty")
    output.mkdir(parents=True, exist_ok=True)

    input_tasks: dict[str, tuple[dict[str, Any], dict[str, Any], Path, Path, int]] = {}
    input_records: list[dict[str, Any]] = []
    publisher_scene_id: str | None = None
    for dual_value, raw_value in zip(dual_input_receipt_paths, raw_result_paths, strict=True):
        dual_path = Path(dual_value).expanduser().resolve()
        raw_path = Path(raw_value).expanduser().resolve()
        dual = _read(dual_path, code="artifixer3d_final_dual_input_unreadable")
        raw = _read(raw_path, code="artifixer3d_final_raw_result_unreadable")
        if (
            dual.get("schema_version") != DUAL_INPUT_SCHEMA
            or dual.get("receipt_digest") != canonical_digest(dual, digest_field="receipt_digest")
            or dual.get("pipeline_mode") != "dual_target_artifixer3d_only"
            or dual.get("status") != "paired_target_inputs_prepared_no_model_no_execution"
            or raw.get("schema_version") != RAW_RESULT_SCHEMA
            or raw.get("result_digest") != canonical_digest(raw, digest_field="result_digest")
            or raw.get("pipeline_mode")
            not in {
                "dual_target_artifixer3d_only",
                "dual_target_artifixer3d_render_only",
            }
            or raw.get("appearance_repair_qualified") is not False
        ):
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_input_receipt_invalid")
        scene_id = str(dual.get("publisher_scene_id") or "")
        if publisher_scene_id is None:
            publisher_scene_id = scene_id
        if not scene_id or publisher_scene_id != scene_id:
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_scene_binding_invalid")
        dual_tasks = dual.get("tasks")
        raw_tasks = raw.get("tasks")
        transition = dual.get("transition_support")
        if not isinstance(dual_tasks, list) or not isinstance(raw_tasks, list):
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_task_inventory_invalid")
        if (
            not isinstance(transition, Mapping)
            or transition.get("morphology") != TRANSITION_MORPHOLOGY
            or isinstance(transition.get("radius_pixels"), bool)
            or not isinstance(transition.get("radius_pixels"), int)
            or not 0 <= transition["radius_pixels"] <= 64
        ):
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_transition_contract_invalid")
        raw_by_id = {str(row.get("task_id")): row for row in raw_tasks if isinstance(row, Mapping)}
        if len(raw_by_id) != len(raw_tasks):
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_task_inventory_invalid")
        for task in dual_tasks:
            if not isinstance(task, Mapping):
                raise ArtiFixer3DFinalCompositeError("artifixer3d_final_task_inventory_invalid")
            task_id = str(task.get("task_id") or "")
            raw_task = raw_by_id.get(task_id)
            if not task_id or raw_task is None or task_id in input_tasks:
                raise ArtiFixer3DFinalCompositeError("artifixer3d_final_task_inventory_invalid")
            input_tasks[task_id] = (
                dict(task),
                dict(raw_task),
                dual_path,
                raw_path,
                int(transition["radius_pixels"]),
            )
        input_records.append(
            {
                "dual_target_inputs": _file_record(dual_path),
                "dual_target_inputs_digest": dual["receipt_digest"],
                "raw_result": _file_record(raw_path),
                "raw_result_digest": raw["result_digest"],
            }
        )
    if not 1 <= len(input_tasks) <= MAX_REPLACEMENT_OBJECTS:
        raise ArtiFixer3DFinalCompositeError("artifixer3d_final_task_count_invalid")

    task_receipts: list[dict[str, Any]] = []
    for task_id, (task, raw_task, _dual_path, _raw_path, radius) in sorted(input_tasks.items()):
        frame_rows = task.get("frames")
        raw_frames = raw_task.get("artifixer3d_review_frames")
        if not isinstance(frame_rows, list) or not isinstance(raw_frames, list):
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_frame_inventory_invalid")
        raw_by_index = {
            row.get("frame_index"): row for row in raw_frames if isinstance(row, Mapping)
        }
        if (
            len(raw_by_index) != len(raw_frames)
            or len(frame_rows) != int(task.get("physical_camera_count") or -1)
            or len(raw_frames) != len(frame_rows)
        ):
            raise ArtiFixer3DFinalCompositeError("artifixer3d_final_frame_inventory_invalid")
        task_root = output / task_id
        frames_root = task_root / "frames"
        masks_root = task_root / "support_masks"
        frames_root.mkdir(parents=True)
        masks_root.mkdir()
        sealed_frames: list[dict[str, Any]] = []
        for frame in frame_rows:
            if not isinstance(frame, Mapping):
                raise ArtiFixer3DFinalCompositeError("artifixer3d_final_frame_inventory_invalid")
            index = frame.get("physical_camera_index")
            raw_frame = raw_by_index.get(index)
            if (
                not isinstance(index, int)
                or not isinstance(raw_frame, Mapping)
                or frame.get("camera_id") != raw_frame.get("camera_id")
            ):
                raise ArtiFixer3DFinalCompositeError("artifixer3d_final_camera_binding_invalid")
            original_path = _bound_absolute(
                frame.get("source_original_frame"),
                code="artifixer3d_final_original_frame_invalid",
            )
            exact_path = _bound_absolute(
                frame.get("source_exact_repair_mask"),
                code="artifixer3d_final_exact_mask_invalid",
            )
            generated_path = _bound_absolute(
                raw_frame,
                code="artifixer3d_final_generated_frame_invalid",
            )
            original = _image(
                original_path, mode="RGB", code="artifixer3d_final_original_frame_invalid"
            )
            generated = _image(
                generated_path,
                mode="RGB",
                code="artifixer3d_final_generated_frame_invalid",
            )
            exact = _image(exact_path, mode="L", code="artifixer3d_final_exact_mask_invalid")
            if original.shape != generated.shape or original.shape[:2] != exact.shape:
                raise ArtiFixer3DFinalCompositeError("artifixer3d_final_frame_shape_invalid")
            alpha, support = _alpha(exact, radius)
            blended = np.rint(
                generated.astype(np.float32) * alpha[:, :, None]
                + original.astype(np.float32) * (1.0 - alpha[:, :, None])
            ).astype(np.uint8)
            outside = ~support
            outside_changes = int(
                np.count_nonzero(np.any(blended[outside] != original[outside], axis=1))
            )
            if outside_changes != 0:
                raise ArtiFixer3DFinalCompositeError("artifixer3d_final_outside_support_changed")
            destination = frames_root / f"{index:05d}.png"
            support_path = masks_root / f"{index:05d}.png"
            Image.fromarray(blended, mode="RGB").save(destination)
            Image.fromarray(np.where(support, 255, 0).astype(np.uint8), mode="L").save(support_path)
            final_record = _file_record(destination, root=task_root)
            sealed_frames.append(
                {
                    "frame_index": index,
                    "camera_id": frame["camera_id"],
                    "path": str(destination),
                    "size_bytes": final_record["size_bytes"],
                    "sha256": final_record["sha256"],
                    "original_frame": _file_record(original_path),
                    "generated_raw_frame": _file_record(generated_path),
                    "exact_repair_mask": _file_record(exact_path),
                    "declared_support_mask": _file_record(support_path, root=task_root),
                    "final_frame": final_record,
                    "exact_repair_pixel_count": int(np.count_nonzero(exact)),
                    "transition_pixel_count": int(np.count_nonzero(support & ~(exact > 0))),
                    "outside_support_pixel_count": int(np.count_nonzero(outside)),
                    "outside_support_changed_pixels": outside_changes,
                }
            )
        task_receipts.append(
            {
                "task_id": task_id,
                "physical_camera_count": len(sealed_frames),
                "transition_radius_pixels": radius,
                "transition_morphology": TRANSITION_MORPHOLOGY,
                "frames": sorted(sealed_frames, key=lambda row: row["frame_index"]),
                "outside_support_changed_pixels_total": sum(
                    row["outside_support_changed_pixels"] for row in sealed_frames
                ),
                "outside_support_invariance_proven": True,
                "semantic_object_absence_review_passed": False,
                "multiview_consistency_review_passed": False,
                "appearance_repair_qualified": False,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "final_composite_materialized_pending_human_multiview_review",
        "publisher_scene_id": publisher_scene_id,
        "replacement_object_count": len(task_receipts),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "inputs": input_records,
        "tasks": task_receipts,
        "outside_support_changed_pixels_total": sum(
            task["outside_support_changed_pixels_total"] for task in task_receipts
        ),
        "outside_support_invariance_proven": True,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "appearance_repair_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "claim_boundary": (
            "bounded_generated_appearance_candidate_pending_human_multiview_review_"
            "not_capture_collision_policy_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{SCHEMA_VERSION}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-input", action="append", required=True)
    parser.add_argument("--raw-result", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = materialize_artifixer3d_final_composite(
        dual_input_receipt_paths=args.dual_input,
        raw_result_paths=args.raw_result,
        output_root=args.output_root,
    )
    print(canonical_json({"status": result["status"], "receipt_digest": result["receipt_digest"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ArtiFixer3DFinalCompositeError",
    "SCHEMA_VERSION",
    "materialize_artifixer3d_final_composite",
]
