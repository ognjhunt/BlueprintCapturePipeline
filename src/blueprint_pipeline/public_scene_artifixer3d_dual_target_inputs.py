"""Materialize paired-target ArtiFixer3D inputs without a pasted silhouette.

Each physical camera contributes two training records at exactly the same pose:

* an unchanged retained-scene render whose loss is disabled inside a declared
  repair support; and
* an unchanged whole-frame semantic-editor candidate with no loss mask.

The adapter performs no model execution.  Whole-frame editor outputs remain
unreviewed candidates, and the resulting packet is neither appearance evidence
nor physical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .public_scene_artifixer3d_candidate_inputs import (
    CAMERA_CONVENTION_FLIP,
    SCHEMA_VERSION as SOURCE_CANDIDATE_SCHEMA,
)
from .fresh_scene_semantic_teacher_image_edit import (
    PACKET_SCHEMA_VERSION as SEMANTIC_TEACHER_PACKET_SCHEMA,
)
from .semantic_teacher_image_edit_paid_lane import RESULT_IMPORT_SCHEMA_VERSION
from .semantic_teacher_image_edit_worker import (
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
)


SCHEMA_VERSION = "public_scene_artifixer3d_dual_target_inputs.v1"
SEMANTIC_TEACHER_SCHEMA = "public_scene_whole_frame_semantic_teacher_candidates.v1"
CAMERA_INDEX_SCHEMA = "public_scene_artifixer3d_dual_target_camera_index.v1"
TRANSITION_MORPHOLOGY = "euclidean_disk_inclusive_radius_constant_zero_border"
HANDOFF_SCHEMA_VERSION = "semantic_teacher_artifixer_handoff.v1"


class DualTargetInputError(ValueError):
    """Stable failure codes for paired-target input materialization."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(value: Any, *, code: str) -> Path:
    unresolved = Path(str(value or "")).expanduser()
    if unresolved.is_symlink():
        raise DualTargetInputError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise DualTargetInputError([code])
    return path


def _directory(value: Any, *, code: str) -> Path:
    unresolved = Path(str(value or "")).expanduser()
    if unresolved.is_symlink():
        raise DualTargetInputError([code])
    path = unresolved.resolve()
    if not path.is_dir():
        raise DualTargetInputError([code])
    return path


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DualTargetInputError([code]) from exc
    if not isinstance(value, dict):
        raise DualTargetInputError([code])
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _absolute_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _relative_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_record(
    value: Any,
    *,
    code: str,
    root: Path | None = None,
) -> Path:
    if not isinstance(value, Mapping):
        raise DualTargetInputError([code])
    if root is None:
        path_value = value.get("path")
    else:
        relative = value.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise DualTargetInputError([code])
        unresolved = Path(relative)
        if unresolved.is_absolute() or ".." in unresolved.parts:
            raise DualTargetInputError([code])
        path_value = root / unresolved
    path = _file(path_value, code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get("sha256"):
        raise DualTargetInputError([code])
    return path


def _image(path: Path, *, mode: str, code: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert(mode), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise DualTargetInputError([code]) from exc


def _link_or_copy(source: Path, destination: Path, *, code: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise DualTargetInputError([code])
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    if destination.stat().st_size != source.stat().st_size or _sha256(destination) != _sha256(
        source
    ):
        raise DualTargetInputError([code])


def _validated_source(path: Path) -> dict[str, Any]:
    source = _read(path, code="dual_target_source_receipt_unreadable")
    tasks = source.get("tasks")
    count = source.get("replacement_object_count")
    if (
        source.get("schema_version") != SOURCE_CANDIDATE_SCHEMA
        or source.get("status") != "candidate_inputs_prepared_no_model_no_execution"
        or source.get("receipt_digest") != canonical_digest(source, digest_field="receipt_digest")
        or not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= MAX_REPLACEMENT_OBJECTS
        or not isinstance(tasks, list)
        or len(tasks) != count
    ):
        raise DualTargetInputError(["dual_target_source_receipt_invalid"])
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise DualTargetInputError(["dual_target_source_receipt_invalid"])
    return source


def _task_root(task: Mapping[str, Any]) -> Path:
    return _directory(task.get("scene_directory"), code="dual_target_source_task_root_invalid")


def _source_task_frames(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_root = _task_root(task)
    frames = task.get("frames")
    if not isinstance(frames, list) or not frames or task.get("camera_count") != len(frames):
        raise DualTargetInputError(["dual_target_source_camera_set_invalid"])
    normalized: list[dict[str, Any]] = []
    camera_ids: list[str] = []
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise DualTargetInputError(["dual_target_source_camera_set_invalid"])
        camera_id = str(frame.get("camera_id") or "")
        if frame.get("frame_index") != expected_index or not camera_id:
            raise DualTargetInputError(["dual_target_source_camera_set_invalid"])
        original = _bound_record(
            frame.get("rendered_rgb"),
            root=task_root,
            code="dual_target_source_original_invalid",
        )
        exact_mask = _bound_record(
            frame.get("exact_repair_mask"),
            root=task_root,
            code="dual_target_source_mask_invalid",
        )
        rgb = _image(original, mode="RGB", code="dual_target_source_original_invalid")
        mask = _image(exact_mask, mode="L", code="dual_target_source_mask_invalid")
        if (
            rgb.shape[:2] != mask.shape
            or set(mask.tobytes()) - {0, 255}
            or not np.any(mask)
            or frame.get("repair_pixel_count") != int(np.count_nonzero(mask))
        ):
            raise DualTargetInputError(["dual_target_source_shape_or_mask_invalid"])
        normalized.append(
            {
                "frame_index": expected_index,
                "camera_id": camera_id,
                "original_path": original,
                "mask_path": exact_mask,
                "rgb": rgb,
                "mask": mask,
            }
        )
        camera_ids.append(camera_id)
    if len(camera_ids) != len(set(camera_ids)):
        raise DualTargetInputError(["dual_target_source_camera_set_invalid"])
    return normalized


def _validated_transforms(
    task: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], Path]:
    path = _bound_record(task.get("transforms"), code="dual_target_source_transforms_invalid")
    transforms = _read(path, code="dual_target_source_transforms_invalid")
    rows = transforms.get("frames")
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise DualTargetInputError(["dual_target_source_transforms_invalid"])
    required_intrinsics = ("w", "h", "fl_x", "fl_y", "cx", "cy")
    for index, (row, frame) in enumerate(zip(rows, frames)):
        matrix = row.get("transform_matrix") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("camera_id") != frame["camera_id"]
            or not isinstance(matrix, list)
            or len(matrix) != 4
            or any(not isinstance(line, list) or len(line) != 4 for line in matrix)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for line in matrix
                for value in line
            )
            or any(
                not isinstance(row.get(field), (int, float))
                or isinstance(row.get(field), bool)
                or not math.isfinite(float(row[field]))
                for field in required_intrinsics
            )
            or int(row["w"]) != frame["rgb"].shape[1]
            or int(row["h"]) != frame["rgb"].shape[0]
            or row.get("file_path") != f"images/{index:05d}.png"
        ):
            raise DualTargetInputError(["dual_target_source_transforms_invalid"])
    return transforms, path


def _validated_semantic_teacher(
    path: Path, *, source_path: Path, source: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _read(path, code="dual_target_semantic_teacher_receipt_unreadable")
    source_record = receipt.get("source_candidate_inputs_receipt")
    frames = receipt.get("frames")
    if not isinstance(source_record, Mapping):
        raise DualTargetInputError(["dual_target_semantic_teacher_receipt_invalid"])
    bound_source = _bound_record(source_record, code="dual_target_semantic_teacher_receipt_invalid")
    if (
        receipt.get("schema_version") != SEMANTIC_TEACHER_SCHEMA
        or receipt.get("status") != "whole_frame_semantic_teacher_candidates_unreviewed"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or source_record.get("receipt_digest") != source.get("receipt_digest")
        or source_record.get("sha256") != _sha256(source_path)
        or source_record.get("size_bytes") != source_path.stat().st_size
        or _sha256(bound_source) != _sha256(source_path)
        or not isinstance(receipt.get("editor_identity"), Mapping)
        or not receipt.get("editor_identity")
        or not str(receipt.get("prompt_policy") or "").strip()
        or receipt.get("semantic_object_absence_review_passed") is not False
        or receipt.get("multiview_consistency_review_passed") is not False
        or not isinstance(frames, list)
        or not frames
        or receipt.get("frame_count") != len(frames)
    ):
        raise DualTargetInputError(["dual_target_semantic_teacher_receipt_invalid"])
    return {**receipt, "receipt_path": path}


def _teacher_frame_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for frame in receipt.get("frames") or []:
        if not isinstance(frame, Mapping):
            raise DualTargetInputError(["dual_target_semantic_teacher_frame_invalid"])
        camera_id = str(frame.get("camera_id") or "")
        if not camera_id or camera_id in result:
            raise DualTargetInputError(["dual_target_semantic_teacher_frame_invalid"])
        for field in (
            "source_original_frame",
            "exact_repair_mask",
            "whole_frame_semantic_teacher",
        ):
            _bound_record(frame.get(field), code="dual_target_semantic_teacher_frame_invalid")
        result[camera_id] = dict(frame)
    return result


def _disk(radius: int) -> np.ndarray:
    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def _rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = (
        np.asarray(
            [
                [rxx - ryy - rzz, 0.0, 0.0, 0.0],
                [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
                [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
                [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
            ],
            dtype=np.float64,
        )
        / 3.0
    )
    values, vectors = np.linalg.eigh(matrix)
    qvec = vectors[[3, 0, 1, 2], int(np.argmax(values))]
    return -qvec if qvec[0] < 0 else qvec


def _write_colmap_calibration(sparse: Path, transform_rows: Sequence[Mapping[str, Any]]) -> None:
    with (
        (sparse / "cameras.bin").open("wb") as cameras,
        (sparse / "images.bin").open("wb") as images,
    ):
        cameras.write(struct.pack("<Q", len(transform_rows)))
        images.write(struct.pack("<Q", len(transform_rows)))
        for image_id, row in enumerate(transform_rows, start=1):
            cameras.write(
                struct.pack(
                    "<iiQQdddddddd",
                    image_id,
                    4,
                    int(row["w"]),
                    int(row["h"]),
                    float(row["fl_x"]),
                    float(row["fl_y"]),
                    float(row["cx"]),
                    float(row["cy"]),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
            camera_to_world_opengl = np.asarray(row["transform_matrix"], dtype=np.float64)
            camera_to_world_opencv = camera_to_world_opengl @ CAMERA_CONVENTION_FLIP
            world_to_camera = np.linalg.inv(camera_to_world_opencv)
            qvec = _rotmat_to_qvec(world_to_camera[:3, :3])
            images.write(
                struct.pack(
                    "<idddddddi",
                    image_id,
                    *qvec,
                    *world_to_camera[:3, 3],
                    image_id,
                )
            )
            images.write(f"{image_id - 1:05d}.png".encode("utf-8") + b"\x00")
            images.write(struct.pack("<Q", 0))


def materialize_whole_frame_semantic_teacher_receipt(
    *,
    source_candidate_inputs_receipt_path: str | Path,
    task_id: str,
    semantic_teacher_frames_root: str | Path,
    editor_identity: Mapping[str, Any],
    prompt_policy: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind full semantic-editor frames without claiming review or locality."""

    source_path = _file(
        source_candidate_inputs_receipt_path,
        code="dual_target_source_receipt_missing",
    )
    source = _validated_source(source_path)
    task_id = str(task_id or "")
    task_matches = [task for task in source["tasks"] if task.get("task_id") == task_id]
    if (
        len(task_matches) != 1
        or not isinstance(editor_identity, Mapping)
        or not editor_identity
        or not isinstance(prompt_policy, str)
        or not prompt_policy.strip()
    ):
        raise DualTargetInputError(["dual_target_semantic_teacher_request_invalid"])
    teacher_root = _directory(
        semantic_teacher_frames_root,
        code="dual_target_semantic_teacher_root_invalid",
    )
    output = Path(output_path).expanduser()
    if output.is_symlink() or output.exists():
        raise DualTargetInputError(["dual_target_semantic_teacher_output_invalid"])
    output = output.resolve()

    task = task_matches[0]
    source_frames = _source_task_frames(task)
    rows: list[dict[str, Any]] = []
    for frame in source_frames:
        index = int(frame["frame_index"])
        teacher_path = _file(
            teacher_root / f"{index:05d}.png",
            code="dual_target_semantic_teacher_frame_missing",
        )
        teacher = _image(
            teacher_path,
            mode="RGB",
            code="dual_target_semantic_teacher_frame_invalid",
        )
        if teacher.shape != frame["rgb"].shape:
            raise DualTargetInputError(["dual_target_semantic_teacher_shape_invalid"])
        support = frame["mask"] > 0
        changed = np.any(teacher != frame["rgb"], axis=2)
        rows.append(
            {
                "frame_index": index,
                "camera_id": frame["camera_id"],
                "source_original_frame": _absolute_record(frame["original_path"]),
                "exact_repair_mask": _absolute_record(frame["mask_path"]),
                "whole_frame_semantic_teacher": _absolute_record(teacher_path),
                "width": int(teacher.shape[1]),
                "height": int(teacher.shape[0]),
                "exact_repair_pixel_count": int(np.count_nonzero(support)),
                "inside_exact_support_changed_pixels": int(np.count_nonzero(changed & support)),
                "outside_exact_support_changed_pixels": int(np.count_nonzero(changed & ~support)),
                "whole_frame_candidate_preserved_without_compositing": True,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SEMANTIC_TEACHER_SCHEMA,
        "status": "whole_frame_semantic_teacher_candidates_unreviewed",
        "task_id": task_id,
        "source_candidate_inputs_receipt": {
            **_absolute_record(source_path),
            "receipt_digest": source["receipt_digest"],
        },
        "editor_identity": dict(editor_identity),
        "prompt_policy": prompt_policy.strip(),
        "frame_count": len(rows),
        "frames": rows,
        "inside_exact_support_changed_pixels_total": sum(
            row["inside_exact_support_changed_pixels"] for row in rows
        ),
        "outside_exact_support_changed_pixels_total": sum(
            row["outside_exact_support_changed_pixels"] for row in rows
        ),
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "appearance_repair_qualified": False,
        "physical_or_deployment_evidence": False,
        "claim_boundary": (
            "unreviewed_whole_frame_semantic_editor_candidates_not_capture_"
            "appearance_qualification_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, receipt)
    return receipt


def _ordered_task_camera_rows(
    value: Mapping[str, Any], *, code: str
) -> list[tuple[str, list[str]]]:
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise DualTargetInputError([code])
    rows: list[tuple[str, list[str]]] = []
    task_ids: set[str] = set()
    for task in tasks:
        frames = task.get("frames") if isinstance(task, Mapping) else None
        task_id = str(task.get("task_id") or "") if isinstance(task, Mapping) else ""
        if (
            not task_id
            or task_id in task_ids
            or not isinstance(frames, list)
            or not frames
            or task.get("camera_count") != len(frames)
        ):
            raise DualTargetInputError([code])
        task_ids.add(task_id)
        camera_ids: list[str] = []
        for expected_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise DualTargetInputError([code])
            camera_id = str(frame.get("camera_id") or "")
            if (
                frame.get("frame_index") != expected_index
                or not camera_id
                or camera_id in camera_ids
            ):
                raise DualTargetInputError([code])
            camera_ids.append(camera_id)
        rows.append((task_id, camera_ids))
    return rows


def _semantic_result_handoff_inputs(
    *,
    result_import_path: str | Path,
    semantic_teacher_packet_path: str | Path,
    source_candidate_inputs_receipt_path: str | Path,
    transition_radius_pixels: int,
) -> dict[str, Any]:
    """Reopen every semantic result byte before any handoff output is written."""

    if (
        not isinstance(transition_radius_pixels, int)
        or isinstance(transition_radius_pixels, bool)
        or transition_radius_pixels < 0
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_transition_radius_invalid"])
    import_path = _file(result_import_path, code="semantic_teacher_handoff_result_import_missing")
    packet_path = _file(
        semantic_teacher_packet_path, code="semantic_teacher_handoff_packet_missing"
    )
    source_path = _file(
        source_candidate_inputs_receipt_path,
        code="semantic_teacher_handoff_source_candidate_missing",
    )
    imported = _read(import_path, code="semantic_teacher_handoff_result_import_invalid")
    packet = _read(packet_path, code="semantic_teacher_handoff_packet_invalid")
    source = _validated_source(source_path)
    if (
        imported.get("schema_version") != RESULT_IMPORT_SCHEMA_VERSION
        or imported.get("status") != "retained_unreviewed_semantic_teacher_candidates"
        or imported.get("result_import_digest")
        != canonical_digest(imported, digest_field="result_import_digest")
        or imported.get("all_generated_teacher_pngs_retained") is not True
        or imported.get("continuing_spend_from_this_run") is not False
        or imported.get("visual_reviewed") is not False
        or imported.get("appearance_qualified") is not False
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_result_import_invalid"])
    if (
        packet.get("schema_version") != SEMANTIC_TEACHER_PACKET_SCHEMA
        or packet.get("status")
        != "semantic_teacher_image_edit_packet_prepared_no_upload_no_execution"
        or packet.get("packet_digest") != canonical_digest(packet, digest_field="packet_digest")
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_packet_invalid"])
    source_record = packet.get("source_candidate_inputs_receipt")
    if not isinstance(source_record, Mapping):
        raise DualTargetInputError(["semantic_teacher_handoff_source_binding_invalid"])
    bound_source = _bound_record(
        source_record, code="semantic_teacher_handoff_source_binding_invalid"
    )
    if (
        bound_source != source_path
        or source_record.get("receipt_digest") != source.get("receipt_digest")
        or source_record.get("sha256") != _sha256(source_path)
        or source_record.get("size_bytes") != source_path.stat().st_size
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_source_binding_invalid"])

    runtime_request_path = _bound_record(
        imported.get("runtime_request"),
        code="semantic_teacher_handoff_runtime_request_invalid",
    )
    runtime_request = _read(
        runtime_request_path, code="semantic_teacher_handoff_runtime_request_invalid"
    )
    runtime_result_path = _bound_record(
        imported.get("runtime_result"),
        code="semantic_teacher_handoff_runtime_result_invalid",
    )
    runtime_result = _read(
        runtime_result_path, code="semantic_teacher_handoff_runtime_result_invalid"
    )
    backend = runtime_request.get("backend")
    execution = backend.get("execution") if isinstance(backend, Mapping) else None
    registry_entry = backend.get("registry_entry") if isinstance(backend, Mapping) else None
    if (
        runtime_request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or runtime_request.get("request_digest")
        != canonical_digest(runtime_request, digest_field="request_digest")
        or runtime_request.get("source_packet_digest") != packet.get("packet_digest")
        or not isinstance(backend, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(registry_entry, Mapping)
        or backend.get("backend_entry_digest") != canonical_digest(registry_entry)
        or not str(runtime_request.get("prompt_policy") or "").strip()
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_runtime_request_invalid"])
    if (
        runtime_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or runtime_result.get("status") != "completed_unreviewed_semantic_teacher_candidates"
        or runtime_result.get("result_digest")
        != canonical_digest(runtime_result, digest_field="result_digest")
        or runtime_result.get("source_runtime_request_digest")
        != runtime_request.get("request_digest")
        or runtime_result.get("backend_entry_digest") != backend.get("backend_entry_digest")
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_runtime_result_invalid"])

    order = _ordered_task_camera_rows(
        runtime_request, code="semantic_teacher_handoff_task_camera_order_invalid"
    )
    if (
        _ordered_task_camera_rows(packet, code="semantic_teacher_handoff_task_camera_order_invalid")
        != order
        or _ordered_task_camera_rows(
            runtime_result, code="semantic_teacher_handoff_task_camera_order_invalid"
        )
        != order
        or _ordered_task_camera_rows(
            source, code="semantic_teacher_handoff_task_camera_order_invalid"
        )
        != order
        or packet.get("task_count") != len(order)
        or imported.get("task_count") != len(order)
        or runtime_result.get("task_count") != len(order)
        or packet.get("request_count") != sum(len(cameras) for _, cameras in order)
        or imported.get("camera_count") != packet.get("request_count")
        or runtime_result.get("request_count") != packet.get("request_count")
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_task_camera_order_invalid"])

    runtime_root = runtime_result_path.parent
    retained_records = imported.get("teacher_frames")
    if not isinstance(retained_records, list):
        raise DualTargetInputError(["semantic_teacher_handoff_frame_inventory_invalid"])
    retained = {
        str(record.get("relative_path") or ""): record
        for record in retained_records
        if isinstance(record, Mapping)
    }
    expected: dict[str, Mapping[str, Any]] = {}
    result_tasks = runtime_result.get("tasks") or []
    source_tasks = {str(task["task_id"]): task for task in source["tasks"]}
    _bound_record(
        source.get("shared_retained_scene"),
        code="semantic_teacher_handoff_source_candidate_invalid",
    )
    _bound_record(
        source.get("shared_colmap_initialization_points3D"),
        code="semantic_teacher_handoff_source_candidate_invalid",
    )
    for task_row in result_tasks:
        task_id = str(task_row["task_id"])
        source_frames = _source_task_frames(source_tasks[task_id])
        _validated_transforms(source_tasks[task_id], source_frames)
        for frame, source_frame in zip(task_row["frames"], source_frames):
            record = frame.get("semantic_teacher_frame")
            if not isinstance(record, Mapping):
                raise DualTargetInputError(["semantic_teacher_handoff_frame_inventory_invalid"])
            relative = str(record.get("relative_path") or "")
            if relative != f"tasks/{task_id}/{frame['frame_index']:05d}.png":
                raise DualTargetInputError(["semantic_teacher_handoff_frame_inventory_invalid"])
            teacher_path = _bound_record(
                record,
                root=runtime_root,
                code="semantic_teacher_handoff_frame_inventory_invalid",
            )
            teacher = _image(
                teacher_path,
                mode="RGB",
                code="semantic_teacher_handoff_frame_inventory_invalid",
            )
            if teacher.shape != source_frame["rgb"].shape:
                raise DualTargetInputError(["semantic_teacher_handoff_frame_inventory_invalid"])
            expected[relative] = record
    if (
        len(retained) != len(retained_records)
        or set(retained) != set(expected)
        or any(
            retained[path].get("size_bytes") != expected[path].get("size_bytes")
            or retained[path].get("sha256") != expected[path].get("sha256")
            or _bound_record(
                retained[path],
                root=runtime_root,
                code="semantic_teacher_handoff_frame_inventory_invalid",
            )
            != runtime_root / path
            for path in expected
        )
    ):
        raise DualTargetInputError(["semantic_teacher_handoff_frame_inventory_invalid"])

    editor_identity = {
        "backend_id": str(registry_entry.get("backend_id") or ""),
        "backend_entry_digest": str(backend.get("backend_entry_digest") or ""),
        "adapter_id": str(execution.get("adapter_id") or ""),
        "model_snapshot": str(execution.get("model_snapshot") or ""),
        "source_runtime_request_digest": str(runtime_request["request_digest"]),
    }
    if any(not value for value in editor_identity.values()):
        raise DualTargetInputError(["semantic_teacher_handoff_editor_identity_invalid"])
    return {
        "import_path": import_path,
        "imported": imported,
        "packet_path": packet_path,
        "packet": packet,
        "source_path": source_path,
        "source": source,
        "runtime_request": runtime_request,
        "runtime_result_path": runtime_result_path,
        "runtime_result": runtime_result,
        "runtime_root": runtime_root,
        "order": order,
        "editor_identity": editor_identity,
    }


def materialize_semantic_teacher_artifixer_handoff(
    *,
    result_import_path: str | Path,
    semantic_teacher_packet_path: str | Path,
    source_candidate_inputs_receipt_path: str | Path,
    transition_radius_pixels: int,
    output_root: str | Path,
) -> dict[str, Any]:
    """Convert one sealed paid result into exact no-spend ArtiFixer inputs."""

    validated = _semantic_result_handoff_inputs(
        result_import_path=result_import_path,
        semantic_teacher_packet_path=semantic_teacher_packet_path,
        source_candidate_inputs_receipt_path=source_candidate_inputs_receipt_path,
        transition_radius_pixels=transition_radius_pixels,
    )
    unresolved = Path(output_root).expanduser()
    if unresolved.is_symlink():
        raise DualTargetInputError(["semantic_teacher_handoff_output_not_empty"])
    output = unresolved.resolve()
    if output.exists() and any(output.iterdir()):
        raise DualTargetInputError(["semantic_teacher_handoff_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    receipt_root = output / "semantic_teacher_receipts"
    receipt_root.mkdir()
    semantic_paths: list[Path] = []
    semantic_rows: list[dict[str, Any]] = []
    for task_id, _camera_ids in validated["order"]:
        path = receipt_root / f"{task_id}.json"
        receipt = materialize_whole_frame_semantic_teacher_receipt(
            source_candidate_inputs_receipt_path=validated["source_path"],
            task_id=task_id,
            semantic_teacher_frames_root=validated["runtime_root"] / "tasks" / task_id,
            editor_identity=validated["editor_identity"],
            prompt_policy=str(validated["runtime_request"]["prompt_policy"]),
            output_path=path,
        )
        semantic_paths.append(path)
        semantic_rows.append(
            {
                "task_id": task_id,
                **_relative_record(path, root=output),
                "receipt_digest": receipt["receipt_digest"],
            }
        )
    dual_root = output / "dual_target_inputs"
    dual = materialize_dual_target_artifixer3d_inputs(
        source_candidate_inputs_receipt_path=validated["source_path"],
        semantic_teacher_receipt_paths=semantic_paths,
        output_root=dual_root,
        transition_radius_pixels=transition_radius_pixels,
        selected_task_ids=[task_id for task_id, _ in validated["order"]],
    )
    receipt: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "status": "semantic_teacher_artifixer_handoff_materialized_no_execution",
        "result_import": {
            **_absolute_record(validated["import_path"]),
            "result_import_digest": validated["imported"]["result_import_digest"],
        },
        "semantic_teacher_packet": {
            **_absolute_record(validated["packet_path"]),
            "packet_digest": validated["packet"]["packet_digest"],
        },
        "source_candidate_inputs_receipt": {
            **_absolute_record(validated["source_path"]),
            "receipt_digest": validated["source"]["receipt_digest"],
        },
        "runtime_request_digest": validated["runtime_request"]["request_digest"],
        "runtime_result_digest": validated["runtime_result"]["result_digest"],
        "editor_identity": validated["editor_identity"],
        "prompt_policy": validated["runtime_request"]["prompt_policy"],
        "transition_radius_pixels": transition_radius_pixels,
        "task_count": len(semantic_rows),
        "camera_count": sum(len(cameras) for _, cameras in validated["order"]),
        "semantic_teacher_receipts": semantic_rows,
        "dual_target_inputs": {
            **_relative_record(dual_root / f"{SCHEMA_VERSION}.json", root=output),
            "receipt_digest": dual["receipt_digest"],
        },
        "paid_execution_started": False,
        "provider_mutations_performed": 0,
        "visual_reviewed": False,
        "appearance_qualified": False,
        "physical_evidence_claimed": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(output / f"{HANDOFF_SCHEMA_VERSION}.json", receipt)
    return receipt


def _materialize_semantic_teacher_artifixer_handoff_from_tool_request(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Execute one already-sealed supervisor request without model-authored paths."""

    return materialize_semantic_teacher_artifixer_handoff(
        result_import_path=str(request.get("result_import_path") or ""),
        semantic_teacher_packet_path=str(request.get("semantic_teacher_packet_path") or ""),
        source_candidate_inputs_receipt_path=str(
            request.get("source_candidate_inputs_receipt_path") or ""
        ),
        transition_radius_pixels=request.get("transition_radius_pixels"),
        output_root=output_root,
    )


def materialize_dual_target_artifixer3d_inputs(
    *,
    source_candidate_inputs_receipt_path: str | Path,
    semantic_teacher_receipt_paths: Sequence[str | Path],
    output_root: str | Path,
    transition_radius_pixels: int,
    selected_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a self-contained paired-target, ArtiFixer3D-only packet."""

    source_path = _file(
        source_candidate_inputs_receipt_path,
        code="dual_target_source_receipt_missing",
    )
    source = _validated_source(source_path)
    if (
        not isinstance(transition_radius_pixels, int)
        or isinstance(transition_radius_pixels, bool)
        or transition_radius_pixels < 0
    ):
        raise DualTargetInputError(["dual_target_transition_radius_invalid"])

    semantic_receipts: dict[str, dict[str, Any]] = {}
    for unresolved in semantic_teacher_receipt_paths:
        path = _file(unresolved, code="dual_target_semantic_teacher_receipt_missing")
        receipt = _validated_semantic_teacher(path, source_path=source_path, source=source)
        task_id = str(receipt.get("task_id") or "")
        if not task_id or task_id in semantic_receipts:
            raise DualTargetInputError(["dual_target_semantic_teacher_task_set_invalid"])
        semantic_receipts[task_id] = receipt
    source_tasks = {str(task["task_id"]): task for task in source["tasks"]}
    if selected_task_ids is None:
        task_ids = sorted(semantic_receipts)
    else:
        task_ids = sorted(str(task_id) for task_id in selected_task_ids)
    if (
        not 1 <= len(task_ids) <= MAX_REPLACEMENT_OBJECTS
        or len(task_ids) != len(set(task_ids))
        or any(not task_id for task_id in task_ids)
        or set(task_ids) != set(semantic_receipts)
        or any(task_id not in source_tasks for task_id in task_ids)
    ):
        raise DualTargetInputError(["dual_target_semantic_teacher_task_set_invalid"])

    output_unresolved = Path(output_root).expanduser()
    if output_unresolved.is_symlink():
        raise DualTargetInputError(["dual_target_output_not_empty"])
    output = output_unresolved.resolve()
    if output.exists() and any(output.iterdir()):
        raise DualTargetInputError(["dual_target_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)

    provenance = output / "provenance"
    shared = output / "shared_initialization"
    provenance.mkdir()
    shared.mkdir()
    copied_source_receipt = provenance / "source_candidate_inputs_receipt.json"
    _link_or_copy(
        source_path,
        copied_source_receipt,
        code="dual_target_source_receipt_copy_invalid",
    )

    source_retained = _bound_record(
        source.get("shared_retained_scene"),
        code="dual_target_source_retained_scene_invalid",
    )
    source_points = _bound_record(
        source.get("shared_colmap_initialization_points3D"),
        code="dual_target_source_seed_invalid",
    )
    retained_name = source_retained.name
    copied_retained = shared / retained_name
    copied_points = shared / "points3D.bin"
    _link_or_copy(
        source_retained,
        copied_retained,
        code="dual_target_source_retained_scene_copy_invalid",
    )
    _link_or_copy(
        source_points,
        copied_points,
        code="dual_target_source_seed_copy_invalid",
    )

    structure = _disk(transition_radius_pixels)
    task_receipts: list[dict[str, Any]] = []
    copied_teacher_receipts: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = source_tasks[task_id]
        source_frames = _source_task_frames(task)
        source_transforms, _source_transforms_path = _validated_transforms(task, source_frames)
        teacher_receipt = semantic_receipts[task_id]
        teacher_by_camera = _teacher_frame_map(teacher_receipt)
        if set(teacher_by_camera) != {str(frame["camera_id"]) for frame in source_frames}:
            raise DualTargetInputError(["dual_target_semantic_teacher_camera_set_invalid"])
        copied_teacher_receipt = provenance / f"semantic_teacher.{task_id}.json"
        _link_or_copy(
            teacher_receipt["receipt_path"],
            copied_teacher_receipt,
            code="dual_target_semantic_teacher_receipt_copy_invalid",
        )
        copied_teacher_receipts.append(
            {
                "task_id": task_id,
                **_relative_record(copied_teacher_receipt, root=output),
                "receipt_digest": teacher_receipt["receipt_digest"],
            }
        )

        task_root = output / task_id
        images = task_root / "images"
        teacher_images = task_root / "semantic_teacher_frames"
        anchor_masks = task_root / "anchor_loss_masks"
        exact_masks = task_root / "exact_masks"
        sparse = task_root / "sparse" / "0"
        for directory in (
            images,
            teacher_images,
            anchor_masks,
            exact_masks,
            sparse,
        ):
            directory.mkdir(parents=True)

        transform_rows: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        anchor_indices: list[int] = []
        teacher_indices: list[int] = []
        for physical_index, (source_frame, transform) in enumerate(
            zip(source_frames, source_transforms["frames"])
        ):
            teacher_frame = teacher_by_camera[str(source_frame["camera_id"])]
            if teacher_frame.get("frame_index") != physical_index:
                raise DualTargetInputError(["dual_target_semantic_teacher_camera_set_invalid"])
            teacher_source = _bound_record(
                teacher_frame.get("whole_frame_semantic_teacher"),
                code="dual_target_semantic_teacher_frame_invalid",
            )
            teacher_original = _bound_record(
                teacher_frame.get("source_original_frame"),
                code="dual_target_semantic_teacher_frame_invalid",
            )
            teacher_mask = _bound_record(
                teacher_frame.get("exact_repair_mask"),
                code="dual_target_semantic_teacher_frame_invalid",
            )
            teacher_pixels = _image(
                teacher_source,
                mode="RGB",
                code="dual_target_semantic_teacher_frame_invalid",
            )
            if (
                _sha256(teacher_original) != _sha256(source_frame["original_path"])
                or _sha256(teacher_mask) != _sha256(source_frame["mask_path"])
                or teacher_pixels.shape != source_frame["rgb"].shape
                or teacher_frame.get("width") != teacher_pixels.shape[1]
                or teacher_frame.get("height") != teacher_pixels.shape[0]
            ):
                raise DualTargetInputError(["dual_target_semantic_teacher_binding_invalid"])
            exact_support = source_frame["mask"] > 0
            teacher_changed = np.any(teacher_pixels != source_frame["rgb"], axis=2)
            if (
                teacher_frame.get("exact_repair_pixel_count")
                != int(np.count_nonzero(exact_support))
                or teacher_frame.get("inside_exact_support_changed_pixels")
                != int(np.count_nonzero(teacher_changed & exact_support))
                or teacher_frame.get("outside_exact_support_changed_pixels")
                != int(np.count_nonzero(teacher_changed & ~exact_support))
            ):
                raise DualTargetInputError(["dual_target_semantic_teacher_binding_invalid"])

            anchor_index = 2 * physical_index
            teacher_index = anchor_index + 1
            anchor_indices.append(anchor_index)
            teacher_indices.append(teacher_index)
            anchor_path = images / f"{anchor_index:05d}.png"
            teacher_path = images / f"{teacher_index:05d}.png"
            teacher_override_path = teacher_images / f"{teacher_index:05d}.png"
            exact_mask_path = exact_masks / f"{physical_index:05d}.png"
            anchor_mask_path = anchor_masks / f"{anchor_index:05d}.png"
            anchor_sibling_mask_path = images / f"{anchor_index:05d}_mask.png"
            _link_or_copy(
                source_frame["original_path"],
                anchor_path,
                code="dual_target_anchor_copy_invalid",
            )
            _link_or_copy(
                teacher_source,
                teacher_path,
                code="dual_target_teacher_copy_invalid",
            )
            _link_or_copy(
                teacher_source,
                teacher_override_path,
                code="dual_target_teacher_copy_invalid",
            )
            _link_or_copy(
                source_frame["mask_path"],
                exact_mask_path,
                code="dual_target_exact_mask_copy_invalid",
            )
            dilated = binary_dilation(
                source_frame["mask"] > 0,
                structure=structure,
                border_value=0,
            )
            anchor_loss = np.where(dilated, 0, 255).astype(np.uint8)
            Image.fromarray(anchor_loss, mode="L").save(anchor_mask_path)
            _link_or_copy(
                anchor_mask_path,
                anchor_sibling_mask_path,
                code="dual_target_anchor_mask_copy_invalid",
            )
            if (images / f"{teacher_index:05d}_mask.png").exists():
                raise DualTargetInputError(["dual_target_teacher_mask_forbidden"])

            common_transform = {
                key: value for key, value in transform.items() if key != "file_path"
            }
            anchor_transform = {
                **common_transform,
                "file_path": f"images/{anchor_index:05d}.png",
                "physical_camera_index": physical_index,
                "training_role": "original_outside_anchor",
            }
            teacher_transform = {
                **common_transform,
                "file_path": f"images/{teacher_index:05d}.png",
                "physical_camera_index": physical_index,
                "training_role": "whole_frame_semantic_teacher",
            }
            transform_rows.extend((anchor_transform, teacher_transform))
            review_rows.append(
                {
                    **common_transform,
                    "file_path": f"images/{teacher_index:05d}.png",
                    "physical_camera_index": physical_index,
                    "review_role": "unique_physical_camera_trajectory",
                }
            )
            exact_pixels = int(np.count_nonzero(source_frame["mask"]))
            excluded_pixels = int(np.count_nonzero(dilated))
            frame_rows.append(
                {
                    "physical_camera_index": physical_index,
                    "camera_id": source_frame["camera_id"],
                    "anchor_training_index": anchor_index,
                    "semantic_teacher_training_index": teacher_index,
                    "source_original_frame": _absolute_record(source_frame["original_path"]),
                    "source_exact_repair_mask": _absolute_record(source_frame["mask_path"]),
                    "source_whole_frame_semantic_teacher": _absolute_record(teacher_source),
                    "anchor_rgb": _relative_record(anchor_path, root=task_root),
                    "semantic_teacher_rgb": _relative_record(teacher_path, root=task_root),
                    "semantic_teacher_override_rgb": _relative_record(
                        teacher_override_path, root=task_root
                    ),
                    "exact_repair_mask": _relative_record(exact_mask_path, root=task_root),
                    "anchor_loss_mask": _relative_record(anchor_mask_path, root=task_root),
                    "anchor_loss_mask_sibling": _relative_record(
                        anchor_sibling_mask_path, root=task_root
                    ),
                    "teacher_loss_mask_materialized": False,
                    "pair_pose_and_intrinsics_exactly_equal": True,
                    "exact_repair_pixel_count": exact_pixels,
                    "excluded_anchor_loss_pixel_count": excluded_pixels,
                    "transition_added_pixel_count": excluded_pixels - exact_pixels,
                    "semantic_teacher_inside_exact_support_changed_pixels": (
                        teacher_frame["inside_exact_support_changed_pixels"]
                    ),
                    "semantic_teacher_outside_exact_support_changed_pixels": (
                        teacher_frame["outside_exact_support_changed_pixels"]
                    ),
                }
            )

        top_intrinsics = {key: value for key, value in source_transforms.items() if key != "frames"}
        transforms_value = {**top_intrinsics, "frames": transform_rows}
        transforms_path = task_root / "transforms.json"
        _write_json(transforms_path, transforms_value)
        review_value = {**top_intrinsics, "frames": review_rows}
        review_path = task_root / "review_transforms.json"
        _write_json(review_path, review_value)
        selected_path = task_root / "selected_anchor_indices.json"
        teacher_indices_path = task_root / "semantic_teacher_indices.json"
        _write_json(selected_path, anchor_indices)
        _write_json(teacher_indices_path, teacher_indices)
        camera_index: dict[str, Any] = {
            "schema_version": CAMERA_INDEX_SCHEMA,
            "ordering": "source_v3_physical_camera_order_anchor_then_teacher",
            "physical_camera_count": len(source_frames),
            "training_record_count": len(transform_rows),
            "frames": [
                {
                    "physical_camera_index": row["physical_camera_index"],
                    "camera_id": row["camera_id"],
                    "anchor_training_index": row["anchor_training_index"],
                    "semantic_teacher_training_index": row["semantic_teacher_training_index"],
                }
                for row in frame_rows
            ],
            "camera_index_digest": "",
        }
        camera_index["camera_index_digest"] = canonical_digest(
            camera_index, digest_field="camera_index_digest"
        )
        camera_index_path = task_root / "camera_index.json"
        _write_json(camera_index_path, camera_index)
        _write_colmap_calibration(sparse, transform_rows)
        _link_or_copy(
            copied_points,
            sparse / "points3D.bin",
            code="dual_target_source_seed_copy_invalid",
        )
        task_receipts.append(
            {
                "task_id": task_id,
                "scene_directory": str(task_root),
                "camera_count": len(source_frames),
                "physical_camera_count": len(source_frames),
                "training_record_count": len(transform_rows),
                "frames": frame_rows,
                "selected_anchor_indices": anchor_indices,
                "semantic_teacher_indices": teacher_indices,
                "selected_anchor_indices_file": _relative_record(selected_path, root=task_root),
                "semantic_teacher_indices_file": _relative_record(
                    teacher_indices_path, root=task_root
                ),
                "transforms": _relative_record(transforms_path, root=task_root),
                "review_trajectory": _relative_record(review_path, root=task_root),
                "camera_index": {
                    **_relative_record(camera_index_path, root=task_root),
                    "camera_index_digest": camera_index["camera_index_digest"],
                },
                "source_colmap": {
                    "cameras": _relative_record(sparse / "cameras.bin", root=task_root),
                    "images": _relative_record(sparse / "images.bin", root=task_root),
                    "points3D": _relative_record(sparse / "points3D.bin", root=task_root),
                },
                "loss_contract": {
                    "original_anchor": (
                        "unchanged_original_rgb_with_binary_loss_mask_zero_inside_"
                        "dilated_repair_support"
                    ),
                    "whole_frame_semantic_teacher": (
                        "unchanged_full_rgb_without_loss_mask_or_hard_composite"
                    ),
                    "same_pose_and_intrinsics_per_pair": True,
                    "direct_artifixer_required": False,
                    "artifixer3d_plus_required_for_this_packet": False,
                },
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "paired_target_inputs_prepared_no_model_no_execution",
        "pipeline_mode": "dual_target_artifixer3d_only",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": source.get("publisher_scene_id"),
        "execution_authority": dict(source["execution_authority"]),
        "source_candidate_inputs_receipt": {
            **_relative_record(copied_source_receipt, root=output),
            "source_path": str(source_path),
            "receipt_digest": source["receipt_digest"],
        },
        "semantic_teacher_receipts": copied_teacher_receipts,
        "replacement_object_count": len(task_ids),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "selected_task_ids": task_ids,
        "shared_seed": {
            "retained_scene": _relative_record(copied_retained, root=output),
            "retained_gaussian_count": source["shared_retained_scene"].get(
                "retained_gaussian_count"
            ),
            "colmap_points3D": _relative_record(copied_points, root=output),
            "source_retained_scene": _absolute_record(source_retained),
            "source_colmap_points3D": _absolute_record(source_points),
            "copied_byte_identical": True,
        },
        "transition_support": {
            "radius_pixels": transition_radius_pixels,
            "morphology": TRANSITION_MORPHOLOGY,
            "anchor_loss_mask_outside_support": 255,
            "anchor_loss_mask_inside_support": 0,
            "semantic_teacher_loss_mask": None,
        },
        "tasks": task_receipts,
        "execution": {
            "model_loaded": False,
            "direct_artifixer_executed": False,
            "artifixer3d_distillation_executed": False,
            "artifixer3d_plus_executed": False,
            "provider_mutations_performed": 0,
            "private_derived_upload_performed": False,
        },
        "claim_boundary": {
            "paired_target_packet_is_not_model_execution": True,
            "semantic_teacher_object_absence_reviewed": False,
            "appearance_repair_qualified": False,
            "source_gaussian_removal_qualified": False,
            "policy_input_use_permitted": False,
            "physical_or_deployment_evidence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{SCHEMA_VERSION}.json"
    _write_json(receipt_path, receipt)
    return receipt


__all__ = [
    "CAMERA_INDEX_SCHEMA",
    "DualTargetInputError",
    "HANDOFF_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SEMANTIC_TEACHER_SCHEMA",
    "TRANSITION_MORPHOLOGY",
    "materialize_dual_target_artifixer3d_inputs",
    "materialize_semantic_teacher_artifixer_handoff",
    "materialize_whole_frame_semantic_teacher_receipt",
]
