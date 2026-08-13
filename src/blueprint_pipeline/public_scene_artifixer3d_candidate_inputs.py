"""Prepare exact-support ArtiFixer3D candidate inputs for 1--5 tasks.

ArtiFixer consumes reconstruction RGB, opacity, calibrated camera rays, and
reference images.  The broad-repair packet has exact deleted-splat support but
does not have a native 3DGRUT opacity render.  This adapter therefore creates a
deliberately labelled binary opacity *surrogate*: pixels outside the frozen
repair support are opaque and pixels inside it are unknown.  References are
the retained renders with the same repair support zeroed, so the removed source
object cannot be copied back from a reference image.

The output is a candidate-input packet, not model execution or a qualified 3D
repair.  It intentionally leaves model, container, rights, and prompt artifacts
as fail-closed blockers for a later immutable paid bundle.
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

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .gaussian_splat_decode import read_standard_3dgs_ply
from .public_scene_aura_exact_residual_preflight import (
    SCHEMA_VERSION as CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA,
)
from .public_scene_segment_mask_repair_preflight import (
    SCHEMA_VERSION as CALIBRATED_SEGMENT_REPAIR_PREFLIGHT_SCHEMA,
)


SCHEMA_VERSION = "public_scene_artifixer3d_candidate_inputs.v3"
OBJECT_ABSENT_REFERENCE_SCHEMA = "public_scene_object_absent_reference_candidates.v1"
CAMERA_INDEX_SCHEMA = "public_scene_artifixer3d_camera_index.v1"
SPLIT_TEMPLATE_SCHEMA = "public_scene_artifixer3d_split_template.v1"
CAMERA_CONVENTION_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
SH_C0 = 0.28209479177387814


class ArtiFixer3DCandidateInputError(ValueError):
    """Stable failures for the no-model, no-upload preparation boundary."""

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
        raise ArtiFixer3DCandidateInputError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise ArtiFixer3DCandidateInputError([code])
    return path


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtiFixer3DCandidateInputError([code]) from exc
    if not isinstance(value, dict):
        raise ArtiFixer3DCandidateInputError([code])
    return value


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    key = "relative_path" if root is not None else "path"
    value = path.relative_to(root).as_posix() if root is not None else str(path)
    return {key: value, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bound_absolute(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise ArtiFixer3DCandidateInputError([code])
    path = _file(value.get("path"), code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get(
        "sha256"
    ):
        raise ArtiFixer3DCandidateInputError([code])
    return path


def _image(path: Path, *, mode: str, code: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert(mode), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ArtiFixer3DCandidateInputError([code]) from exc


def _validated_preflight(path: Path) -> dict[str, Any]:
    value = _read(path, code="artifixer3d_calibrated_preflight_unreadable")
    count = value.get("replacement_object_count")
    if (
        value.get("schema_version")
        not in {
            CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA,
            CALIBRATED_SEGMENT_REPAIR_PREFLIGHT_SCHEMA,
        }
        or value.get("status") != "prepared_no_upload_no_execution"
        or value.get("preflight_digest")
        != canonical_digest(value, digest_field="preflight_digest")
        or not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= MAX_REPLACEMENT_OBJECTS
        or len(value.get("lanes") or []) != count
        or value.get("execution", {}).get("provider_mutations_performed") != 0
        or value.get("execution", {}).get("aura_inpainting_executed") is not False
        or value.get("required_result_checks", {}).get(
            "outside_mask_pixel_delta_required"
        )
        != 0
        or value.get("required_result_checks", {}).get(
            "locality_mask_dilation_pixels"
        )
        != 0
    ):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_calibrated_preflight_invalid"]
        )
    return value


def _validated_object_absent_reference_receipts(
    paths: Sequence[str | Path],
) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for unresolved in paths:
        path = _file(
            unresolved, code="artifixer3d_object_absent_reference_receipt_missing"
        )
        value = _read(
            path, code="artifixer3d_object_absent_reference_receipt_unreadable"
        )
        task_id = str(value.get("task_id") or "")
        frames = value.get("frames")
        if (
            value.get("schema_version") != OBJECT_ABSENT_REFERENCE_SCHEMA
            or value.get("status")
            != "candidate_frames_exact_support_composited"
            or value.get("receipt_digest")
            != canonical_digest(value, digest_field="receipt_digest")
            or not task_id
            or task_id in receipts
            or not isinstance(frames, list)
            or not frames
            or value.get("frame_count") != len(frames)
            or value.get("outside_support_changed_pixels_total") != 0
        ):
            raise ArtiFixer3DCandidateInputError(
                ["artifixer3d_object_absent_reference_receipt_invalid"]
            )
        camera_ids: list[str] = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ArtiFixer3DCandidateInputError(
                    ["artifixer3d_object_absent_reference_receipt_invalid"]
                )
            camera_id = str(frame.get("camera_id") or "")
            if (
                not camera_id
                or frame.get("outside_support_changed_pixels") != 0
            ):
                raise ArtiFixer3DCandidateInputError(
                    ["artifixer3d_object_absent_reference_receipt_invalid"]
                )
            for field in ("source_render", "exact_repair_mask", "object_absent_frame"):
                _bound_absolute(
                    frame.get(field),
                    code="artifixer3d_object_absent_reference_frame_invalid",
                )
            camera_ids.append(camera_id)
        if len(camera_ids) != len(set(camera_ids)):
            raise ArtiFixer3DCandidateInputError(
                ["artifixer3d_object_absent_reference_receipt_invalid"]
            )
        receipts[task_id] = {**value, "receipt_path": path}
    return receipts


def _scene_identity(preflight: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    backend = preflight.get("backend_admission")
    authority_record = (
        backend.get("execution_authority") if isinstance(backend, Mapping) else None
    )
    authority_path = _bound_absolute(
        authority_record, code="artifixer3d_execution_authority_invalid"
    )
    authority = _read(authority_path, code="artifixer3d_execution_authority_invalid")
    scene_id = str(authority.get("publisher_scene_id") or "")
    if (
        not scene_id
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or not isinstance(authority_record, Mapping)
        or authority_record.get("authority_digest") != authority["authority_digest"]
    ):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_execution_authority_invalid"]
        )
    return scene_id, {
        **_record(authority_path),
        "authority_digest": authority["authority_digest"],
    }


def _camera_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_camera_input_invalid"])
    task_id = str(row.get("task_id") or "")
    camera_id = str(row.get("camera_id") or "")
    calibration = row.get("calibration")
    spec = calibration.get("spec") if isinstance(calibration, Mapping) else None
    pose = spec.get("pose") if isinstance(spec, Mapping) else None
    intrinsics = spec.get("intrinsics") if isinstance(spec, Mapping) else None
    matrix = pose.get("T_world_camera_opencv") if isinstance(pose, Mapping) else None
    if (
        not task_id
        or not camera_id
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
        or [float(value) for value in matrix[3]] != [0.0, 0.0, 0.0, 1.0]
        or not isinstance(intrinsics, Mapping)
        or intrinsics.get("model") != "PINHOLE"
        or any(
            not isinstance(intrinsics.get(field), (int, float))
            or isinstance(intrinsics.get(field), bool)
            or not math.isfinite(float(intrinsics[field]))
            for field in ("fx", "fy", "cx", "cy", "width", "height")
        )
        or float(intrinsics["fx"]) <= 0
        or float(intrinsics["fy"]) <= 0
        or int(intrinsics["width"]) <= 0
        or int(intrinsics["height"]) <= 0
    ):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_camera_input_invalid"])
    before = _bound_absolute(
        row.get("retained_scene_before"), code="artifixer3d_retained_frame_invalid"
    )
    mask = _bound_absolute(
        row.get("exact_residual_mask"), code="artifixer3d_exact_mask_invalid"
    )
    rgb = _image(before, mode="RGB", code="artifixer3d_retained_frame_invalid")
    mask_pixels = _image(mask, mode="L", code="artifixer3d_exact_mask_invalid")
    if (
        rgb.shape[:2] != mask_pixels.shape
        or rgb.shape[1] != int(intrinsics["width"])
        or rgb.shape[0] != int(intrinsics["height"])
        or set(mask_pixels.tobytes()) - {0, 255}
        or not np.any(mask_pixels)
        or int(np.count_nonzero(mask_pixels))
        != (row.get("exact_residual_mask") or {}).get("pixel_count")
    ):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_frame_shape_or_mask_invalid"]
        )
    matrix_array = np.asarray(matrix, dtype=np.float64)
    return {
        "task_id": task_id,
        "camera_id": camera_id,
        "T_world_camera_opencv": matrix_array,
        "T_world_camera_opengl": matrix_array @ CAMERA_CONVENTION_FLIP,
        "intrinsics": {
            "camera_model": "OPENCV",
            "w": int(intrinsics["width"]),
            "h": int(intrinsics["height"]),
            "fl_x": float(intrinsics["fx"]),
            "fl_y": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        },
        "before_path": before,
        "mask_path": mask,
        "rgb": rgb,
        "mask": mask_pixels,
    }


def _ordered_cameras(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a deterministic nearest-camera path without semantic labels."""

    remaining = {str(row["camera_id"]): row for row in rows}
    current_id = min(remaining)
    ordered = [remaining.pop(current_id)]
    while remaining:
        current = ordered[-1]["T_world_camera_opencv"][:3, 3]
        next_id = min(
            remaining,
            key=lambda camera_id: (
                float(
                    np.linalg.norm(
                        remaining[camera_id]["T_world_camera_opencv"][:3, 3]
                        - current
                    )
                ),
                camera_id,
            ),
        )
        ordered.append(remaining.pop(next_id))
    return ordered


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.asarray(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    values, vectors = np.linalg.eigh(matrix)
    qvec = vectors[[3, 0, 1, 2], int(np.argmax(values))]
    return -qvec if qvec[0] < 0 else qvec


def _retained_splat(preflight: Mapping[str, Any]):
    record = preflight.get("shared_retained_scene")
    path = _bound_absolute(record, code="artifixer3d_retained_splat_invalid")
    splat = read_standard_3dgs_ply(path)
    if (
        not isinstance(record, Mapping)
        or record.get("retained_gaussian_count") != splat.count
        or splat.count <= 0
    ):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_retained_splat_invalid"])
    return path, splat


def _write_colmap_points3d(path: Path, splat: Any) -> None:
    """Convert retained splat centers to a deterministic COLMAP seed cloud."""

    rgb = np.rint(
        np.clip(0.5 + SH_C0 * np.asarray(splat.f_dc), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", int(splat.count)))
        for point_id, (xyz, color) in enumerate(zip(splat.xyz, rgb), start=1):
            stream.write(
                struct.pack(
                    "<QdddBBBdQ",
                    point_id,
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    0.0,
                    0,
                )
            )


def _write_colmap_cameras_and_images(
    sparse: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write the binary COLMAP calibration ArtiFixer3D reads directly."""

    camera_path = sparse / "cameras.bin"
    image_path = sparse / "images.bin"
    with camera_path.open("wb") as cameras, image_path.open("wb") as images:
        cameras.write(struct.pack("<Q", len(rows)))
        images.write(struct.pack("<Q", len(rows)))
        for index, row in enumerate(rows, start=1):
            intrinsic = row["intrinsics"]
            cameras.write(
                struct.pack(
                    "<iiQQdddddddd",
                    index,
                    4,  # COLMAP OPENCV
                    int(intrinsic["w"]),
                    int(intrinsic["h"]),
                    float(intrinsic["fl_x"]),
                    float(intrinsic["fl_y"]),
                    float(intrinsic["cx"]),
                    float(intrinsic["cy"]),
                    float(intrinsic["k1"]),
                    float(intrinsic["k2"]),
                    float(intrinsic["p1"]),
                    float(intrinsic["p2"]),
                )
            )
            world_to_camera = np.linalg.inv(row["T_world_camera_opencv"])
            qvec = _rotmat_to_qvec(world_to_camera[:3, :3])
            images.write(
                struct.pack(
                    "<idddddddi",
                    index,
                    *qvec,
                    *world_to_camera[:3, 3],
                    index,
                )
            )
            images.write(f"{index - 1:05d}.png".encode("utf-8") + b"\x00")
            images.write(struct.pack("<Q", 0))


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    if _sha256(source) != _sha256(destination):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_colmap_seed_copy_invalid"]
        )


def _split_template(
    *,
    task_id: str,
    selected_name: str,
    target_name: str | None,
    render_trajectory: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "transforms_path": "transforms.json",
        "image_root": ".",
        "render_dir": "renders",
        "opacity_dir": "opacity",
        "selected_indices_path": selected_name,
        "prompt_path": "captions/unconditioned_zero_prompt.h5",
        "camera_scale": 1.0,
        "has_gt": False,
    }
    if target_name is not None:
        metadata["target_indices_path"] = target_name
    value: dict[str, Any] = {
        "schema_version": SPLIT_TEMPLATE_SCHEMA,
        "upstream_evalset": "reconstructed_colmap",
        "upstream_split": {"test": {task_id: metadata}},
        "prompt_artifact_materialized": False,
        "render_trajectory": render_trajectory,
        "split_template_digest": "",
    }
    value["split_template_digest"] = canonical_digest(
        value, digest_field="split_template_digest"
    )
    return value


def materialize_object_absent_reference_candidate_receipt(
    *,
    source_candidate_inputs_receipt_path: str | Path,
    task_id: str,
    object_absent_frames_root: str | Path,
    editor_identity: Mapping[str, Any],
    prompt_policy: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind precomputed object-absent views and prove their exact-support locality."""

    source_path = _file(
        source_candidate_inputs_receipt_path,
        code="artifixer3d_source_candidate_receipt_missing",
    )
    source = _read(source_path, code="artifixer3d_source_candidate_receipt_unreadable")
    tasks = source.get("tasks")
    if (
        source.get("schema_version") != SCHEMA_VERSION
        or source.get("receipt_digest")
        != canonical_digest(source, digest_field="receipt_digest")
        or not isinstance(tasks, list)
        or not task_id
        or not isinstance(editor_identity, Mapping)
        or not editor_identity
        or not isinstance(prompt_policy, str)
        or not prompt_policy.strip()
    ):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_source_candidate_receipt_invalid"]
        )
    matches = [task for task in tasks if task.get("task_id") == task_id]
    if len(matches) != 1:
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_object_absent_reference_task_invalid"]
        )
    task = matches[0]
    task_root = Path(str(task["scene_directory"])).expanduser().resolve()
    generated_root = Path(object_absent_frames_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if (
        generated_root.is_symlink()
        or not generated_root.is_dir()
        or output.is_symlink()
        or output.exists()
    ):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_object_absent_reference_output_invalid"]
        )
    rows: list[dict[str, Any]] = []
    for frame in task.get("frames") or []:
        index = int(frame["frame_index"])
        candidate_path = _file(
            generated_root / f"{index:05d}.png",
            code="artifixer3d_object_absent_reference_frame_missing",
        )
        source_render = _bound_absolute(
            {
                **frame["rendered_rgb"],
                "path": str(task_root / frame["rendered_rgb"]["relative_path"]),
            },
            code="artifixer3d_object_absent_reference_source_invalid",
        )
        mask_path = _bound_absolute(
            {
                **frame["exact_repair_mask"],
                "path": str(task_root / frame["exact_repair_mask"]["relative_path"]),
            },
            code="artifixer3d_object_absent_reference_mask_invalid",
        )
        before = _image(
            source_render,
            mode="RGB",
            code="artifixer3d_object_absent_reference_source_invalid",
        )
        repair = _image(
            candidate_path,
            mode="RGB",
            code="artifixer3d_object_absent_reference_frame_invalid",
        )
        support = (
            _image(
                mask_path,
                mode="L",
                code="artifixer3d_object_absent_reference_mask_invalid",
            )
            > 0
        )
        if before.shape != repair.shape or before.shape[:2] != support.shape:
            raise ArtiFixer3DCandidateInputError(
                ["artifixer3d_object_absent_reference_shape_invalid"]
            )
        outside_changes = int(
            np.count_nonzero(
                np.any(repair[~support] != before[~support], axis=1)
            )
        )
        if outside_changes != 0:
            raise ArtiFixer3DCandidateInputError(
                ["artifixer3d_object_absent_reference_outside_change"]
            )
        rows.append(
            {
                "frame_index": index,
                "camera_id": frame["camera_id"],
                "source_render": _record(source_render),
                "exact_repair_mask": _record(mask_path),
                "object_absent_frame": _record(candidate_path),
                "repair_pixel_count": int(np.count_nonzero(support)),
                "outside_support_changed_pixels": outside_changes,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": OBJECT_ABSENT_REFERENCE_SCHEMA,
        "status": "candidate_frames_exact_support_composited",
        "task_id": task_id,
        "source_candidate_inputs_receipt": {
            **_record(source_path),
            "receipt_digest": source["receipt_digest"],
        },
        "editor_identity": dict(editor_identity),
        "prompt_policy": prompt_policy.strip(),
        "frame_count": len(rows),
        "frames": rows,
        "outside_support_changed_pixels_total": sum(
            row["outside_support_changed_pixels"] for row in rows
        ),
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "physical_or_deployment_evidence": False,
        "claim_boundary": (
            "precomputed_object_absent_candidate_views_not_capture_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, receipt)
    return receipt


def materialize_artifixer3d_candidate_inputs(
    *,
    calibrated_residual_preflight_path: str | Path,
    output_root: str | Path,
    selected_task_ids: Sequence[str] | None = None,
    object_absent_reference_receipt_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Materialize no-model ArtiFixer candidate inputs from exact repair support."""

    preflight_path = _file(
        calibrated_residual_preflight_path,
        code="artifixer3d_calibrated_preflight_missing",
    )
    preflight = _validated_preflight(preflight_path)
    object_absent_receipts = _validated_object_absent_reference_receipts(
        object_absent_reference_receipt_paths
    )
    publisher_scene_id, execution_authority = _scene_identity(preflight)
    retained_splat_path, retained_splat = _retained_splat(preflight)
    rows = preflight.get("camera_inputs")
    if not isinstance(rows, list) or not rows:
        raise ArtiFixer3DCandidateInputError(["artifixer3d_camera_inputs_missing"])
    normalized = [_camera_row(row) for row in rows]
    keys = [(str(row["task_id"]), str(row["camera_id"])) for row in normalized]
    if len(keys) != len(set(keys)):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_camera_input_duplicate"])
    available_task_ids = sorted({task_id for task_id, _camera_id in keys})
    if len(available_task_ids) != preflight["replacement_object_count"]:
        raise ArtiFixer3DCandidateInputError(["artifixer3d_task_set_mismatch"])
    if selected_task_ids is None:
        task_ids = available_task_ids
    else:
        task_ids = [str(task_id) for task_id in selected_task_ids]
        if (
            not 1 <= len(task_ids) <= MAX_REPLACEMENT_OBJECTS
            or len(task_ids) != len(set(task_ids))
            or any(not task_id or task_id not in available_task_ids for task_id in task_ids)
        ):
            raise ArtiFixer3DCandidateInputError(
                ["artifixer3d_selected_task_set_invalid"]
            )
        task_ids = sorted(task_ids)
    normalized = [row for row in normalized if row["task_id"] in set(task_ids)]
    if set(object_absent_receipts) - set(task_ids):
        raise ArtiFixer3DCandidateInputError(
            ["artifixer3d_object_absent_reference_task_invalid"]
        )

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    shared_seed_root = output / "shared_initialization"
    shared_seed_root.mkdir()
    shared_points = shared_seed_root / "points3D.bin"
    _write_colmap_points3d(shared_points, retained_splat)
    task_receipts: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_root = output / task_id
        images_root = task_root / "images"
        renders_root = task_root / "renders"
        opacity_root = task_root / "opacity"
        masks_root = task_root / "exact_masks"
        sparse_root = task_root / "sparse" / "0"
        for directory in (
            images_root,
            renders_root,
            opacity_root,
            masks_root,
            sparse_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        task_rows = _ordered_cameras(
            [row for row in normalized if row["task_id"] == task_id]
        )
        bound_reference = object_absent_receipts.get(task_id)
        bound_frames_by_camera = {
            str(frame["camera_id"]): frame
            for frame in (bound_reference or {}).get("frames", [])
        }
        if bound_reference is not None and set(bound_frames_by_camera) != {
            str(row["camera_id"]) for row in task_rows
        }:
            raise ArtiFixer3DCandidateInputError(
                ["artifixer3d_object_absent_reference_camera_set_invalid"]
            )
        frame_rows: list[dict[str, Any]] = []
        repair_support_fractions: list[float] = []
        transforms_frames: list[dict[str, Any]] = []
        for index, row in enumerate(task_rows):
            filename = f"{index:05d}.png"
            repair = row["mask"] > 0
            bound_frame = bound_frames_by_camera.get(str(row["camera_id"]))
            if bound_frame is None:
                reference = row["rgb"].copy()
                reference[repair] = 0
                reference_source = "retained_rgb_with_exact_repair_support_zeroed"
            else:
                bound_source = _bound_absolute(
                    bound_frame.get("source_render"),
                    code="artifixer3d_object_absent_reference_source_invalid",
                )
                bound_mask = _bound_absolute(
                    bound_frame.get("exact_repair_mask"),
                    code="artifixer3d_object_absent_reference_mask_invalid",
                )
                bound_candidate = _bound_absolute(
                    bound_frame.get("object_absent_frame"),
                    code="artifixer3d_object_absent_reference_frame_invalid",
                )
                bound_source_pixels = _image(
                    bound_source,
                    mode="RGB",
                    code="artifixer3d_object_absent_reference_source_invalid",
                )
                bound_mask_pixels = _image(
                    bound_mask,
                    mode="L",
                    code="artifixer3d_object_absent_reference_mask_invalid",
                )
                if not np.array_equal(bound_source_pixels, row["rgb"]) or not np.array_equal(
                    bound_mask_pixels, row["mask"]
                ):
                    raise ArtiFixer3DCandidateInputError(
                        ["artifixer3d_object_absent_reference_binding_invalid"]
                    )
                reference = _image(
                    bound_candidate,
                    mode="RGB",
                    code="artifixer3d_object_absent_reference_frame_invalid",
                )
                if reference.shape != row["rgb"].shape or np.any(
                    reference[~repair] != row["rgb"][~repair]
                ):
                    raise ArtiFixer3DCandidateInputError(
                        ["artifixer3d_object_absent_reference_outside_change"]
                    )
                reference_source = "bound_object_absent_exact_support_candidate"
            opacity = np.where(repair, 0, 255).astype(np.uint8)
            reference_path = images_root / filename
            render_path = renders_root / filename
            opacity_path = opacity_root / filename
            mask_path = masks_root / filename
            Image.fromarray(reference, mode="RGB").save(reference_path)
            Image.fromarray(row["rgb"], mode="RGB").save(render_path)
            Image.fromarray(opacity, mode="L").save(opacity_path)
            Image.fromarray(row["mask"], mode="L").save(mask_path)
            outside = ~repair
            outside_changes = int(
                np.count_nonzero(np.any(reference[outside] != row["rgb"][outside], axis=1))
            )
            if outside_changes != 0 or not np.array_equal(opacity == 0, repair):
                raise ArtiFixer3DCandidateInputError(
                    ["artifixer3d_exact_support_transform_invalid"]
                )
            repair_pixel_count = int(np.count_nonzero(repair))
            image_pixel_count = int(repair.size)
            repair_support_fraction = repair_pixel_count / image_pixel_count
            repair_support_fractions.append(repair_support_fraction)
            intrinsics = row["intrinsics"]
            transforms_frames.append(
                {
                    "file_path": f"images/{filename}",
                    "camera_id": row["camera_id"],
                    "transform_matrix": row["T_world_camera_opengl"].tolist(),
                    **intrinsics,
                }
            )
            frame_rows.append(
                {
                    "frame_index": index,
                    "camera_id": row["camera_id"],
                    "input_retained_frame": _record(row["before_path"]),
                    "input_exact_repair_mask": _record(row["mask_path"]),
                    "masked_reference_rgb": _record(reference_path, root=task_root),
                    "rendered_rgb": _record(render_path, root=task_root),
                    "binary_opacity_surrogate": _record(
                        opacity_path, root=task_root
                    ),
                    "exact_repair_mask": _record(mask_path, root=task_root),
                    "repair_pixel_count": repair_pixel_count,
                    "image_pixel_count": image_pixel_count,
                    "repair_support_fraction": repair_support_fraction,
                    "outside_support_changed_pixels": outside_changes,
                    "reference_source": reference_source,
                }
            )
        top_intrinsics = dict(task_rows[0]["intrinsics"])
        transforms = {**top_intrinsics, "frames": transforms_frames}
        transforms_path = task_root / "transforms.json"
        _write_json(transforms_path, transforms)
        _write_colmap_cameras_and_images(sparse_root, task_rows)
        _link_or_copy(shared_points, sparse_root / "points3D.bin")

        all_indices = list(range(len(task_rows)))
        selected_indices = all_indices[::2]
        target_indices = all_indices[1::2]
        selected_path = task_root / "selected_indices.json"
        _write_json(selected_path, selected_indices)
        target_path = task_root / "target_indices.json"
        if target_indices:
            _write_json(target_path, target_indices)
        camera_index: dict[str, Any] = {
            "schema_version": CAMERA_INDEX_SCHEMA,
            "ordering": (
                "lexicographically_smallest_camera_then_greedy_nearest_camera_center_"
                "with_camera_id_tiebreak"
            ),
            "camera_count": len(frame_rows),
            "frames": [
                {"frame_index": row["frame_index"], "camera_id": row["camera_id"]}
                for row in frame_rows
            ],
            "camera_index_digest": "",
        }
        camera_index["camera_index_digest"] = canonical_digest(
            camera_index, digest_field="camera_index_digest"
        )
        camera_index_path = task_root / "camera_index.json"
        _write_json(camera_index_path, camera_index)
        split_rows: list[dict[str, Any]] = []
        fold_sets = [("fold_0", selected_indices, target_indices)]
        if target_indices:
            fold_sets.append(("fold_1", target_indices, selected_indices))
        for fold_id, anchors, targets in fold_sets:
            selected_name = f"selected_indices.{fold_id}.json"
            _write_json(task_root / selected_name, anchors)
            target_name = None
            trajectory = "all_frames"
            if targets:
                target_name = f"target_indices.{fold_id}.json"
                _write_json(task_root / target_name, targets)
                trajectory = "trajectory"
            split_template = _split_template(
                task_id=task_id,
                selected_name=selected_name,
                target_name=target_name,
                render_trajectory=trajectory,
            )
            split_path = task_root / f"split.direct_{fold_id}.template.json"
            _write_json(split_path, split_template)
            split_rows.append(
                {
                    "fold_id": fold_id,
                    "selected_indices": anchors,
                    "target_indices": targets,
                    "split_template": {
                        **_record(split_path),
                        "split_template_digest": split_template[
                            "split_template_digest"
                        ],
                    },
                }
            )
        task_receipts.append(
            {
                "task_id": task_id,
                "scene_directory": str(task_root),
                "camera_count": len(frame_rows),
                "object_absent_reference_receipt": (
                    {
                        **_record(bound_reference["receipt_path"]),
                        "receipt_digest": bound_reference["receipt_digest"],
                    }
                    if bound_reference is not None
                    else None
                ),
                "frames": frame_rows,
                "repair_support_coverage": {
                    "minimum_fraction": min(repair_support_fractions),
                    "mean_fraction": sum(repair_support_fractions)
                    / len(repair_support_fractions),
                    "maximum_fraction": max(repair_support_fractions),
                    "interpretation": (
                        "pre_execution_large_hole_risk_metric_not_method_"
                        "quality_or_qualification_verdict"
                    ),
                },
                "transforms": _record(transforms_path),
                "selected_indices": _record(selected_path),
                "target_indices": _record(target_path)
                if target_indices
                else None,
                "direct_inference_folds": split_rows,
                "direct_prediction_coverage_indices": sorted(
                    {index for row in split_rows for index in row["target_indices"]}
                    or set(all_indices)
                ),
                "artifixer3d_distillation": {
                    "selected_anchor_indices": selected_indices,
                    "generated_prediction_indices": target_indices,
                    "required_repaired_input_indices": all_indices,
                    "camera_partition_eligible": bool(target_indices),
                    "execution_eligible": False,
                    "required_image_role": (
                        "exact_support_composited_object_free_background_"
                        "prediction_from_complementary_direct_folds"
                    ),
                    "masked_reference_placeholders_permitted_as_distillation_images": False,
                    "source_colmap": {
                        "cameras": _record(sparse_root / "cameras.bin"),
                        "images": _record(sparse_root / "images.bin"),
                        "points3D": _record(sparse_root / "points3D.bin"),
                    },
                },
                "camera_index": {
                    **_record(camera_index_path),
                    "camera_index_digest": camera_index["camera_index_digest"],
                },
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_inputs_prepared_no_model_no_execution",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": publisher_scene_id,
        "calibrated_residual_preflight": {
            **_record(preflight_path),
            "preflight_digest": preflight["preflight_digest"],
        },
        "shared_retained_scene": {
            **_record(retained_splat_path),
            "retained_gaussian_count": retained_splat.count,
        },
        "shared_colmap_initialization_points3D": _record(shared_points),
        "execution_authority": execution_authority,
        "replacement_object_count": len(task_ids),
        "source_preflight_replacement_object_count": preflight[
            "replacement_object_count"
        ],
        "selected_task_ids": task_ids,
        "object_absent_reference_receipts": [
            {
                **_record(object_absent_receipts[task_id]["receipt_path"]),
                "task_id": task_id,
                "receipt_digest": object_absent_receipts[task_id]["receipt_digest"],
            }
            for task_id in sorted(object_absent_receipts)
        ],
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "all_replacements_co_present_in_shared_retained_scene": True,
        "tasks": task_receipts,
        "adapter": {
            "upstream_evalset": "reconstructed_colmap",
            "camera_input_convention": "T_world_camera_opencv",
            "upstream_camera_convention": "T_world_camera_opengl",
            "camera_conversion": "T_world_camera_opencv @ diag(1,-1,-1,1)",
            "camera_scale": 1.0,
            "camera_scale_basis": "sealed_metric_world_camera_translations_in_meters",
            "reference_role": (
                "per_task_bound_object_absent_candidate_or_zeroed_exact_support"
                if object_absent_receipts
                else "retained_rgb_with_exact_repair_support_zeroed"
            ),
            "source_object_pixels_available_to_reference": False,
            "bound_object_absent_reference_task_count": len(
                object_absent_receipts
            ),
            "opacity_role": "binary_exact_repair_support_surrogate_not_native_3dgrut_opacity",
            "opacity_outside_support": 1.0,
            "opacity_inside_support": 0.0,
            "mask_dilation_pixels": 0,
            "direct_output_must_be_composited_inside_exact_support": True,
            "artifixer3d_distillation_input": (
                "all_camera_exact_support_composited_object_free_background_"
                "predictions_from_both_complementary_direct_folds_only"
            ),
            "artifixer3d_plus_input": "renders_from_candidate_artifixer3d_representation",
        },
        "repair_target_semantics": {
            "inside_exact_support": (
                "plausible_object_free_background_consistent_with_calibrated_"
                "outside_support_context"
            ),
            "source_washer_or_notebook_restoration_permitted": False,
            "black_unknown_placeholder_preservation_permitted": False,
            "outside_exact_support_changed_pixels_permitted": 0,
        },
        "execution_blockers": [
            "artifixer_checkpoint_bytes_not_bound",
            "artifixer_wan_base_model_snapshot_not_bound",
            "artifixer_cuda_container_image_not_bound",
            "artifixer_noncommercial_research_development_attestation_not_bound",
            "artifixer_unconditioned_zero_prompt_hdf5_not_materialized",
            "artifixer_all_camera_exact_support_composited_repairs_not_materialized",
            *(
                []
                if all(
                    task["artifixer3d_distillation"]["camera_partition_eligible"]
                    for task in task_receipts
                )
                else ["artifixer3d_requires_at_least_two_calibrated_views_per_task"]
            ),
        ],
        "scientific_limitations": [
            "native_continuous_3dgrut_opacity_unavailable_binary_exact_support_surrogate_used",
            "references_are_masked_derived_renders_not_clean_observed_background_views",
            "hidden_background_truth_unavailable",
        ],
        "execution": {
            "model_loaded": False,
            "artifixer_direct_inference_executed": False,
            "artifixer3d_distillation_executed": False,
            "artifixer3d_plus_inference_executed": False,
            "provider_mutations_performed": 0,
            "private_derived_upload_performed": False,
        },
        "claim_boundary": {
            "candidate_input_packet_is_not_method_execution": True,
            "generated_pixels_are_captured_evidence": False,
            "appearance_repair_qualified": False,
            "source_gaussian_removal_qualified": False,
            "native_simulator_import_qualified": False,
            "policy_input_use_permitted": False,
            "physical_or_deployment_evidence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output / f"{SCHEMA_VERSION}.json"
    _write_json(receipt_path, receipt)
    return receipt


__all__ = [
    "ArtiFixer3DCandidateInputError",
    "OBJECT_ABSENT_REFERENCE_SCHEMA",
    "SCHEMA_VERSION",
    "materialize_artifixer3d_candidate_inputs",
    "materialize_object_absent_reference_candidate_receipt",
]
