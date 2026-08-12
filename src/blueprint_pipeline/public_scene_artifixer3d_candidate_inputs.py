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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .public_scene_aura_exact_residual_preflight import (
    SCHEMA_VERSION as CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA,
)


SCHEMA_VERSION = "public_scene_artifixer3d_candidate_inputs.v1"
CAMERA_INDEX_SCHEMA = "public_scene_artifixer3d_camera_index.v1"
SPLIT_TEMPLATE_SCHEMA = "public_scene_artifixer3d_split_template.v1"
CAMERA_CONVENTION_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


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
        value.get("schema_version") != CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA
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


def materialize_artifixer3d_candidate_inputs(
    *, calibrated_residual_preflight_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Materialize no-model ArtiFixer candidate inputs from exact repair support."""

    preflight_path = _file(
        calibrated_residual_preflight_path,
        code="artifixer3d_calibrated_preflight_missing",
    )
    preflight = _validated_preflight(preflight_path)
    publisher_scene_id, execution_authority = _scene_identity(preflight)
    rows = preflight.get("camera_inputs")
    if not isinstance(rows, list) or not rows:
        raise ArtiFixer3DCandidateInputError(["artifixer3d_camera_inputs_missing"])
    normalized = [_camera_row(row) for row in rows]
    keys = [(str(row["task_id"]), str(row["camera_id"])) for row in normalized]
    if len(keys) != len(set(keys)):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_camera_input_duplicate"])
    task_ids = sorted({task_id for task_id, _camera_id in keys})
    if len(task_ids) != preflight["replacement_object_count"]:
        raise ArtiFixer3DCandidateInputError(["artifixer3d_task_set_mismatch"])

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArtiFixer3DCandidateInputError(["artifixer3d_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    task_receipts: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_root = output / task_id
        images_root = task_root / "images"
        renders_root = task_root / "renders"
        opacity_root = task_root / "opacity"
        masks_root = task_root / "exact_masks"
        for directory in (images_root, renders_root, opacity_root, masks_root):
            directory.mkdir(parents=True, exist_ok=True)
        task_rows = _ordered_cameras(
            [row for row in normalized if row["task_id"] == task_id]
        )
        frame_rows: list[dict[str, Any]] = []
        transforms_frames: list[dict[str, Any]] = []
        for index, row in enumerate(task_rows):
            filename = f"{index:05d}.png"
            repair = row["mask"] > 0
            reference = row["rgb"].copy()
            reference[repair] = 0
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
                    "repair_pixel_count": int(np.count_nonzero(repair)),
                    "outside_support_changed_pixels": outside_changes,
                }
            )
        top_intrinsics = dict(task_rows[0]["intrinsics"])
        transforms = {**top_intrinsics, "frames": transforms_frames}
        transforms_path = task_root / "transforms.json"
        _write_json(transforms_path, transforms)
        selected_indices = list(range(len(task_rows)))
        selected_path = task_root / "selected_indices.json"
        _write_json(selected_path, selected_indices)
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
        prompt_path = task_root / "captions" / "unconditioned_zero_prompt.h5"
        split_template: dict[str, Any] = {
            "schema_version": SPLIT_TEMPLATE_SCHEMA,
            "upstream_evalset": "reconstructed_colmap",
            "upstream_split": {
                "test": {
                    task_id: {
                        "transforms_path": "transforms.json",
                        "image_root": "images",
                        "render_dir": "renders",
                        "opacity_dir": "opacity",
                        "selected_indices_path": "selected_indices.json",
                        "prompt_path": "captions/unconditioned_zero_prompt.h5",
                        "camera_scale": 1.0,
                        "has_gt": False,
                    }
                }
            },
            "prompt_artifact_materialized": prompt_path.is_file(),
            "render_trajectory": "all_frames",
            "split_template_digest": "",
        }
        split_template["split_template_digest"] = canonical_digest(
            split_template, digest_field="split_template_digest"
        )
        split_path = task_root / "split.template.json"
        _write_json(split_path, split_template)
        task_receipts.append(
            {
                "task_id": task_id,
                "scene_directory": str(task_root),
                "camera_count": len(frame_rows),
                "frames": frame_rows,
                "transforms": _record(transforms_path),
                "selected_indices": _record(selected_path),
                "camera_index": {
                    **_record(camera_index_path),
                    "camera_index_digest": camera_index["camera_index_digest"],
                },
                "split_template": {
                    **_record(split_path),
                    "split_template_digest": split_template[
                        "split_template_digest"
                    ],
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
        "execution_authority": execution_authority,
        "replacement_object_count": preflight["replacement_object_count"],
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
            "reference_role": "retained_rgb_with_exact_repair_support_zeroed",
            "source_object_pixels_available_to_reference": False,
            "opacity_role": "binary_exact_repair_support_surrogate_not_native_3dgrut_opacity",
            "opacity_outside_support": 1.0,
            "opacity_inside_support": 0.0,
            "mask_dilation_pixels": 0,
            "direct_output_must_be_composited_inside_exact_support": True,
            "artifixer3d_distillation_input": (
                "exact_support_composited_direct_predictions_only"
            ),
            "artifixer3d_plus_input": "renders_from_candidate_artifixer3d_representation",
        },
        "execution_blockers": [
            "artifixer_checkpoint_bytes_not_bound",
            "artifixer_wan_base_model_snapshot_not_bound",
            "artifixer_cuda_container_image_not_bound",
            "artifixer_noncommercial_research_development_attestation_not_bound",
            "artifixer_unconditioned_zero_prompt_hdf5_not_materialized",
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
    "SCHEMA_VERSION",
    "materialize_artifixer3d_candidate_inputs",
]
