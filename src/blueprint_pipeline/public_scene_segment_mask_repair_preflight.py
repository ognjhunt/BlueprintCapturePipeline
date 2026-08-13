"""Seal calibrated object-segment masks as the only generated repair support.

The segment-contribution cutout deliberately records every source Gaussian that
contributes to an object segment.  Removing those Gaussians can create a much
larger 3D render hole because a single Gaussian may also contribute background
pixels.  This adapter keeps that deletion provenance, but gives an image repair
backend only the original object segment.  Every pixel outside the exact binary
segment remains canonical source-frame context and must be restored byte for
byte by the final compositor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS, validate_task_freeze
from .gaussian_splat_decode import read_standard_3dgs_ply


SCHEMA_VERSION = "public_scene_calibrated_exact_segment_repair_preflight.v1"
CANDIDATE_SCHEMA = "adp009d_segment_contribution_cutout_set.v1"


class SegmentMaskRepairPreflightError(ValueError):
    """Stable fail-closed errors for exact-segment repair preparation."""

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
        raise SegmentMaskRepairPreflightError([code]) from exc
    if not isinstance(value, dict):
        raise SegmentMaskRepairPreflightError([code])
    return value


def _file(value: Any, *, code: str) -> Path:
    unresolved = Path(str(value or "")).expanduser()
    if unresolved.is_symlink():
        raise SegmentMaskRepairPreflightError([code])
    path = unresolved.resolve()
    if not path.is_file():
        raise SegmentMaskRepairPreflightError([code])
    return path


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bound_absolute(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise SegmentMaskRepairPreflightError([code])
    path = _file(value.get("path"), code=code)
    if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256(path):
        raise SegmentMaskRepairPreflightError([code])
    return path


def _bound_relative(root: Path, value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise SegmentMaskRepairPreflightError([code])
    relative = str(value.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise SegmentMaskRepairPreflightError([code])
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise SegmentMaskRepairPreflightError([code])
    if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256(path):
        raise SegmentMaskRepairPreflightError([code])
    return path


def _camera_rows(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentMaskRepairPreflightError(["segment_repair_camera_contract_invalid"]) from exc
    if not isinstance(value, list) or not value:
        raise SegmentMaskRepairPreflightError(["segment_repair_camera_contract_invalid"])
    rows: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise SegmentMaskRepairPreflightError(["segment_repair_camera_contract_invalid"])
        camera_id = str(row.get("camera_id") or "")
        matrix = row.get("T_world_camera_provider_frame")
        intrinsics = row.get("intrinsics")
        if (
            not camera_id
            or camera_id in rows
            or not isinstance(matrix, list)
            or len(matrix) != 4
            or any(not isinstance(line, list) or len(line) != 4 for line in matrix)
            or not isinstance(intrinsics, Mapping)
            or intrinsics.get("model", "PINHOLE") != "PINHOLE"
        ):
            raise SegmentMaskRepairPreflightError(["segment_repair_camera_contract_invalid"])
        try:
            normalized_matrix = [[float(item) for item in line] for line in matrix]
            normalized_intrinsics = {
                "model": "PINHOLE",
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "width": int(intrinsics["width"]),
                "height": int(intrinsics["height"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmentMaskRepairPreflightError(
                ["segment_repair_camera_contract_invalid"]
            ) from exc
        if (
            normalized_matrix[3] != [0.0, 0.0, 0.0, 1.0]
            or normalized_intrinsics["fx"] <= 0
            or normalized_intrinsics["fy"] <= 0
            or normalized_intrinsics["width"] <= 0
            or normalized_intrinsics["height"] <= 0
        ):
            raise SegmentMaskRepairPreflightError(["segment_repair_camera_contract_invalid"])
        rows[camera_id] = {
            "id": camera_id,
            "spec": {
                "pose": {"T_world_camera_opencv": normalized_matrix},
                "intrinsics": normalized_intrinsics,
            },
        }
    return rows


def _task_inputs(
    *, candidate_root: Path, row: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    task_id = str(row.get("task_id") or "")
    freeze_path = _bound_absolute(row.get("task_freeze"), code="segment_repair_task_freeze_invalid")
    freeze = validate_task_freeze(_read(freeze_path, code="segment_repair_task_freeze_invalid"))
    if (
        not task_id
        or freeze.get("task_id") != task_id
        or freeze.get("task_freeze_digest") != row.get("task_freeze_digest")
    ):
        raise SegmentMaskRepairPreflightError(["segment_repair_task_freeze_invalid"])
    sweep_path = _bound_absolute(
        row.get("sweep_freeze"), code="segment_repair_sweep_freeze_invalid"
    )
    sweep = _read(sweep_path, code="segment_repair_sweep_freeze_invalid")
    scene = sweep.get("scene")
    publisher_scene_id = (
        str(scene.get("publisher_scene_id") or "") if isinstance(scene, Mapping) else ""
    )
    if (
        sweep.get("freeze_digest") != row.get("sweep_freeze", {}).get("freeze_digest")
        or sweep.get("freeze_digest") != canonical_digest(sweep, digest_field="freeze_digest")
        or sweep.get("learned_policy_outcomes_observed") is not False
        or sweep.get("replacement_usd_inserted") is not False
        or not publisher_scene_id
        or not isinstance(scene, Mapping)
        or scene.get("task_id") != task_id
    ):
        raise SegmentMaskRepairPreflightError(["segment_repair_sweep_freeze_invalid"])
    camera_path = _bound_absolute(
        sweep.get("camera_contract"), code="segment_repair_camera_contract_invalid"
    )
    cameras = _camera_rows(camera_path)
    mask_rows = sweep.get("masks")
    image_rows = sweep.get("source_images")
    if not isinstance(mask_rows, list) or not isinstance(image_rows, list):
        raise SegmentMaskRepairPreflightError(["segment_repair_camera_set_invalid"])
    masks = {
        str(item.get("camera_id") or ""): item for item in mask_rows if isinstance(item, Mapping)
    }
    images = {
        str(item.get("camera_id") or ""): item for item in image_rows if isinstance(item, Mapping)
    }
    if (
        not 2 <= len(cameras)
        or set(cameras) != set(masks)
        or set(cameras) != set(images)
        or len(masks) != len(mask_rows)
        or len(images) != len(image_rows)
    ):
        raise SegmentMaskRepairPreflightError(["segment_repair_camera_set_invalid"])
    camera_inputs: list[dict[str, Any]] = []
    mask_pixel_counts: list[int] = []
    for camera_id in sorted(cameras):
        mask_path = _bound_absolute(
            masks[camera_id].get("historical_outer_mask"),
            code="segment_repair_exact_segment_mask_invalid",
        )
        image_path = _bound_absolute(images[camera_id], code="segment_repair_source_frame_invalid")
        try:
            with Image.open(mask_path) as image:
                mask = image.convert("L")
                values = set(mask.tobytes())
                mask_size = mask.size
                pixel_count = sum(value > 0 for value in mask.tobytes())
            with Image.open(image_path) as image:
                image_size = image.size
        except (OSError, ValueError) as exc:
            raise SegmentMaskRepairPreflightError(
                ["segment_repair_frame_or_mask_unreadable"]
            ) from exc
        intrinsics = cameras[camera_id]["spec"]["intrinsics"]
        if (
            values != {0, 255}
            or pixel_count <= 0
            or mask_size != image_size
            or image_size != (intrinsics["width"], intrinsics["height"])
        ):
            raise SegmentMaskRepairPreflightError(["segment_repair_frame_or_mask_invalid"])
        mask_pixel_counts.append(pixel_count)
        camera_inputs.append(
            {
                "task_id": task_id,
                "camera_id": camera_id,
                "calibration": cameras[camera_id],
                "retained_scene_before": {"camera_id": camera_id, **_record(image_path)},
                "exact_residual_mask": {
                    "camera_id": camera_id,
                    **_record(mask_path),
                    "pixel_count": pixel_count,
                },
            }
        )
    removal = freeze["removal_plan"]
    lane = {
        "task_id": task_id,
        "task_freeze_digest": freeze["task_freeze_digest"],
        "removal_id": removal["removal_id"],
        "mask_set_id": removal["mask_set_id"],
        "replacement_asset_id": removal["replacement_asset_id"],
        "camera_count": len(camera_inputs),
        "segment_mask_pixel_count_minimum": min(mask_pixel_counts),
        "segment_mask_pixel_count_maximum": max(mask_pixel_counts),
        "source": {
            "candidate_set_root": str(candidate_root),
            "sweep_freeze": {
                **_record(sweep_path),
                "freeze_digest": sweep["freeze_digest"],
            },
            "camera_contract": _record(camera_path),
            "task_freeze": {
                **_record(freeze_path),
                "task_freeze_digest": freeze["task_freeze_digest"],
            },
        },
    }
    return lane, camera_inputs, publisher_scene_id


def materialize_segment_mask_repair_preflight(
    *,
    segment_cutout_set_path: str | Path,
    execution_authority_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one 1--5 task exact-segment, zero-dilation repair preflight."""

    candidate_path = _file(segment_cutout_set_path, code="segment_repair_candidate_set_missing")
    candidate = _read(candidate_path, code="segment_repair_candidate_set_unreadable")
    task_rows = candidate.get("task_candidates")
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA
        or candidate.get("receipt_digest")
        != canonical_digest(candidate, digest_field="receipt_digest")
        or not isinstance(task_rows, list)
        or not 1 <= len(task_rows) <= MAX_REPLACEMENT_OBJECTS
    ):
        raise SegmentMaskRepairPreflightError(["segment_repair_candidate_set_invalid"])
    authority_path = _file(
        execution_authority_path, code="segment_repair_execution_authority_missing"
    )
    authority = _read(authority_path, code="segment_repair_execution_authority_invalid")
    if (
        not str(authority.get("publisher_scene_id") or "")
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or authority.get("private_rights_admitted_scene_derived_uploads_authorized") is not True
        or authority.get("raw_interiorgs_upload_authorized") is not False
    ):
        raise SegmentMaskRepairPreflightError(["segment_repair_execution_authority_invalid"])
    shared = candidate.get("shared_scene_union")
    outputs = shared.get("outputs") if isinstance(shared, Mapping) else None
    counts = shared.get("counts") if isinstance(shared, Mapping) else None
    retained = _bound_relative(
        candidate_path.parent,
        outputs.get("retained_scene_gaussians") if isinstance(outputs, Mapping) else None,
        code="segment_repair_retained_splat_invalid",
    )
    retained_count = read_standard_3dgs_ply(retained).count
    if not isinstance(counts, Mapping) or counts.get("retained_total") != retained_count:
        raise SegmentMaskRepairPreflightError(["segment_repair_retained_splat_invalid"])
    lanes: list[dict[str, Any]] = []
    camera_inputs: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    publisher_scene_ids: set[str] = set()
    for row in task_rows:
        if not isinstance(row, Mapping):
            raise SegmentMaskRepairPreflightError(["segment_repair_task_invalid"])
        lane, cameras, publisher_scene_id = _task_inputs(
            candidate_root=candidate_path.parent, row=row
        )
        if lane["task_id"] in task_ids:
            raise SegmentMaskRepairPreflightError(["segment_repair_task_duplicate"])
        task_ids.add(lane["task_id"])
        publisher_scene_ids.add(publisher_scene_id)
        lanes.append(lane)
        camera_inputs.extend(cameras)
    if publisher_scene_ids != {str(authority["publisher_scene_id"])}:
        raise SegmentMaskRepairPreflightError(["segment_repair_execution_authority_scene_mismatch"])
    replacement_asset_ids = sorted(str(lane["replacement_asset_id"]) for lane in lanes)
    preflight: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_upload_no_execution",
        "segment_cutout_set": {
            **_record(candidate_path),
            "receipt_digest": candidate["receipt_digest"],
        },
        "backend_admission": {
            "execution_authority": {
                **_record(authority_path),
                "authority_digest": authority["authority_digest"],
            }
        },
        "shared_retained_scene": {
            **_record(retained),
            "retained_gaussian_count": retained_count,
            "all_replacements_co_present": True,
        },
        "replacement_object_count": len(lanes),
        "replacement_asset_ids": replacement_asset_ids,
        "lanes": sorted(lanes, key=lambda lane: str(lane["task_id"])),
        "camera_inputs": sorted(
            camera_inputs,
            key=lambda row: (str(row["task_id"]), str(row["camera_id"])),
        ),
        "repair_authority": {
            "generated_pixel_support": "historical_outer_object_segment_exact_binary_mask",
            "mask_dilation_pixels": 0,
            "full_deleted_gaussian_projection_is_diagnostic_only": True,
            "full_deleted_gaussian_projection_is_generated_edit_support": False,
            "canonical_source_frame_is_outside_mask_authority": True,
        },
        "required_result_checks": {
            "outside_mask_pixel_delta_required": 0,
            "locality_mask_dilation_pixels": 0,
            "exact_mask_composited_frames_retained": True,
            "multi_view_consistency_required": True,
            "source_object_absent_inside_segment_review_required": True,
        },
        "execution": {
            "private_derived_upload_performed": False,
            "provider_mutations_performed": 0,
            "aura_inpainting_executed": False,
            "artifixer3d_executed": False,
            "learned_policy_outcomes_accessed": False,
        },
        "claim_boundary": {
            "candidate_preflight_only": True,
            "source_gaussian_removal_qualified": False,
            "appearance_repair_qualified": False,
            "generated_pixels_are_observed_physical_evidence": False,
            "view_compositor_not_global_clean_3dgs": True,
        },
        "preflight_digest": "",
    }
    preflight["preflight_digest"] = canonical_digest(preflight, digest_field="preflight_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise SegmentMaskRepairPreflightError(["segment_repair_output_exists"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(preflight) + "\n", encoding="utf-8")
    return preflight



def main(argv: list[str] | None = None) -> int:
    """Materialize the calibrated preflight the ArtiFixer3D chain starts from.

    This is the root of the appearance path: the candidate-inputs receipt is
    derived from a calibrated preflight, and the only two producers are this one
    and the retired Aura residual preflight. With Aura retired and this having
    no entry point, the chain had no reachable root at all -- the head of the
    chain reported a missing input, several steps downstream of the actual gap.

    Performs no provider mutation and rents nothing.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize a calibrated segment mask repair preflight."
    )
    parser.add_argument("--segment-cutout-set", required=True)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        receipt = materialize_segment_mask_repair_preflight(
            segment_cutout_set_path=args.segment_cutout_set,
            execution_authority_path=args.execution_authority,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

__all__ = [
    "SCHEMA_VERSION",
    "SegmentMaskRepairPreflightError",
    "materialize_segment_mask_repair_preflight",
]
