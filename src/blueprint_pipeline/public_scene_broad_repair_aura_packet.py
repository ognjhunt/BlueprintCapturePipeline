"""Bridge broad deleted-projection support into the strict Aura input contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS, validate_task_freeze
from .public_scene_broad_repair_support import PACKET_SCHEMA as BROAD_SUPPORT_SCHEMA
from .public_scene_residual_inpainting_packet import (
    BACKEND_ADMISSION_SCHEMA,
    PACKET_SCHEMA,
)


class BroadRepairAuraPacketError(ValueError):
    """Stable fail-closed errors for the broad-repair Aura bridge."""

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
        raise BroadRepairAuraPacketError([code]) from exc
    if not isinstance(value, dict):
        raise BroadRepairAuraPacketError([code])
    return value


def _file(value: Any, *, code: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise BroadRepairAuraPacketError([code])
    return path


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bound(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise BroadRepairAuraPacketError([code])
    path = _file(record.get("path"), code=code)
    if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != _sha256(path):
        raise BroadRepairAuraPacketError([code])
    return path


def _relative(root: Path, record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise BroadRepairAuraPacketError([code])
    relative = str(record.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise BroadRepairAuraPacketError([code])
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise BroadRepairAuraPacketError([code])
    if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != _sha256(path):
        raise BroadRepairAuraPacketError([code])
    return path


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _manifest_frames(path: Path) -> dict[str, dict[str, Any]]:
    manifest = _read(path, code="broad_repair_aura_render_manifest_unreadable")
    if (
        manifest.get("schema_version") != "sealed_camera_render_manifest.v1"
        or manifest.get("status") != "rendered_exact_cameras"
        or manifest.get("sealed_camera_render_manifest_digest")
        != canonical_digest(manifest, digest_field="sealed_camera_render_manifest_digest")
    ):
        raise BroadRepairAuraPacketError(["broad_repair_aura_render_manifest_invalid"])
    frames: dict[str, dict[str, Any]] = {}
    for row in manifest.get("renders") or []:
        if not isinstance(row, Mapping):
            raise BroadRepairAuraPacketError(["broad_repair_aura_render_frame_invalid"])
        camera_id = str(row.get("camera_id") or "")
        relative = str(row.get("relative_path") or "")
        frame = (path.parent / relative).resolve()
        if (
            not camera_id
            or camera_id in frames
            or path.parent.resolve() not in frame.parents
            or not frame.is_file()
            or frame.is_symlink()
            or row.get("size_bytes") != frame.stat().st_size
            or row.get("digest") != _sha256(frame)
        ):
            raise BroadRepairAuraPacketError(["broad_repair_aura_render_frame_invalid"])
        frames[camera_id] = {
            "camera_id": camera_id,
            "relative_path": relative,
            "size_bytes": frame.stat().st_size,
            "sha256": _sha256(frame),
        }
    if not frames:
        raise BroadRepairAuraPacketError(["broad_repair_aura_render_frame_invalid"])
    return frames


def materialize_broad_repair_aura_packet(
    *,
    broad_support_packet_path: str | Path,
    backend_admission_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create a standard strict-mask packet without editing or uploading bytes."""

    support_path = _file(
        broad_support_packet_path,
        code="broad_repair_aura_support_packet_missing",
    )
    support = _read(support_path, code="broad_repair_aura_support_packet_unreadable")
    if (
        support.get("schema_version") != BROAD_SUPPORT_SCHEMA
        or support.get("status") != "full_deleted_projection_repair_support_materialized"
        or support.get("receipt_digest")
        != canonical_digest(support, digest_field="receipt_digest")
        or (support.get("claim_boundary") or {}).get(
            "all_detectable_deleted_projection_inside_repair_support"
        )
        is not True
        or (support.get("claim_boundary") or {}).get("outside_repair_support_change_authorized")
        is not False
    ):
        raise BroadRepairAuraPacketError(["broad_repair_aura_support_packet_invalid"])
    candidate_path = _bound(
        support.get("candidate_set"),
        code="broad_repair_aura_candidate_invalid",
    )
    candidate = _read(candidate_path, code="broad_repair_aura_candidate_invalid")
    if (
        candidate.get("receipt_digest")
        != canonical_digest(candidate, digest_field="receipt_digest")
        or candidate.get("receipt_digest") != support["candidate_set"].get("receipt_digest")
    ):
        raise BroadRepairAuraPacketError(["broad_repair_aura_candidate_invalid"])
    backend_path = _file(backend_admission_path, code="broad_repair_aura_backend_missing")
    backend = _read(backend_path, code="broad_repair_aura_backend_unreadable")
    if (
        backend.get("schema_version") != BACKEND_ADMISSION_SCHEMA
        or backend.get("status") != "rights_admitted_for_private_derived_inpainting"
        or backend.get("backend_id") != "aurafusion360_exact_residual_multiview"
        or backend.get("receipt_digest")
        != canonical_digest(backend, digest_field="receipt_digest")
    ):
        raise BroadRepairAuraPacketError(["broad_repair_aura_backend_invalid"])
    task_rows = candidate.get("task_candidates")
    support_lanes = support.get("task_lanes")
    if (
        not isinstance(task_rows, list)
        or not isinstance(support_lanes, list)
        or not 1 <= len(task_rows) <= MAX_REPLACEMENT_OBJECTS
        or len(task_rows) != len(support_lanes)
    ):
        raise BroadRepairAuraPacketError(["broad_repair_aura_task_set_invalid"])
    tasks: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        if not isinstance(row, Mapping):
            raise BroadRepairAuraPacketError(["broad_repair_aura_task_invalid"])
        task_id = str(row.get("task_id") or "")
        freeze_path = _bound(row.get("task_freeze"), code="broad_repair_aura_task_freeze_invalid")
        freeze = validate_task_freeze(
            _read(freeze_path, code="broad_repair_aura_task_freeze_invalid")
        )
        if (
            not task_id
            or task_id in tasks
            or freeze.get("task_id") != task_id
            or freeze.get("task_freeze_digest") != row.get("task_freeze_digest")
        ):
            raise BroadRepairAuraPacketError(["broad_repair_aura_task_freeze_invalid"])
        tasks[task_id] = freeze
    lane_by_task = {
        str(row.get("task_id") or ""): row
        for row in support_lanes
        if isinstance(row, Mapping)
    }
    if set(lane_by_task) != set(tasks):
        raise BroadRepairAuraPacketError(["broad_repair_aura_task_set_invalid"])
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise BroadRepairAuraPacketError(["broad_repair_aura_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    replacement_asset_ids = sorted(
        str(tasks[task_id]["removal_plan"]["replacement_asset_id"])
        for task_id in tasks
    )
    lane_records: list[dict[str, Any]] = []
    for slot, task_id in enumerate(sorted(tasks), start=1):
        support_lane = lane_by_task[task_id]
        freeze = tasks[task_id]
        removal = freeze["removal_plan"]
        black_manifest = _bound(
            support_lane.get("deleted_source_black_manifest"),
            code="broad_repair_aura_black_manifest_invalid",
        )
        white_manifest = _bound(
            support_lane.get("deleted_source_white_manifest"),
            code="broad_repair_aura_white_manifest_invalid",
        )
        retained_manifest = _bound(
            support_lane.get("retained_scene_manifest"),
            code="broad_repair_aura_retained_manifest_invalid",
        )
        black_frames = _manifest_frames(black_manifest)
        white_frames = _manifest_frames(white_manifest)
        retained_frames = _manifest_frames(retained_manifest)
        if not (set(black_frames) == set(white_frames) == set(retained_frames)):
            raise BroadRepairAuraPacketError(["broad_repair_aura_camera_set_invalid"])
        frame_rows = support_lane.get("frames")
        if not isinstance(frame_rows, list):
            raise BroadRepairAuraPacketError(["broad_repair_aura_support_frames_invalid"])
        support_frames = {
            str(row.get("camera_id") or ""): row
            for row in frame_rows
            if isinstance(row, Mapping)
        }
        if set(support_frames) != set(black_frames):
            raise BroadRepairAuraPacketError(["broad_repair_aura_camera_set_invalid"])
        lane_root = output / f"lane_{slot:02d}"
        masks: list[dict[str, Any]] = []
        retained_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        for camera_id in sorted(support_frames):
            row = support_frames[camera_id]
            source_mask = _relative(
                support_path.parent,
                row.get("repair_support_mask"),
                code="broad_repair_aura_mask_invalid",
            )
            mask = lane_root / "uncovered_source_support_masks" / f"{camera_id}.png"
            _link(source_mask, mask)
            masks.append(
                {
                    **{
                        "relative_path": mask.relative_to(lane_root).as_posix(),
                        "size_bytes": mask.stat().st_size,
                        "sha256": _sha256(mask),
                    },
                    "camera_id": camera_id,
                    "pixel_count": row.get("repair_support_pixel_count"),
                    "derived_from_all_state_cells": 1,
                }
            )
            retained_rows.append(retained_frames[camera_id])
            pair_rows.append(
                {
                    "camera_id": camera_id,
                    "black_background": {
                        key: value for key, value in black_frames[camera_id].items() if key != "camera_id"
                    },
                    "white_background": {
                        key: value for key, value in white_frames[camera_id].items() if key != "camera_id"
                    },
                }
            )
        coverage: dict[str, Any] = {
            "schema_version": "public_scene_broad_repair_support_lane.v1",
            "status": "full_deleted_projection_is_repair_authority",
            "task_id": task_id,
            "task_freeze_digest": freeze["task_freeze_digest"],
            "broad_support_packet_receipt_digest": support["receipt_digest"],
            "repair_support_masks": masks,
            "manifest_digest": "",
        }
        coverage["manifest_digest"] = canonical_digest(coverage, digest_field="manifest_digest")
        coverage_path = lane_root / "public_scene_broad_repair_support_lane.v1.json"
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path.write_text(canonical_json(coverage) + "\n", encoding="utf-8")
        lane: dict[str, Any] = {
            "task_id": task_id,
            "task_freeze_digest": freeze["task_freeze_digest"],
            "removal_id": removal["removal_id"],
            "mask_set_id": removal["mask_set_id"],
            "replacement_asset_id": removal["replacement_asset_id"],
            "co_present_replacement_asset_ids": replacement_asset_ids,
            "coverage_audit": {
                **_record(coverage_path),
                "manifest_digest": coverage["manifest_digest"],
            },
            "retained_scene_render": {
                **_record(retained_manifest),
                "sealed_camera_render_manifest_digest": support_lane[
                    "retained_scene_manifest"
                ]["sealed_camera_render_manifest_digest"],
            },
            "source_layer_black_render": {
                **_record(black_manifest),
                "sealed_camera_render_manifest_digest": support_lane[
                    "deleted_source_black_manifest"
                ]["sealed_camera_render_manifest_digest"],
            },
            "source_layer_white_render": {
                **_record(white_manifest),
                "sealed_camera_render_manifest_digest": support_lane[
                    "deleted_source_white_manifest"
                ]["sealed_camera_render_manifest_digest"],
            },
            "exact_residual_masks": masks,
            "retained_scene_frames": retained_rows,
            "source_layer_black_white_frames": pair_rows,
            "inpainting_execution_authorized": False,
            "inpainting_result_qualified": False,
            "lane_digest": "",
        }
        lane["lane_digest"] = canonical_digest(lane, digest_field="lane_digest")
        lane_path = lane_root / "residual_inpainting_input_lane.v1.json"
        lane_path.write_text(canonical_json(lane) + "\n", encoding="utf-8")
        lane_records.append(
            {
                **_record(lane_path),
                "lane_digest": lane["lane_digest"],
                "source_layer_black_white_frames": pair_rows,
            }
        )
    shared = candidate["shared_scene_union"]
    retained_record = shared["outputs"]["retained_scene_gaussians"]
    packet: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA,
        "status": "exact_mask_contained_inpainting_input_packet_materialized",
        "replacement_object_count": len(lane_records),
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "candidate_set": {
            **_record(candidate_path),
            "receipt_digest": candidate["receipt_digest"],
        },
        "backend_admission": {
            **_record(backend_path),
            "receipt_digest": backend["receipt_digest"],
        },
        "shared_retained_scene": {
            **retained_record,
            "retained_gaussian_count": shared["counts"]["retained_total"],
            "all_replacements_co_present": True,
        },
        "lanes": lane_records,
        "broad_repair_support": {
            **_record(support_path),
            "receipt_digest": support["receipt_digest"],
        },
        "claim_boundary": {
            "released_code_inpainting_executed": False,
            "inpainting_result_qualified": False,
            "source_gaussian_removal_qualified": False,
            "outside_mask_locality_measured": False,
            "private_derived_upload_not_yet_performed": True,
            "raw_dataset_bytes_upload_authorized": False,
            "native_simulator_import_qualified": False,
            "repair_support_includes_all_detectable_deleted_projection": True,
            "repaired_pixels_are_observed_physical_evidence": False,
        },
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = output / "public_scene_residual_inpainting_input_packet.v1.json"
    packet_path.write_text(canonical_json(packet) + "\n", encoding="utf-8")
    return packet


__all__ = ["BroadRepairAuraPacketError", "materialize_broad_repair_aura_packet"]
