"""Prepare a strict-mask AuraFusion360 residual-edit execution plan.

Aura's published InteriorGS example first trains a scene, removes an object,
asks SAM2 to expand masks, and then dilates them again.  None of those steps
is admissible for a residual packet: the scene is already the shared retained
layer and the exact per-camera residual masks are the only edit authority.

This module turns a 1--5 replacement residual packet into one *shared* Aura
preflight.  It deliberately emits a plan only: no model is loaded, no upload
is made, and no image is edited.  A future executor must consume this plan,
run a separately admitted exact-mask 2-D reference completion, invoke only
Aura's ``inpaint.py`` stage, and retain both native and exact-mask-composited
outputs for independent checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS
from .public_scene_residual_inpainting_packet import (
    BACKEND_ADMISSION_SCHEMA,
    PACKET_SCHEMA,
)


SCHEMA_VERSION = "public_scene_aura_exact_residual_preflight.v1"
SOURCE_RENDER_SCHEMA = "sealed_camera_render_manifest.v1"
BIG_LAMA_ARTIFACT_ID = "big_lama_author_linked_archive"
BIG_LAMA_RIGHTS_AUTHORITY_ID = "big_lama_apache_2_0"
BIG_LAMA_REPOSITORY = "https://github.com/advimman/lama"
BIG_LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
BIG_LAMA_TREE = "25f9902ca0c2ec4bf6c31c2b4427f0a4f05f2fd1"


class AuraExactResidualPreflightError(ValueError):
    """Stable errors for the no-upload direct-Aura preparation boundary."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    value = str(value or "")
    return len(value) == 71 and value.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in value[7:]
    )


def _file(path_value: Any, *, code: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise AuraExactResidualPreflightError([code])
    return path


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraExactResidualPreflightError([code]) from exc
    if not isinstance(value, dict):
        raise AuraExactResidualPreflightError([code])
    return value


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bound_file(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise AuraExactResidualPreflightError([code])
    path = _file(record.get("path"), code=code)
    if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
        raise AuraExactResidualPreflightError([code])
    return path


def _bound_relative(
    root: Path, record: Any, *, code: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise AuraExactResidualPreflightError([code])
    relative = str(record.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise AuraExactResidualPreflightError([code])
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise AuraExactResidualPreflightError([code])
    expected = record.get("sha256") if record.get("sha256") is not None else record.get("digest")
    if (
        record.get("sha256") is not None
        and record.get("digest") is not None
        and record.get("sha256") != record.get("digest")
    ):
        raise AuraExactResidualPreflightError([code])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != expected
    ):
        raise AuraExactResidualPreflightError([code])
    return path, {"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validated_packet(path: Path) -> dict[str, Any]:
    packet = _read(path, code="aura_exact_residual_packet_unreadable")
    if (
        packet.get("schema_version") != PACKET_SCHEMA
        or packet.get("status") != "exact_mask_contained_inpainting_input_packet_materialized"
        or packet.get("packet_digest") != canonical_digest(packet, digest_field="packet_digest")
        or not isinstance(packet.get("replacement_object_count"), int)
        or not 1 <= packet["replacement_object_count"] <= MAX_REPLACEMENT_OBJECTS
        or packet.get("replacement_object_count") != len(packet.get("lanes") or [])
        or packet.get("maximum_replacement_objects") != MAX_REPLACEMENT_OBJECTS
        or (packet.get("claim_boundary") or {}).get("released_code_inpainting_executed")
        is not False
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_packet_invalid"])
    return packet


def _validated_backend(packet: Mapping[str, Any]) -> dict[str, Any]:
    path = _bound_file(
        packet.get("backend_admission"), code="aura_exact_residual_backend_record_invalid"
    )
    backend = _read(path, code="aura_exact_residual_backend_unreadable")
    policy = backend.get("private_derived_upload_policy")
    if (
        backend.get("schema_version") != BACKEND_ADMISSION_SCHEMA
        or backend.get("status") != "rights_admitted_for_private_derived_inpainting"
        or backend.get("backend_id") != "aurafusion360_exact_residual_multiview"
        or backend.get("receipt_digest")
        != canonical_digest(backend, digest_field="receipt_digest")
        or backend.get("strict_exact_residual_masks_required") is not True
        or backend.get("mask_dilation_pixels") != 0
        or backend.get("outside_mask_pixel_delta_required") != 0
        or backend.get("multi_view_consistency_required") is not True
        or backend.get("execution_authorized") is not False
        or not isinstance(policy, Mapping)
        or policy.get("raw_dataset_bytes_upload") is not False
        or policy.get("private_derived_upload") is not True
        or policy.get("provider_training") is not False
        or policy.get("publication") is not False
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_backend_invalid"])
    return {
        **_record(path),
        "receipt_digest": backend["receipt_digest"],
        "backend_id": backend["backend_id"],
        "_backend": backend,
    }


def _big_lama_reference_completion(
    backend_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind just the Apache Big-LaMa checkpoint, not Inpaint360GS code/data."""

    backend = backend_record["_backend"]
    prerequisite_record = backend.get("method_prerequisite")
    prerequisite_path = _bound_file(
        prerequisite_record, code="aura_exact_residual_prerequisite_record_invalid"
    )
    prerequisite = _read(
        prerequisite_path, code="aura_exact_residual_prerequisite_unreadable"
    )
    if (
        prerequisite.get("schema_version") != "public_scene_method_prerequisite_receipt.v1"
        or prerequisite.get("receipt_digest")
        != canonical_digest(prerequisite, digest_field="receipt_digest")
        or not isinstance(prerequisite_record, Mapping)
        or prerequisite_record.get("receipt_digest") != prerequisite["receipt_digest"]
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_prerequisite_invalid"])
    method = (prerequisite.get("methods") or {}).get("inpaint360_author_smoke")
    if not isinstance(method, Mapping):
        raise AuraExactResidualPreflightError(["aura_exact_residual_big_lama_rights_missing"])
    artifacts = [
        row
        for row in method.get("artifacts") or []
        if isinstance(row, Mapping) and row.get("artifact_id") == BIG_LAMA_ARTIFACT_ID
    ]
    authorities = [
        row
        for row in method.get("rights_authorities") or []
        if isinstance(row, Mapping)
        and row.get("authority_id") == BIG_LAMA_RIGHTS_AUTHORITY_ID
    ]
    if len(artifacts) != 1 or len(authorities) != 1:
        raise AuraExactResidualPreflightError(["aura_exact_residual_big_lama_rights_missing"])
    artifact, authority = artifacts[0], authorities[0]
    if (
        artifact.get("rights_established") is not True
        or artifact.get("rights_authority_id") != BIG_LAMA_RIGHTS_AUTHORITY_ID
        or artifact.get("role") != "method_checkpoint"
        or not isinstance(artifact.get("size_bytes"), int)
        or artifact["size_bytes"] <= 0
        or not _digest(artifact.get("sha256"))
        or authority.get("established") is not True
        or authority.get("license_id") != "Apache-2.0"
        or authority.get("repository") != BIG_LAMA_REPOSITORY
        or authority.get("revision") != BIG_LAMA_COMMIT
        or authority.get("repository_tree") != BIG_LAMA_TREE
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_big_lama_rights_invalid"])
    relative = str(artifact.get("relative_path") or "")
    prerequisite_root = prerequisite_path.parent.parent
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise AuraExactResidualPreflightError(["aura_exact_residual_big_lama_archive_invalid"])
    archive = (prerequisite_root / relative).resolve()
    if (
        prerequisite_root != archive
        and prerequisite_root not in archive.parents
        or not archive.is_file()
        or archive.is_symlink()
        or archive.stat().st_size != artifact["size_bytes"]
        or _sha256(archive) != artifact["sha256"]
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_big_lama_archive_invalid"])
    return {
        "backend": "Big-LaMa",
        "role": "single_exact_mask_reference_completion_for_Aura_only",
        "checkpoint": {**_record(archive), "artifact_id": BIG_LAMA_ARTIFACT_ID},
        "rights_authority": {
            "authority_id": BIG_LAMA_RIGHTS_AUTHORITY_ID,
            "license_id": authority["license_id"],
            "repository": authority["repository"],
            "revision": authority["revision"],
            "repository_tree": authority["repository_tree"],
        },
        "method_prerequisite": {
            **_record(prerequisite_path),
            "receipt_digest": prerequisite["receipt_digest"],
        },
        "stock_inpaint360gs_code_or_author_data_used": False,
    }


def _render_manifest(record: Any, *, code: str) -> tuple[Path, dict[str, Any]]:
    path = _bound_file(record, code=code)
    value = _read(path, code=code)
    settings = value.get("render_settings")
    dimensions = settings.get("dimensions") if isinstance(settings, Mapping) else None
    if (
        value.get("schema_version") != SOURCE_RENDER_SCHEMA
        or value.get("status") != "rendered_exact_cameras"
        or value.get("sealed_camera_render_manifest_digest")
        != canonical_digest(value, digest_field="sealed_camera_render_manifest_digest")
        or not isinstance(dimensions, Mapping)
        or not isinstance(dimensions.get("width"), int)
        or not isinstance(dimensions.get("height"), int)
        or dimensions["width"] <= 0
        or dimensions["height"] <= 0
    ):
        raise AuraExactResidualPreflightError([code])
    return path, value


def _camera_frames(
    *, manifest_path: Path, manifest: Mapping[str, Any], code: str
) -> dict[str, dict[str, Any]]:
    rows = manifest.get("renders")
    settings = manifest["render_settings"]
    dimensions = settings["dimensions"]
    if not isinstance(rows, list) or not rows:
        raise AuraExactResidualPreflightError([code])
    frames: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AuraExactResidualPreflightError([code])
        camera_id = str(row.get("camera_id") or "")
        if not camera_id or camera_id in frames:
            raise AuraExactResidualPreflightError([code])
        path, bound = _bound_relative(manifest_path.parent, row, code=code)
        try:
            with Image.open(path) as image:
                if image.size != (dimensions["width"], dimensions["height"]):
                    raise AuraExactResidualPreflightError([code])
        except (OSError, ValueError) as exc:
            if isinstance(exc, AuraExactResidualPreflightError):
                raise
            raise AuraExactResidualPreflightError([code]) from exc
        frames[camera_id] = {"camera_id": camera_id, **bound}
    return frames


def _lane(
    *, packet_path: Path, row: Any, expected_asset_ids: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lane_path = _bound_file(row, code="aura_exact_residual_lane_record_invalid")
    lane = _read(lane_path, code="aura_exact_residual_lane_unreadable")
    if (
        lane.get("lane_digest") != canonical_digest(lane, digest_field="lane_digest")
        or lane.get("inpainting_execution_authorized") is not False
        or lane.get("inpainting_result_qualified") is not False
        or sorted(lane.get("co_present_replacement_asset_ids") or []) != expected_asset_ids
        or not str(lane.get("task_id") or "")
        or not str(lane.get("removal_id") or "")
        or not str(lane.get("mask_set_id") or "")
        or str(lane.get("replacement_asset_id") or "") not in expected_asset_ids
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_lane_invalid"])
    coverage_path = _bound_file(
        lane.get("coverage_audit"), code="aura_exact_residual_coverage_record_invalid"
    )
    coverage = _read(coverage_path, code="aura_exact_residual_coverage_unreadable")
    if (
        coverage.get("manifest_digest") != canonical_digest(coverage, digest_field="manifest_digest")
        or coverage.get("task_id") != lane["task_id"]
        or coverage.get("task_freeze_digest") != lane.get("task_freeze_digest")
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_coverage_invalid"])
    retained_path, retained = _render_manifest(
        lane.get("retained_scene_render"), code="aura_exact_residual_retained_render_invalid"
    )
    retained_frames = _camera_frames(
        manifest_path=retained_path, manifest=retained, code="aura_exact_residual_retained_frame_invalid"
    )
    black_path, black = _render_manifest(
        lane.get("source_layer_black_render"), code="aura_exact_residual_source_black_invalid"
    )
    white_path, white = _render_manifest(
        lane.get("source_layer_white_render"), code="aura_exact_residual_source_white_invalid"
    )
    black_frames = _camera_frames(
        manifest_path=black_path, manifest=black, code="aura_exact_residual_source_black_frame_invalid"
    )
    white_frames = _camera_frames(
        manifest_path=white_path, manifest=white, code="aura_exact_residual_source_white_frame_invalid"
    )
    masks = lane.get("exact_residual_masks")
    if not isinstance(masks, list) or not masks:
        raise AuraExactResidualPreflightError(["aura_exact_residual_masks_missing"])
    mask_records: dict[str, dict[str, Any]] = {}
    for mask in masks:
        if not isinstance(mask, Mapping):
            raise AuraExactResidualPreflightError(["aura_exact_residual_mask_invalid"])
        camera_id = str(mask.get("camera_id") or "")
        path, bound = _bound_relative(
            coverage_path.parent, mask, code="aura_exact_residual_mask_invalid"
        )
        try:
            with Image.open(path) as image:
                pixels = image.convert("L")
                values = set(pixels.tobytes())
                if (
                    pixels.size != (
                        retained["render_settings"]["dimensions"]["width"],
                        retained["render_settings"]["dimensions"]["height"],
                    )
                    or not values.issubset({0, 255})
                    or values != {0, 255}
                    or int(sum(value > 0 for value in pixels.tobytes()))
                    != mask.get("pixel_count")
                ):
                    raise AuraExactResidualPreflightError(["aura_exact_residual_mask_invalid"])
        except (OSError, ValueError) as exc:
            if isinstance(exc, AuraExactResidualPreflightError):
                raise
            raise AuraExactResidualPreflightError(["aura_exact_residual_mask_invalid"]) from exc
        if not camera_id or camera_id in mask_records:
            raise AuraExactResidualPreflightError(["aura_exact_residual_mask_invalid"])
        mask_records[camera_id] = {"camera_id": camera_id, **bound, "pixel_count": mask["pixel_count"]}
    if not (
        set(mask_records) == set(retained_frames) == set(black_frames) == set(white_frames)
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_camera_set_mismatch"])
    planned = []
    for camera_id in sorted(mask_records):
        planned.append(
            {
                "camera_id": camera_id,
                "retained_scene_before": {
                    "camera_id": camera_id,
                    **_record(
                        retained_path.parent
                        / retained_frames[camera_id]["relative_path"]
                    ),
                },
                "exact_residual_mask": {
                    "camera_id": camera_id,
                    **_record(
                        coverage_path.parent
                        / mask_records[camera_id]["relative_path"]
                    ),
                    "pixel_count": mask_records[camera_id]["pixel_count"],
                },
                "deleted_source_black": {
                    "camera_id": camera_id,
                    **_record(
                        black_path.parent / black_frames[camera_id]["relative_path"]
                    ),
                },
                "deleted_source_white": {
                    "camera_id": camera_id,
                    **_record(
                        white_path.parent / white_frames[camera_id]["relative_path"]
                    ),
                },
            }
        )
    lane_record = {
        "task_id": lane["task_id"],
        "task_freeze_digest": lane["task_freeze_digest"],
        "removal_id": lane["removal_id"],
        "mask_set_id": lane["mask_set_id"],
        "replacement_asset_id": lane["replacement_asset_id"],
        "co_present_replacement_asset_ids": expected_asset_ids,
        "lane": {**_record(lane_path), "lane_digest": lane["lane_digest"]},
        "retained_render": {
            **_record(retained_path),
            "sealed_camera_render_manifest_digest": retained[
                "sealed_camera_render_manifest_digest"
            ],
        },
        "source_black_render": {
            **_record(black_path),
            "sealed_camera_render_manifest_digest": black[
                "sealed_camera_render_manifest_digest"
            ],
        },
        "source_white_render": {
            **_record(white_path),
            "sealed_camera_render_manifest_digest": white[
                "sealed_camera_render_manifest_digest"
            ],
        },
        "camera_count": len(planned),
    }
    return lane_record, planned


def _lane_asset_id(row: Any) -> str:
    """Read only the signed lane identity needed to build the shared set."""

    lane_path = _bound_file(row, code="aura_exact_residual_lane_record_invalid")
    lane = _read(lane_path, code="aura_exact_residual_lane_unreadable")
    asset_id = str(lane.get("replacement_asset_id") or "")
    if (
        lane.get("lane_digest") != canonical_digest(lane, digest_field="lane_digest")
        or not asset_id
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_lane_invalid"])
    return asset_id


def materialize_aura_exact_residual_preflight(
    *, input_packet_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Validate a residual packet and seal one no-mutation direct-Aura plan."""

    packet_path = _file(input_packet_path, code="aura_exact_residual_packet_missing")
    packet = _validated_packet(packet_path)
    backend = _validated_backend(packet)
    big_lama = _big_lama_reference_completion(backend)
    backend = {key: value for key, value in backend.items() if key != "_backend"}
    candidate_set_path = _bound_file(
        packet.get("candidate_set"), code="aura_exact_residual_candidate_set_invalid"
    )
    candidate_set = _read(
        candidate_set_path, code="aura_exact_residual_candidate_set_unreadable"
    )
    if candidate_set.get("schema_version") not in {
        "adp009b_direct_evidence_expansion_set.v1",
        "adp009b_ownership_coverage_cutout_set.v1",
    }:
        raise AuraExactResidualPreflightError(["aura_exact_residual_candidate_set_invalid"])
    shared = packet.get("shared_retained_scene")
    if not isinstance(shared, Mapping) or not _digest(shared.get("sha256")) or not isinstance(
        shared.get("retained_gaussian_count"), int
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_shared_scene_invalid"])
    candidate_root = candidate_set_path.parent
    shared_path = (candidate_root / str(shared.get("relative_path") or "")).resolve()
    if candidate_root != shared_path and candidate_root not in shared_path.parents:
        raise AuraExactResidualPreflightError(["aura_exact_residual_shared_scene_invalid"])
    if (
        not shared_path.is_file()
        or shared_path.is_symlink()
        or shared_path.stat().st_size != shared.get("size_bytes")
        or _sha256(shared_path) != shared.get("sha256")
        or shared["retained_gaussian_count"] <= 0
    ):
        raise AuraExactResidualPreflightError(["aura_exact_residual_shared_scene_invalid"])
    expected_asset_ids = sorted(_lane_asset_id(row) for row in packet["lanes"])
    if not all(expected_asset_ids) or len(set(expected_asset_ids)) != len(expected_asset_ids):
        raise AuraExactResidualPreflightError(["aura_exact_residual_replacement_set_invalid"])
    lanes: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    for row in packet["lanes"]:
        lane, lane_cameras = _lane(
            packet_path=packet_path, row=row, expected_asset_ids=expected_asset_ids
        )
        lanes.append(lane)
        for camera in lane_cameras:
            cameras.append({"task_id": lane["task_id"], **camera})
    if not cameras:
        raise AuraExactResidualPreflightError(["aura_exact_residual_camera_set_mismatch"])
    selected = sorted(
        cameras,
        key=lambda row: (-int(row["exact_residual_mask"]["pixel_count"]), row["task_id"], row["camera_id"]),
    )[0]
    preflight: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_upload_no_execution",
        "input_packet": {**_record(packet_path), "packet_digest": packet["packet_digest"]},
        "backend_admission": backend,
        "shared_retained_scene": {
            **_record(shared_path),
            "retained_gaussian_count": shared["retained_gaussian_count"],
            "all_replacements_co_present": True,
        },
        "replacement_object_count": packet["replacement_object_count"],
        "replacement_asset_ids": expected_asset_ids,
        "lanes": lanes,
        "camera_inputs": cameras,
        "reference_completion": {
            "reference_selection": "largest_exact_residual_mask_pixel_count_then_task_and_camera_id",
            "selected_task_id": selected["task_id"],
            "selected_camera_id": selected["camera_id"],
            "input_before": selected["retained_scene_before"],
            "input_exact_mask": selected["exact_residual_mask"],
            "required_backend": big_lama["backend"],
            "backend_provenance": big_lama,
            "stock_inpaint360gs_not_invoked": True,
            "mask_dilation_pixels": 0,
            "output_must_be_exact_mask_composited": True,
        },
        "aura_workflow": {
            "released_entrypoint": "inpaint.py",
            "initial_shared_gaussian_iteration": "shared_retained",
            "excluded_stock_stages": ["train.py", "remove.py", "utils/sam2_utils.py"],
            "exact_mask_materialization": "hardlink_or_byte_verified_copy_only",
            "inpaint_dilate_mask_iter": 0,
            "inpaint_dilate_mask_kernel_size": 1,
            "agdd_dilate_iter": 0,
            "agdd_kernel_size": 1,
            "inpaint_init_finetune_iteration": -1,
            "multiview_scene_camera_count": len(cameras),
        },
        "required_result_checks": {
            "raw_native_aura_frames_retained": True,
            "exact_mask_composited_frames_retained": True,
            "outside_mask_pixel_delta_required": 0,
            "locality_mask_dilation_pixels": 0,
            "multi_view_consistency_required": True,
            "native_output_is_not_automatically_a_qualified_shared_scene": True,
        },
        "execution": {
            "private_derived_upload_performed": False,
            "provider_mutations_performed": 0,
            "aura_inpainting_executed": False,
            "learned_policy_outcomes_accessed": False,
        },
        "claim_boundary": {
            "raw_dataset_bytes_upload_authorized": False,
            "inpaint360gs_executed": False,
            "inpainting_result_qualified": False,
            "source_gaussian_removal_qualified": False,
            "native_simulator_import_qualified": False,
        },
        "preflight_digest": "",
    }
    preflight["preflight_digest"] = canonical_digest(
        preflight, digest_field="preflight_digest"
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(preflight) + "\n", encoding="utf-8")
    return preflight


__all__ = [
    "AuraExactResidualPreflightError",
    "SCHEMA_VERSION",
    "materialize_aura_exact_residual_preflight",
]
