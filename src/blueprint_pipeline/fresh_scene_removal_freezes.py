"""Materialize the deterministic mask-to-Gaussian-removal freeze handoff.

One digest-bound request drives the existing excision and all-camera segment
sweep materializers for one to five task objects.  The caller cannot supply
replacement masks, camera rows, or Gaussian indices: those are reopened from
the reviewed calibrated-mask receipt and produced by the existing deterministic
modules.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_gaussian_excision_audit import (
    FREEZE_SCHEMA,
    materialize_excision_audit_freeze,
)
from .public_scene_segment_contribution_cutout import (
    materialize_segment_contribution_sweep_freeze,
)


REQUEST_SCHEMA_VERSION = "fresh_scene_removal_freeze_tool_request.v1"
RECEIPT_SCHEMA_VERSION = "fresh_scene_removal_freeze_set.v1"


class FreshSceneRemovalFreezeError(ValueError):
    """The reviewed mask-to-removal-freeze handoff is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    import json

    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSceneRemovalFreezeError(code) from exc
    if unresolved.is_symlink() or not resolved.is_file() or not isinstance(value, dict):
        raise FreshSceneRemovalFreezeError(code)
    return resolved, value


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }
    if root is None:
        record["path"] = str(resolved)
    else:
        record["relative_path"] = resolved.relative_to(root.resolve()).as_posix()
    return record


def _verified_receipt_relative(*, receipt_root: Path, record: object, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise FreshSceneRemovalFreezeError(code)
    relative = str(record.get("relative_path") or "")
    path = (receipt_root / relative).resolve()
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or receipt_root.resolve() not in path.parents
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise FreshSceneRemovalFreezeError(code)
    return path


def _verified_receipt_record(*, receipt_root: Path, record: object, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise FreshSceneRemovalFreezeError(code)
    if record.get("relative_path"):
        return _verified_receipt_relative(receipt_root=receipt_root, record=record, code=code)
    unresolved = Path(str(record.get("path") or "")).expanduser()
    path = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise FreshSceneRemovalFreezeError(code)
    return path


def materialize_fresh_scene_removal_freezes(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Build per-task excision and all-camera segment freezes, without spend."""

    value = dict(request)
    if value.get("schema_version") != REQUEST_SCHEMA_VERSION or value.get(
        "request_digest"
    ) != canonical_digest(value, digest_field="request_digest"):
        raise FreshSceneRemovalFreezeError("fresh_scene_removal_request_invalid")
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise FreshSceneRemovalFreezeError("fresh_scene_removal_output_not_empty")
    source_splat = Path(str(value.get("source_standard_splat_path") or "")).expanduser().resolve()
    source_collision = Path(str(value.get("source_collision_path") or "")).expanduser().resolve()
    registered_frame = (
        Path(str(value.get("registered_frame_receipt_path") or "")).expanduser().resolve()
    )
    for path in (source_splat, source_collision, registered_frame):
        if path.is_symlink() or not path.is_file():
            raise FreshSceneRemovalFreezeError("fresh_scene_removal_source_missing")
    mask_receipt_path, mask_receipt = _read(
        str(value.get("calibrated_mask_set_receipt_path") or ""),
        code="fresh_scene_removal_mask_receipt_invalid",
    )
    if (
        mask_receipt.get("schema_version") != "public_scene_calibrated_object_mask_set.v1"
        or mask_receipt.get("status")
        != "calibrated_inferred_object_masks_materialized_pending_review"
        or mask_receipt.get("receipt_digest")
        != canonical_digest(mask_receipt, digest_field="receipt_digest")
        or not (
            (mask_receipt.get("selection_authority") or {}).get(
                "all_selected_tracks_review_accepted"
            )
            is True
            and (mask_receipt.get("selection_authority") or {}).get("reviewer_kind")
            in {"human", "ai"}
            or (mask_receipt.get("selection_authority") or {}).get(
                "all_selected_tracks_human_review_accepted"
            )
            is True
        )
    ):
        raise FreshSceneRemovalFreezeError("fresh_scene_removal_mask_receipt_invalid")
    task_rows = mask_receipt.get("tasks")
    task_requests = value.get("tasks")
    if not isinstance(task_rows, list) or not isinstance(task_requests, Mapping):
        raise FreshSceneRemovalFreezeError("fresh_scene_removal_task_set_invalid")
    masks_by_task = {
        str(row.get("task_id") or ""): row for row in task_rows if isinstance(row, Mapping)
    }
    if (
        not 1 <= len(masks_by_task) <= 5
        or "" in masks_by_task
        or set(masks_by_task) != set(task_requests)
        or mask_receipt.get("task_count") != len(masks_by_task)
    ):
        raise FreshSceneRemovalFreezeError("fresh_scene_removal_task_set_invalid")

    output.mkdir(parents=True)
    output_tasks: list[dict[str, Any]] = []
    for task_id in sorted(masks_by_task):
        mask_row = masks_by_task[task_id]
        task = task_requests[task_id]
        if not isinstance(task, Mapping):
            raise FreshSceneRemovalFreezeError("fresh_scene_removal_task_request_invalid")
        receipt_root = mask_receipt_path.parent
        task_freeze_path = _verified_receipt_record(
            receipt_root=receipt_root,
            record=mask_row.get("task_freeze"),
            code="fresh_scene_removal_task_freeze_binding_invalid",
        )
        _verified_receipt_record(
            receipt_root=receipt_root,
            record=mask_row.get("source_track_result"),
            code="fresh_scene_removal_source_track_binding_invalid",
        )
        _task_freeze_path, task_freeze = _read(
            task_freeze_path, code="fresh_scene_removal_task_freeze_binding_invalid"
        )
        task_freeze_record = mask_row["task_freeze"]
        if (
            task_freeze.get("task_id") != task_id
            or task_freeze.get("task_freeze_digest") != task_freeze_record.get("task_freeze_digest")
            or task_freeze.get("task_freeze_digest")
            != canonical_digest(task_freeze, digest_field="task_freeze_digest")
        ):
            raise FreshSceneRemovalFreezeError("fresh_scene_removal_task_freeze_binding_invalid")
        camera_path = _verified_receipt_relative(
            receipt_root=receipt_root,
            record=mask_row.get("camera_contract"),
            code="fresh_scene_removal_camera_binding_invalid",
        )
        image_root = Path(str(mask_row.get("source_images_root") or "")).resolve()
        mask_root = Path(str(mask_row.get("mask_root") or "")).resolve()
        expected_root = (receipt_root / "tasks" / task_id).resolve()
        if (
            image_root != expected_root / "images"
            or mask_root != expected_root / "masks"
            or image_root.is_symlink()
            or mask_root.is_symlink()
            or not image_root.is_dir()
            or not mask_root.is_dir()
        ):
            raise FreshSceneRemovalFreezeError("fresh_scene_removal_mask_root_invalid")
        for row in mask_row.get("source_images") or []:
            _verified_receipt_relative(
                receipt_root=receipt_root,
                record=row.get("image") if isinstance(row, Mapping) else None,
                code="fresh_scene_removal_image_binding_invalid",
            )
        for row in mask_row.get("masks") or []:
            _verified_receipt_relative(
                receipt_root=receipt_root,
                record=row.get("mask") if isinstance(row, Mapping) else None,
                code="fresh_scene_removal_mask_binding_invalid",
            )
        task_root = output / "tasks" / task_id
        excision_root = task_root / "excision_freeze"
        freeze = materialize_excision_audit_freeze(
            source_standard_splat_path=source_splat,
            source_collision_path=source_collision,
            target_collision_prim_path=str(task.get("target_collision_prim_path") or ""),
            registered_frame_receipt_path=registered_frame,
            camera_contract_path=camera_path,
            source_image_root=image_root,
            historical_outer_mask_root=mask_root,
            scene=dict(task.get("scene") or {}),
            policy=dict(task.get("policy") or {}),
            historical_baseline=dict(task.get("historical_baseline") or {}),
            output_root=excision_root,
            supersample=int(task.get("supersample", 2)),
            render_input_receipt_path=task.get("render_input_receipt_path"),
            adp_item=str(task.get("adp_item") or "ADP-009D"),
        )
        freeze_path = excision_root / f"{FREEZE_SCHEMA}.json"
        sweep_root = task_root / "segment_sweep_freeze"
        sweep = materialize_segment_contribution_sweep_freeze(
            excision_freeze_path=freeze_path, output_root=sweep_root
        )
        sweep_path = sweep_root / f"{FREEZE_SCHEMA}.json"
        output_tasks.append(
            {
                "task_id": task_id,
                "excision_freeze": {
                    **_record(freeze_path, root=output),
                    "freeze_digest": freeze["freeze_digest"],
                },
                "segment_sweep_freeze": {
                    **_record(sweep_path, root=output),
                    "freeze_digest": sweep["freeze_digest"],
                },
                "camera_count": int(sweep["camera_split"]["camera_count"]),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "excision_and_segment_sweep_freezes_materialized_no_execution",
        "task_count": len(output_tasks),
        "source_standard_splat": _record(source_splat),
        "source_collision": _record(source_collision),
        "registered_frame_receipt": _record(registered_frame),
        "calibrated_mask_set_receipt": {
            **_record(mask_receipt_path),
            "receipt_digest": mask_receipt["receipt_digest"],
        },
        "tasks": output_tasks,
        "paid_execution_started": False,
        "provider_mutations_performed": 0,
        "agent_selected_gaussian_indices": False,
        "canonical_source_altered": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (output / f"{RECEIPT_SCHEMA_VERSION}.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "REQUEST_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "FreshSceneRemovalFreezeError",
    "materialize_fresh_scene_removal_freezes",
]
