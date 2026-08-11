"""Bind independent source removals to co-present simulator replacements.

The runtime must never infer that two tasks may share a mask, collider deletion,
or replacement qualification merely because they use one scene.  This contract
joins each frozen task to one independently qualified construction lane while
retaining one digest for the shared scene construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    MAX_REPLACEMENT_OBJECTS,
    validate_scene_freeze,
    validate_task_freeze,
    validate_task_freeze_set,
)
from .simready_graph_asset_static_qualification import (
    SCHEMA_VERSION as STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
)
from .simready_replacement_native_qualification import (
    NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION,
)


SCHEMA_VERSION = "replacement_construction_bindings.v2"
MASK_SET_QUALIFICATION_SCHEMA_VERSION = "calibrated_removal_mask_set_qualification.v1"
GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION = "gaussian_source_removal_qualification.v1"
REPLACEMENT_QUALIFICATION_SCHEMA_VERSION = "simready_replacement_native_qualification.v1"
SOURCE_COLLIDER_DELETION_SCHEMA_VERSION = "source_collider_subtree_removal.v1"
SOURCE_COLLIDER_BATCH_DELETION_SCHEMA_VERSION = "source_collider_batch_removal.v1"
_LANE_PATH_FIELDS = frozenset(
    {
        "task_freeze_receipt_path",
        "mask_set_receipt_path",
        "gaussian_removal_receipt_path",
        "source_collider_deletion_receipt_path",
        "replacement_qualification_receipt_path",
    }
)


class ReplacementConstructionBindingsError(ValueError):
    """Stable validation failures for removal/replacement construction joins."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _identifier(value: Any) -> str:
    text = str(value or "")
    if not text or not text.replace("_", "a").replace("-", "a").isalnum():
        return ""
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json_receipt(
    path_value: Any,
    *,
    role: str,
    schema_version: str,
    status: str | None,
    digest_field: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(path_value, (str, Path)) or isinstance(path_value, bool):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_path_invalid"]
        )
    path = Path(path_value).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_path_invalid"]
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_receipt_invalid"]
        ) from exc
    if not isinstance(value, dict):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_receipt_invalid"]
        )
    errors: list[str] = []
    if value.get("schema_version") != schema_version:
        errors.append(f"replacement_construction_{role}_schema_invalid")
    if status is None and "status" in value:
        errors.append(f"replacement_construction_{role}_status_invalid")
    elif status is not None and value.get("status") != status:
        errors.append(f"replacement_construction_{role}_status_unqualified")
    if value.get(digest_field) != canonical_digest(value, digest_field=digest_field):
        errors.append(f"replacement_construction_{role}_digest_invalid")
    if errors:
        raise ReplacementConstructionBindingsError(errors)
    return path, value


def _receipt_file_record(
    path: Path, value: Mapping[str, Any], *, digest_field: str
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema_version": value["schema_version"],
        "canonical_digest": value[digest_field],
    }


def _verify_file_record(
    record: Any,
    *,
    parent_path: Path,
    role: str,
    schema_version: str,
    status: str,
    digest_field: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_evidence_missing"]
        )
    record_path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        record_path.is_symlink()
        or not record_path.is_file()
        or record_path.stat().st_size != record.get("size_bytes")
        or _sha256(record_path) != record.get("sha256")
        or record.get("schema_version") != schema_version
    ):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_evidence_invalid"]
        )
    path, value = _read_json_receipt(
        record_path,
        role=role,
        schema_version=schema_version,
        status=status,
        digest_field=digest_field,
    )
    if (
        record.get("canonical_digest") != value[digest_field]
        or path == parent_path
    ):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_evidence_invalid"]
        )
    return path, value


def _verify_replacement_native_qualification_evidence(
    *,
    replacement_path: Path,
    replacement: Mapping[str, Any],
    expected: Mapping[str, str],
    index: int,
) -> None:
    role = f"replacement_qualification:{index}"
    evidence = replacement.get("evidence_receipts")
    if not isinstance(evidence, Mapping):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_native_evidence_missing"]
        )
    _, static = _verify_file_record(
        evidence.get("static_qualification"),
        parent_path=replacement_path,
        role=f"{role}:static_qualification",
        schema_version=STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
        status="authored_structure_statically_qualified",
        digest_field="receipt_digest",
    )
    _, native = _verify_file_record(
        evidence.get("native_import"),
        parent_path=replacement_path,
        role=f"{role}:native_import",
        schema_version=NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION,
        status="native_import_qualified",
        digest_field="receipt_digest",
    )
    _require_identity(
        receipt=static,
        expected={
            key: expected[key] for key in ("task_id", "task_freeze_digest", "asset_id")
        },
        role=f"{role}:static",
    )
    _require_identity(receipt=native, expected=expected, role=f"{role}:native_import")
    static_asset = (static.get("replacement_usd") or {}).get("sha256")
    errors: list[str] = []
    if static_asset != replacement.get("replacement_asset_sha256"):
        errors.append(f"replacement_construction_{role}_static_asset_mismatch")
    if native.get("replacement_asset_sha256") != replacement.get("replacement_asset_sha256"):
        errors.append(f"replacement_construction_{role}_native_asset_mismatch")
    if native.get("native_isaac_executed") is not True:
        errors.append(f"replacement_construction_{role}_native_execution_missing")
    if native.get("native_simulator_import_qualified") is not True:
        errors.append(f"replacement_construction_{role}_native_import_not_qualified")
    if replacement.get("native_import_receipt_digest") != native.get("receipt_digest"):
        errors.append(f"replacement_construction_{role}_native_receipt_mismatch")
    if (
        replacement.get("static_qualification_receipt_digest")
        != static.get("receipt_digest")
    ):
        errors.append(f"replacement_construction_{role}_static_receipt_mismatch")
    if errors:
        raise ReplacementConstructionBindingsError(errors)


def _require_identity(
    *,
    receipt: Mapping[str, Any],
    expected: Mapping[str, str],
    role: str,
) -> None:
    errors = [
        f"replacement_construction_{role}_identity_mismatch:{field}"
        for field, expected_value in expected.items()
        if str(receipt.get(field) or "") != expected_value
    ]
    if errors:
        raise ReplacementConstructionBindingsError(errors)


def _resolve_source_collider_deletion(
    path_value: Any,
    *,
    index: int,
    collider_deletion_id: str,
    target_prim_path: str,
    sage_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    role = f"source_collider_deletion:{index}"
    if not isinstance(path_value, (str, Path)) or isinstance(path_value, bool):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_path_invalid"]
        )
    batch_path = Path(path_value).expanduser().resolve()
    if batch_path.is_symlink() or not batch_path.is_file():
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_path_invalid"]
        )
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_receipt_invalid"]
        ) from exc
    if (
        isinstance(batch, Mapping)
        and batch.get("schema_version") == SOURCE_COLLIDER_DELETION_SCHEMA_VERSION
    ):
        verified_path, child = _read_json_receipt(
            batch_path,
            role=role,
            schema_version=SOURCE_COLLIDER_DELETION_SCHEMA_VERSION,
            status="exact_source_collider_subtree_removed",
            digest_field="receipt_digest",
        )
        if (
            child.get("removal_id") != collider_deletion_id
            or child.get("sage_collision_usd_sha256") != sage_sha256
            or child.get("removed_prim_path") != target_prim_path
            or not isinstance(child.get("removed_prim_count"), int)
            or child.get("removed_prim_count", 0) <= 0
            or child.get("source_bytes_unchanged") is not True
            or child.get("unrelated_prim_inventory_unchanged") is not True
            or child.get("remaining_target_collision_prim_count") != 0
            or child.get("replacement_inserted") is not False
        ):
            raise ReplacementConstructionBindingsError(
                [f"replacement_construction_{role}_identity_mismatch"]
            )
        return (
            verified_path,
            child,
            {
                "selected_deletion_id": collider_deletion_id,
                "independent": _receipt_file_record(
                    verified_path, child, digest_field="receipt_digest"
                ),
            },
        )
    errors: list[str] = []
    if not isinstance(batch, dict):
        errors.append(f"replacement_construction_{role}_receipt_invalid")
        batch = {}
    if batch.get("schema_version") != SOURCE_COLLIDER_BATCH_DELETION_SCHEMA_VERSION:
        errors.append(f"replacement_construction_{role}_schema_invalid")
    if batch.get("status") != "independent_and_shared_source_colliders_removed":
        errors.append(f"replacement_construction_{role}_status_unqualified")
    if batch.get("receipt_digest") != canonical_digest(batch, digest_field="receipt_digest"):
        errors.append(f"replacement_construction_{role}_digest_invalid")
    source_scene = batch.get("source_scene_usd")
    rows = batch.get("target_removals")
    if (
        not isinstance(source_scene, Mapping)
        or source_scene.get("sha256") != sage_sha256
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_REPLACEMENT_OBJECTS
        or batch.get("target_count") != len(rows)
        or batch.get("source_bytes_unchanged") is not True
        or batch.get("unrelated_prim_inventory_unchanged") is not True
        or batch.get("remaining_target_collision_prim_count") != 0
        or batch.get("replacement_inserted") is not False
        or batch.get("independent_receipts_share_exact_source_digest") is not True
        or batch.get("independent_removed_scenes_are_distinct") is not True
    ):
        errors.append(f"replacement_construction_{role}_batch_not_qualified")
    matches = (
        [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("removal_id") == collider_deletion_id
        ]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1:
        errors.append(f"replacement_construction_{role}_deletion_id_mismatch")
    if errors:
        raise ReplacementConstructionBindingsError(errors)
    selected = dict(matches[0])
    if (
        selected.get("target_prim_path") != target_prim_path
        or selected.get("source_scene_sha256") != sage_sha256
        or not isinstance(selected.get("removed_prim_count"), int)
        or selected.get("removed_prim_count", 0) <= 0
    ):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_identity_mismatch"]
        )
    child_record = selected.get("receipt")
    if not isinstance(child_record, Mapping):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_child_receipt_invalid"]
        )
    relative_path = str(child_record.get("relative_path") or "")
    child_path = (batch_path.parent / relative_path).resolve()
    if (
        child_path == batch_path.parent
        or batch_path.parent not in child_path.parents
        or child_path.is_symlink()
        or not child_path.is_file()
        or child_path.stat().st_size != child_record.get("size_bytes")
        or _sha256(child_path) != child_record.get("sha256")
    ):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_child_receipt_invalid"]
        )
    verified_path, child = _read_json_receipt(
        child_path,
        role=f"source_collider_deletion_child:{index}",
        schema_version=SOURCE_COLLIDER_DELETION_SCHEMA_VERSION,
        status="exact_source_collider_subtree_removed",
        digest_field="receipt_digest",
    )
    if (
        child.get("receipt_digest") != selected.get("receipt_digest")
        or child.get("sage_collision_usd_sha256") != sage_sha256
        or child.get("removed_prim_path") != target_prim_path
        or child.get("removed_prim_count") != selected.get("removed_prim_count")
        or child.get("source_bytes_unchanged") is not True
        or child.get("unrelated_prim_inventory_unchanged") is not True
        or child.get("remaining_target_collision_prim_count") != 0
        or child.get("replacement_inserted") is not False
    ):
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_{role}_child_binding_mismatch"]
        )
    return (
        verified_path,
        child,
        {
            "batch": _receipt_file_record(batch_path, batch, digest_field="receipt_digest"),
            "selected_deletion_id": collider_deletion_id,
            "independent": _receipt_file_record(
                verified_path, child, digest_field="receipt_digest"
            ),
        },
    )


def validate_replacement_construction_bindings(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a sealed, scene-shared set of independent construction lanes."""

    try:
        payload = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReplacementConstructionBindingsError(
            ["replacement_construction_bindings_invalid"]
        ) from exc
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("replacement_construction_schema_invalid")
    for field in ("scene_freeze_digest", "task_freeze_set_digest"):
        if not _digest(payload.get(field)):
            errors.append(f"replacement_construction_{field}_invalid")
    rows = payload.get("bindings")
    if not isinstance(rows, list):
        errors.append("replacement_construction_bindings_missing")
        rows = []
    elif not 1 <= len(rows) <= MAX_REPLACEMENT_OBJECTS:
        errors.append("replacement_construction_binding_count_out_of_range")
        rows = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            errors.append(f"replacement_construction_binding_invalid:{index}")
            continue
        row = dict(raw)
        for field in (
            "task_id",
            "asset_id",
            "source_object_instance_id",
            "removal_id",
            "mask_set_id",
            "collider_deletion_id",
            "source_collider_prim_path",
            "replacement_qualification_id",
        ):
            if field == "source_collider_prim_path":
                if not str(row.get(field) or "").startswith("/"):
                    errors.append(f"replacement_construction_identity_invalid:{index}:{field}")
            elif not _identifier(row.get(field)):
                errors.append(f"replacement_construction_identity_invalid:{index}:{field}")
        for field in (
            "task_freeze_digest",
            "mask_set_receipt_digest",
            "source_removal_receipt_digest",
            "collider_deletion_receipt_digest",
            "replacement_qualification_receipt_digest",
            "replacement_asset_sha256",
        ):
            if not _digest(row.get(field)):
                errors.append(f"replacement_construction_digest_invalid:{index}:{field}")
        for field in (
            "source_removal_qualified",
            "collider_deletion_qualified",
            "replacement_simulator_import_qualified",
        ):
            if row.get(field) is not True:
                errors.append(f"replacement_construction_qualification_missing:{index}:{field}")
        normalized.append(row)

    independent_fields = (
        "task_id",
        "asset_id",
        "task_freeze_digest",
        "source_object_instance_id",
        "removal_id",
        "mask_set_id",
        "mask_set_receipt_digest",
        "source_removal_receipt_digest",
        "collider_deletion_id",
        "source_collider_prim_path",
        "collider_deletion_receipt_digest",
        "replacement_qualification_id",
        "replacement_qualification_receipt_digest",
        "replacement_asset_sha256",
    )
    for field in independent_fields:
        values = [str(row.get(field) or "") for row in normalized]
        if len(values) != len(set(values)):
            errors.append(f"replacement_construction_shared_identity:{field}")

    payload["bindings"] = sorted(normalized, key=lambda row: str(row.get("asset_id")))
    expected = canonical_digest(payload, digest_field="construction_digest")
    if payload.get("construction_digest") != expected:
        errors.append("replacement_construction_digest_invalid")
    if errors:
        raise ReplacementConstructionBindingsError(errors)
    return payload


def seal_replacement_construction_bindings(
    *,
    scene_freeze_digest: str,
    task_freeze_set_digest: str | None = None,
    task_freeze_join_digest: str | None = None,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonicalize and seal construction rows after all qualifications exist."""

    freeze_set_digest = task_freeze_set_digest or task_freeze_join_digest

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_freeze_digest": scene_freeze_digest,
        "task_freeze_set_digest": freeze_set_digest,
        "bindings": sorted(
            (json.loads(json.dumps(row)) for row in bindings),
            key=lambda row: str(row.get("asset_id")),
        ),
        "construction_digest": "",
    }
    payload["construction_digest"] = canonical_digest(payload, digest_field="construction_digest")
    return validate_replacement_construction_bindings(payload)


def materialize_replacement_construction_bindings(
    *,
    scene_freeze_receipt_path: str | Path,
    evidence_lanes: Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Join path-backed construction evidence without trusting caller claims.

    Each lane is path-only.  Task, source-object, removal, mask, collider, and
    replacement identities are read from canonical receipts and joined to the
    frozen task.  The compatibility sealer remains available for already
    trusted internal mappings, but this materializer is the evidence boundary
    used before native controls or policy execution.
    """

    scene_path, raw_scene = _read_json_receipt(
        scene_freeze_receipt_path,
        role="scene_freeze",
        schema_version="dual_task_scene_freeze.v1",
        status=None,
        digest_field="scene_freeze_digest",
    )
    try:
        scene = validate_scene_freeze(raw_scene)
    except DualTaskRehearsalContractError as exc:
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_scene_freeze_invalid:{error}" for error in exc.errors]
        ) from exc

    if (
        isinstance(evidence_lanes, (str, bytes))
        or not 1 <= len(evidence_lanes) <= MAX_REPLACEMENT_OBJECTS
    ):
        raise ReplacementConstructionBindingsError(
            ["replacement_construction_evidence_lane_count_out_of_range"]
        )

    task_rows: list[dict[str, Any]] = []
    opened: list[tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]], Any]] = []
    for index, lane_value in enumerate(evidence_lanes):
        if not isinstance(lane_value, Mapping) or set(lane_value) != _LANE_PATH_FIELDS:
            raise ReplacementConstructionBindingsError(
                [f"replacement_construction_lane_paths_invalid:{index}"]
            )
        lane = dict(lane_value)
        task_path, raw_task = _read_json_receipt(
            lane["task_freeze_receipt_path"],
            role=f"task_freeze:{index}",
            schema_version="dual_task_task_freeze.v1",
            status=None,
            digest_field="task_freeze_digest",
        )
        try:
            task = validate_task_freeze(raw_task)
        except DualTaskRehearsalContractError as exc:
            raise ReplacementConstructionBindingsError(
                [
                    f"replacement_construction_task_freeze_invalid:{index}:{error}"
                    for error in exc.errors
                ]
            ) from exc
        if task["scene_freeze_digest"] != scene["scene_freeze_digest"]:
            raise ReplacementConstructionBindingsError(
                [f"replacement_construction_task_scene_mismatch:{index}"]
            )
        task_rows.append(task)
        opened.append(
            (
                task,
                {
                    "task_freeze": (task_path, task),
                    "mask_set": _read_json_receipt(
                        lane["mask_set_receipt_path"],
                        role=f"mask_set:{index}",
                        schema_version=MASK_SET_QUALIFICATION_SCHEMA_VERSION,
                        status="calibrated_mask_set_qualified",
                        digest_field="receipt_digest",
                    ),
                    "gaussian_removal": _read_json_receipt(
                        lane["gaussian_removal_receipt_path"],
                        role=f"gaussian_removal:{index}",
                        schema_version=GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
                        status="source_gaussian_removal_qualified",
                        digest_field="receipt_digest",
                    ),
                    "replacement_qualification": _read_json_receipt(
                        lane["replacement_qualification_receipt_path"],
                        role=f"replacement_qualification:{index}",
                        schema_version=REPLACEMENT_QUALIFICATION_SCHEMA_VERSION,
                        status="native_simulator_import_qualified",
                        digest_field="receipt_digest",
                    ),
                },
                lane["source_collider_deletion_receipt_path"],
            )
        )

    try:
        task_set = validate_task_freeze_set(task_rows)
    except DualTaskRehearsalContractError as exc:
        raise ReplacementConstructionBindingsError(
            [f"replacement_construction_task_set_invalid:{error}" for error in exc.errors]
        ) from exc

    scene_id = str(scene["selected_scene_id"])
    interiorgs_sha256 = str(scene["source_components"]["interiorgs"]["sha256"])
    sage_sha256 = str(scene["source_components"]["sage_collision"]["sha256"])
    bindings: list[dict[str, Any]] = []
    for index, (task, receipts, collider_batch_path) in enumerate(opened):
        source = task["source_object"]
        removal = task["removal_plan"]
        common_identity = {
            "scene_id": scene_id,
            "scene_freeze_digest": str(scene["scene_freeze_digest"]),
            "task_id": str(task["task_id"]),
            "task_freeze_digest": str(task["task_freeze_digest"]),
            "source_object_instance_id": str(source["instance_id"]),
            "removal_id": str(removal["removal_id"]),
            "mask_set_id": str(removal["mask_set_id"]),
        }
        mask_path, mask = receipts["mask_set"]
        gaussian_path, gaussian = receipts["gaussian_removal"]
        _, collider, collider_evidence = _resolve_source_collider_deletion(
            collider_batch_path,
            index=index,
            collider_deletion_id=str(removal["collider_deletion_id"]),
            target_prim_path=str(removal["source_collider_prim_path"]),
            sage_sha256=sage_sha256,
        )
        replacement_path, replacement = receipts["replacement_qualification"]
        _require_identity(receipt=mask, expected=common_identity, role=f"mask_set:{index}")
        _require_identity(
            receipt=gaussian,
            expected=common_identity,
            role=f"gaussian_removal:{index}",
        )
        replacement_identity = {
            **{
                key: common_identity[key]
                for key in (
                    "scene_id",
                    "scene_freeze_digest",
                    "task_id",
                    "task_freeze_digest",
                    "source_object_instance_id",
                )
            },
            "asset_id": str(removal["replacement_asset_id"]),
            "replacement_qualification_id": str(
                removal["replacement_qualification_id"]
            ),
        }
        _require_identity(
            receipt=replacement,
            expected=replacement_identity,
            role=f"replacement_qualification:{index}",
        )
        errors: list[str] = []
        if mask.get("source_scene_sha256") != interiorgs_sha256:
            errors.append(f"replacement_construction_mask_set_source_mismatch:{index}")
        if mask.get("calibrated_masks_qualified") is not True:
            errors.append(f"replacement_construction_mask_set_not_qualified:{index}")
        if gaussian.get("source_scene_sha256") != interiorgs_sha256:
            errors.append(f"replacement_construction_gaussian_removal_source_mismatch:{index}")
        if gaussian.get("mask_set_receipt_digest") != mask.get("receipt_digest"):
            errors.append(f"replacement_construction_gaussian_removal_mask_mismatch:{index}")
        if (
            gaussian.get("source_removal_qualified") is not True
            or gaussian.get("retained_records_byte_exact") is not True
            or gaussian.get("protected_geometry_deleted") is not False
        ):
            errors.append(f"replacement_construction_gaussian_removal_not_qualified:{index}")
        if replacement.get("native_simulator_import_qualified") is not True:
            errors.append(f"replacement_construction_replacement_import_not_qualified:{index}")
        if not _digest(replacement.get("replacement_asset_sha256")):
            errors.append(f"replacement_construction_replacement_asset_digest_invalid:{index}")
        if errors:
            raise ReplacementConstructionBindingsError(errors)
        _verify_replacement_native_qualification_evidence(
            replacement_path=replacement_path,
            replacement=replacement,
            expected=replacement_identity,
            index=index,
        )

        task_path, _ = receipts["task_freeze"]
        bindings.append(
            {
                "task_id": task["task_id"],
                "asset_id": removal["replacement_asset_id"],
                "task_freeze_digest": task["task_freeze_digest"],
                "source_object_instance_id": source["instance_id"],
                "removal_id": removal["removal_id"],
                "mask_set_id": removal["mask_set_id"],
                "mask_set_receipt_digest": mask["receipt_digest"],
                "source_removal_receipt_digest": gaussian["receipt_digest"],
                "source_removal_qualified": True,
                "collider_deletion_id": removal["collider_deletion_id"],
                "source_collider_prim_path": removal["source_collider_prim_path"],
                "collider_deletion_receipt_digest": collider["receipt_digest"],
                "collider_deletion_qualified": True,
                "replacement_qualification_id": removal["replacement_qualification_id"],
                "replacement_qualification_receipt_digest": replacement["receipt_digest"],
                "replacement_asset_sha256": replacement["replacement_asset_sha256"],
                "replacement_simulator_import_qualified": True,
                "evidence_receipts": {
                    "task_freeze": _receipt_file_record(
                        task_path, task, digest_field="task_freeze_digest"
                    ),
                    "mask_set": _receipt_file_record(
                        mask_path, mask, digest_field="receipt_digest"
                    ),
                    "gaussian_removal": _receipt_file_record(
                        gaussian_path, gaussian, digest_field="receipt_digest"
                    ),
                    "source_collider_deletion": collider_evidence,
                    "replacement_qualification": _receipt_file_record(
                        replacement_path, replacement, digest_field="receipt_digest"
                    ),
                },
            }
        )

    result = seal_replacement_construction_bindings(
        scene_freeze_digest=scene["scene_freeze_digest"],
        task_freeze_set_digest=task_set["set_digest"],
        bindings=bindings,
    )
    result["scene_freeze_receipt"] = _receipt_file_record(
        scene_path, scene, digest_field="scene_freeze_digest"
    )
    result["construction_digest"] = canonical_digest(result, digest_field="construction_digest")
    result = validate_replacement_construction_bindings(result)
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination.exists() or destination.is_symlink():
            raise ReplacementConstructionBindingsError(["replacement_construction_output_exists"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


__all__ = [
    "GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION",
    "MASK_SET_QUALIFICATION_SCHEMA_VERSION",
    "REPLACEMENT_QUALIFICATION_SCHEMA_VERSION",
    "ReplacementConstructionBindingsError",
    "SCHEMA_VERSION",
    "SOURCE_COLLIDER_BATCH_DELETION_SCHEMA_VERSION",
    "SOURCE_COLLIDER_DELETION_SCHEMA_VERSION",
    "materialize_replacement_construction_bindings",
    "seal_replacement_construction_bindings",
    "validate_replacement_construction_bindings",
]
