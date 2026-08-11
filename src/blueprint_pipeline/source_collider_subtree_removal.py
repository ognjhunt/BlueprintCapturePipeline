"""Remove one exact source-collider subtree without rewriting scene meaning.

The replacement runtime must never leave the captured/source proxy collider
under the SimReady twin.  This module performs that operation generically for
OpenUSD stages and compares a digest of every unrelated composed prim,
attribute value, and relationship before and after export.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


SCHEMA_VERSION = "source_collider_subtree_removal.v1"
BATCH_SCHEMA_VERSION = "source_collider_batch_removal.v1"


class SourceColliderSubtreeRemovalError(ValueError):
    """Stable, sorted collider-removal failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _value(value: Any) -> Any:
    """Stable JSON-shaped representation of composed USD property values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    # Gf, Vt, Sdf and token values all have deterministic string forms for a
    # fixed OpenUSD revision. The receipt binds that revision separately at the
    # caller/runtime layer.
    return str(value)


def _prim_inventory(
    stage: Any,
    *,
    excluded_prefix: str | None = None,
    excluded_prefixes: Sequence[str] = (),
) -> list[dict[str, Any]]:
    excluded = tuple(
        prefix
        for prefix in (excluded_prefix, *excluded_prefixes)
        if isinstance(prefix, str) and prefix
    )
    rows: list[dict[str, Any]] = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in excluded
        ):
            continue
        attributes = []
        for attribute in sorted(prim.GetAttributes(), key=lambda item: item.GetName()):
            attributes.append(
                {
                    "name": attribute.GetName(),
                    "type": str(attribute.GetTypeName()),
                    "authored": bool(attribute.HasAuthoredValue()),
                    "value": _value(attribute.Get()),
                }
            )
        relationships = [
            {
                "name": relationship.GetName(),
                "targets": sorted(str(target) for target in relationship.GetTargets()),
            }
            for relationship in sorted(
                prim.GetRelationships(), key=lambda item: item.GetName()
            )
        ]
        rows.append(
            {
                "path": path,
                "type_name": prim.GetTypeName(),
                "active": prim.IsActive(),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def remove_source_collider_subtree(
    *,
    source_usd_path: str | Path,
    target_prim_path: str,
    output_usda_path: str | Path,
    expected_source_sha256: str | None = None,
    removal_id: str | None = None,
) -> dict[str, Any]:
    """Delete exactly ``target_prim_path`` and verify unrelated composed data."""

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(output_usda_path).expanduser().resolve()
    target = str(target_prim_path or "")
    normalized_removal_id = str(removal_id or "")
    errors: list[str] = []
    if not source.is_file() or source.is_symlink():
        errors.append("source_collider_usd_missing_or_symlink")
    if not target.startswith("/") or target == "/" or "//" in target:
        errors.append("source_collider_target_prim_path_invalid")
    if output.exists() or output.suffix.lower() != ".usda":
        errors.append("source_collider_output_must_be_new_usda")
    if removal_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", normalized_removal_id
    ):
        errors.append("source_collider_removal_id_invalid")
    if errors:
        raise SourceColliderSubtreeRemovalError(errors)

    source_digest = _sha256(source)
    if expected_source_sha256 is not None and source_digest != expected_source_sha256:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_usd_digest_mismatch"]
        )
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise SourceColliderSubtreeRemovalError(["source_collider_usd_unreadable"])
    target_prim = source_stage.GetPrimAtPath(target)
    if not target_prim.IsValid():
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_target_prim_missing"]
        )
    removed_paths = sorted(
        str(prim.GetPath())
        for prim in source_stage.TraverseAll()
        if str(prim.GetPath()) == target
        or str(prim.GetPath()).startswith(target + "/")
    )
    before_retained = _prim_inventory(source_stage, excluded_prefix=target)
    before_digest = canonical_digest({"prims": before_retained})

    # Never author removal opinions into the source layer. OpenUSD caches root
    # layers by identifier, so mutating the stage returned for the source can
    # leak into another independent removal in the same process even without a
    # disk save. A flattened anonymous layer isolates every execution.
    working_layer = source_stage.Flatten()
    stage = Usd.Stage.Open(working_layer)
    if stage is None or not stage.GetPrimAtPath(target).IsValid():
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_working_stage_invalid"]
        )
    if not stage.RemovePrim(target):
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_target_remove_failed"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_output_export_failed"]
        )
    reopened = Usd.Stage.Open(str(output))
    if reopened is None:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_output_unreadable"]
        )
    after = _prim_inventory(reopened)
    after_digest = canonical_digest({"prims": after})
    remaining_target_count = sum(
        str(prim.GetPath()) == target
        or str(prim.GetPath()).startswith(target + "/")
        for prim in reopened.TraverseAll()
    )
    if remaining_target_count:
        errors.append("source_collider_target_subtree_still_present")
    if before_digest != after_digest or len(before_retained) != len(after):
        errors.append("source_collider_unrelated_prim_inventory_changed")
    if _sha256(source) != source_digest:
        errors.append("source_collider_source_bytes_changed")
    if errors:
        raise SourceColliderSubtreeRemovalError(errors)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "exact_source_collider_subtree_removed",
        "removal_id": normalized_removal_id or None,
        "source_scene_usd": {
            "path": str(source),
            "sha256": source_digest,
            "size_bytes": source.stat().st_size,
        },
        "removed_scene_usd": {
            "path": str(output),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
        # Flat fields preserve the existing articulated join contract.
        "sage_collision_usd_sha256": source_digest,
        "removed_scene_usd_sha256": _sha256(output),
        "removed_prim_path": target,
        "removed_prim_count": len(removed_paths),
        "removed_prim_paths_digest": canonical_digest({"paths": removed_paths}),
        "remaining_target_collision_prim_count": remaining_target_count,
        "retained_prim_count": len(after),
        "retained_prim_inventory_before_digest": before_digest,
        "retained_prim_inventory_after_digest": after_digest,
        "unrelated_prim_inventory_unchanged": True,
        "source_bytes_unchanged": True,
        "caller_asserted_removal_accepted": False,
        "replacement_inserted": False,
        "claim_ceiling": "source_collision_subtree_removal_only",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def materialize_source_collider_batch_removal(
    *,
    source_usd_path: str | Path,
    targets: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize independent target receipts and one shared removed stage.

    Every target is first removed from the same immutable source stage, yielding
    an independent receipt.  A second deterministic pass removes the complete
    target set into the shared collision scene used by a multi-task runtime.
    Unrelated composed prims must be byte-stably represented by the inventory
    digest in both the independent and shared passes.
    """

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    errors: list[str] = []
    if not source.is_file() or source.is_symlink():
        errors.append("source_collider_usd_missing_or_symlink")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        errors.append("source_collider_batch_output_not_empty")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        errors.append("source_collider_batch_targets_invalid")
        targets = ()

    normalized: list[dict[str, str]] = []
    for index, row in enumerate(targets):
        if not isinstance(row, Mapping):
            errors.append(f"source_collider_batch_target_invalid:{index}")
            continue
        removal_id = str(row.get("removal_id") or "")
        target = str(row.get("target_prim_path") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", removal_id):
            errors.append(f"source_collider_batch_removal_id_invalid:{index}")
        if not target.startswith("/") or target == "/" or "//" in target:
            errors.append(f"source_collider_batch_target_path_invalid:{index}")
        normalized.append(
            {"removal_id": removal_id, "target_prim_path": target}
        )
    if len(normalized) < 2:
        errors.append("source_collider_batch_requires_two_targets")
    if len(normalized) > MAX_REPLACEMENT_OBJECTS:
        errors.append("source_collider_batch_target_count_exceeds_limit")
    removal_ids = [row["removal_id"] for row in normalized]
    target_paths = [row["target_prim_path"] for row in normalized]
    if len(set(removal_ids)) != len(removal_ids):
        errors.append("source_collider_batch_removal_ids_duplicate")
    if len(set(target_paths)) != len(target_paths):
        errors.append("source_collider_batch_target_paths_duplicate")
    for left in target_paths:
        for right in target_paths:
            if left != right and right.startswith(left + "/"):
                errors.append("source_collider_batch_targets_nested")
    if errors:
        raise SourceColliderSubtreeRemovalError(errors)

    source_digest = _sha256(source)
    if expected_source_sha256 is not None and source_digest != expected_source_sha256:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_usd_digest_mismatch"]
        )

    output.mkdir(parents=True, exist_ok=True)
    independent_root = output / "independent"
    independent_root.mkdir(parents=True, exist_ok=False)
    independent_rows: list[dict[str, Any]] = []
    for row in sorted(normalized, key=lambda item: item["removal_id"]):
        removed_path = independent_root / f"{row['removal_id']}.removed.usda"
        receipt = remove_source_collider_subtree(
            source_usd_path=source,
            target_prim_path=row["target_prim_path"],
            output_usda_path=removed_path,
            expected_source_sha256=source_digest,
            removal_id=row["removal_id"],
        )
        receipt_path = independent_root / f"{row['removal_id']}.receipt.json"
        receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
        independent_rows.append(
            {
                **row,
                "source_scene_sha256": receipt["sage_collision_usd_sha256"],
                "receipt_digest": receipt["receipt_digest"],
                "receipt": _artifact_record(receipt_path, output),
                "removed_scene": _artifact_record(removed_path, output),
            }
        )

    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise SourceColliderSubtreeRemovalError(["source_collider_usd_unreadable"])
    removed_paths_by_target: dict[str, list[str]] = {}
    for row in normalized:
        target = row["target_prim_path"]
        prim = source_stage.GetPrimAtPath(target)
        if not prim.IsValid():
            raise SourceColliderSubtreeRemovalError(
                [f"source_collider_batch_target_prim_missing:{row['removal_id']}"]
            )
        removed_paths_by_target[row["removal_id"]] = sorted(
            str(candidate.GetPath())
            for candidate in source_stage.TraverseAll()
            if str(candidate.GetPath()) == target
            or str(candidate.GetPath()).startswith(target + "/")
        )
    before = _prim_inventory(source_stage, excluded_prefixes=target_paths)
    before_digest = canonical_digest({"prims": before})
    working_layer = source_stage.Flatten()
    stage = Usd.Stage.Open(working_layer)
    if stage is None:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_batch_working_stage_invalid"]
        )
    for target in sorted(target_paths, reverse=True):
        if not stage.RemovePrim(target):
            raise SourceColliderSubtreeRemovalError(
                ["source_collider_batch_target_remove_failed"]
            )
    shared_path = output / "scene_without_source_colliders.usda"
    if not stage.GetRootLayer().Export(str(shared_path)):
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_batch_output_export_failed"]
        )
    reopened = Usd.Stage.Open(str(shared_path))
    if reopened is None:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_batch_output_unreadable"]
        )
    remaining = [
        target
        for target in target_paths
        if reopened.GetPrimAtPath(target).IsValid()
    ]
    after = _prim_inventory(reopened)
    after_digest = canonical_digest({"prims": after})
    if remaining:
        errors.append("source_collider_batch_target_subtree_still_present")
    if before_digest != after_digest or len(before) != len(after):
        errors.append("source_collider_batch_unrelated_prim_inventory_changed")
    if _sha256(source) != source_digest:
        errors.append("source_collider_source_bytes_changed")
    if errors:
        raise SourceColliderSubtreeRemovalError(errors)

    receipt: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "status": "independent_and_shared_source_colliders_removed",
        "source_scene_usd": {
            "path": str(source),
            "sha256": source_digest,
            "size_bytes": source.stat().st_size,
        },
        "shared_removed_scene_usd": _artifact_record(shared_path, output),
        "target_removals": [
            {
                **row,
                "removed_prim_count": len(
                    removed_paths_by_target[row["removal_id"]]
                ),
                "removed_prim_paths_digest": canonical_digest(
                    {"paths": removed_paths_by_target[row["removal_id"]]}
                ),
            }
            for row in independent_rows
        ],
        "target_count": len(independent_rows),
        "remaining_target_collision_prim_count": 0,
        "retained_prim_count": len(after),
        "retained_prim_inventory_before_digest": before_digest,
        "retained_prim_inventory_after_digest": after_digest,
        "unrelated_prim_inventory_unchanged": True,
        "source_bytes_unchanged": True,
        "independent_receipts_share_exact_source_digest": all(
            row["source_scene_sha256"] == source_digest
            for row in independent_rows
        ),
        "independent_removed_scenes_are_distinct": len(
            {row["removed_scene"]["sha256"] for row in independent_rows}
        )
        == len(independent_rows),
        "caller_asserted_removal_accepted": False,
        "replacement_inserted": False,
        "claim_ceiling": "source_collision_subtree_removal_only",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output / f"{BATCH_SCHEMA_VERSION}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    # Verify serialization itself before returning a receipt intended for a
    # downstream construction binding.
    serialized = json.loads(receipt_path.read_text(encoding="utf-8"))
    if serialized != receipt:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_batch_receipt_roundtrip_changed"]
        )
    return receipt


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SourceColliderSubtreeRemovalError",
    "materialize_source_collider_batch_removal",
    "remove_source_collider_subtree",
]
