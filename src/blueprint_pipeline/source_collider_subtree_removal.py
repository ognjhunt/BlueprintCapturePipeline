"""Remove one exact source-collider subtree without rewriting scene meaning.

The replacement runtime must never leave the captured/source proxy collider
under the SimReady twin.  This module performs that operation generically for
OpenUSD stages and compares a digest of every unrelated composed prim,
attribute value, and relationship before and after export.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "source_collider_subtree_removal.v1"


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


def _prim_inventory(stage: Any, *, excluded_prefix: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if excluded_prefix and (
            path == excluded_prefix or path.startswith(excluded_prefix + "/")
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
    errors: list[str] = []
    if not source.is_file() or source.is_symlink():
        errors.append("source_collider_usd_missing_or_symlink")
    if not target.startswith("/") or target == "/" or "//" in target:
        errors.append("source_collider_target_prim_path_invalid")
    if output.exists() or output.suffix.lower() != ".usda":
        errors.append("source_collider_output_must_be_new_usda")
    if errors:
        raise SourceColliderSubtreeRemovalError(errors)

    source_digest = _sha256(source)
    if expected_source_sha256 is not None and source_digest != expected_source_sha256:
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_usd_digest_mismatch"]
        )
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise SourceColliderSubtreeRemovalError(["source_collider_usd_unreadable"])
    target_prim = stage.GetPrimAtPath(target)
    if not target_prim.IsValid():
        raise SourceColliderSubtreeRemovalError(
            ["source_collider_target_prim_missing"]
        )
    removed_paths = sorted(
        str(prim.GetPath())
        for prim in stage.TraverseAll()
        if str(prim.GetPath()) == target
        or str(prim.GetPath()).startswith(target + "/")
    )
    before_retained = _prim_inventory(stage, excluded_prefix=target)
    before_digest = canonical_digest({"prims": before_retained})

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
    if errors:
        raise SourceColliderSubtreeRemovalError(errors)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "exact_source_collider_subtree_removed",
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
        "caller_asserted_removal_accepted": False,
        "replacement_inserted": False,
        "claim_ceiling": "source_collision_subtree_removal_only",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "SourceColliderSubtreeRemovalError",
    "remove_source_collider_subtree",
]
