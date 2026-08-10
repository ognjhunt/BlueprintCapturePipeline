"""Bind independent source removals to co-present simulator replacements.

The runtime must never infer that two tasks may share a mask, collider deletion,
or replacement qualification merely because they use one scene.  This contract
joins each frozen task to one independently qualified construction lane while
retaining one digest for the shared scene construction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "replacement_construction_bindings.v1"


class ReplacementConstructionBindingsError(ValueError):
    """Stable validation failures for removal/replacement construction joins."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _identifier(value: Any) -> str:
    text = str(value or "")
    if (
        not text
        or not text.replace("_", "a").replace("-", "a").isalnum()
    ):
        return ""
    return text


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
    for field in ("scene_freeze_digest", "task_freeze_join_digest"):
        if not _digest(payload.get(field)):
            errors.append(f"replacement_construction_{field}_invalid")
    rows = payload.get("bindings")
    if not isinstance(rows, list) or not rows:
        errors.append("replacement_construction_bindings_missing")
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
                    errors.append(
                        f"replacement_construction_identity_invalid:{index}:{field}"
                    )
            elif not _identifier(row.get(field)):
                errors.append(
                    f"replacement_construction_identity_invalid:{index}:{field}"
                )
        for field in (
            "task_freeze_digest",
            "mask_set_receipt_digest",
            "source_removal_receipt_digest",
            "collider_deletion_receipt_digest",
            "replacement_qualification_receipt_digest",
            "replacement_asset_sha256",
        ):
            if not _digest(row.get(field)):
                errors.append(
                    f"replacement_construction_digest_invalid:{index}:{field}"
                )
        for field in (
            "source_removal_qualified",
            "collider_deletion_qualified",
            "replacement_simulator_import_qualified",
        ):
            if row.get(field) is not True:
                errors.append(
                    f"replacement_construction_qualification_missing:{index}:{field}"
                )
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
    task_freeze_join_digest: str,
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonicalize and seal construction rows after all qualifications exist."""

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_freeze_digest": scene_freeze_digest,
        "task_freeze_join_digest": task_freeze_join_digest,
        "bindings": sorted(
            (json.loads(json.dumps(row)) for row in bindings),
            key=lambda row: str(row.get("asset_id")),
        ),
        "construction_digest": "",
    }
    payload["construction_digest"] = canonical_digest(
        payload, digest_field="construction_digest"
    )
    return validate_replacement_construction_bindings(payload)


__all__ = [
    "ReplacementConstructionBindingsError",
    "SCHEMA_VERSION",
    "seal_replacement_construction_bindings",
    "validate_replacement_construction_bindings",
]
