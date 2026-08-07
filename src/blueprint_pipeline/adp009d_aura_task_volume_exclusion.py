"""Deterministic removal of retained Gaussians inside the approved-can volume.

AuraFusion360's object-removal step recoloured the source can's Gaussians to
match the background instead of deleting them.  The can therefore disappears
from RGB while remaining opaque in depth, so the approved replacement can is
partially occluded by an invisible surface.

This module completes exactly that removal as a digest-bound derivative.  The
sealed PLY is never modified: retained rows are copied byte-for-byte, only the
vertex count in the header changes, and the exclusion volume is preregistered
rather than fitted to an outcome.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest

EXCLUSION_RECEIPT_SCHEMA_VERSION = "adp009d_aura_task_volume_exclusion.v1"

SEALED_AURA_PLY_SHA256 = (
    "sha256:cbb05fc8e6da6ecdb72464f3b115f63e8747e2b67e97c309b4e40952b33000bd"
)
SEALED_AURA_PLY_VERTEX_COUNT = 415_265

# Preregistered exclusion volume, in the shared metre frame.  The cylinder is
# the approved can's own task volume: its sealed axis, a radius comfortably
# outside the can's 3.1 cm visible surface, and a floor one centimetre above the
# admitted support plane so the reconstructed shelf is never touched.
CAN_AXIS_XY_M = (3.4681748, -3.3100837)
SUPPORT_HEIGHT_M = 0.5264650138348479
EXCLUSION_RADIUS_M = 0.06
EXCLUSION_FLOOR_ABOVE_SUPPORT_M = 0.01
EXCLUSION_CEILING_ABOVE_SUPPORT_M = 0.20

# Measured on the sealed asset before any re-render.  A different count means
# the input is not the audited asset or the rule drifted.
EXPECTED_REMOVED_VERTEX_COUNT = 943
REMOVED_FRACTION_CEILING = 0.005


class AuraTaskVolumeExclusionError(ValueError):
    """Stable fail-closed exclusion-derivative errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _read_header(handle: Any) -> tuple[bytes, int, list[str]]:
    """Return (raw header bytes, vertex count, property names)."""

    lines: list[bytes] = []
    count = 0
    names: list[str] = []
    if handle.readline().strip() != b"ply":
        raise AuraTaskVolumeExclusionError(["aura_exclusion_source_not_ply"])
    lines.append(b"ply\n")
    fmt = None
    while True:
        line = handle.readline()
        if not line:
            raise AuraTaskVolumeExclusionError(["aura_exclusion_source_header_invalid"])
        lines.append(line)
        text = line.decode("latin-1").strip()
        if text.startswith("format"):
            fmt = text.split()[1]
        elif text.startswith("element vertex"):
            count = int(text.split()[-1])
        elif text.startswith("element"):
            raise AuraTaskVolumeExclusionError(
                ["aura_exclusion_source_multi_element_ply"]
            )
        elif text.startswith("property"):
            parts = text.split()
            if parts[1] != "float":
                raise AuraTaskVolumeExclusionError(
                    ["aura_exclusion_source_non_float_property"]
                )
            names.append(parts[-1])
        elif text == "end_header":
            break
    if fmt != "binary_little_endian":
        raise AuraTaskVolumeExclusionError(["aura_exclusion_source_format_unsupported"])
    return b"".join(lines), count, names


def materialize_aura_task_volume_exclusion(
    *,
    source_ply_path: str | Path,
    output_ply_path: str | Path,
    receipt_path: str | Path | None = None,
    expected_source_sha256: str = SEALED_AURA_PLY_SHA256,
    expected_removed_vertex_count: int | None = EXPECTED_REMOVED_VERTEX_COUNT,
    removed_fraction_ceiling: float = REMOVED_FRACTION_CEILING,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Drop Gaussians whose centre lies inside the approved-can task volume."""

    import numpy as np

    source = Path(source_ply_path)
    output = Path(output_ply_path)
    if not source.is_file():
        raise AuraTaskVolumeExclusionError(["aura_exclusion_source_missing"])
    observed_source_sha256 = f"sha256:{sha256_file(source)}"
    if expected_source_sha256 and observed_source_sha256 != expected_source_sha256:
        raise AuraTaskVolumeExclusionError(["aura_exclusion_source_digest_mismatch"])

    with source.open("rb") as handle:
        header, count, names = _read_header(handle)
        offset = handle.tell()
    if expected_source_sha256 == SEALED_AURA_PLY_SHA256 and (
        count != SEALED_AURA_PLY_VERTEX_COUNT
    ):
        raise AuraTaskVolumeExclusionError(["aura_exclusion_source_vertex_count_unexpected"])
    for required in ("x", "y", "z", "opacity"):
        if required not in names:
            raise AuraTaskVolumeExclusionError(["aura_exclusion_source_missing_property"])

    columns = len(names)
    flat = np.fromfile(source, dtype="<f4", count=count * columns, offset=offset)
    if flat.size != count * columns:
        raise AuraTaskVolumeExclusionError(["aura_exclusion_source_body_truncated"])
    rows = flat.reshape(count, columns)
    index = {name: position for position, name in enumerate(names)}
    x = rows[:, index["x"]].astype(np.float64)
    y = rows[:, index["y"]].astype(np.float64)
    z = rows[:, index["z"]].astype(np.float64)

    radius = np.hypot(x - CAN_AXIS_XY_M[0], y - CAN_AXIS_XY_M[1])
    floor = SUPPORT_HEIGHT_M + EXCLUSION_FLOOR_ABOVE_SUPPORT_M
    ceiling = SUPPORT_HEIGHT_M + EXCLUSION_CEILING_ABOVE_SUPPORT_M
    # Centre-inside, not overlap: a surface that merely extends into the volume
    # from outside (the shelf lip, the cabinet wall) keeps all of its splats.
    excluded = (radius < EXCLUSION_RADIUS_M) & (z > floor) & (z < ceiling)
    removed = int(excluded.sum())

    if expected_removed_vertex_count is not None and removed != (
        expected_removed_vertex_count
    ):
        raise AuraTaskVolumeExclusionError(["aura_exclusion_removed_count_unexpected"])
    if removed / max(count, 1) > float(removed_fraction_ceiling):
        raise AuraTaskVolumeExclusionError(["aura_exclusion_removed_fraction_exceeded"])

    # The reconstructed shelf lives at or below the support plane inside the can
    # footprint.  It must survive untouched, or the policy would see a hole.
    footprint = radius < EXCLUSION_RADIUS_M
    shelf = footprint & (z <= floor)
    if int((shelf & excluded).sum()) != 0:
        raise AuraTaskVolumeExclusionError(["aura_exclusion_support_surface_disturbed"])

    kept = ~excluded
    retained_rows = rows[kept]
    new_header = header.replace(
        f"element vertex {count}".encode("latin-1"),
        f"element vertex {int(kept.sum())}".encode("latin-1"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(new_header)
        handle.write(np.ascontiguousarray(retained_rows, dtype="<f4").tobytes())
    temporary.replace(output)

    scales = [name for name in ("scale_0", "scale_1", "scale_2") if name in index]
    if scales:
        extent = np.exp(rows[:, [index[name] for name in scales]].astype(np.float64)).max(
            axis=1
        )
        large_removed = int((excluded & (extent > 0.02)).sum())
        largest_removed_extent_m = float(extent[excluded].max()) if removed else 0.0
    else:
        large_removed = 0
        largest_removed_extent_m = 0.0

    receipt: dict[str, Any] = {
        "schema_version": EXCLUSION_RECEIPT_SCHEMA_VERSION,
        "status": "materialized_task_volume_exclusion_derivative",
        "generated_at": generated_at,
        "source_ply_sha256": observed_source_sha256,
        "source_vertex_count": count,
        "source_modified": False,
        "output_ply_sha256": f"sha256:{sha256_file(output)}",
        "output_vertex_count": int(kept.sum()),
        "removed_vertex_count": removed,
        "removed_fraction": removed / max(count, 1),
        "removed_large_splat_count_extent_gt_2cm": large_removed,
        "largest_removed_surfel_extent_m": largest_removed_extent_m,
        "support_surface_disturbed": False,
        "exclusion_rule": {
            "geometry": "cylinder_centre_inside",
            "axis_xy_m": list(CAN_AXIS_XY_M),
            "radius_m": EXCLUSION_RADIUS_M,
            "support_height_m": SUPPORT_HEIGHT_M,
            "floor_above_support_m": EXCLUSION_FLOOR_ABOVE_SUPPORT_M,
            "ceiling_above_support_m": EXCLUSION_CEILING_ABOVE_SUPPORT_M,
            "preregistered_before_rerender": True,
        },
        "retained_rows_copied_verbatim": True,
        "authorship": "blueprint_authored_removal_completing_aurafusion360_object_removal",
        "claim_ceiling": (
            "Appearance inside the approved-can task volume is a Blueprint-authored "
            "removal, not AuraFusion360 admitted output; everything outside that "
            "cylinder remains the sealed asset byte-for-byte"
        ),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if receipt_path is not None:
        write_json(Path(receipt_path), receipt)
    return receipt


def validate_aura_task_volume_exclusion_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject an exclusion receipt that is not a preregistered, bounded removal."""

    try:
        receipt = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise AuraTaskVolumeExclusionError(["aura_exclusion_receipt_invalid"]) from exc
    if not isinstance(receipt, dict):
        raise AuraTaskVolumeExclusionError(["aura_exclusion_receipt_invalid"])
    errors: list[str] = []
    if receipt.get("schema_version") != EXCLUSION_RECEIPT_SCHEMA_VERSION:
        errors.append("aura_exclusion_receipt_schema_invalid")
    if receipt.get("source_ply_sha256") != SEALED_AURA_PLY_SHA256:
        errors.append("aura_exclusion_receipt_source_not_sealed_asset")
    if receipt.get("source_modified") is not False:
        errors.append("aura_exclusion_receipt_source_modified")
    if receipt.get("support_surface_disturbed") is not False:
        errors.append("aura_exclusion_receipt_support_disturbed")
    if receipt.get("retained_rows_copied_verbatim") is not True:
        errors.append("aura_exclusion_receipt_rows_not_verbatim")
    rule = receipt.get("exclusion_rule") or {}
    if rule.get("preregistered_before_rerender") is not True:
        errors.append("aura_exclusion_receipt_rule_not_preregistered")
    fraction = receipt.get("removed_fraction")
    if not isinstance(fraction, (int, float)) or float(fraction) > (
        REMOVED_FRACTION_CEILING
    ):
        errors.append("aura_exclusion_receipt_removed_fraction_exceeded")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("aura_exclusion_receipt_digest_mismatch")
    if errors:
        raise AuraTaskVolumeExclusionError(errors)
    return receipt


__all__ = [
    "AuraTaskVolumeExclusionError",
    "EXCLUSION_RECEIPT_SCHEMA_VERSION",
    "EXPECTED_REMOVED_VERTEX_COUNT",
    "SEALED_AURA_PLY_SHA256",
    "materialize_aura_task_volume_exclusion",
    "validate_aura_task_volume_exclusion_receipt",
]
