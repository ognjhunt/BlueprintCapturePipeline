"""Materialize qualified source-Gaussian removal receipts from upstream evidence.

Construction bindings must not accept caller-authored booleans as proof that a
source object was removed safely.  This module is the file-backed seam that joins
the frozen scene/task, calibrated masks, Gaussian ownership/held-out audit, and
replacement-coverage/inpainting decision into the compact qualification receipt
consumed by replacement construction.
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
    validate_scene_freeze,
    validate_task_freeze,
)
from .replacement_construction_bindings import (
    GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
    MASK_SET_QUALIFICATION_SCHEMA_VERSION,
)


OWNERSHIP_RECEIPT_SCHEMA_VERSION = "adp009b_gaussian_excision_ownership_receipt.v1"
HELDOUT_AUDIT_SCHEMA_VERSION = "adp009b_gaussian_excision_heldout_audit.v1"
EXCISION_JOIN_SCHEMA_VERSION = "articulated_excision_join.v1"
ADMITTED_INPAINTING_POLICIES = frozenset(
    {"inpainting_not_required", "narrow_mask_contained_seam_repair_only"}
)


class GaussianSourceRemovalQualificationError(ValueError):
    """Stable fail-closed errors for source-removal qualification joins."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path_value: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise GaussianSourceRemovalQualificationError([code])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GaussianSourceRemovalQualificationError([code]) from exc
    if not isinstance(value, dict):
        raise GaussianSourceRemovalQualificationError([code])
    return path, value


def _canonical_receipt(
    path_value: str | Path,
    *,
    schema_version: str,
    digest_field: str,
    code: str,
    status: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path, value = _read_json(path_value, code=code)
    errors: list[str] = []
    if value.get("schema_version") != schema_version:
        errors.append(f"{code}:schema")
    if status is not None and value.get("status") != status:
        errors.append(f"{code}:status")
    if value.get(digest_field) != canonical_digest(value, digest_field=digest_field):
        errors.append(f"{code}:digest")
    if errors:
        raise GaussianSourceRemovalQualificationError(errors)
    return path, value


def _file_record(path: Path, *, receipt: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "receipt_digest": receipt[digest_field],
    }


def _require_common_identity(
    *,
    receipt: Mapping[str, Any],
    expected: Mapping[str, str],
    role: str,
    errors: list[str],
) -> None:
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            errors.append(f"gaussian_source_removal_{role}_identity_mismatch:{field}")


def materialize_gaussian_source_removal_qualification(
    *,
    scene_freeze_path: str | Path,
    task_freeze_path: str | Path,
    mask_set_receipt_path: str | Path,
    ownership_receipt_path: str | Path,
    heldout_audit_receipt_path: str | Path,
    excision_join_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join upstream removal evidence into one construction-consumable receipt."""

    scene_path, raw_scene = _read_json(
        scene_freeze_path, code="gaussian_source_removal_scene_freeze_invalid"
    )
    task_path, raw_task = _read_json(
        task_freeze_path, code="gaussian_source_removal_task_freeze_invalid"
    )
    try:
        scene = validate_scene_freeze(raw_scene)
        task = validate_task_freeze(raw_task)
    except DualTaskRehearsalContractError as exc:
        raise GaussianSourceRemovalQualificationError(exc.errors) from exc
    errors: list[str] = []
    if task["scene_freeze_digest"] != scene["scene_freeze_digest"]:
        errors.append("gaussian_source_removal_task_scene_mismatch")

    mask_path, mask = _canonical_receipt(
        mask_set_receipt_path,
        schema_version=MASK_SET_QUALIFICATION_SCHEMA_VERSION,
        status="calibrated_mask_set_qualified",
        digest_field="receipt_digest",
        code="gaussian_source_removal_mask_set_invalid",
    )
    ownership_path, ownership = _canonical_receipt(
        ownership_receipt_path,
        schema_version=OWNERSHIP_RECEIPT_SCHEMA_VERSION,
        digest_field="receipt_digest",
        code="gaussian_source_removal_ownership_invalid",
    )
    heldout_path, heldout = _canonical_receipt(
        heldout_audit_receipt_path,
        schema_version=HELDOUT_AUDIT_SCHEMA_VERSION,
        status="heldout_gaussian_ownership_gate_passed",
        digest_field="receipt_digest",
        code="gaussian_source_removal_heldout_invalid",
    )
    join_path, join = _canonical_receipt(
        excision_join_receipt_path,
        schema_version=EXCISION_JOIN_SCHEMA_VERSION,
        status="join_admitted",
        digest_field="receipt_digest",
        code="gaussian_source_removal_excision_join_invalid",
    )

    source = task["source_object"]
    removal = task["removal_plan"]
    common_identity = {
        "scene_id": str(scene["selected_scene_id"]),
        "scene_freeze_digest": str(scene["scene_freeze_digest"]),
        "task_id": str(task["task_id"]),
        "task_freeze_digest": str(task["task_freeze_digest"]),
        "source_object_instance_id": str(source["instance_id"]),
        "removal_id": str(removal["removal_id"]),
        "mask_set_id": str(removal["mask_set_id"]),
    }
    _require_common_identity(
        receipt=mask, expected=common_identity, role="mask_set", errors=errors
    )
    if mask.get("source_scene_sha256") != scene["source_components"]["interiorgs"]["sha256"]:
        errors.append("gaussian_source_removal_mask_set_source_mismatch")
    if mask.get("calibrated_masks_qualified") is not True:
        errors.append("gaussian_source_removal_mask_set_not_qualified")

    if ownership.get("freeze_digest") != heldout.get("freeze_digest"):
        errors.append("gaussian_source_removal_ownership_heldout_freeze_mismatch")
    if heldout.get("ownership_receipt_digest") != ownership.get("receipt_digest"):
        errors.append("gaussian_source_removal_heldout_ownership_mismatch")
    if heldout.get("heldout_gate_passed") is not True:
        errors.append("gaussian_source_removal_heldout_not_passed")
    if heldout.get("replacement_coverage_sweep_authorized") is not True:
        errors.append("gaussian_source_removal_coverage_not_authorized")
    bindings = join.get("bindings") or {}
    claim_boundary = join.get("claim_boundary") or {}
    suppression = join.get("suppression") or {}
    if bindings.get("ownership_receipt_digest") != ownership.get("receipt_digest"):
        errors.append("gaussian_source_removal_join_ownership_mismatch")
    if join.get("inpainting_policy") not in ADMITTED_INPAINTING_POLICIES:
        errors.append("gaussian_source_removal_inpainting_policy_invalid")
    if claim_boundary.get("gaussian_ownership_established") is not True:
        errors.append("gaussian_source_removal_join_ownership_not_established")
    if suppression.get("mode") != "deletion" or suppression.get("canonical_scan_modified") is not True:
        errors.append("gaussian_source_removal_join_did_not_delete_source")

    ownership_counts = ownership.get("ownership") or {}
    if (
        ownership.get("heldout_cameras_accessed_for_classification") is not False
        or ownership.get("replacement_usd_inserted") is not False
        or ownership_counts.get("exhaustive") is not True
        or ownership_counts.get("pairwise_disjoint") is not True
    ):
        errors.append("gaussian_source_removal_ownership_not_qualified")

    if errors:
        raise GaussianSourceRemovalQualificationError(errors)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
        "status": "source_gaussian_removal_qualified",
        **common_identity,
        "source_scene_sha256": scene["source_components"]["interiorgs"]["sha256"],
        "mask_set_receipt_digest": mask["receipt_digest"],
        "ownership_receipt_digest": ownership["receipt_digest"],
        "heldout_audit_receipt_digest": heldout["receipt_digest"],
        "excision_join_receipt_digest": join["receipt_digest"],
        "source_removal_qualified": True,
        "retained_records_byte_exact": True,
        "protected_geometry_deleted": False,
        "inpainting_policy": join["inpainting_policy"],
        "deleted_index_set_sha256": bindings.get("deleted_index_set_sha256")
        or bindings.get("owned_index_set_sha256"),
        "retained_scene_ply_sha256": bindings.get("retained_scene_ply_sha256"),
        "upstream_evidence": {
            "scene_freeze": _file_record(
                scene_path, receipt=scene, digest_field="scene_freeze_digest"
            ),
            "task_freeze": _file_record(
                task_path, receipt=task, digest_field="task_freeze_digest"
            ),
            "mask_set": _file_record(
                mask_path, receipt=mask, digest_field="receipt_digest"
            ),
            "ownership": _file_record(
                ownership_path, receipt=ownership, digest_field="receipt_digest"
            ),
            "heldout_audit": _file_record(
                heldout_path, receipt=heldout, digest_field="receipt_digest"
            ),
            "excision_join": _file_record(
                join_path, receipt=join, digest_field="receipt_digest"
            ),
        },
        "claim_boundary": {
            "source_gaussian_removal_qualified": True,
            "protected_scene_geometry_deleted": False,
            "replacement_native_import_qualified": False,
            "learned_policy_outcomes_used": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "ADMITTED_INPAINTING_POLICIES",
    "GaussianSourceRemovalQualificationError",
    "materialize_gaussian_source_removal_qualification",
]
