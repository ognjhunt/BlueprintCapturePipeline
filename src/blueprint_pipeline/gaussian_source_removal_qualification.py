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
import math
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
COVERAGE_CONDITIONED_CUTOUT_SCHEMA_VERSION = "adp009b_coverage_conditioned_cutout.v1"
OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA_VERSION = (
    "adp009b_ownership_coverage_cutout_candidate.v1"
)
DELETED_SOURCE_LAYER_COVERAGE_SCHEMA_VERSION = "articulated_excision_coverage.v1"
DELETED_SOURCE_LAYER_COVERAGE_STATUS = (
    "deleted_source_layer_replacement_coverage_qualified"
)
SOURCE_LAYER_COVERAGE_AUDIT_SCHEMA_VERSION = (
    "adp009b_source_layer_replacement_coverage_audit.v1"
)
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


def _verified_relative_artifact(
    *, record: object, root: Path, code: str
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(record, Mapping):
        raise GaussianSourceRemovalQualificationError([code])
    relative = str(record.get("relative_path") or "")
    path = (root / relative).resolve()
    if (
        not relative
        or not path.is_file()
        or path.is_symlink()
        or record.get("size_bytes") != path.stat().st_size
        or record.get("sha256") != _sha256(path)
    ):
        raise GaussianSourceRemovalQualificationError([code])
    return path, record


def _verify_source_layer_coverage_provenance(
    *, coverage: Mapping[str, Any], errors: list[str]
) -> None:
    """Re-open zero-residue evidence so a self-sealed coverage JSON is insufficient."""

    record = coverage.get("source_layer_coverage_audit")
    if not isinstance(record, Mapping):
        errors.append("gaussian_source_removal_source_layer_audit_missing")
        return
    path_value = str(record.get("path") or "")
    path = Path(path_value).expanduser().resolve()
    if (
        not path_value
        or not path.is_file()
        or path.is_symlink()
        or record.get("size_bytes") != path.stat().st_size
        or record.get("sha256") != _sha256(path)
    ):
        errors.append("gaussian_source_removal_source_layer_audit_file_invalid")
        return
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("gaussian_source_removal_source_layer_audit_file_invalid")
        return
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != SOURCE_LAYER_COVERAGE_AUDIT_SCHEMA_VERSION
        or audit.get("status") != "source_layer_coverage_measured"
        or audit.get("manifest_digest")
        != canonical_digest(audit, digest_field="manifest_digest")
        or record.get("manifest_digest") != audit.get("manifest_digest")
        or audit.get("source_layer_splat_digest")
        != coverage.get("source_layer_splat_digest")
        or audit.get("camera_ids") != coverage.get("camera_ids")
    ):
        errors.append("gaussian_source_removal_source_layer_audit_join_invalid")
        return
    threshold = audit.get("significant_alpha_threshold")
    margin = audit.get("coverage_margin_pixels")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isclose(float(threshold), 1.0 / 255.0, abs_tol=1e-12, rel_tol=0.0)
        or isinstance(margin, bool)
        or not isinstance(margin, int)
        or margin < 1
    ):
        errors.append("gaussian_source_removal_source_layer_audit_policy_invalid")
        return
    audit_cells = audit.get("cells")
    coverage_cells = coverage.get("cells")
    if (
        not isinstance(audit_cells, list)
        or not audit_cells
        or not isinstance(coverage_cells, list)
        or len(audit_cells) != len(coverage_cells)
    ):
        errors.append("gaussian_source_removal_source_layer_audit_cells_invalid")
        return
    for audit_cell, coverage_cell in zip(audit_cells, coverage_cells, strict=True):
        if not isinstance(audit_cell, Mapping) or not isinstance(coverage_cell, Mapping):
            errors.append("gaussian_source_removal_source_layer_audit_cells_invalid")
            return
        values = (
            audit_cell.get("uncovered_significant_pixel_count"),
            audit_cell.get("largest_uncovered_component_pixels"),
            audit_cell.get("uncovered_alpha_sum"),
            audit_cell.get("uncovered_alpha_fraction"),
        )
        if (
            audit_cell.get("camera_id") != coverage_cell.get("camera_id")
            or values[0] != 0
            or values[1] != 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) > 1e-12
                for value in values[2:]
            )
            or coverage_cell.get("residual_significant_pixels") != 0
            or coverage_cell.get("residual_max_connected_component_pixels") != 0
            or coverage_cell.get("outside_mask_changed_pixels") != 0
        ):
            errors.append("gaussian_source_removal_source_layer_residue_observed")
            return


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
    heldout_audit_receipt_path: str | Path | None,
    excision_join_receipt_path: str | Path,
    output_path: str | Path,
    coverage_conditioned_cutout_receipt_path: str | Path | None = None,
    coverage_cutout_candidate_path: str | Path | None = None,
    source_layer_coverage_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Join source ownership or coverage-conditioned deletion evidence.

    The standard lane establishes factual ownership in a held-out audit.  The
    coverage-conditioned lane instead deletes a calibration-only
    owned-plus-ambiguous candidate only after actual USD depth proves that its
    complete source layer has zero residual in every frozen camera/state.  The
    latter remains a visibility/safety qualification, never a factual ownership
    claim for the ambiguous records.
    """

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
    heldout_path: Path | None = None
    heldout: dict[str, Any] | None = None
    if heldout_audit_receipt_path is not None:
        heldout_path, heldout = _canonical_receipt(
            heldout_audit_receipt_path,
            schema_version=HELDOUT_AUDIT_SCHEMA_VERSION,
            status=None,
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
    coverage_mode = (
        coverage_conditioned_cutout_receipt_path is not None
        or coverage_cutout_candidate_path is not None
        or source_layer_coverage_receipt_path is not None
    )
    cutout_path: Path | None = None
    cutout: dict[str, Any] | None = None
    candidate_path: Path | None = None
    candidate: dict[str, Any] | None = None
    coverage_path: Path | None = None
    coverage: dict[str, Any] | None = None
    if coverage_mode:
        if (
            coverage_conditioned_cutout_receipt_path is None
            or coverage_cutout_candidate_path is None
            or source_layer_coverage_receipt_path is None
        ):
            errors.append("gaussian_source_removal_coverage_conditioned_inputs_incomplete")
        else:
            cutout_path, cutout = _canonical_receipt(
                coverage_conditioned_cutout_receipt_path,
                schema_version=COVERAGE_CONDITIONED_CUTOUT_SCHEMA_VERSION,
                status="coverage_conditioned_cutout_admitted",
                digest_field="receipt_digest",
                code="gaussian_source_removal_coverage_conditioned_cutout_invalid",
            )
            candidate_path, candidate = _canonical_receipt(
                coverage_cutout_candidate_path,
                schema_version=OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA_VERSION,
                status=(
                    "ownership_coverage_cutout_materialized_pending_actual_usd_source_layer_coverage"
                ),
                digest_field="receipt_digest",
                code="gaussian_source_removal_coverage_cutout_candidate_invalid",
            )
            coverage_path, coverage = _canonical_receipt(
                source_layer_coverage_receipt_path,
                schema_version=DELETED_SOURCE_LAYER_COVERAGE_SCHEMA_VERSION,
                status=DELETED_SOURCE_LAYER_COVERAGE_STATUS,
                digest_field="receipt_digest",
                code="gaussian_source_removal_source_layer_coverage_invalid",
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

    bindings = join.get("bindings") or {}
    claim_boundary = join.get("claim_boundary") or {}
    suppression = join.get("suppression") or {}
    if join.get("inpainting_policy") not in ADMITTED_INPAINTING_POLICIES:
        errors.append("gaussian_source_removal_inpainting_policy_invalid")
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

    if coverage_mode:
        if cutout is None or candidate is None or coverage is None:
            pass
        else:
            source_ownership_digest = cutout.get("source_ownership_receipt_digest")
            if source_ownership_digest != ownership.get("receipt_digest"):
                errors.append(
                    "gaussian_source_removal_coverage_conditioned_ownership_mismatch"
                )
            if cutout.get("coverage_receipt_digest") != coverage.get("receipt_digest"):
                errors.append(
                    "gaussian_source_removal_coverage_conditioned_coverage_mismatch"
                )
            source_splat = ownership.get("source_standard_splat") or {}
            candidate_ownership = candidate.get("source_ownership_receipt") or {}
            candidate_source = candidate.get("source_standard_splat") or {}
            if (
                cutout.get("bound_cutout_candidate_digest")
                != candidate.get("receipt_digest")
                or candidate_ownership.get("receipt_digest")
                != ownership.get("receipt_digest")
                or candidate_source.get("sha256") != source_splat.get("sha256")
            ):
                errors.append(
                    "gaussian_source_removal_coverage_cutout_candidate_join_mismatch"
                )
            candidate_outputs = candidate.get("outputs") or {}
            try:
                _deleted_path, deleted_record = _verified_relative_artifact(
                    record=candidate_outputs.get("deleted_source_indices"),
                    root=candidate_path.parent if candidate_path is not None else Path(),
                    code="gaussian_source_removal_coverage_cutout_deleted_artifact_invalid",
                )
                _retained_path, retained_record = _verified_relative_artifact(
                    record=candidate_outputs.get("retained_scene_gaussians"),
                    root=candidate_path.parent if candidate_path is not None else Path(),
                    code="gaussian_source_removal_coverage_cutout_retained_artifact_invalid",
                )
            except GaussianSourceRemovalQualificationError as exc:
                errors.extend(exc.codes)
                deleted_record = {}
                retained_record = {}
            if (
                deleted_record.get("sha256") != cutout.get("deleted_index_set_sha256")
                or retained_record.get("sha256")
                != cutout.get("retained_scene_ply_sha256")
            ):
                errors.append(
                    "gaussian_source_removal_coverage_cutout_artifact_join_mismatch"
                )
            if (
                coverage.get("coverage_scope") != "deleted_source_layer"
                or coverage.get("coverage_qualified") is not True
                or coverage.get("all_deleted_source_contribution_occluded") is not True
                or coverage.get("source_layer_splat_digest")
                != source_splat.get("sha256")
            ):
                errors.append(
                    "gaussian_source_removal_coverage_conditioned_source_layer_invalid"
                )
            _verify_source_layer_coverage_provenance(
                coverage=coverage, errors=errors
            )
            if bindings.get("ownership_receipt_digest") != cutout.get("receipt_digest"):
                errors.append("gaussian_source_removal_join_cutout_mismatch")
            if bindings.get("source_ownership_receipt_digest") != ownership.get(
                "receipt_digest"
            ):
                errors.append("gaussian_source_removal_join_source_ownership_mismatch")
            if bindings.get("coverage_receipt_digest") != coverage.get("receipt_digest"):
                errors.append("gaussian_source_removal_join_coverage_mismatch")
            if claim_boundary.get("gaussian_ownership_established") is not False:
                errors.append(
                    "gaussian_source_removal_join_coverage_conditioned_claim_invalid"
                )
            if claim_boundary.get("visibility_after_replacement_is_the_criterion") is not True:
                errors.append(
                    "gaussian_source_removal_join_coverage_visibility_claim_invalid"
                )
            if join.get("inpainting_policy") != "inpainting_not_required":
                errors.append(
                    "gaussian_source_removal_coverage_conditioned_inpainting_not_strict"
                )
    else:
        if heldout is None:
            errors.append("gaussian_source_removal_heldout_missing")
        else:
            if ownership.get("freeze_digest") != heldout.get("freeze_digest"):
                errors.append("gaussian_source_removal_ownership_heldout_freeze_mismatch")
            if heldout.get("ownership_receipt_digest") != ownership.get("receipt_digest"):
                errors.append("gaussian_source_removal_heldout_ownership_mismatch")
            if heldout.get("status") != "heldout_gaussian_ownership_gate_passed":
                errors.append("gaussian_source_removal_heldout_not_passed")
            if heldout.get("heldout_gate_passed") is not True:
                errors.append("gaussian_source_removal_heldout_not_passed")
            if heldout.get("replacement_coverage_sweep_authorized") is not True:
                errors.append("gaussian_source_removal_coverage_not_authorized")
        if bindings.get("ownership_receipt_digest") != ownership.get("receipt_digest"):
            errors.append("gaussian_source_removal_join_ownership_mismatch")
        if claim_boundary.get("gaussian_ownership_established") is not True:
            errors.append("gaussian_source_removal_join_ownership_not_established")

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
        "heldout_audit_receipt_digest": (
            heldout["receipt_digest"] if heldout is not None else None
        ),
        "coverage_conditioned_cutout_receipt_digest": (
            cutout["receipt_digest"] if cutout is not None else None
        ),
        "source_layer_coverage_receipt_digest": (
            coverage["receipt_digest"] if coverage is not None else None
        ),
        "excision_join_receipt_digest": join["receipt_digest"],
        "ownership_proof_mode": (
            "coverage_conditioned_visibility" if coverage_mode else "heldout_factual_ownership"
        ),
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
            **(
                {
                    "heldout_audit": _file_record(
                        heldout_path, receipt=heldout, digest_field="receipt_digest"
                    )
                }
                if heldout_path is not None and heldout is not None
                else {}
            ),
            **(
                {
                    "coverage_conditioned_cutout": _file_record(
                        cutout_path, receipt=cutout, digest_field="receipt_digest"
                    ),
                    "coverage_cutout_candidate": _file_record(
                        candidate_path, receipt=candidate, digest_field="receipt_digest"
                    ),
                    "source_layer_coverage": _file_record(
                        coverage_path, receipt=coverage, digest_field="receipt_digest"
                    ),
                }
                if cutout_path is not None
                and cutout is not None
                and candidate_path is not None
                and candidate is not None
                and coverage_path is not None
                and coverage is not None
                else {}
            ),
            "excision_join": _file_record(
                join_path, receipt=join, digest_field="receipt_digest"
            ),
        },
        "claim_boundary": {
            "source_gaussian_removal_qualified": True,
            "protected_scene_geometry_deleted": False,
            "factual_gaussian_ownership_established": not coverage_mode,
            "coverage_conditioned_visibility_qualified": coverage_mode,
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
