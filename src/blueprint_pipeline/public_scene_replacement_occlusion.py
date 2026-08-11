"""Materialize an exact 3DGS cutout admitted by replacement occlusion evidence.

The released segmentation method and the replacement renderer remain replaceable
adapters. This module owns the stable evidence boundary: contribution totals,
three-way Gaussian ownership, per-cell mesh-depth coverage, byte-exact retained
rows, and a fail-closed inpainting disposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    validate_task_freeze_set,
)
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply_subset_exact,
)

REQUEST_SCHEMA = "adp009b_replacement_occlusion_request.v1"
CONTRIBUTION_SCHEMA = "adp009b_gaussian_contribution_evidence.v1"
COVERAGE_SCHEMA = "adp009b_replacement_depth_coverage.v1"
RECEIPT_SCHEMA = "adp009b_replacement_occlusion_receipt.v1"
OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA = (
    "adp009b_ownership_coverage_cutout_candidate.v1"
)
OWNERSHIP_COVERAGE_CUTOUT_SET_SCHEMA = "adp009b_ownership_coverage_cutout_set.v1"

_OWNERSHIP_RECEIPT_SCHEMA = "adp009b_gaussian_excision_ownership_receipt.v1"
_OWNERSHIP_RECEIPT_STATUSES = frozenset(
    {
        "three_way_ownership_materialized_heldout_not_evaluated",
        "three_way_ownership_materialized_by_frozen_conservative_aggregation_heldout_not_evaluated",
    }
)

LABEL_RETAINED = np.uint8(0)
LABEL_OWNED_DELETED = np.uint8(1)
LABEL_AMBIGUOUS_RETAINED = np.uint8(2)
LABEL_AMBIGUOUS_DELETED = np.uint8(3)


class ReplacementOcclusionError(ValueError):
    """Stable fail-closed cutout and coverage errors."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _input_record(path: Path) -> dict[str, Any]:
    """Record a sealed input without pretending it lives under an output root."""

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplacementOcclusionError([code]) from exc
    if not isinstance(value, dict):
        raise ReplacementOcclusionError([code])
    return value


def _under(root: Path, path: str | Path, *, code: str) -> Path:
    root = root.expanduser().resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.expanduser().resolve()
    if candidate != root and root not in candidate.parents:
        raise ReplacementOcclusionError([code])
    return candidate


def _verify_artifact(root: Path, record: Mapping[str, Any], *, code: str) -> Path:
    path = _under(root, str(record.get("relative_path") or ""), code=code)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ReplacementOcclusionError([code])
    return path


def _finite_number(value: Any, *, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def build_replacement_occlusion_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and digest a frozen scene-neutral cutout request."""

    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReplacementOcclusionError(["replacement_occlusion_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("replacement_occlusion_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1" or request.get("adp_item") != "ADP-009B":
        errors.append("replacement_occlusion_program_identity_invalid")
    if request.get("frozen_before_cutout") is not True:
        errors.append("replacement_occlusion_request_not_frozen")
    if request.get("learned_policy_outcomes_observed") is not False:
        errors.append("replacement_occlusion_policy_outcome_leakage")
    if {"status", "admitted", "inpainting_not_required"}.intersection(request):
        errors.append("replacement_occlusion_caller_outcome_forbidden")
    scene = request.get("scene")
    if not isinstance(scene, Mapping):
        errors.append("replacement_occlusion_scene_missing")
    else:
        for key in ("publisher_scene_id", "target_instance_id", "target_semantic_label"):
            if not str(scene.get(key) or "").strip():
                errors.append(f"replacement_occlusion_scene_{key}_missing")
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("replacement_occlusion_inputs_missing")
    else:
        for key in ("source_standard_splat", "contribution_manifest", "coverage_manifest"):
            record = inputs.get(key)
            if not isinstance(record, Mapping):
                errors.append(f"replacement_occlusion_{key}_missing")
                continue
            if (
                not str(record.get("relative_path") or "")
                or not isinstance(record.get("size_bytes"), int)
                or not str(record.get("sha256") or "").startswith("sha256:")
            ):
                errors.append(f"replacement_occlusion_{key}_invalid")
    policy = request.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("replacement_occlusion_policy_missing")
    else:
        low = _finite_number(
            policy.get("retained_max_foreground_fraction"), minimum=0.0, maximum=1.0
        )
        high = _finite_number(
            policy.get("owned_min_foreground_fraction"), minimum=0.0, maximum=1.0
        )
        if low is None or high is None or not low < high:
            errors.append("replacement_occlusion_ownership_thresholds_invalid")
        for key, minimum, maximum in (
            ("minimum_total_contribution", 0.0, 1e12),
            ("minimum_cell_visible_contribution", 0.0, 1e12),
            ("maximum_ambiguous_uncovered_fraction", 0.0, 1.0),
            ("maximum_ambiguous_uncovered_contribution", 0.0, 1e12),
            ("confident_removal_alpha_threshold", 0.0, 1.0),
            ("maximum_residual_alpha_fraction_per_cell", 0.0, 1.0),
            ("door_angle_readback_tolerance_deg", 0.0, 5.0),
        ):
            if _finite_number(policy.get(key), minimum=minimum, maximum=maximum) is None:
                errors.append(f"replacement_occlusion_{key}_invalid")
        max_pixels = policy.get("maximum_confident_uncovered_pixels_per_cell")
        if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels < 0:
            errors.append("replacement_occlusion_maximum_confident_uncovered_pixels_invalid")
        camera_ids = policy.get("required_camera_ids")
        if (
            not isinstance(camera_ids, list)
            or not camera_ids
            or len(camera_ids) != len(set(map(str, camera_ids)))
            or any(not str(item).strip() for item in camera_ids)
        ):
            errors.append("replacement_occlusion_required_camera_ids_invalid")
        angles = policy.get("required_door_angles_deg")
        if (
            not isinstance(angles, list)
            or not angles
            or any(_finite_number(item, minimum=-180.0, maximum=180.0) is None for item in angles)
        ):
            errors.append("replacement_occlusion_required_door_angles_invalid")
        elif any(float(b) <= float(a) for a, b in zip(angles, angles[1:], strict=False)):
            errors.append("replacement_occlusion_required_door_angles_not_increasing")
    if errors:
        raise ReplacementOcclusionError(errors)
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        raise ReplacementOcclusionError(["replacement_occlusion_request_digest_mismatch"])
    request["request_digest"] = expected
    return request


def classify_gaussian_contributions(
    foreground: np.ndarray,
    background: np.ndarray,
    *,
    retained_max_foreground_fraction: float,
    owned_min_foreground_fraction: float,
    minimum_total_contribution: float,
) -> dict[str, np.ndarray]:
    """Turn released-method alpha/transmittance totals into three ownership states."""

    foreground = np.asarray(foreground, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if foreground.ndim != 1 or foreground.shape != background.shape or foreground.size == 0:
        raise ReplacementOcclusionError(["replacement_occlusion_contribution_shape_invalid"])
    if (
        not np.isfinite(foreground).all()
        or not np.isfinite(background).all()
        or np.any(foreground < 0)
        or np.any(background < 0)
    ):
        raise ReplacementOcclusionError(["replacement_occlusion_contribution_values_invalid"])
    if not 0 <= retained_max_foreground_fraction < owned_min_foreground_fraction <= 1:
        raise ReplacementOcclusionError(["replacement_occlusion_ownership_thresholds_invalid"])
    total = foreground + background
    enough = total >= float(minimum_total_contribution)
    fraction = np.zeros_like(total)
    np.divide(foreground, total, out=fraction, where=total > 0)
    owned = enough & (fraction >= float(owned_min_foreground_fraction))
    retained = ~enough | (fraction <= float(retained_max_foreground_fraction))
    ambiguous = ~(owned | retained)
    return {
        "foreground_fraction": fraction,
        "total_contribution": total,
        "owned": owned,
        "retained": retained,
        "ambiguous": ambiguous,
    }


def coverage_safe_ambiguous(
    ambiguous_indices: np.ndarray,
    evidence_indices: np.ndarray,
    visible_contribution: np.ndarray,
    uncovered_contribution: np.ndarray,
    *,
    minimum_cell_visible_contribution: float,
    maximum_uncovered_fraction: float,
    maximum_uncovered_contribution: float,
) -> np.ndarray:
    """Return ambiguous Gaussians whose contribution is covered in every seen cell."""

    ambiguous_indices = np.asarray(ambiguous_indices, dtype=np.int64)
    evidence_indices = np.asarray(evidence_indices, dtype=np.int64)
    visible = np.asarray(visible_contribution, dtype=np.float64)
    uncovered = np.asarray(uncovered_contribution, dtype=np.float64)
    if (
        evidence_indices.ndim != 1
        or visible.ndim != 2
        or uncovered.shape != visible.shape
        or visible.shape[1] != evidence_indices.size
        or not np.isfinite(visible).all()
        or not np.isfinite(uncovered).all()
        or np.any(visible < 0)
        or np.any(uncovered < 0)
        or np.any(uncovered > visible + 1e-12)
    ):
        raise ReplacementOcclusionError(["replacement_occlusion_ambiguous_coverage_invalid"])
    if len(set(evidence_indices.tolist())) != evidence_indices.size:
        raise ReplacementOcclusionError(["replacement_occlusion_coverage_indices_duplicate"])
    lookup = {int(index): column for column, index in enumerate(evidence_indices)}
    if any(int(index) not in lookup for index in ambiguous_indices):
        raise ReplacementOcclusionError(["replacement_occlusion_ambiguous_coverage_incomplete"])
    safe = np.zeros(ambiguous_indices.size, dtype=bool)
    for output_column, index in enumerate(ambiguous_indices):
        column = lookup[int(index)]
        seen = visible[:, column] >= float(minimum_cell_visible_contribution)
        if not np.any(seen):
            continue
        ratios = np.zeros(visible.shape[0], dtype=np.float64)
        np.divide(uncovered[:, column], visible[:, column], out=ratios, where=visible[:, column] > 0)
        safe[output_column] = bool(
            np.all(
                ~seen
                | (
                    (uncovered[:, column] <= float(maximum_uncovered_contribution))
                    & (ratios <= float(maximum_uncovered_fraction))
                )
            )
        )
    return safe


def select_direct_calibration_evidence_expansion(
    candidate_indices: np.ndarray,
    owned: np.ndarray,
    protected_camera_count: np.ndarray,
    core_camera_count: np.ndarray,
    core_fraction: np.ndarray,
    geometry_score: np.ndarray,
    *,
    minimum_core_camera_count: int,
    minimum_core_fraction: float,
    minimum_geometry_score: float,
) -> np.ndarray:
    """Select non-owned candidates supported directly by calibration evidence.

    This deliberately ignores a neighborhood-smoothed score: smoothing may
    make an appliance-edge Gaussian inherit nearby background uncertainty even
    when its own evidence is target-only in multiple calibration views.
    Held-out pixels and replacement outcomes are not inputs to this selector.
    """

    indices = np.asarray(candidate_indices, dtype=np.int64)
    owned_values = np.asarray(owned, dtype=bool)
    protected = np.asarray(protected_camera_count)
    core_count = np.asarray(core_camera_count)
    fraction = np.asarray(core_fraction, dtype=np.float64)
    geometry = np.asarray(geometry_score, dtype=np.float64)
    size = owned_values.size
    if (
        indices.ndim != 1
        or len(set(indices.tolist())) != indices.size
        or np.any(indices < 0)
        or np.any(indices >= size)
        or any(
            values.ndim != 1 or values.size != size
            for values in (protected, core_count, fraction, geometry)
        )
        or not np.isfinite(fraction).all()
        or not np.isfinite(geometry).all()
        or np.any(protected < 0)
        or np.any(core_count < 0)
        or isinstance(minimum_core_camera_count, bool)
        or not isinstance(minimum_core_camera_count, int)
        or minimum_core_camera_count < 1
        or not 0.0 <= minimum_core_fraction <= 1.0
        or not 0.0 <= minimum_geometry_score <= 1.0
    ):
        raise ReplacementOcclusionError(
            ["replacement_occlusion_direct_evidence_input_invalid"]
        )
    selected = indices[
        (~owned_values[indices])
        & (protected[indices] == 0)
        & (core_count[indices] >= minimum_core_camera_count)
        & (fraction[indices] >= minimum_core_fraction)
        & (geometry[indices] >= minimum_geometry_score)
    ]
    return selected.astype(np.int64, copy=False)


def _load_array(path: Path, code: str) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise ReplacementOcclusionError([code])
    try:
        return np.asarray(np.load(path, allow_pickle=False))
    except (OSError, ValueError) as exc:
        raise ReplacementOcclusionError([code]) from exc


def materialize_direct_evidence_expansion_candidate(
    *,
    source_standard_splat_path: str | Path,
    owned_indices_path: str | Path,
    candidate_indices_path: str | Path,
    protected_camera_count_path: str | Path,
    core_camera_count_path: str | Path,
    core_fraction_path: str | Path,
    geometry_score_path: str | Path,
    output_root: str | Path,
    minimum_core_camera_count: int,
    minimum_core_fraction: float,
    minimum_geometry_score: float,
) -> dict[str, Any]:
    """Materialize a byte-exact diagnostic cutout from direct evidence only."""

    source = Path(source_standard_splat_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReplacementOcclusionError(
            ["replacement_occlusion_direct_output_not_empty"]
        )
    splat = read_standard_3dgs_ply(source)
    paths = {
        "owned_indices": Path(owned_indices_path).expanduser().resolve(),
        "candidate_indices": Path(candidate_indices_path).expanduser().resolve(),
        "protected_camera_count": Path(protected_camera_count_path)
        .expanduser()
        .resolve(),
        "core_camera_count": Path(core_camera_count_path).expanduser().resolve(),
        "core_fraction": Path(core_fraction_path).expanduser().resolve(),
        "geometry_score": Path(geometry_score_path).expanduser().resolve(),
    }
    arrays = {
        name: _load_array(path, f"replacement_occlusion_direct_{name}_invalid")
        for name, path in paths.items()
    }
    owned_indices = np.asarray(arrays["owned_indices"], dtype=np.int64)
    candidate_indices = np.asarray(arrays["candidate_indices"], dtype=np.int64)
    if (
        owned_indices.ndim != 1
        or candidate_indices.ndim != 1
        or len(set(owned_indices.tolist())) != owned_indices.size
        or len(set(candidate_indices.tolist())) != candidate_indices.size
        or np.any(owned_indices < 0)
        or np.any(owned_indices >= splat.count)
    ):
        raise ReplacementOcclusionError(
            ["replacement_occlusion_direct_indices_invalid"]
        )
    owned = np.zeros(splat.count, dtype=bool)
    owned[owned_indices] = True
    expansion = select_direct_calibration_evidence_expansion(
        candidate_indices,
        owned,
        arrays["protected_camera_count"],
        arrays["core_camera_count"],
        arrays["core_fraction"],
        arrays["geometry_score"],
        minimum_core_camera_count=minimum_core_camera_count,
        minimum_core_fraction=minimum_core_fraction,
        minimum_geometry_score=minimum_geometry_score,
    )
    deleted = np.union1d(owned_indices, expansion).astype(np.int64)
    retained = np.setdiff1d(
        np.arange(splat.count, dtype=np.int64), deleted, assume_unique=True
    )
    output.mkdir(parents=True, exist_ok=True)
    deleted_indices_path = output / "deleted_source_indices.npy"
    retained_indices_path = output / "retained_source_indices.npy"
    expansion_indices_path = output / "direct_evidence_expansion_indices.npy"
    np.save(deleted_indices_path, deleted, allow_pickle=False)
    np.save(retained_indices_path, retained, allow_pickle=False)
    np.save(expansion_indices_path, expansion, allow_pickle=False)
    deleted_ply = write_standard_3dgs_ply_subset_exact(
        source, output / "deleted_source_gaussians.ply", deleted
    )
    retained_ply = write_standard_3dgs_ply_subset_exact(
        source, output / "retained_scene_gaussians.ply", retained
    )
    preservation = verify_standard_3dgs_ply_subset_exact(
        source, retained_ply, retained
    )
    if preservation.get("retained_rows_byte_exact") is not True:
        raise ReplacementOcclusionError(
            ["replacement_occlusion_direct_retained_rows_changed"]
        )
    receipt: dict[str, Any] = {
        "schema_version": "adp009b_direct_evidence_expansion_candidate.v1",
        "status": "diagnostic_cutout_materialized_pending_exact_camera_audit",
        "source_standard_splat": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "inputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "policy": {
            "minimum_core_camera_count": minimum_core_camera_count,
            "minimum_core_fraction": float(minimum_core_fraction),
            "minimum_geometry_score": float(minimum_geometry_score),
            "maximum_protected_camera_count": 0,
            "neighborhood_score_used": False,
            "heldout_pixels_used": False,
            "learned_policy_outcomes_used": False,
        },
        "counts": {
            "source": splat.count,
            "owned": int(owned_indices.size),
            "direct_evidence_expansion": int(expansion.size),
            "deleted_total": int(deleted.size),
            "retained_total": int(retained.size),
        },
        "preservation": preservation,
        "outputs": {
            "deleted_source_indices": _record(deleted_indices_path, output),
            "retained_source_indices": _record(retained_indices_path, output),
            "direct_evidence_expansion_indices": _record(
                expansion_indices_path, output
            ),
            "deleted_source_gaussians": _record(deleted_ply, output),
            "retained_scene_gaussians": _record(retained_ply, output),
        },
        "claim_ceiling": "diagnostic_byte_exact_cutout_pending_heldout_and_hybrid_review",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output / "adp009b_direct_evidence_expansion_candidate.v1.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_bound_index_union_candidate(
    *,
    source_standard_splat_path: str | Path,
    required_deletion_indices_path: str | Path,
    registered_volume_indices_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize the exact union of owned and registered-volume indices.

    This is a construction candidate, not a success assertion.  Replacement
    coverage and authorized seam containment remain independent downstream
    gates.
    """

    source = Path(source_standard_splat_path).expanduser().resolve()
    required_path = Path(required_deletion_indices_path).expanduser().resolve()
    registered_path = Path(registered_volume_indices_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReplacementOcclusionError(
            ["replacement_occlusion_bound_union_output_not_empty"]
        )
    splat = read_standard_3dgs_ply(source)
    required = np.asarray(
        _load_array(
            required_path, "replacement_occlusion_bound_required_indices_invalid"
        ),
        dtype=np.int64,
    )
    registered = np.asarray(
        _load_array(
            registered_path,
            "replacement_occlusion_bound_registered_indices_invalid",
        ),
        dtype=np.int64,
    )
    if any(
        values.ndim != 1
        or len(set(values.tolist())) != values.size
        or np.any(values < 0)
        or np.any(values >= splat.count)
        for values in (required, registered)
    ):
        raise ReplacementOcclusionError(
            ["replacement_occlusion_bound_union_indices_invalid"]
        )
    deleted = np.union1d(required, registered).astype(np.int64)
    registered_only = np.setdiff1d(registered, required, assume_unique=True)
    retained = np.setdiff1d(
        np.arange(splat.count, dtype=np.int64), deleted, assume_unique=True
    )
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "deleted_source_indices": output / "deleted_source_indices.npy",
        "retained_source_indices": output / "retained_source_indices.npy",
        "registered_volume_only_indices": output
        / "registered_volume_only_indices.npy",
    }
    np.save(outputs["deleted_source_indices"], deleted, allow_pickle=False)
    np.save(outputs["retained_source_indices"], retained, allow_pickle=False)
    np.save(
        outputs["registered_volume_only_indices"],
        registered_only,
        allow_pickle=False,
    )
    deleted_ply = write_standard_3dgs_ply_subset_exact(
        source, output / "deleted_source_gaussians.ply", deleted
    )
    retained_ply = write_standard_3dgs_ply_subset_exact(
        source, output / "retained_scene_gaussians.ply", retained
    )
    preservation = verify_standard_3dgs_ply_subset_exact(
        source, retained_ply, retained
    )
    if preservation.get("retained_rows_byte_exact") is not True:
        raise ReplacementOcclusionError(
            ["replacement_occlusion_bound_union_retained_rows_changed"]
        )
    receipt: dict[str, Any] = {
        "schema_version": "adp009b_bound_index_union_candidate.v1",
        "status": "bound_cutout_materialized_pending_coverage_and_seam_gates",
        "source_standard_splat": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "required_deletion_indices": {
            "path": str(required_path),
            "size_bytes": required_path.stat().st_size,
            "sha256": _sha256(required_path),
        },
        "registered_volume_indices": {
            "path": str(registered_path),
            "size_bytes": registered_path.stat().st_size,
            "sha256": _sha256(registered_path),
        },
        "selection": {
            "rule": "set_union_of_required_deletion_and_registered_volume_indices",
            "heldout_pixels_used_to_select_indices": False,
            "learned_policy_outcomes_used": False,
            "caller_asserted_coverage": False,
        },
        "counts": {
            "source": splat.count,
            "required_deletion": int(required.size),
            "registered_volume": int(registered.size),
            "registered_volume_only": int(registered_only.size),
            "deleted_total": int(deleted.size),
            "retained_total": int(retained.size),
        },
        "preservation": preservation,
        "outputs": {
            **{name: _record(path, output) for name, path in outputs.items()},
            "deleted_source_gaussians": _record(deleted_ply, output),
            "retained_scene_gaussians": _record(retained_ply, output),
        },
        "claim_ceiling": "byte_exact_cutout_candidate_pending_hybrid_coverage_and_seam_review",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output / "adp009b_bound_index_union_candidate.v1.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def _ownership_output_array(
    *, ownership_path: Path, record: object, code: str
) -> tuple[Path, np.ndarray]:
    if not isinstance(record, Mapping):
        raise ReplacementOcclusionError([code])
    relative = str(record.get("relative_path") or "")
    path = (ownership_path.parent / relative).resolve()
    if (
        not relative
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ReplacementOcclusionError([code])
    values = _load_array(path, code)
    if (
        values.ndim != 1
        or values.dtype.kind not in {"i", "u"}
        or (values.size > 1 and np.any(values[1:] <= values[:-1]))
    ):
        raise ReplacementOcclusionError([code])
    return path, values.astype(np.int64, copy=False)


def materialize_ownership_coverage_cutout_candidate(
    *,
    source_standard_splat_path: str | Path,
    ownership_receipt_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Prepare owned-plus-ambiguous deletion for later actual-USD coverage.

    Ambiguous records are deliberately not relabelled as owned.  They are only
    included in this byte-exact *candidate* so a later all-camera/state
    source-layer coverage audit can independently decide whether their removal
    is safe.  The source ownership receipt must have been made without either
    held-out classification pixels or a replacement asset.
    """

    source = Path(source_standard_splat_path).expanduser().resolve()
    ownership_path = Path(ownership_receipt_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_output_not_empty"]
        )
    if not source.is_file() or source.is_symlink():
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_source_splat_missing"]
        )
    ownership = _read_object(
        ownership_path, code="ownership_coverage_cutout_ownership_unreadable"
    )
    if (
        ownership.get("schema_version") != _OWNERSHIP_RECEIPT_SCHEMA
        or ownership.get("status") not in _OWNERSHIP_RECEIPT_STATUSES
        or ownership.get("receipt_digest")
        != canonical_digest(ownership, digest_field="receipt_digest")
        or ownership.get("heldout_cameras_accessed_for_classification") is not False
        or ownership.get("replacement_usd_inserted") is not False
    ):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_ownership_receipt_invalid"]
        )
    source_record = ownership.get("source_standard_splat")
    if (
        not isinstance(source_record, Mapping)
        or source_record.get("sha256") != _sha256(source)
        or source_record.get("size_bytes") != source.stat().st_size
    ):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_source_identity_mismatch"]
        )
    outputs = ownership.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_ownership_outputs_invalid"]
        )
    _owned_path, owned = _ownership_output_array(
        ownership_path=ownership_path,
        record=outputs.get("owned_indices"),
        code="ownership_coverage_cutout_owned_indices_invalid",
    )
    _ambiguous_path, ambiguous = _ownership_output_array(
        ownership_path=ownership_path,
        record=outputs.get("ambiguous_indices"),
        code="ownership_coverage_cutout_ambiguous_indices_invalid",
    )
    _retained_path, retained = _ownership_output_array(
        ownership_path=ownership_path,
        record=outputs.get("retained_indices"),
        code="ownership_coverage_cutout_retained_indices_invalid",
    )
    splat = read_standard_3dgs_ply(source)
    if any(
        values.size and (values[0] < 0 or values[-1] >= splat.count)
        for values in (owned, ambiguous, retained)
    ):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_indices_out_of_range"]
        )
    deleted = np.union1d(owned, ambiguous).astype(np.int64)
    if (
        not deleted.size
        or not retained.size
        or np.intersect1d(owned, ambiguous, assume_unique=True).size
        or np.intersect1d(owned, retained, assume_unique=True).size
        or np.intersect1d(ambiguous, retained, assume_unique=True).size
        or np.intersect1d(deleted, retained, assume_unique=True).size
        or not np.array_equal(
            np.union1d(deleted, retained), np.arange(splat.count, dtype=np.int64)
        )
    ):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_partition_invalid"]
        )
    counts = ownership.get("ownership")
    if (
        not isinstance(counts, Mapping)
        or counts.get("source_gaussian_count") != splat.count
        or counts.get("owned_count") != int(owned.size)
        or counts.get("ambiguous_count") != int(ambiguous.size)
        or counts.get("retained_count") != int(retained.size)
        or int(owned.size) + int(ambiguous.size) + int(retained.size)
        != splat.count
        or counts.get("exhaustive") is not True
        or counts.get("pairwise_disjoint") is not True
    ):
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_ownership_counts_invalid"]
        )

    output.mkdir(parents=True, exist_ok=True)
    deleted_path = output / "deleted_source_indices.npy"
    retained_path = output / "retained_source_indices.npy"
    ambiguous_path = output / "coverage_conditioned_ambiguous_indices.npy"
    np.save(deleted_path, deleted, allow_pickle=False)
    np.save(retained_path, retained, allow_pickle=False)
    np.save(ambiguous_path, ambiguous, allow_pickle=False)
    deleted_ply = write_standard_3dgs_ply_subset_exact(
        source, output / "deleted_source_gaussians.ply", deleted
    )
    retained_ply = write_standard_3dgs_ply_subset_exact(
        source, output / "retained_scene_gaussians.ply", retained
    )
    preservation = verify_standard_3dgs_ply_subset_exact(
        source, retained_ply, retained
    )
    if preservation.get("retained_rows_byte_exact") is not True:
        raise ReplacementOcclusionError(
            ["ownership_coverage_cutout_retained_rows_changed"]
        )
    receipt: dict[str, Any] = {
        "schema_version": OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA,
        "status": "ownership_coverage_cutout_materialized_pending_actual_usd_source_layer_coverage",
        "source_standard_splat": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "source_ownership_receipt": {
            "path": str(ownership_path),
            "size_bytes": ownership_path.stat().st_size,
            "sha256": _sha256(ownership_path),
            "receipt_digest": ownership["receipt_digest"],
            "freeze_digest": ownership.get("freeze_digest"),
        },
        "selection": {
            "rule": "owned_and_ambiguous_union_from_calibration_only_ownership.v1",
            "heldout_pixels_used_to_select_indices": False,
            "learned_policy_outcomes_used": False,
            "caller_asserted_coverage": False,
            "replacement_usd_used_to_select_indices": False,
        },
        "counts": {
            "source": splat.count,
            "owned": int(owned.size),
            "ambiguous_pending_coverage": int(ambiguous.size),
            "deleted_total": int(deleted.size),
            "retained_total": int(retained.size),
        },
        "preservation": preservation,
        "outputs": {
            "deleted_source_indices": _record(deleted_path, output),
            "retained_source_indices": _record(retained_path, output),
            "coverage_conditioned_ambiguous_indices": _record(
                ambiguous_path, output
            ),
            "deleted_source_gaussians": _record(deleted_ply, output),
            "retained_scene_gaussians": _record(retained_ply, output),
        },
        "claim_boundary": {
            "factual_gaussian_ownership_established": False,
            "ambiguous_records_deleted_only_if_actual_usd_source_layer_coverage_qualifies": True,
            "replacement_asset_inserted": False,
            "native_simulator_import_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_ownership_coverage_cutout_set(
    *,
    source_standard_splat_path: str | Path,
    task_freeze_paths: Sequence[str | Path],
    excision_freeze_paths_by_task: Mapping[str, str | Path],
    ownership_receipt_paths_by_task: Mapping[str, str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize independent coverage-conditioned candidates for one to five objects.

    A failed factual-ownership audit must not turn into a scene-specific manual
    crop.  This successor path selects ``owned ∪ ambiguous`` solely from each
    task's calibration-only ownership receipt, retains every other source row
    byte-for-byte, and emits a shared-scene union.  The ambiguous records stay
    explicitly *not factually owned*: a later actual-USD, all-camera/state
    coverage audit is the only route that can qualify their deletion.

    The set fails closed when two task slices nominate the same source Gaussian.
    That makes a shared source contribution visible instead of silently letting
    two replacements share a deletion assumption.  A caller that needs to
    model a genuinely shared source layer must supply a separate joint-object
    construction contract; it cannot pass through this independent-object
    seam.
    """

    source = Path(source_standard_splat_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ReplacementOcclusionError(["coverage_cutout_set_source_splat_missing"])
    if output.exists() and any(output.iterdir()):
        raise ReplacementOcclusionError(["coverage_cutout_set_output_not_empty"])
    if isinstance(task_freeze_paths, (str, bytes)) or not task_freeze_paths:
        raise ReplacementOcclusionError(["coverage_cutout_set_task_freezes_missing"])

    task_paths = [Path(path).expanduser().resolve() for path in task_freeze_paths]
    raw_tasks: list[dict[str, Any]] = []
    for path in task_paths:
        if not path.is_file() or path.is_symlink():
            raise ReplacementOcclusionError(["coverage_cutout_set_task_freeze_missing"])
        raw_tasks.append(
            _read_object(path, code="coverage_cutout_set_task_freeze_unreadable")
        )
    try:
        task_set = validate_task_freeze_set(raw_tasks)
    except DualTaskRehearsalContractError as exc:
        raise ReplacementOcclusionError(
            [f"coverage_cutout_set_{code}" for code in exc.errors]
        ) from exc

    tasks_by_id = {str(task["task_id"]): task for task in raw_tasks}
    paths_by_id = {
        str(task["task_id"]): path for task, path in zip(raw_tasks, task_paths, strict=True)
    }
    task_ids = sorted(tasks_by_id)
    if set(excision_freeze_paths_by_task) != set(task_ids):
        raise ReplacementOcclusionError(["coverage_cutout_set_excision_freeze_keys_invalid"])
    if set(ownership_receipt_paths_by_task) != set(task_ids):
        raise ReplacementOcclusionError(["coverage_cutout_set_ownership_receipt_keys_invalid"])

    source_sha256 = _sha256(source)
    source_size = source.stat().st_size
    source_splat = read_standard_3dgs_ply(source)
    validated: list[tuple[str, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path]] = []
    preselected_indices: dict[str, np.ndarray] = {}
    errors: list[str] = []
    for task_id in task_ids:
        task = tasks_by_id[task_id]
        excision_path = Path(excision_freeze_paths_by_task[task_id]).expanduser().resolve()
        ownership_path = Path(ownership_receipt_paths_by_task[task_id]).expanduser().resolve()
        if not excision_path.is_file() or excision_path.is_symlink():
            errors.append(f"coverage_cutout_set_excision_freeze_missing:{task_id}")
            continue
        if not ownership_path.is_file() or ownership_path.is_symlink():
            errors.append(f"coverage_cutout_set_ownership_receipt_missing:{task_id}")
            continue
        excision = _read_object(
            excision_path, code=f"coverage_cutout_set_excision_freeze_unreadable:{task_id}"
        )
        ownership = _read_object(
            ownership_path, code=f"coverage_cutout_set_ownership_receipt_unreadable:{task_id}"
        )
        expected_excision_digest = canonical_digest(excision, digest_field="freeze_digest")
        if (
            excision.get("schema_version") != "adp009b_gaussian_excision_audit_freeze.v1"
            or excision.get("status") != "frozen_before_excision_execution"
            or excision.get("freeze_digest") != expected_excision_digest
            or excision.get("learned_policy_outcomes_observed") is not False
            or excision.get("replacement_usd_inserted") is not False
        ):
            errors.append(f"coverage_cutout_set_excision_freeze_invalid:{task_id}")
            continue
        scene = excision.get("scene")
        if not isinstance(scene, Mapping) or (
            scene.get("task_id") != task_id
            or scene.get("target_instance_id")
            != task["source_object"]["instance_id"]
            or scene.get("removal_id") != task["removal_plan"]["removal_id"]
            or scene.get("mask_set_id") != task["removal_plan"]["mask_set_id"]
        ):
            errors.append(f"coverage_cutout_set_excision_task_join_invalid:{task_id}")
            continue
        excision_source = excision.get("source_standard_splat")
        if not isinstance(excision_source, Mapping) or (
            excision_source.get("sha256") != source_sha256
            or excision_source.get("size_bytes") != source_size
        ):
            errors.append(f"coverage_cutout_set_excision_source_mismatch:{task_id}")
            continue
        if (
            ownership.get("schema_version") != _OWNERSHIP_RECEIPT_SCHEMA
            or ownership.get("status") not in _OWNERSHIP_RECEIPT_STATUSES
            or ownership.get("receipt_digest")
            != canonical_digest(ownership, digest_field="receipt_digest")
            or ownership.get("freeze_digest") != excision.get("freeze_digest")
            or ownership.get("heldout_cameras_accessed_for_classification") is not False
            or ownership.get("replacement_usd_inserted") is not False
        ):
            errors.append(f"coverage_cutout_set_ownership_receipt_invalid:{task_id}")
            continue
        ownership_source = ownership.get("source_standard_splat")
        if not isinstance(ownership_source, Mapping) or (
            ownership_source.get("sha256") != source_sha256
            or ownership_source.get("size_bytes") != source_size
        ):
            errors.append(f"coverage_cutout_set_ownership_source_mismatch:{task_id}")
            continue
        outputs = ownership.get("outputs")
        if not isinstance(outputs, Mapping):
            errors.append(f"coverage_cutout_set_ownership_outputs_invalid:{task_id}")
            continue
        _owned_path, owned = _ownership_output_array(
            ownership_path=ownership_path,
            record=outputs.get("owned_indices"),
            code=f"coverage_cutout_set_owned_indices_invalid:{task_id}",
        )
        _ambiguous_path, ambiguous = _ownership_output_array(
            ownership_path=ownership_path,
            record=outputs.get("ambiguous_indices"),
            code=f"coverage_cutout_set_ambiguous_indices_invalid:{task_id}",
        )
        _retained_path, retained = _ownership_output_array(
            ownership_path=ownership_path,
            record=outputs.get("retained_indices"),
            code=f"coverage_cutout_set_retained_indices_invalid:{task_id}",
        )
        selected = np.union1d(owned, ambiguous).astype(np.int64)
        source_count = source_splat.count
        if (
            not selected.size
            or any(
                values.size and (values[0] < 0 or values[-1] >= source_count)
                for values in (owned, ambiguous, retained)
            )
            or np.intersect1d(owned, ambiguous, assume_unique=True).size
            or np.intersect1d(owned, retained, assume_unique=True).size
            or np.intersect1d(ambiguous, retained, assume_unique=True).size
            or not np.array_equal(
                np.union1d(selected, retained),
                np.arange(source_count, dtype=np.int64),
            )
        ):
            errors.append(f"coverage_cutout_set_ownership_partition_invalid:{task_id}")
            continue
        preselected_indices[task_id] = selected
        validated.append(
            (
                task_id,
                task,
                paths_by_id[task_id],
                excision,
                excision_path,
                ownership,
                ownership_path,
            )
        )
    if errors:
        raise ReplacementOcclusionError(errors)

    for left_index, left_task_id in enumerate(task_ids):
        for right_task_id in task_ids[left_index + 1 :]:
            if np.intersect1d(
                preselected_indices[left_task_id],
                preselected_indices[right_task_id],
                assume_unique=True,
            ).size:
                raise ReplacementOcclusionError(
                    [
                        "coverage_cutout_set_independent_candidate_overlap:"
                        f"{left_task_id}:{right_task_id}"
                    ]
                )

    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    candidate_indices: dict[str, np.ndarray] = {}
    for slot, (
        task_id,
        task,
        task_path,
        excision,
        excision_path,
        ownership,
        ownership_path,
    ) in enumerate(validated, start=1):
        candidate_root = output / "task_candidates" / f"slot_{slot:02d}"
        candidate = materialize_ownership_coverage_cutout_candidate(
            source_standard_splat_path=source,
            ownership_receipt_path=ownership_path,
            output_root=candidate_root,
        )
        candidate_receipt_path = candidate_root / f"{OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA}.json"
        deleted_record = candidate["outputs"]["deleted_source_indices"]
        deleted_path = _verify_artifact(
            candidate_root,
            deleted_record,
            code=f"coverage_cutout_set_candidate_indices_invalid:{task_id}",
        )
        deleted = np.asarray(np.load(deleted_path, allow_pickle=False), dtype=np.int64)
        if (
            deleted.ndim != 1
            or not deleted.size
            or (deleted.size > 1 and np.any(deleted[1:] <= deleted[:-1]))
            or not np.array_equal(deleted, preselected_indices[task_id])
        ):
            raise ReplacementOcclusionError(
                [f"coverage_cutout_set_candidate_indices_invalid:{task_id}"]
            )
        candidate_indices[task_id] = deleted
        candidates.append(
            {
                "slot": slot,
                "task_id": task_id,
                "task_freeze_digest": task["task_freeze_digest"],
                "source_object_instance_id": task["source_object"]["instance_id"],
                "removal_id": task["removal_plan"]["removal_id"],
                "mask_set_id": task["removal_plan"]["mask_set_id"],
                "task_freeze": _input_record(task_path),
                "excision_freeze": {
                    **_input_record(excision_path),
                    "freeze_digest": excision["freeze_digest"],
                },
                "ownership_receipt": {
                    **_input_record(ownership_path),
                    "receipt_digest": ownership["receipt_digest"],
                },
                "candidate_receipt": {
                    **_record(candidate_receipt_path, output),
                    "receipt_digest": candidate["receipt_digest"],
                },
                "counts": dict(candidate["counts"]),
            }
        )

    deleted_union = np.unique(np.concatenate([candidate_indices[task_id] for task_id in task_ids]))
    retained_union = np.setdiff1d(
        np.arange(source_splat.count, dtype=np.int64), deleted_union, assume_unique=True
    )
    shared_root = output / "shared_scene_union"
    shared_root.mkdir()
    deleted_indices_path = shared_root / "deleted_source_indices.npy"
    retained_indices_path = shared_root / "retained_source_indices.npy"
    np.save(deleted_indices_path, deleted_union, allow_pickle=False)
    np.save(retained_indices_path, retained_union, allow_pickle=False)
    deleted_ply = write_standard_3dgs_ply_subset_exact(
        source, shared_root / "deleted_source_gaussians.ply", deleted_union
    )
    retained_ply = write_standard_3dgs_ply_subset_exact(
        source, shared_root / "retained_scene_gaussians.ply", retained_union
    )
    preservation = verify_standard_3dgs_ply_subset_exact(source, retained_ply, retained_union)
    if preservation.get("retained_rows_byte_exact") is not True:
        raise ReplacementOcclusionError(["coverage_cutout_set_retained_rows_changed"])

    receipt: dict[str, Any] = {
        "schema_version": OWNERSHIP_COVERAGE_CUTOUT_SET_SCHEMA,
        "status": (
            "coverage_conditioned_successor_set_materialized_"
            "pending_per_task_actual_usd_source_layer_coverage"
        ),
        "source_standard_splat": _input_record(source),
        "task_set": task_set,
        "selection": {
            "rule": "owned_and_ambiguous_union_from_calibration_only_ownership.v1",
            "heldout_pixels_used_to_select_indices": False,
            "replacement_usd_used_to_select_indices": False,
            "learned_policy_outcomes_used": False,
            "factual_gaussian_ownership_established_for_ambiguous_records": False,
            "fresh_confirmation_coverage_required_before_qualification": True,
        },
        "task_candidates": candidates,
        "shared_scene_union": {
            "counts": {
                "source": source_splat.count,
                "deleted_total": int(deleted_union.size),
                "retained_total": int(retained_union.size),
            },
            "preservation": preservation,
            "outputs": {
                "deleted_source_indices": _record(deleted_indices_path, output),
                "retained_source_indices": _record(retained_indices_path, output),
                "deleted_source_gaussians": _record(deleted_ply, output),
                "retained_scene_gaussians": _record(retained_ply, output),
            },
        },
        "claim_boundary": {
            "source_gaussians_deleted_from_canonical_scene": False,
            "candidate_derived_layers_only": True,
            "all_task_slices_independent": True,
            "overlapping_task_deletions_allowed": False,
            "replacement_depth_coverage_qualified": False,
            "inpainting_decision_qualified": False,
            "native_simulator_import_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{OWNERSHIP_COVERAGE_CUTOUT_SET_SCHEMA}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def evaluate_depth_coverage(
    removal_alpha: np.ndarray,
    replacement_depth_m: np.ndarray,
    *,
    confident_alpha_threshold: float,
) -> list[dict[str, Any]]:
    """Measure actual replacement-depth coverage without trusting a caller mask."""

    alpha = np.asarray(removal_alpha, dtype=np.float64)
    depth = np.asarray(replacement_depth_m, dtype=np.float64)
    if alpha.ndim != 3 or depth.shape != alpha.shape or alpha.shape[0] == 0:
        raise ReplacementOcclusionError(["replacement_occlusion_depth_coverage_shape_invalid"])
    if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise ReplacementOcclusionError(["replacement_occlusion_removal_alpha_invalid"])
    covered = np.isfinite(depth) & (depth > 0)
    rows = []
    for cell in range(alpha.shape[0]):
        confident = alpha[cell] >= float(confident_alpha_threshold)
        uncovered = ~covered[cell]
        alpha_total = float(alpha[cell].sum())
        residual = float(alpha[cell][uncovered].sum())
        rows.append(
            {
                "cell_index": cell,
                "confident_removal_pixel_count": int(confident.sum()),
                "confident_uncovered_pixel_count": int((confident & uncovered).sum()),
                "removal_alpha_sum": alpha_total,
                "uncovered_alpha_sum": residual,
                "residual_alpha_fraction": residual / alpha_total if alpha_total > 0 else 0.0,
                "replacement_depth_covered_pixel_count": int(covered[cell].sum()),
            }
        )
    return rows


def _load_npz(path: Path, *, required: Sequence[str], code: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in required if key not in archive]
            if missing:
                raise ReplacementOcclusionError([code])
            return {key: np.asarray(archive[key]) for key in required}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ReplacementOcclusionError):
            raise
        raise ReplacementOcclusionError([code]) from exc


def _verify_scene_join(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, code: str
) -> None:
    for key in ("publisher_scene_id", "target_instance_id", "target_semantic_label"):
        if str(expected.get(key)).lower() != str(observed.get(key)).lower():
            raise ReplacementOcclusionError([code])


def materialize_replacement_occlusion_cutout(
    *,
    request_path: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Create a byte-preserving cutout and a measured inpainting disposition."""

    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    request_file = _under(repo, request_path, code="replacement_occlusion_request_outside_repo")
    output = _under(data, output_root, code="replacement_occlusion_output_outside_data")
    request = build_replacement_occlusion_request(
        _read_object(request_file, code="replacement_occlusion_request_unreadable")
    )
    inputs = request["inputs"]
    source_path = _verify_artifact(
        data, inputs["source_standard_splat"], code="replacement_occlusion_source_splat_changed"
    )
    contribution_manifest_path = _verify_artifact(
        data,
        inputs["contribution_manifest"],
        code="replacement_occlusion_contribution_manifest_changed",
    )
    coverage_manifest_path = _verify_artifact(
        data,
        inputs["coverage_manifest"],
        code="replacement_occlusion_coverage_manifest_changed",
    )
    contribution_manifest = _read_object(
        contribution_manifest_path, code="replacement_occlusion_contribution_manifest_invalid"
    )
    coverage_manifest = _read_object(
        coverage_manifest_path, code="replacement_occlusion_coverage_manifest_invalid"
    )
    if contribution_manifest.get("schema_version") != CONTRIBUTION_SCHEMA:
        raise ReplacementOcclusionError(["replacement_occlusion_contribution_schema_invalid"])
    if coverage_manifest.get("schema_version") != COVERAGE_SCHEMA:
        raise ReplacementOcclusionError(["replacement_occlusion_coverage_schema_invalid"])
    if contribution_manifest.get("manifest_digest") != canonical_digest(
        contribution_manifest, digest_field="manifest_digest"
    ):
        raise ReplacementOcclusionError(["replacement_occlusion_contribution_digest_mismatch"])
    if coverage_manifest.get("manifest_digest") != canonical_digest(
        coverage_manifest, digest_field="manifest_digest"
    ):
        raise ReplacementOcclusionError(["replacement_occlusion_coverage_digest_mismatch"])
    _verify_scene_join(
        request["scene"], contribution_manifest.get("scene") or {},
        code="replacement_occlusion_contribution_scene_mismatch",
    )
    _verify_scene_join(
        request["scene"], coverage_manifest.get("scene") or {},
        code="replacement_occlusion_coverage_scene_mismatch",
    )
    if contribution_manifest.get("source_standard_splat_sha256") != _sha256(source_path):
        raise ReplacementOcclusionError(["replacement_occlusion_contribution_source_mismatch"])
    method = contribution_manifest.get("method") or {}
    if (
        method.get("name") != "FlashSplat"
        or method.get("released_code_executed") is not True
        or method.get("source_modified") is not False
        or not str(method.get("repository") or "").startswith("https://github.com/")
        or len(str(method.get("commit") or "")) != 40
    ):
        raise ReplacementOcclusionError(["replacement_occlusion_released_method_identity_invalid"])
    if coverage_manifest.get("actual_mesh_depth_rasterized") is not True:
        raise ReplacementOcclusionError(["replacement_occlusion_actual_mesh_depth_missing"])
    if coverage_manifest.get("caller_supplied_coverage_mask") is not False:
        raise ReplacementOcclusionError(["replacement_occlusion_caller_coverage_forbidden"])

    contribution_npz = _verify_artifact(
        contribution_manifest_path.parent,
        contribution_manifest.get("arrays") or {},
        code="replacement_occlusion_contribution_arrays_changed",
    )
    coverage_npz = _verify_artifact(
        coverage_manifest_path.parent,
        coverage_manifest.get("arrays") or {},
        code="replacement_occlusion_coverage_arrays_changed",
    )
    contribution = _load_npz(
        contribution_npz,
        required=("foreground_contribution", "background_contribution"),
        code="replacement_occlusion_contribution_arrays_invalid",
    )
    coverage = _load_npz(
        coverage_npz,
        required=(
            "removal_alpha", "replacement_depth_m", "gaussian_indices",
            "gaussian_visible_contribution", "gaussian_uncovered_contribution",
        ),
        code="replacement_occlusion_coverage_arrays_invalid",
    )
    splat = read_standard_3dgs_ply(source_path)
    if contribution["foreground_contribution"].shape != (splat.count,):
        raise ReplacementOcclusionError(["replacement_occlusion_contribution_count_mismatch"])
    policy = request["policy"]
    classification = classify_gaussian_contributions(
        contribution["foreground_contribution"],
        contribution["background_contribution"],
        retained_max_foreground_fraction=float(policy["retained_max_foreground_fraction"]),
        owned_min_foreground_fraction=float(policy["owned_min_foreground_fraction"]),
        minimum_total_contribution=float(policy["minimum_total_contribution"]),
    )

    cells = coverage_manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != coverage["removal_alpha"].shape[0]:
        raise ReplacementOcclusionError(["replacement_occlusion_coverage_cells_invalid"])
    required_cells = {
        (str(camera), float(angle))
        for camera in policy["required_camera_ids"]
        for angle in policy["required_door_angles_deg"]
    }
    observed_cells: set[tuple[str, float]] = set()
    readback_tolerance = float(policy["door_angle_readback_tolerance_deg"])
    for row in cells:
        if not isinstance(row, Mapping):
            raise ReplacementOcclusionError(["replacement_occlusion_coverage_cells_invalid"])
        commanded = float(row.get("commanded_door_angle_deg"))
        readback = float(row.get("readback_door_angle_deg"))
        observed_cells.add((str(row.get("camera_id")), commanded))
        if abs(commanded - readback) > readback_tolerance:
            raise ReplacementOcclusionError(["replacement_occlusion_door_angle_readback_mismatch"])
    if observed_cells != required_cells or len(cells) != len(required_cells):
        raise ReplacementOcclusionError(["replacement_occlusion_coverage_matrix_mismatch"])

    ambiguous_indices = np.flatnonzero(classification["ambiguous"])
    safe = coverage_safe_ambiguous(
        ambiguous_indices,
        coverage["gaussian_indices"],
        coverage["gaussian_visible_contribution"],
        coverage["gaussian_uncovered_contribution"],
        minimum_cell_visible_contribution=float(policy["minimum_cell_visible_contribution"]),
        maximum_uncovered_fraction=float(policy["maximum_ambiguous_uncovered_fraction"]),
        maximum_uncovered_contribution=float(
            policy["maximum_ambiguous_uncovered_contribution"]
        ),
    )
    delete = classification["owned"].copy()
    delete[ambiguous_indices[safe]] = True
    retained = ~delete
    labels = np.full(splat.count, LABEL_RETAINED, dtype=np.uint8)
    labels[classification["owned"]] = LABEL_OWNED_DELETED
    labels[ambiguous_indices[~safe]] = LABEL_AMBIGUOUS_RETAINED
    labels[ambiguous_indices[safe]] = LABEL_AMBIGUOUS_DELETED

    depth_rows = evaluate_depth_coverage(
        coverage["removal_alpha"],
        coverage["replacement_depth_m"],
        confident_alpha_threshold=float(policy["confident_removal_alpha_threshold"]),
    )
    for row, cell in zip(depth_rows, cells, strict=True):
        row.update(
            camera_id=str(cell["camera_id"]),
            commanded_door_angle_deg=float(cell["commanded_door_angle_deg"]),
            readback_door_angle_deg=float(cell["readback_door_angle_deg"]),
        )
    coverage_passed = all(
        row["confident_uncovered_pixel_count"]
        <= int(policy["maximum_confident_uncovered_pixels_per_cell"])
        and row["residual_alpha_fraction"]
        <= float(policy["maximum_residual_alpha_fraction_per_cell"])
        for row in depth_rows
    )

    output.mkdir(parents=True, exist_ok=True)
    retained_indices = np.flatnonzero(retained).astype(np.int64)
    deleted_indices = np.flatnonzero(delete).astype(np.int64)
    retained_ply = write_standard_3dgs_ply_subset_exact(
        source_path, output / "scene_without_target_gaussians.ply", retained_indices
    )
    deleted_ply = write_standard_3dgs_ply_subset_exact(
        source_path, output / "deleted_target_gaussians.ply", deleted_indices
    )
    retained_index_path = output / "retained_source_indices.npy"
    deleted_index_path = output / "deleted_source_indices.npy"
    labels_path = output / "gaussian_ownership_labels.npy"
    np.save(retained_index_path, retained_indices, allow_pickle=False)
    np.save(deleted_index_path, deleted_indices, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)
    exact = verify_standard_3dgs_ply_subset_exact(
        source_path, retained_ply, retained_indices
    )
    if exact["retained_rows_byte_exact"] is not True:
        raise ReplacementOcclusionError(["replacement_occlusion_retained_rows_changed"])

    blockers = []
    if not coverage_passed:
        blockers.append("replacement_depth_coverage_residual_observed")
    status = (
        "cutout_admitted_inpainting_not_required"
        if coverage_passed
        else "cutout_candidate_residual_measured"
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": status,
        "blockers": blockers,
        "request_digest": request["request_digest"],
        "scene": dict(request["scene"]),
        "source_standard_splat": dict(inputs["source_standard_splat"]),
        "method": method,
        "replacement_depth_renderer": coverage_manifest.get("renderer"),
        "replacement_usd": coverage_manifest.get("replacement_usd"),
        "ownership": {
            "scene_gaussian_count": int(splat.count),
            "owned_count": int(classification["owned"].sum()),
            "retained_owned_evidence_count": int(classification["retained"].sum()),
            "ambiguous_count": int(classification["ambiguous"].sum()),
            "coverage_safe_ambiguous_deleted_count": int(safe.sum()),
            "ambiguous_retained_count": int((~safe).sum()),
            "total_deleted_count": int(delete.sum()),
            "total_retained_count": int(retained.sum()),
            "policy": dict(policy),
        },
        "coverage": {
            "passed": coverage_passed,
            "cell_count": len(depth_rows),
            "worst_confident_uncovered_pixel_count": max(
                row["confident_uncovered_pixel_count"] for row in depth_rows
            ),
            "worst_residual_alpha_fraction": max(
                row["residual_alpha_fraction"] for row in depth_rows
            ),
            "cells": depth_rows,
        },
        "preservation": exact,
        "outputs": {
            "scene_without_target_gaussians": _record(retained_ply, output),
            "deleted_target_gaussians": _record(deleted_ply, output),
            "retained_source_indices": _record(retained_index_path, output),
            "deleted_source_indices": _record(deleted_index_path, output),
            "gaussian_ownership_labels": _record(labels_path, output),
        },
        "inpainting_disposition": (
            "inpainting_not_required_by_replacement_occlusion"
            if coverage_passed
            else "conditional_seam_ladder_required_for_measured_residual"
        ),
        "source_collider_removed": False,
        "hybrid_render_qualified": False,
        "candidate_policy_queried": False,
        "claim_ceiling": (
            "byte_exact_3dgs_cutout_admitted_by_independent_replacement_depth_coverage"
            if coverage_passed
            else "byte_exact_3dgs_cutout_candidate_with_measured_coverage_residual"
        ),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{RECEIPT_SCHEMA}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    if receipt_output is not None:
        retained_receipt = _under(
            repo, receipt_output, code="replacement_occlusion_receipt_outside_repo"
        )
        retained_receipt.parent.mkdir(parents=True, exist_ok=True)
        retained_receipt.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args(argv)
    receipt = materialize_replacement_occlusion_cutout(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_root=args.output_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


__all__ = [
    "CONTRIBUTION_SCHEMA",
    "COVERAGE_SCHEMA",
    "OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA",
    "OWNERSHIP_COVERAGE_CUTOUT_SET_SCHEMA",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "ReplacementOcclusionError",
    "build_replacement_occlusion_request",
    "classify_gaussian_contributions",
    "coverage_safe_ambiguous",
    "evaluate_depth_coverage",
    "materialize_replacement_occlusion_cutout",
    "materialize_direct_evidence_expansion_candidate",
    "materialize_ownership_coverage_cutout_candidate",
    "materialize_ownership_coverage_cutout_set",
    "materialize_bound_index_union_candidate",
    "select_direct_calibration_evidence_expansion",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
