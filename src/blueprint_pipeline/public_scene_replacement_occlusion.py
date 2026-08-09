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
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply_subset_exact,
)

REQUEST_SCHEMA = "adp009b_replacement_occlusion_request.v1"
CONTRIBUTION_SCHEMA = "adp009b_gaussian_contribution_evidence.v1"
COVERAGE_SCHEMA = "adp009b_replacement_depth_coverage.v1"
RECEIPT_SCHEMA = "adp009b_replacement_occlusion_receipt.v1"

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
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "ReplacementOcclusionError",
    "build_replacement_occlusion_request",
    "classify_gaussian_contributions",
    "coverage_safe_ambiguous",
    "evaluate_depth_coverage",
    "materialize_replacement_occlusion_cutout",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
