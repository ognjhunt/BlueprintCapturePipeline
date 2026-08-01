"""Independent held-out baseline evaluation for ArtiFixer support assets."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .common import read_json_any, sha256_file, write_json
from .decision_evidence_contracts import canonical_digest


MANIFEST_SCHEMA = "artifixer_heldout_evaluation_manifest.v2"
RESULT_SCHEMA = "artifixer_heldout_real_view_evaluation.v2"
MINIMUM_HELDOUT_VIEW_COUNT = 3
MINIMUM_MEAN_PSNR_DB = 20.0
MAXIMUM_MEAN_ABSOLUTE_ERROR = 0.1
MINIMUM_MEAN_PSNR_IMPROVEMENT_DB = 0.1
MINIMUM_MEAN_MAE_IMPROVEMENT = 0.001
MINIMUM_IMPROVED_VIEW_FRACTION = 0.5


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _resolve_root(value: str | Path, *, code: str, blockers: list[str]) -> Path | None:
    path = Path(value).expanduser()
    if path.is_symlink():
        blockers.append(f"{code}_symlink_forbidden")
        return None
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        blockers.append(f"{code}_missing")
        return None
    if not resolved.is_dir():
        blockers.append(f"{code}_invalid")
        return None
    return resolved


def _safe_image(
    root: Path | None,
    *,
    reference: Any,
    expected_sha256: Any,
    code: str,
    blockers: list[str],
) -> Path | None:
    text = str(reference or "").strip().replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        root is None
        or not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        blockers.append(f"{code}_reference_unsafe")
        return None
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        blockers.append(f"{code}_symlink_forbidden")
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        blockers.append(f"{code}_missing")
        return None
    if root != resolved and root not in resolved.parents:
        blockers.append(f"{code}_path_escape")
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        blockers.append(f"{code}_format_invalid")
        return None
    if not isinstance(expected_sha256, str) or sha256_file(resolved) != expected_sha256:
        blockers.append(f"{code}_digest_mismatch")
        return None
    return resolved


def _metrics(real: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    difference = real - candidate
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return psnr, mae


def _mean_psnr(values: list[float]) -> float:
    return float("inf") if any(math.isinf(value) for value in values) else float(np.mean(values))


def _metric_value(value: float, *, digits: int) -> float | str:
    if math.isinf(value):
        return "infinity" if value > 0 else "negative_infinity"
    return round(value, digits)


def evaluate_artifixer_heldout_views(
    *,
    manifest_path: str | Path,
    heldout_root: str | Path,
    baseline_root: str | Path,
    generated_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare baseline and enhanced renders against frozen real observations.

    Paths in the evaluator-owned manifest are relative to three disjoint roots.
    Candidate execution receives none of the real-view paths through this API.
    Thresholds are implementation constants and cannot be supplied by an agent.
    """

    manifest_file = Path(manifest_path).expanduser()
    blockers: list[str] = []
    if manifest_file.is_symlink():
        blockers.append("artifixer_heldout_manifest_symlink_forbidden")
        manifest: dict[str, Any] = {}
        manifest_resolved = manifest_file.absolute()
    else:
        try:
            manifest_resolved = manifest_file.resolve(strict=True)
            loaded = read_json_any(manifest_resolved)
            manifest = _mapping(loaded)
        except (OSError, ValueError):
            manifest_resolved = manifest_file.absolute()
            manifest = {}
            blockers.append("artifixer_heldout_manifest_unreadable")

    supplied_manifest_digest = manifest.get("manifest_digest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("frozen") is not True
        or manifest.get("frozen_before_candidate_execution") is not True
        or supplied_manifest_digest != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        blockers.append("artifixer_heldout_manifest_not_accepted")
    for field in (
        "source_capture_digest",
        "frozen_split_digest",
        "baseline_reconstruction_digest",
        "enhancement_method_audit_digest",
    ):
        value = str(manifest.get(field) or "")
        if len(value) != 71 or not value.startswith("sha256:"):
            blockers.append(f"artifixer_heldout_{field}_invalid")
    if not isinstance(manifest.get("timestamp"), str) or not manifest.get("timestamp"):
        blockers.append("artifixer_heldout_timestamp_missing")

    real_dir = _resolve_root(heldout_root, code="artifixer_heldout_root", blockers=blockers)
    baseline_dir = _resolve_root(baseline_root, code="artifixer_baseline_root", blockers=blockers)
    generated_dir = _resolve_root(
        generated_root, code="artifixer_generated_root", blockers=blockers
    )
    if len({path for path in (real_dir, baseline_dir, generated_dir) if path is not None}) != 3:
        blockers.append("artifixer_evaluation_roots_must_be_disjoint")

    training_ids = {str(value) for value in manifest.get("training_view_ids", [])}
    raw_pairs = manifest.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) < MINIMUM_HELDOUT_VIEW_COUNT:
        blockers.append("artifixer_heldout_pair_count_insufficient")
        raw_pairs = [] if not isinstance(raw_pairs, list) else raw_pairs
    rows: list[dict[str, Any]] = []
    heldout_ids: set[str] = set()
    for index, value in enumerate(raw_pairs):
        pair = _mapping(value)
        view_id = str(pair.get("view_id") or "").strip()
        if not view_id or view_id in heldout_ids:
            blockers.append(f"artifixer_heldout_view_id_missing_or_duplicate:{index}")
        heldout_ids.add(view_id)
        if view_id in training_ids or pair.get("excluded_from_candidate_training") is not True:
            blockers.append(f"artifixer_view_not_proven_held_out:{view_id or index}")
        real_path = _safe_image(
            real_dir,
            reference=pair.get("real_view_reference"),
            expected_sha256=pair.get("real_view_sha256"),
            code=f"artifixer_real_view:{view_id or index}",
            blockers=blockers,
        )
        baseline_path = _safe_image(
            baseline_dir,
            reference=pair.get("baseline_view_reference"),
            expected_sha256=pair.get("baseline_view_sha256"),
            code=f"artifixer_baseline_view:{view_id or index}",
            blockers=blockers,
        )
        generated_path = _safe_image(
            generated_dir,
            reference=pair.get("generated_view_reference"),
            expected_sha256=pair.get("generated_view_sha256"),
            code=f"artifixer_generated_view:{view_id or index}",
            blockers=blockers,
        )
        if real_path is None or baseline_path is None or generated_path is None:
            continue
        try:
            real = _image(real_path)
            baseline = _image(baseline_path)
            generated = _image(generated_path)
        except (OSError, ValueError):
            blockers.append(f"artifixer_view_decode_failed:{view_id}")
            continue
        if real.shape != baseline.shape or real.shape != generated.shape:
            blockers.append(f"artifixer_view_shape_mismatch:{view_id}")
            continue
        baseline_psnr, baseline_mae = _metrics(real, baseline)
        generated_psnr, generated_mae = _metrics(real, generated)
        rows.append(
            {
                "view_id": view_id,
                "real_view_reference": pair["real_view_reference"],
                "real_view_sha256": pair["real_view_sha256"],
                "baseline_view_reference": pair["baseline_view_reference"],
                "baseline_view_sha256": pair["baseline_view_sha256"],
                "generated_view_reference": pair["generated_view_reference"],
                "generated_view_sha256": pair["generated_view_sha256"],
                "baseline_psnr_db": _metric_value(baseline_psnr, digits=6),
                "generated_psnr_db": _metric_value(generated_psnr, digits=6),
                "baseline_mean_absolute_error": round(baseline_mae, 8),
                "generated_mean_absolute_error": round(generated_mae, 8),
                "psnr_improvement_db": (
                    "infinity"
                    if math.isinf(generated_psnr) and not math.isinf(baseline_psnr)
                    else (
                        "negative_infinity"
                        if math.isinf(baseline_psnr) and not math.isinf(generated_psnr)
                        else (
                            0.0
                            if math.isinf(generated_psnr) and math.isinf(baseline_psnr)
                            else round(generated_psnr - baseline_psnr, 6)
                        )
                    )
                ),
                "mae_improvement": round(baseline_mae - generated_mae, 8),
            }
        )

    baseline_psnr_values = [
        float("inf") if row["baseline_psnr_db"] == "infinity" else float(row["baseline_psnr_db"])
        for row in rows
    ]
    generated_psnr_values = [
        float("inf") if row["generated_psnr_db"] == "infinity" else float(row["generated_psnr_db"])
        for row in rows
    ]
    baseline_mean_psnr = _mean_psnr(baseline_psnr_values) if rows else 0.0
    generated_mean_psnr = _mean_psnr(generated_psnr_values) if rows else 0.0
    baseline_mean_mae = (
        float(np.mean([row["baseline_mean_absolute_error"] for row in rows])) if rows else 1.0
    )
    generated_mean_mae = (
        float(np.mean([row["generated_mean_absolute_error"] for row in rows])) if rows else 1.0
    )
    if math.isinf(generated_mean_psnr) and not math.isinf(baseline_mean_psnr):
        psnr_improvement = float("inf")
    elif math.isinf(generated_mean_psnr) and math.isinf(baseline_mean_psnr):
        psnr_improvement = 0.0
    else:
        psnr_improvement = generated_mean_psnr - baseline_mean_psnr
    mae_improvement = baseline_mean_mae - generated_mean_mae
    improved_count = sum(
        1
        for row in rows
        if (
            row["psnr_improvement_db"] == "infinity"
            or (
                row["psnr_improvement_db"] != "negative_infinity"
                and float(row["psnr_improvement_db"]) > 0
            )
        )
        and float(row["mae_improvement"]) > 0
    )
    improved_fraction = float(improved_count / len(rows)) if rows else 0.0
    thresholds_passed = bool(
        len(rows) >= MINIMUM_HELDOUT_VIEW_COUNT
        and generated_mean_psnr >= MINIMUM_MEAN_PSNR_DB
        and generated_mean_mae <= MAXIMUM_MEAN_ABSOLUTE_ERROR
        and psnr_improvement >= MINIMUM_MEAN_PSNR_IMPROVEMENT_DB
        and mae_improvement >= MINIMUM_MEAN_MAE_IMPROVEMENT
        and improved_fraction >= MINIMUM_IMPROVED_VIEW_FRACTION
    )
    if not thresholds_passed:
        blockers.append("artifixer_heldout_baseline_improvement_not_established")

    result = {
        "schema_version": RESULT_SCHEMA,
        "timestamp": manifest.get("timestamp"),
        "status": "passed_advisory" if not blockers else "blocked_or_failed",
        "manifest_reference": manifest_resolved.name,
        "manifest_sha256": sha256_file(manifest_resolved) if manifest_resolved.is_file() else None,
        "manifest_digest": supplied_manifest_digest,
        "source_capture_digest": manifest.get("source_capture_digest"),
        "frozen_split_digest": manifest.get("frozen_split_digest"),
        "baseline_reconstruction_digest": manifest.get("baseline_reconstruction_digest"),
        "enhancement_method_audit_digest": manifest.get("enhancement_method_audit_digest"),
        "heldout_view_count": len(rows),
        "rows": rows,
        "aggregate": {
            "baseline_mean_psnr_db": _metric_value(baseline_mean_psnr, digits=6),
            "generated_mean_psnr_db": _metric_value(generated_mean_psnr, digits=6),
            "mean_psnr_improvement_db": _metric_value(psnr_improvement, digits=6),
            "baseline_mean_absolute_error": round(baseline_mean_mae, 8),
            "generated_mean_absolute_error": round(generated_mean_mae, 8),
            "mean_absolute_error_improvement": round(mae_improvement, 8),
            "improved_view_fraction": round(improved_fraction, 8),
            "threshold_profile": "artifixer_independent_real_heldout_v2",
            "minimum_heldout_view_count": MINIMUM_HELDOUT_VIEW_COUNT,
            "minimum_mean_psnr_db": MINIMUM_MEAN_PSNR_DB,
            "maximum_mean_absolute_error": MAXIMUM_MEAN_ABSOLUTE_ERROR,
            "minimum_mean_psnr_improvement_db": MINIMUM_MEAN_PSNR_IMPROVEMENT_DB,
            "minimum_mean_mae_improvement": MINIMUM_MEAN_MAE_IMPROVEMENT,
            "minimum_improved_view_fraction": MINIMUM_IMPROVED_VIEW_FRACTION,
            "thresholds_passed": thresholds_passed,
        },
        "blockers": sorted(set(blockers)),
        "proof_effect": "generated_visual_support_quality_only" if not blockers else "none",
        "claim_ceiling": "generated_visual_support",
        "claim_boundary": {
            "generated_support_quality_measured_on_heldout_real_views": not blockers,
            "baseline_improvement_established": not blockers,
            "generated_pixels_are_capture_truth": False,
            "generated_geometry_is_collision_truth": False,
            "metric_or_collision_qualification_changed": False,
            "simulator_execution_proven": False,
            "physical_or_deployment_success_proven": False,
        },
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_json(Path(output_path), result)
    return result


__all__ = [
    "MANIFEST_SCHEMA",
    "RESULT_SCHEMA",
    "evaluate_artifixer_heldout_views",
]
