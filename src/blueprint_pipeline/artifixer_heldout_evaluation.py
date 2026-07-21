"""Held-out real-view evaluation for ArtiFixer generated support assets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def evaluate_artifixer_heldout_views(
    *,
    manifest_path: str | Path,
    generated_root: str | Path,
    output_path: str | Path,
    minimum_mean_psnr_db: float = 20.0,
    maximum_mean_absolute_error: float = 0.1,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    generated_dir = Path(generated_root).resolve()
    loaded = read_json_any(manifest_file)
    manifest = _mapping(loaded)
    blockers: list[str] = []
    if manifest.get("frozen") is not True:
        blockers.append("artifixer_heldout_manifest_not_frozen")
    training_ids = {str(value) for value in manifest.get("training_view_ids", [])}
    raw_pairs = manifest.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        blockers.append("artifixer_heldout_pairs_missing")
        raw_pairs = []
    rows: list[dict[str, Any]] = []
    heldout_ids: set[str] = set()
    for index, value in enumerate(raw_pairs):
        pair = _mapping(value)
        view_id = str(pair.get("view_id") or "").strip()
        real_path = Path(str(pair.get("real_view_path") or "")).resolve()
        generated_path = Path(str(pair.get("generated_view_path") or "")).resolve()
        if not view_id or view_id in heldout_ids:
            blockers.append(f"artifixer_heldout_view_id_missing_or_duplicate:{index}")
        heldout_ids.add(view_id)
        if view_id in training_ids or pair.get("excluded_from_training") is not True:
            blockers.append(f"artifixer_view_not_proven_held_out:{view_id or index}")
        try:
            inside_generated = generated_path.is_relative_to(generated_dir)
        except ValueError:
            inside_generated = False
        if not real_path.is_file() or not generated_path.is_file() or not inside_generated:
            blockers.append(
                f"artifixer_view_pair_missing_or_generated_outside_root:{view_id or index}"
            )
            continue
        real = _image(real_path)
        generated = _image(generated_path)
        if real.shape != generated.shape:
            blockers.append(f"artifixer_view_shape_mismatch:{view_id}")
            continue
        difference = real - generated
        mse = float(np.mean(np.square(difference)))
        mae = float(np.mean(np.abs(difference)))
        psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
        rows.append(
            {
                "view_id": view_id,
                "real_view_path": str(real_path),
                "real_view_sha256": sha256_file(real_path),
                "generated_view_path": str(generated_path),
                "generated_view_sha256": sha256_file(generated_path),
                "mean_absolute_error": round(mae, 8),
                "psnr_db": "infinity" if math.isinf(psnr) else round(psnr, 6),
            }
        )
    finite_psnr = [float(row["psnr_db"]) for row in rows if row["psnr_db"] != "infinity"]
    mean_psnr = (
        float("inf")
        if rows and len(finite_psnr) < len(rows)
        else (float(np.mean(finite_psnr)) if finite_psnr else 0.0)
    )
    mean_mae = float(np.mean([row["mean_absolute_error"] for row in rows])) if rows else 1.0
    thresholds_passed = bool(
        rows and mean_psnr >= minimum_mean_psnr_db and mean_mae <= maximum_mean_absolute_error
    )
    if not thresholds_passed:
        blockers.append("artifixer_heldout_real_view_thresholds_not_met")
    result = {
        "schema_version": "artifixer_heldout_real_view_evaluation.v1",
        "generated_at": utc_now_iso(),
        "status": "passed_advisory" if not blockers else "blocked_or_failed",
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file) if manifest_file.is_file() else None,
        "heldout_view_count": len(rows),
        "rows": rows,
        "aggregate": {
            "mean_psnr_db": "infinity" if math.isinf(mean_psnr) else round(mean_psnr, 6),
            "mean_absolute_error": round(mean_mae, 8),
            "minimum_mean_psnr_db": minimum_mean_psnr_db,
            "maximum_mean_absolute_error": maximum_mean_absolute_error,
            "thresholds_passed": thresholds_passed,
        },
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "generated_support_quality_measured_on_heldout_real_views": not blockers,
            "generated_pixels_are_capture_truth": False,
            "generated_geometry_is_collision_truth": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
        },
    }
    result["result_fingerprint"] = canonical_sha256(result)
    write_json(Path(output_path), result)
    return result
