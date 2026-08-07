"""Fail-closed exact-camera conformance for the official native Aura renderer."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adp009d_aura_native_vast import (
    EXPECTED_AURA_PLY_SHA256,
    PROBE_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TREE,
    _sha256,
)
from .adp009d_aura_renderer_conformance import (
    FROZEN_THRESHOLDS,
    THRESHOLD_DEFINITION_COMMIT,
)
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .heldout_appearance_evaluation_v2 import _global_ssim, windowed_ssim


RECEIPT_SCHEMA_VERSION = "adp009d_aura_native_renderer_conformance_receipt.v1"


class AuraNativeConformanceError(ValueError):
    """Stable fail-closed native-renderer conformance error."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _mapping(path: Path, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuraNativeConformanceError([error]) from exc
    if not isinstance(value, dict):
        raise AuraNativeConformanceError([error])
    return value


def _bound_file(root: Path, path: Path, *, error: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AuraNativeConformanceError([f"{error}_outside_evidence_root"]) from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise AuraNativeConformanceError([f"{error}_missing"])
    return resolved


def _artifact(
    *,
    result_root: Path,
    row: Mapping[str, Any],
    suffix: str,
    evidence_root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    matches = [
        dict(item)
        for item in row.get("artifacts", [])
        if isinstance(item, Mapping) and str(item.get("path") or "").endswith(suffix)
    ]
    if len(matches) != 1:
        raise AuraNativeConformanceError([f"{label}_artifact_identity_invalid"])
    artifact = matches[0]
    path = _bound_file(
        evidence_root,
        result_root / str(artifact.get("path") or ""),
        error=label,
    )
    if _sha256(path) != artifact.get("sha256"):
        raise AuraNativeConformanceError([f"{label}_digest_mismatch"])
    return path, artifact


def _rgb(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    except (OSError, ValueError) as exc:
        raise AuraNativeConformanceError([f"{label}_unreadable"]) from exc
    if value.size == 0 or not np.isfinite(value).all():
        raise AuraNativeConformanceError([f"{label}_invalid"])
    return value


def materialize_aura_native_conformance_receipt(
    *,
    probe_manifest_path: str | Path,
    provider_bundle_receipt_path: str | Path,
    native_result_path: str | Path,
    evidence_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive conformance only from digest-bound native outputs and oracle bytes."""

    root = Path(evidence_root).expanduser().resolve()
    if not root.is_dir():
        raise AuraNativeConformanceError(["aura_native_conformance_evidence_root_invalid"])
    probe_path = _bound_file(
        root,
        Path(probe_manifest_path),
        error="aura_native_conformance_probe",
    )
    bundle_path = _bound_file(
        root,
        Path(provider_bundle_receipt_path),
        error="aura_native_conformance_bundle_receipt",
    )
    result_path = _bound_file(
        root,
        Path(native_result_path),
        error="aura_native_conformance_result",
    )
    probe = _mapping(probe_path, "aura_native_conformance_probe_invalid")
    bundle = _mapping(bundle_path, "aura_native_conformance_bundle_receipt_invalid")
    result = _mapping(result_path, "aura_native_conformance_result_invalid")
    errors: list[str] = []
    if (
        probe.get("schema_version") != PROBE_SCHEMA_VERSION
        or probe.get("status") != "materialized_unexecuted"
        or probe.get("manifest_digest")
        != canonical_digest(probe, digest_field="manifest_digest")
        or probe.get("conformance_thresholds") != FROZEN_THRESHOLDS
        or probe.get("threshold_definition_commit") != THRESHOLD_DEFINITION_COMMIT
        or probe.get("thresholds_frozen_before_execution") is not True
        or probe.get("renderer_outcomes_observed_before_freeze") is not False
        or probe.get("aura_ply_sha256") != EXPECTED_AURA_PLY_SHA256
    ):
        errors.append("aura_native_conformance_probe_invalid")
    if (
        bundle.get("status") != "ready"
        or bundle.get("source_probe_manifest_digest") != probe.get("manifest_digest")
        or bundle.get("conformance_thresholds") != FROZEN_THRESHOLDS
        or bundle.get("threshold_definition_commit") != THRESHOLD_DEFINITION_COMMIT
        or bundle.get("source_repository") != SOURCE_REPOSITORY
        or bundle.get("source_commit") != SOURCE_COMMIT
        or bundle.get("source_tree") != SOURCE_TREE
        or bundle.get("aura_ply_sha256") != EXPECTED_AURA_PLY_SHA256
    ):
        errors.append("aura_native_conformance_bundle_binding_invalid")
    if (
        result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or result.get("status") != "completed"
        or result.get("blockers")
        or result.get("input_digest") != bundle.get("input_digest")
        or result.get("source_probe_manifest_digest") != probe.get("manifest_digest")
        or result.get("source_repository") != SOURCE_REPOSITORY
        or result.get("source_commit") != SOURCE_COMMIT
        or result.get("source_tree") != SOURCE_TREE
        or result.get("source_modified") is not False
        or result.get("aura_ply_sha256") != EXPECTED_AURA_PLY_SHA256
        or result.get("depth_output") != "surf_depth_expected_camera_z_m"
        or result.get("depth_ratio") != 0.0
        or result.get("metric_scene_units") != "meters"
        or result.get("candidate_policy_queried") is not False
        or result.get("candidate_outcomes_accessed") is not False
    ):
        errors.append("aura_native_conformance_result_invalid")
    probe_rows = {
        str(row.get("camera_id")): dict(row)
        for row in probe.get("camera_configs", [])
        if isinstance(row, Mapping)
    }
    result_rows = {
        str(row.get("camera_id")): dict(row)
        for row in result.get("camera_rows", [])
        if isinstance(row, Mapping)
    }
    if set(probe_rows) != set(result_rows) or len(probe_rows) < 2:
        errors.append("aura_native_conformance_camera_set_mismatch")
    if errors:
        raise AuraNativeConformanceError(errors)

    rows: list[dict[str, Any]] = []
    result_root = result_path.parent
    for camera_id in sorted(probe_rows):
        expected = probe_rows[camera_id]
        observed = result_rows[camera_id]
        if (
            observed.get("valid") is not True
            or observed.get("calibration_digest") != expected.get("calibration_digest")
            or observed.get("native_reference_sha256")
            != expected.get("native_reference_sha256")
        ):
            raise AuraNativeConformanceError(
                [f"aura_native_conformance_camera_binding_invalid:{camera_id}"]
            )
        native_path = _bound_file(
            root,
            Path(str(expected.get("native_reference_path") or "")),
            error=f"aura_native_reference:{camera_id}",
        )
        if _sha256(native_path) != expected.get("native_reference_sha256"):
            raise AuraNativeConformanceError(
                [f"aura_native_reference_digest_mismatch:{camera_id}"]
            )
        rgb_path, rgb_artifact = _artifact(
            result_root=result_root,
            row=observed,
            suffix=f"{camera_id}/rgb.png",
            evidence_root=root,
            label=f"aura_native_rgb:{camera_id}",
        )
        depth_path, depth_artifact = _artifact(
            result_root=result_root,
            row=observed,
            suffix=f"{camera_id}/depth_m.npy",
            evidence_root=root,
            label=f"aura_native_depth:{camera_id}",
        )
        alpha_path, alpha_artifact = _artifact(
            result_root=result_root,
            row=observed,
            suffix=f"{camera_id}/alpha.npy",
            evidence_root=root,
            label=f"aura_native_alpha:{camera_id}",
        )
        native = _rgb(native_path, label=f"aura_native_reference:{camera_id}")
        rendered = _rgb(rgb_path, label=f"aura_native_rgb:{camera_id}")
        if native.shape != rendered.shape:
            raise AuraNativeConformanceError(
                [f"aura_native_conformance_rgb_shape_mismatch:{camera_id}"]
            )
        try:
            depth = np.load(depth_path, allow_pickle=False)
            alpha = np.load(alpha_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AuraNativeConformanceError(
                [f"aura_native_conformance_depth_unreadable:{camera_id}"]
            ) from exc
        expected_shape = native.shape[:2]
        valid_depth = np.isfinite(depth) & (depth > 0) & (alpha > 0)
        if (
            depth.shape != expected_shape
            or alpha.shape != expected_shape
            or depth.dtype != np.float32
            or alpha.dtype != np.float32
            or not np.isfinite(alpha).all()
            or float(alpha.min()) < -1.0e-6
            or float(alpha.max()) > 1.0 + 1.0e-6
            or not valid_depth.any()
            or int(valid_depth.sum()) != observed.get("positive_finite_depth_count")
        ):
            raise AuraNativeConformanceError(
                [f"aura_native_conformance_metric_depth_invalid:{camera_id}"]
            )
        difference = native - rendered
        mse = float(np.mean(difference**2))
        rows.append(
            {
                "camera_id": camera_id,
                "calibration_digest": expected["calibration_digest"],
                "native_frame_sha256": expected["native_reference_sha256"],
                "rendered_frame_sha256": rgb_artifact["sha256"],
                "depth_m_sha256": depth_artifact["sha256"],
                "alpha_sha256": alpha_artifact["sha256"],
                "psnr_db": (
                    "infinity" if mse == 0 else round(10 * math.log10(1 / mse), 6)
                ),
                "global_ssim": round(_global_ssim(native, rendered), 8),
                "windowed_ssim": round(windowed_ssim(native, rendered), 8),
                "mean_absolute_error": round(float(np.mean(np.abs(difference))), 8),
                "positive_finite_depth_count": int(valid_depth.sum()),
                "minimum_positive_depth_m": round(float(depth[valid_depth].min()), 8),
                "maximum_positive_depth_m": round(float(depth[valid_depth].max()), 8),
            }
        )
    finite_psnr = [
        float(row["psnr_db"]) for row in rows if row["psnr_db"] != "infinity"
    ]
    mean_psnr: str | float = (
        "infinity"
        if len(finite_psnr) != len(rows)
        else round(float(np.mean(finite_psnr)), 6)
    )
    aggregate = {
        "camera_count": len(rows),
        "mean_psnr_db": mean_psnr,
        "mean_global_ssim": round(
            float(np.mean([row["global_ssim"] for row in rows])), 8
        ),
        "mean_windowed_ssim": round(
            float(np.mean([row["windowed_ssim"] for row in rows])), 8
        ),
        "mean_absolute_error": round(
            float(np.mean([row["mean_absolute_error"] for row in rows])), 8
        ),
        "minimum_windowed_ssim": min(float(row["windowed_ssim"]) for row in rows),
        "maximum_absolute_error": max(
            float(row["mean_absolute_error"]) for row in rows
        ),
    }
    passed = (
        (mean_psnr == "infinity" or float(mean_psnr) >= FROZEN_THRESHOLDS["minimum_mean_psnr_db"])
        and aggregate["mean_global_ssim"]
        >= FROZEN_THRESHOLDS["minimum_mean_global_ssim"]
        and aggregate["mean_windowed_ssim"]
        >= FROZEN_THRESHOLDS["minimum_mean_windowed_ssim"]
        and aggregate["mean_absolute_error"]
        <= FROZEN_THRESHOLDS["maximum_mean_absolute_error"]
        and aggregate["minimum_windowed_ssim"]
        >= FROZEN_THRESHOLDS["minimum_per_camera_windowed_ssim"]
        and aggregate["maximum_absolute_error"]
        <= FROZEN_THRESHOLDS["maximum_per_camera_absolute_error"]
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": (
            "passed_exact_camera_conformance"
            if passed
            else "rejected_exact_camera_conformance"
        ),
        "passed": bool(passed),
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_modified": False,
        "source_probe_manifest_digest": probe["manifest_digest"],
        "provider_bundle_sha256": bundle.get("bundle_sha256"),
        "run_input_digest": result["input_digest"],
        "aura_ply_sha256": EXPECTED_AURA_PLY_SHA256,
        "aura_native_render_manifest_digest": probe[
            "aura_native_render_manifest_digest"
        ],
        "threshold_definition_commit": THRESHOLD_DEFINITION_COMMIT,
        "thresholds": FROZEN_THRESHOLDS,
        "thresholds_frozen_before_execution": True,
        "renderer_outcomes_observed_before_threshold_freeze": False,
        "depth_output": "surf_depth_expected_camera_z_m",
        "depth_ratio": 0.0,
        "metric_scene_units": "meters",
        "rows": rows,
        "aggregate": aggregate,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "claim_ceiling": (
            "Official native Aura construction equivalence and metric expected "
            "depth at two frozen exact cameras only"
        ),
        "policy_observation_admitted_by_this_receipt_alone": False,
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    validated = validate_aura_native_conformance_receipt(receipt)
    write_json(Path(output_path).expanduser().resolve(), validated)
    return validated


def validate_aura_native_conformance_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AuraNativeConformanceError(["aura_native_conformance_receipt_invalid"])
    receipt = json.loads(json.dumps(value))
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("aura_native_conformance_receipt_schema_invalid")
    if (
        receipt.get("status") != "passed_exact_camera_conformance"
        or receipt.get("passed") is not True
    ):
        errors.append("aura_native_conformance_not_passed")
    if receipt.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("aura_native_conformance_thresholds_invalid")
    if (
        receipt.get("source_repository") != SOURCE_REPOSITORY
        or receipt.get("source_commit") != SOURCE_COMMIT
        or receipt.get("source_tree") != SOURCE_TREE
        or receipt.get("source_modified") is not False
    ):
        errors.append("aura_native_conformance_source_identity_invalid")
    if receipt.get("aura_ply_sha256") != EXPECTED_AURA_PLY_SHA256:
        errors.append("aura_native_conformance_ply_identity_invalid")
    if receipt.get("candidate_policy_queried") is not False:
        errors.append("aura_native_conformance_policy_query_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("aura_native_conformance_receipt_digest_mismatch")
    if errors:
        raise AuraNativeConformanceError(errors)
    return receipt


__all__ = [
    "AuraNativeConformanceError",
    "RECEIPT_SCHEMA_VERSION",
    "materialize_aura_native_conformance_receipt",
    "validate_aura_native_conformance_receipt",
]
