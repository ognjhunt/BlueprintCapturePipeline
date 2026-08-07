"""Exact-camera conformance gate for an Aura OVRTX renderer.

An OVRTX frame is not admitted merely because it is nonblank or has finite
depth.  This evaluator compares renderer output with Aura's method-native
render at identical frozen cameras and binds every consumed image byte.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest
from .heldout_appearance_evaluation_v2 import _global_ssim, windowed_ssim


REQUEST_SCHEMA_VERSION = "adp009d_aura_renderer_conformance_request.v1"
RECEIPT_SCHEMA_VERSION = "adp009d_aura_renderer_conformance_receipt.v1"
OVRTX_REPOSITORY = "https://github.com/NVIDIA-Omniverse/ovrtx"
OVRTX_REVISION = "4b9a5fe6f8becf6c5ff031e167cd4201054a96ce"

# These construction thresholds are code-frozen before the exact-camera OVRTX
# run.  They are intentionally a renderer-conformance gate, not a claim that
# the underlying Aura reconstruction is photorealistically accurate.
FROZEN_THRESHOLDS = {
    "minimum_mean_psnr_db": 18.0,
    "minimum_mean_global_ssim": 0.65,
    "minimum_mean_windowed_ssim": 0.55,
    "maximum_mean_absolute_error": 0.15,
    "minimum_per_camera_windowed_ssim": 0.45,
    "maximum_per_camera_absolute_error": 0.20,
}


class AuraRendererConformanceError(ValueError):
    """Stable fail-closed renderer-conformance errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _safe_file(root: Path, raw: Any, *, label: str) -> Path:
    value = str(raw or "").replace("\\", "/")
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise AuraRendererConformanceError([f"{label}_path_invalid"])
    lexical = root / Path(*relative.parts)
    if lexical.is_symlink():
        raise AuraRendererConformanceError([f"{label}_symlink_forbidden"])
    path = lexical.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AuraRendererConformanceError([f"{label}_path_escape"]) from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise AuraRendererConformanceError([f"{label}_missing"])
    return path


def _rgb(path: Path, *, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    except (OSError, ValueError) as exc:
        raise AuraRendererConformanceError([f"{label}_unreadable"]) from exc
    if value.size == 0 or not np.isfinite(value).all():
        raise AuraRendererConformanceError([f"{label}_pixels_invalid"])
    return value


def _calibration_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise AuraRendererConformanceError([f"{label}_calibration_invalid"])
    intrinsic = value.get("intrinsic_matrix")
    world_from_camera = value.get("world_from_camera")
    resolution = value.get("resolution")
    valid = (
        value.get("camera_model") == "pinhole"
        and isinstance(intrinsic, list)
        and len(intrinsic) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in intrinsic)
        and isinstance(world_from_camera, list)
        and len(world_from_camera) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in world_from_camera)
        and isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(item, int) and item > 0 for item in resolution)
    )
    try:
        numbers = [float(item) for row in intrinsic for item in row]
        numbers.extend(float(item) for row in world_from_camera for item in row)
        valid = valid and all(math.isfinite(item) for item in numbers)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise AuraRendererConformanceError([f"{label}_calibration_invalid"])
    return canonical_digest(dict(value))


def _validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = json.loads(json.dumps(value))
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("aura_renderer_conformance_request_schema_invalid")
    if request.get("status") != "frozen_before_ovrtx_execution":
        errors.append("aura_renderer_conformance_request_not_prospective")
    if request.get("thresholds_frozen_before_ovrtx_execution") is not True:
        errors.append("aura_renderer_conformance_thresholds_not_frozen")
    if request.get("ovrtx_outcomes_observed_before_freeze") is not False:
        errors.append("aura_renderer_conformance_outcomes_seen_before_freeze")
    if request.get("ovrtx_repository") != OVRTX_REPOSITORY:
        errors.append("aura_renderer_conformance_ovrtx_repository_invalid")
    if request.get("ovrtx_revision") != OVRTX_REVISION:
        errors.append("aura_renderer_conformance_ovrtx_revision_invalid")
    if request.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("aura_renderer_conformance_thresholds_not_code_frozen")
    for field in (
        "aura_particlefield_sha256",
        "aura_source_ply_sha256",
        "aura_native_render_manifest_digest",
        "ovrtx_run_input_digest",
    ):
        if not _is_digest(request.get(field)):
            errors.append(f"aura_renderer_conformance_{field}_invalid")
    root = Path(str(request.get("evidence_root") or "")).expanduser().resolve()
    if not root.is_dir():
        errors.append("aura_renderer_conformance_evidence_root_invalid")
    pairs = request.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 2:
        errors.append("aura_renderer_conformance_exact_camera_pairs_insufficient")
    else:
        seen: set[str] = set()
        for index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                errors.append(f"aura_renderer_conformance_pair_invalid:{index}")
                continue
            camera_id = str(pair.get("camera_id") or "")
            if not camera_id or camera_id in seen:
                errors.append(f"aura_renderer_conformance_camera_id_invalid:{index}")
            seen.add(camera_id)
            for field in ("native_frame_sha256", "ovrtx_frame_sha256"):
                if not _is_digest(pair.get(field)):
                    errors.append(
                        f"aura_renderer_conformance_pair_{field}_invalid:{camera_id or index}"
                    )
    supplied_digest = request.pop("request_digest", None)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    if supplied_digest != request["request_digest"]:
        errors.append("aura_renderer_conformance_request_digest_mismatch")
    if errors:
        raise AuraRendererConformanceError(errors)
    return request


def evaluate_aura_renderer_conformance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Measure OVRTX against Aura-native frames at byte-bound exact cameras."""

    request = _validate_request(value)
    root = Path(request["evidence_root"]).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for pair in request["pairs"]:
        camera_id = str(pair["camera_id"])
        native_path = _safe_file(root, pair.get("native_frame_path"), label="aura_native_frame")
        ovrtx_path = _safe_file(root, pair.get("ovrtx_frame_path"), label="aura_ovrtx_frame")
        if _sha256(native_path) != pair["native_frame_sha256"]:
            raise AuraRendererConformanceError(
                [f"aura_renderer_conformance_native_frame_digest_mismatch:{camera_id}"]
            )
        if _sha256(ovrtx_path) != pair["ovrtx_frame_sha256"]:
            raise AuraRendererConformanceError(
                [f"aura_renderer_conformance_ovrtx_frame_digest_mismatch:{camera_id}"]
            )
        native_calibration = _calibration_digest(
            pair.get("native_calibration"), label="aura_native"
        )
        ovrtx_calibration = _calibration_digest(
            pair.get("ovrtx_calibration"), label="aura_ovrtx"
        )
        if native_calibration != ovrtx_calibration:
            raise AuraRendererConformanceError(
                [f"aura_renderer_conformance_camera_mismatch:{camera_id}"]
            )
        native = _rgb(native_path, label="aura_native_frame")
        ovrtx = _rgb(ovrtx_path, label="aura_ovrtx_frame")
        if native.shape != ovrtx.shape:
            raise AuraRendererConformanceError(
                [f"aura_renderer_conformance_frame_shape_mismatch:{camera_id}"]
            )
        resolution = pair["native_calibration"]["resolution"]
        if list(native.shape[1::-1]) != resolution:
            raise AuraRendererConformanceError(
                [f"aura_renderer_conformance_resolution_mismatch:{camera_id}"]
            )
        difference = native - ovrtx
        mse = float(np.mean(np.square(difference)))
        rows.append(
            {
                "camera_id": camera_id,
                "camera_calibration_digest": native_calibration,
                "native_frame_sha256": pair["native_frame_sha256"],
                "ovrtx_frame_sha256": pair["ovrtx_frame_sha256"],
                "psnr_db": "infinity" if mse == 0 else round(10 * math.log10(1 / mse), 6),
                "global_ssim": round(_global_ssim(native, ovrtx), 8),
                "windowed_ssim": round(windowed_ssim(native, ovrtx), 8),
                "mean_absolute_error": round(float(np.mean(np.abs(difference))), 8),
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
        "mean_global_ssim": round(float(np.mean([row["global_ssim"] for row in rows])), 8),
        "mean_windowed_ssim": round(
            float(np.mean([row["windowed_ssim"] for row in rows])), 8
        ),
        "mean_absolute_error": round(
            float(np.mean([row["mean_absolute_error"] for row in rows])), 8
        ),
        "minimum_windowed_ssim": round(
            min(float(row["windowed_ssim"]) for row in rows), 8
        ),
        "maximum_absolute_error": round(
            max(float(row["mean_absolute_error"]) for row in rows), 8
        ),
    }
    thresholds = FROZEN_THRESHOLDS
    passed = (
        (mean_psnr == "infinity" or float(mean_psnr) >= thresholds["minimum_mean_psnr_db"])
        and aggregate["mean_global_ssim"] >= thresholds["minimum_mean_global_ssim"]
        and aggregate["mean_windowed_ssim"] >= thresholds["minimum_mean_windowed_ssim"]
        and aggregate["mean_absolute_error"] <= thresholds["maximum_mean_absolute_error"]
        and aggregate["minimum_windowed_ssim"]
        >= thresholds["minimum_per_camera_windowed_ssim"]
        and aggregate["maximum_absolute_error"]
        <= thresholds["maximum_per_camera_absolute_error"]
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "passed_exact_camera_conformance" if passed else "rejected_exact_camera_conformance",
        "request_digest": request["request_digest"],
        "aura_particlefield_sha256": request["aura_particlefield_sha256"],
        "aura_source_ply_sha256": request["aura_source_ply_sha256"],
        "aura_native_render_manifest_digest": request["aura_native_render_manifest_digest"],
        "ovrtx_run_input_digest": request["ovrtx_run_input_digest"],
        "ovrtx_repository": OVRTX_REPOSITORY,
        "ovrtx_revision": OVRTX_REVISION,
        "thresholds": thresholds,
        "thresholds_frozen_before_ovrtx_execution": True,
        "metric_definitions": {
            "psnr": "rgb_unit_interval_mse",
            "global_ssim": "repository_deterministic_global_equivalent",
            "windowed_ssim": "wang2004_gaussian_11x11_sigma1.5_valid_region_L1",
            "mean_absolute_error": "rgb_unit_interval",
        },
        "rows": rows,
        "aggregate": aggregate,
        "passed": bool(passed),
        "claim_ceiling": "Aura OVRTX construction equivalence at the frozen exact cameras only",
        "policy_observation_admitted_by_this_receipt_alone": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def validate_aura_renderer_conformance_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AuraRendererConformanceError(["aura_renderer_conformance_receipt_invalid"])
    receipt = json.loads(json.dumps(value))
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("aura_renderer_conformance_receipt_schema_invalid")
    if receipt.get("status") != "passed_exact_camera_conformance" or receipt.get("passed") is not True:
        errors.append("aura_renderer_conformance_not_passed")
    if receipt.get("thresholds") != FROZEN_THRESHOLDS:
        errors.append("aura_renderer_conformance_receipt_thresholds_invalid")
    if receipt.get("thresholds_frozen_before_ovrtx_execution") is not True:
        errors.append("aura_renderer_conformance_receipt_not_prospective")
    if receipt.get("ovrtx_repository") != OVRTX_REPOSITORY or receipt.get("ovrtx_revision") != OVRTX_REVISION:
        errors.append("aura_renderer_conformance_receipt_ovrtx_identity_invalid")
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        errors.append("aura_renderer_conformance_receipt_digest_mismatch")
    if errors:
        raise AuraRendererConformanceError(errors)
    return receipt


__all__ = [
    "AuraRendererConformanceError",
    "FROZEN_THRESHOLDS",
    "REQUEST_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "evaluate_aura_renderer_conformance",
    "validate_aura_renderer_conformance_receipt",
]
