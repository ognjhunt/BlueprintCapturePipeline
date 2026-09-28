"""Admit an immutable training seed before models or paid GPU execution.

Only a tiny inherited far-field population outside every admitted camera's
bounded support can be quarantined. Original bytes and index lineage survive;
this is source conditioning, never a repair of learned tensors or pixel proof.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_field_quality import (
    MAX_CENTER_DISTANCE_TO_ROBUST_DIAGONAL,
    measure_gaussian_field_quality,
)
from .gaussian_splat_decode import read_standard_3dgs_ply, write_standard_3dgs_ply_subset_exact

MAX_QUARANTINE_FRACTION = 0.001
SUPPORT_SIGMA = 8.0
IMAGE_PADDING_PIXELS = 64.0


def _record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + digest.hexdigest(),
    }


def _camera_exclusion(positions, scales, cameras):
    if not cameras:
        raise ValueError("artifixer_source_geometry_cameras_missing")
    radius = SUPPORT_SIGMA * scales.max(axis=1)
    proof = []
    for row in cameras:
        matrix = np.asarray(row["T_world_camera_opencv"], dtype=np.float64)
        k = row["intrinsics"]
        values = np.asarray([k[name] for name in ("fl_x", "fl_y", "cx", "cy", "w", "h")])
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.isfinite(values).all()
            or min(values[[0, 1, 4, 5]]) <= 0
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)
            or any(float(k.get(name, 0)) != 0 for name in ("k1", "k2", "p1", "p2"))
        ):
            raise ValueError("artifixer_source_geometry_camera_invalid")
        fx, fy, cx, cy, w, h = values
        pad = IMAGE_PADDING_PIXELS
        planes = np.array(
            [
                [0, 0, 1],
                [fx, 0, cx + pad],
                [-fx, 0, w - cx + pad],
                [0, fy, cy + pad],
                [0, -fy, h - cy + pad],
            ]
        )
        centers = (positions - matrix[:3, 3]) @ matrix[:3, :3]
        upper = centers @ planes.T + radius[:, None] * np.linalg.norm(planes, axis=1)
        separation = np.min(upper / np.linalg.norm(planes, axis=1), axis=1)
        if not np.all(separation < 0):
            raise ValueError("artifixer_source_geometry_outlier_camera_support_overlap")
        proof.append(
            {
                "camera_id": str(row["camera_id"]),
                "T_world_camera_opencv": matrix.tolist(),
                "intrinsics": dict(k),
                "excluded_count": int(len(positions)),
                "minimum_separation_m": float(-separation.max()),
            }
        )
    return proof


def admit_source_geometry(
    *, source: Path, cameras: Sequence[Mapping[str, Any]], output_root: Path
) -> tuple[Path, dict[str, Any]]:
    splat = read_standard_3dgs_ply(source)
    scales = np.exp(np.asarray(splat.scales, dtype=np.float64))
    before = measure_gaussian_field_quality(positions=splat.xyz, activated_scales=scales)
    output_root.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema_version": "artifixer_source_geometry_admission.v1",
        "status": "blocked",
        "source": _record(source),
        "source_quality": before,
        "source_gaussian_count": splat.count,
        "quarantined_gaussian_count": 0,
        "source_bytes_mutated": False,
        "learned_tensors_mutated": False,
        "pixel_equality_claimed": False,
        "physical_evidence": False,
        "claim_boundary": "development_only_pretraining_source_conditioning",
        "blockers": [],
    }
    result = source
    try:
        if before["blockers"]:
            if before["blockers"] != ["gaussian_field_center_outlier_above_ceiling"]:
                raise ValueError("artifixer_source_geometry_quality_invalid")
            metrics = before["metrics"]
            center = (np.asarray(metrics["robust_bounds_min"]) + metrics["robust_bounds_max"]) / 2
            outside = np.linalg.norm(splat.xyz - center, axis=1) > (
                MAX_CENTER_DISTANCE_TO_ROBUST_DIAGONAL * metrics["robust_diagonal"]
            )
            indices = np.flatnonzero(outside)
            if not len(indices) or len(indices) / splat.count > MAX_QUARANTINE_FRACTION:
                raise ValueError("artifixer_source_geometry_quarantine_fraction_exceeded")
            proof = _camera_exclusion(splat.xyz[outside], scales[outside], cameras)
            after = measure_gaussian_field_quality(
                positions=splat.xyz[~outside], activated_scales=scales[~outside]
            )
            if after["status"] != "qualified":
                raise ValueError("artifixer_source_geometry_conditioned_quality_invalid")
            result = output_root / "initialization.ply"
            excluded = output_root / "quarantined_source_rows.ply"
            write_standard_3dgs_ply_subset_exact(source, result, np.flatnonzero(~outside))
            write_standard_3dgs_ply_subset_exact(source, excluded, indices)
            index_path = output_root / "quarantined_source_indices.json"
            index_path.write_text(
                canonical_json(
                    {
                        "index_space": "zero_based_original_retained_source",
                        "indices": indices.tolist(),
                    }
                )
                + "\n"
            )
            receipt.update(
                quarantined_gaussian_count=len(indices),
                quarantine_rows=_record(excluded),
                quarantine_indices=_record(index_path),
                camera_exclusion=proof,
                support_sigma=SUPPORT_SIGMA,
                image_padding_pixels=IMAGE_PADDING_PIXELS,
                maximum_quarantine_fraction=MAX_QUARANTINE_FRACTION,
            )
        else:
            after = before
        receipt.update(
            status="admitted",
            initialization=_record(result),
            initialization_quality=after,
            initialization_gaussian_count=after["metrics"]["count"],
        )
    except ValueError as exc:
        receipt["blockers"] = [str(exc)]
        raise
    finally:
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        (output_root / "source_geometry_admission.json").write_text(canonical_json(receipt) + "\n")
    return result, receipt
