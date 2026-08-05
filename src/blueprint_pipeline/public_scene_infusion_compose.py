"""Compose InFusion's RGB point supplement without dropping publisher SH bands.

The released InFusion ``compose.py`` fixes ``max_sh_degree=0``.  InteriorGS's
decoded publisher PLY uses degree-3 spherical harmonics, so passing it through
that script would either fail or erase 45 ``f_rest_*`` fields.  This narrow
adapter preserves every retained publisher row and gives newly inferred points
zero higher-order coefficients.  Generated points remain visual candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply

SCHEMA_VERSION = "adp009b_infusion_composition_receipt.v1"
SH_C0 = 0.28209479177387814


class InFusionCompositionError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: str | Path, root: Path, code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise InFusionCompositionError([code])
    return resolved


def _read_ascii_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with path.open("r", encoding="ascii") as stream:
            if stream.readline().strip() != "ply":
                raise InFusionCompositionError(["infusion_supplement_not_ply"])
            if stream.readline().strip() != "format ascii 1.0":
                raise InFusionCompositionError(["infusion_supplement_format_unsupported"])
            count: int | None = None
            properties: list[str] = []
            for raw in stream:
                line = raw.strip()
                if line.startswith("element vertex "):
                    count = int(line.rsplit(" ", 1)[1])
                elif line.startswith("property "):
                    properties.append(line.rsplit(" ", 1)[1])
                elif line == "end_header":
                    break
            required = ["x", "y", "z", "red", "green", "blue"]
            if count is None or properties != required:
                raise InFusionCompositionError(["infusion_supplement_schema_invalid"])
            table = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, InFusionCompositionError):
            raise
        raise InFusionCompositionError(["infusion_supplement_unreadable"]) from exc
    if table.shape != (count, 6) or count < 4 or not np.isfinite(table).all():
        raise InFusionCompositionError(["infusion_supplement_values_invalid"])
    colors = table[:, 3:]
    if np.any(colors < 0) or np.any(colors > 255) or not np.allclose(colors, np.rint(colors)):
        raise InFusionCompositionError(["infusion_supplement_colors_invalid"])
    return table[:, :3].astype(np.float32), np.rint(colors).astype(np.uint8)


def _statistical_inliers(points: np.ndarray, *, neighbors: int, std_ratio: float) -> np.ndarray:
    if points.shape[0] <= neighbors:
        return np.ones(points.shape[0], dtype=bool)
    distances, _ = cKDTree(points).query(points, k=neighbors + 1, workers=1)
    means = np.asarray(distances[:, 1:], dtype=np.float64).mean(axis=1)
    threshold = float(means.mean() + std_ratio * means.std())
    return means <= threshold


def _supplement_scales(points: np.ndarray) -> np.ndarray:
    distances, _ = cKDTree(points).query(points, k=4, workers=1)
    mean_squared = np.square(np.asarray(distances[:, 1:], dtype=np.float64)).mean(axis=1)
    scale = np.log(np.sqrt(np.maximum(mean_squared, 1e-7))).astype(np.float32)
    return np.repeat(scale[:, None], 3, axis=1)


def _remove_author_default_floaters(
    original: SplatData,
    supplement_xyz: np.ndarray,
    *,
    similarity_threshold_m: float,
    radius_m: float,
    radius_min_neighbors: int,
) -> np.ndarray:
    supplement_tree = cKDTree(supplement_xyz)
    near = np.asarray(
        supplement_tree.query(original.xyz, k=1, workers=1)[0] < similarity_threshold_m,
        dtype=bool,
    )
    original_tree = cKDTree(original.xyz)
    neighbor_counts = original_tree.query_ball_point(
        original.xyz, radius_m, return_length=True, workers=1
    )
    # Open3D includes the query point in the radius count.  InFusion removes only
    # sparse original points that are also near the predicted supplement.
    radius_inlier = np.asarray(neighbor_counts) >= radius_min_neighbors
    return ~(near & ~radius_inlier)


def _subset(splat: SplatData, keep: np.ndarray) -> SplatData:
    return SplatData(
        count=int(np.count_nonzero(keep)),
        xyz=splat.xyz[keep].copy(),
        opacity=splat.opacity[keep].copy(),
        f_dc=splat.f_dc[keep].copy(),
        scales=splat.scales[keep].copy(),
        quats=splat.quats[keep].copy(),
        properties=splat.properties,
        sh_rest=None if splat.sh_rest is None else splat.sh_rest[keep].copy(),
    )


def _atomic_write_splat(splat: SplatData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".ply", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_standard_3dgs_ply(splat, temporary)
        if output.exists():
            if not output.is_file() or _sha256(output) != _sha256(temporary):
                raise InFusionCompositionError(["infusion_composition_output_conflict"])
            return
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def compose_infusion_supplement(
    *,
    original_ply: str | Path,
    supplement_ply: str | Path,
    output_ply: str | Path,
    data_root: str | Path,
    similarity_threshold_m: float = 1.0,
    radius_m: float = 0.1,
    radius_min_neighbors: int = 100,
) -> dict[str, Any]:
    """Compose an observed InFusion supplement with a degree-preserving PLY."""

    root = Path(data_root).expanduser().resolve()
    original_path = _under(original_ply, root, "infusion_original_outside_data_root")
    supplement_path = _under(supplement_ply, root, "infusion_supplement_outside_data_root")
    output_path = _under(output_ply, root, "infusion_output_outside_data_root")
    if not original_path.is_file() or original_path.is_symlink():
        raise InFusionCompositionError(["infusion_original_missing_or_symlink"])
    if not supplement_path.is_file() or supplement_path.is_symlink():
        raise InFusionCompositionError(["infusion_supplement_missing_or_symlink"])
    if not math.isfinite(similarity_threshold_m) or similarity_threshold_m <= 0:
        raise InFusionCompositionError(["infusion_similarity_threshold_invalid"])
    if not math.isfinite(radius_m) or radius_m <= 0 or radius_min_neighbors < 1:
        raise InFusionCompositionError(["infusion_radius_filter_invalid"])

    try:
        original = read_standard_3dgs_ply(original_path)
    except ValueError as exc:
        raise InFusionCompositionError(["infusion_original_ply_invalid"]) from exc
    if original.sh_rest is None:
        raise InFusionCompositionError(["infusion_original_higher_order_sh_missing"])
    coefficient_count = 1 + original.sh_rest.shape[1] // 3
    sh_degree = int(round(coefficient_count**0.5)) - 1
    if sh_degree < 1 or (sh_degree + 1) ** 2 != coefficient_count:
        raise InFusionCompositionError(["infusion_original_sh_degree_invalid"])

    supplement_xyz, supplement_rgb = _read_ascii_rgb_ply(supplement_path)
    supplement_keep = _statistical_inliers(supplement_xyz, neighbors=5, std_ratio=4.0)
    supplement_xyz = supplement_xyz[supplement_keep]
    supplement_rgb = supplement_rgb[supplement_keep]
    if supplement_xyz.shape[0] < 4:
        raise InFusionCompositionError(["infusion_supplement_too_sparse_after_filter"])

    original_keep = _remove_author_default_floaters(
        original,
        supplement_xyz,
        similarity_threshold_m=similarity_threshold_m,
        radius_m=radius_m,
        radius_min_neighbors=radius_min_neighbors,
    )
    retained = _subset(original, original_keep)
    supplement_count = supplement_xyz.shape[0]
    f_dc = (supplement_rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0
    supplement = SplatData(
        count=supplement_count,
        xyz=supplement_xyz,
        opacity=np.full(supplement_count, np.float32(math.log(0.999 / 0.001))),
        f_dc=f_dc,
        scales=_supplement_scales(supplement_xyz),
        quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (supplement_count, 1)),
        properties=original.properties,
        sh_rest=np.zeros((supplement_count, original.sh_rest.shape[1]), dtype=np.float32),
    )
    composed = SplatData(
        count=retained.count + supplement.count,
        xyz=np.concatenate((retained.xyz, supplement.xyz)),
        opacity=np.concatenate((retained.opacity, supplement.opacity)),
        f_dc=np.concatenate((retained.f_dc, supplement.f_dc)),
        scales=np.concatenate((retained.scales, supplement.scales)),
        quats=np.concatenate((retained.quats, supplement.quats)),
        properties=original.properties,
        sh_rest=np.concatenate((retained.sh_rest, supplement.sh_rest)),
    )
    _atomic_write_splat(composed, output_path)
    observed = read_standard_3dgs_ply(output_path)
    if observed.count != composed.count or observed.sh_rest is None:
        raise InFusionCompositionError(["infusion_composition_roundtrip_failed"])
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "inputs": {
            "original": {"sha256": _sha256(original_path), "size_bytes": original_path.stat().st_size},
            "supplement": {"sha256": _sha256(supplement_path), "size_bytes": supplement_path.stat().st_size},
        },
        "parameters": {
            "supplement_statistical_neighbors": 5,
            "supplement_statistical_std_ratio": 4.0,
            "similarity_threshold_m": similarity_threshold_m,
            "radius_m": radius_m,
            "radius_min_neighbors": radius_min_neighbors,
        },
        "counts": {
            "original": original.count,
            "original_retained": retained.count,
            "original_floater_rows_removed": original.count - retained.count,
            "supplement_input": int(supplement_keep.shape[0]),
            "supplement_retained": supplement.count,
            "composed": composed.count,
        },
        "spherical_harmonics": {
            "publisher_degree": sh_degree,
            "publisher_f_rest_fields_preserved": original.sh_rest.shape[1],
            "supplement_f_rest_initialization": "zero",
        },
        "output": {"sha256": _sha256(output_path), "size_bytes": output_path.stat().st_size},
        "proof_boundaries": {
            "generated_supplement_is_visual_candidate_only": True,
            "composition_is_not_metric_surface_truth": True,
            "composition_is_not_inpaint_method_execution": True,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-ply", required=True)
    parser.add_argument("--supplement-ply", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--receipt-output", required=True)
    args = parser.parse_args(argv)
    receipt = compose_infusion_supplement(
        original_ply=args.original_ply,
        supplement_ply=args.supplement_ply,
        output_ply=args.output_ply,
        data_root=args.data_root,
    )
    receipt_path = _under(args.receipt_output, Path(args.data_root).expanduser().resolve(), "infusion_receipt_outside_data_root")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(receipt) + "\n"
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != payload:
        raise InFusionCompositionError(["infusion_receipt_output_conflict"])
    receipt_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
