"""Reference model of NVIDIA 3DGRUT's direct NuRec -> LightField transcode.

Scene 839873's rendering audit needed to know whether Blueprint's private
NuRec-tensor -> standard-PLY -> ``usd-convert-gsplat`` conversion could ever be
treated as equivalent to the transcode NVIDIA documents.  Equivalence cannot
be asserted from digests of the *inputs*; it has to be shown attribute by
attribute on the *outputs*.  This module transcribes, in pure numpy, exactly
what the pinned upstream revision does between reading ``.nurec`` state and
authoring ``ParticleField3DGaussianSplat`` attributes:

* ``threedgrut/export/importers/nurec_usd.py`` -- reads the six pre-activation
  tensors as float32, applies the Volume's local-to-world transform to
  positions (row-vector ``p @ M^T``), rotations (``q_vol * q``) and scales
  (column norms), and leaves albedo/specular untouched.
* ``threedgrut/export/adapter.py`` -- ``exp`` on scales, ``sigmoid`` on
  densities, ``normalize`` on rotations (wxyz).
* ``threedgrut/export/usd/writers/lightfield.py`` -- clips opacities to
  ``[0, 1]`` and lays the radiance out as ``[albedo, specular.reshape(N, 15, 3)]``
  flattened to ``(N * 16, 3)``, i.e. coefficient-major RGB triplets with no
  channel transpose.

Everything here is pinned to :data:`THREEDGRUT_REFERENCE_REVISION`.  It is a
comparator, not a converter: production assets come from the real transcode
(:func:`blueprint_pipeline.isaac_nurec_export.transcode_nurec_usdz_to_particlefield`),
and this model exists so a hermetic test can pin the contract and so a receipt
can carry per-attribute digests computed the upstream way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

THREEDGRUT_REFERENCE_REVISION = "a37ef721012dea0f29c0fcfff2d525023b4e854a"
THREEDGRUT_REFERENCE_FILES = (
    "threedgrut/export/importers/nurec_usd.py",
    "threedgrut/export/adapter.py",
    "threedgrut/export/usd/writers/lightfield.py",
)
#: Prim-level authoring the upstream LightField writer performs that Blueprint's
#: current ``usd-convert-gsplat`` route does not.  These are the only known
#: differences between the two outputs; the learned attributes are identical.
UPSTREAM_ONLY_PRIM_AUTHORING = {
    "projectionModeHint": "perspective",
    "sortingModeHint": "cameraDistance",
    "colorSpace:name": "srgb_rec709_display",
    "customLayerData.renderSettings": {"rtx:post:tonemap:op": 2},
}


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Transcribed from ``nurec_usd._rotation_matrix_to_quat_wxyz``."""

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / ((1.0 + trace) ** 0.5 + 1e-8)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def apply_nurec_volume_transform(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    matrix_row_major: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transcribed from ``nurec_usd._apply_volume_transform`` (Gf row-major)."""

    t = np.asarray(matrix_row_major, dtype=np.float64)
    ones = np.ones((positions.shape[0], 1), dtype=np.float64)
    p4 = np.hstack([positions.astype(np.float64), ones])
    positions_out = (p4 @ t.T)[:, :3].astype(np.float32)
    lin = t[:3, :3]
    scale_factors = np.maximum(
        np.array([np.linalg.norm(lin[:, i]) for i in range(3)], dtype=np.float32), 1e-8
    )
    scales_out = scales * scale_factors
    q_vol = _rotation_matrix_to_quat_wxyz(lin / scale_factors.astype(np.float64))
    qw, qx, qy, qz = q_vol
    rw, rx, ry, rz = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    rotations_out = np.stack(
        [
            qw * rw - qx * rx - qy * ry - qz * rz,
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
        ],
        axis=1,
    ).astype(np.float32)
    return positions_out, rotations_out, scales_out


def threedgrut_lightfield_attributes(
    arrays: Mapping[str, np.ndarray],
    *,
    volume_transform_row_major: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Post-activation ParticleField attributes exactly as 3DGRUT would author them.

    ``arrays`` are the pre-activation ``.nurec`` tensors (``positions``,
    ``rotations``, ``scales``, ``densities``, ``features_albedo``,
    ``features_specular``) at their stored precision; they are widened to
    float32 first, as the upstream importer does.
    """

    positions = np.asarray(arrays["positions"]).astype(np.float32)
    rotations = np.asarray(arrays["rotations"]).astype(np.float32)
    scales = np.asarray(arrays["scales"]).astype(np.float32)
    densities = np.asarray(arrays["densities"]).astype(np.float32).reshape(-1, 1)
    albedo = np.asarray(arrays["features_albedo"]).astype(np.float32)
    specular = np.asarray(arrays["features_specular"]).astype(np.float32)
    n = positions.shape[0]
    if volume_transform_row_major is not None:
        positions, rotations, scales = apply_nurec_volume_transform(
            positions, rotations, scales, volume_transform_row_major
        )
    n_spec = specular.shape[1]
    sh_degree = max(0, min(3, int(round((n_spec // 3 + 1) ** 0.5 - 1))))
    # adapter.py: exp / sigmoid / normalize, evaluated in float32 like torch.
    activated_scales = np.exp(scales).astype(np.float32)
    with np.errstate(over="ignore"):
        opacities = (1.0 / (1.0 + np.exp(-densities.astype(np.float32)))).astype(np.float32)
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    orientations = (rotations / np.maximum(norms, 1e-12)).astype(np.float32)
    # lightfield.py: clip densities; radiance = [albedo | specular (N, M, 3)].
    opacities = np.clip(opacities.reshape(-1), 0.0, 1.0).astype(np.float32)
    if sh_degree == 0:
        sh = albedo.reshape(-1, 3)
    else:
        rest = (sh_degree + 1) ** 2 - 1
        sh = np.concatenate([albedo.reshape(n, 1, 3), specular.reshape(n, rest, 3)], axis=1)
        sh = sh.reshape(-1, 3)
    return {
        "count": int(n),
        "positions": np.ascontiguousarray(positions, dtype=np.float32),
        "scales": np.ascontiguousarray(activated_scales, dtype=np.float32),
        "orientations": np.ascontiguousarray(orientations, dtype=np.float32),
        "opacities": np.ascontiguousarray(opacities, dtype=np.float32),
        "sh_coefficients": np.ascontiguousarray(sh, dtype=np.float32),
        "sh_degree": sh_degree,
        "sh_element_size": (sh_degree + 1) ** 2,
        "reference_revision": THREEDGRUT_REFERENCE_REVISION,
    }


def attribute_digests(attributes: Mapping[str, Any]) -> dict[str, str]:
    """sha256 of each float32 attribute buffer, for receipt-level comparison."""

    return {
        name: "sha256:"
        + hashlib.sha256(np.ascontiguousarray(attributes[name], dtype=np.float32).tobytes()).hexdigest()
        for name in ("positions", "scales", "orientations", "opacities", "sh_coefficients")
    }


def compare_particlefield_attributes(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Per-attribute parity report; quaternions compare up to sign."""

    report: dict[str, Any] = {"passed": True, "attributes": {}}
    if int(candidate["count"]) != int(reference["count"]):
        report["passed"] = False
        report["count"] = {"candidate": int(candidate["count"]), "reference": int(reference["count"])}
        return report
    for name in ("positions", "scales", "opacities", "sh_coefficients"):
        a = np.asarray(candidate[name], dtype=np.float64)
        b = np.asarray(reference[name], dtype=np.float64)
        if a.shape != b.shape:
            row = {"passed": False, "shape_candidate": list(a.shape), "shape_reference": list(b.shape)}
        else:
            diff = float(np.abs(a - b).max()) if a.size else 0.0
            row = {"passed": diff <= atol, "max_abs_diff": diff}
        report["attributes"][name] = row
        report["passed"] = report["passed"] and row["passed"]
    qa = np.asarray(candidate["orientations"], dtype=np.float64)
    qb = np.asarray(reference["orientations"], dtype=np.float64)
    if qa.shape != qb.shape:
        row = {"passed": False, "shape_candidate": list(qa.shape), "shape_reference": list(qb.shape)}
    else:
        sign = np.sign(np.sum(qa * qb, axis=1, keepdims=True))
        sign[sign == 0] = 1.0
        diff = float(np.abs(qa - qb * sign).max()) if qa.size else 0.0
        row = {"passed": diff <= atol, "max_abs_diff_up_to_sign": diff}
    report["attributes"]["orientations"] = row
    report["passed"] = report["passed"] and row["passed"]
    for name in ("sh_degree", "sh_element_size"):
        same = int(candidate[name]) == int(reference[name])
        report["attributes"][name] = {"passed": same, "candidate": int(candidate[name]), "reference": int(reference[name])}
        report["passed"] = report["passed"] and same
    return report


__all__ = [
    "THREEDGRUT_REFERENCE_FILES",
    "THREEDGRUT_REFERENCE_REVISION",
    "UPSTREAM_ONLY_PRIM_AUTHORING",
    "apply_nurec_volume_transform",
    "attribute_digests",
    "compare_particlefield_attributes",
    "threedgrut_lightfield_attributes",
]
