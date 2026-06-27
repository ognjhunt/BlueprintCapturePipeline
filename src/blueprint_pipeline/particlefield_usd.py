"""Author an Isaac-renderable ParticleField3DGaussianSplat USD from a standard 3DGS PLY.

Isaac Sim 6.0 RTX renders the OpenUSD ``ParticleField3DGaussianSplat`` schema (UsdVol)
natively — the preferred, non-deprecated path for Gaussian splats. This module authors
that prim **directly from our standard INRIA 3DGS data in pure Python/pxr**, with no
ncore / 3dgrut / NRE dependency, so a base Isaac image can load + RTX-render free cameras.

Conventions (canonical, taken from NVIDIA 3dgrut's ParticleField exporter):
- ``scales``       = exp(log-scale)                 (activation applied)
- ``orientations`` = normalized quaternion (w,x,y,z) (Gf.Quatf real=w)
- ``opacities``    = sigmoid(opacity logit)
- ``radiance:sphericalHarmonicsCoefficients`` = raw SH coefficients as float3 per coeff
  (the renderer evaluates SH → color); ``radiance:sphericalHarmonicsDegree`` set to match.
  Degree 0 (DC only, from ``f_dc``) is the default safe path; pass higher with ``sh_rest``.

The array math (:func:`build_particlefield_arrays`) is pure numpy and unit-tested. The USD
writer (:func:`write_particlefield_usd`) needs ``pxr`` (OpenUSD / usd-core); it is
fail-closed when pxr is unavailable. This module claims authoring only — not rendering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .gaussian_splat_decode import SplatData, read_standard_3dgs_ply

PARTICLEFIELD_SCHEMA = "ParticleField3DGaussianSplat"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def build_particlefield_arrays(splat: SplatData, *, sh_rest: np.ndarray | None = None) -> dict:
    """Compute the ParticleField attribute arrays from standard 3DGS data (pure numpy).

    Returns float32 arrays ready for USD authoring. ``sh_rest`` (N, 45) higher-order SH,
    INRIA channel-major layout, is optional; without it the field is degree-0 (DC only).
    """
    xyz = np.ascontiguousarray(splat.xyz, dtype=np.float32)
    scales = np.exp(np.asarray(splat.scales, dtype=np.float32)).astype(np.float32)

    q = np.asarray(splat.quats, dtype=np.float64)  # (N,4) = (w, x, y, z) INRIA order
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    quats = (q / norm).astype(np.float32)

    opac = _sigmoid(np.asarray(splat.opacity, dtype=np.float32)).astype(np.float32)

    n = xyz.shape[0]
    dc = np.asarray(splat.f_dc, dtype=np.float32).reshape(n, 1, 3)  # coeff 0 (RGB)
    if sh_rest is not None and np.asarray(sh_rest).size:
        rest = np.asarray(sh_rest, dtype=np.float32)
        n_rest = rest.shape[1] // 3
        # INRIA f_rest is channel-major: [R*n_rest, G*n_rest, B*n_rest] -> (n, n_rest, 3)
        rest = rest.reshape(n, 3, n_rest).transpose(0, 2, 1)
        coeffs = np.concatenate([dc, rest], axis=1)  # (n, 1+n_rest, 3)
        total = coeffs.shape[1]
        degree = int(round(total ** 0.5)) - 1
        sh = coeffs.reshape(n * total, 3).astype(np.float32)
    else:
        degree = 0
        sh = dc.reshape(n, 3).astype(np.float32)

    extent = np.stack([xyz.min(axis=0), xyz.max(axis=0)]).astype(np.float32)
    return {
        "count": int(n),
        "positions": xyz,
        "scales": scales,
        "orientations": quats,  # (w, x, y, z)
        "opacities": opac,
        "sh_coefficients": sh,
        "sh_degree": degree,
        "extent": extent,
    }


def write_particlefield_usd(
    source: str | Path | SplatData,
    out_path: str | Path,
    *,
    sh_rest: np.ndarray | None = None,
    prim_path: str = "/World/CapturedScene/Gaussians",
    up_axis: str = "Z",
    sorting_mode: str = "zDepth",
) -> dict:
    """Author a ParticleField3DGaussianSplat USD from a standard 3DGS PLY (or SplatData).

    Returns a fail-closed status dict; ``status == 'completed'`` means ``out_path`` is a
    valid USD that Isaac Sim 6.0 RTX can render.
    """
    out_path = Path(out_path)
    try:
        from pxr import Usd, UsdGeom, Gf, Sdf, Vt  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - pxr/usd-core not installed here
        return {
            "status": "blocked",
            "blockers": ["usd_core_unavailable"],
            "remediation": "pip install usd-core (authoring runs where pxr exists: locally or the Isaac pod)",
            "error": repr(exc),
        }
    splat = source if isinstance(source, SplatData) else read_standard_3dgs_ply(source)
    arr = build_particlefield_arrays(splat, sh_rest=sh_rest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, "z" if up_axis.upper() == "Z" else "y"))
    UsdGeom.Xform.Define(stage, "/World")
    prim = stage.DefinePrim(prim_path, PARTICLEFIELD_SCHEMA)
    if not prim or not prim.IsValid():
        return {"status": "blocked", "blockers": ["particlefield_schema_unavailable"],
                "remediation": "usd-core build lacks ParticleField3DGaussianSplat (need a recent UsdVol)"}

    def vec3f(a: np.ndarray):
        return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, dtype=np.float32))

    prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(vec3f(arr["positions"]))
    prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set(vec3f(arr["scales"]))
    prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray.FromNumpy(np.ascontiguousarray(arr["opacities"], dtype=np.float32))
    )
    prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array).Set(
        vec3f(arr["sh_coefficients"])
    )
    prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(int(arr["sh_degree"]))
    prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(vec3f(arr["extent"]))
    prim.CreateAttribute("projectionModeHint", Sdf.ValueTypeNames.Token).Set("perspective")
    prim.CreateAttribute("sortingModeHint", Sdf.ValueTypeNames.Token).Set(sorting_mode)

    # quaternions: try numpy fast path, fall back to per-element Gf.Quatf (w, x, y, z)
    q = arr["orientations"]
    quat_attr = prim.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray)
    try:
        quat_attr.Set(Vt.QuatfArray.FromNumpy(np.ascontiguousarray(q, dtype=np.float32)))
    except Exception:  # noqa: BLE001
        quat_attr.Set(Vt.QuatfArray([Gf.Quatf(float(w), float(x), float(y), float(z)) for w, x, y, z in q]))

    stage.GetRootLayer().Save()
    return {
        "status": "completed",
        "output": str(out_path),
        "output_bytes": out_path.stat().st_size if out_path.is_file() else 0,
        "schema": PARTICLEFIELD_SCHEMA,
        "splat_count": arr["count"],
        "sh_degree": arr["sh_degree"],
        "prim_path": prim_path,
        "proof_boundary": "ParticleField USD authoring only; Isaac RTX render is the GPU step.",
    }
