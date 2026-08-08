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

from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import (
    GaussianSurfelData,
    SplatData,
    read_aura_2dgs_surfel_ply,
    read_standard_3dgs_ply,
)

PARTICLEFIELD_SCHEMA = "ParticleField3DGaussianSplat"
GAUSSIAN_SURFLET_SCHEMA = "ParticleField+ParticleFieldKernelGaussianSurfletAPI"
GAUSSIAN_SURFLET_RECEIPT_SCHEMA_VERSION = "aura_ovrtx_particlefield_receipt.v1"


# Zeroth-order spherical-harmonic basis constant, for reading a DC coefficient
# back as a display colour: colour = 0.5 + C0 * dc.
SH_C0 = 0.28209479177387814

# The structural Z extent, as a fraction of the smaller learned planar extent.
# Flat has to be relative: a constant epsilon would be thicker than wide for the
# smallest surfels in this field, which is the bug it replaces at a new scale.
STRUCTURAL_Z_SCALE_FRACTION = 0.01

def _sigmoid(x: np.ndarray) -> np.ndarray:
    result = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    result = np.where(np.isposinf(x), 1.0, result)
    return np.where(np.isneginf(x), 0.0, result)


def build_particlefield_arrays(splat: SplatData, *, sh_rest: np.ndarray | None = None) -> dict:
    """Compute the ParticleField attribute arrays from standard 3DGS data (pure numpy).

    Returns float32 arrays ready for USD authoring. ``sh_rest`` (N, 45) higher-order SH,
    INRIA channel-major layout, is optional; without it the field is degree-0 (DC only).
    """
    if isinstance(splat.count, bool) or splat.count < 1:
        raise ValueError("particlefield_splat_count_invalid")
    xyz = np.ascontiguousarray(splat.xyz, dtype=np.float32)
    if xyz.shape != (splat.count, 3):
        raise ValueError("particlefield_position_shape_invalid")
    arrays = (xyz, splat.scales, splat.quats, splat.opacity, splat.f_dc)
    if any(not np.isfinite(np.asarray(value)).all() for value in arrays):
        raise ValueError("particlefield_nonfinite_input")
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
        if (
            rest.ndim != 2
            or rest.shape[0] != n
            or rest.shape[1] % 3
            or not np.isfinite(rest).all()
        ):
            raise ValueError("particlefield_sh_rest_invalid")
        n_rest = rest.shape[1] // 3
        # INRIA f_rest is channel-major: [R*n_rest, G*n_rest, B*n_rest] -> (n, n_rest, 3)
        rest = rest.reshape(n, 3, n_rest).transpose(0, 2, 1)
        coeffs = np.concatenate([dc, rest], axis=1)  # (n, 1+n_rest, 3)
        total = coeffs.shape[1]
        degree = int(round(total ** 0.5)) - 1
        if (degree + 1) ** 2 != total:
            raise ValueError("particlefield_sh_coefficient_count_invalid")
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


def build_gaussian_surflet_arrays(surfel: GaussianSurfelData) -> dict:
    """Decode Aura's exact 2DGS parameters for OpenUSD Gaussian surflets."""

    if isinstance(surfel.count, bool) or surfel.count < 1:
        raise ValueError("aura_2dgs_count_invalid")
    expected_shapes = {
        "positions": (surfel.count, 3),
        "scales": (surfel.count, 2),
        "orientations": (surfel.count, 4),
        "opacities": (surfel.count,),
        "sh_dc": (surfel.count, 3),
        "sh_rest": (surfel.count, 45),
        "mask_logits": (surfel.count, 3),
    }
    values = {
        "positions": surfel.xyz,
        "scales": surfel.scales,
        "orientations": surfel.quats,
        "opacities": surfel.opacity,
        "sh_dc": surfel.f_dc,
        "sh_rest": surfel.sh_rest,
        "mask_logits": surfel.mask_logits,
    }
    for name, expected in expected_shapes.items():
        value = np.asarray(values[name])
        if value.shape != expected:
            raise ValueError(f"aura_2dgs_{name}_shape_invalid")
        if name == "opacities":
            if np.isnan(value).any() or np.isneginf(value).any():
                raise ValueError("aura_2dgs_nonfinite_input")
        elif not np.isfinite(value).all():
            raise ValueError("aura_2dgs_nonfinite_input")

    positions = np.ascontiguousarray(surfel.xyz, dtype=np.float32)
    planar_scales = np.exp(np.asarray(surfel.scales, dtype=np.float32)).astype(np.float32)
    if not np.isfinite(planar_scales).all() or (planar_scales <= 0).any():
        raise ValueError("aura_2dgs_activated_scale_invalid")
    # GaussianSurflet is planar in local XY. Z is a structural API component,
    # not a learned thickness or an invented third ellipsoid scale.
    #
    # It was authored as 1.0 on the reasoning that a structural component is
    # "unused", which is multiplicative-identity thinking applied to a value
    # that is an extent in metres, where the neutral value is zero.  With a
    # median learned extent of 0.8 mm that made every surfel a one-metre
    # needle, 1237x thicker than wide, and 414k of them at mean opacity 0.90
    # put 47 m^3 of opaque geometry inside a 117 m^3 room with the camera in
    # it.  Every frame rendered came back at max 1 of 255.
    #
    # Flat means proportional to the surfel, not a constant: a fixed epsilon
    # would be thicker than wide for the smallest surfels here, which is the
    # same error at a different magnitude.
    structural_z = (
        planar_scales.min(axis=1, keepdims=True) * STRUCTURAL_Z_SCALE_FRACTION
    ).astype(np.float32)
    scales = np.concatenate([planar_scales, structural_z], axis=1)

    raw_quats = np.asarray(surfel.quats, dtype=np.float64)
    norms = np.linalg.norm(raw_quats, axis=1, keepdims=True)
    if (norms <= np.finfo(np.float32).eps).any():
        raise ValueError("aura_2dgs_zero_quaternion")
    orientations = (raw_quats / norms).astype(np.float32)
    opacities = _sigmoid(np.asarray(surfel.opacity, dtype=np.float32)).astype(np.float32)

    rest = np.asarray(surfel.sh_rest, dtype=np.float32).reshape(surfel.count, 3, 15)
    rest = rest.transpose(0, 2, 1)
    dc = np.asarray(surfel.f_dc, dtype=np.float32).reshape(surfel.count, 1, 3)
    sh_coefficients = np.concatenate([dc, rest], axis=1).reshape(-1, 3)
    extent = np.stack([positions.min(axis=0), positions.max(axis=0)]).astype(np.float32)
    return {
        "count": surfel.count,
        "positions": positions,
        "scales": np.ascontiguousarray(scales),
        "orientations": np.ascontiguousarray(orientations),
        "opacities": np.ascontiguousarray(opacities),
        "sh_coefficients": np.ascontiguousarray(sh_coefficients, dtype=np.float32),
        "sh_degree": 3,
        "extent": extent,
        "mask_logits": np.ascontiguousarray(surfel.mask_logits, dtype=np.float32),
        "structural_z_scale_fraction": STRUCTURAL_Z_SCALE_FRACTION,
        "structural_z_scale_max_m": float(structural_z.max()),
        "structural_z_scale_median_m": float(np.median(structural_z)),
        "positive_infinite_opacity_logit_count": int(np.isposinf(surfel.opacity).sum()),
        # Reported, never clamped: these are the sealed learned coefficients,
        # and silently rescaling them would change the appearance the capture
        # actually recorded.  A reader can see how much of the field renders
        # outside displayable range and decide, rather than be told nothing.
        "sh_dc_out_of_display_range_fraction": float(
            1.0
            - (
                (0.5 + SH_C0 * dc.reshape(-1, 3) >= 0.0)
                & (0.5 + SH_C0 * dc.reshape(-1, 3) <= 1.0)
            )
            .all(axis=1)
            .mean()
        ),
        "sh_dc_radiance_max": float((0.5 + SH_C0 * dc).max()),
    }


def write_gaussian_surflet_particlefield_usd(
    source: str | Path | GaussianSurfelData,
    out_path: str | Path,
    *,
    prim_path: str = "/World/AuraAppearance/GaussianSurflets",
    expected_source_sha256: str | None = None,
    receipt_path: str | Path | None = None,
) -> dict:
    """Author Aura 2DGS as ``ParticleField`` plus Gaussian-surflet APIs.

    This never mutates the sealed PLY and intentionally does not author the
    ellipsoidal ``ParticleField3DGaussianSplat`` schema.
    """

    out_path = Path(out_path)
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol, Vt
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "blocked",
            "blockers": ["usd_core_gaussian_surflet_schema_unavailable"],
            "error": repr(exc),
        }
    source_path = None if isinstance(source, GaussianSurfelData) else Path(source)
    source_sha256 = None
    if source_path is not None:
        if not source_path.is_file():
            return {"status": "blocked", "blockers": ["aura_2dgs_source_missing"]}
        source_sha256 = f"sha256:{sha256_file(source_path)}"
        if expected_source_sha256 is None:
            return {
                "status": "blocked",
                "blockers": ["aura_2dgs_expected_source_sha256_missing"],
                "observed_source_sha256": source_sha256,
            }
        if source_sha256 != expected_source_sha256:
            return {
                "status": "blocked",
                "blockers": ["aura_2dgs_source_sha256_mismatch"],
                "expected_source_sha256": expected_source_sha256,
                "observed_source_sha256": source_sha256,
            }
    surfel = source if isinstance(source, GaussianSurfelData) else read_aura_2dgs_surfel_ply(source_path)
    arrays = build_gaussian_surflet_arrays(surfel)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    # Without a default prim a reference resolves to nothing.  Arena brings this
    # asset in with Object(usd_path=...), i.e. a USD reference, and a live run
    # added it to the scene and produced frames byte-comparable to a run with no
    # appearance at all -- max difference 12 of 255.  Nothing composed, so there
    # was nothing to render, which is also why authoring gaussian accumulation
    # settings changed nothing.  Both assets that do compose in this scene carry
    # one: sage_task_collision has /Root, the approved can has /canned_beverage.
    # The standalone OVRTX worker never noticed because it opens the file as a
    # stage rather than referencing it.
    stage.SetDefaultPrim(world.GetPrim())
    field = UsdVol.ParticleField.Define(stage, prim_path)
    prim = field.GetPrim()
    api_classes = (
        UsdVol.ParticleFieldPositionAttributeAPI,
        UsdVol.ParticleFieldOrientationAttributeAPI,
        UsdVol.ParticleFieldScaleAttributeAPI,
        UsdVol.ParticleFieldOpacityAttributeAPI,
        UsdVol.ParticleFieldKernelGaussianSurfletAPI,
        UsdVol.ParticleFieldSphericalHarmonicsAttributeAPI,
    )
    if not prim or not prim.IsValid() or not all(api.CanApply(prim) for api in api_classes):
        return {
            "status": "blocked",
            "blockers": ["usd_core_gaussian_surflet_schema_unavailable"],
        }
    position_api, orientation_api, scale_api, opacity_api, kernel_api, sh_api = (
        api.Apply(prim) for api in api_classes
    )
    if not all((position_api, orientation_api, scale_api, opacity_api, kernel_api, sh_api)):
        return {
            "status": "blocked",
            "blockers": ["usd_core_gaussian_surflet_api_application_failed"],
        }

    def vec3f(value: np.ndarray):
        return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(value, dtype=np.float32))

    position_api.CreatePositionsAttr().Set(vec3f(arrays["positions"]))
    scale_api.CreateScalesAttr().Set(vec3f(arrays["scales"]))
    opacity_api.CreateOpacitiesAttr().Set(
        Vt.FloatArray.FromNumpy(arrays["opacities"])
    )
    sh_api.CreateRadianceSphericalHarmonicsCoefficientsAttr().Set(
        vec3f(arrays["sh_coefficients"])
    )
    sh_api.CreateRadianceSphericalHarmonicsDegreeAttr().Set(arrays["sh_degree"])
    field.CreateExtentAttr().Set(vec3f(arrays["extent"]))
    quaternions = arrays["orientations"]
    orientation_api.CreateOrientationsAttr().Set(
        Vt.QuatfArray(
            [
                Gf.Quatf(float(w), float(x), float(y), float(z))
                for w, x, y, z in quaternions
            ]
        )
    )

    # Bind the same ParticleField emissive material used by Isaac Lab's
    # official Gaussian camera test asset. A generic ParticleField does not
    # inherit the concrete ParticleField3DGaussianSplat material contract, so
    # leaving this implicit makes SH radiance interpretation renderer-dependent.
    material_path = "/World/AuraAppearance/Looks/ParticleFieldEmissive"
    shader_path = f"{material_path}/Shader"
    material = stage.DefinePrim(material_path, "Material")
    shader = stage.DefinePrim(shader_path, "Shader")
    shader.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set(
        "sourceAsset"
    )
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        "ParticleFieldEmissive.mdl"
    )
    shader.CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
    ).Set("ParticleFieldEmissive")
    shader.CreateAttribute(
        "inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    shader.CreateAttribute(
        "inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    shader.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token, custom=True)
    for output_name in ("mdl:displacement", "mdl:surface", "mdl:volume"):
        material.CreateAttribute(f"outputs:{output_name}", Sdf.ValueTypeNames.Token).AddConnection(
            shader.GetPath().AppendProperty("outputs:out")
        )
    prim.CreateRelationship("material:binding").SetTargets([material.GetPath()])
    stage.GetRootLayer().Save()
    if source_path is not None and f"sha256:{sha256_file(source_path)}" != source_sha256:
        return {"status": "blocked", "blockers": ["aura_2dgs_sealed_source_mutated"]}
    result = {
        "schema_version": GAUSSIAN_SURFLET_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "output": str(out_path),
        "output_bytes": out_path.stat().st_size,
        "schema": GAUSSIAN_SURFLET_SCHEMA,
        "prim_path": prim_path,
        "default_prim": "/World",
        "surfel_count": arrays["count"],
        "sh_degree": arrays["sh_degree"],
        "source_frame": "right_handed_z_up_meters_identity_to_admitted_world",
        "source_sha256": source_sha256,
        "output_sha256": f"sha256:{sha256_file(out_path)}",
        "learned_scale_components": 2,
        "structural_z_scale_fraction": arrays["structural_z_scale_fraction"],
        "structural_z_scale_median_m": arrays["structural_z_scale_median_m"],
        "structural_z_scale_max_m": arrays["structural_z_scale_max_m"],
        "sh_dc_out_of_display_range_fraction": arrays[
            "sh_dc_out_of_display_range_fraction"
        ],
        "sh_dc_radiance_max": arrays["sh_dc_radiance_max"],
        "positive_infinite_opacity_logit_count": arrays[
            "positive_infinite_opacity_logit_count"
        ],
        "material": {
            "path": material_path,
            "shader": "ParticleFieldEmissive.mdl",
            "sub_identifier": "ParticleFieldEmissive",
            "apply_inverse_tonemap": False,
            "apply_srgb_linear": False,
            "basis": "official_isaac_lab_gaussian_camera_test_asset",
        },
        "sealed_source_mutated": False,
        "proof_boundary": "OpenUSD Gaussian-surflet authoring only; live OVRTX rendering remains required.",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if receipt_path is not None:
        write_json(Path(receipt_path), result)
    return result


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
    effective_sh_rest = sh_rest if sh_rest is not None else splat.sh_rest
    arr = build_particlefield_arrays(splat, sh_rest=effective_sh_rest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, "z" if up_axis.upper() == "Z" else "y"))
    # Splat centers/scales are METERS (straight from the 3DGS PLY). Without this,
    # the stage defaults to Kit's 0.01 (centimeters) and unit-aware consumers
    # mis-correct anything referenced in from meter-authored assets (a G1 robot
    # referenced into a cm-declared stage renders at 1/100 scale).
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
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
