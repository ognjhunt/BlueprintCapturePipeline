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

import hashlib
from pathlib import Path
from typing import Any, Sequence
import zipfile

import numpy as np

from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_field_quality import measure_gaussian_field_quality
from .gaussian_splat_decode import (
    GaussianSurfelData,
    SplatData,
    read_aura_2dgs_surfel_ply,
    read_standard_3dgs_ply,
)
from .nurec_volume_codec import (
    NuRecCodecError,
    decode_nurec_bytes,
    describe_volume,
    gaussian_arrays,
)

PARTICLEFIELD_SCHEMA = "ParticleField3DGaussianSplat"
PARTICLEFIELD_RECEIPT_SCHEMA_VERSION = "particlefield_3dgs_authoring_receipt.v1"
PARTICLEFIELD_REFERENCE_CONVERTERS = {
    "openusd_py3dgs_ply_to_usd": (
        "https://github.com/PixarAnimationStudios/OpenUSD/blob/"
        "47154dc7b5e28df623745495a7a508b69535ba24/"
        "extras/imaging/examples/hdParticleField/py3dgsPlyToUsd.py"
    ),
    "nvidia_usd_convert_gsplat": (
        "https://github.com/NVIDIA-Omniverse/usd-convert-gsplat/blob/"
        "621017ebf78394488260c70ec4eadd70ff621131/"
        "source/python/usd_convert_gsplat/usd_writer.py"
    ),
}
GAUSSIAN_SURFLET_SCHEMA = "ParticleField+ParticleFieldKernelGaussianSurfletAPI"
GAUSSIAN_SURFLET_RECEIPT_SCHEMA_VERSION = "aura_ovrtx_particlefield_receipt.v1"

_GAUSSIAN_SURFLET_SCHEMA_MEMBERS = (
    "ParticleField",
    "ParticleFieldPositionAttributeAPI",
    "ParticleFieldOrientationAttributeAPI",
    "ParticleFieldScaleAttributeAPI",
    "ParticleFieldOpacityAttributeAPI",
    "ParticleFieldKernelGaussianSurfletAPI",
    "ParticleFieldSphericalHarmonicsAttributeAPI",
)


# Zeroth-order spherical-harmonic basis constant, for reading a DC coefficient
# back as a display colour: colour = 0.5 + C0 * dc.
SH_C0 = 0.28209479177387814

# The structural Z extent, as a fraction of the smaller learned planar extent.
# Flat has to be relative: a constant epsilon would be thicker than wide for the
# smallest surfels in this field, which is the bug it replaces at a new scale.
STRUCTURAL_Z_SCALE_FRACTION = 0.01
NUREC_VOLUME_MARKER = "omni:nurec:isNuRecVolume"


def gaussian_surflet_schema_available(usd_vol: object) -> bool:
    """Whether this OpenUSD build carries every Isaac ParticleField binding."""

    return all(hasattr(usd_vol, name) for name in _GAUSSIAN_SURFLET_SCHEMA_MEMBERS)


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
    arrays = (xyz, splat.scales, splat.quats, splat.f_dc)
    if any(not np.isfinite(np.asarray(value)).all() for value in arrays):
        raise ValueError("particlefield_nonfinite_input")
    raw_opacity = np.asarray(splat.opacity, dtype=np.float32)
    # Standard 3DGS serializers commonly represent exact alpha endpoints as
    # inverse-sigmoid logits: +inf is fully opaque and -inf fully transparent.
    # The activation below handles both exactly.  NaN has no appearance
    # meaning and remains forbidden, as do non-finite geometry/material fields.
    if raw_opacity.shape != (splat.count,) or np.isnan(raw_opacity).any():
        raise ValueError("particlefield_nonfinite_input")
    with np.errstate(over="ignore", invalid="ignore"):
        scales = np.exp(np.asarray(splat.scales, dtype=np.float32)).astype(np.float32)
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        raise ValueError("particlefield_activated_scale_invalid")

    q = np.asarray(splat.quats, dtype=np.float64)  # (N,4) = (w, x, y, z) INRIA order
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if (norm <= np.finfo(np.float32).eps).any():
        raise ValueError("particlefield_zero_quaternion")
    quats = (q / norm).astype(np.float32)

    opac = _sigmoid(raw_opacity).astype(np.float32)

    n = xyz.shape[0]
    dc = np.asarray(splat.f_dc, dtype=np.float32).reshape(n, 1, 3)  # coeff 0 (RGB)
    if sh_rest is not None and np.asarray(sh_rest).size:
        rest = np.asarray(sh_rest, dtype=np.float32)
        if rest.ndim != 2 or rest.shape[0] != n or rest.shape[1] % 3 or not np.isfinite(rest).all():
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
    # Both pinned reference converters author displayColor as the deterministic
    # per-vertex DC-colour fallback alongside spherical-harmonic radiance.
    display_colors = np.clip(
        0.5 + SH_C0 * np.asarray(splat.f_dc, dtype=np.float32), 0.0, 1.0
    ).astype(np.float32)
    return {
        "count": int(n),
        "positions": xyz,
        "scales": scales,
        "orientations": quats,  # (w, x, y, z)
        "opacities": opac,
        "sh_coefficients": sh,
        "sh_degree": degree,
        "sh_element_size": int((degree + 1) ** 2),
        "display_colors": display_colors,
        "extent": extent,
        "positive_infinite_opacity_logit_count": int(np.isposinf(raw_opacity).sum()),
        "negative_infinite_opacity_logit_count": int(np.isneginf(raw_opacity).sum()),
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
    structural_z = (planar_scales.min(axis=1, keepdims=True) * STRUCTURAL_Z_SCALE_FRACTION).astype(
        np.float32
    )
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
            - ((0.5 + SH_C0 * dc.reshape(-1, 3) >= 0.0) & (0.5 + SH_C0 * dc.reshape(-1, 3) <= 1.0))
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
    surfel = (
        source if isinstance(source, GaussianSurfelData) else read_aura_2dgs_surfel_ply(source_path)
    )
    arrays = build_gaussian_surflet_arrays(surfel)
    if not gaussian_surflet_schema_available(UsdVol):
        return {
            "status": "blocked",
            "blockers": ["usd_core_gaussian_surflet_schema_unavailable"],
        }
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
    # one: the collision scene and task asset can have different default prims.
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
    opacity_api.CreateOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(arrays["opacities"]))
    sh_api.CreateRadianceSphericalHarmonicsCoefficientsAttr().Set(vec3f(arrays["sh_coefficients"]))
    sh_api.CreateRadianceSphericalHarmonicsDegreeAttr().Set(arrays["sh_degree"])
    field.CreateExtentAttr().Set(vec3f(arrays["extent"]))
    quaternions = arrays["orientations"]
    orientation_api.CreateOrientationsAttr().Set(
        Vt.QuatfArray(
            [Gf.Quatf(float(w), float(x), float(y), float(z)) for w, x, y, z in quaternions]
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
    shader.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        "ParticleFieldEmissive.mdl"
    )
    shader.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set(
        "ParticleFieldEmissive"
    )
    shader.CreateAttribute(
        "inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    shader.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool, custom=True).Set(
        False
    )
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
        "sh_dc_out_of_display_range_fraction": arrays["sh_dc_out_of_display_range_fraction"],
        "sh_dc_radiance_max": arrays["sh_dc_radiance_max"],
        "positive_infinite_opacity_logit_count": arrays["positive_infinite_opacity_logit_count"],
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
    expected_source_sha256: str | None = None,
    receipt_path: str | Path | None = None,
    layer_transform_row_major: Sequence[Sequence[float]] | None = None,
) -> dict:
    """Author a ParticleField3DGaussianSplat USD from a standard 3DGS PLY (or SplatData).

    Returns a fail-closed status dict; ``status == 'completed'`` means ``out_path`` is a
    valid USD that Isaac Sim 6.0 RTX can render.
    """
    out_path = Path(out_path)
    try:
        from pxr import Usd, UsdGeom, UsdVol, Gf, Sdf, Vt  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - pxr/usd-core not installed here
        return {
            "status": "blocked",
            "blockers": ["usd_core_unavailable"],
            "remediation": "pip install usd-core (authoring runs where pxr exists: locally or the Isaac pod)",
            "error": repr(exc),
        }
    source_path = None if isinstance(source, SplatData) else Path(source)
    source_sha256 = None
    if source_path is not None:
        if not source_path.is_file():
            return {
                "status": "blocked",
                "blockers": ["particlefield_3dgs_source_missing"],
            }
        source_sha256 = f"sha256:{sha256_file(source_path)}"
        if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
            return {
                "status": "blocked",
                "blockers": ["particlefield_3dgs_source_sha256_mismatch"],
                "expected_source_sha256": expected_source_sha256,
                "observed_source_sha256": source_sha256,
            }
    splat = source if isinstance(source, SplatData) else read_standard_3dgs_ply(source_path)
    effective_sh_rest = sh_rest if sh_rest is not None else splat.sh_rest
    arr = build_particlefield_arrays(splat, sh_rest=effective_sh_rest)
    field_quality = measure_gaussian_field_quality(
        positions=arr["positions"],
        activated_scales=arr["scales"],
        opacities=arr["opacities"],
    )
    if field_quality.get("status") != "qualified" or field_quality.get("blockers"):
        return {
            "status": "blocked",
            "blockers": ["particlefield_gaussian_field_quality_invalid"],
            "gaussian_field_quality": field_quality,
            "proof_boundary": (
                "No ParticleField authored because the exact learned field failed "
                "scene-relative geometry quality."
            ),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, "z" if up_axis.upper() == "Z" else "y"))
    # Splat centers/scales are METERS (straight from the 3DGS PLY). Without this,
    # the stage defaults to Kit's 0.01 (centimeters) and unit-aware consumers
    # mis-correct anything referenced in from meter-authored assets (a G1 robot
    # referenced into a cm-declared stage renders at 1/100 scale).
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    # Arena composes this file as a reference. A standalone stage without a
    # default prim can open successfully yet contribute no referenced scene.
    stage.SetDefaultPrim(world.GetPrim())
    typed_schema = getattr(UsdVol, "ParticleField3DGaussianSplat", None)
    field = typed_schema.Define(stage, prim_path) if typed_schema else None
    prim = field.GetPrim() if field else stage.DefinePrim(prim_path, PARTICLEFIELD_SCHEMA)
    if not prim or not prim.IsValid():
        return {
            "status": "blocked",
            "blockers": ["particlefield_schema_unavailable"],
            "remediation": "usd-core build lacks ParticleField3DGaussianSplat (need a recent UsdVol)",
        }

    transform = np.asarray(
        layer_transform_row_major if layer_transform_row_major is not None else np.eye(4),
        dtype=np.float64,
    )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return {
            "status": "blocked",
            "blockers": ["particlefield_layer_transform_invalid"],
        }
    if not np.allclose(transform, np.eye(4)):
        matrix = Gf.Matrix4d(*[float(value) for value in transform.reshape(-1)])
        UsdGeom.Xformable(prim).AddTransformOp().Set(matrix)

    def vec3f(a: np.ndarray):
        return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, dtype=np.float32))

    if field:
        field.CreatePositionsAttr(vec3f(arr["positions"]))
        field.CreateScalesAttr(vec3f(arr["scales"]))
        field.CreateOpacitiesAttr(
            Vt.FloatArray.FromNumpy(np.ascontiguousarray(arr["opacities"], dtype=np.float32))
        )
        sh_attr = field.CreateRadianceSphericalHarmonicsCoefficientsAttr(
            vec3f(arr["sh_coefficients"])
        )
        field.CreateRadianceSphericalHarmonicsDegreeAttr(int(arr["sh_degree"]))
    else:
        prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
            vec3f(arr["positions"])
        )
        prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set(vec3f(arr["scales"]))
        prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set(
            Vt.FloatArray.FromNumpy(np.ascontiguousarray(arr["opacities"], dtype=np.float32))
        )
        sh_attr = prim.CreateAttribute(
            "radiance:sphericalHarmonicsCoefficients",
            Sdf.ValueTypeNames.Float3Array,
        )
        sh_attr.Set(vec3f(arr["sh_coefficients"]))
        prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(
            int(arr["sh_degree"])
        )
    UsdGeom.Boundable(prim).CreateExtentAttr(vec3f(arr["extent"]))

    # The coefficient array is flattened as (degree + 1)^2 float3 values per
    # Gaussian. Without these load-bearing primvar opinions, RTX interprets it
    # as one constant value even though every coefficient byte is present.
    sh_primvar = UsdGeom.Primvar(sh_attr)
    sh_primvar.SetElementSize(arr["sh_element_size"])
    sh_primvar.SetInterpolation(UsdGeom.Tokens.vertex)

    display_color = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    )
    display_color.Set(vec3f(arr["display_colors"]))

    # Match Isaac Lab's own known-working Gaussian camera fixture.  The
    # ParticleField schema carries geometry/radiance attributes, but the RTX
    # camera path still needs the emissive MDL bound to turn those attributes
    # into renderable radiance.  Leave its inputs at MDL defaults: the Isaac
    # Lab PPISP test overrides them only because that test installs a separate
    # ISP authority, which this normal LDR camera path does not.
    material_path = f"{prim.GetParent().GetPath()}/Looks/ParticleFieldEmissive"
    shader_path = f"{material_path}/Shader"
    material = stage.DefinePrim(material_path, "Material")
    shader = stage.DefinePrim(shader_path, "Shader")
    shader.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        "ParticleFieldEmissive.mdl"
    )
    shader.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set(
        "ParticleFieldEmissive"
    )
    shader.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token, custom=True)
    for output_name in ("mdl:displacement", "mdl:surface", "mdl:volume"):
        material.CreateAttribute(f"outputs:{output_name}", Sdf.ValueTypeNames.Token).AddConnection(
            shader.GetPath().AppendProperty("outputs:out")
        )
    prim.CreateRelationship("material:binding").SetTargets([material.GetPath()])
    prim.CreateAttribute("projectionModeHint", Sdf.ValueTypeNames.Token).Set("perspective")
    prim.CreateAttribute("sortingModeHint", Sdf.ValueTypeNames.Token).Set(sorting_mode)

    # quaternions: try numpy fast path, fall back to per-element Gf.Quatf (w, x, y, z)
    q = arr["orientations"]
    quat_attr = (
        field.CreateOrientationsAttr()
        if field
        else prim.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray)
    )
    try:
        quat_attr.Set(Vt.QuatfArray.FromNumpy(np.ascontiguousarray(q, dtype=np.float32)))
    except Exception:  # noqa: BLE001
        quat_attr.Set(
            Vt.QuatfArray([Gf.Quatf(float(w), float(x), float(y), float(z)) for w, x, y, z in q])
        )

    stage.GetRootLayer().Save()
    if source_path is not None and f"sha256:{sha256_file(source_path)}" != source_sha256:
        return {
            "status": "blocked",
            "blockers": ["particlefield_3dgs_sealed_source_mutated"],
        }
    result = {
        "schema_version": PARTICLEFIELD_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "output": str(out_path),
        "output_bytes": out_path.stat().st_size if out_path.is_file() else 0,
        "output_sha256": f"sha256:{sha256_file(out_path)}",
        "schema": PARTICLEFIELD_SCHEMA,
        "splat_count": arr["count"],
        "sh_degree": arr["sh_degree"],
        "sh_primvar_element_size": arr["sh_element_size"],
        "sh_primvar_interpolation": "vertex",
        "display_color_fallback_authored": True,
        "particlefield_emissive_material_binding_authored": True,
        "particlefield_emissive_material_inputs": "mdl_defaults",
        "particlefield_emissive_material_path": material_path,
        "reference_converters": PARTICLEFIELD_REFERENCE_CONVERTERS,
        "prim_path": prim_path,
        "default_prim": "/World",
        "source_sha256": source_sha256,
        "source_kind": ("in_memory_splat_data" if source_path is None else "standard_3dgs_ply"),
        "positive_infinite_opacity_logit_count": arr["positive_infinite_opacity_logit_count"],
        "negative_infinite_opacity_logit_count": arr["negative_infinite_opacity_logit_count"],
        "gaussian_field_quality": field_quality,
        "sealed_source_mutated": False,
        "layer_transform_row_major": transform.tolist(),
        "proof_boundary": "ParticleField USD authoring only; Isaac RTX render is the GPU step.",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if receipt_path is not None:
        write_json(Path(receipt_path), result)
    return result


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _nurec_payload_and_transform(
    source_path: Path,
) -> tuple[dict[str, Any], list[list[float]], str, bytes]:
    """Read one self-contained NuRec volume without changing its learned arrays."""

    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:  # noqa: BLE001
        raise NuRecCodecError(["nurec_particlefield_usd_runtime_unavailable"]) from exc
    try:
        stage = Usd.Stage.Open(str(source_path))
    except Exception as exc:  # noqa: BLE001
        raise NuRecCodecError(["nurec_particlefield_source_unreadable"]) from exc
    if stage is None or not stage.GetDefaultPrim():
        raise NuRecCodecError(["nurec_particlefield_source_unreadable"])
    volumes = [
        prim for prim in stage.Traverse() if bool(prim.GetAttribute(NUREC_VOLUME_MARKER).Get())
    ]
    if len(volumes) != 1:
        raise NuRecCodecError(["nurec_particlefield_volume_not_exact"])
    volume = volumes[0]
    payloads = {
        str(child.GetAttribute("filePath").Get().path)
        for child in volume.GetChildren()
        if child.GetAttribute("filePath").IsValid()
        and child.GetAttribute("filePath").Get() is not None
    }
    if len(payloads) != 1:
        raise NuRecCodecError(["nurec_particlefield_payload_not_exact"])
    payload_name = Path(payloads.pop().replace("\\", "/")).name
    try:
        with zipfile.ZipFile(source_path) as archive:
            members = [
                info
                for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).name == payload_name
            ]
            if len(members) != 1:
                raise NuRecCodecError(["nurec_particlefield_payload_not_exact"])
            payload = archive.read(members[0])
    except (OSError, zipfile.BadZipFile) as exc:
        raise NuRecCodecError(["nurec_particlefield_source_unreadable"]) from exc
    document = decode_nurec_bytes(payload)
    matrix = UsdGeom.Xformable(volume).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transform = [[float(matrix[row][column]) for column in range(4)] for row in range(4)]
    return document, transform, str(volume.GetPath()), payload


def write_particlefield_usd_from_nurec(
    source_path: str | Path,
    out_path: str | Path,
    *,
    expected_source_sha256: str,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert one sealed NuRec USDZ to the Isaac 6 native ParticleField schema.

    NuRec remains the configured-scene appearance truth. This produces a derived,
    digest-bound runtime representation from the same learned Gaussian arrays.
    """

    source = Path(source_path)
    observed_source_sha256 = f"sha256:{sha256_file(source)}" if source.is_file() else None
    if (
        source.is_symlink()
        or not source.is_file()
        or observed_source_sha256 != expected_source_sha256
    ):
        return {
            "status": "blocked",
            "blockers": ["nurec_particlefield_source_identity_mismatch"],
            "expected_source_sha256": expected_source_sha256,
            "observed_source_sha256": observed_source_sha256,
        }
    try:
        document, transform, source_prim_path, payload = _nurec_payload_and_transform(source)
        description = describe_volume(document)
        arrays = gaussian_arrays(document)
    except (NuRecCodecError, KeyError, ValueError) as exc:
        return {
            "status": "blocked",
            "blockers": ["nurec_particlefield_source_invalid"],
            "detail_codes": list(getattr(exc, "errors", (str(exc),))),
        }
    if (
        description.get("density_activation") != "sigmoid"
        or description.get("scale_activation") != "exp"
        or description.get("rotation_activation") != "normalize"
        or description.get("radiance_sph_degree") != 3
        or description.get("density_kernel_planar") is not False
    ):
        return {
            "status": "blocked",
            "blockers": ["nurec_particlefield_activation_contract_unsupported"],
            "nurec_description": description,
        }
    count = int(arrays["positions"].shape[0])
    splat = SplatData(
        count=count,
        xyz=np.asarray(arrays["positions"], dtype=np.float32),
        opacity=np.asarray(arrays["densities"], dtype=np.float32).reshape(count),
        f_dc=np.asarray(arrays["features_albedo"], dtype=np.float32),
        scales=np.asarray(arrays["scales"], dtype=np.float32),
        quats=np.asarray(arrays["rotations"], dtype=np.float32),
        properties=(),
        sh_rest=np.asarray(arrays["features_specular"], dtype=np.float32),
    )
    result = write_particlefield_usd(
        splat,
        out_path,
        sh_rest=splat.sh_rest,
        layer_transform_row_major=transform,
    )
    if result.get("status") != "completed":
        return result
    result.update(
        source_sha256=observed_source_sha256,
        source_kind="nurec_usdz",
        source_nurec_payload_sha256=_sha256_bytes(payload),
        source_nurec_prim_path=source_prim_path,
        source_nurec_description=description,
        exact_learned_arrays_preserved=True,
        representation_conversion_only=True,
        proof_boundary=(
            "Deterministic NuRec-to-ParticleField representation conversion only; "
            "Isaac RTX render remains the GPU gate."
        ),
    )
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if receipt_path is not None:
        write_json(Path(receipt_path), result)
    return result
