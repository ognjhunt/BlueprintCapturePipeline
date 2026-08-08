"""Encode Aura's ghost-removed 2DGS as a NuRec volume Isaac can render.

Removing the ghost is what pushed this appearance off NuRec: AuraFusion360
emits a 2DGS PLY, NuRec is what Isaac renders natively, and nothing could
author the format back.  So the scene had to choose between the inpainting and
the renderer.  This removes the choice.

Two decisions carry the risk here.

**The config is a template, not an invention.**  A shipped NuRec volume is
adopted wholesale and only the fields that must differ are overridden, so the
projection, culling and render settings stay exactly what NVIDIA's own
reconstruction wrote.  Authoring those from scratch would be guessing at a
renderer contract we cannot test locally.

**The third scale is flat in log space.**  NuRec stores scales pre-activation,
so "flat" is a large negative log, not a small metre value -- and the last time
a structural third component was authored as a linear ``1.0`` on the reasoning
that it was unused, it became a one-metre thickness on sub-millimetre surfels
and buried the camera in opaque needles.  Doing this in the wrong space is the
same mistake with a different sign.

Nothing here renders or admits a volume.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .nurec_volume_codec import (
    GAUSSIAN_KEY_PREFIX,
    NuRecCodecError,
    build_state_dict,
    decode_nurec_bytes,
    gaussian_arrays,
)

AURA_NUREC_SCHEMA_VERSION = "aura_nurec_volume.v1"

# The structural third extent, as a fraction of the smaller learned planar
# extent.  Applied in log space, where NuRec stores scales.
STRUCTURAL_Z_SCALE_FRACTION = 0.01
STRUCTURAL_Z_LOG_OFFSET = math.log(STRUCTURAL_Z_SCALE_FRACTION)

# float16 saturates at 65504, and Aura's PLY carries genuine +inf opacity
# logits.  sigmoid is already 1.0 to float precision well before this, so
# clamping preserves the decoded opacity exactly while keeping the array
# finite -- an infinity would make the buffer unreadable rather than opaque.
FINITE_LOGIT_CLAMP = 30.0

BLOCKER_TEMPLATE_NOT_GAUSSIAN = "aura_nurec_template_not_a_gaussian_volume"
BLOCKER_SH_WIDTH = "aura_nurec_sh_rest_width_mismatch"


def _template_document(template: bytes | Mapping[str, Any]) -> dict[str, Any]:
    document = (
        decode_nurec_bytes(template) if isinstance(template, (bytes, bytearray))
        else dict(template)
    )
    config = document.get("config") or {}
    gaussians = ((config.get("layers") or {}).get("gaussians")) or {}
    if not gaussians or "particle" not in gaussians:
        raise NuRecCodecError([BLOCKER_TEMPLATE_NOT_GAUSSIAN])
    return document


def build_aura_nurec_document(
    surfel: Any,
    *,
    template: bytes | Mapping[str, Any],
    planar: bool = True,
) -> dict[str, Any]:
    """Lay Aura's learned 2DGS parameters into a NuRec container document.

    ``surfel`` is a :class:`~blueprint_pipeline.gaussian_splat_decode.GaussianSurfelData`
    holding the sealed PLY values.  They are written through unactivated,
    because NuRec applies ``exp`` and ``sigmoid`` itself and every activation
    applied here is a chance to be wrong about units.
    """

    document = _template_document(template)
    config = {k: v for k, v in (document.get("config") or {}).items()}
    layers = {k: v for k, v in (config.get("layers") or {}).items()}
    gaussians = {k: v for k, v in (layers.get("gaussians") or {}).items()}
    particle = {k: v for k, v in (gaussians.get("particle") or {}).items()}

    # The one substantive override.  Aura's field is planar; the template's is
    # not, and a 2D gaussian rendered by a volumetric kernel is not the same
    # surface.
    particle["density_kernel_planar"] = bool(planar)
    gaussians["particle"] = particle
    layers["gaussians"] = gaussians
    config["layers"] = layers

    count = int(surfel.count)
    positions = np.asarray(surfel.xyz, dtype=np.float32).reshape(count, 3)
    rotations = np.asarray(surfel.quats, dtype=np.float32).reshape(count, 4)
    planar_log = np.asarray(surfel.scales, dtype=np.float32).reshape(count, 2)
    albedo = np.asarray(surfel.f_dc, dtype=np.float32).reshape(count, 3)
    specular = np.asarray(surfel.sh_rest, dtype=np.float32).reshape(count, -1)
    if specular.shape[1] != 45:
        raise NuRecCodecError([f"{BLOCKER_SH_WIDTH}:{specular.shape[1]}"])

    # Flat, in the space the value is stored in.  log(min_planar * fraction)
    # is min_planar_log + log(fraction), so this stays proportional to each
    # surfel rather than a constant that would be thicker than the smallest
    # ones are wide.
    structural_log = planar_log.min(axis=1, keepdims=True) + STRUCTURAL_Z_LOG_OFFSET
    scales = np.concatenate([planar_log, structural_log], axis=1)

    raw_density = np.asarray(surfel.opacity, dtype=np.float32).reshape(count, 1)
    infinite_logits = int(np.isposinf(raw_density).sum() + np.isneginf(raw_density).sum())
    densities = np.clip(
        np.nan_to_num(raw_density, nan=0.0, posinf=FINITE_LOGIT_CLAMP, neginf=-FINITE_LOGIT_CLAMP),
        -FINITE_LOGIT_CLAMP,
        FINITE_LOGIT_CLAMP,
    )

    precision = int(gaussians.get("precision") or 16)
    state = build_state_dict(
        {
            "positions": positions,
            "rotations": rotations,
            "scales": scales,
            "densities": densities,
            "features_albedo": albedo,
            "features_specular": specular,
        },
        precision=precision,
    )
    # Carried from the template verbatim: these are container bookkeeping whose
    # meaning is the renderer's, not ours, and a guessed value is worse than a
    # copied one.
    template_state = document.get("state_dict") or {}
    for key in ("._extra_state",):
        if key in template_state:
            state[key] = template_state[key]
    empty_signal = f"{GAUSSIAN_KEY_PREFIX}extra_signal"
    state[empty_signal] = b""
    state[f"{empty_signal}.shape"] = [count, 0]
    active = f"{GAUSSIAN_KEY_PREFIX}n_active_features"
    if active in template_state:
        state[active] = template_state[active]
        state[f"{active}.shape"] = template_state.get(f"{active}.shape", [])

    built = {
        "version": document.get("version"),
        "model": document.get("model"),
        "config": config,
        "state_dict": state,
    }
    built["_blueprint_authoring"] = {
        "schema_version": AURA_NUREC_SCHEMA_VERSION,
        "gaussian_count": count,
        "precision": precision,
        "density_kernel_planar": bool(planar),
        "structural_z_scale_fraction": STRUCTURAL_Z_SCALE_FRACTION,
        "structural_z_log_offset": STRUCTURAL_Z_LOG_OFFSET,
        "infinite_opacity_logits_clamped": infinite_logits,
        "finite_logit_clamp": FINITE_LOGIT_CLAMP,
        "values_written": "pre_activation_learned_parameters",
    }
    return built


def describe_authored_volume(document: Mapping[str, Any]) -> dict[str, Any]:
    """Report what was authored, from the document itself rather than intent."""

    arrays = gaussian_arrays(document)
    scales = arrays["scales"].astype(np.float32)
    activated = np.exp(scales)
    planar = activated[:, :2]
    authoring = dict(document.get("_blueprint_authoring") or {})
    authoring.update(
        {
            "activated_planar_median_m": float(np.median(planar)),
            "activated_structural_median_m": float(np.median(activated[:, 2])),
            # The check the linear-space version of this failed: flatter than
            # the surfel is wide, for every surfel.
            "structural_is_flatter_than_planar": bool(
                (activated[:, 2] < planar.min(axis=1)).all()
            ),
        }
    )
    return authoring


__all__ = [
    "AURA_NUREC_SCHEMA_VERSION",
    "FINITE_LOGIT_CLAMP",
    "STRUCTURAL_Z_LOG_OFFSET",
    "STRUCTURAL_Z_SCALE_FRACTION",
    "build_aura_nurec_document",
    "describe_authored_volume",
]
