"""Lay a standard 3DGS splat into a NuRec container document.

The sibling module :mod:`blueprint_pipeline.aura_nurec_volume` does this for
AuraFusion360's 2D surfels, which need a synthetic third scale and a planar
density kernel. A standard 3DGS splat is the case with nothing to invent: it
already carries three learned log-scales, and the template's own volumetric
kernel - the configuration the shipped InteriorGS package actually renders
with - is the correct one. So this builder's whole job is to move arrays
without editorializing: positions recentred for float16 (offset carried on
the volume, not baked into data), non-finite opacity logits clamped, and
everything else written through pre-activation exactly as the PLY stored it,
because NuRec applies ``exp`` and ``sigmoid`` itself.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from blueprint_pipeline.aura_nurec_volume import (
    FINITE_LOGIT_CLAMP,
    MORTON_BITS,
    _template_document,
    morton_order,
)
from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.nurec_volume_codec import (
    GAUSSIAN_KEY_PREFIX,
    NuRecCodecError,
    build_state_dict,
)


SPLAT_NUREC_SCHEMA_VERSION = "splat_nurec_authoring.v1"
SPECULAR_WIDTH = 45


def build_splat_nurec_document(
    splat: SplatData,
    *,
    template: bytes | Mapping[str, Any],
    z_order: bool = False,
    precision: int | None = None,
    recentre: bool = True,
) -> dict[str, Any]:
    """Lay standard 3DGS arrays into ``template`` without touching its kernel."""

    document = _template_document(template)
    config = {k: v for k, v in (document.get("config") or {}).items()}
    layers = {k: v for k, v in (config.get("layers") or {}).items()}
    gaussians = {k: v for k, v in (layers.get("gaussians") or {}).items()}
    # Deliberately no particle override: the template's density kernel is the
    # one the shipped package renders with, and this splat is volumetric like
    # the template's own field. The aura builder must flip it; we must not.
    config["layers"] = layers

    count = int(splat.count)
    if splat.sh_rest is None:
        raise NuRecCodecError(["splat_nurec_sh_rest_missing"])
    specular = np.asarray(splat.sh_rest, dtype=np.float32).reshape(count, -1)
    if specular.shape[1] != SPECULAR_WIDTH:
        raise NuRecCodecError(
            [f"splat_nurec_sh_width_unexpected:{specular.shape[1]}"]
        )

    positions = np.asarray(splat.xyz, dtype=np.float32).reshape(count, 3)
    rotations = np.asarray(splat.quats, dtype=np.float32).reshape(count, 4)
    scales = np.asarray(splat.scales, dtype=np.float32).reshape(count, 3)
    albedo = np.asarray(splat.f_dc, dtype=np.float32).reshape(count, 3)
    raw_density = np.asarray(splat.opacity, dtype=np.float32).reshape(count, 1)

    # float16 resolution is relative to magnitude; centring the field halves
    # the rounding error for free, with the offset carried as a translation
    # on the volume prim rather than baked into the learned data.
    centre = (
        ((positions.min(axis=0) + positions.max(axis=0)) / 2.0).astype(np.float32)
        if recentre
        else np.zeros(3, dtype=np.float32)
    )
    positions = positions - centre

    order = morton_order(positions) if z_order else np.arange(count)
    positions = positions[order]
    rotations = rotations[order]
    scales = scales[order]
    albedo = albedo[order]
    specular = specular[order]
    raw_density = raw_density[order]

    infinite_logits = int(
        np.isposinf(raw_density).sum() + np.isneginf(raw_density).sum()
    )
    densities = np.clip(
        np.nan_to_num(
            raw_density,
            nan=0.0,
            posinf=FINITE_LOGIT_CLAMP,
            neginf=-FINITE_LOGIT_CLAMP,
        ),
        -FINITE_LOGIT_CLAMP,
        FINITE_LOGIT_CLAMP,
    )

    resolved_precision = int(precision or gaussians.get("precision") or 16)
    gaussians["precision"] = resolved_precision
    layers["gaussians"] = gaussians
    state = build_state_dict(
        {
            "positions": positions,
            "rotations": rotations,
            "scales": scales,
            "densities": densities,
            "features_albedo": albedo,
            "features_specular": specular,
        },
        precision=resolved_precision,
    )
    # Container bookkeeping carried verbatim: its meaning is the renderer's.
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
        "schema_version": SPLAT_NUREC_SCHEMA_VERSION,
        "gaussian_count": count,
        "precision": resolved_precision,
        "density_kernel": "template_verbatim",
        "infinite_opacity_logits_clamped": infinite_logits,
        "finite_logit_clamp": FINITE_LOGIT_CLAMP,
        "values_written": "pre_activation_learned_parameters",
        "z_ordered": bool(z_order),
        "precision_source": (
            "explicit" if precision is not None else "template_default"
        ),
        "recentred": bool(recentre),
        # The translation the volume must re-apply, or the room renders in
        # the wrong place - correctly, sharply, metres from the arm.
        "centre_offset_m": [float(v) for v in centre],
        "z_order_bits": MORTON_BITS,
    }
    return built


__all__ = [
    "SPECULAR_WIDTH",
    "SPLAT_NUREC_SCHEMA_VERSION",
    "build_splat_nurec_document",
]
