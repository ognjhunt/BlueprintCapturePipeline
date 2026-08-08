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


MORTON_BITS = 10
"""Quantisation per axis for the Z-order key.  Ten bits fills a 32-bit key."""


def morton_order(positions: np.ndarray) -> np.ndarray:
    """Index order that sorts gaussians by Morton (Z-order) curve.

    Provided, but **off by default**, and the reason is worth recording.  The
    renderer config carries ``global_z_order: True``, which looked like a
    requirement that the payload arrive pre-sorted.  A check appeared to
    confirm it, reporting the shipped gaussians as perfectly Morton-monotone.

    That check was wrong: it evaluated ``np.diff(keys) >= 0`` on a **uint32**
    array, where subtraction wraps, so every difference is non-negative by
    construction and the answer is 1.000 whatever the data says.  Cast to
    int64 the shipped payload scores 0.500 -- arbitrary order.  The renderer
    orders internally, and sorting our payload would make it differ from the
    only reference known to render.

    Kept because the ordering question will come back, and a named function
    with this note attached is cheaper than rediscovering the trap.
    """

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise NuRecCodecError([f"aura_nurec_morton_positions_invalid:{positions.shape}"])
    span = positions.max(axis=0) - positions.min(axis=0)
    # A degenerate axis would divide by zero; a flat scene is still orderable.
    span = np.where(span > 0, span, 1.0)
    scale = (1 << MORTON_BITS) - 1
    quantised = np.clip(
        ((positions - positions.min(axis=0)) / span * scale).astype(np.uint64), 0, scale
    )

    def _spread(value: np.ndarray) -> np.ndarray:
        value = (value | (value << 16)) & 0x030000FF
        value = (value | (value << 8)) & 0x0300F00F
        value = (value | (value << 4)) & 0x030C30C3
        value = (value | (value << 2)) & 0x09249249
        return value

    key = (
        _spread(quantised[:, 0])
        | (_spread(quantised[:, 1]) << np.uint64(1))
        | (_spread(quantised[:, 2]) << np.uint64(2))
    )
    # Stable, so an already-ordered payload keeps its exact arrangement rather
    # than being permuted within ties.
    return np.argsort(key, kind="stable")


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
    z_order: bool = False,
    precision: int | None = None,
    recentre: bool = True,
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
    raw_opacity = np.asarray(surfel.opacity, dtype=np.float32).reshape(count, 1)

    # float16 resolution is relative to magnitude: the grid is coarse at 8.4m
    # and fine near zero.  Centring the field costs nothing and more than
    # halves the rounding error -- 1.15x the median surfel width down to
    # 0.53x -- with the offset carried as a translation on the volume rather
    # than baked into the data.  Free insurance even at float32, and the only
    # fix that cannot fail: it is arithmetic, not a format feature the
    # renderer might not implement.
    centre = (
        ((positions.min(axis=0) + positions.max(axis=0)) / 2.0).astype(np.float32)
        if recentre
        else np.zeros(3, dtype=np.float32)
    )
    positions = positions - centre

    # Off by default: the shipped payload is not Z-ordered, so sorting would
    # make ours differ from the only reference known to render.  When it is
    # enabled, every per-gaussian array is permuted by the same index, because
    # a positions/colour mismatch renders a plausible-looking scene made of
    # the wrong colours -- worse than one that fails outright.
    order = morton_order(positions) if z_order else np.arange(count)
    positions = positions[order]
    rotations = rotations[order]
    planar_log = planar_log[order]
    albedo = albedo[order]
    specular = specular[order]
    raw_opacity = raw_opacity[order]
    if specular.shape[1] != 45:
        raise NuRecCodecError([f"{BLOCKER_SH_WIDTH}:{specular.shape[1]}"])

    # Flat, in the space the value is stored in.  log(min_planar * fraction)
    # is min_planar_log + log(fraction), so this stays proportional to each
    # surfel rather than a constant that would be thicker than the smallest
    # ones are wide.
    structural_log = planar_log.min(axis=1, keepdims=True) + STRUCTURAL_Z_LOG_OFFSET
    scales = np.concatenate([planar_log, structural_log], axis=1)

    raw_density = raw_opacity
    infinite_logits = int(np.isposinf(raw_density).sum() + np.isneginf(raw_density).sum())
    densities = np.clip(
        np.nan_to_num(raw_density, nan=0.0, posinf=FINITE_LOGIT_CLAMP, neginf=-FINITE_LOGIT_CLAMP),
        -FINITE_LOGIT_CLAMP,
        FINITE_LOGIT_CLAMP,
    )

    # float16 costs this field more than it costs the template's.  Rounding
    # displaces an Aura surfel by 0.93mm at p95 against a median surfel width
    # of 0.81mm -- more than its own size -- so the field is smeared onto a
    # grid coarser than its own detail.  InteriorGS is unharmed by the same
    # grid because its gaussians are 6.1mm, three times coarser than the
    # spacing.  The precision is a config field rather than a constant, so it
    # can be raised for a finer field; the payload doubles in size.
    precision = int(precision or gaussians.get("precision") or 16)
    gaussians["precision"] = precision
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
        "z_ordered": bool(z_order),
        "precision_source": "explicit" if precision != 16 else "template_default",
        "recentred": bool(recentre),
        # The translation the volume must re-apply, or the room renders in the
        # wrong place -- correctly, sharply, and several metres from the arm.
        "centre_offset_m": [float(v) for v in centre],
        "z_order_bits": MORTON_BITS,
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
