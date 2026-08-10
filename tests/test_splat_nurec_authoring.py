"""Laying standard 3DGS splats into a NuRec container without guessing.

The aura builder exists for 2D surfels: it synthesizes a structural third
scale and flips the density kernel to planar. A standard 3DGS splat needs
neither - it already has three learned log-scales and the template's
volumetric kernel is the right one. The dangerous edit is the helpful one:
touching the kernel or the scales "for consistency" produces a field that
renders plausibly and wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.nurec_volume_codec import (
    NuRecCodecError,
    decode_nurec_bytes,
    encode_nurec_bytes,
    gaussian_arrays,
)
from blueprint_pipeline.splat_nurec_authoring import build_splat_nurec_document


def _template() -> dict:
    return {
        "version": "0.0.0-test",
        "model": "gaussians",
        "config": {
            "layers": {
                "gaussians": {
                    "precision": 16,
                    "particle": {"density_kernel_planar": False},
                }
            }
        },
        "state_dict": {},
    }


def _splat(count=4, sh_width=45, opacity=None) -> SplatData:
    # float16-exact values so round-trip assertions are equality, not faith.
    xyz = np.array(
        [[1.0, 2.0, 0.5], [3.0, -2.0, 1.5], [-1.0, 4.0, 0.25], [5.0, 0.0, 2.0]],
        dtype=np.float32,
    )[:count]
    return SplatData(
        count=count,
        xyz=xyz,
        opacity=np.asarray(opacity if opacity is not None else [0.5, -1.0, 2.0, 0.0][:count], dtype=np.float32),
        f_dc=np.full((count, 3), 0.25, dtype=np.float32),
        scales=np.array([[-4.0, -3.5, -5.0]] * count, dtype=np.float32),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]] * count, dtype=np.float32),
        properties=("x",),
        sh_rest=np.full((count, sh_width), 0.125, dtype=np.float32),
    )


def test_three_learned_scales_are_written_verbatim():
    """No synthetic structural axis: scale_2 is learned, not derived."""

    built = build_splat_nurec_document(_splat(), template=_template())

    arrays = gaussian_arrays(built)
    assert arrays["scales"].shape == (4, 3)
    np.testing.assert_array_equal(
        arrays["scales"].astype(np.float32),
        np.array([[-4.0, -3.5, -5.0]] * 4, dtype=np.float32),
    )


def test_the_template_kernel_is_left_alone():
    """A volumetric field through a planar kernel is a different surface."""

    built = build_splat_nurec_document(_splat(), template=_template())

    particle = built["config"]["layers"]["gaussians"]["particle"]
    assert particle["density_kernel_planar"] is False
    authoring = built["_blueprint_authoring"]
    assert authoring["density_kernel"] == "template_verbatim"


def test_recentring_moves_positions_and_records_the_offset():
    built = build_splat_nurec_document(_splat(), template=_template())

    authoring = built["_blueprint_authoring"]
    centre = np.asarray(authoring["centre_offset_m"], dtype=np.float32)
    arrays = gaussian_arrays(built)
    restored = arrays["positions"].astype(np.float32) + centre
    np.testing.assert_allclose(restored, _splat().xyz, atol=5e-3)
    assert authoring["recentred"] is True


def test_infinite_opacity_logits_are_clamped_and_counted():
    splat = _splat(opacity=[np.inf, -np.inf, 0.5, np.nan])

    built = build_splat_nurec_document(splat, template=_template())

    arrays = gaussian_arrays(built)
    densities = arrays["densities"].astype(np.float32)
    assert np.isfinite(densities).all()
    assert built["_blueprint_authoring"]["infinite_opacity_logits_clamped"] == 2


def test_a_wrong_spherical_harmonic_width_is_refused():
    with pytest.raises(NuRecCodecError):
        build_splat_nurec_document(_splat(sh_width=9), template=_template())


def test_a_splat_without_sh_rest_is_refused():
    splat = _splat()
    splat = SplatData(
        count=splat.count,
        xyz=splat.xyz,
        opacity=splat.opacity,
        f_dc=splat.f_dc,
        scales=splat.scales,
        quats=splat.quats,
        properties=splat.properties,
        sh_rest=None,
    )

    with pytest.raises(NuRecCodecError):
        build_splat_nurec_document(splat, template=_template())


def test_the_document_survives_the_wire_format():
    """encode -> decode -> arrays is the exact path the renderer takes."""

    built = build_splat_nurec_document(_splat(), template=_template())

    decoded = decode_nurec_bytes(encode_nurec_bytes(built))
    arrays = gaussian_arrays(decoded)
    assert int(arrays["positions"].shape[0]) == 4
    assert arrays["features_specular"].shape == (4, 45)
