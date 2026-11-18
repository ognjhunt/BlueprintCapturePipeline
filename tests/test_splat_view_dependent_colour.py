# Covers (for impacted-test selection):
#   src/blueprint_pipeline/atlas_depth_splat.py
#   src/blueprint_pipeline/splat_sh_rotation.py
"""View-dependent colour (spherical harmonics above degree 0) survives the SPZ codec, and turns with a splat.

Atlas's refined splats seen so far are degree 0 (capture-7c42, 2026-11-17: SPZ v2, 2,000,000 Gaussians,
header degree 0). A degree-0 file must decode and encode byte for byte as before; a degree 1 to 3 file used to
be refused and now keeps its coefficients.
"""
from __future__ import annotations

import dataclasses
import gzip
import struct

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth
from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.splat_sh_rotation import rotate_sh_rest, sh_basis


def _splat(count=40, seed=0) -> SplatData:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4))
    return SplatData(count=count, xyz=rng.uniform(-3, 3, (count, 3)).astype(np.float32),
                     opacity=rng.normal(0, 2, count).astype(np.float32),
                     f_dc=rng.normal(0, 1, (count, 3)).astype(np.float32),
                     scales=rng.uniform(-6, -1, (count, 3)).astype(np.float32),
                     quats=(quats / np.linalg.norm(quats, axis=1, keepdims=True)).astype(np.float32),
                     properties=depth.PLY_PROPERTIES)


def _encode_before(splat: SplatData) -> bytes:
    """``encode_spz`` as it was before higher-order SH (main 573a132e), the degree-0 reference."""
    fixed = np.round(splat.xyz.astype(np.float64) * (1 << depth.SPZ_FRACTIONAL_BITS)).astype(np.int64)
    positions = ((fixed[..., None] & 0xFFFFFF) >> np.array([0, 8, 16])) & 0xFF
    quats = splat.quats.astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats *= np.where(quats[:, :1] < 0, -1.0, 1.0)
    payload = b"".join([
        struct.pack("<IIIBBBB", depth.SPZ_MAGIC, depth.SPZ_VERSION, splat.count, 0, depth.SPZ_FRACTIONAL_BITS, 0, 0),
        positions.astype(np.uint8).tobytes(),
        depth._uint8(255.0 / (1.0 + np.exp(-splat.opacity.astype(np.float64)))).tobytes(),
        depth._uint8(splat.f_dc.astype(np.float64) * (depth.SPZ_COLOR_SCALE * 255.0) + 127.5).tobytes(),
        depth._uint8((splat.scales.astype(np.float64) + 10.0) * 16.0).tobytes(),
        depth._uint8(quats[:, 1:] * 127.5 + 127.5).tobytes(),
    ])
    return gzip.compress(payload, compresslevel=9, mtime=0)


def test_a_degree_zero_splat_encodes_and_decodes_exactly_as_before():
    splat = _splat()
    blob = depth.encode_spz(splat)
    assert blob == _encode_before(splat)
    decoded = depth.decode_spz(blob)
    assert decoded.sh_rest is None and decoded.properties == depth.PLY_PROPERTIES
    again = depth.decode_spz(_encode_before(splat))
    for name in ("xyz", "opacity", "f_dc", "scales", "quats"):
        assert np.array_equal(getattr(decoded, name), getattr(again, name))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_higher_order_coefficients_survive_the_codec_channel_major(degree):
    dim = (degree + 1) ** 2 - 1
    raw = np.random.default_rng(degree).integers(0, 256, (40, 3 * dim))
    splat = dataclasses.replace(_splat(), sh_rest=((raw - 128) / 128).astype(np.float32))
    decoded = depth.decode_spz(depth.encode_spz(splat))
    assert np.array_equal(decoded.sh_rest, splat.sh_rest)
    assert decoded.properties[-1] == f"f_rest_{3 * dim - 1}"
    # The base fields decode as a degree-0 file of the same splat does.
    base = depth.decode_spz(depth.encode_spz(_splat()))
    for name in ("xyz", "opacity", "f_dc", "scales", "quats"):
        assert np.array_equal(getattr(decoded, name), getattr(base, name))


def test_the_spz_layout_keeps_the_colour_channel_innermost():
    """SPZ stores each Gaussian's coefficients coefficient by coefficient, red, green, blue; PLY is channel-major."""
    splat = _splat(count=1)
    head = gzip.decompress(depth.encode_spz(splat))
    sh = bytes([128 + 10 * k + c for k in range(3) for c in range(3)])  # degree 1: coefficient k, channel c
    raw = struct.pack("<IIIBBBB", depth.SPZ_MAGIC, depth.SPZ_VERSION, 1, 1, depth.SPZ_FRACTIONAL_BITS, 0, 0) + \
        head[16:] + sh
    rest = depth.decode_spz(gzip.compress(raw)).sh_rest[0] * 128
    assert rest.tolist() == [0, 10, 20, 1, 11, 21, 2, 12, 22]  # f_rest_0..2 red, 3..5 green, 6..8 blue


def test_a_degree_above_three_or_a_short_body_is_refused():
    splat = dataclasses.replace(_splat(), sh_rest=np.zeros((40, 9), np.float32))
    raw = bytearray(gzip.decompress(depth.encode_spz(splat)))
    with pytest.raises(ValueError, match="spz_invalid"):
        depth.decode_spz(gzip.compress(bytes(raw[:-1])))
    raw[12] = 4
    with pytest.raises(ValueError, match="spz_invalid"):
        depth.decode_spz(gzip.compress(bytes(raw)))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_turning_a_splat_turns_its_view_dependent_colour(degree):
    rng = np.random.default_rng(10 + degree)
    dim = (degree + 1) ** 2 - 1
    rest = rng.normal(size=(6, 3 * dim))
    turn, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    turn *= np.sign(np.linalg.det(turn))
    seen = rng.normal(size=(12, 3))
    seen /= np.linalg.norm(seen, axis=1, keepdims=True)

    def colour(coefficients, directions):
        return np.einsum("nck,mk->nmc", coefficients.reshape(6, 3, dim), sh_basis(directions, degree))
    turned = rotate_sh_rest(rest, turn)
    assert np.abs(colour(turned, seen @ turn.T) - colour(rest, seen)).max() < 1e-9
    assert np.allclose(rotate_sh_rest(rest, np.eye(3)), rest)


@pytest.mark.parametrize("degree", [0, 3])
def test_the_refined_splat_ply_carries_its_degree_into_the_packages_particlefield(tmp_path, degree):
    """The package's ParticleField look (``website_native_appearance``) reads the PLY's ``f_rest_*``."""
    from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply, write_standard_3dgs_ply
    from blueprint_pipeline.particlefield_usd import write_particlefield_usd
    dim = (degree + 1) ** 2 - 1
    splat = _splat()
    if degree:
        splat = dataclasses.replace(splat, sh_rest=np.random.default_rng(1).normal(0, .1, (40, 3 * dim)).astype(np.float32))
    decoded = depth.decode_spz(depth.encode_spz(splat))
    write_standard_3dgs_ply(decoded, tmp_path / "splat.ply")
    assert read_standard_3dgs_ply(tmp_path / "splat.ply").properties[-1] == (f"f_rest_{3 * dim - 1}" if degree
                                                                            else "rot_3")
    pytest.importorskip("pxr")
    authored = write_particlefield_usd(tmp_path / "splat.ply", tmp_path / "look.usdc")
    assert authored["status"] == "completed" and authored["sh_degree"] == degree
