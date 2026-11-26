"""A PLY source's look is read and built a run at a time: the same USD bytes and receipt as the whole arrays.

``write_particlefield_usd`` given ``SplatData`` still builds every array at once
(``build_particlefield_arrays``), as every look was written before 2026-11-25; it is the
reference here. A path source is read from the PLY's memory map, an attribute and a run of
rows at a time, so the traced (numpy) memory stays near one attribute's array.
"""
from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import particlefield_usd
from blueprint_pipeline.gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply

pytest.importorskip("pxr")

#: Receipt fields that name where the look came from, not what it holds.
_SOURCE_FIELDS = ("source_sha256", "source_kind", "output", "receipt_digest")


def _splat(count: int, *, degree: int = 0, seed: int = 0) -> SplatData:
    rng = np.random.default_rng(seed)
    opacity = rng.normal(0.0, 3.0, count).astype(np.float32)
    opacity[::97] = np.inf
    opacity[5::89] = -np.inf
    rest = None
    if degree:
        rest = rng.normal(0.0, 0.2, (count, 3 * ((degree + 1) ** 2 - 1))).astype(np.float32)
    return SplatData(count=count, xyz=rng.normal(0.0, 1.0, (count, 3)).astype(np.float32), opacity=opacity,
                     f_dc=rng.normal(0.0, 0.5, (count, 3)).astype(np.float32),
                     scales=rng.normal(-4.0, 0.3, (count, 3)).astype(np.float32),
                     quats=rng.normal(0.0, 1.0, (count, 4)).astype(np.float32), properties=(), sh_rest=rest)


def _written(tmp_path: Path, splat: SplatData, name: str = "look.ply") -> Path:
    return write_standard_3dgs_ply(splat, tmp_path / name)


def _content(receipt: dict) -> dict:
    return {key: value for key, value in receipt.items() if key not in _SOURCE_FIELDS}


@pytest.mark.parametrize("degree", [0, 3])
@pytest.mark.parametrize("chunk", [1, 7, 1 << 18])
def test_streamed_look_is_the_whole_arrays_look(tmp_path, monkeypatch, degree, chunk):
    ply = _written(tmp_path, _splat(513, degree=degree))
    whole = particlefield_usd.write_particlefield_usd(read_standard_3dgs_ply(ply), tmp_path / "whole.usdc")
    monkeypatch.setattr(particlefield_usd, "LOOK_CHUNK_ROWS", chunk)
    streamed = particlefield_usd.write_particlefield_usd(ply, tmp_path / "streamed.usdc")
    assert whole["status"] == streamed["status"] == "completed"
    assert (tmp_path / "streamed.usdc").read_bytes() == (tmp_path / "whole.usdc").read_bytes()
    assert _content(streamed) == _content(whole)
    assert streamed["source_kind"] == "standard_3dgs_ply"
    assert streamed["positive_infinite_opacity_logit_count"] == 6
    assert streamed["sh_degree"] == degree


def test_streamed_look_holds_about_one_attribute(tmp_path, monkeypatch):
    """Base (decode, every attribute, the float64 rotations): about 230 traced bytes a Gaussian."""
    count = 400_000
    ply = _written(tmp_path, _splat(count))
    monkeypatch.setattr(particlefield_usd, "LOOK_CHUNK_ROWS", 1 << 14)
    tracemalloc.start()
    try:
        result = particlefield_usd.write_particlefield_usd(ply, tmp_path / "look.usdc")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["status"] == "completed"
    # The orientations' float32 array (16 bytes a Gaussian) is the largest built at once; the quality measure
    # reads in fixed blocks (``gaussian_field_quality.CHUNK_ROWS``), about 25 MB whatever the field's size.
    assert peak < 20 * count + (32 << 20), peak


@pytest.mark.parametrize("faults", [
    ({"quats": 10}, {"xyz": 400}),             # a zero rotation early, a NaN position later: the NaN is named
    ({"scales": 3}, {"opacity": 500}),          # an overflowing scale early, a NaN opacity later
    ({"quats": 450}, {"scales": 2}),            # the scale is checked before any rotation
    ({"rest": 300}, {"quats": 1}),              # the rotation before the coefficients
])
def test_streamed_look_refuses_as_the_whole_arrays_do(tmp_path, monkeypatch, faults):
    splat = _splat(513, degree=1)
    for fault in faults:
        for field, row in fault.items():
            if field == "quats":
                splat.quats[row] = 0.0
            elif field == "scales":
                splat.scales[row, 1] = 200.0
            elif field == "rest":
                splat.sh_rest[row, 0] = np.nan
            else:
                getattr(splat, field)[row] = np.nan
    ply = _written(tmp_path, splat)
    with pytest.raises(ValueError) as whole:
        particlefield_usd.write_particlefield_usd(read_standard_3dgs_ply(ply), tmp_path / "whole.usdc")
    monkeypatch.setattr(particlefield_usd, "LOOK_CHUNK_ROWS", 64)
    with pytest.raises(ValueError) as streamed:
        particlefield_usd.write_particlefield_usd(ply, tmp_path / "streamed.usdc")
    assert str(streamed.value) == str(whole.value)
    assert not (tmp_path / "streamed.usdc").exists()


def test_streamed_look_reads_the_header_as_the_decoder_does(tmp_path):
    ply = _written(tmp_path, _splat(16))
    data = ply.read_bytes()
    truncated = tmp_path / "truncated.ply"
    truncated.write_bytes(data[:-4])
    with pytest.raises(ValueError) as decoded:
        read_standard_3dgs_ply(truncated)
    with pytest.raises(ValueError) as streamed:
        particlefield_usd.write_particlefield_usd(truncated, tmp_path / "look.usdc")
    assert str(streamed.value) == str(decoded.value)
