"""Tests for blueprint_pipeline.gaussian_splat_decode."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    convert_to_standard_ply,
    find_splat_transform_cli,
    read_standard_3dgs_ply,
    write_standard_3dgs_ply,
)


def _make_splat(count: int = 5) -> SplatData:
    rng = np.random.default_rng(7)
    return SplatData(
        count=count,
        xyz=rng.standard_normal((count, 3)).astype(np.float32),
        opacity=rng.standard_normal(count).astype(np.float32),
        f_dc=rng.standard_normal((count, 3)).astype(np.float32),
        scales=rng.standard_normal((count, 3)).astype(np.float32),
        quats=rng.standard_normal((count, 4)).astype(np.float32),
        properties=(),
    )


def test_round_trip_standard_3dgs_ply(tmp_path: Path) -> None:
    original = _make_splat(11)
    out = write_standard_3dgs_ply(original, tmp_path / "scene.ply")
    loaded = read_standard_3dgs_ply(out)
    assert loaded.count == 11
    np.testing.assert_allclose(loaded.xyz, original.xyz, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loaded.opacity, original.opacity, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loaded.f_dc, original.f_dc, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loaded.scales, original.scales, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loaded.quats, original.quats, rtol=0, atol=1e-6)
    assert "x" in loaded.properties and "rot_3" in loaded.properties


def test_opacity_sigmoid_and_aabb() -> None:
    splat = _make_splat(4)
    sig = splat.opacity_sigmoid
    assert sig.shape == (4,)
    assert np.all((sig >= 0.0) & (sig <= 1.0))
    lo, hi = splat.aabb()
    assert np.all(lo <= hi)


def test_reader_rejects_compressed_ply(tmp_path: Path) -> None:
    # PlayCanvas compressed PLY has a 'chunk' element before 'vertex'.
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element chunk 1\n"
        "property float min_x\n"
        "element vertex 1\n"
        "property uint packed_position\n"
        "end_header\n"
    ).encode("ascii")
    path = tmp_path / "compressed.ply"
    path.write_bytes(header + b"\x00" * 64)
    with pytest.raises(ValueError, match="not_a_standard_3dgs_ply"):
        read_standard_3dgs_ply(path)


def test_reader_rejects_non_binary_format(tmp_path: Path) -> None:
    header = (
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
        "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
        "property float rot_2\nproperty float rot_3\nproperty float f_dc_0\n"
        "property float f_dc_1\nproperty float f_dc_2\nend_header\n"
    ).encode("ascii")
    path = tmp_path / "ascii.ply"
    path.write_bytes(header)
    with pytest.raises(ValueError, match="binary_little_endian"):
        read_standard_3dgs_ply(path)


def test_reader_rejects_missing_properties(tmp_path: Path) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    path = tmp_path / "partial.ply"
    path.write_bytes(header + struct.pack("<3f", 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="missing 3dgs properties"):
        read_standard_3dgs_ply(path)


def test_convert_blocked_when_cli_missing(tmp_path: Path) -> None:
    # repo_root pointed at an empty dir => no splat-transform CLI present.
    src = tmp_path / "in.ply"
    src.write_bytes(b"ply\n")
    result = convert_to_standard_ply(src, tmp_path / "out.ply", repo_root=tmp_path)
    assert result["status"] == "blocked"
    assert "splat_transform_cli_unavailable" in result["blockers"]


def test_convert_blocked_when_source_missing(tmp_path: Path, monkeypatch) -> None:
    # Pretend the CLI exists so we exercise the missing-source branch.
    fake_cli = tmp_path / "tools/splat_render/node_modules/@playcanvas/splat-transform/bin"
    fake_cli.mkdir(parents=True)
    (fake_cli / "cli.mjs").write_text("// fake")
    result = convert_to_standard_ply(
        tmp_path / "does_not_exist.ply", tmp_path / "out.ply", repo_root=tmp_path
    )
    assert result["status"] == "blocked"
    assert "splat_source_missing" in result["blockers"]


def test_find_cli_in_real_repo() -> None:
    # In the real checkout the CLI is installed under tools/splat_render.
    cli = find_splat_transform_cli()
    # Either present (installed) or cleanly absent; must not raise.
    assert cli is None or cli.name == "cli.mjs"
