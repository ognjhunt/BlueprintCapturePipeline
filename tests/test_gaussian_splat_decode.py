"""Tests for blueprint_pipeline.gaussian_splat_decode."""
from __future__ import annotations

import dataclasses
import inspect
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import blueprint_pipeline.gaussian_splat_decode as gsd
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    convert_to_standard_ply,
    find_splat_transform_cli,
    read_standard_3dgs_ply,
    run_splat_transform_cleanup,
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


def test_splat_data_contract_fields_and_dtypes() -> None:
    """Pin the exact SplatData field set + array dtypes/shapes that the hot lane
    (Isaac NuRec geometry analysis) imports this module as a leaf to rely on."""
    field_names = tuple(f.name for f in dataclasses.fields(SplatData))
    assert field_names == (
        "count",
        "xyz",
        "opacity",
        "f_dc",
        "scales",
        "quats",
        "properties",
        "sh_rest",
    )

    count = 6
    splat = _make_splat(count)
    assert isinstance(splat.count, int) and splat.count == count

    # float32 geometry/color/opacity arrays with the documented column counts.
    assert splat.xyz.dtype == np.float32 and splat.xyz.shape == (count, 3)
    assert splat.f_dc.dtype == np.float32 and splat.f_dc.shape == (count, 3)
    assert splat.scales.dtype == np.float32 and splat.scales.shape == (count, 3)
    assert splat.quats.dtype == np.float32 and splat.quats.shape == (count, 4)
    assert splat.opacity.dtype == np.float32 and splat.opacity.shape == (count,)
    assert isinstance(splat.properties, tuple)
    assert splat.sh_rest is None


def test_standard_ply_round_trip_preserves_higher_order_sh(tmp_path: Path) -> None:
    splat = _make_splat(3)
    splat.sh_rest = np.arange(3 * 45, dtype=np.float32).reshape(3, 45)
    output = write_standard_3dgs_ply(splat, tmp_path / "degree3.ply")
    loaded = read_standard_3dgs_ply(output)
    assert loaded.sh_rest is not None
    np.testing.assert_array_equal(loaded.sh_rest, splat.sh_rest)


def test_opacity_sigmoid_clip_bounds_and_aabb_shape() -> None:
    """opacity_sigmoid clips raw logits to [-30, 30] before the sigmoid, so even
    saturating logits stay strictly inside (0, 1); aabb() returns (min, max) (3,)."""
    splat = SplatData(
        count=3,
        xyz=np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0], [-1.0, 4.0, -5.0]], dtype=np.float32),
        opacity=np.array([-1000.0, 0.0, 1000.0], dtype=np.float32),
        f_dc=np.zeros((3, 3), dtype=np.float32),
        scales=np.zeros((3, 3), dtype=np.float32),
        quats=np.zeros((3, 4), dtype=np.float32),
        properties=(),
    )
    sig = splat.opacity_sigmoid
    assert sig.shape == (3,)
    # Clipping to [-30, 30] keeps the sigmoid finite and inside [0, 1]; no overflow
    # warnings or NaNs even for saturating +/-1000 logits.
    assert np.all((sig >= 0.0) & (sig <= 1.0))
    assert np.all(np.isfinite(sig))
    # The clip bound is exactly +/-30: the result matches sigmoid(clip(.., -30, 30)).
    expected = 1.0 / (1.0 + np.exp(-np.clip(splat.opacity, -30.0, 30.0)))
    np.testing.assert_allclose(sig, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sig[1], 0.5, rtol=0, atol=1e-12)

    lo, hi = splat.aabb()
    assert lo.shape == (3,) and hi.shape == (3,)
    np.testing.assert_array_equal(lo, np.array([-1.0, -2.0, -5.0], dtype=np.float32))
    np.testing.assert_array_equal(hi, np.array([1.0, 4.0, 3.0], dtype=np.float32))


def test_public_api_signatures_pinned() -> None:
    """Pin current public signatures so the hot-lane leaf contract cannot drift."""
    assert (
        str(inspect.signature(read_standard_3dgs_ply))
        == "(path: 'str | Path') -> 'SplatData'"
    )
    assert (
        str(inspect.signature(write_standard_3dgs_ply))
        == "(splat: 'SplatData', path: 'str | Path') -> 'Path'"
    )
    assert (
        str(inspect.signature(convert_to_standard_ply))
        == "(src: 'str | Path', dst: 'str | Path', *, repo_root: 'str | Path | None' = None, "
        "node: 'str' = 'node', timeout_seconds: 'int' = 900) -> 'dict'"
    )
    assert (
        str(inspect.signature(find_splat_transform_cli))
        == "(repo_root: 'str | Path | None' = None) -> 'Path | None'"
    )
    # All four are real callables exported from the module under their current names.
    for name in (
        "read_standard_3dgs_ply",
        "write_standard_3dgs_ply",
        "convert_to_standard_ply",
        "find_splat_transform_cli",
    ):
        assert callable(getattr(gsd, name))


def test_splat_transform_cleanup_wraps_upstream_without_decimation(
    tmp_path: Path, monkeypatch
) -> None:
    cli = (
        tmp_path
        / "tools"
        / "splat_render"
        / "node_modules"
        / "@playcanvas"
        / "splat-transform"
        / "bin"
        / "cli.mjs"
    )
    cli.parent.mkdir(parents=True)
    cli.write_text("// fixture", encoding="utf-8")
    source = tmp_path / "scene.spz"
    source.write_bytes(b"source-splat")
    output = tmp_path / "derived" / "cleaned.spz"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="splat-transform v2.7.0\n", stderr="")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"cleaned-splat")
        return SimpleNamespace(
            returncode=0,
            stdout="**Row Count:** 100\n**Row Count:** 96\n",
            stderr="",
        )

    monkeypatch.setattr(gsd.subprocess, "run", fake_run)
    result = run_splat_transform_cleanup(
        source,
        output,
        repo_root=tmp_path,
        gpu=0,
        minimum_opacity=0.02,
        robust_bounds=[-2, -1, -3, 4, 2, 5],
    )

    assert result["status"] == "completed_unqualified_candidate"
    assert result["source"]["splat_count"] == 100
    assert result["render_input_candidate"]["splat_count"] == 96
    assert result["removed_splat_count"] == 4
    assert result["global_decimation_applied"] is False
    assert result["evaluation_render_authorized"] is False
    cleanup_command = calls[-1]
    assert "--no-tty" in cleanup_command
    assert not any(token in {"--decimate", "-F"} for token in cleanup_command)
    assert any(token.startswith("--filter-box=") for token in cleanup_command)


def test_splat_transform_cleanup_forbids_source_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "scene.spz"
    source.write_bytes(b"source-splat")
    result = run_splat_transform_cleanup(source, source, repo_root=tmp_path)
    assert result["status"] == "blocked"
    assert "immutable_source_overwrite_forbidden" in result["blockers"]


def test_splat_transform_cleanup_rejects_inverted_bounds(tmp_path: Path) -> None:
    source = tmp_path / "scene.spz"
    source.write_bytes(b"source-splat")
    result = run_splat_transform_cleanup(
        source,
        tmp_path / "clean.spz",
        repo_root=tmp_path,
        robust_bounds=[1, -1, -1, 0, 1, 1],
    )
    assert result["status"] == "blocked"
    assert "splat_cleanup_robust_bounds_invalid" in result["blockers"]
