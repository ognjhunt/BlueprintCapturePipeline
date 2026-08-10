"""Tests for blueprint_pipeline.particlefield_usd.

The convention math is pure-numpy and tested here; the USD writer needs pxr/usd-core
(skipped when absent — it's exercised via the usd-core venv / Isaac pod).
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.common import sha256_file
from blueprint_pipeline import particlefield_usd
from blueprint_pipeline.particlefield_usd import (
    build_particlefield_arrays,
    write_particlefield_usd,
)

_HAS_PXR = importlib.util.find_spec("pxr") is not None


def _splat(n: int = 6, with_rest: bool = False) -> tuple[SplatData, np.ndarray | None]:
    rng = np.random.default_rng(3)
    splat = SplatData(
        count=n,
        xyz=rng.standard_normal((n, 3)).astype(np.float32),
        opacity=rng.standard_normal(n).astype(np.float32),
        f_dc=rng.standard_normal((n, 3)).astype(np.float32),
        scales=rng.standard_normal((n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        properties=(),
    )
    rest = rng.standard_normal((n, 45)).astype(np.float32) if with_rest else None
    return splat, rest


def test_conventions_exp_sigmoid_normalized() -> None:
    splat, _ = _splat()
    arr = build_particlefield_arrays(splat)
    np.testing.assert_allclose(arr["scales"], np.exp(splat.scales), rtol=1e-5)
    np.testing.assert_allclose(arr["opacities"], 1.0 / (1.0 + np.exp(-splat.opacity)), rtol=1e-5)
    norms = np.linalg.norm(arr["orientations"], axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)
    assert np.all((arr["opacities"] >= 0) & (arr["opacities"] <= 1))


def test_signed_infinite_opacity_logits_are_exact_alpha_endpoints() -> None:
    splat, _ = _splat(4)
    splat.opacity[0] = np.inf
    splat.opacity[1] = -np.inf

    arr = build_particlefield_arrays(splat)

    assert arr["opacities"][0] == 1.0
    assert arr["opacities"][1] == 0.0
    assert arr["positive_infinite_opacity_logit_count"] == 1
    assert arr["negative_infinite_opacity_logit_count"] == 1


def test_nan_opacity_and_nonfinite_geometry_remain_forbidden() -> None:
    nan_opacity, _ = _splat(3)
    nan_opacity.opacity[0] = np.nan
    with pytest.raises(ValueError, match="particlefield_nonfinite_input"):
        build_particlefield_arrays(nan_opacity)

    infinite_position, _ = _splat(3)
    infinite_position.xyz[0, 0] = np.inf
    with pytest.raises(ValueError, match="particlefield_nonfinite_input"):
        build_particlefield_arrays(infinite_position)


def test_zero_quaternion_and_scale_overflow_fail_closed() -> None:
    zero_quaternion, _ = _splat(3)
    zero_quaternion.quats[0] = 0.0
    with pytest.raises(ValueError, match="particlefield_zero_quaternion"):
        build_particlefield_arrays(zero_quaternion)

    scale_overflow, _ = _splat(3)
    scale_overflow.scales[0] = 1.0e4
    with pytest.raises(ValueError, match="particlefield_activated_scale_invalid"):
        build_particlefield_arrays(scale_overflow)


def test_degree0_dc_only() -> None:
    splat, _ = _splat(5)
    arr = build_particlefield_arrays(splat)
    assert arr["sh_degree"] == 0
    assert arr["sh_coefficients"].shape == (5, 3)
    np.testing.assert_allclose(arr["sh_coefficients"], splat.f_dc, rtol=1e-6)


def test_full_sh_layout_degree3() -> None:
    n = 4
    splat, _ = _splat(n)
    # construct f_rest so coeff k (1..15) per channel is identifiable: R=k, G=100+k, B=200+k
    rest = np.zeros((n, 45), dtype=np.float32)
    for k in range(15):
        rest[:, k] = k + 1            # R band
        rest[:, 15 + k] = 100 + k + 1  # G band
        rest[:, 30 + k] = 200 + k + 1  # B band
    arr = build_particlefield_arrays(splat, sh_rest=rest)
    assert arr["sh_degree"] == 3
    assert arr["sh_coefficients"].shape == (n * 16, 3)
    coeffs = arr["sh_coefficients"].reshape(n, 16, 3)
    np.testing.assert_allclose(coeffs[:, 0, :], splat.f_dc, rtol=1e-6)  # DC first
    # coeff 1 == (R_0, G_0, B_0) == (1, 101, 201)
    np.testing.assert_allclose(coeffs[0, 1, :], [1.0, 101.0, 201.0], rtol=1e-6)
    np.testing.assert_allclose(coeffs[0, 15, :], [15.0, 115.0, 215.0], rtol=1e-6)


def test_extent_is_aabb() -> None:
    splat, _ = _splat(20)
    arr = build_particlefield_arrays(splat)
    np.testing.assert_allclose(arr["extent"][0], splat.xyz.min(0), rtol=1e-6)
    np.testing.assert_allclose(arr["extent"][1], splat.xyz.max(0), rtol=1e-6)


def test_write_blocked_without_pxr_or_writes_usd(tmp_path: Path) -> None:
    splat, _ = _splat(8)
    out = tmp_path / "scene.usd"
    result = write_particlefield_usd(splat, out)
    if not _HAS_PXR:
        assert result["status"] == "blocked"
        assert "usd_core_unavailable" in result["blockers"]
    else:
        assert result["status"] == "completed"
        assert out.is_file() and result["splat_count"] == 8
        assert result["schema"] == "ParticleField3DGaussianSplat"
        assert result["default_prim"] == "/World"
        assert result["output_sha256"] == f"sha256:{sha256_file(out)}"


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_path_source_is_digest_bound_and_authors_default_prim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pxr import Usd

    splat, _ = _splat(8)
    source = tmp_path / "retained_scene_gaussians.ply"
    source.write_bytes(b"sealed-standard-3dgs-fixture")
    source_sha256 = f"sha256:{sha256_file(source)}"
    monkeypatch.setattr(
        particlefield_usd, "read_standard_3dgs_ply", lambda _: splat
    )
    out = tmp_path / "scene.usdc"
    receipt = tmp_path / "particlefield_receipt.json"

    result = write_particlefield_usd(
        source,
        out,
        expected_source_sha256=source_sha256,
        receipt_path=receipt,
    )

    assert result["status"] == "completed"
    assert result["source_sha256"] == source_sha256
    assert result["source_kind"] == "standard_3dgs_ply"
    assert result["sealed_source_mutated"] is False
    assert json.loads(receipt.read_text(encoding="utf-8")) == result
    stage = Usd.Stage.Open(str(out))
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_path_source_digest_mismatch_fails_before_authoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splat, _ = _splat(8)
    source = tmp_path / "retained_scene_gaussians.ply"
    source.write_bytes(b"actual-source")
    monkeypatch.setattr(
        particlefield_usd, "read_standard_3dgs_ply", lambda _: splat
    )
    out = tmp_path / "scene.usdc"

    result = write_particlefield_usd(
        source,
        out,
        expected_source_sha256="sha256:" + ("0" * 64),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["particlefield_3dgs_source_sha256_mismatch"]
    assert not out.exists()
