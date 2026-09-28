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
from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz
from blueprint_pipeline.nurec_volume_codec import build_state_dict
from blueprint_pipeline.particlefield_usd import (
    build_particlefield_arrays,
    write_particlefield_usd,
    write_particlefield_usd_from_nurec,
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
    assert arr["sh_element_size"] == 16
    assert arr["sh_coefficients"].shape == (n * 16, 3)
    coeffs = arr["sh_coefficients"].reshape(n, 16, 3)
    np.testing.assert_allclose(coeffs[:, 0, :], splat.f_dc, rtol=1e-6)  # DC first
    # coeff 1 == (R_0, G_0, B_0) == (1, 101, 201)
    np.testing.assert_allclose(coeffs[0, 1, :], [1.0, 101.0, 201.0], rtol=1e-6)
    np.testing.assert_allclose(coeffs[0, 15, :], [15.0, 115.0, 215.0], rtol=1e-6)
    expected_display = np.clip(0.5 + particlefield_usd.SH_C0 * splat.f_dc, 0.0, 1.0)
    np.testing.assert_allclose(arr["display_colors"], expected_display, rtol=1e-6)


def test_nurec_coefficient_major_sh_layout_is_not_transposed_as_inria_ply() -> None:
    n = 4
    splat, _ = _splat(n)
    rest = np.zeros((n, 45), dtype=np.float32)
    for coefficient in range(15):
        rest[:, coefficient * 3 : coefficient * 3 + 3] = [
            coefficient + 1,
            100 + coefficient + 1,
            200 + coefficient + 1,
        ]

    arr = build_particlefield_arrays(
        splat,
        sh_rest=rest,
        sh_rest_layout=particlefield_usd.SH_REST_LAYOUT_COEFFICIENT_MAJOR,
    )
    coeffs = arr["sh_coefficients"].reshape(n, 16, 3)

    np.testing.assert_allclose(coeffs[0, 1, :], [1.0, 101.0, 201.0])
    np.testing.assert_allclose(coeffs[0, 15, :], [15.0, 115.0, 215.0])
    assert arr["source_sh_rest_layout"] == "coefficient_major_rgb_triplets"


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

    splat, rest = _splat(8, with_rest=True)
    source = tmp_path / "retained_scene_gaussians.ply"
    source.write_bytes(b"sealed-standard-3dgs-fixture")
    source_sha256 = f"sha256:{sha256_file(source)}"
    monkeypatch.setattr(particlefield_usd, "read_standard_3dgs_ply", lambda _: splat)
    out = tmp_path / "scene.usdc"
    receipt = tmp_path / "particlefield_receipt.json"

    result = write_particlefield_usd(
        source,
        out,
        sh_rest=rest,
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
    prim = stage.GetPrimAtPath(result["prim_path"])
    from pxr import UsdGeom

    sh_primvar = UsdGeom.Primvar(prim.GetAttribute("radiance:sphericalHarmonicsCoefficients"))
    assert sh_primvar.GetElementSize() == 16
    assert sh_primvar.GetInterpolation() == UsdGeom.Tokens.vertex
    display_color = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    assert display_color
    assert display_color.GetInterpolation() == UsdGeom.Tokens.vertex
    assert len(display_color.Get()) == 8
    assert result["sh_primvar_element_size"] == 16
    assert result["sh_primvar_interpolation"] == "vertex"
    assert result["display_color_fallback_authored"] is True
    assert result["particlefield_emissive_material_binding_authored"] is False
    assert result["particlefield_emissive_material_inputs"] == "upstream_native_unbound"
    assert result["particlefield_emissive_material_input_values"] == {}
    assert result["particlefield_custom_render_hints_authored"] is False
    binding = prim.GetRelationship("material:binding").GetTargets()
    assert list(binding) == []
    assert not prim.GetAttribute("projectionModeHint").HasAuthoredValueOpinion()
    assert not prim.GetAttribute("sortingModeHint").HasAuthoredValueOpinion()


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_path_source_digest_mismatch_fails_before_authoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splat, _ = _splat(8)
    source = tmp_path / "retained_scene_gaussians.ply"
    source.write_bytes(b"actual-source")
    monkeypatch.setattr(particlefield_usd, "read_standard_3dgs_ply", lambda _: splat)
    out = tmp_path / "scene.usdc"

    result = write_particlefield_usd(
        source,
        out,
        expected_source_sha256="sha256:" + ("0" * 64),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["particlefield_3dgs_source_sha256_mismatch"]
    assert not out.exists()


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_nurec_is_represented_as_particlefield_without_changing_gaussians(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    from blueprint_pipeline.native_task_appearance_frame_alignment import (
        measure_native_task_appearance_frame,
    )

    splat, rest = _splat(8, with_rest=True)
    document = {
        "version": "0.2.576",
        "model": "nre",
        "config": {
            "layers": {
                "gaussians": {
                    "precision": 32,
                    "density_activation": "sigmoid",
                    "scale_activation": "exp",
                    "rotation_activation": "normalize",
                    "particle": {
                        "density_kernel_planar": False,
                        "radiance_sph_degree": 3,
                    },
                }
            },
            "renderer": {"name": "3dgut-nrend"},
        },
        "state_dict": build_state_dict(
            {
                "positions": splat.xyz,
                "rotations": splat.quats,
                "scales": splat.scales,
                "densities": splat.opacity[:, None],
                "features_albedo": splat.f_dc,
                "features_specular": rest,
            },
            precision=32,
        ),
    }
    source = tmp_path / "configured_appearance.usdz"
    write_aura_nurec_usdz(document, source)
    source_sha256 = f"sha256:{sha256_file(source)}"
    output = tmp_path / "scene_appearance.usdc"
    receipt_path = tmp_path / "particlefield_authoring_receipt.v1.json"

    result = write_particlefield_usd_from_nurec(
        source,
        output,
        expected_source_sha256=source_sha256,
        receipt_path=receipt_path,
    )

    assert result["status"] == "completed"
    assert result["source_kind"] == "nurec_usdz"
    assert result["source_sha256"] == source_sha256
    assert result["exact_learned_arrays_preserved"] is True
    assert result["representation_conversion_only"] is True
    assert result["source_nurec_sh_rest_layout"] == (
        "coefficient_major_rgb_triplets"
    )
    assert result["splat_count"] == splat.count
    assert result["particlefield_authoring_implementation"] == (
        "nvidia_usd_convert_gsplat"
    )
    assert result["upstream_converter"]["version"] == "0.1.15"
    assert result["particlefield_emissive_material_binding_authored"] is False
    assert result["particlefield_custom_render_hints_authored"] is False
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == result
    before = measure_native_task_appearance_frame(source)
    after = measure_native_task_appearance_frame(output)
    assert before["representation"] == "nurec_volume"
    assert after["representation"] == "particlefield_3d_gaussian_splat"
    assert after["gaussian_count"] == before["gaussian_count"]
    stage = Usd.Stage.Open(str(output))
    prim = stage.GetPrimAtPath(result["prim_path"])
    authored_sh = np.asarray(
        prim.GetAttribute("radiance:sphericalHarmonicsCoefficients").Get()
    ).reshape(splat.count, 16, 3)
    expected_sh = np.concatenate(
        [splat.f_dc[:, None, :], rest.reshape(splat.count, 15, 3)],
        axis=1,
    )
    np.testing.assert_array_equal(authored_sh, expected_sh)
    np.testing.assert_allclose(
        after["stored_tensor_occupied_bounds_m"]["minimum"],
        before["stored_tensor_occupied_bounds_m"]["minimum"],
    )
    np.testing.assert_allclose(
        after["stored_tensor_occupied_bounds_m"]["maximum"],
        before["stored_tensor_occupied_bounds_m"]["maximum"],
    )


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_nurec_with_scene_relative_gaussian_divergence_is_refused(
    tmp_path: Path,
) -> None:
    count = 1_200
    rng = np.random.default_rng(839873)
    splat = SplatData(
        count=count,
        xyz=rng.uniform([-4.0, -6.0, 0.0], [4.0, 6.0, 3.0], (count, 3)).astype(np.float32),
        opacity=np.full(count, 2.0, dtype=np.float32),
        f_dc=np.zeros((count, 3), dtype=np.float32),
        scales=np.full((count, 3), -3.0, dtype=np.float32),
        quats=np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (count, 1),
        ),
        properties=(),
        sh_rest=np.zeros((count, 45), dtype=np.float32),
    )
    splat.xyz[-1] = [8_000.0, -7_000.0, 900.0]
    splat.scales[-1] = [6.88, 0.0, 0.0]
    document = {
        "version": "0.2.576",
        "model": "nre",
        "config": {
            "layers": {
                "gaussians": {
                    "precision": 32,
                    "density_activation": "sigmoid",
                    "scale_activation": "exp",
                    "rotation_activation": "normalize",
                    "particle": {
                        "density_kernel_planar": False,
                        "radiance_sph_degree": 3,
                    },
                }
            },
            "renderer": {"name": "3dgut-nrend"},
        },
        "state_dict": build_state_dict(
            {
                "positions": splat.xyz,
                "rotations": splat.quats,
                "scales": splat.scales,
                "densities": splat.opacity[:, None],
                "features_albedo": splat.f_dc,
                "features_specular": splat.sh_rest,
            },
            precision=32,
        ),
    }
    source = tmp_path / "configured_appearance.usdz"
    write_aura_nurec_usdz(document, source)
    output = tmp_path / "scene_appearance.usdc"
    receipt_path = tmp_path / "particlefield_authoring_receipt.v1.json"

    result = write_particlefield_usd_from_nurec(
        source,
        output,
        expected_source_sha256=f"sha256:{sha256_file(source)}",
        receipt_path=receipt_path,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["particlefield_gaussian_field_quality_invalid"]
    assert result["gaussian_field_quality"]["status"] == "blocked"
    assert not output.exists()
    assert not receipt_path.exists()
