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


# --- the look read back: the inverse of the writer, where the file places it -----------------------------------

#: The website look's frame (``website_native_appearance``): a provider's Y-up world, scaled, turned to Z-up and moved.
_WEBSITE_TRANSFORM = np.eye(4)
_WEBSITE_TRANSFORM[:3, :3] = 0.8 * np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_WEBSITE_TRANSFORM[:3, 3] = [0.3, -0.2, 0.1]


def _matrices(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices of (w, x, y, z) quaternions."""
    w, x, y, z = (np.asarray(quats, float) / np.linalg.norm(quats, axis=1, keepdims=True)).T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def _field(path: Path, quats_wxyz, *, transform=None, opacities=None, degree: int = 0, prim: str = "/World/G",
           stage=None):
    """A ParticleField3DGaussianSplat authored by hand, orientations as Gf.Quatf(real, i, j, k)."""
    from pxr import Gf, Usd, UsdGeom, UsdVol, Vt

    count = len(quats_wxyz)
    rng = np.random.default_rng(count + degree)
    own = stage is None
    stage = stage or Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    field = UsdVol.ParticleField3DGaussianSplat.Define(stage, prim)
    field.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(rng.uniform(-1, 1, (count, 3)).astype(np.float32)))
    field.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(rng.uniform(0.01, 0.1, (count, 3)).astype(np.float32)))
    field.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(np.asarray(
        rng.uniform(0.05, 0.95, count) if opacities is None else opacities, dtype=np.float32)))
    field.CreateOrientationsAttr(Vt.QuatfArray([Gf.Quatf(*map(float, q)) for q in quats_wxyz]))
    field.CreateRadianceSphericalHarmonicsCoefficientsAttr(Vt.Vec3fArray.FromNumpy(
        rng.standard_normal((count * (degree + 1) ** 2, 3)).astype(np.float32)))
    field.CreateRadianceSphericalHarmonicsDegreeAttr(degree)
    if transform is not None:
        UsdGeom.Xformable(field.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*np.asarray(transform).T.ravel()))
    if own:
        stage.GetRootLayer().Save()
    return field


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_the_writers_look_reads_back_as_its_values_where_the_file_places_them(tmp_path: Path) -> None:
    from pxr import Usd
    from blueprint_pipeline.splat_sh_rotation import sh_basis

    rng = np.random.default_rng(5)
    n = 1500
    splat = SplatData(count=n, xyz=rng.uniform([-3, 0, -3], [3, 2.5, 3], (n, 3)).astype(np.float32),
                      opacity=rng.normal(0, 3, n).astype(np.float32), f_dc=rng.normal(0, 1, (n, 3)).astype(np.float32),
                      scales=rng.uniform(-5, -3, (n, 3)).astype(np.float32),
                      quats=rng.standard_normal((n, 4)).astype(np.float32), properties=(),
                      sh_rest=(0.1 * rng.standard_normal((n, 45))).astype(np.float32))
    authored = write_particlefield_usd(splat, tmp_path / "look.usdc",
                                       layer_transform_row_major=_WEBSITE_TRANSFORM.T.tolist())
    assert authored["status"] == "completed"
    back = particlefield_usd.read_particlefield_splat(tmp_path / "look.usdc")
    stage = Usd.Stage.Open(str(tmp_path / "look.usdc"))
    prim = stage.GetPrimAtPath(authored["prim_path"])
    linear, scale = _WEBSITE_TRANSFORM[:3, :3], 0.8
    rotation = linear / scale

    def stored(name):
        return np.asarray(prim.GetAttribute(name).Get(), dtype=float)
    assert back.count == n and back.sh_rest.shape == (n, 45)
    np.testing.assert_allclose(back.xyz, stored("positions") @ linear.T + _WEBSITE_TRANSFORM[:3, 3], atol=1e-6)
    # Fed back through the writer's own activations, the PLY's values are the look's.
    again = build_particlefield_arrays(back)
    np.testing.assert_allclose(again["scales"], scale * stored("scales"), rtol=1e-6)
    np.testing.assert_allclose(again["opacities"], stored("opacities"), atol=1e-7)
    coefficients = stored("radiance:sphericalHarmonicsCoefficients").reshape(n, 16, 3)
    np.testing.assert_array_equal(back.f_dc, coefficients[:, 0, :].astype(np.float32))
    gf = [(q.GetReal(), *q.GetImaginary()) for q in prim.GetAttribute("orientations").Get()]
    np.testing.assert_allclose(_matrices(back.quats), rotation @ _matrices(np.asarray(gf)), atol=1e-6)
    # View-dependent colour: a world direction through the PLY's bands, the look's direction through the USD's.
    directions = rng.standard_normal((40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ply_bands = back.sh_rest.reshape(n, 3, 15).transpose(0, 2, 1)
    np.testing.assert_allclose(np.einsum("mk,nkc->nmc", sh_basis(directions, 3), ply_bands),
                               np.einsum("mk,nkc->nmc", sh_basis(directions @ rotation, 3), coefficients[:, 1:]),
                               atol=1e-6)


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_orientations_are_read_real_part_first_and_any_turn_is_applied(tmp_path: Path) -> None:
    quats = np.random.default_rng(2).standard_normal((50, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    _field(tmp_path / "plain.usda", quats)
    np.testing.assert_allclose(particlefield_usd.read_particlefield_splat(tmp_path / "plain.usda").quats, quats,
                               atol=1e-6)
    for axis in range(3):  # half turns: the quaternion's largest part is never w
        half = np.diag([-1.0, -1.0, -1.0])
        half[axis, axis] = 1.0
        transform = np.eye(4)
        transform[:3, :3] = half
        _field(tmp_path / f"half{axis}.usda", quats, transform=transform)
        turned = particlefield_usd.read_particlefield_splat(tmp_path / f"half{axis}.usda").quats
        np.testing.assert_allclose(_matrices(turned), half @ _matrices(quats), atol=1e-6)


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_an_exact_opacity_of_zero_or_one_is_a_bounded_logit(tmp_path: Path) -> None:
    _field(tmp_path / "ends.usda", np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)), opacities=[0.0, 1.0, 0.5])
    logits = particlefield_usd.read_particlefield_splat(tmp_path / "ends.usda").opacity
    assert logits.tolist() == [-particlefield_usd.OPACITY_LOGIT_BOUND, particlefield_usd.OPACITY_LOGIT_BOUND, 0.0]
    assert np.isfinite(logits).all()


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_fields_of_different_degree_join_with_zero_bands(tmp_path: Path) -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateNew(str(tmp_path / "two.usda"))
    _field(tmp_path, np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)), degree=0, prim="/World/A", stage=stage)
    _field(tmp_path, np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)), degree=1, prim="/World/B", stage=stage)
    stage.GetRootLayer().Save()
    joined = particlefield_usd.read_particlefield_splat(tmp_path / "two.usda")
    assert joined.count == 9 and joined.sh_rest.shape == (9, 9)
    assert not joined.sh_rest[:4].any() and joined.sh_rest[4:].any()


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
@pytest.mark.parametrize("case", ["no_field", "shear", "mirror"])
def test_no_splat_or_a_transform_a_gaussian_cannot_take_is_refused(tmp_path: Path, case: str) -> None:
    from pxr import Usd, UsdGeom

    path = tmp_path / f"{case}.usda"
    if case == "no_field":
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.Cube.Define(stage, "/World/Box")
        stage.GetRootLayer().Save()
        reason = particlefield_usd.PARTICLEFIELD_ABSENT
    else:
        transform = np.eye(4)
        transform[:3, :3] = [[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] if case == "shear" else np.diag([-1, 1, 1])
        _field(path, np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)), transform=transform)
        reason = "particlefield_transform_not_similarity"
    with pytest.raises(ValueError, match=reason):
        particlefield_usd.read_particlefield_splat(path)


def _covariances(quats_wxyz: np.ndarray, scales: np.ndarray, linear: np.ndarray | None = None) -> np.ndarray:
    """The 3D covariance a 3DGS renderer builds (``build_scaling_rotation``: L = R S, Sigma = L L^T), optionally
    carried by a linear map (Sigma' = A Sigma A^T)."""
    covariance = _matrices(quats_wxyz) @ (np.asarray(scales, float)[:, :, None] ** 2 * np.eye(3)) @ np.swapaxes(
        _matrices(quats_wxyz), 1, 2)
    return covariance if linear is None else linear @ covariance @ linear.T


def _axis_angle(axis, degrees: float) -> list[float]:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    half = np.radians(degrees) / 2
    return [np.cos(half), *(np.sin(half) * axis)]


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_known_rotations_render_the_same_covariance_in_the_source_the_usd_and_the_ply(tmp_path: Path) -> None:
    """Owner, 2026-11-18: the writer stored every Gaussian's rotation scrambled (``FromNumpy`` reads x, y, z, w).
    Known turns about oblique axes, with three unequal scales, must render the same ellipsoid from the source
    splat, from the USD as any USD reader takes it (Gf real part first), and from the PLY made of that USD."""
    from pxr import Usd
    from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply, write_viewer_3dgs_ply

    known = np.asarray([_axis_angle((1, 2, 3), 37), _axis_angle((-2, 0.5, 1), 115), _axis_angle((0.3, -1, 0.2), 170),
                        _axis_angle((0, 0, 1), 90), _axis_angle((1, 1, 0), 63)])
    n = 1500
    rng = np.random.default_rng(18)
    splat = SplatData(count=n, xyz=rng.uniform([-3, 0, -3], [3, 2.5, 3], (n, 3)).astype(np.float32),
                      opacity=np.full(n, 2.0, np.float32), f_dc=np.zeros((n, 3), np.float32),
                      scales=np.tile(np.log([0.02, 0.05, 0.11]), (n, 1)).astype(np.float32),
                      quats=np.resize(known, (n, 4)).astype(np.float32), properties=())
    source = _covariances(splat.quats, np.exp(splat.scales.astype(float)), _WEBSITE_TRANSFORM[:3, :3])
    # The scrambled layout renders these as other ellipsoids: the test can tell them apart.
    scrambled = _covariances(splat.quats[:, [3, 0, 1, 2]], np.exp(splat.scales.astype(float)), _WEBSITE_TRANSFORM[:3, :3])
    assert np.abs(scrambled - source).max() > 1e-3

    authored = write_particlefield_usd(splat, tmp_path / "look.usdc",
                                       layer_transform_row_major=_WEBSITE_TRANSFORM.T.tolist())
    assert authored["status"] == "completed"
    stage = Usd.Stage.Open(str(tmp_path / "look.usdc"))
    prim = stage.GetPrimAtPath(authored["prim_path"])
    gf = np.asarray([(q.GetReal(), *q.GetImaginary()) for q in prim.GetAttribute("orientations").Get()])
    usd = _covariances(gf, np.asarray(prim.GetAttribute("scales").Get(), float), _WEBSITE_TRANSFORM[:3, :3])
    np.testing.assert_allclose(usd, source, rtol=0, atol=1e-8)

    write_viewer_3dgs_ply(particlefield_usd.read_particlefield_splat(tmp_path / "look.usdc"), tmp_path / "look.ply")
    ply = read_standard_3dgs_ply(tmp_path / "look.ply")
    np.testing.assert_allclose(_covariances(ply.quats, np.exp(ply.scales.astype(float))), source, rtol=0, atol=1e-8)
