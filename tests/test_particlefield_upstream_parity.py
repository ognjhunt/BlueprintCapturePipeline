"""Attribute parity: Blueprint NuRec conversion vs NVIDIA 3DGRUT direct transcode.

Scene 839873 (2026-09-02): the failing rainbow renders raised the question of
whether Blueprint's private ``.nurec`` tensor reinterpretation could ever be
treated as equivalent to the transcode NVIDIA documents.  The reference model
in ``particlefield_upstream_parity`` transcribes the pinned upstream importer,
adapter and LightField writer; it was checked against the real
``threedgrut.export.scripts.transcode`` output at that revision (positions and
SH exact, other attributes within float32 rounding).  These tests pin that
contract hermetically so a future conversion change cannot silently diverge.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.particlefield_upstream_parity import (
    THREEDGRUT_REFERENCE_REVISION,
    UPSTREAM_ONLY_PRIM_AUTHORING,
    apply_nurec_volume_transform,
    attribute_digests,
    compare_particlefield_attributes,
    threedgrut_lightfield_attributes,
)
from blueprint_pipeline.particlefield_usd import (
    SH_REST_LAYOUT_INRIA_CHANNEL_MAJOR,
    build_particlefield_arrays,
)

_HAS_PXR = importlib.util.find_spec("pxr") is not None
_HAS_UPSTREAM_CONVERTER = importlib.util.find_spec("usd_convert_gsplat") is not None


def _nurec_arrays(n: int = 512, *, seed: int = 839873) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    densities = (rng.normal(size=(n, 1)) * 3).astype(np.float16)
    # 58.56% of the configured Scene 839873 field is +inf (exact opacity 1.0).
    densities[: int(n * 0.5856)] = np.inf
    albedo = (rng.normal(size=(n, 3)) * 2).astype(np.float16)
    albedo[:8] = 29.1875  # the configured field's albedo tail
    return {
        "positions": rng.uniform(-3, 3, (n, 3)).astype(np.float16),
        "rotations": rng.normal(size=(n, 4)).astype(np.float16),
        "scales": rng.uniform(-6.0, -2.0, (n, 3)).astype(np.float16),
        "densities": densities,
        "features_albedo": albedo,
        "features_specular": rng.normal(size=(n, 45)).astype(np.float16),
    }


def _blueprint_arrays(arrays: dict[str, np.ndarray]) -> dict:
    """Exactly what ``write_particlefield_usd_from_nurec`` feeds the converter."""

    count = int(arrays["positions"].shape[0])
    rest = np.asarray(arrays["features_specular"], dtype=np.float32)
    n_rest = rest.shape[1] // 3
    standard_rest = rest.reshape(count, n_rest, 3).transpose(0, 2, 1).reshape(count, n_rest * 3)
    splat = SplatData(
        count=count,
        xyz=np.asarray(arrays["positions"], dtype=np.float32),
        opacity=np.asarray(arrays["densities"], dtype=np.float32).reshape(count),
        f_dc=np.asarray(arrays["features_albedo"], dtype=np.float32),
        scales=np.asarray(arrays["scales"], dtype=np.float32),
        quats=np.asarray(arrays["rotations"], dtype=np.float32),
        properties=(),
        sh_rest=np.ascontiguousarray(standard_rest),
    )
    return build_particlefield_arrays(
        splat, sh_rest=splat.sh_rest, sh_rest_layout=SH_REST_LAYOUT_INRIA_CHANNEL_MAJOR
    )


def test_reference_revision_is_pinned() -> None:
    assert THREEDGRUT_REFERENCE_REVISION == "a37ef721012dea0f29c0fcfff2d525023b4e854a"


def test_blueprint_private_conversion_matches_upstream_direct_transcode_attributes() -> None:
    arrays = _nurec_arrays()
    reference = threedgrut_lightfield_attributes(arrays, volume_transform_row_major=np.eye(4).tolist())
    candidate = _blueprint_arrays(arrays)

    report = compare_particlefield_attributes(candidate, reference, atol=1e-6)

    assert report["passed"] is True, report
    # SH radiance and positions are bit-identical: the one NuRec->INRIA
    # transpose Blueprint applies is exactly undone by the converter's
    # INRIA->vec3 read, so the coefficient-major layout survives unchanged.
    assert report["attributes"]["sh_coefficients"]["max_abs_diff"] == 0.0
    assert report["attributes"]["positions"]["max_abs_diff"] == 0.0
    assert reference["sh_degree"] == candidate["sh_degree"] == 3
    assert reference["sh_element_size"] == candidate["sh_element_size"] == 16
    # +inf logits are exact alpha 1.0 on both routes.
    assert float(reference["opacities"][:8].min()) == 1.0
    assert float(candidate["opacities"][:8].min()) == 1.0


def test_a_channel_major_misread_of_nurec_specular_is_detected() -> None:
    """PR #1555's defect: treating NuRec triplets as INRIA channel bands."""

    arrays = _nurec_arrays(64)
    reference = threedgrut_lightfield_attributes(arrays)
    count = 64
    rest = np.asarray(arrays["features_specular"], dtype=np.float32)
    # Skip the transpose Blueprint performs: the converter's INRIA read then
    # scrambles coefficients and channels while preserving every scalar.
    wrong_splat = SplatData(
        count=count,
        xyz=np.asarray(arrays["positions"], dtype=np.float32),
        opacity=np.asarray(arrays["densities"], dtype=np.float32).reshape(count),
        f_dc=np.asarray(arrays["features_albedo"], dtype=np.float32),
        scales=np.asarray(arrays["scales"], dtype=np.float32),
        quats=np.asarray(arrays["rotations"], dtype=np.float32),
        properties=(),
        sh_rest=np.ascontiguousarray(rest),
    )
    candidate = build_particlefield_arrays(
        wrong_splat, sh_rest=wrong_splat.sh_rest, sh_rest_layout=SH_REST_LAYOUT_INRIA_CHANNEL_MAJOR
    )
    report = compare_particlefield_attributes(candidate, reference)
    assert report["passed"] is False
    assert report["attributes"]["sh_coefficients"]["passed"] is False
    assert report["attributes"]["positions"]["passed"] is True
    assert sorted(np.asarray(candidate["sh_coefficients"]).ravel()) == pytest.approx(
        sorted(np.asarray(reference["sh_coefficients"]).ravel())
    )


def test_volume_transform_is_applied_to_geometry_only_like_the_upstream_importer() -> None:
    """A non-identity Volume xform moves positions/rotations/scales, not SH."""

    arrays = _nurec_arrays(32)
    theta = np.pi / 2
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    row_major = rotation.T.tolist()  # Gf stores row vectors; p' = p @ M^T
    identity = threedgrut_lightfield_attributes(arrays)
    moved = threedgrut_lightfield_attributes(arrays, volume_transform_row_major=row_major)
    p, r, s = apply_nurec_volume_transform(
        np.asarray(arrays["positions"], np.float32),
        np.asarray(arrays["rotations"], np.float32),
        np.asarray(arrays["scales"], np.float32),
        row_major,
    )
    assert np.allclose(moved["positions"], p)
    assert np.allclose(moved["positions"][:, 2], identity["positions"][:, 2])
    assert not np.allclose(moved["positions"][:, :2], identity["positions"][:, :2])
    assert np.array_equal(moved["sh_coefficients"], identity["sh_coefficients"])
    assert np.allclose(moved["scales"], identity["scales"])
    # This is also why Blueprint's converter refuses a non-identity transform:
    # baking it would require rotating the SH basis, which the upstream
    # importer does not do either (it keeps the transform on the prim).


def test_reference_attribute_digests_are_stable() -> None:
    arrays = _nurec_arrays(16)
    first = attribute_digests(threedgrut_lightfield_attributes(arrays))
    second = attribute_digests(threedgrut_lightfield_attributes(arrays))
    assert first == second
    assert set(first) == {"positions", "scales", "orientations", "opacities", "sh_coefficients"}


def test_known_prim_level_differences_are_declared() -> None:
    assert UPSTREAM_ONLY_PRIM_AUTHORING["sortingModeHint"] == "cameraDistance"
    assert UPSTREAM_ONLY_PRIM_AUTHORING["projectionModeHint"] == "perspective"
    assert UPSTREAM_ONLY_PRIM_AUTHORING["colorSpace:name"] == "srgb_rec709_display"


@pytest.mark.skipif(
    not (_HAS_PXR and _HAS_UPSTREAM_CONVERTER),
    reason="usd-core and usd-convert-gsplat unavailable",
)
def test_full_production_converter_output_matches_upstream_reference(tmp_path: Path) -> None:
    """End to end: sealed NuRec USDZ -> production converter -> attribute parity."""

    from pxr import Usd

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz
    from blueprint_pipeline.common import sha256_file
    from blueprint_pipeline.isaac_nurec_export import validate_transcoded_particlefield
    from blueprint_pipeline.particlefield_usd import write_particlefield_usd_from_nurec

    arrays = _nurec_arrays(256)
    state = {}
    for name, value in arrays.items():
        value = np.ascontiguousarray(value, dtype=np.float16)
        state[f".gaussians_nodes.gaussians.{name}"] = value.tobytes()
        state[f".gaussians_nodes.gaussians.{name}.shape"] = list(value.shape)
    document = {
        "version": "0.2.576",
        "model": "nre",
        "config": {
            "layers": {
                "gaussians": {
                    "precision": 16,
                    "density_activation": "sigmoid",
                    "scale_activation": "exp",
                    "rotation_activation": "normalize",
                    "particle": {"density_kernel_planar": False, "radiance_sph_degree": 3},
                }
            },
            "renderer": {"name": "3dgut-nrend"},
        },
        "state_dict": state,
    }
    source = tmp_path / "configured_appearance.usdz"
    write_aura_nurec_usdz(document, source, payload_name="repaired_scene.nurec")
    out = tmp_path / "scene_appearance.usdc"
    result = write_particlefield_usd_from_nurec(
        source, out, expected_source_sha256=f"sha256:{sha256_file(source)}"
    )
    assert result["status"] == "completed", result

    stage = Usd.Stage.Open(str(out))
    prim = stage.GetPrimAtPath("/World/CapturedScene/Gaussians")
    quats = prim.GetAttribute("orientations").Get()
    candidate = {
        "count": len(quats),
        "positions": np.asarray(prim.GetAttribute("positions").Get()),
        "scales": np.asarray(prim.GetAttribute("scales").Get()),
        "orientations": np.array([[q.GetReal(), *q.GetImaginary()] for q in quats]),
        "opacities": np.asarray(prim.GetAttribute("opacities").Get()),
        "sh_coefficients": np.asarray(
            prim.GetAttribute("radiance:sphericalHarmonicsCoefficients").Get()
        ),
        "sh_degree": prim.GetAttribute("radiance:sphericalHarmonicsDegree").Get(),
        "sh_element_size": prim.GetAttribute(
            "radiance:sphericalHarmonicsCoefficients"
        ).GetMetadata("elementSize"),
    }
    reference = threedgrut_lightfield_attributes(arrays, volume_transform_row_major=np.eye(4).tolist())
    report = compare_particlefield_attributes(candidate, reference, atol=1e-6)
    assert report["passed"] is True, report

    # The production converter's prim carries none of the upstream hints or
    # colour-space metadata; the direct-transcode validator therefore refuses
    # it, which is the documented difference between the two routes.
    validation = validate_transcoded_particlefield(out, expected_gaussian_count=256)
    assert validation["passed"] is False
    assert set(validation["blockers"]) == {
        "nurec_transcode_projection_hint_unexpected",
        "nurec_transcode_sorting_hint_unexpected",
        "nurec_transcode_color_space_unexpected",
    }
    assert validation["attribute_digests"]["sh_coefficients"] == attribute_digests(reference)["sh_coefficients"]
    assert validation["attribute_digests"]["positions"] == attribute_digests(reference)["positions"]
