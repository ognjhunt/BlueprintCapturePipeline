from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz
from blueprint_pipeline.common import sha256_file
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.nurec_volume_codec import build_state_dict
from blueprint_pipeline.nvidia_3dgrut_particlefield_transcode import (
    AUTHORING_IMPLEMENTATION,
    COLOR_SPACE,
    SORTING_MODE_HINT,
    validate_direct_particlefield,
    write_direct_particlefield_from_nurec,
    write_direct_particlefield_from_ply,
)


pxr = pytest.importorskip("pxr")


def _source(tmp_path: Path, *, count: int = 4) -> Path:
    rng = np.random.default_rng(839873)
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
                "positions": rng.normal(size=(count, 3)).astype(np.float32),
                "rotations": np.tile(
                    np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    (count, 1),
                ),
                "scales": np.full((count, 3), -2.0, dtype=np.float32),
                "densities": np.zeros((count, 1), dtype=np.float32),
                "features_albedo": np.zeros((count, 3), dtype=np.float32),
                "features_specular": np.zeros((count, 45), dtype=np.float32),
            },
            precision=32,
        ),
    }
    source = tmp_path / "source.usdz"
    write_aura_nurec_usdz(document, source)
    return source


def _fake_direct_transcode(source, output, **kwargs) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdVol, Vt

    assert Path(source).is_file()
    assert kwargs == {
        "output_format": "lightfield",
        "render_order_hint": "cameraDistance",
        "validate_usd": True,
    }
    count = 4
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    field = UsdVol.ParticleField3DGaussianSplat.Define(
        stage, "/World/Gaussians/gaussians"
    )
    field.CreatePositionsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(index), 0.0, 0.0) for index in range(count)])
    )
    field.CreateOrientationsAttr().Set(
        Vt.QuatfArray([Gf.Quatf(1.0)] * count)
    )
    field.CreateScalesAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.1)] * count)
    )
    field.CreateOpacitiesAttr().Set(Vt.FloatArray([0.5] * count))
    field.CreateRadianceSphericalHarmonicsDegreeAttr().Set(3)
    coefficients = field.CreateRadianceSphericalHarmonicsCoefficientsAttr()
    coefficients.Set(Vt.Vec3fArray([Gf.Vec3f()] * (count * 16)))
    coefficients.SetMetadata("elementSize", 16)
    field.CreateExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.0), Gf.Vec3f(3.0, 0.0, 0.0)])
    )
    field.CreateProjectionModeHintAttr().Set("perspective")
    field.CreateSortingModeHintAttr().Set("cameraDistance")
    Usd.ColorSpaceAPI.Apply(field.GetPrim()).CreateColorSpaceNameAttr().Set(
        "srgb_rec709_display"
    )
    stage.GetRootLayer().Save()


def _ply_source(tmp_path: Path, *, count: int = 4) -> Path:
    names = [
        "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
        "scale_0", "scale_1", "scale_2", "opacity",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    source = tmp_path / "source.ply"
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
        *(f"property float {name}" for name in names),
        "end_header",
        "",
    ]
    rows = np.zeros((count, len(names)), dtype="<f4")
    source.write_bytes("\n".join(header).encode("ascii") + rows.tobytes())
    return source


def test_direct_transcode_seals_upstream_lightfield_contract(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "direct.usdc"
    receipt_path = tmp_path / "receipt.json"

    result = write_direct_particlefield_from_nurec(
        source,
        output,
        expected_source_sha256="sha256:" + sha256_file(source),
        receipt_path=receipt_path,
        transcode_runner=_fake_direct_transcode,
    )

    assert result["status"] == "completed"
    assert result["particlefield_authoring_implementation"] == AUTHORING_IMPLEMENTATION
    assert result["upstream_sorting_mode_hint"] == SORTING_MODE_HINT
    assert result["upstream_color_space"] == COLOR_SPACE
    assert result["sh_primvar_interpolation"] == "constant"
    assert result["splat_count"] == 4
    assert result["provider_mutation_performed"] is False
    assert result["receipt_digest"] == canonical_digest(
        result, digest_field="receipt_digest"
    )
    assert json.loads(receipt_path.read_text()) == result
    assert validate_direct_particlefield(output)["output_sha256"] == result[
        "output_sha256"
    ]


def test_direct_transcode_refuses_legacy_unhinted_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "legacy.usdc"

    def unhinted(source, output, **_kwargs):
        _fake_direct_transcode(
            source,
            output,
            output_format="lightfield",
            render_order_hint="cameraDistance",
            validate_usd=True,
        )
        stage = pxr.Usd.Stage.Open(str(output))
        field = stage.GetPrimAtPath("/World/Gaussians/gaussians")
        field.GetAttribute("sortingModeHint").Clear()
        stage.GetRootLayer().Save()

    result = write_direct_particlefield_from_nurec(
        source,
        output,
        expected_source_sha256="sha256:" + sha256_file(source),
        transcode_runner=unhinted,
    )

    assert result["status"] == "blocked"
    assert "nvidia_3dgrut_particlefield_contract_invalid" in result["blockers"][0]
    assert not output.exists()


def test_direct_ply_transcode_seals_distinct_source_contract(tmp_path: Path) -> None:
    source = _ply_source(tmp_path)
    output = tmp_path / "direct-from-ply.usdc"

    result = write_direct_particlefield_from_ply(
        source,
        output,
        expected_source_sha256="sha256:" + sha256_file(source),
        transcode_runner=_fake_direct_transcode,
    )

    assert result["status"] == "completed"
    assert result["source_kind"] == "standard_3dgs_ply"
    assert result["source_vertex_count"] == 4
    assert result["splat_count"] == 4
    assert result["exact_learned_arrays_preserved"] is True
    assert result["receipt_digest"] == canonical_digest(
        result, digest_field="receipt_digest"
    )


def test_direct_ply_transcode_refuses_nonstandard_header(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_text("ply\nformat ascii 1.0\nelement vertex 4\nend_header\n")

    result = write_direct_particlefield_from_ply(
        source,
        tmp_path / "output.usdc",
        expected_source_sha256="sha256:" + sha256_file(source),
        transcode_runner=_fake_direct_transcode,
    )

    assert result["status"] == "blocked"
    assert "nvidia_3dgrut_standard_ply_contract_invalid" in result["blockers"][0]
