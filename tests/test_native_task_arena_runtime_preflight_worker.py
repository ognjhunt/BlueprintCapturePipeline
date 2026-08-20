from __future__ import annotations

import pytest

from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
    _particlefield_stage_readback,
)


pxr = pytest.importorskip("pxr")


def _stage(*, interpolation: str = "vertex", element_size: int = 16):
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim(
        "/World/envs/env_0/scene_appearance/gauss/gauss",
        "ParticleField3DGaussianSplat",
    )
    count = 2
    prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray([Gf.Vec3f()] * count)
    )
    prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(1.0)] * count)
    )
    prim.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray).Set(
        Vt.QuatfArray([Gf.Quatf(1.0)] * count)
    )
    prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray([1.0] * count)
    )
    prim.CreateAttribute(
        "radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int
    ).Set(3)
    sh = prim.CreateAttribute(
        "radiance:sphericalHarmonicsCoefficients",
        Sdf.ValueTypeNames.Float3Array,
    )
    sh.Set(Vt.Vec3fArray([Gf.Vec3f()] * (count * 16)))
    primvar = UsdGeom.Primvar(sh)
    primvar.SetElementSize(element_size)
    primvar.SetInterpolation(interpolation)
    prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(-1.0), Gf.Vec3f(1.0)])
    )
    return stage


def test_live_particlefield_readback_accepts_official_layout() -> None:
    result = _particlefield_stage_readback(_stage())

    assert result["passed"] is True
    assert result["particlefield_prim_count"] == 1
    row = result["particlefields"][0]
    assert row["prim_path"].endswith("/scene_appearance/gauss/gauss")
    assert row["sh_element_size"] == 16
    assert row["sh_interpolation"] == "vertex"
    assert row["sh_coefficient_count"] == 32


@pytest.mark.parametrize(
    ("interpolation", "element_size"),
    [("constant", 16), ("vertex", 1)],
)
def test_live_particlefield_readback_rejects_old_layout(
    interpolation: str, element_size: int
) -> None:
    result = _particlefield_stage_readback(
        _stage(interpolation=interpolation, element_size=element_size)
    )

    assert result["passed"] is False
    assert result["blockers"] == [
        "native_task_arena_particlefield_composition_invalid"
    ]
