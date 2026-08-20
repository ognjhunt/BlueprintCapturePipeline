from __future__ import annotations

import pytest

from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
    _gripper_pad_geometry_axis_readback,
    _official_nurec_render_setup_and_warmup,
    _particlefield_stage_readback,
    _robot_reset_task_space_readback,
)


pxr = pytest.importorskip("pxr")


def test_gripper_pad_geometry_uses_live_bounds_not_coincident_body_origins() -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    for side, y in (("left", 0.04), ("right", -0.04)):
        parent = UsdGeom.Xform.Define(
            stage,
            f"/World/envs/env_0/Robot/Gripper/{side}_inner_finger",
        )
        parent.AddTranslateOp().Set(Gf.Vec3d(0.0, y, 0.12))
        cube = UsdGeom.Cube.Define(stage, f"{parent.GetPath()}/pad_geometry")
        cube.CreateSizeAttr(0.02)

    result = _gripper_pad_geometry_axis_readback(
        stage=stage,
        body_axis_readback={
            "controlled_body_name": "base_link",
            "measured": {
                "controlled_body_position_world_m": [0.0, 0.0, 0.0],
                "controlled_body_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    )

    assert result["passed"] is True
    axis = result["axis_readback"]
    assert axis["measured"]["finger_separation_m"] == pytest.approx(0.08)
    assert axis["measured"]["body_origin_to_finger_midpoint_m"] == pytest.approx(
        0.12
    )
    assert abs(axis["derived"]["jaw_approach_orthogonality_dot"]) < 1.0e-9


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
    material_path = "/World/envs/env_0/scene_appearance/gauss/Looks/ParticleFieldEmissive"
    material = stage.DefinePrim(material_path, "Material")
    shader = stage.DefinePrim(f"{material_path}/Shader", "Shader")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        "ParticleFieldEmissive.mdl"
    )
    shader.CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
    ).Set("ParticleFieldEmissive")
    prim.CreateRelationship("material:binding").SetTargets([material.GetPath()])
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
    assert row["material_shader_source_asset"] == "ParticleFieldEmissive.mdl"


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


def test_official_nurec_setup_runs_the_full_800_update_warmup() -> None:
    class App:
        updates = 0

        def update(self):
            self.updates += 1

    app = App()
    progress = []
    result = _official_nurec_render_setup_and_warmup(
        app,
        object(),
        setup_for_rendering_factory=lambda _stage: (True, True, False, []),
        progress_callback=progress.append,
    )

    assert result["passed"] is True
    assert result["stage_classified_nurec"] is True
    assert result["stage_classified_spg"] is False
    assert result["prime_app_update_count"] == 5
    assert result["warmup_app_update_count"] == 800
    assert result["app_update_count"] == 805
    assert app.updates == 805
    assert result["orchestrator_attempts"] == 0
    assert result["camera_warmup_method"] == (
        "isaaclab_camera_app_updates_without_replicator_orchestrator"
    )
    assert [row["round"] for row in progress] == list(range(9))
    assert [row["warmup_updates_completed"] for row in progress] == [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
    ]


def test_official_nurec_setup_refuses_a_non_nurec_stage() -> None:
    result = _official_nurec_render_setup_and_warmup(
        object(),
        object(),
        setup_for_rendering_factory=lambda _stage: (True, False, False, []),
    )

    assert result["passed"] is False
    assert result["blockers"] == [
        "native_task_arena_nurec_official_setup_not_qualified"
    ]


def _reset_task_space_result(midpoint) -> dict:
    return _robot_reset_task_space_readback(
        plan={
            "robot": {
                "base_pose_world": {
                    "position_world_m": [3.7634863, 8.906664, 0.090782],
                    "orientation_xyzw": [0.0, 0.0, 2**-0.5, 2**-0.5],
                }
            }
        },
        gripper_frame_axis_readback={
            "measured": {"finger_midpoint_world_m": midpoint}
        },
        object_reset_readback={
            "task_link_frame_equivalence": {
                "observed_contact_position_world_m": [
                    3.7634863,
                    9.456664,
                    0.405,
                ]
            }
        },
    )


def test_robot_reset_task_space_accepts_fingers_above_and_in_front() -> None:
    result = _reset_task_space_result([3.7634863, 9.20, 0.50])

    assert result["passed"] is True
    assert result["finger_forward_projection_m"] > 0.0
    assert result["approach_standoff_position_world_m"] == pytest.approx(
        [3.7634863, 9.336664, 0.405]
    )


def test_robot_reset_task_space_rejects_the_prior_behind_floor_pose() -> None:
    result = _reset_task_space_result([3.875189, 8.742135, 0.090626])

    assert result["passed"] is False
    assert result["checks"]["finger_midpoint_above_floor"] is False
    assert result["checks"]["finger_midpoint_in_front_of_base"] is False
