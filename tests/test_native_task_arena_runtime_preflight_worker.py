from __future__ import annotations

import pytest

from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
    _bind_measured_gripper_servo,
    _gripper_pad_geometry_axis_readback,
    _particlefield_stage_readback,
    _prepolicy_visual_gate_from_snapshot,
    _robot_reset_task_space_readback,
)
from blueprint_pipeline.native_task_nurec_render_setup import (
    setup_and_warm_native_nurec_renderer as _official_nurec_render_setup_and_warmup,
)


pxr = pytest.importorskip("pxr")


def test_runtime_preflight_measures_gripper_before_live_pad_readback() -> None:
    calls = []

    class Env:
        def reset(self, *, seed):
            calls.append(("reset", seed))

    measured = {"status": "measured", "blockers": [], "open_command": 0.0}

    def probe(**kwargs):
        calls.append(("probe", kwargs["seed"]))
        return measured

    def servo_factory(**kwargs):
        calls.append(("servo", kwargs["gripper_convention"]))
        return object()

    gripper, servo = _bind_measured_gripper_servo(
        env=Env(),
        robot=object(),
        seed=85423473,
        torch=object(),
        gripper_probe=probe,
        servo_factory=servo_factory,
    )

    assert gripper is measured
    assert servo is not None
    assert calls == [
        ("probe", 85423473),
        ("reset", 85423473),
        ("servo", measured),
    ]


def test_runtime_preflight_refuses_unmeasured_gripper_without_servo() -> None:
    gripper, servo = _bind_measured_gripper_servo(
        env=object(),
        robot=object(),
        seed=1,
        torch=object(),
        gripper_probe=lambda **_kwargs: {
            "status": "blocked",
            "blockers": ["probe_failed"],
        },
        servo_factory=lambda **_kwargs: pytest.fail("servo must not be built"),
    )

    assert gripper["blockers"] == ["probe_failed"]
    assert servo is None


def test_runtime_preflight_applies_visual_gate_to_exact_retained_pngs(
    tmp_path,
) -> None:
    import numpy as np
    from PIL import Image

    cameras = []
    for index, role in enumerate(("external", "wrist", "overview"), start=1):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frame[:, :, index - 1] = np.tile(
            np.linspace(20 + index, 220 - index, 60, dtype=np.uint8),
            (40, 1),
        )
        path = tmp_path / f"{role}.png"
        Image.fromarray(frame).save(path)
        cameras.append(
            {"role": role, "rgb_png": {"path": path.name}}
        )

    result = _prepolicy_visual_gate_from_snapshot(
        snapshot={"cameras": cameras}, output_root=tmp_path
    )

    assert result["passed"] is True
    assert result["candidate_policy_loaded"] is False
    assert result["candidate_policy_queried"] is False


def test_runtime_preflight_visual_gate_refuses_dark_splat_signature(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    cameras = []
    for index, role in enumerate(("external", "wrist", "overview"), start=1):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frame[:8, :, index - 1] = np.tile(
            np.linspace(10, 200, 60, dtype=np.uint8), (8, 1)
        )
        path = tmp_path / f"{role}.png"
        Image.fromarray(frame).save(path)
        cameras.append(
            {"role": role, "rgb_png": {"path": path.name}}
        )

    result = _prepolicy_visual_gate_from_snapshot(
        snapshot={"cameras": cameras}, output_root=tmp_path
    )

    assert result["passed"] is False
    assert any("near_black_fraction_above_ceiling" in value for value in result["blockers"])


def test_gripper_pad_geometry_uses_live_bounds_not_coincident_body_origins() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

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


def _stage(
    *, interpolation: str = "vertex", element_size: int = 16, legacy_material: bool = False
):
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
    if legacy_material:
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
    assert row["material_contract"] == "upstream_native_unbound"
    assert row["material_binding_targets"] == []


def test_live_particlefield_readback_preserves_legacy_replay_compatibility() -> None:
    result = _particlefield_stage_readback(_stage(legacy_material=True))

    assert result["passed"] is True
    assert result["particlefields"][0]["material_contract"] == (
        "legacy_particlefield_emissive"
    )


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
