"""NVIDIA's NuRec setup cannot repair a splat forced through the tonemapper.

scene-839873 r13 ran ``setup_for_rendering`` (stage classified nurec, not SPG,
so NVIDIA's override set was all ``null`` and wrote nothing) plus 805 warmup
ticks, and still rendered the sealed ParticleField as linear radiance up to
60x display white because ``/rtx/rtpt/gaussian/skipTonemapping/enabled`` had
been forced off.  Omniverse composites ParticleField prims as-is only while
that flag stays at its default, so a display-referred splat must read the
flag back as skipped before any warmup tick or policy frame is spent.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from blueprint_pipeline.native_task_nurec_render_setup import (
    BLOCKER_PARTICLEFIELD_INPUTS_NOT_APPLIED,
    BLOCKER_PARTICLEFIELD_MATERIAL_MISSING,
    BLOCKER_PARTICLEFIELD_PRIM_MISSING,
    BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED,
    BLOCKER_PARTICLEFIELD_TONEMAPPING_UNREADABLE,
    DISPLAY_REFERRED_MATERIAL_INPUTS,
    GAUSSIAN_SKIP_TONEMAPPING_SETTING,
    apply_display_referred_particlefield_material,
    prepare_site_appearance_renderer,
    read_gaussian_skip_tonemapping_setting,
    setup_and_warm_native_nurec_renderer,
)


class _Attr:
    """A settable USD-like attribute; ``sticky=False`` ignores writes."""

    def __init__(self, value=None, *, sticky: bool = True) -> None:
        self.value = value
        self.sticky = sticky

    def Get(self):
        return self.value

    def Set(self, value) -> None:
        if self.sticky:
            self.value = value

    def __bool__(self) -> bool:
        return True


class _Missing:
    def __bool__(self) -> bool:
        return False

    def Get(self):
        return None


class _Prim:
    def __init__(self, path: str, type_name: str, attrs=None, targets=()) -> None:
        self.path = path
        self.type_name = type_name
        self.attrs = dict(attrs or {})
        self.targets = list(targets)

    def GetTypeName(self):
        return self.type_name

    def GetPath(self):
        return self.path

    def GetRelationship(self, _name: str):
        return SimpleNamespace(GetTargets=lambda: list(self.targets))

    def GetAttribute(self, name: str):
        return self.attrs.get(name) or _Missing()

    def CreateAttribute(self, name: str, _type, custom: bool = False):
        self.attrs[name] = _Attr()
        return self.attrs[name]

    def __bool__(self) -> bool:
        return True


class _Stage:
    def __init__(self, *prims: _Prim) -> None:
        self.prims = {prim.path: prim for prim in prims}

    def Traverse(self):
        return list(self.prims.values())

    def GetPrimAtPath(self, path):
        return self.prims.get(str(path)) or _Missing()


def _particlefield_stage(*, shader_attrs=None, source_asset="ParticleFieldEmissive.mdl"):
    field = _Prim(
        "/World/envs/env_0/scene_appearance/CapturedScene/Gaussians",
        "ParticleField3DGaussianSplat",
        targets=["/World/envs/env_0/scene_appearance/CapturedScene/Looks/ParticleFieldEmissive"],
    )
    shader = _Prim(
        "/World/envs/env_0/scene_appearance/CapturedScene/Looks/ParticleFieldEmissive/Shader",
        "Shader",
        attrs={"info:mdl:sourceAsset": _Attr(source_asset), **(shader_attrs or {})},
    )
    return _Stage(field, shader), shader


class _App:
    def __init__(self) -> None:
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


def _qualified(_stage: object) -> tuple[bool, bool, bool, list[str]]:
    return (True, True, False, [])


def _reader(value: object):
    def read(path: str) -> object:
        assert path == GAUSSIAN_SKIP_TONEMAPPING_SETTING
        return value

    return read


def test_a_particlefield_forced_through_the_tonemapper_is_refused_before_any_tick() -> None:
    app = _App()
    result = setup_and_warm_native_nurec_renderer(
        app,
        object(),
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(False),
        require_display_referred_particlefield=True,
    )

    assert result["passed"] is False
    assert result["blockers"] == [BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED]
    assert result["gaussian_skip_tonemapping"] == {
        "setting": GAUSSIAN_SKIP_TONEMAPPING_SETTING,
        "readback": "read",
        "enabled": False,
    }
    assert result["display_referred_particlefield_required"] is True
    assert result["app_update_count"] == 0
    assert app.updates == 0


@pytest.mark.parametrize("value", [None, True, 1])
def test_engine_default_or_explicit_skip_warms_and_records_the_flag(value) -> None:
    app = _App()
    stage, shader = _particlefield_stage()
    result = setup_and_warm_native_nurec_renderer(
        app,
        stage,
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(value),
        require_display_referred_particlefield=True,
    )

    assert result["passed"] is True
    assert result["blockers"] == []
    # A sealed asset with MDL-default inputs is overridden live, before the
    # warmup, and read back.
    assert shader.attrs["inputs:apply_srgb_linear"].Get() is True
    assert shader.attrs["inputs:apply_inverse_tonemap"].Get() is False
    material = result["display_referred_material"]
    assert material["passed"] is True
    assert material["particlefields"][0]["inputs_before"] == {
        "apply_srgb_linear": None,
        "apply_inverse_tonemap": None,
    }
    assert material["particlefields"][0]["inputs_after"] == DISPLAY_REFERRED_MATERIAL_INPUTS
    assert result["gaussian_skip_tonemapping"]["readback"] == "read"
    assert result["gaussian_skip_tonemapping"]["enabled"] is (
        None if value is None else True
    )
    assert result["display_referred_particlefield_required"] is True
    assert result["app_update_count"] == 805
    assert app.updates == 805


def test_an_unreadable_flag_is_a_refusal_for_a_particlefield_not_a_pass() -> None:
    def broken(_path: str) -> object:
        raise RuntimeError("no carb settings in this process")

    app = _App()
    result = setup_and_warm_native_nurec_renderer(
        app,
        object(),
        setup_for_rendering_factory=_qualified,
        settings_reader=broken,
        require_display_referred_particlefield=True,
    )

    assert result["passed"] is False
    assert result["blockers"] == [BLOCKER_PARTICLEFIELD_TONEMAPPING_UNREADABLE]
    assert result["gaussian_skip_tonemapping"] == {
        "setting": GAUSSIAN_SKIP_TONEMAPPING_SETTING,
        "readback": "unavailable",
        "readback_error_type": "RuntimeError",
        "enabled": None,
    }
    assert app.updates == 0


def test_the_default_reader_refuses_when_carb_is_absent_from_the_process() -> None:
    pytest.importorskip("numpy")
    try:
        import carb  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - a Kit process
        pytest.skip("carb is importable here; the live path is exercised on the lane")

    readback = read_gaussian_skip_tonemapping_setting()

    assert readback["readback"] == "unavailable"
    assert readback["enabled"] is None


def test_a_plain_nurec_volume_records_the_flag_without_gating_on_it() -> None:
    app = _App()
    result = setup_and_warm_native_nurec_renderer(
        app,
        object(),
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(False),
    )

    assert result["passed"] is True
    assert result["display_referred_particlefield_required"] is False
    assert result["gaussian_skip_tonemapping"]["enabled"] is False
    assert app.updates == 805


def test_prepare_site_appearance_renderer_gates_only_display_referred_splats() -> None:
    blocked = prepare_site_appearance_renderer(
        simulation_app=_App(),
        plan={
            "appearance_frame_alignment": {
                "representation": "particlefield_3d_gaussian_splat"
            }
        },
        stage=object(),
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(False),
    )
    assert blocked["passed"] is False
    assert blocked["representation"] == "particlefield_3d_gaussian_splat"
    assert blocked["blockers"] == [BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED]

    volume = prepare_site_appearance_renderer(
        simulation_app=_App(),
        plan={"appearance_frame_alignment": {"representation": "nurec_volume"}},
        stage=object(),
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(False),
        warmup_steps=40,
    )
    assert volume["passed"] is True
    assert volume["display_referred_particlefield_required"] is False


# --- display-referred material override -----------------------------------


def test_the_shader_override_precedes_every_warmup_tick_and_is_read_back() -> None:
    stage, shader = _particlefield_stage(
        shader_attrs={"inputs:apply_srgb_linear": _Attr(False)}
    )

    receipt = apply_display_referred_particlefield_material(stage)

    assert receipt["passed"] is True
    assert receipt["inputs"] == {"apply_srgb_linear": True, "apply_inverse_tonemap": False}
    row = receipt["particlefields"][0]
    assert row["shader_source_asset"] == "ParticleFieldEmissive.mdl"
    assert row["inputs_before"] == {"apply_srgb_linear": False, "apply_inverse_tonemap": None}
    assert row["inputs_after"] == {"apply_srgb_linear": True, "apply_inverse_tonemap": False}
    assert shader.attrs["inputs:apply_srgb_linear"].Get() is True


def test_a_particlefield_without_the_emissive_material_is_refused() -> None:
    stage, _ = _particlefield_stage(source_asset="OmniPBR.mdl")
    app = _App()

    result = setup_and_warm_native_nurec_renderer(
        app,
        stage,
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(True),
        require_display_referred_particlefield=True,
    )

    assert result["passed"] is False
    assert result["blockers"] == [BLOCKER_PARTICLEFIELD_MATERIAL_MISSING]
    assert result["display_referred_material"]["passed"] is False
    assert app.updates == 0


def test_an_override_that_does_not_read_back_is_refused() -> None:
    stage, _ = _particlefield_stage(
        shader_attrs={"inputs:apply_srgb_linear": _Attr(False, sticky=False)}
    )
    app = _App()

    result = setup_and_warm_native_nurec_renderer(
        app,
        stage,
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(True),
        require_display_referred_particlefield=True,
    )

    assert result["passed"] is False
    assert result["blockers"] == [BLOCKER_PARTICLEFIELD_INPUTS_NOT_APPLIED]
    assert app.updates == 0


def test_a_stage_with_no_particlefield_cannot_claim_a_display_referred_splat() -> None:
    receipt = apply_display_referred_particlefield_material(_Stage())

    assert receipt["passed"] is False
    assert receipt["blockers"] == [BLOCKER_PARTICLEFIELD_PRIM_MISSING]


def test_a_plain_nurec_volume_stage_is_not_touched() -> None:
    app = _App()
    result = setup_and_warm_native_nurec_renderer(
        app,
        _Stage(),
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(True),
    )

    assert result["passed"] is True
    assert result["display_referred_material"] is None
    assert app.updates == 805


def test_writer_and_runtime_agree_on_the_display_referred_inputs() -> None:
    from blueprint_pipeline.particlefield_usd import (
        PARTICLEFIELD_DISPLAY_REFERRED_MATERIAL_INPUTS,
    )

    assert PARTICLEFIELD_DISPLAY_REFERRED_MATERIAL_INPUTS == DISPLAY_REFERRED_MATERIAL_INPUTS


def test_a_sealed_pxr_asset_with_default_inputs_is_overridden_on_the_live_stage() -> None:
    pxr = pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    field = stage.DefinePrim("/World/scene_appearance/Gaussians", "ParticleField3DGaussianSplat")
    material = stage.DefinePrim("/World/scene_appearance/Looks/ParticleFieldEmissive", "Material")
    shader = stage.DefinePrim(f"{material.GetPath()}/Shader", "Shader")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        "ParticleFieldEmissive.mdl"
    )
    field.CreateRelationship("material:binding").SetTargets([material.GetPath()])
    del pxr

    receipt = apply_display_referred_particlefield_material(stage)

    assert receipt["passed"] is True, receipt
    assert shader.GetAttribute("inputs:apply_srgb_linear").Get() is True
    assert shader.GetAttribute("inputs:apply_inverse_tonemap").Get() is False
    assert receipt["particlefields"][0]["prim_path"] == "/World/scene_appearance/Gaussians"
