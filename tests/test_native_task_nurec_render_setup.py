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

from blueprint_pipeline.native_task_nurec_render_setup import (
    BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED,
    BLOCKER_PARTICLEFIELD_TONEMAPPING_UNREADABLE,
    GAUSSIAN_SKIP_TONEMAPPING_SETTING,
    prepare_site_appearance_renderer,
    read_gaussian_skip_tonemapping_setting,
    setup_and_warm_native_nurec_renderer,
)


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
    result = setup_and_warm_native_nurec_renderer(
        app,
        object(),
        setup_for_rendering_factory=_qualified,
        settings_reader=_reader(value),
        require_display_referred_particlefield=True,
    )

    assert result["passed"] is True
    assert result["blockers"] == []
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
