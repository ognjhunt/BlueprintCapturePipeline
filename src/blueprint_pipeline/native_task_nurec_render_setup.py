"""NVIDIA NuRec renderer setup shared by preflight and native execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


OFFICIAL_NUREC_WARMUP_STEPS = 800

#: Omniverse RTX composites ParticleField prims "as-is": their light fields
#: are display-referred sRGB and the tonemapping pipeline is skipped for them
#: (Omniverse Materials and Rendering, "Gaussian Splats (Particle Fields)").
#: NVIDIA's shipped ``nurec_config.yaml`` forces this flag off only for PPISP
#: (``info:spg:sourceAsset``) stages; its plain-gaussian override set is all
#: ``null`` and ``apply_carb_overrides`` skips ``None``, so the official setup
#: writes nothing for a plain splat and cannot restore the flag either.  When
#: the flag is forced off anyway (a launch argument, or Isaac Lab's camera
#: whenever ``rgb_hdr`` or an ISP is requested) the splat is rendered as linear
#: radiance up to 60x display white and the LDR annotator clamps it per
#: channel into white blobs with chromatic fringes.  scene-839873 r13 measured
#: 22-24 percent of world-camera pixels above 1.0 after this exact setup and
#: 805 warmup ticks, so the warmup never repairs it; the setting must read
#: back skipped before a display-referred splat is rendered for a policy.
GAUSSIAN_SKIP_TONEMAPPING_SETTING = "/rtx/rtpt/gaussian/skipTonemapping/enabled"
DISPLAY_REFERRED_REPRESENTATIONS = frozenset({"particlefield_3d_gaussian_splat"})
BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED = (
    "native_task_arena_particlefield_tonemapping_not_skipped"
)
BLOCKER_PARTICLEFIELD_TONEMAPPING_UNREADABLE = (
    "native_task_arena_particlefield_tonemapping_setting_unreadable"
)


def read_gaussian_skip_tonemapping_setting(
    settings_reader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Read the live carb value; ``None`` is the engine default (skipped)."""

    if settings_reader is None:

        def settings_reader(path: str) -> Any:
            import carb

            return carb.settings.get_settings().get(path)

    try:
        value = settings_reader(GAUSSIAN_SKIP_TONEMAPPING_SETTING)
    except Exception as exc:  # noqa: BLE001 - retained diagnostic boundary
        return {
            "setting": GAUSSIAN_SKIP_TONEMAPPING_SETTING,
            "readback": "unavailable",
            "readback_error_type": type(exc).__name__,
            "enabled": None,
        }
    if value is not None and not isinstance(value, bool):
        # carb may hand an int back for a bool setting.
        value = bool(value)
    return {
        "setting": GAUSSIAN_SKIP_TONEMAPPING_SETTING,
        "readback": "read",
        "enabled": value,
    }


def particlefield_display_referred_blockers(
    readback: Mapping[str, Any],
) -> list[str]:
    """Blockers for a display-referred splat given the flag readback."""

    if readback.get("readback") != "read":
        return [BLOCKER_PARTICLEFIELD_TONEMAPPING_UNREADABLE]
    if readback.get("enabled") is False:
        return [BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED]
    return []


def prepare_site_appearance_renderer(
    *,
    simulation_app: Any,
    plan: Mapping[str, Any],
    stage: Any = None,
    setup_for_rendering_factory: Any = None,
    warmup_steps: int = OFFICIAL_NUREC_WARMUP_STEPS,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    settings_reader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run NVIDIA's accumulation path only for supported NuRec appearances."""

    representation = str(
        (plan.get("appearance_frame_alignment") or {}).get("representation") or ""
    )
    if representation not in {"nurec_volume", "particlefield_3d_gaussian_splat"}:
        return {
            "schema_version": "native_task_arena_nurec_warmup.v1",
            "status": "not_required",
            "representation": representation or None,
            "passed": True,
            "blockers": [],
        }
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    result = setup_and_warm_native_nurec_renderer(
        simulation_app,
        stage,
        warmup_steps=warmup_steps,
        setup_for_rendering_factory=setup_for_rendering_factory,
        progress_callback=progress_callback,
        settings_reader=settings_reader,
        require_display_referred_particlefield=(
            representation in DISPLAY_REFERRED_REPRESENTATIONS
        ),
    )
    result["representation"] = representation
    return result


def setup_and_warm_native_nurec_renderer(
    simulation_app: Any,
    stage: Any,
    *,
    warmup_steps: int = OFFICIAL_NUREC_WARMUP_STEPS,
    setup_for_rendering_factory: Any = None,
    progress_callback: Any = None,
    settings_reader: Callable[[str], Any] | None = None,
    require_display_referred_particlefield: bool = False,
) -> dict[str, Any]:
    """Apply NVIDIA's shipped NuRec setup and accumulation procedure."""

    if (
        isinstance(warmup_steps, bool)
        or int(warmup_steps) < 40
        or int(warmup_steps) > 2_000
    ):
        return {
            "schema_version": "native_task_arena_nurec_warmup.v1",
            "passed": False,
            "blockers": ["native_task_arena_nurec_warmup_steps_invalid"],
        }
    try:
        if setup_for_rendering_factory is None:
            from isaacsim.replicator.nurec_utils.rendering_setup import (
                setup_for_rendering,
            )

            setup_for_rendering_factory = setup_for_rendering
        success, nurec, spg, problems = setup_for_rendering_factory(stage)
    except Exception as exc:  # noqa: BLE001 - retained diagnostic boundary
        return {
            "schema_version": "native_task_arena_nurec_warmup.v1",
            "passed": False,
            "blockers": [
                "native_task_arena_nurec_official_setup_failed:"
                f"{type(exc).__name__}"
            ],
        }
    if not success or not nurec:
        return {
            "schema_version": "native_task_arena_nurec_warmup.v1",
            "official_setup_success": bool(success),
            "stage_classified_nurec": bool(nurec),
            "stage_classified_spg": bool(spg),
            "official_setup_problems": list(problems or []),
            "passed": False,
            "blockers": ["native_task_arena_nurec_official_setup_not_qualified"],
        }

    # Read the flag after NVIDIA's overrides so the receipt shows the value the
    # first rendered frame will actually use.  A display-referred splat is
    # refused before a single warmup tick is spent when the flag is forced off.
    display_referred_required = require_display_referred_particlefield is True
    tonemapping = read_gaussian_skip_tonemapping_setting(settings_reader)
    tonemapping_blockers = (
        particlefield_display_referred_blockers(tonemapping)
        if display_referred_required
        else []
    )
    if tonemapping_blockers:
        return {
            "schema_version": "native_task_arena_nurec_warmup.v1",
            "official_setup_success": True,
            "stage_classified_nurec": True,
            "stage_classified_spg": bool(spg),
            "official_setup_problems": [],
            "gaussian_skip_tonemapping": tonemapping,
            "display_referred_particlefield_required": True,
            "warmup_app_update_count": 0,
            "app_update_count": 0,
            "passed": False,
            "blockers": tonemapping_blockers,
        }

    attempts = 8
    updates_per_attempt = max(int(warmup_steps) // attempts, 5)
    warmup_update_count = 0
    prime_update_count = 0
    # Isaac Lab owns the camera annotators and has no Replicator trigger graph.
    # NVIDIA's pinned camera test advances this path with application ticks.
    for _ in range(5):
        simulation_app.update()
        prime_update_count += 1
    if progress_callback is not None:
        progress_callback(
            {
                "round": 0,
                "prime_updates_completed": prime_update_count,
                "warmup_updates_completed": warmup_update_count,
            }
        )
    for attempt in range(attempts):
        for _ in range(updates_per_attempt):
            simulation_app.update()
            warmup_update_count += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "round": attempt + 1,
                    "prime_updates_completed": prime_update_count,
                    "warmup_updates_completed": warmup_update_count,
                }
            )
    return {
        "schema_version": "native_task_arena_nurec_warmup.v1",
        "official_setup_success": True,
        "stage_classified_nurec": True,
        "stage_classified_spg": bool(spg),
        "official_setup_problems": [],
        "gaussian_skip_tonemapping": tonemapping,
        "display_referred_particlefield_required": display_referred_required,
        "requested_warmup_steps": int(warmup_steps),
        "orchestrator_attempts": 0,
        "orchestrator_error_types": [],
        "prime_app_update_count": prime_update_count,
        "warmup_app_update_count": warmup_update_count,
        "app_update_count": prime_update_count + warmup_update_count,
        "procedure_sources": [
            (
                "isaac-sim/IsaacSim:source/standalone_examples/nurec/"
                "nurec_render.py@987015050efebfd0cd5d3736ae47fffe5adee308"
            ),
            (
                "isaac-sim/IsaacLab:source/isaaclab/test/sensors/"
                "test_camera_ppisp_gaussian.py@"
                "ffff603eafc6b74264a5261cc0183d6a65390d78"
            ),
        ],
        "camera_warmup_method": (
            "isaaclab_camera_app_updates_without_replicator_orchestrator"
        ),
        "passed": warmup_update_count >= int(warmup_steps),
        "blockers": [],
    }


__all__ = [
    "BLOCKER_PARTICLEFIELD_TONEMAPPING_NOT_SKIPPED",
    "BLOCKER_PARTICLEFIELD_TONEMAPPING_UNREADABLE",
    "DISPLAY_REFERRED_REPRESENTATIONS",
    "GAUSSIAN_SKIP_TONEMAPPING_SETTING",
    "OFFICIAL_NUREC_WARMUP_STEPS",
    "particlefield_display_referred_blockers",
    "prepare_site_appearance_renderer",
    "read_gaussian_skip_tonemapping_setting",
    "setup_and_warm_native_nurec_renderer",
]
