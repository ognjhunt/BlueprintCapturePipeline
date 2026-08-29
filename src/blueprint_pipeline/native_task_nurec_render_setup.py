"""NVIDIA NuRec renderer setup shared by preflight and native execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


OFFICIAL_NUREC_WARMUP_STEPS = 800


def prepare_site_appearance_renderer(
    *,
    simulation_app: Any,
    plan: Mapping[str, Any],
    stage: Any = None,
    setup_for_rendering_factory: Any = None,
    warmup_steps: int = OFFICIAL_NUREC_WARMUP_STEPS,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
    "OFFICIAL_NUREC_WARMUP_STEPS",
    "prepare_site_appearance_renderer",
    "setup_and_warm_native_nurec_renderer",
]
