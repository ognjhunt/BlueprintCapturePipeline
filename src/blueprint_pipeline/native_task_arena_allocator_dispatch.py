"""Small closed dispatch for the Native Task Arena allocator lane that remains."""

from __future__ import annotations

from collections.abc import Callable

from .rigid_robot_free_probe_bundle import PROBE_KIND as RIGID_ROBOT_FREE_PROBE_KIND


def isolated_native_task_arena_lane(probe_kind: str) -> Callable[..., int]:
    """The isolated lane module that owns ``probe_kind``'s whole allocator branch.

    The rigid robot-free probe (beta L5) admits, funds and runs its own single
    capped session. The import stays lazy so the allocator's module load is
    unchanged.
    """

    if probe_kind != RIGID_ROBOT_FREE_PROBE_KIND:
        raise ValueError("isolated_native_task_arena_lane_unknown:" + str(probe_kind))
    from .rigid_robot_free_probe_lane import run_rigid_robot_free_probe_allocator_lane

    return run_rigid_robot_free_probe_allocator_lane


__all__ = ["RIGID_ROBOT_FREE_PROBE_KIND", "isolated_native_task_arena_lane"]
