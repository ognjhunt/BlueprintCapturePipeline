"""How wide a base the reconstructed floor near a website job can stand, and where, for the robot fit table.

The standard station is a 0.3 m square (``team_scene_packet.STATION_FOOTPRINT_HALF_EXTENT_M``). Rather than
one search per robot whose published base is larger, the table asks the scene one question: the widest
square base that stands clear near the task, searched the way the standard station is
(``website_rigid_feasibility.carry_station_search``: the same mount height, standoff, clearance margin
and task-point ranking; only the footprint grows). Each robot is then decided by arithmetic, so the
cost is a handful of CPU searches per job whatever the number of robots. The table runs it only when
standard stations were found and some robot needs it: a base the standard station may not hold, or,
when the capability verdict recorded no floor height, a robot judged by its hand heights.

The answer is in scene units; the scene's true scale lies in the preparation's band, so the floor it
offers is the found width times the band. It is bisected to ``RESOLUTION_M`` of half extent, at most
``MAX_SEARCHES`` searches and within ``BUDGET_S`` of wall-clock time, checked before each search. Each
clear search keeps where it stood (``ground_stance`` ranks by reach to the task points), so a fixed arm's
reach is measured from where its own base can stand. A miss on a scanned room's own mesh may be a hole
in the reconstructed floor, so ``upper_hole_possible`` says the floor may offer more than found.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

#: The bisection stops when the clear and refused half extents are this close (half the search grid).
RESOLUTION_M = 0.025
#: At most this many searches per job, the first at the largest half extent any robot needs.
MAX_SEARCHES = 7
#: Wall-clock budget for the whole search, checked before each search. On the toy room a search takes
#: about 0.1 s; a website collider read once into the scope's binary copy is searched in seconds, so
#: 20 s holds the bisection on a typical scene and caps what a large one adds to the handoff.
BUDGET_S = 20.0
#: The stance search's own grid (``team_embodiment_stance.GRID_STEP_M``): a refusal on the grid does not
#: prove that no square stands between its points, so a fail allows one step of slack.
GRID_STEP_M = 0.05
#: Blockers that mean a refusal may come from floor the reconstruction never observed.
HOLE_BLOCKERS = frozenset({"team_embodiment_room_mesh_no_clear_stance"})
#: The refusals that mean no stance of this size stands clear; any other (no floor found, say) is not an
#: answer about the floor's width, so the search stops and the robot's stance stays unsearched.
NO_STANCE_BLOCKERS = HOLE_BLOCKERS | {"team_embodiment_free_base_no_clear_stance",
                                      "team_embodiment_task_motion_no_clear_stance"}


class FloorNotSearched(Exception):
    """The search could not answer how wide a base the floor stands."""


def widest_clear_square(*, preparation: Mapping[str, Any], collision_path: Path, collision_sha256: str,
                        known_clear_half_m: float, needed_half_m: float, want_floor: bool = True,
                        search: Any = None, clock: Callable[[], float] = time.monotonic,
                        budget_s: float = BUDGET_S) -> dict[str, Any]:
    """The widest clear square base, as a bound on its half extent in scene units, and where each stood.

    ``known_clear_half_m`` stands clear already (the standard station); ``needed_half_m`` is the largest
    half extent any robot could need. With ``want_floor``, one more search at the standard station finds
    the floor's height when no clear search did. Past ``budget_s`` it stops: ``budget_exhausted`` is
    true and the bounds are not an answer. ``search`` and ``clock`` stand in for ``ground_stance`` and
    the clock in tests. The elapsed time is logged, never returned: the table that holds this is digested.
    """
    from .team_embodiment import TeamEmbodimentError
    from .website_rigid_feasibility import carry_station_search

    if search is None:
        from .team_embodiment_stance import ground_stance as search
    arguments = carry_station_search(preparation)
    arguments.pop("ranked", None)  # one clear stance answers whether this base fits

    started = clock()
    searches: list[dict[str, Any]] = []
    floor: list[float] = []
    exhausted: list[bool] = []

    def clear(half: float) -> bool | None:
        if clock() - started > budget_s:
            exhausted.append(True)
            return None
        try:
            found = search(collision_path=collision_path, collision_sha256=collision_sha256,
                           **{**arguments, "footprint_half_extent_m": [half, half]})
        except TeamEmbodimentError as exc:
            if not set(exc.blockers) <= NO_STANCE_BLOCKERS:
                raise FloorNotSearched(",".join(sorted(exc.blockers))) from exc
            searches.append({"half_extent_m": round(half, 6), "clear": False, "blockers": sorted(exc.blockers)})
            return False
        row: dict[str, Any] = {"half_extent_m": round(half, 6), "clear": True}
        if isinstance(found, Mapping):
            position = found.get("position_world_m")
            if isinstance(position, Sequence) and len(position) == 3:
                row["position_world_m"] = [round(float(v), 6) for v in position]
            if isinstance(found.get("floor_z_m"), (int, float)):
                floor.append(float(found["floor_z_m"]))  # the floor the search stood on, for hand heights
        searches.append(row)
        return True

    lower, upper = float(known_clear_half_m), float(needed_half_m)
    first = None if upper <= lower else clear(upper)
    if upper <= lower or first:
        lower, upper = max(lower, upper), None
    elif first is False:
        while upper - lower > RESOLUTION_M and len(searches) < MAX_SEARCHES:
            middle = (lower + upper) / 2.0
            answer = clear(middle)
            if answer is None:
                break
            if answer:
                lower = middle
            else:
                upper = middle
    if want_floor and not floor:
        clear(float(known_clear_half_m))  # one more search, at the standard station, for the floor's height
    LOGGER.info("website_robot_fit_floor_search searches=%d seconds=%.3f budget_exhausted=%s",
                len(searches), clock() - started, bool(exhausted))
    refused_at = [row for row in searches if not row["clear"] and upper is not None
                  and row["half_extent_m"] == round(upper, 6)]
    return {"floor_z_m": round(floor[0], 6) if floor else None, "lower_half_m": round(lower, 6),
            "upper_half_m": None if upper is None else round(upper, 6),
            "upper_hole_possible": bool(refused_at and HOLE_BLOCKERS & set(refused_at[0]["blockers"])),
            "budget_exhausted": bool(exhausted), "searches": searches,
            "rule": "widest_square_base_clear_near_the_task_with_the_standard_station_clearance"}


def floor_stance(sizes: Sequence[Sequence[float]], floor: Mapping[str, Any], band: Sequence[float]) -> dict[str, Any]:
    """Whether a base of the published ``sizes`` (each [length, width], m) stands on the floor ``floor`` offers.

    ``pass`` when its longest side fits even at the smaller scale; ``fail`` when even its shortest side,
    less one search-grid step of slack, exceeds the widest floor at the larger scale and no hole in the
    reconstruction could hide more; ``conditional`` when the floor the job seems to offer straddles its
    longest side; ``not_found`` otherwise (the widest floor found is narrower than the base at every
    scale, but that is not proven). ``positions`` are where a square at least that wide stood, in scene
    coordinates: for a pass, every clear search that holds the base at the smaller scale; for a
    conditional one, the widest found, which holds it whenever the condition holds.
    """
    longest = max(max(size) for size in sizes)
    shortest = min(min(size) for size in sizes)
    lower, upper = float(floor["lower_half_m"]), floor["upper_half_m"]
    offered_min = 2.0 * lower * band[0]
    offered_max = None if upper is None else 2.0 * float(upper) * band[1]
    clear_rows = [row for row in floor["searches"] if row["clear"] and "position_world_m" in row]

    def where(half: float) -> list[list[float]]:
        return [row["position_world_m"] for row in clear_rows if row["half_extent_m"] >= half - 1e-9]

    # Clear at the largest half extent searched, which every base needs at most at the smaller scale.
    if upper is None or offered_min >= longest - 1e-9:
        return {"result": "pass", "base_m": longest, "offered_m": [offered_min, offered_max],
                "positions": where(longest / 2.0 / band[0])}
    if not floor["upper_hole_possible"] and offered_max + GRID_STEP_M * band[1] < shortest:
        return {"result": "fail", "base_m": shortest, "offered_m": [offered_min, offered_max], "positions": []}
    if offered_max >= longest:
        return {"result": "conditional", "base_m": longest, "offered_m": [offered_min, offered_max],
                "positions": where(lower)}
    return {"result": "not_found", "base_m": longest, "offered_m": [offered_min, offered_max], "positions": []}


__all__ = ["BUDGET_S", "FloorNotSearched", "GRID_STEP_M", "MAX_SEARCHES", "RESOLUTION_M", "floor_stance",
           "widest_clear_square"]
