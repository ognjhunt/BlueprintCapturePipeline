"""How wide a base the reconstructed floor near a website job can stand, for the robot fit table.

The standard station is a 0.3 m square (``team_scene_packet.STATION_FOOTPRINT_HALF_EXTENT_M``); a robot
whose published base is larger was never searched for. Rather than one search per robot, the table asks
the scene one question: the widest square base that stands clear near the task, searched the way the
standard station is (``website_rigid_feasibility.carry_station_search``: the same mount height, standoff
and clearance margin; only the footprint grows). Each robot is then decided by arithmetic, so the cost
is a handful of CPU searches per job whatever the number of robots, and none when no robot needs it.

The answer is in scene units; the scene's true scale lies in the preparation's band, so the floor it
offers is the found width times the band. The search's own step bounds the answer from both sides
(``lower_m`` stands clear, ``upper_m`` does not): it is bisected to ``RESOLUTION_M`` of half extent, at
most ``MAX_SEARCHES`` searches. A miss is refused by name; a miss on a scanned room's own mesh may be a
hole in the reconstructed floor, so ``upper_hole_possible`` says the floor may offer more than found.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The bisection stops when the clear and refused half extents are this close (half the search grid).
RESOLUTION_M = 0.025
#: At most this many searches per job, the first at the largest half extent any robot needs.
MAX_SEARCHES = 7
#: Blockers that mean a refusal may come from floor the reconstruction never observed.
HOLE_BLOCKERS = frozenset({"team_embodiment_room_mesh_no_clear_stance"})
#: The refusals that mean no stance of this size stands clear; any other (no floor found, say) is not an
#: answer about the floor's width, so the search stops and the robot's stance stays unsearched.
NO_STANCE_BLOCKERS = HOLE_BLOCKERS | {"team_embodiment_free_base_no_clear_stance",
                                      "team_embodiment_task_motion_no_clear_stance"}


class FloorNotSearched(Exception):
    """The search could not answer how wide a base the floor stands."""


def widest_clear_square(*, preparation: Mapping[str, Any], collision_path: Path, collision_sha256: str,
                        known_clear_half_m: float, needed_half_m: float,
                        search: Any = None) -> dict[str, Any]:
    """The widest clear square base, as a bound on its half extent in scene units.

    ``known_clear_half_m`` stands clear already (the standard station); ``needed_half_m`` is the largest
    half extent any robot could need. ``search`` stands in for ``ground_stance`` in tests.
    """
    from .team_embodiment import TeamEmbodimentError
    from .website_rigid_feasibility import carry_station_search

    if search is None:
        from .team_embodiment_stance import ground_stance as search
    arguments = carry_station_search(preparation)
    arguments.pop("ranked", None)  # one clear stance answers whether this base fits

    started = time.monotonic()
    searches: list[dict[str, Any]] = []
    floor: list[float] = []

    def clear(half: float) -> bool:
        try:
            found = search(collision_path=collision_path, collision_sha256=collision_sha256,
                           **{**arguments, "footprint_half_extent_m": [half, half]})
        except TeamEmbodimentError as exc:
            if not set(exc.blockers) <= NO_STANCE_BLOCKERS:
                raise FloorNotSearched(",".join(sorted(exc.blockers))) from exc
            searches.append({"half_extent_m": round(half, 6), "clear": False, "blockers": sorted(exc.blockers)})
            return False
        searches.append({"half_extent_m": round(half, 6), "clear": True})
        if isinstance(found, Mapping) and isinstance(found.get("floor_z_m"), (int, float)):
            floor.append(float(found["floor_z_m"]))  # the floor the search stood on, for hand heights
        return True

    lower, upper = float(known_clear_half_m), float(needed_half_m)
    if upper <= lower or clear(upper):
        lower, upper = max(lower, upper), None
    else:
        while upper - lower > RESOLUTION_M and len(searches) < MAX_SEARCHES:
            middle = (lower + upper) / 2.0
            if clear(middle):
                lower = middle
            else:
                upper = middle
    if not floor:
        clear(float(known_clear_half_m))  # one more search, at the standard station, for the floor's height
    refused_at = [row for row in searches if not row["clear"] and upper is not None
                  and row["half_extent_m"] == round(upper, 6)]
    return {"floor_z_m": round(floor[0], 6) if floor else None, "lower_half_m": round(lower, 6), "upper_half_m": None if upper is None else round(upper, 6),
            "upper_hole_possible": bool(refused_at and HOLE_BLOCKERS & set(refused_at[0]["blockers"])),
            "searches": searches, "seconds": round(time.monotonic() - started, 3),
            "rule": "widest_square_base_clear_near_the_task_with_the_standard_station_clearance"}


def floor_stance(sizes: Sequence[Sequence[float]], floor: Mapping[str, Any], band: Sequence[float]) -> dict[str, Any]:
    """Whether a base of the published ``sizes`` (each [length, width], m) stands on the floor ``floor`` offers.

    It passes when its longest side fits even at the smaller scale; fails when even its shortest side
    exceeds the widest floor at the larger scale and no hole in the reconstruction could hide more; and
    is conditional otherwise: the base's longest side beside the floor width the job seems to offer.
    """
    longest = max(max(size) for size in sizes)
    shortest = min(min(size) for size in sizes)
    offered_min = 2.0 * float(floor["lower_half_m"]) * band[0]
    upper = floor["upper_half_m"]
    offered_max = None if upper is None else 2.0 * float(upper) * band[1]
    # Clear at the largest half extent searched, which every base needs at most at the smaller scale.
    if upper is None or offered_min >= longest - 1e-9:
        return {"result": "pass", "base_m": longest, "offered_m": [offered_min, offered_max]}
    if offered_max is not None and offered_max < shortest and not floor["upper_hole_possible"]:
        return {"result": "fail", "base_m": shortest, "offered_m": [offered_min, offered_max]}
    return {"result": "unknown", "base_m": longest, "offered_m": [offered_min, offered_max]}


__all__ = ["FloorNotSearched", "MAX_SEARCHES", "RESOLUTION_M", "floor_stance", "widest_clear_square"]
