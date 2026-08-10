"""Find a robot asset inside an image, and say where you looked if you cannot.

Isaac ships robot USDs, but where has moved between releases and between the
container's bundled copy and its remote asset root. Hardcoding one path means a
launch that boots, spends four minutes bringing Isaac up, and then reports that
one guess was wrong - and the next attempt guesses again.

So the candidates are a list, and a failure names every place that was checked.
When the search fails the search itself is the useful output: one launch then
resolves the layout for good instead of one launch per guess.

Nothing here reaches the network. A remote asset root may or may not be
reachable from a provider container, and discovering that mid-run is its own
wasted launch; this looks only at what is already on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


ROBOT_ASSET_DISCOVERY_SCHEMA_VERSION = "robot_asset_discovery.v1"

# Layouts seen across Isaac Sim releases and their bundled asset trees.
FRANKA_CANDIDATE_RELATIVE_PATHS = (
    "Isaac/Robots/Franka/franka.usd",
    "Isaac/Robots/Franka/franka_instanceable.usd",
    "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "Isaac/Robots/Franka/franka_alt_fingers.usd",
    "Robots/Franka/franka.usd",
)
DEFAULT_SEARCH_ROOTS = (
    "/isaac-sim/assets",
    "/isaac-sim/data/assets",
    "/root/.local/share/ov/pkg/isaac-sim/assets",
    "/isaac-sim",
)


# Trees that contain USDs for reasons other than shipping a robot.
NON_ASSET_PATH_MARKERS = ("/data/tests/", "/tests/", "/unittests/", "/extscache/")


def is_usable_robot_asset(path: str) -> bool:
    """Whether a Franka-named USD is actually a robot rather than a test scene.

    The capability probe found exactly one Franka-named USD on the bare image
    and it was a viewport regression fixture - a bolt-tightening scene. Counting
    it reported a route as viable on an image that cannot support it, which is
    worse than reporting nothing, because it sends the next launch somewhere
    there is nothing to find.
    """

    lowered = str(path).lower()
    return not any(marker in lowered for marker in NON_ASSET_PATH_MARKERS)


class RobotAssetDiscoveryError(ValueError):
    """Stable, sorted robot-asset discovery failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def discover_robot_asset(
    *,
    search_roots: Sequence[str | Path] = DEFAULT_SEARCH_ROOTS,
    relative_candidates: Sequence[str] = FRANKA_CANDIDATE_RELATIVE_PATHS,
) -> dict[str, Any]:
    """Return the first candidate that exists, or report the whole search."""

    roots = [Path(str(value)).expanduser() for value in search_roots]
    candidates = [str(value) for value in relative_candidates if str(value)]
    if not roots or not candidates:
        raise RobotAssetDiscoveryError(["robot_asset_discovery_search_space_empty"])

    checked: list[str] = []
    for root in roots:
        for relative in candidates:
            path = root / relative
            checked.append(str(path))
            # A directory of that name is not the asset; accepting it would
            # fail later and less clearly.
            if path.is_file() and is_usable_robot_asset(str(path)):
                return {
                    "schema_version": ROBOT_ASSET_DISCOVERY_SCHEMA_VERSION,
                    "resolved_path": str(path),
                    "matched_relative_path": relative,
                    "search_root": str(root),
                    "paths_checked": checked,
                    "claim_boundary": {
                        "existence_only_not_a_load_check": True,
                        "no_network_asset_root_consulted": True,
                    },
                }
    raise RobotAssetDiscoveryError(
        ["robot_asset_discovery_robot_asset_not_found"]
        + [f"robot_asset_discovery_checked:{path}" for path in checked]
    )


__all__ = [
    "DEFAULT_SEARCH_ROOTS",
    "NON_ASSET_PATH_MARKERS",
    "is_usable_robot_asset",
    "FRANKA_CANDIDATE_RELATIVE_PATHS",
    "ROBOT_ASSET_DISCOVERY_SCHEMA_VERSION",
    "RobotAssetDiscoveryError",
    "discover_robot_asset",
]
