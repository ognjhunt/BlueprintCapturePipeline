"""Shared contract types for the dynamic robot-placement pipeline.

This module is the foundation every other ``scene_placement`` component imports.
It is deliberately tiny and dependency-free (stdlib ``dataclasses`` + ``typing``
only) so that the whole package — and its unit tests — import with NO isaacsim,
NO google-genai, NO torch, NO network, and NO GPU. Heavy/optional backends
(USD, VLM, perception, PhysX probes) are injected by the callers that need them.

The shapes here are a hard contract: ``usd_index``, ``perception_index``,
``target_resolver``, and ``placement`` all depend on these exact fields, so keep
them stable. Geometry is expressed in world coordinates with ``z`` pointing up
(floor at ``z = floor_z``), which is what the placement math assumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol, Tuple, runtime_checkable

# A 3-tuple of world coordinates (meters). Aliased for readable annotations.
Vec3 = Tuple[float, float, float]


@dataclass
class SceneObject:
    """A single meaningful scene object with a world-aligned bounding box.

    Produced by the spatial indices (USD walk or perception unprojection) and
    consumed by the target resolver + placement solver. The AABB is the geometric
    truth the placement math reasons about; ``label`` is the human handle the
    task/VLM matches against ("faucet", "sink", "stove").
    """

    id: str
    label: str  # human label, e.g. "faucet", "sink", "stove"
    bbox_min: Vec3  # world AABB lower corner (x, y, z)
    bbox_max: Vec3  # world AABB upper corner (x, y, z)
    centroid: Vec3  # world centroid (the thing the robot turns to face)
    category: str = ""  # optional coarse class
    source: str = ""  # provenance: "usd" | "perception"
    confidence: float = 1.0
    extra: Dict = field(default_factory=dict)

    def size(self) -> Vec3:
        """Axis-aligned extents ``(dx, dy, dz)`` of the bounding box.

        Used to derive a standoff that scales with how big the target is.
        """
        return (
            self.bbox_max[0] - self.bbox_min[0],
            self.bbox_max[1] - self.bbox_min[1],
            self.bbox_max[2] - self.bbox_min[2],
        )

    def footprint_center(self) -> Tuple[float, float]:
        """Center of the object's floor footprint ``(cx, cy)`` (xy of the AABB).

        Placement steps outward from this point to find open floor; using the
        AABB center (not the centroid) keeps it stable for irregular meshes.
        """
        return (
            0.5 * (self.bbox_min[0] + self.bbox_max[0]),
            0.5 * (self.bbox_min[1] + self.bbox_max[1]),
        )

    def min_z(self) -> float:
        """Lowest ``z`` of the AABB (e.g. where the object meets the floor)."""
        return self.bbox_min[2]

    def max_z(self) -> float:
        """Highest ``z`` of the AABB (e.g. counter height for reach checks)."""
        return self.bbox_max[2]


@dataclass
class StandPose:
    """Where the robot should stand to act on a target, plus why we trust it.

    ``position`` is the pelvis world pose (its ``z`` is the pelvis height above
    the floor, not floor level). ``clear`` records whether a probe verified the
    floor is actually free at this spot; ``notes`` carries human-readable context
    (e.g. a "no clear side, fell back to max_out" explanation).
    """

    position: Vec3  # robot pelvis world pos (z = pelvis height above floor)
    yaw: float  # radians, faces the target centroid
    target_id: str
    clear: bool  # probe-verified clear floor
    standoff_m: float  # distance from target footprint
    notes: str = ""


# A footprint-overlap probe: given a candidate pelvis pose + yaw, return the
# number of PhysX-style collision hits under the robot's footprint. ``0`` means
# clear floor. Injected so placement is unit-testable with a mock that marks
# occupied cells (no PhysX/GPU needed). Callable signature is
# ``(pose: Vec3, yaw: float) -> int``.
Probe = Callable[[Vec3, float], int]


@runtime_checkable
class SceneSpatialIndex(Protocol):
    """Anything that can enumerate scene objects for placement.

    Both the USD-backed and perception-backed indices satisfy this; the
    orchestrator depends only on this surface so backends stay swappable.
    ``runtime_checkable`` lets tests/callers ``isinstance``-check a duck-typed
    fake without importing a concrete index.
    """

    def objects(self) -> List[SceneObject]:
        """Return the meaningful, placement-relevant objects in the scene."""
        ...
