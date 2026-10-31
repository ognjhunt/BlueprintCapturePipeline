"""The one surface-target admission for a website pick-and-place task.

Website preparation and native submission both derive the task's target region
here, so a region evaluation would refuse is reported when the scene is
prepared rather than hours later. On capture-447ae4f4, preparation reported
intake_ready at 03:39Z; evaluation preparation refused the same geometry at
05:01Z, because only native submission ran ``derive_surface_target``.

Pure: no file, collider or provider access. The region and its digest are
exactly the ones native submission sealed before this module existed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .task_evaluation_surface_target import SUPPORT_HEIGHT_TOLERANCE_M, derive_surface_target
from .website_support_geometry import DESTINATION_MARGIN_M

#: How the native request names the website's surfaces; the region is bound to these.
SUPPORT_PRIM_PATH = "/Root"
SUPPORT_INSTANCE_ID = "website-support"
DESTINATION_SUPPORT_INSTANCE_ID = "website-destination-support"


def target_radius_m(subject_min: Sequence[float], subject_max: Sequence[float]) -> float:
    """The region holds the subject at any yaw: its plan half-diagonal plus the margin."""
    return math.hypot(*[(subject_max[i] - subject_min[i]) / 2 for i in (0, 1)]) + DESTINATION_MARGIN_M


def _surface(instance_id: str, bounds: Mapping[str, Any]) -> dict[str, Any]:
    # The fields of the native support record that the region reads.
    return {"publisher_instance_id": instance_id, "sage_prim_path": SUPPORT_PRIM_PATH,
            "bounds_min_xyz_m": [float(value) for value in bounds["aabb_min_xyz"]],
            "bounds_max_xyz_m": [float(value) for value in bounds["aabb_max_xyz"]]}


def admit_surface_target(*, destination: Mapping[str, Any], support: Mapping[str, Any],
                         subject_min: Sequence[float], subject_max: Sequence[float],
                         placement_uncertainty_m: Any, development_support: bool) -> dict[str, Any]:
    """The sealed region on the destination's surface, or the typed code evaluation refuses it with.

    All inputs are in the simulator frame. ``destination`` is the preparation's
    destination: ``position_world_m``, ``relation``, ``visible_label``, the
    interior bounds inside a container, and ``support`` when the object ends
    on another surface than the one it starts on. ``support`` is the surface
    under the subject (``aabb_min_xyz``/``aabb_max_xyz``). A captured support's
    top is known only to its ``placement_uncertainty_m``; an authored
    development surface keeps the 5 mm check. Raises ``ValueError`` with the
    ``surface_target_*`` code of ``derive_surface_target``.
    """
    start = _surface(SUPPORT_INSTANCE_ID, support)
    target_support, start_support = start, None
    if destination.get("support") is not None:
        # The object ends on another surface (floor to desk): the region is on that one.
        target_support, start_support = _surface(DESTINATION_SUPPORT_INSTANCE_ID, destination["support"]), start
    # An estimated support's top is known only to the placement uncertainty
    # its destination was found within (capture-447ae4f4: 7.2 cm; the region
    # sat 1.2 cm under the support's highest face, the 5 mm check refused it).
    tolerance = SUPPORT_HEIGHT_TOLERANCE_M
    if (not development_support and isinstance(placement_uncertainty_m, (int, float))
            and not isinstance(placement_uncertainty_m, bool) and math.isfinite(placement_uncertainty_m)):
        tolerance = max(SUPPORT_HEIGHT_TOLERANCE_M, float(placement_uncertainty_m))
    return derive_surface_target(
        destination={**destination, "kind": "green_region", "radius_m": target_radius_m(subject_min, subject_max)},
        support=target_support, source_min=subject_min, source_max=subject_max,
        support_instance_id=target_support["publisher_instance_id"], start_support=start_support,
        height_tolerance_m=tolerance)
