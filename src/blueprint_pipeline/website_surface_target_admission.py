"""The one surface-target admission for a website pick-and-place task.

Website preparation and native submission both derive the task's target region
here, so a region evaluation would refuse is reported when the scene is
prepared rather than hours later. On capture-447ae4f4, preparation reported
intake_ready at 03:39Z; evaluation preparation refused the same geometry at
05:01Z, because only native submission ran ``derive_surface_target``.

Pure: no file, collider or provider access. For a subject built to size the
region and its digest are exactly the ones native submission sealed before this
module existed; an estimated subject's region also holds the largest
replacement static qualification admits (``website_target_radius``).
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


def _checked_tolerance(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("website_target_size_tolerance_invalid")
    return float(value)


def replacement_size_tolerance(*, dimension_authority: Any, relative_error: Any, development: bool) -> float:
    """How much larger than its estimate, per dimension, a qualified replacement may be.

    Static qualification admits a replacement within ``relative_error`` of each
    estimated dimension (``maximum_dimension_relative_error``, 0.25). An
    authored development environment, a designed subject and published
    dimensions are built to size: zero, the region as before.
    """
    if development or dimension_authority != "estimated":
        return 0.0
    return _checked_tolerance(relative_error)


def preparation_size_tolerance(preparation: Mapping[str, Any]) -> float:
    """``replacement_size_tolerance`` as a website preparation recorded it."""
    authoring = preparation.get("authoring_inputs") or {}
    return replacement_size_tolerance(dimension_authority=authoring.get("dimension_authority"),
        relative_error=(authoring.get("metric_envelope") or {}).get("maximum_dimension_relative_error"),
        development=preparation.get("development_test") is not None)


def website_target_radius(lower: Sequence[float], upper: Sequence[float], *, relative_error: float,
                          up: int = 2, meters_per_unit: float = 1.0) -> float:
    """The one target-disc radius: it holds the subject, or its largest admitted replacement, at any yaw.

    The plan half-diagonal of the estimate with each half scaled by
    ``1 + relative_error``, plus the margin. Admission, every relocation
    clearance, the start rule, the scorer and native bind read this radius
    (capture-447ae4f4: a replacement 3.3% larger than a 31x52 cm tray's
    estimate was refused at bind, after the scene configuration was paid for).
    Simulator frame by default; ``up`` and ``meters_per_unit`` give it in
    runtime units. With ``relative_error`` zero it is exactly the radius before.
    """
    scale = 1.0 + _checked_tolerance(relative_error)
    return (math.hypot(*[(upper[axis] - lower[axis]) / 2 * scale for axis in range(3) if axis != up])
            + DESTINATION_MARGIN_M / meters_per_unit)


def _surface(instance_id: str, bounds: Mapping[str, Any]) -> dict[str, Any]:
    # The fields of the native support record that the region reads.
    return {"publisher_instance_id": instance_id, "sage_prim_path": SUPPORT_PRIM_PATH,
            "bounds_min_xyz_m": [float(value) for value in bounds["aabb_min_xyz"]],
            "bounds_max_xyz_m": [float(value) for value in bounds["aabb_max_xyz"]]}


def admit_surface_target(*, destination: Mapping[str, Any], support: Mapping[str, Any],
                         subject_min: Sequence[float], subject_max: Sequence[float],
                         placement_uncertainty_m: Any, development_support: bool,
                         relative_error: float = 0.0) -> dict[str, Any]:
    """The sealed region on the destination's surface, or the typed code evaluation refuses it with.

    All inputs are in the simulator frame. ``destination`` is the preparation's
    destination: ``position_world_m``, ``relation``, ``visible_label``, the
    interior bounds inside a container, and ``support`` when the object ends
    on another surface than the one it starts on. ``support`` is the surface
    under the subject (``aabb_min_xyz``/``aabb_max_xyz``). A captured support's
    top is known only to its ``placement_uncertainty_m``; an authored
    development surface keeps the 5 mm check. ``relative_error`` is the
    subject's ``replacement_size_tolerance``. Raises ``ValueError`` with the
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
        destination={**destination, "kind": "green_region", "radius_m": website_target_radius(
            subject_min, subject_max, relative_error=relative_error)},
        support=target_support, source_min=subject_min, source_max=subject_max,
        support_instance_id=target_support["publisher_instance_id"], start_support=start_support,
        height_tolerance_m=tolerance)
