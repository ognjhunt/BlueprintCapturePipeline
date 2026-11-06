"""The one surface-target admission for a website pick-and-place task.

Website preparation and native submission both derive the task's target region
here, so a region evaluation would refuse is reported when the scene is
prepared rather than hours later. On capture-447ae4f4, preparation reported
intake_ready at 03:39Z; evaluation preparation refused the same geometry at
05:01Z, because only native submission ran ``derive_surface_target``.

Pure: no file, collider or provider access.

Since the owner's G3 switch (2026-11-06) every website region is a v2 target
(``task_evaluation_surface_target``): the subject's centre must end within
``WEBSITE_TARGET_TOLERANCE_M`` of the region's centre, so the disc is that
tolerance, not wider than the object, and it is kept clear of the start
footprint by ``website_target_reach``. Regions sealed before stay v1 and keep
scoring whole-footprint containment.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .task_evaluation_surface_target import CENTRE_RULE, SUPPORT_HEIGHT_TOLERANCE_M, derive_surface_target, require
from .website_support_geometry import REGION_EDGE_ROUNDING_M, SupportOutline

#: How far the subject's centre may end from the region's centre: the disc's radius
#: (owner decision 2026-11-06, G3: score the centre within a tolerance). The owner chose
#: the rule; 5 cm is the harness's default, not a number the owner set.
WEBSITE_TARGET_TOLERANCE_M = 0.05

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


def website_target_radius(*, meters_per_unit: float = 1.0) -> float:
    """The one target-disc radius: the tolerance the subject's centre must end within.

    Whatever the subject's size: the disc no longer has to hold the object, so a
    larger replacement does not widen it. Simulator metres by default;
    ``meters_per_unit`` gives it in runtime units.
    """
    return WEBSITE_TARGET_TOLERANCE_M / meters_per_unit


def website_target_reach(lower: Sequence[float], upper: Sequence[float], *, relative_error: float,
                         up: int = 2, meters_per_unit: float = 1.0) -> float:
    """How far from the region's centre an accepted subject's footprint can reach.

    The plan half-diagonal of the estimate, each half scaled by ``1 + relative_error``
    (the largest replacement static qualification admits, at any yaw), plus the
    radius. A start centre farther than this keeps the whole disc off the start
    footprint, so every relocation clearance and the start rule read it
    (capture-447ae4f4: a replacement 3.3% larger than a 31x52 cm tray's estimate
    was refused at bind, after the scene configuration was paid for). Simulator
    frame by default; ``up`` and ``meters_per_unit`` give it in runtime units.
    """
    scale = 1.0 + _checked_tolerance(relative_error)
    return (math.hypot(*[(upper[axis] - lower[axis]) / 2 * scale for axis in range(3) if axis != up])
            + website_target_radius(meters_per_unit=meters_per_unit))


def _surface(instance_id: str, bounds: Mapping[str, Any]) -> dict[str, Any]:
    # The fields of the native support record that the region reads.
    return {"publisher_instance_id": instance_id, "sage_prim_path": SUPPORT_PRIM_PATH,
            "bounds_min_xyz_m": [float(value) for value in bounds["aabb_min_xyz"]],
            "bounds_max_xyz_m": [float(value) for value in bounds["aabb_max_xyz"]]}


def admit_surface_target(*, destination: Mapping[str, Any], support: Mapping[str, Any],
                         subject_min: Sequence[float], subject_max: Sequence[float],
                         placement_uncertainty_m: Any, development_support: bool,
                         relative_error: float = 0.0, support_outline: SupportOutline | None = None) -> dict[str, Any]:
    """The sealed region on the destination's surface, or the typed code evaluation refuses it with.

    All inputs are in the simulator frame. ``destination`` is the preparation's
    destination: ``position_world_m``, ``relation``, ``visible_label``, the
    interior bounds inside a container, and ``support`` when the object ends
    on another surface than the one it starts on. ``support`` is the surface
    under the subject (``aabb_min_xyz``/``aabb_max_xyz``). A captured support's
    top is known only to its ``placement_uncertainty_m``; an authored
    development surface keeps the 5 mm check. ``relative_error`` is the
    subject's ``replacement_size_tolerance``: the start footprint the disc must
    stay clear of is widened by it in plan, so a larger qualified replacement
    cannot start over the disc. ``support_outline`` is the
    region's surface itself (``SupportOutline.in_frame``): the region must lie
    wholly on it, not only in its bounding box (capture-447ae4f4: an oval
    table). Preparation, which holds the collider, passes it and records a
    refusal as advisory; native submission seals without it, its relocation
    having placed the region on the surface where any spot allows. Raises
    ``ValueError`` with the ``surface_target_*`` code of
    ``derive_surface_target``.
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
    scale = 1.0 + _checked_tolerance(relative_error)
    middle = [(float(a) + float(b)) / 2 for a, b in zip(subject_min, subject_max, strict=True)]
    # The largest admitted replacement's start footprint (plan axes only; the height keeps the estimate's).
    start_min = [m - (m - float(a)) * scale if axis != 2 else float(a)
                 for axis, (m, a) in enumerate(zip(middle, subject_min, strict=True))]
    start_max = [m + (float(b) - m) * scale if axis != 2 else float(b)
                 for axis, (m, b) in enumerate(zip(middle, subject_max, strict=True))]
    target = derive_surface_target(
        destination={**destination, "kind": "green_region", "radius_m": website_target_radius()},
        support=target_support, source_min=start_min, source_max=start_max,
        support_instance_id=target_support["publisher_instance_id"], start_support=start_support,
        height_tolerance_m=tolerance, success_rule=CENTRE_RULE)
    if support_outline is not None:
        # Twice placement's rounding allowance: converting a placed region to this frame cannot refuse it.
        require(support_outline.holds(target["surface_position_world_m"][:2], target["radius_m"],
                                      tolerance=2 * REGION_EDGE_ROUNDING_M), "surface_target_outside_support")
    return target


def submission_relocates_destination(preparation: Mapping[str, Any], destination: Mapping[str, Any], *,
                                     subject_min: Sequence[float], subject_max: Sequence[float]) -> bool:
    """Whether native submission moves this destination off the start before it seals the region.

    A preparation from before the same-surface rule, or one that moved its
    destination by a smaller clearance than this region's
    (``website_task_preparation.relocated_destination``). Simulator frame.
    """
    from .website_support_geometry import destination_clear_of_start

    size_tolerance = preparation_size_tolerance(preparation)
    recorded = preparation.get("destination_relocation")
    return (destination.get("support") is None and destination.get("relation") == "on"
            and (recorded is not None and recorded.get("replacement_size_tolerance", 0.0) < size_tolerance
                 or not destination_clear_of_start(subject_min, subject_max, destination["position_world_m"], up=2,
                     clearance=website_target_reach(subject_min, subject_max, relative_error=size_tolerance))))


def sealed_surface_target(preparation: Mapping[str, Any], *, destination: Mapping[str, Any],
                          support: Mapping[str, Any], subject_min: Sequence[float], subject_max: Sequence[float],
                          development: bool) -> dict[str, Any]:
    """The region native submission seals into the success document, from the preparation it was compiled from.

    ``admit_surface_target`` with the preparation's placement uncertainty and
    replacement size tolerance; ``support`` is the intake task's support under
    the subject. The website preparation reads the same region to model the
    support's collider before submission (``website_support_pad``).
    """
    return admit_surface_target(destination=destination, support=support, subject_min=subject_min,
                                subject_max=subject_max,
                                placement_uncertainty_m=preparation["coordinate_frame"].get("placement_uncertainty_m"),
                                development_support=bool(development),
                                relative_error=preparation_size_tolerance(preparation))
