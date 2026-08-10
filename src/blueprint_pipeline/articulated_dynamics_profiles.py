"""Measured dynamics bands, researched once per object class and reused.

Finding out how hard a real refrigerator door is to open took a literature
search that ended at one instrumented survey. That is not work to repeat per
asset, and it is not work to skip either - the numbers it produced caught a
twin whose hinge resisted three times too much and had no gasket at all, and
nothing else in the pipeline would have.

The resolution is that a band belongs to an object *class*, not an object. Every
refrigerator we ever build reuses this entry; a dishwasher needs its own, once.
So the research step is: look the class up, and if it is missing, go and find
it - which is exactly what the lookup enforces by failing closed.

Failing closed rather than falling back to a neighbouring class is the point.
Substituting a cupboard's numbers for a refrigerator's would produce a receipt
identical in shape to a real one, and quietly convert "we checked this against
measurements" into "we guessed from something adjacent". A missing profile is a
visible gap; a borrowed profile is an invisible error.

Entries here are reviewable data, deliberately not constructible at call time.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence


DYNAMICS_PROFILE_REGISTRY_VERSION = "articulated_dynamics_profiles.v1"

_REGISTRY: dict[str, dict[str, Any]] = {
    "household_refrigerator_door": {
        "profile_id": "household_refrigerator_door",
        "measurement_source": (
            "Jain, Nguyen, Rath, Okerman & Kemp, 'The Complex Structure of Simple "
            "Devices: A Survey of Trajectories and Forces that Open Doors and "
            "Drawers', IEEE BioRob 2010, DOI 10.1109/BIOROB.2010.5626754"
        ),
        "sample_description": (
            "Instrumented force capture on 29 doors and 15 drawers across 6 homes "
            "and 1 office, 10 trials each; 451 further doors and drawers measured "
            "for kinematics in 11 homes. Appliance breakaway 15-36 N at the "
            "handle, sustained 0-3 N once the gasket releases."
        ),
        # Torque bands are the measured handle forces converted across the
        # surveyed handle radii, which is why the lever-arm band travels with
        # them: applying these torques at a radius outside it is extrapolation.
        "breakaway_torque_n_m": [6.0, 28.0],
        "breakaway_angular_width_degrees": [3.0, 8.0],
        "sustained_torque_n_m": [0.4, 2.1],
        "lever_arm_m": [0.42, 0.70],
        "notes": (
            "Mainstream freestanding units self-close gently from about halfway "
            "on installed backward tilt; cam-hinged built-ins hold open past 90 "
            "degrees. Hold-open behaviour is therefore a per-unit property and "
            "is not part of this band."
        ),
    },
}


class ArticulatedDynamicsProfileError(ValueError):
    """Stable, sorted dynamics-profile lookup failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def available_profile_ids() -> tuple[str, ...]:
    """Object classes whose dynamics someone has actually looked up."""

    return tuple(sorted(_REGISTRY))


def resolve_dynamics_profile(object_class: str) -> dict[str, Any]:
    """Return the measured band for one object class, or refuse."""

    key = str(object_class or "").strip()
    if not key:
        raise ArticulatedDynamicsProfileError(
            ["articulated_dynamics_profile_object_class_missing"]
        )
    profile = _REGISTRY.get(key)
    if profile is None:
        raise ArticulatedDynamicsProfileError(
            [
                f"articulated_dynamics_profile_not_researched:{key}",
                "articulated_dynamics_profile_researched_classes:"
                + ",".join(available_profile_ids()),
            ]
        )
    # Deep copy: a caller that edits what it resolved must not reach the next.
    return copy.deepcopy(profile)


__all__ = [
    "DYNAMICS_PROFILE_REGISTRY_VERSION",
    "ArticulatedDynamicsProfileError",
    "available_profile_ids",
    "resolve_dynamics_profile",
]
