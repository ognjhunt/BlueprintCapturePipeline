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
import difflib
from typing import Any, Sequence


DYNAMICS_PROFILE_REGISTRY_VERSION = "articulated_dynamics_profiles.v1"
BAND_FIELDS = (
    "breakaway_torque_n_m",
    "breakaway_angular_width_degrees",
    "sustained_torque_n_m",
    "lever_arm_m",
)

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
        # Contact properties, separate because they come from a different
        # place: the force survey measured how hard doors are to pull, not what
        # their panels are made of. Attributing these to it would be a
        # misattribution, so they carry their own source line.
        "material_bands": {
            "measurement_source": (
                "Engineering-admissible bands for household appliance shell "
                "materials - painted and stainless sheet steel, ABS and "
                "polystyrene liners, tempered glass shelving, and closed-cell "
                "gasket rubber - taken from standard dry-contact coefficient "
                "tables and appliance mass ratings rather than from an "
                "instrumented capture of this object class."
            ),
            "band_basis": "material_class_envelope_not_per_asset_measurement",
            # Steel on steel dry runs about 0.4-0.6, filled ABS about 0.3-0.5,
            # and gasket rubber up to about 0.9. The envelope spans the mix.
            "dynamic_friction_range": [0.25, 0.90],
            # Static exceeds dynamic for every pairing in that set.
            "static_friction_range": [0.30, 1.00],
            # Painted steel, plastic liner and glass against rigid surfaces sit
            # low; a gasket is damped, not elastic. Nothing on an appliance
            # rebounds like rubber ball stock, so the ceiling is well under it.
            "restitution_range": [0.0, 0.50],
            # A door leaf or shelf, not the cabinet: full-size doors run about
            # 8-20 kg with liner, gasket and any in-door dispenser.
            "link_mass_range_kg": [3.0, 45.0],
            # Whole unit: freestanding full-size refrigerators are rated about
            # 70-140 kg empty.
            "assembly_mass_range_kg": [55.0, 160.0],
        },
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
        errors = [
            f"articulated_dynamics_profile_not_researched:{key}",
            "articulated_dynamics_profile_researched_classes:"
            + ",".join(available_profile_ids()),
            # The error is the work order, so it states the job rather than
            # leaving whoever hits it to reverse-engineer the schema.
            "articulated_dynamics_profile_required_fields:"
            + ",".join(("measurement_source", "sample_description") + BAND_FIELDS),
        ]
        # Failing closed identically for a typo and for a class nobody has ever
        # measured sends a person off to redo a search that was already done.
        near = difflib.get_close_matches(key, available_profile_ids(), n=1, cutoff=0.6)
        if near:
            errors.append(f"articulated_dynamics_profile_did_you_mean:{near[0]}")
        raise ArticulatedDynamicsProfileError(errors)
    # Deep copy: a caller that edits what it resolved must not reach the next.
    return copy.deepcopy(profile)


__all__ = [
    "BAND_FIELDS",
    "DYNAMICS_PROFILE_REGISTRY_VERSION",
    "ArticulatedDynamicsProfileError",
    "available_profile_ids",
    "resolve_dynamics_profile",
]
