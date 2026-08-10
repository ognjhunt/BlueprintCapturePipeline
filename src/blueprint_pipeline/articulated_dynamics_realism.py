"""Check an articulated twin's authored dynamics against measured reality.

Drive constants are the easiest thing in a generated twin to get wrong without
noticing. Geometry is visible, textures are visible, a missing joint stops the
run - but a hinge that resists fifteen times too much still opens, still
renders, and still produces a clean scripted-positive receipt. The error only
shows up as a policy that learned to shove.

The 840796 refrigerator shipped exactly that way. Its hinge damping put 6.1 N*m
of steady resistance behind a door whose measured class tops out near 2.1, and
it carried no gasket detent at all - and a gasket is most of what distinguishes
a refrigerator door from a cupboard door, since the measured peak is an order
of magnitude above the sustained force and lives in the first few degrees.

So the band is checked, and the band has to come from somewhere. A profile
without a citation is refused rather than defaulted, because an invented range
is worse than none: it launders a guess into a receipt that reads like
evidence. What the profile asserts is a published measurement of real objects;
what this module asserts is only that the authored numbers do or do not fall
inside it. Neither is a claim that the simulated door behaves like a real one.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DYNAMICS_REALISM_SCHEMA_VERSION = "articulated_dynamics_realism.v1"
BAND_FIELDS = (
    "breakaway_torque_n_m",
    "breakaway_angular_width_degrees",
    "sustained_torque_n_m",
    "lever_arm_m",
)


class ArticulatedDynamicsRealismError(ValueError):
    """Stable, sorted dynamics-realism failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _band(profile: Mapping[str, Any], field: str, errors: list[str]):
    value = profile.get(field)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        errors.append(f"articulated_dynamics_band_invalid:{field}")
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        errors.append(f"articulated_dynamics_band_invalid:{field}")
        return None
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        errors.append(f"articulated_dynamics_band_invalid:{field}")
        return None
    return (low, high)


def _number(value: Any, name: str, errors: list[str], *, allow_none: bool = False):
    if value is None:
        if not allow_none:
            errors.append(f"articulated_dynamics_{name}_missing")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"articulated_dynamics_{name}_invalid")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"articulated_dynamics_{name}_invalid")
        return None
    return number


def evaluate_articulated_dynamics_realism(
    *,
    lever_arm_m: float,
    joint_damping_n_m_s_per_rad: float,
    breakaway_torque_n_m: float | None,
    breakaway_angular_width_degrees: float | None,
    nominal_open_angle_degrees: float,
    nominal_sweep_duration_s: float,
    reference_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare authored drive constants against a cited measured band."""

    errors: list[str] = []
    if not isinstance(reference_profile, Mapping):
        raise ArticulatedDynamicsRealismError(
            ["articulated_dynamics_reference_profile_invalid"]
        )
    source = str(reference_profile.get("measurement_source") or "").strip()
    if not source:
        # An unsourced band is an opinion wearing a receipt's clothes.
        errors.append("articulated_dynamics_measurement_source_missing")
    profile_id = str(reference_profile.get("profile_id") or "").strip()
    if not profile_id:
        errors.append("articulated_dynamics_profile_id_missing")

    bands = {field: _band(reference_profile, field, errors) for field in BAND_FIELDS}

    lever = _number(lever_arm_m, "lever_arm", errors)
    damping = _number(joint_damping_n_m_s_per_rad, "joint_damping", errors)
    breakaway = _number(
        breakaway_torque_n_m, "breakaway_torque", errors, allow_none=True
    )
    breakaway_width = _number(
        breakaway_angular_width_degrees,
        "breakaway_angular_width",
        errors,
        allow_none=True,
    )
    open_angle = _number(nominal_open_angle_degrees, "nominal_open_angle", errors)
    duration = _number(nominal_sweep_duration_s, "nominal_sweep_duration", errors)
    if duration is not None and duration <= 0.0:
        errors.append("articulated_dynamics_nominal_sweep_duration_invalid")
    if lever is not None and lever <= 0.0:
        errors.append("articulated_dynamics_lever_arm_invalid")
    if errors:
        raise ArticulatedDynamicsRealismError(errors)

    velocity = math.radians(open_angle) / duration
    sustained = damping * velocity
    ratio = None if not breakaway or sustained <= 0.0 else breakaway / sustained

    findings: list[str] = []

    def _check(name: str, value: float | None, band, absent_finding: str) -> None:
        if band is None:
            return
        if value is None:
            findings.append(absent_finding)
            return
        low, high = band
        if value < low:
            findings.append(f"{name}_below_measured_band:{value:.4g}<{low:.4g}")
        elif value > high:
            findings.append(f"{name}_above_measured_band:{value:.4g}>{high:.4g}")

    _check(
        "articulated_dynamics_sustained_torque",
        sustained,
        bands["sustained_torque_n_m"],
        "articulated_dynamics_sustained_torque_absent",
    )
    _check(
        "articulated_dynamics_breakaway_torque",
        breakaway,
        bands["breakaway_torque_n_m"],
        "articulated_dynamics_breakaway_torque_absent",
    )
    _check(
        "articulated_dynamics_breakaway_width",
        breakaway_width,
        bands["breakaway_angular_width_degrees"],
        "articulated_dynamics_breakaway_width_absent",
    )
    _check(
        "articulated_dynamics_lever_arm",
        lever,
        bands["lever_arm_m"],
        "articulated_dynamics_lever_arm_absent",
    )

    return {
        "schema_version": DYNAMICS_REALISM_SCHEMA_VERSION,
        "within_measured_band": not findings,
        "findings": sorted(findings),
        "observed": {
            "lever_arm_m": lever,
            "joint_damping_n_m_s_per_rad": damping,
            "nominal_angular_velocity_rad_s": velocity,
            "sustained_torque_n_m": sustained,
            "sustained_handle_force_n": sustained / lever,
            "breakaway_torque_n_m": breakaway,
            "breakaway_angular_width_degrees": breakaway_width,
            "breakaway_handle_force_n": None if breakaway is None else breakaway / lever,
            "peak_to_sustained_ratio": ratio,
        },
        "reference_profile": {
            "profile_id": profile_id,
            "measurement_source": source,
            "sample_description": str(
                reference_profile.get("sample_description") or ""
            ),
            **{field: list(bands[field]) for field in BAND_FIELDS if bands[field]},
        },
        "claim_boundary": {
            "band_is_published_measurement_not_our_own": True,
            "inside_the_band_is_plausibility_not_fidelity": True,
            "authored_constants_are_not_a_measurement_of_this_object": True,
        },
    }


__all__ = [
    "DYNAMICS_REALISM_SCHEMA_VERSION",
    "ArticulatedDynamicsRealismError",
    "evaluate_articulated_dynamics_realism",
]
