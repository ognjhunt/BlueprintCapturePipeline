"""Pinned ADP-009D PhysX contact-envelope contract.

The approved can uses an SDF collider while the frozen Robotiq finger
colliders retain their 5 mm PhysX contact offset.  PhysX can therefore create
a contact before the rendered surfaces meet.  This module makes that envelope
an explicit, immutable input to both the native collider validation and the
controls plan; neither layer is allowed to silently use a different margin.

It contains only standard-library code because it is shipped flat with the
provider runtime and must be testable without Isaac or USD installed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


CONTACT_ENVELOPE_SCHEMA_VERSION = "adp009d_contact_envelope.v1"
APPROVED_CAN_SDF_MARGIN_M = 0.0025
APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M = 0.0025
FINGER_COLLIDER_CONTACT_OFFSET_M = 0.005
APPROVED_CAN_SDF_RESOLUTION = 256
APPROVED_CAN_SDF_SUBGRID_RESOLUTION = 6
CONTACT_ENVELOPE_CALCULATION = (
    "sdf_margin_m_plus_sdf_narrow_band_thickness_m_plus_"
    "finger_collider_contact_offset_m"
)
FINGER_COLLIDER_CONTACT_OFFSET_SOURCE = (
    "adp009d_franka_eval_harness_manifest.v1.json:physics.settings.contact_offset_m"
)


class ContactEnvelopeError(ValueError):
    """Stable fail-closed errors for drifted ADP-009D contact settings."""


def _finite_number(value: Any, *, error: str) -> float:
    if isinstance(value, bool):
        raise ContactEnvelopeError(error)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContactEnvelopeError(error) from exc
    if not math.isfinite(result):
        raise ContactEnvelopeError(error)
    return result


def _exact_number(value: Any, *, expected: float, error: str) -> float:
    result = _finite_number(value, error=error)
    if not math.isclose(result, expected, rel_tol=0.0, abs_tol=1.0e-9):
        raise ContactEnvelopeError(error)
    return expected


def _exact_integer(value: Any, *, expected: int, error: str) -> int:
    if isinstance(value, bool):
        raise ContactEnvelopeError(error)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ContactEnvelopeError(error) from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or int(numeric) != expected
    ):
        raise ContactEnvelopeError(error)
    return expected


def canonical_contact_envelope() -> dict[str, Any]:
    """Return the sole contact-generating envelope admitted for ADP-009D."""

    effective = (
        APPROVED_CAN_SDF_MARGIN_M
        + APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M
        + FINGER_COLLIDER_CONTACT_OFFSET_M
    )
    return {
        "schema_version": CONTACT_ENVELOPE_SCHEMA_VERSION,
        "sdf_margin_m": APPROVED_CAN_SDF_MARGIN_M,
        "sdf_narrow_band_thickness_m": APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M,
        "finger_collider_contact_offset_m": FINGER_COLLIDER_CONTACT_OFFSET_M,
        "finger_collider_contact_offset_source": (
            FINGER_COLLIDER_CONTACT_OFFSET_SOURCE
        ),
        "effective_contact_envelope_m": effective,
        "effective_contact_envelope_calculation": CONTACT_ENVELOPE_CALCULATION,
        "sdf_resolution": APPROVED_CAN_SDF_RESOLUTION,
        "sdf_subgrid_resolution": APPROVED_CAN_SDF_SUBGRID_RESOLUTION,
    }


def contact_envelope_from_physx_sdf_settings(
    *,
    sdf_margin_m: Any,
    sdf_narrow_band_thickness_m: Any,
    sdf_resolution: Any,
    sdf_subgrid_resolution: Any,
) -> dict[str, Any]:
    """Validate the authored PhysX SDF settings and return their envelope."""

    _exact_number(
        sdf_margin_m,
        expected=APPROVED_CAN_SDF_MARGIN_M,
        error="adp009d_contact_envelope_sdf_margin_invalid",
    )
    _exact_number(
        sdf_narrow_band_thickness_m,
        expected=APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M,
        error="adp009d_contact_envelope_sdf_narrow_band_thickness_invalid",
    )
    _exact_integer(
        sdf_resolution,
        expected=APPROVED_CAN_SDF_RESOLUTION,
        error="adp009d_contact_envelope_sdf_resolution_invalid",
    )
    _exact_integer(
        sdf_subgrid_resolution,
        expected=APPROVED_CAN_SDF_SUBGRID_RESOLUTION,
        error="adp009d_contact_envelope_sdf_subgrid_resolution_invalid",
    )
    return canonical_contact_envelope()


def contact_envelope_from_harness_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the modeled finger contact offset to the frozen harness input."""

    physics = value.get("physics") if isinstance(value, Mapping) else None
    settings = physics.get("settings") if isinstance(physics, Mapping) else None
    canonical = value.get("canonical_condition") if isinstance(value, Mapping) else None
    parameters = canonical.get("parameters") if isinstance(canonical, Mapping) else None
    if not isinstance(settings, Mapping) or not isinstance(parameters, Mapping):
        raise ContactEnvelopeError("adp009d_contact_envelope_harness_settings_missing")
    _exact_number(
        settings.get("contact_offset_m"),
        expected=FINGER_COLLIDER_CONTACT_OFFSET_M,
        error="adp009d_contact_envelope_harness_contact_offset_invalid",
    )
    _exact_number(
        parameters.get("object_contact_offset_m"),
        expected=FINGER_COLLIDER_CONTACT_OFFSET_M,
        error="adp009d_contact_envelope_harness_object_contact_offset_invalid",
    )
    return canonical_contact_envelope()


def validate_contact_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a retained envelope exactly matches the frozen one."""

    if not isinstance(value, Mapping):
        raise ContactEnvelopeError("adp009d_contact_envelope_not_mapping")
    expected = canonical_contact_envelope()
    if set(value) != set(expected):
        raise ContactEnvelopeError("adp009d_contact_envelope_fields_invalid")
    if value.get("schema_version") != CONTACT_ENVELOPE_SCHEMA_VERSION:
        raise ContactEnvelopeError("adp009d_contact_envelope_schema_invalid")
    if value.get("finger_collider_contact_offset_source") != (
        FINGER_COLLIDER_CONTACT_OFFSET_SOURCE
    ):
        raise ContactEnvelopeError("adp009d_contact_envelope_offset_source_invalid")
    if value.get("effective_contact_envelope_calculation") != (
        CONTACT_ENVELOPE_CALCULATION
    ):
        raise ContactEnvelopeError("adp009d_contact_envelope_calculation_invalid")
    _exact_number(
        value.get("sdf_margin_m"),
        expected=APPROVED_CAN_SDF_MARGIN_M,
        error="adp009d_contact_envelope_sdf_margin_invalid",
    )
    _exact_number(
        value.get("sdf_narrow_band_thickness_m"),
        expected=APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M,
        error="adp009d_contact_envelope_sdf_narrow_band_thickness_invalid",
    )
    _exact_number(
        value.get("finger_collider_contact_offset_m"),
        expected=FINGER_COLLIDER_CONTACT_OFFSET_M,
        error="adp009d_contact_envelope_finger_contact_offset_invalid",
    )
    _exact_number(
        value.get("effective_contact_envelope_m"),
        expected=float(expected["effective_contact_envelope_m"]),
        error="adp009d_contact_envelope_effective_distance_invalid",
    )
    _exact_integer(
        value.get("sdf_resolution"),
        expected=APPROVED_CAN_SDF_RESOLUTION,
        error="adp009d_contact_envelope_sdf_resolution_invalid",
    )
    _exact_integer(
        value.get("sdf_subgrid_resolution"),
        expected=APPROVED_CAN_SDF_SUBGRID_RESOLUTION,
        error="adp009d_contact_envelope_sdf_subgrid_resolution_invalid",
    )
    return expected


def apply_contact_envelope_to_clearance(
    open_jaw_clearance_m: Any, *, envelope: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Subtract the approved envelope from a planned open-jaw clearance.

    The controls plan must plan against the clearance that actually remains
    once the contact-generating envelope is accounted for, not against the raw
    jaw aperture. Without this the envelope was defined and cross-checked but
    never applied, so a plan that leaves no usable clearance looked identical
    to a correct one.

    The envelope is re-validated rather than trusted, and a clearance the
    envelope fully consumes is a typed blocker: no positive clearance remains,
    so contact is certain rather than planned.
    """
    resolved_envelope = validate_contact_envelope(
        canonical_contact_envelope() if envelope is None else envelope
    )
    clearance = _finite_number(
        open_jaw_clearance_m,
        error="adp009d_contact_envelope_open_jaw_clearance_invalid",
    )
    if clearance <= 0:
        raise ContactEnvelopeError("adp009d_contact_envelope_open_jaw_clearance_invalid")
    effective = float(resolved_envelope["effective_contact_envelope_m"])
    remaining = clearance - effective
    if remaining <= 0:
        raise ContactEnvelopeError("adp009d_contact_envelope_exceeds_open_jaw_clearance")
    return {
        "schema_version": CONTACT_ENVELOPE_SCHEMA_VERSION,
        "open_jaw_clearance_m": clearance,
        "effective_contact_envelope_m": effective,
        "effective_contact_envelope_calculation": resolved_envelope[
            "effective_contact_envelope_calculation"
        ],
        "resolved_clearance_m": remaining,
    }


def validate_dynamics_receipt_contact_envelope(
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require a native arm-dynamics receipt to retain the approved envelope.

    A receipt that drops the resolved value, or carries one that drifted from
    the approved envelope, must block controls before policy execution. Left
    unchecked such a receipt would be scored as a policy result when it is
    really a runtime configuration mismatch.
    """
    if not isinstance(receipt, Mapping):
        raise ContactEnvelopeError("adp009d_dynamics_receipt_contact_envelope_missing")
    expected = canonical_contact_envelope()
    for field in (
        "effective_contact_envelope_m",
        "effective_contact_envelope_calculation",
    ):
        if field not in receipt:
            raise ContactEnvelopeError(
                "adp009d_dynamics_receipt_contact_envelope_missing"
            )
    _exact_number(
        receipt.get("effective_contact_envelope_m"),
        expected=float(expected["effective_contact_envelope_m"]),
        error="adp009d_dynamics_receipt_contact_envelope_mismatch",
    )
    if (
        receipt.get("effective_contact_envelope_calculation")
        != expected["effective_contact_envelope_calculation"]
    ):
        raise ContactEnvelopeError(
            "adp009d_dynamics_receipt_contact_envelope_mismatch"
        )
    return {
        "effective_contact_envelope_m": float(expected["effective_contact_envelope_m"]),
        "effective_contact_envelope_calculation": expected[
            "effective_contact_envelope_calculation"
        ],
    }


__all__ = [
    "APPROVED_CAN_SDF_MARGIN_M",
    "APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M",
    "APPROVED_CAN_SDF_RESOLUTION",
    "APPROVED_CAN_SDF_SUBGRID_RESOLUTION",
    "CONTACT_ENVELOPE_CALCULATION",
    "CONTACT_ENVELOPE_SCHEMA_VERSION",
    "ContactEnvelopeError",
    "FINGER_COLLIDER_CONTACT_OFFSET_M",
    "FINGER_COLLIDER_CONTACT_OFFSET_SOURCE",
    "apply_contact_envelope_to_clearance",
    "canonical_contact_envelope",
    "contact_envelope_from_harness_manifest",
    "contact_envelope_from_physx_sdf_settings",
    "validate_contact_envelope",
    "validate_dynamics_receipt_contact_envelope",
]
