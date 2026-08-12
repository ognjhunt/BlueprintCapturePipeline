from __future__ import annotations

import pytest

from blueprint_pipeline.adp009d_contact_envelope import (
    APPROVED_CAN_SDF_MARGIN_M,
    APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M,
    APPROVED_CAN_SDF_RESOLUTION,
    APPROVED_CAN_SDF_SUBGRID_RESOLUTION,
    ContactEnvelopeError,
    FINGER_COLLIDER_CONTACT_OFFSET_M,
    canonical_contact_envelope,
    contact_envelope_from_harness_manifest,
    contact_envelope_from_physx_sdf_settings,
    validate_contact_envelope,
)


def test_effective_contact_envelope_is_explicit_and_pinned() -> None:
    envelope = canonical_contact_envelope()

    assert envelope["sdf_margin_m"] == APPROVED_CAN_SDF_MARGIN_M == 0.0025
    assert envelope["sdf_narrow_band_thickness_m"] == (
        APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M
    ) == 0.0025
    assert envelope["finger_collider_contact_offset_m"] == (
        FINGER_COLLIDER_CONTACT_OFFSET_M
    ) == 0.005
    assert envelope["effective_contact_envelope_m"] == pytest.approx(0.01)
    assert validate_contact_envelope(envelope) == envelope


def test_physx_or_harness_drift_fails_before_controls() -> None:
    with pytest.raises(ContactEnvelopeError, match="sdf_margin_invalid"):
        contact_envelope_from_physx_sdf_settings(
            sdf_margin_m=0.01,
            sdf_narrow_band_thickness_m=APPROVED_CAN_SDF_NARROW_BAND_THICKNESS_M,
            sdf_resolution=APPROVED_CAN_SDF_RESOLUTION,
            sdf_subgrid_resolution=APPROVED_CAN_SDF_SUBGRID_RESOLUTION,
        )

    with pytest.raises(ContactEnvelopeError, match="harness_contact_offset_invalid"):
        contact_envelope_from_harness_manifest(
            {
                "physics": {"settings": {"contact_offset_m": 0.004}},
                "canonical_condition": {
                    "parameters": {"object_contact_offset_m": 0.005}
                },
            }
        )
