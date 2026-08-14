"""The scripted-control receipt must retain the approved contact envelope.

`PRODUCTION_WEBSITE_LAUNCH.md`: "Every native arm-dynamics receipt retains the
same envelope. A mismatch blocks controls before policy execution and must
remain a typed runtime blocker rather than becoming a policy result."

`adp009d_scripted_control_ik_receipt.v1` is that receipt, and it carried no
envelope at all. Without it a run whose colliders were built against a
different margin produced controls evidence indistinguishable from a correct
one, and the mismatch could only surface later as an unexplained policy score.

The receipt is built inside the Isaac runtime, so the envelope binding lives in
a pure helper that can be checked without a simulator.
"""

import pytest

from blueprint_pipeline.adp009d_contact_envelope import (
    CONTACT_ENVELOPE_CALCULATION,
    ContactEnvelopeError,
    canonical_contact_envelope,
    contact_envelope_receipt_fields,
    validate_dynamics_receipt_contact_envelope,
)

EFFECTIVE_ENVELOPE_M = 0.010


def test_receipt_fields_carry_value_and_calculation() -> None:
    fields = contact_envelope_receipt_fields()
    assert fields["effective_contact_envelope_m"] == EFFECTIVE_ENVELOPE_M
    assert fields["effective_contact_envelope_calculation"] == CONTACT_ENVELOPE_CALCULATION


def test_receipt_fields_are_exactly_what_the_validator_requires() -> None:
    """Producer and consumer must agree, or the seam is decorative."""
    receipt = {
        "schema_version": "adp009d_scripted_control_ik_receipt.v1",
        **contact_envelope_receipt_fields(),
    }
    validated = validate_dynamics_receipt_contact_envelope(receipt)
    assert validated["effective_contact_envelope_m"] == EFFECTIVE_ENVELOPE_M


def test_receipt_fields_match_the_canonical_envelope() -> None:
    envelope = canonical_contact_envelope()
    fields = contact_envelope_receipt_fields()
    assert fields["effective_contact_envelope_m"] == envelope["effective_contact_envelope_m"]
    assert (
        fields["effective_contact_envelope_calculation"]
        == envelope["effective_contact_envelope_calculation"]
    )


def test_a_receipt_built_without_the_helper_is_still_refused() -> None:
    """The validator must not be satisfied by an envelope-free receipt."""
    with pytest.raises(ContactEnvelopeError, match="dynamics_receipt_contact_envelope_missing"):
        validate_dynamics_receipt_contact_envelope(
            {"schema_version": "adp009d_scripted_control_ik_receipt.v1"}
        )


def test_isaac_runtime_binds_the_envelope_into_the_scripted_control_receipt() -> None:
    """Pin the actual producer, not just the helper it should call."""
    from pathlib import Path

    import blueprint_pipeline.adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    marker = '"schema_version": "adp009d_scripted_control_ik_receipt.v1"'
    assert marker in source
    block = source.split(marker, 1)[1][:600]
    assert "contact_envelope_receipt_fields()" in block, (
        "the scripted-control receipt must bind the approved contact envelope"
    )
