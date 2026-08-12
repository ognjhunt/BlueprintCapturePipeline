"""The two contact-envelope seams the ADP-009D controls gate depends on.

`PRODUCTION_WEBSITE_LAUNCH.md` states both requirements plainly: "the v12
controls plan subtracts that envelope from the open-jaw clearance" and "every
native arm-dynamics receipt retains the same envelope", with a mismatch
blocking controls before policy execution as a typed runtime blocker.

Neither existed as code. A repo-wide search on 2026-08-12 found
`effective_contact_envelope_m` in exactly two places: the module that defines
it and that module's own test. `adp009d_live_readiness` checks only that
envelopes supplied from outside match *each other*, so an envelope that is
never applied to a clearance, or a dynamics receipt that silently drops it,
was indistinguishable from a correct run.

These functions transcribe the runbook, they do not invent physics: the
subtraction is the single subtraction the runbook describes, and the retention
check asserts the exact fields the canonical envelope already publishes.
"""

import pytest

from blueprint_pipeline.adp009d_contact_envelope import (
    CONTACT_ENVELOPE_CALCULATION,
    ContactEnvelopeError,
    apply_contact_envelope_to_clearance,
    canonical_contact_envelope,
    validate_dynamics_receipt_contact_envelope,
)

# 2.5 mm SDF margin + 2.5 mm narrow band + 5 mm finger contact offset.
EFFECTIVE_ENVELOPE_M = 0.010


def _dynamics_receipt(**overrides) -> dict:
    envelope = canonical_contact_envelope()
    receipt = {
        "schema_version": "adp009d_native_arm_dynamics_receipt.v1",
        "effective_contact_envelope_m": envelope["effective_contact_envelope_m"],
        "effective_contact_envelope_calculation": envelope[
            "effective_contact_envelope_calculation"
        ],
    }
    receipt.update(overrides)
    return receipt


# --- seam one: planner clearance consumes the envelope ---------------------

def test_subtracts_the_envelope_from_the_open_jaw_clearance() -> None:
    resolved = apply_contact_envelope_to_clearance(0.050)
    assert resolved["open_jaw_clearance_m"] == 0.050
    assert resolved["effective_contact_envelope_m"] == EFFECTIVE_ENVELOPE_M
    assert resolved["resolved_clearance_m"] == pytest.approx(0.040)


def test_retains_the_calculation_string_with_the_resolved_clearance() -> None:
    """A number without its derivation cannot be audited after the fact."""
    resolved = apply_contact_envelope_to_clearance(0.050)
    assert resolved["effective_contact_envelope_calculation"] == CONTACT_ENVELOPE_CALCULATION


@pytest.mark.parametrize("clearance", [0.010, 0.005])
def test_blocks_when_the_envelope_consumes_the_whole_clearance(clearance) -> None:
    """No positive clearance remains, so contact is certain rather than planned."""
    with pytest.raises(ContactEnvelopeError, match="contact_envelope_exceeds_open_jaw_clearance"):
        apply_contact_envelope_to_clearance(clearance)


@pytest.mark.parametrize("clearance", [0.0, -0.001, None, float("nan"), "wide"])
def test_refuses_a_clearance_that_is_not_a_positive_number(clearance) -> None:
    with pytest.raises(ContactEnvelopeError):
        apply_contact_envelope_to_clearance(clearance)


def test_rejects_a_tampered_envelope_rather_than_trusting_the_caller() -> None:
    tampered = canonical_contact_envelope() | {"effective_contact_envelope_m": 0.001}
    with pytest.raises(ContactEnvelopeError):
        apply_contact_envelope_to_clearance(0.050, envelope=tampered)


# --- seam two: dynamics receipts retain the envelope -----------------------

def test_admits_a_receipt_that_retains_value_and_calculation() -> None:
    validated = validate_dynamics_receipt_contact_envelope(_dynamics_receipt())
    assert validated["effective_contact_envelope_m"] == EFFECTIVE_ENVELOPE_M
    assert validated["effective_contact_envelope_calculation"] == CONTACT_ENVELOPE_CALCULATION


def test_blocks_a_receipt_that_drops_the_resolved_value() -> None:
    receipt = _dynamics_receipt()
    del receipt["effective_contact_envelope_m"]
    with pytest.raises(ContactEnvelopeError, match="dynamics_receipt_contact_envelope_missing"):
        validate_dynamics_receipt_contact_envelope(receipt)


def test_blocks_a_receipt_that_drops_the_calculation_string() -> None:
    receipt = _dynamics_receipt()
    del receipt["effective_contact_envelope_calculation"]
    with pytest.raises(ContactEnvelopeError, match="dynamics_receipt_contact_envelope_missing"):
        validate_dynamics_receipt_contact_envelope(receipt)


def test_blocks_a_receipt_whose_envelope_drifted_from_the_approved_value() -> None:
    """A drifted envelope is a runtime blocker, never a policy result."""
    receipt = _dynamics_receipt(effective_contact_envelope_m=0.0125)
    with pytest.raises(ContactEnvelopeError, match="dynamics_receipt_contact_envelope_mismatch"):
        validate_dynamics_receipt_contact_envelope(receipt)


def test_blocks_a_receipt_whose_calculation_string_drifted() -> None:
    receipt = _dynamics_receipt(effective_contact_envelope_calculation="sdf_margin_only")
    with pytest.raises(ContactEnvelopeError, match="dynamics_receipt_contact_envelope_mismatch"):
        validate_dynamics_receipt_contact_envelope(receipt)


def test_blocks_a_receipt_that_is_not_a_mapping() -> None:
    with pytest.raises(ContactEnvelopeError):
        validate_dynamics_receipt_contact_envelope(None)
