from __future__ import annotations

import pytest

from blueprint_pipeline.articulated_dynamics_realism import (
    DYNAMICS_REALISM_SCHEMA_VERSION,
    ArticulatedDynamicsRealismError,
    evaluate_articulated_dynamics_realism,
)


def _profile(**overrides) -> dict:
    """A measured band for one object class, carrying its own provenance."""

    profile = {
        "profile_id": "household_refrigerator_door",
        "measurement_source": (
            "Jain, Nguyen, Rath, Okerman & Kemp, 'The Complex Structure of Simple "
            "Devices', IEEE BioRob 2010, DOI 10.1109/BIOROB.2010.5626754"
        ),
        "sample_description": "29 doors and 15 drawers, 6 homes and 1 office",
        "breakaway_torque_n_m": [6.0, 28.0],
        "breakaway_angular_width_degrees": [3.0, 8.0],
        "sustained_torque_n_m": [0.4, 2.1],
        "lever_arm_m": [0.42, 0.70],
    }
    profile.update(overrides)
    return profile


def _evaluate(**overrides):
    arguments = {
        "lever_arm_m": 0.495,
        "joint_damping_n_m_s_per_rad": 2.5,
        "breakaway_torque_n_m": 12.0,
        "breakaway_angular_width_degrees": 5.0,
        "nominal_open_angle_degrees": 50.0,
        "nominal_sweep_duration_s": 2.0,
        "reference_profile": _profile(),
    }
    arguments.update(overrides)
    return evaluate_articulated_dynamics_realism(**arguments)


def test_constants_inside_the_measured_band_are_admitted() -> None:
    receipt = _evaluate()

    assert receipt["within_measured_band"] is True
    assert receipt["findings"] == []
    assert receipt["schema_version"] == DYNAMICS_REALISM_SCHEMA_VERSION


def test_the_authored_840796_damping_is_rejected_as_far_too_stiff() -> None:
    """14 N*m*s/rad is what the twin actually shipped with, and it is wrong.

    At the nominal sweep that is 6.1 N*m of steady resistance where the
    measured band tops out near 2.1 - a door that behaves like it is moving
    through treacle. A scripted positive would still pass, which is exactly why
    this has to be caught here rather than by watching the video.
    """

    receipt = _evaluate(joint_damping_n_m_s_per_rad=14.0)

    assert receipt["within_measured_band"] is False
    assert any("sustained_torque_above_measured" in f for f in receipt["findings"])
    assert receipt["observed"]["sustained_torque_n_m"] == pytest.approx(6.11, abs=0.02)


def test_a_missing_seal_detent_is_reported_in_its_own_right() -> None:
    """A fridge without its gasket is a cupboard.

    The measured peak is fifteen-odd times the sustained force and lives
    entirely in the first few degrees. Omitting it does not make the task
    slightly easier, it removes the part that makes it a fridge.
    """

    receipt = _evaluate(breakaway_torque_n_m=None)

    assert receipt["within_measured_band"] is False
    assert any("breakaway_torque_absent" in f for f in receipt["findings"])
    assert receipt["observed"]["peak_to_sustained_ratio"] is None


def test_a_lever_arm_outside_the_surveyed_range_says_which_way() -> None:
    """Too short and too long are different defects with different fixes."""

    short = _evaluate(lever_arm_m=0.15)
    long = _evaluate(lever_arm_m=1.20)

    assert any("lever_arm_below_measured_band" in f for f in short["findings"])
    assert any("lever_arm_above_measured_band" in f for f in long["findings"])


def test_a_profile_with_no_measurement_source_fails_closed() -> None:
    """An unsourced band is an opinion, and would launder one into a receipt."""

    with pytest.raises(ArticulatedDynamicsRealismError) as excinfo:
        _evaluate(reference_profile=_profile(measurement_source=""))

    assert any("measurement_source_missing" in e for e in excinfo.value.errors)


def test_an_inverted_band_fails_closed() -> None:
    with pytest.raises(ArticulatedDynamicsRealismError) as excinfo:
        _evaluate(reference_profile=_profile(sustained_torque_n_m=[2.1, 0.4]))

    assert any("band_invalid" in e for e in excinfo.value.errors)


def test_the_receipt_carries_the_citation_it_judged_against() -> None:
    receipt = _evaluate()

    assert "BioRob 2010" in receipt["reference_profile"]["measurement_source"]
    assert receipt["claim_boundary"]["band_is_published_measurement_not_our_own"] is True


def test_evaluation_is_deterministic() -> None:
    assert _evaluate() == _evaluate()


from blueprint_pipeline.articulated_dynamics_realism import seal_detent_torque  # noqa: E402


def test_the_seal_resists_hardest_at_fully_closed() -> None:
    assert seal_detent_torque(
        joint_angle_degrees=0.0, breakaway_torque_n_m=12.0, angular_width_degrees=5.0
    ) == pytest.approx(12.0)


def test_the_seal_is_gone_once_the_door_is_off_the_gasket() -> None:
    """Measured traces fall to the sustained level by roughly ten degrees.

    A seal that kept pulling through the whole sweep would be a spring, and
    would make the door close itself from any angle.
    """

    assert seal_detent_torque(
        joint_angle_degrees=12.0, breakaway_torque_n_m=12.0, angular_width_degrees=5.0
    ) == 0.0


def test_the_seal_decays_monotonically_across_its_width() -> None:
    previous = None
    for angle in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        torque = seal_detent_torque(
            joint_angle_degrees=angle,
            breakaway_torque_n_m=12.0,
            angular_width_degrees=5.0,
        )
        if previous is not None:
            assert torque < previous
        previous = torque


def test_the_seal_opposes_opening_in_both_directions() -> None:
    """A door hung the other way swings to negative angles and seals the same."""

    opening = seal_detent_torque(
        joint_angle_degrees=2.0, breakaway_torque_n_m=12.0, angular_width_degrees=5.0
    )
    mirrored = seal_detent_torque(
        joint_angle_degrees=-2.0, breakaway_torque_n_m=12.0, angular_width_degrees=5.0
    )
    assert mirrored == pytest.approx(-opening)


def test_a_zero_width_seal_fails_closed() -> None:
    with pytest.raises(ArticulatedDynamicsRealismError) as excinfo:
        seal_detent_torque(
            joint_angle_degrees=0.0,
            breakaway_torque_n_m=12.0,
            angular_width_degrees=0.0,
        )

    assert any("angular_width_invalid" in e for e in excinfo.value.errors)
