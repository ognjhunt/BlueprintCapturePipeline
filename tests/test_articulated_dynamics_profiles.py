from __future__ import annotations

import pytest

from blueprint_pipeline.articulated_dynamics_profiles import (
    DYNAMICS_PROFILE_REGISTRY_VERSION,
    ArticulatedDynamicsProfileError,
    available_profile_ids,
    resolve_dynamics_profile,
)
from blueprint_pipeline.articulated_dynamics_realism import (
    evaluate_articulated_dynamics_realism,
)


def test_a_class_we_have_researched_resolves_to_its_cited_band() -> None:
    profile = resolve_dynamics_profile("household_refrigerator_door")

    assert "BioRob 2010" in profile["measurement_source"]
    assert profile["sustained_torque_n_m"] == [0.4, 2.1]


def test_an_unresearched_class_fails_closed_rather_than_defaulting() -> None:
    """Reaching for a nearby profile is how a cupboard gets a fridge's gasket.

    The whole value of the band is that someone measured *this* kind of object.
    Substituting a neighbour silently converts "we checked" into "we guessed",
    and the receipt would look identical either way.
    """

    with pytest.raises(ArticulatedDynamicsProfileError) as excinfo:
        resolve_dynamics_profile("dishwasher_door")

    assert any("profile_not_researched" in e for e in excinfo.value.errors)
    assert any("dishwasher_door" in e for e in excinfo.value.errors)


def test_the_error_lists_what_has_been_researched_so_far() -> None:
    """A new class needs a person to go and measure, so say what exists."""

    with pytest.raises(ArticulatedDynamicsProfileError) as excinfo:
        resolve_dynamics_profile("oven_door")

    joined = ";".join(excinfo.value.errors)
    assert "household_refrigerator_door" in joined


def test_every_registered_profile_carries_a_citation_and_a_sample() -> None:
    """An uncited profile in the registry would be laundered by every use."""

    for profile_id in available_profile_ids():
        profile = resolve_dynamics_profile(profile_id)
        assert profile["measurement_source"].strip()
        assert profile["sample_description"].strip()
        assert profile["profile_id"] == profile_id


def test_every_registered_profile_is_accepted_by_the_gate_it_feeds() -> None:
    """A malformed band would only surface on the object that first used it."""

    for profile_id in available_profile_ids():
        profile = resolve_dynamics_profile(profile_id)
        receipt = evaluate_articulated_dynamics_realism(
            lever_arm_m=sum(profile["lever_arm_m"]) / 2.0,
            joint_damping_n_m_s_per_rad=1.0,
            breakaway_torque_n_m=sum(profile["breakaway_torque_n_m"]) / 2.0,
            breakaway_angular_width_degrees=(
                sum(profile["breakaway_angular_width_degrees"]) / 2.0
            ),
            nominal_open_angle_degrees=50.0,
            nominal_sweep_duration_s=2.0,
            reference_profile=profile,
        )
        assert receipt["reference_profile"]["profile_id"] == profile_id


def test_the_registry_is_immutable_from_the_outside() -> None:
    """A caller editing a resolved profile must not reach the next caller."""

    first = resolve_dynamics_profile("household_refrigerator_door")
    first["sustained_torque_n_m"] = [0.0, 999.0]

    second = resolve_dynamics_profile("household_refrigerator_door")
    assert second["sustained_torque_n_m"] == [0.4, 2.1]


def test_the_registry_declares_its_version() -> None:
    assert DYNAMICS_PROFILE_REGISTRY_VERSION.startswith(
        "articulated_dynamics_profiles."
    )


def test_a_near_miss_is_named_so_a_typo_does_not_read_as_unresearched() -> None:
    """"refrigerator_door" and the real key differ by one word.

    Failing closed is right, but failing closed identically for "someone
    mistyped a key" and "nobody has ever measured this class" sends a person
    off to do a literature search that was already done.
    """

    with pytest.raises(ArticulatedDynamicsProfileError) as excinfo:
        resolve_dynamics_profile("refrigerator_door")

    joined = ";".join(excinfo.value.errors)
    assert "did_you_mean:household_refrigerator_door" in joined


def test_a_genuinely_new_class_is_not_given_a_bogus_suggestion() -> None:
    """Suggesting the only entry we have would be worse than saying nothing."""

    with pytest.raises(ArticulatedDynamicsProfileError) as excinfo:
        resolve_dynamics_profile("pallet_jack_lever")

    assert not any("did_you_mean" in error for error in excinfo.value.errors)


def test_a_miss_states_what_a_new_profile_has_to_supply() -> None:
    """The error is the work order, so it should say what the job is."""

    with pytest.raises(ArticulatedDynamicsProfileError) as excinfo:
        resolve_dynamics_profile("pallet_jack_lever")

    joined = ";".join(excinfo.value.errors)
    for field in ("breakaway_torque_n_m", "sustained_torque_n_m", "lever_arm_m"):
        assert field in joined
    assert "measurement_source" in joined
