from __future__ import annotations

import inspect

from blueprint_pipeline import adp009d_franka_vast


def test_the_transport_can_be_told_which_arena_payload_it_carries() -> None:
    """Both Arena payloads share a transport but not their required entries.

    Hardcoding the kind here meant an articulated bundle was validated against
    the rigid lane's entry list and refused for missing a SAGE overlay it can
    never produce.
    """

    signature = inspect.signature(
        adp009d_franka_vast.run_adp009d_native_microcheck_vast
    )

    assert "provider_bundle_kind" in signature.parameters


def test_it_still_defaults_to_the_rigid_kind() -> None:
    """The qualified lane must behave identically when nothing is passed."""

    signature = inspect.signature(
        adp009d_franka_vast.run_adp009d_native_microcheck_vast
    )

    assert signature.parameters["provider_bundle_kind"].default == "adp009d_isaac"


def test_only_the_two_arena_kinds_are_accepted() -> None:
    """This transport carries Arena payloads; other kinds have other transports."""

    assert adp009d_franka_vast.SUPPORTED_BUNDLE_KINDS == (
        "adp009d_isaac",
        "adp009d_articulated_arena",
    )
