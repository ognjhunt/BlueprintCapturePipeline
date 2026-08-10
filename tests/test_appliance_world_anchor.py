"""A floor-standing appliance must be anchored, or it falls over."""

from __future__ import annotations

import pytest

from blueprint_pipeline.appliance_world_anchor import (
    ApplianceWorldAnchorError,
    assess_world_anchor,
)


def test_a_free_floating_articulation_is_refused():
    """The 840796 twin: articulation root, two hinges, nothing holding it up.

    It drops at step zero, settles, and the doors swing from the base motion -
    35 degrees before the arm arrives, identical across four twins because
    nothing about the anchoring ever changed. Four geometry fixes could not
    touch it because none of them was the fault.
    """

    verdict = assess_world_anchor(
        support_link_path="/Asset/cabinet",
        fixed_joint_paths=[],
        kinematic_links=[],
        rigid_body_links=["/Asset/cabinet", "/Asset/upper_door", "/Asset/lower_door"],
    )

    assert verdict["anchored"] is False
    assert "support_link_is_free" in ";".join(verdict["reasons"])


def test_a_fixed_joint_on_the_support_link_anchors_it():
    verdict = assess_world_anchor(
        support_link_path="/Asset/cabinet",
        fixed_joint_paths=["/Asset/joints/world_anchor"],
        fixed_joint_bodies={"/Asset/joints/world_anchor": ["/Asset/cabinet"]},
        kinematic_links=[],
        rigid_body_links=["/Asset/cabinet"],
    )

    assert verdict["anchored"] is True
    assert verdict["mechanism"] == "fixed_joint"


def test_a_kinematic_support_link_also_counts():
    """Immovable and still collidable is what a floor-standing body needs."""

    verdict = assess_world_anchor(
        support_link_path="/Asset/cabinet",
        fixed_joint_paths=[],
        kinematic_links=["/Asset/cabinet"],
        rigid_body_links=["/Asset/cabinet"],
    )

    assert verdict["anchored"] is True
    assert verdict["mechanism"] == "kinematic_support_link"


def test_a_fixed_joint_on_the_wrong_body_does_not_count():
    """Anchoring a door is not anchoring the appliance."""

    verdict = assess_world_anchor(
        support_link_path="/Asset/cabinet",
        fixed_joint_paths=["/Asset/joints/anchor"],
        fixed_joint_bodies={"/Asset/joints/anchor": ["/Asset/upper_door"]},
        kinematic_links=[],
        rigid_body_links=["/Asset/cabinet", "/Asset/upper_door"],
    )

    assert verdict["anchored"] is False


def test_a_support_link_that_is_not_a_rigid_body_refuses():
    with pytest.raises(ApplianceWorldAnchorError):
        assess_world_anchor(
            support_link_path="/Asset/cabinet",
            fixed_joint_paths=[],
            kinematic_links=[],
            rigid_body_links=["/Asset/upper_door"],
        )
