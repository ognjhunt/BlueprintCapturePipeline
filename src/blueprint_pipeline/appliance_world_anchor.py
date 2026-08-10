"""Check that a floor-standing appliance is actually standing on something.

A rigid task object is meant to be free: the whole point of a can is that the
arm can move it. An appliance is the opposite. A refrigerator that is not
anchored is a 90 kg body released at the origin, and it falls, settles and
rocks - and every joint hanging off it swings while it does.

The 840796 twin had an articulation root, two revolute hinges and no anchor of
any kind. It opened its own door 35 degrees in the first few physics steps,
identically across four successive twins, because nothing about the anchoring
ever changed and none of the four geometry fixes was the fault.

Two mechanisms count. A fixed joint from the world to the support link is the
explicit one. A kinematic support link is the other: immovable and still
collidable, which is what a floor-standing body needs. Anchoring some other
link does not count - a door bolted to the world is not an appliance standing
up.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


APPLIANCE_WORLD_ANCHOR_SCHEMA_VERSION = "appliance_world_anchor.v1"


class ApplianceWorldAnchorError(ValueError):
    """Stable, sorted anchor-assessment failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def assess_world_anchor(
    *,
    support_link_path: str,
    fixed_joint_paths: Sequence[str],
    rigid_body_links: Sequence[str],
    kinematic_links: Sequence[str] = (),
    fixed_joint_bodies: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Whether the appliance's support link is held against gravity."""

    support = str(support_link_path)
    bodies = {str(value) for value in rigid_body_links}
    if support not in bodies:
        raise ApplianceWorldAnchorError(
            [
                "appliance_world_anchor_support_link_not_a_rigid_body:"
                f"{support}:bodies={','.join(sorted(bodies))}"
            ]
        )

    if support in {str(value) for value in kinematic_links}:
        return {
            "schema_version": APPLIANCE_WORLD_ANCHOR_SCHEMA_VERSION,
            "anchored": True,
            "mechanism": "kinematic_support_link",
            "support_link_path": support,
            "reasons": [],
        }

    joint_bodies = dict(fixed_joint_bodies or {})
    for joint in fixed_joint_paths:
        targets = {str(value) for value in joint_bodies.get(str(joint), ())}
        if support in targets:
            return {
                "schema_version": APPLIANCE_WORLD_ANCHOR_SCHEMA_VERSION,
                "anchored": True,
                "mechanism": "fixed_joint",
                "fixed_joint_path": str(joint),
                "support_link_path": support,
                "reasons": [],
            }

    return {
        "schema_version": APPLIANCE_WORLD_ANCHOR_SCHEMA_VERSION,
        "anchored": False,
        "mechanism": None,
        "support_link_path": support,
        "reasons": [
            "appliance_world_anchor_support_link_is_free:"
            f"{support}:fixed_joints={len(list(fixed_joint_paths))}"
        ],
        "claim_boundary": {
            "an_unanchored_appliance_moves_before_any_robot_does": True,
        },
    }


__all__ = [
    "APPLIANCE_WORLD_ANCHOR_SCHEMA_VERSION",
    "ApplianceWorldAnchorError",
    "assess_world_anchor",
]
