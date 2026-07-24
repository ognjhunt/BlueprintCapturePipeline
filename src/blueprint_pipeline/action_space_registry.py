"""Registered action spaces for multi-embodiment evaluation.

Two independent copies of a hardcoded ``7`` governed action handling in this
repository: the normalization contract and the Cosmos command adapter both
required exactly seven dimensions in the SC3 delta end-effector layout. That is
correct for the SC3 cell and wrong as a platform invariant -- the executing
Unitree G1 whole-body action is 78-dimensional, so the very embodiment the
pipeline drives could not be described by its own action contract.

The fix is not to relax the check. A 7-D delta end-effector vector, a 43-joint
arm/hand chunk and a 78-D whole-body command are different physical objects that
happen to share a Python type, and letting them flow through one adapter because
they are all arrays of numbers is exactly the failure mode the strict contract
was defending against.

So dimensionality becomes a property of a *registered action space* rather than
a constant. Each space pins its identifier, dimension, ordered component names,
units and representation aliases. Callers name the space they intend; an
unregistered name fails closed, and a vector is validated against that space's
exact layout with the same strictness the 7-D path always had.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


ACTION_SPACE_SCHEMA_VERSION = "action_space.v1"

SC3_7D_DELTA_EE = "sc3_7d_delta_end_effector.v1"
UNITREE_G1_WHOLE_BODY_78D = "unitree_g1_whole_body_78d.v1"
UNITREE_G1_ARM_HAND_43D = "unitree_g1_arm_hand_43d.v1"


@dataclass(frozen=True)
class ActionSpace:
    """One registered, exactly-specified action layout."""

    action_schema_id: str
    dim: int
    representation: str
    order: tuple[str, ...]
    units: tuple[str, ...]
    representation_aliases: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def __post_init__(self) -> None:
        if self.order and len(self.order) != self.dim:
            raise ValueError(f"action_space_order_length_mismatch:{self.action_schema_id}")
        if self.units and len(self.units) != self.dim:
            raise ValueError(f"action_space_units_length_mismatch:{self.action_schema_id}")

    @property
    def dim_blocker(self) -> str:
        """Blocker emitted when a vector's width does not match this space."""

        return f"action_space_dim_must_equal_{self.dim}"

    def accepts_representation(self, representation: str) -> bool:
        text = str(representation or "").strip()
        return text == self.representation or text in self.representation_aliases

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SPACE_SCHEMA_VERSION,
            "action_schema_id": self.action_schema_id,
            "dim": self.dim,
            "representation": self.representation,
            "order": list(self.order),
            "units": list(self.units),
            "description": self.description,
        }


_SC3_ORDER = (
    "delta_x_m",
    "delta_y_m",
    "delta_z_m",
    "delta_roll_rad",
    "delta_pitch_rad",
    "delta_yaw_rad",
    "gripper_normalized",
)
_SC3_UNITS = ("m", "m", "m", "rad", "rad", "rad", "normalized")

# 64 motion tokens plus two 7-DOF hands, matching the executing controller's
# wire format in gear_sonic_official_zmq_executor.
_G1_WHOLE_BODY_ORDER = tuple(
    [f"motion_token_{index}" for index in range(64)]
    + [f"left_hand_{index}" for index in range(7)]
    + [f"right_hand_{index}" for index in range(7)]
)
_G1_WHOLE_BODY_UNITS = tuple(["normalized"] * 64 + ["rad"] * 14)

_G1_ARM_HAND_ORDER = tuple(
    [f"arm_joint_{index}_rad" for index in range(29)]
    + [f"left_hand_{index}" for index in range(7)]
    + [f"right_hand_{index}" for index in range(7)]
)
_G1_ARM_HAND_UNITS = tuple(["rad"] * 43)


ACTION_SPACES: dict[str, ActionSpace] = {
    SC3_7D_DELTA_EE: ActionSpace(
        action_schema_id=SC3_7D_DELTA_EE,
        dim=7,
        representation="7d_delta_end_effector_pose",
        order=_SC3_ORDER,
        units=_SC3_UNITS,
        representation_aliases=frozenset(
            {
                "7d_delta_end_effector_pose",
                "sc3_7d_delta_end_effector_pose",
                "ee_delta_pose_gripper",
            }
        ),
        description="SC3-Eval delta translation (3), delta rotation (3), gripper (1)",
    ),
    UNITREE_G1_WHOLE_BODY_78D: ActionSpace(
        action_schema_id=UNITREE_G1_WHOLE_BODY_78D,
        dim=78,
        representation="unitree_g1_whole_body_motion_tokens_plus_hands",
        order=_G1_WHOLE_BODY_ORDER,
        units=_G1_WHOLE_BODY_UNITS,
        representation_aliases=frozenset({"gear_sonic_whole_body_78d"}),
        description="64 GEAR-SONIC motion tokens plus two 7-DOF hands",
    ),
    UNITREE_G1_ARM_HAND_43D: ActionSpace(
        action_schema_id=UNITREE_G1_ARM_HAND_43D,
        dim=43,
        representation="unitree_g1_arm_hand_joint_positions",
        order=_G1_ARM_HAND_ORDER,
        units=_G1_ARM_HAND_UNITS,
        representation_aliases=frozenset({"unitree_g1_whole_body_arm_hand_chunks_v1"}),
        description="29 body/arm joints plus two 7-DOF hands, joint positions",
    ),
}

DEFAULT_ACTION_SPACE_ID = SC3_7D_DELTA_EE


class UnknownActionSpaceError(KeyError):
    """Raised when a caller names an action space that is not registered."""


def get_action_space(action_schema_id: str | None = None) -> ActionSpace:
    """Look up a registered action space, failing closed on unknown ids."""

    name = str(action_schema_id or DEFAULT_ACTION_SPACE_ID).strip()
    space = ACTION_SPACES.get(name)
    if space is None:
        raise UnknownActionSpaceError(
            f"action_space_not_registered:{name}; "
            f"registered={sorted(ACTION_SPACES)}"
        )
    return space


def registered_action_space_ids() -> list[str]:
    return sorted(ACTION_SPACES)


def validate_action_vector(
    vector: Sequence[Any], *, action_schema_id: str | None = None
) -> list[str]:
    """Validate one action vector against a registered space.

    Returns blockers rather than raising so callers can accumulate them
    alongside their own contract checks.
    """

    space = get_action_space(action_schema_id)
    blockers: list[str] = []
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        return ["action_vector_missing_or_not_a_sequence"]
    if len(vector) != space.dim:
        blockers.append(space.dim_blocker)
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            blockers.append("action_vector_non_numeric")
            break
        if not math.isfinite(value):
            blockers.append("action_vector_non_finite")
            break
    return sorted(set(blockers))


def validate_action_space_contract(
    contract: Mapping[str, Any], *, action_schema_id: str | None = None
) -> list[str]:
    """Validate a declared action-space contract against its registered space."""

    space = get_action_space(action_schema_id)
    blockers: list[str] = []
    if not contract:
        return ["action_space_contract_missing"]
    declared_dim = contract.get("dim")
    if not isinstance(declared_dim, int) or isinstance(declared_dim, bool):
        declared_dim = None
    if declared_dim != space.dim:
        blockers.append(space.dim_blocker)
    representation = str(
        contract.get("representation")
        or contract.get("name")
        or contract.get("layout_id")
        or ""
    ).strip()
    if not space.accepts_representation(representation):
        blockers.append(f"action_representation_not_{space.action_schema_id}")
    if space.order and tuple(contract.get("order") or []) != space.order:
        blockers.append("action_dimension_order_missing_or_invalid")
    if space.units and tuple(contract.get("units") or []) != space.units:
        blockers.append("action_dimension_units_missing_or_invalid")
    return sorted(set(blockers))
