"""Pinned protocol-v4 joint-name/order contract for the official GEAR-SONIC stack.

The official ``/opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml`` model
drives the G1 with 29 body joints plus 7 joints per Inspire hand. The names and
order below are pinned from the official Unitree G1 29-DOF layout already used
across this repository (``UNITREE_RL_GYM_LEG_JOINT_NAMES`` and
``UNITREE_G1_SONIC_STATE_JOINT_GROUPS`` in
``mujoco_g1_wam_vla_policy_endpoint_eval`` and ``JOINT_NAMES`` in
``official_g1_policy_handoff``). Controller results and FK/Isaac target
application must validate against this contract by name; positional-only
mappings are rejected fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

JOINT_ORDER_SCHEMA_VERSION = "gear_sonic_joint_order.protocol_v4.v1"
PINNED_WBC_SOURCE_REVISION = "6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b"

PROTOCOL_V4_BODY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

PROTOCOL_V4_LEFT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)

PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

PROTOCOL_V4_FULL_JOINT_ORDER: tuple[str, ...] = (
    PROTOCOL_V4_BODY_JOINT_NAMES
    + PROTOCOL_V4_LEFT_HAND_JOINT_NAMES
    + PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES
)


def compute_mapping_digest(
    *,
    schema_version: str,
    body_joint_names: Sequence[str],
    left_hand_joint_names: Sequence[str],
    right_hand_joint_names: Sequence[str],
) -> str:
    """Canonical digest over the joint-order mapping the controller applied."""

    payload = {
        "schema_version": str(schema_version),
        "body_joint_names": [str(item) for item in body_joint_names],
        "left_hand_joint_names": [str(item) for item in left_hand_joint_names],
        "right_hand_joint_names": [str(item) for item in right_hand_joint_names],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


PROTOCOL_V4_MAPPING_DIGEST = compute_mapping_digest(
    schema_version=JOINT_ORDER_SCHEMA_VERSION,
    body_joint_names=PROTOCOL_V4_BODY_JOINT_NAMES,
    left_hand_joint_names=PROTOCOL_V4_LEFT_HAND_JOINT_NAMES,
    right_hand_joint_names=PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES,
)


def pinned_controller_joint_order(controller_revision: str) -> dict[str, Any]:
    """Return the authoritative MuJoCo order for the exact pinned WBC source.

    The official ``g1_debug`` wire payload carries numeric arrays in MuJoCo
    order but does not repeat joint names on every state message. The names are
    therefore bound to the immutable WBC source revision installed in the
    sealed image, not trusted from caller-supplied state fields.
    """
    if str(controller_revision or "").strip().lower() != PINNED_WBC_SOURCE_REVISION:
        raise ValueError("official_gear_sonic_controller_revision_mismatch")
    return {
        "schema_version": JOINT_ORDER_SCHEMA_VERSION,
        "body_joint_names": list(PROTOCOL_V4_BODY_JOINT_NAMES),
        "left_hand_joint_names": list(PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        "right_hand_joint_names": list(PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
        "mapping_source": "pinned_wbc_mujoco_order",
        "controller_revision": PINNED_WBC_SOURCE_REVISION,
    }


def _string_names(value: Any, *, source: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"official_gear_sonic_{source}_joint_names_invalid")
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"official_gear_sonic_{source}_joint_names_invalid")
        names.append(item)
    return names


def _classify_names(
    names: Sequence[str], *, expected: Sequence[str], source: str, ordered: bool
) -> tuple[str, ...]:
    values = _string_names(names, source=source)
    if len(set(values)) != len(values):
        raise ValueError(f"official_gear_sonic_{source}_joint_names_duplicate")
    expected_set = set(expected)
    if any(value not in expected_set for value in values):
        raise ValueError(f"official_gear_sonic_{source}_joint_names_unknown")
    provided = set(values)
    if any(name not in provided for name in expected):
        raise ValueError(f"official_gear_sonic_{source}_joint_names_missing")
    if ordered and tuple(values) != tuple(expected):
        raise ValueError(f"official_gear_sonic_{source}_joint_names_permuted")
    return tuple(values)


def validate_ordered_joint_names(
    names: Sequence[str], *, expected: Sequence[str], source: str
) -> tuple[str, ...]:
    """Require ``names`` to equal ``expected`` exactly (set and order)."""

    return _classify_names(names, expected=expected, source=source, ordered=True)


def validate_full_joint_order(
    names: Sequence[str], *, source: str = "executor"
) -> tuple[str, ...]:
    """Require the full 43-joint protocol-v4 order, exactly."""

    return _classify_names(
        names, expected=PROTOCOL_V4_FULL_JOINT_ORDER, source=source, ordered=True
    )


def validate_model_joint_names(
    names: Sequence[str], *, source: str = "mujoco_model"
) -> tuple[str, ...]:
    """Require the model to expose exactly the pinned joint set.

    Order inside the model is free because targets are applied by joint name,
    never positionally.
    """

    return _classify_names(
        names, expected=PROTOCOL_V4_FULL_JOINT_ORDER, source=source, ordered=False
    )


def validate_controller_joint_order(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the joint-order schema a protocol-v4 controller result carries.

    Positional-only results (no ordered joint names) are rejected fail-closed,
    as are duplicate, unknown, missing, or permuted mappings and any mapping
    digest that does not match the pinned protocol-v4 contract.
    """

    schema_version = state.get("joint_order_schema_version")
    if schema_version is None:
        raise ValueError("official_gear_sonic_controller_joint_order_schema_version_missing")
    if str(schema_version) != JOINT_ORDER_SCHEMA_VERSION:
        raise ValueError(
            "official_gear_sonic_controller_joint_order_schema_version_unsupported"
        )
    segments = (
        ("body_joint_names", PROTOCOL_V4_BODY_JOINT_NAMES, "controller_body"),
        ("left_hand_joint_names", PROTOCOL_V4_LEFT_HAND_JOINT_NAMES, "controller_left_hand"),
        ("right_hand_joint_names", PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES, "controller_right_hand"),
    )
    for key, _, _ in segments:
        if state.get(key) is None:
            raise ValueError(
                "official_gear_sonic_controller_joint_names_missing_positional_only_rejected"
            )
    if state.get("mapping_digest") is None:
        raise ValueError("official_gear_sonic_controller_mapping_digest_missing")
    validated: dict[str, Any] = {"schema_version": JOINT_ORDER_SCHEMA_VERSION}
    for key, expected, source in segments:
        validated[key] = list(
            validate_ordered_joint_names(state[key], expected=expected, source=source)
        )
    digest = str(state.get("mapping_digest"))
    expected_digest = compute_mapping_digest(
        schema_version=JOINT_ORDER_SCHEMA_VERSION,
        body_joint_names=validated["body_joint_names"],
        left_hand_joint_names=validated["left_hand_joint_names"],
        right_hand_joint_names=validated["right_hand_joint_names"],
    )
    if digest != expected_digest or digest != PROTOCOL_V4_MAPPING_DIGEST:
        raise ValueError("official_gear_sonic_controller_mapping_digest_mismatch")
    validated["mapping_digest"] = digest
    return validated


def build_isaac_dof_mapping(
    live_joint_names: Sequence[str], *, source: str = "isaac_articulation"
) -> list[dict[str, Any]]:
    """Map protocol-v4 joint order onto a live Isaac articulation DOF list.

    The live articulation must expose exactly the pinned joint set; the
    returned rows carry the explicit permutation so targets are applied by
    name, never positionally. Expose this as the validation hook for live
    Isaac articulations before applying any targets.
    """

    names = _classify_names(
        live_joint_names,
        expected=PROTOCOL_V4_FULL_JOINT_ORDER,
        source=source,
        ordered=False,
    )
    dof_index_by_name = {name: index for index, name in enumerate(names)}
    return [
        {
            "joint_name": joint_name,
            "protocol_index": protocol_index,
            "articulation_dof_index": dof_index_by_name[joint_name],
        }
        for protocol_index, joint_name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)
    ]
