from __future__ import annotations

import random

import pytest

from blueprint_pipeline.g1_proprioception_map import (
    G1_CANONICAL_DOF_ALIASES,
    G1_CANONICAL_DOF_GROUPS,
    G1_SONIC_PROPRIOCEPTION_STATE_DIMS,
    resolve_g1_proprioception_map,
    validate_g1_sonic_state_dims,
)


def _body_29() -> list[tuple[str, float]]:
    names = [
        name
        for group in ("left_leg", "right_leg", "waist", "left_arm", "right_arm")
        for name in G1_CANONICAL_DOF_GROUPS[group]
    ]
    return [(name, 0.01 * index) for index, name in enumerate(names)]


def _hand_extended_43() -> list[tuple[str, float]]:
    inventory = _body_29()
    offset = len(inventory)
    for group in ("left_hand", "right_hand"):
        for name in G1_CANONICAL_DOF_GROUPS[group]:
            inventory.append((name, 0.01 * (offset + len(inventory))))
    return inventory


def _positions(inventory: list[tuple[str, float]], names: tuple[str, ...]) -> list[float]:
    lookup = dict(inventory)
    return [lookup[name] for name in names]


def test_realistic_29_dof_inventory_maps_deterministically():
    inventory = _body_29()
    first = resolve_g1_proprioception_map(inventory, require_hands=False)
    second = resolve_g1_proprioception_map(inventory, require_hands=False)
    assert first["status"] == "passed"
    assert first["blockers"] == []
    assert first == second
    assert first["group_values"]["left_leg"] == _positions(
        inventory, G1_CANONICAL_DOF_GROUPS["left_leg"]
    )
    assert first["group_values"]["right_arm"] == _positions(
        inventory, G1_CANONICAL_DOF_GROUPS["right_arm"]
    )
    assert first["dimensions"]["left_leg"] == 6
    assert first["dimensions"]["waist"] == 3
    assert first["dimensions"]["left_hand"] == 0
    assert len(first["mapping_digest"]) == 64
    assert len(first["observed_dof_inventory"]) == 29


def test_hand_extended_inventory_maps_deterministically():
    inventory = _hand_extended_43()
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "passed"
    assert resolution["group_values"]["left_hand"] == _positions(
        inventory, G1_CANONICAL_DOF_GROUPS["left_hand"]
    )
    assert resolution["dimensions"] == {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 7,
        "right_hand": 7,
    }


def test_shuffled_observation_order_preserves_canonical_group_order():
    inventory = _hand_extended_43()
    shuffled = list(inventory)
    random.Random(7).shuffle(shuffled)
    resolution = resolve_g1_proprioception_map(shuffled, require_hands=True)
    assert resolution["status"] == "passed"
    assert resolution["group_values"]["left_leg"] == _positions(
        inventory, G1_CANONICAL_DOF_GROUPS["left_leg"]
    )
    assert resolution["group_values"]["right_hand"] == _positions(
        inventory, G1_CANONICAL_DOF_GROUPS["right_hand"]
    )


def test_extra_arm_and_hand_dofs_do_not_enter_leg_vectors():
    inventory = _hand_extended_43()
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "passed"
    assert len(resolution["group_values"]["left_leg"]) == 6
    assert len(resolution["group_values"]["right_leg"]) == 6
    left_leg_names = {
        row["canonical_name"] for row in resolution["resolved_map"]["left_leg"]
    }
    assert left_leg_names == set(G1_CANONICAL_DOF_GROUPS["left_leg"])
    assert not any("shoulder" in name or "hand" in name for name in left_leg_names)


def test_unknown_extra_dof_is_recorded_unmapped_not_grouped():
    inventory = _hand_extended_43() + [("left_gripper_aux_joint", 0.9)]
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "passed"
    assert resolution["unmapped_observed_dofs"] == ["left_gripper_aux_joint"]
    assert 0.9 not in resolution["group_values"]["left_leg"]
    assert 0.9 not in resolution["group_values"]["left_hand"]


def test_missing_required_joint_blocks():
    inventory = [row for row in _hand_extended_43() if row[0] != "left_knee_joint"]
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "blocked"
    assert "g1_proprioception_required_dof_missing:left_knee_joint" in resolution["blockers"]
    assert resolution["mapping_digest"] is None
    assert resolution["group_values"] == {}


def test_duplicate_observed_dof_blocks():
    inventory = _hand_extended_43() + [("left_knee_joint", 0.5)]
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "blocked"
    assert "g1_proprioception_observed_dof_duplicate:left_knee_joint" in resolution["blockers"]


def test_alias_resolves_deliberately():
    inventory = [
        ("left_elbow_pitch_joint" if name == "left_elbow_joint" else name, value)
        for name, value in _hand_extended_43()
    ]
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "passed"
    rows = {
        row["canonical_name"]: row["observed_name"]
        for row in resolution["resolved_map"]["left_arm"]
    }
    assert rows["left_elbow_joint"] == "left_elbow_pitch_joint"


def test_alias_collision_blocks():
    inventory = _hand_extended_43() + [("left_elbow_pitch_joint", 0.8)]
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "blocked"
    assert "g1_proprioception_alias_collision:left_elbow_joint" in resolution["blockers"]


def test_left_right_swap_blocks():
    swapped = [
        (name.replace("left_", "right_", 1), value)
        for name, value in _hand_extended_43()
    ]
    resolution = resolve_g1_proprioception_map(swapped, require_hands=True)
    assert resolution["status"] == "blocked"
    assert any(
        blocker.startswith("g1_proprioception_observed_dof_duplicate:right_")
        for blocker in resolution["blockers"]
    )
    assert any(
        blocker.startswith("g1_proprioception_required_dof_missing:left_")
        for blocker in resolution["blockers"]
    )


def test_hands_required_blocks_29_dof_inventory():
    resolution = resolve_g1_proprioception_map(_body_29(), require_hands=True)
    assert resolution["status"] == "blocked"
    assert (
        "g1_proprioception_required_dof_missing:left_hand_thumb_0_joint"
        in resolution["blockers"]
    )


def test_partial_hand_inventory_blocks_even_when_hands_optional():
    inventory = _body_29() + [("left_hand_thumb_0_joint", 0.4)]
    resolution = resolve_g1_proprioception_map(inventory, require_hands=False)
    assert resolution["status"] == "blocked"
    assert (
        "g1_proprioception_required_dof_missing:left_hand_thumb_1_joint"
        in resolution["blockers"]
    )


def test_nonfinite_observed_position_blocks():
    inventory = _hand_extended_43()
    inventory[0] = (inventory[0][0], float("nan"))
    resolution = resolve_g1_proprioception_map(inventory, require_hands=True)
    assert resolution["status"] == "blocked"
    assert (
        "g1_proprioception_observed_position_invalid:left_hip_pitch_joint"
        in resolution["blockers"]
    )


def test_dims_match_unitree_g1_sonic_state_contract():
    from blueprint_pipeline.oscar_isaac_closed_loop_eval import UNITREE_G1_SONIC_STATE_DIMS

    assert G1_SONIC_PROPRIOCEPTION_STATE_DIMS == UNITREE_G1_SONIC_STATE_DIMS


def test_groups_match_repo_canonical_joint_groups():
    from blueprint_pipeline.mujoco_g1_wam_vla_policy_endpoint_eval import (
        UNITREE_G1_SONIC_STATE_JOINT_GROUPS,
    )

    assert {
        group: tuple(names) for group, names in G1_CANONICAL_DOF_GROUPS.items()
    } == {
        group: tuple(names)
        for group, names in UNITREE_G1_SONIC_STATE_JOINT_GROUPS.items()
    }


def test_alias_table_targets_are_canonical():
    canonical = {
        name for names in G1_CANONICAL_DOF_GROUPS.values() for name in names
    }
    for target, aliases in G1_CANONICAL_DOF_ALIASES.items():
        assert target in canonical
        for alias in aliases:
            assert alias not in canonical


def test_validate_g1_sonic_state_dims_accepts_resolved_state_and_rejects_bad():
    resolution = resolve_g1_proprioception_map(_hand_extended_43(), require_hands=True)
    state = {**resolution["group_values"], "projected_gravity": [0.0, 0.0, -1.0]}
    assert validate_g1_sonic_state_dims(state) == []
    short = dict(state)
    short["left_leg"] = short["left_leg"][:5]
    assert validate_g1_sonic_state_dims(short) == ["g1_sonic_state_dim_invalid:left_leg"]
    missing = dict(state)
    del missing["projected_gravity"]
    assert validate_g1_sonic_state_dims(missing) == [
        "g1_sonic_state_dim_invalid:projected_gravity"
    ]
    with pytest.raises(ValueError):
        resolve_g1_proprioception_map(_hand_extended_43(), require_hands="yes")
