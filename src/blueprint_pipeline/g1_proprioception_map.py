"""Explicit canonical Unitree G1 DOF mapping for initial proprioception.

Pure CPU logic with no Isaac imports so the mapping is hermetically testable.
Canonical joint names follow the official Unitree G1 29-DOF rev_1_0 naming
already pinned across this repository (``UNITREE_G1_SONIC_STATE_JOINT_GROUPS``
in ``mujoco_g1_wam_vla_policy_endpoint_eval`` and the GEAR-SONIC state dims in
``oscar_isaac_closed_loop_eval``). Substring token grouping is forbidden here:
every group member is an exact canonical joint resolved through a deliberate
alias table, and any missing, duplicate, ambiguous, or colliding required
joint blocks fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION = "g1_proprioception_map.v1"

G1_CANONICAL_DOF_GROUPS: dict[str, tuple[str, ...]] = {
    "left_leg": (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    "right_leg": (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
    "waist": (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ),
    "left_arm": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right_arm": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
    "left_hand": (
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
    ),
    "right_hand": (
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    ),
}

G1_HAND_GROUPS = ("left_hand", "right_hand")

# Earlier official Unitree G1 model revisions published the elbow/wrist-roll
# pair as ``*_elbow_pitch_joint`` / ``*_elbow_roll_joint``; rev_1_0 renamed
# them. Accept exactly those historical spellings and nothing else.
G1_CANONICAL_DOF_ALIASES: dict[str, tuple[str, ...]] = {
    "left_elbow_joint": ("left_elbow_pitch_joint",),
    "right_elbow_joint": ("right_elbow_pitch_joint",),
    "left_wrist_roll_joint": ("left_elbow_roll_joint",),
    "right_wrist_roll_joint": ("right_elbow_roll_joint",),
}

G1_SONIC_PROPRIOCEPTION_STATE_DIMS: dict[str, int] = {
    **{group: len(names) for group, names in G1_CANONICAL_DOF_GROUPS.items()},
    "projected_gravity": 3,
}


def _normalize_dof_name(name: Any) -> str:
    text = str(name or "").strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.strip().lower()


def _accepted_spellings() -> tuple[dict[str, str], list[str]]:
    accepted: dict[str, str] = {}
    blockers: list[str] = []
    for names in G1_CANONICAL_DOF_GROUPS.values():
        for canonical in names:
            for spelling in (canonical, *G1_CANONICAL_DOF_ALIASES.get(canonical, ())):
                existing = accepted.get(spelling)
                if existing is not None and existing != canonical:
                    blockers.append(f"g1_proprioception_alias_table_ambiguous:{spelling}")
                    continue
                accepted[spelling] = canonical
    return accepted, blockers


def resolve_g1_proprioception_map(
    observed_dofs: Sequence[Any], *, require_hands: bool = True
) -> dict[str, Any]:
    """Resolve a live articulation DOF inventory onto the canonical G1 map.

    ``observed_dofs`` is an ordered sequence of ``(name, position)`` pairs or
    mappings with ``name``/``position`` keys, in articulation index order.
    Returns a fail-closed result: ``group_values`` and ``mapping_digest`` are
    populated only when ``status`` is ``passed``.
    """

    if not isinstance(require_hands, bool):
        raise ValueError("g1_proprioception_require_hands_not_strict_boolean")
    blockers: list[str] = []
    inventory: list[dict[str, Any]] = []
    entries_by_name: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(observed_dofs):
        if isinstance(item, Mapping):
            raw_name, raw_position = item.get("name"), item.get("position")
        else:
            raw_name, raw_position = item
        observed_name = str(raw_name or "").strip()
        normalized = _normalize_dof_name(observed_name)
        if not normalized:
            blockers.append(f"g1_proprioception_observed_dof_name_empty:{index}")
            continue
        try:
            position = float(raw_position)
        except (TypeError, ValueError):
            position = math.nan
        if not math.isfinite(position):
            blockers.append(f"g1_proprioception_observed_position_invalid:{normalized}")
        entry = {
            "observed_index": index,
            "observed_name": observed_name,
            "normalized_name": normalized,
            "position": position,
        }
        inventory.append(entry)
        entries_by_name.setdefault(normalized, []).append(entry)
    for normalized, entries in sorted(entries_by_name.items()):
        if len(entries) > 1:
            blockers.append(f"g1_proprioception_observed_dof_duplicate:{normalized}")

    accepted, table_blockers = _accepted_spellings()
    blockers.extend(table_blockers)
    canonical_matches: dict[str, list[dict[str, Any]]] = {}
    unmapped: list[str] = []
    for normalized, entries in entries_by_name.items():
        canonical = accepted.get(normalized)
        if canonical is None:
            unmapped.append(normalized)
            continue
        canonical_matches.setdefault(canonical, []).extend(entries)

    resolved_map: dict[str, list[dict[str, Any]]] = {}
    group_values: dict[str, list[float]] = {}
    dimensions: dict[str, int] = {}
    for group, names in G1_CANONICAL_DOF_GROUPS.items():
        group_observed = any(canonical_matches.get(name) for name in names)
        if group in G1_HAND_GROUPS and not require_hands and not group_observed:
            resolved_map[group] = []
            group_values[group] = []
            dimensions[group] = 0
            continue
        rows: list[dict[str, Any]] = []
        values: list[float] = []
        for canonical in names:
            entries = canonical_matches.get(canonical) or []
            if not entries:
                blockers.append(f"g1_proprioception_required_dof_missing:{canonical}")
                continue
            if len(entries) > 1:
                blockers.append(f"g1_proprioception_alias_collision:{canonical}")
                continue
            entry = entries[0]
            rows.append(
                {
                    "canonical_name": canonical,
                    "observed_name": entry["observed_name"],
                    "observed_index": entry["observed_index"],
                }
            )
            values.append(entry["position"])
        resolved_map[group] = rows
        group_values[group] = values
        dimensions[group] = len(values)

    blockers = sorted(set(blockers))
    if blockers:
        return {
            "schema_version": G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": blockers,
            "group_values": {},
            "resolved_map": {},
            "dimensions": {},
            "observed_dof_inventory": inventory,
            "unmapped_observed_dofs": sorted(unmapped),
            "mapping_digest": None,
        }
    mapping_digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
                "resolved_map": resolved_map,
                "unmapped_observed_dofs": sorted(unmapped),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": G1_PROPRIOCEPTION_MAP_SCHEMA_VERSION,
        "status": "passed",
        "blockers": [],
        "group_values": group_values,
        "resolved_map": resolved_map,
        "dimensions": dimensions,
        "observed_dof_inventory": inventory,
        "unmapped_observed_dofs": sorted(unmapped),
        "mapping_digest": mapping_digest,
    }


def validate_g1_sonic_state_dims(state: Mapping[str, Any]) -> list[str]:
    """Check a proprioception state against the GEAR-SONIC dimension contract."""

    blockers: list[str] = []
    for key, dim in G1_SONIC_PROPRIOCEPTION_STATE_DIMS.items():
        value = state.get(key) if isinstance(state, Mapping) else None
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) != dim
            or not all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in value
            )
        ):
            blockers.append(f"g1_sonic_state_dim_invalid:{key}")
    return sorted(blockers)
