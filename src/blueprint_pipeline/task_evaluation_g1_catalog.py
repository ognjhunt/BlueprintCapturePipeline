"""Unavailable G1 choices in the existing Task Evaluation Run setup catalog.

The development worker can exercise these checkpoints, but no production
policy-canary profile has executed them on this site's sealed scene. Publishing
their identities here does not admit them for a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .native_g1_run_preflight import _candidate
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


G1_PRESET_ID = "unitree_g1_dex3_sonic_v1"
G1_EMBODIMENT_ID = "unitree_g1_dex3_v1"
G1_CANDIDATE_IDS = (
    "humanoidarena_dp_g1_dex3_sonic",
    "humanoidarena_pi05_g1_dex3_sonic",
    "humanoidarena_dp_g1_dex3_sonic_vision_navi",
    "humanoidarena_pi05_g1_dex3_sonic_vision_navi",
)
INVENTORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
)


def unavailable_g1_preset(*, inventory_path: Path = INVENTORY_PATH) -> dict[str, Any]:
    """List exact pinned policies while keeping all execution gates closed."""

    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise ValueError("g1_catalog_inventory_missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("source_revision") != "68479287a784a69be9ce6ad739311d2f11f75ef9"
        or [row.get("candidate_id") for row in inventory.get("candidates", [])]
        != list(G1_CANDIDATE_IDS)
    ):
        raise ValueError("g1_catalog_inventory_identity_invalid")
    policies = []
    for candidate_id in G1_CANDIDATE_IDS:
        row = _candidate(inventory, candidate_id)
        navigation = row["policy_role"] == "movement_navigation"
        if navigation != candidate_id.endswith("_vision_navi"):
            raise ValueError("g1_catalog_candidate_role_invalid")
        policies.append({
            "candidate_id": candidate_id,
            "display_name": (
                "HumanoidArena DP G1 navigation" if navigation and "_dp_" in candidate_id
                else "HumanoidArena π0.5 G1 navigation" if navigation
                else "HumanoidArena DP G1 manipulation" if "_dp_" in candidate_id
                else "HumanoidArena π0.5 G1 manipulation"
            ),
            "evaluation_objective_id": "g1_navigation_goal" if navigation else "task_success",
            "checkpoint": {
                "uri": "https://modelscope.cn/models/Twang2026/HumanoidArena_models",
                "digest": row["inventory_digest"],
                "size_bytes": sum(file["size_bytes"] for file in row["files"]),
            },
            "adapter_id": "humanoidarena_semantic_v3_to_sonic_v1",
            "license_id": "publisher_apache_2_inherited_terms_pending",
            "compatibility": {
                "robot_preset_ids": [G1_PRESET_ID],
                "embodiment_ids": [G1_EMBODIMENT_ID],
                "observation_schema_ids": ["humanoidarena_head_rgb_state64_v1"],
                "action_schema_ids": ["humanoidarena_semantic_v3"],
                "simulator_runtime_ids": ["isaac_native_arena_g1_v1"],
                "task_family_ids": ["rigid_relocation"],
            },
            "readiness": {
                "status": "unavailable",
                "receipt": None,
                "reason": (
                    "A confirmed site movement goal, live checkpoint/SONIC episode, "
                    "and movement score are required before this policy can run."
                    if navigation else
                    "A sealed G1 scene, reviewed model/SONIC terms, and live checkpoint/SONIC "
                    "episode are required before this policy can run."
                ),
            },
        })
    return {
        "robot_preset_id": G1_PRESET_ID,
        "display_name": "Unitree G1 + Dex3 / SONIC",
        "embodiment_id": G1_EMBODIMENT_ID,
        "task_family_id": "rigid_relocation",
        "simulator_runtime_id": "isaac_native_arena_g1_v1",
        "runtime_image": {
            "uri": NATIVE_TASK_ARENA_IMAGE,
            "digest": "sha256:" + NATIVE_TASK_ARENA_IMAGE.rsplit("@sha256:", 1)[1],
        },
        "observation_schema": {
            "schema_id": "humanoidarena_head_rgb_state64_v1",
            "cameras": ["head"],
            "modalities": ["rgb_uint8", "state_64", "language_instruction"],
        },
        "action_schema": {
            "schema_id": "humanoidarena_semantic_v3",
            "space": "semantic_v3_to_sonic_targets",
            "control_hz": 50,
        },
        "readiness": {
            "status": "unavailable",
            "receipt": None,
            "reason": "G1 has no verified production episode for this site and task.",
        },
        "policy_candidates": policies,
    }
