"""Validate a planning-only pair chosen in the shared Task Evaluation Run setup.

The choice contains no scene packet, rights approval, spend authority, or run
request. Embodiment-specific staging joins it to those inputs later.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_policy_canary_setup import validate_policy_canary_setup


SCHEMA = "task_evaluation_policy_pair_choice.v1"
FIELDS = frozenset({
    "schema_version", "claim_ceiling", "source_launch_id", "offering_digest",
    "scene_revision_digest", "setup_digest", "robot_preset_id",
    "policy_candidate_ids", "objective_id", "choice_digest",
})


def validate_policy_pair_choice(
    value: Mapping[str, Any], *, setup: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind two compatible catalog candidates to one exact published setup."""

    published = validate_policy_canary_setup(setup)
    choice = dict(value)
    if (
        set(choice) != FIELDS
        or choice.get("schema_version") != SCHEMA
        or choice.get("claim_ceiling") != "planning_only"
        or choice.get("source_launch_id") != published["source_launch_id"]
        or choice.get("offering_digest") != published["offering_digest"]
        or choice.get("scene_revision_digest") != published["scene_revision_digest"]
        or choice.get("setup_digest") != published["setup_digest"]
        or choice.get("choice_digest") != canonical_digest(choice, digest_field="choice_digest")
    ):
        raise ValueError("policy_pair_choice_binding_invalid")
    robots = [
        row for row in published["robot_presets"]
        if row["robot_preset_id"] == choice.get("robot_preset_id")
    ]
    if len(robots) != 1:
        raise ValueError("policy_pair_choice_robot_invalid")
    robot = robots[0]
    chosen = choice.get("policy_candidate_ids")
    if (
        not isinstance(chosen, list)
        or len(chosen) != 2
        or any(not isinstance(item, str) or not item for item in chosen)
        or len(set(chosen)) != 2
        or choice.get("objective_id") not in {"task_success", "g1_navigation_goal"}
    ):
        raise ValueError("policy_pair_choice_candidates_invalid")
    by_id = {row["candidate_id"]: row for row in robot["policy_candidates"]}
    if len(by_id) != len(robot["policy_candidates"]) or any(item not in by_id for item in chosen):
        raise ValueError("policy_pair_choice_candidates_invalid")
    expected = {
        "robot_preset_ids": robot["robot_preset_id"],
        "embodiment_ids": robot["embodiment_id"],
        "observation_schema_ids": robot["observation_schema"]["schema_id"],
        "action_schema_ids": robot["action_schema"]["schema_id"],
        "simulator_runtime_ids": robot["simulator_runtime_id"],
        "task_family_ids": robot["task_family_id"],
    }
    if any(
        by_id[item].get("evaluation_objective_id", "task_success") != choice["objective_id"]
        or any(value not in by_id[item]["compatibility"][field]
               for field, value in expected.items())
        for item in chosen
    ):
        raise ValueError("policy_pair_choice_compatibility_invalid")
    return choice
