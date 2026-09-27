"""Score one registered team policy in the retained G1 task scene.

This is the worker-side episode seam. The trusted caller must admit the
runtime, model and scene rights, site-observation exchange, and paid resources
before invoking it. A score here is development-only simulator evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_navigation_goal import score_g1_navigation_episode, validate_g1_navigation_goal
from .native_g1_shared_scene_episode import (
    run_g1_shared_scene_episode,
    team_policy_candidate_id,
)
from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID, G1_PRESET_ID
from .task_evaluation_packet_planning_setup import validate_packet_planning_setup
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


RESULT_SCHEMA = "native_g1_team_scored_scene_episode.v1"
TRACE_FILENAME = "native_g1_shared_scene_episode_trace.v1.json"
RESULT_FILENAME = RESULT_SCHEMA + ".json"


def run_g1_team_scored_scene_episode(
    *,
    built: Any,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    policy_client: Any,
    sonic_bridge: Any,
    objective_id: str,
    max_steps: int,
    output_dir: Path,
    to_tensor: Callable[[Any], Any],
    make_action_tensor: Callable[..., Any],
) -> dict[str, Any]:
    """Run the same G1 scene, recorder and independent scorer for a team client."""

    setup = validate_packet_planning_setup(trusted_setup)
    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=setup, authenticated_owner=authenticated_owner
    )
    plan = getattr(built, "plan", None)
    task = plan.get("task_spec") if isinstance(plan, Mapping) else None
    if (
        bound["robot_preset_id"] != G1_PRESET_ID
        or bound["embodiment_id"] != G1_EMBODIMENT_ID
        or bound["observation_schema_id"] != "humanoidarena_head_rgb_state64_v1"
        or bound["action_schema_id"] != "humanoidarena_semantic_v3"
        or objective_id not in {"task_success", "g1_navigation_goal"}
        or not isinstance(plan, Mapping)
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
        or plan.get("scene_id") != bound["source_scene_id"]
        or plan.get("task_id") != bound["source_task_id"]
        or plan.get("task_kind") != "rigid_pick_place"
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or not isinstance(task, Mapping)
        or not isinstance(setup.get("task_success_contract_digest"), str)
        or task.get("task_success_contract_digest") != setup["task_success_contract_digest"]
        or not isinstance(output_dir, Path)
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise ValueError("g1_team_scored_scene_binding_invalid")
    candidate_id = team_policy_candidate_id(bound["profile_digest"])
    if getattr(policy_client, "profile_digest", None) != bound["profile_digest"]:
        raise ValueError("g1_team_scored_scene_client_identity_invalid")

    from .native_g1_joint_episode_environment import NativeG1JointEpisodeEnvironment

    environment = NativeG1JointEpisodeEnvironment(
        built=built, to_tensor=to_tensor, make_action_tensor=make_action_tensor
    )
    if objective_id == "g1_navigation_goal":
        goal = validate_g1_navigation_goal(task)

        def read_task_sample() -> dict[str, Any]:
            state = environment.read_state()
            return {
                "step_index": state["step_index"],
                "root_position_world_m": state["root_position_world_m"],
            }

        task_prompt = goal["task_instruction"]
    else:
        from .native_task_arena_readback import NativeRigidTaskArenaReadback

        readback = NativeRigidTaskArenaReadback(built)

        def read_task_sample() -> dict[str, Any]:
            return {
                **readback.read_task_sample(),
                "step_index": environment.read_state()["step_index"],
            }

        task_prompt = str(task["prompt"])

    trace = run_g1_shared_scene_episode(
        environment=environment,
        policy_client=policy_client,
        sonic_bridge=sonic_bridge,
        candidate_id=candidate_id,
        task_prompt=task_prompt,
        max_steps=max_steps,
        output_dir=output_dir,
        read_task_sample=read_task_sample,
        team_policy_profile_digest=bound["profile_digest"],
        team_objective_id=objective_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / TRACE_FILENAME
    with trace_path.open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    samples = [trace["initial_task_sample"]]
    samples.extend(row["task_sample"] for row in trace["steps"])
    if objective_id == "g1_navigation_goal":
        score = score_g1_navigation_episode(task_spec=task, samples=samples)
    else:
        from .adp_task_scoring import score_task_episode_from_spec

        score = score_task_episode_from_spec(task_spec=task, samples=samples)
    if not isinstance(score, Mapping) or score.get("status") != "scored":
        raise ValueError("g1_team_scored_scene_task_score_incomplete")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "development_only_scored_episode",
        "claim_ceiling": "development_only",
        "profile_digest": bound["profile_digest"],
        "owner": bound["owner"],
        "delivery_mode": bound["delivery"]["mode"],
        "source_setup_digest": bound["source_setup_digest"],
        "source_packet_receipt_digest": setup["source_packet_receipt_digest"],
        "scene_id": plan["scene_id"],
        "task_id": plan["task_id"],
        "task_success_contract_digest": setup["task_success_contract_digest"],
        "scene_plan_digest": plan["plan_digest"],
        "candidate_id": candidate_id,
        "objective_id": objective_id,
        "policy_query_count": trace["policy_query_count"],
        "scene_step_count": trace["scene_step_count"],
        "trace_digest": trace["trace_digest"],
        "trace_relative_path": trace_path.name,
        "score": dict(score),
        "policy_runtime_identity_verified": False,
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
        "public_redistribution_authorized": False,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with (output_dir / RESULT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result
