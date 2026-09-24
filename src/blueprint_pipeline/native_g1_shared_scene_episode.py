"""Run one G1 candidate in the existing native task/site scene.

The caller supplies a sealed Arena environment, an attested policy server,
the pinned SONIC target bridge, and a lossless media recorder. This module
only orders their calls. It does not infer task success or attest model bytes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import (
    finalize_multicamera_visual_evidence,
    persist_multicamera_observation,
)


G1_BOX_CANDIDATES = frozenset({
    "humanoidarena_dp_g1_dex3_sonic",
    "humanoidarena_pi05_g1_dex3_sonic",
})


def run_g1_shared_scene_episode(
    *,
    environment: Any,
    policy_client: Any,
    sonic_bridge: Any,
    candidate_id: str,
    task_prompt: str,
    max_steps: int,
    output_dir: Path,
    read_task_sample: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain every policy input, action, scene step and task readback.

    A trace is development-only until a worker verifies checkpoint/runtime
    identity, scores the frozen task, and seals the usual episode receipt.
    """

    plan = getattr(environment, "plan", None)
    if (
        not isinstance(plan, Mapping)
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or candidate_id not in G1_BOX_CANDIDATES
        or not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or not 1 <= max_steps <= 3000
        or not isinstance(task_prompt, str)
        or not task_prompt.strip()
        or task_prompt != (plan.get("task_spec") or {}).get("prompt")
        or not isinstance(output_dir, Path)
        or not callable(read_task_sample)
    ):
        raise ValueError("g1_shared_scene_episode_configuration_invalid")
    try:
        seed = int(plan["scenario"]["seed"])
        plan_digest = str(plan["plan_digest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("g1_shared_scene_episode_plan_invalid") from exc
    if seed < 0 or len(plan_digest) != 71 or not plan_digest.startswith("sha256:"):
        raise ValueError("g1_shared_scene_episode_plan_invalid")

    environment.reset(seed=seed)
    policy_client.reset(seed=seed)
    initial_task_sample = dict(read_task_sample())
    if initial_task_sample.get("step_index") != 0:
        raise ValueError("g1_shared_scene_initial_task_sample_invalid")

    def retain_observation(
        images: Mapping[str, Any], *, kind: str
    ) -> dict[str, Any]:
        metadata = environment.read_observation_metadata(tuple(images))
        return persist_multicamera_observation(
            images,
            output_dir=output_dir,
            episode_id=candidate_id,
            observation_index=len(policy_observations) + len(review_observations),
            kind=kind,
            **metadata,
        )

    steps: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    policy_observations: list[dict[str, Any]] = []
    review_observations: list[dict[str, Any]] = []
    while len(steps) < max_steps:
        inputs = environment.read_policy_inputs()
        if not isinstance(inputs, Mapping):
            raise ValueError("g1_shared_scene_policy_inputs_invalid")
        query_index = len(queries)
        policy_observation = retain_observation(
            {"head": inputs["front_rgb"]}, kind="policy-input"
        )
        policy_observations.append(policy_observation)
        policy_frame = policy_observation["views"]["head"]
        chunk = policy_client.infer_chunk(
            front_rgb=inputs["front_rgb"],
            observation_state=inputs["observation_state"],
            task=task_prompt,
        )
        if not isinstance(chunk, list) or not chunk:
            raise ValueError("g1_shared_scene_policy_chunk_invalid")
        queries.append({
            "query_index": query_index,
            "step_index": len(steps),
            "policy_input_frame": policy_frame,
            "sensor_freshness": inputs["sensor_freshness"],
            "observation_state": inputs["observation_state"],
            "returned_action_count": len(chunk),
        })
        for action_index, action in enumerate(chunk):
            if len(steps) >= max_steps:
                break
            targets = sonic_bridge.targets_for_action(action)
            state = environment.step_controller_targets(targets)
            review = environment.read_review_inputs()
            review_observation = retain_observation(
                {"head": review["head_rgb"], "overview": review["overview_rgb"]},
                kind="review-sample",
            )
            review_observations.append(review_observation)
            step_index = len(steps) + 1
            row = {
                "step_index": step_index,
                "query_index": query_index,
                "action_index": action_index,
                "semantic_action": action,
                "controller_targets_rad": targets,
                "robot_state": state,
                "task_sample": dict(read_task_sample()),
                "review_frames": {
                    role: review_observation["views"][role]
                    for role in ("head", "overview")
                },
                "review_sensor_freshness": review["sensor_freshness"],
            }
            steps.append(row)

    terminal_inputs = environment.read_review_inputs()
    terminal = retain_observation(
        {
            "head": terminal_inputs["head_rgb"],
            "overview": terminal_inputs["overview_rgb"],
        },
        kind="terminal-observation",
    )
    visual, artifacts = finalize_multicamera_visual_evidence(
        output_dir=output_dir,
        episode_id=candidate_id,
        identity={
            "scene_plan_digest": plan_digest,
            "candidate_id": candidate_id,
            "seed": seed,
            "claim_ceiling": "simulator_only_unscored",
        },
        policy_input_observations=policy_observations,
        review_observations=review_observations,
        terminal_observation=terminal,
        required_camera_ids=("head", "overview"),
        review_only_camera_ids=("overview",),
    )
    trace = {
        "schema_version": "native_g1_shared_scene_episode_trace.v1",
        "status": "development_trace_recorded",
        "claim_ceiling": "simulator_only_unscored",
        "scene_plan_digest": plan_digest,
        "candidate_id": candidate_id,
        "task_prompt": task_prompt,
        "seed": seed,
        "initial_task_sample": initial_task_sample,
        "policy_query_count": len(queries),
        "scene_step_count": len(steps),
        "queries": queries,
        "steps": steps,
        "policy_input_observations": policy_observations,
        "review_observations": review_observations,
        "terminal_observation": terminal,
        "visual_evidence": visual,
        "media_artifacts": artifacts,
    }
    try:
        trace = json.loads(json.dumps(trace, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("g1_shared_scene_episode_trace_invalid") from exc
    trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
    return trace


def run_g1_built_scene_policy_episode(
    *,
    built: Any,
    policy_client: Any,
    sonic_bridge: Any,
    candidate_id: str,
    max_steps: int,
    output_dir: Path,
    preflight_inputs: Mapping[str, Any],
    to_tensor: Callable[[Any], Any],
    make_action_tensor: Callable[..., Any],
) -> dict[str, Any]:
    """Score a G1 box episode from the same native scene and task readback.

    This executes the shared scene but does not attest the running policy
    server, model loaded in its process, or physical outcome. The existing
    qualified policy worker must own those gates before promotion.
    """

    from .adp_task_scoring import score_task_episode_from_spec
    from .native_g1_joint_episode_environment import NativeG1JointEpisodeEnvironment
    from .native_g1_run_preflight import preflight_g1_shared_scene_run
    from .native_task_arena_readback import NativeRigidTaskArenaReadback

    plan = getattr(built, "plan", None)
    if (
        not isinstance(plan, Mapping)
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or plan.get("task_kind") != "rigid_pick_place"
        or candidate_id not in G1_BOX_CANDIDATES
    ):
        raise ValueError("g1_built_scene_policy_configuration_invalid")
    preflight = preflight_g1_shared_scene_run(**preflight_inputs)
    if (
        preflight.get("status") != "staged_inputs_verified"
        or preflight.get("robot_id") != "unitree_g1"
        or preflight.get("candidate_id") != candidate_id
        or preflight.get("scene_plan_digest") != plan.get("plan_digest")
        or preflight.get("policy_role") != "manipulation"
    ):
        raise ValueError("g1_built_scene_policy_preflight_binding_invalid")
    environment = NativeG1JointEpisodeEnvironment(
        built=built, to_tensor=to_tensor, make_action_tensor=make_action_tensor
    )
    readback = NativeRigidTaskArenaReadback(built)

    def read_task_sample() -> dict[str, Any]:
        return {
            **readback.read_task_sample(),
            "step_index": environment.read_state()["step_index"],
        }

    trace = run_g1_shared_scene_episode(
        environment=environment,
        policy_client=policy_client,
        sonic_bridge=sonic_bridge,
        candidate_id=candidate_id,
        task_prompt=str(plan["task_spec"]["prompt"]),
        max_steps=max_steps,
        output_dir=output_dir,
        read_task_sample=read_task_sample,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "native_g1_shared_scene_episode_trace.v1.json"
    with trace_path.open("x", encoding="utf-8") as stream:
        json.dump(trace, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    samples = [trace["initial_task_sample"]]
    samples.extend(row["task_sample"] for row in trace["steps"])
    score = score_task_episode_from_spec(task_spec=plan["task_spec"], samples=samples)
    if not isinstance(score, Mapping) or score.get("status") != "scored":
        raise ValueError("g1_built_scene_task_score_incomplete")
    result = {
        "schema_version": "native_g1_built_scene_policy_episode.v1",
        "status": "development_only_scored_episode",
        "scene_plan_digest": plan["plan_digest"],
        "candidate_id": candidate_id,
        "preflight_receipt_digest": canonical_digest(preflight),
        "trace_digest": trace["trace_digest"],
        "trace_relative_path": trace_path.name,
        "score": score,
        "policy_runtime_identity_verified": False,
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with (output_dir / "native_g1_built_scene_policy_episode.v1.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result
