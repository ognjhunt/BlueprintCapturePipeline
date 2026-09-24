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
