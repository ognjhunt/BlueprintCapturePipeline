"""Bounded multimodal agent loop for task-aware robot base placement.

The model proposes and visually reviews poses; deterministic site geometry and
robot-reach gates remain the only placement acceptance authority.  The accepted
pose is digest-bound before a native/GPU construction lane may consume it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    AgentsSDKInvoker,
    OpenAIAgentsSDKConfig,
)
from .scene_placement.robot_profile import get_robot_profile
from .task_evaluation_robot_placement_orientation import (
    RobotPlacementOrientationError,
    evaluate_orientation_slew_feasibility,
    solve_base_yaw_for_orientation,
)
from .task_evaluation_robot_placement_trajectory import (
    RobotPlacementTrajectoryError,
    validate_robot_placement_trajectory,
)


ROBOT_PLACEMENT_AGENT_MODEL = "gpt-5.6-sol"
ROBOT_PLACEMENT_AGENT_REASONING_EFFORT = "high"
ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS = 8_000
ROBOT_PLACEMENT_AGENT_SCHEMA_VERSION = "task_evaluation_robot_placement_agent.v1"
ROBOT_PLACEMENT_RECEIPT_SCHEMA_VERSION = "task_evaluation_robot_placement_receipt.v1"
ROBOT_PLACEMENT_CLAIM_CEILING = "analytic_and_visual_robot_placement_candidate"
DEFAULT_MAX_PLACEMENT_ROUNDS = 4
NATIVE_REJECTED_POSE_EXCLUSION_RADIUS_M = 0.08
NATIVE_REJECTED_POSE_ORIENTATION_EXCLUSION_RAD = math.radians(5.0)


class RobotPlacementAgentError(ValueError):
    """The bounded placement loop or its receipt is invalid."""


class RobotBasePoseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_world_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4, max_length=4)

    @field_validator("position_world_m", "orientation_xyzw")
    @classmethod
    def _finite(cls, values: list[float]) -> list[float]:
        result = [float(value) for value in values]
        if not all(math.isfinite(value) for value in result):
            raise ValueError("robot_placement_pose_non_finite")
        return result

    @field_validator("orientation_xyzw")
    @classmethod
    def _unit_quaternion(cls, values: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
            raise ValueError("robot_placement_orientation_not_unit")
        return values


class RobotPlacementProposalOutput(BaseModel):
    """Non-authoritative pose proposal returned by the placement agent."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=160)
    pose: RobotBasePoseOutput
    support_surface_id: str = Field(min_length=1, max_length=300)
    rationale: str = Field(min_length=1, max_length=4_000)
    addressed_blockers: list[str] = Field(default_factory=list, max_length=50)
    uncertainty: str = Field(min_length=1, max_length=2_000)


class RobotPlacementVisualReviewOutput(BaseModel):
    """Advisory visual verdict; it can veto but never approve geometry."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "rejected", "uncertain"]
    robot_supported_by_declared_surface: bool
    robot_not_visibly_clipping_site_geometry: bool
    robot_faces_task_workspace: bool
    task_workspace_visually_reachable: bool
    camera_views_are_sufficient: bool
    reason: str = Field(min_length=1, max_length=4_000)
    revision_guidance: list[str] = Field(default_factory=list, max_length=30)


PlacementValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PlacementRenderer = Callable[
    [Mapping[str, Any], int], Sequence[Mapping[str, Any]]
]
PlacementExecutor = Callable[
    [Mapping[str, Any], Mapping[str, Any], int], Mapping[str, Any]
]


def robot_placement_agents_sdk_config(
    *,
    max_inference_cost_usd: float,
    allow_live_invocation: bool,
    tracing_disabled: bool = False,
) -> OpenAIAgentsSDKConfig:
    """Explicit production configuration for the user-selected placement model."""

    return OpenAIAgentsSDKConfig(
        model=ROBOT_PLACEMENT_AGENT_MODEL,
        max_turns=1,
        max_output_tokens=ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS,
        allow_live_invocation=allow_live_invocation,
        tracing_disabled=tracing_disabled,
        max_inference_cost_usd=max_inference_cost_usd,
        input_cost_per_million_tokens_usd=4.0,
        output_cost_per_million_tokens_usd=20.0,
    )


def _image_metadata(images: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, image in enumerate(images):
        digest = str(image.get("digest") or "")
        label = str(image.get("label") or f"image_{index}")
        if not (
            digest.startswith("sha256:")
            and len(digest) == 71
            and all(character in "0123456789abcdef" for character in digest[7:])
        ):
            raise RobotPlacementAgentError("robot_placement_image_digest_invalid")
        result.append({"label": label, "digest": digest})
    return result


def _multimodal_input(
    *, prompt: Mapping[str, Any], images: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    metadata = _image_metadata(images)
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": json.dumps(
                {**dict(prompt), "image_inventory": metadata},
                sort_keys=True,
            ),
        }
    ]
    for image in images:
        image_url = str(image.get("image_url") or "")
        if not image_url.startswith(("data:image/", "https://")):
            raise RobotPlacementAgentError("robot_placement_image_url_invalid")
        content.append(
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": str(image.get("detail") or "high"),
            }
        )
    return [{"role": "user", "content": content}]


def _validated_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    gate = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        gate.get("schema_version") != "task_evaluation_robot_placement_geometry_gate.v1"
        or gate.get("status") not in {"passed", "rejected"}
        or not isinstance(gate.get("blockers"), list)
        or gate.get("geometry_gate_digest")
        != canonical_digest(gate, digest_field="geometry_gate_digest")
    ):
        raise RobotPlacementAgentError("robot_placement_geometry_gate_invalid")
    return gate


def _position_world_m(value: object) -> list[float] | None:
    if not isinstance(value, Mapping):
        return None
    position = value.get("position_world_m")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 3
    ):
        return None
    try:
        result = [float(item) for item in position]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _orientation_world_xyzw(value: object) -> list[float] | None:
    if not isinstance(value, Mapping):
        return None
    orientation = value.get("orientation_xyzw")
    if orientation is None:
        return None
    if (
        not isinstance(orientation, Sequence)
        or isinstance(orientation, (str, bytes))
        or len(orientation) != 4
    ):
        raise RobotPlacementAgentError("robot_placement_rejected_native_poses_invalid")
    try:
        result = [float(item) for item in orientation]
    except (TypeError, ValueError) as exc:
        raise RobotPlacementAgentError(
            "robot_placement_rejected_native_poses_invalid"
        ) from exc
    norm = math.sqrt(sum(item * item for item in result))
    if not all(math.isfinite(item) for item in result) or not math.isclose(
        norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4
    ):
        raise RobotPlacementAgentError("robot_placement_rejected_native_poses_invalid")
    return result


def _quaternion_distance_rad(left: Sequence[float], right: Sequence[float]) -> float:
    # q and -q encode the same rotation, so use the absolute dot product.
    dot = abs(math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _native_rejected_poses(
    *,
    scene_context: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    prior_native_attempts: Sequence[Mapping[str, Any]] = (),
) -> list[tuple[list[float], list[float] | None]]:
    poses: list[tuple[list[float], list[float] | None]] = []
    configured = scene_context.get("rejected_native_base_poses") or []
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise RobotPlacementAgentError("robot_placement_rejected_native_poses_invalid")
    for value in configured:
        position = _position_world_m(value)
        if position is None:
            raise RobotPlacementAgentError("robot_placement_rejected_native_poses_invalid")
        poses.append((position, _orientation_world_xyzw(value)))
    for attempt in prior_native_attempts:
        feedback = attempt.get("native_feedback")
        pose = (
            feedback.get("initial_robot_root_pose_world")
            if isinstance(feedback, Mapping)
            else None
        )
        if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)):
            raise RobotPlacementAgentError(
                "robot_placement_prior_native_attempt_pose_invalid"
            )
        pose_mapping = {
            "position_world_m": list(pose[:3]),
            "orientation_xyzw": list(pose[3:]),
        }
        position = _position_world_m(pose_mapping)
        orientation = _orientation_world_xyzw(pose_mapping)
        if position is None or orientation is None:
            raise RobotPlacementAgentError(
                "robot_placement_prior_native_attempt_pose_invalid"
            )
        poses.append((position, orientation))
    for round_record in history:
        native_attempt = round_record.get("native_attempt")
        if not isinstance(native_attempt, Mapping) or native_attempt.get("status") != "rejected":
            continue
        proposal = round_record.get("proposal")
        pose = proposal.get("pose") if isinstance(proposal, Mapping) else None
        position = _position_world_m(pose)
        if position is not None:
            poses.append((position, _orientation_world_xyzw(pose)))
    return poses


def _reject_reused_native_pose(
    *,
    gate: Mapping[str, Any],
    proposal: Mapping[str, Any],
    rejected_poses: Sequence[tuple[Sequence[float], Sequence[float] | None]],
) -> dict[str, Any]:
    result = dict(gate)
    pose = proposal.get("pose")
    position = _position_world_m(pose)
    orientation = _orientation_world_xyzw(pose)
    if position is None or orientation is None:
        raise RobotPlacementAgentError("robot_placement_pose_invalid")
    reused = any(
        math.dist(position, [float(item) for item in rejected_position])
        < NATIVE_REJECTED_POSE_EXCLUSION_RADIUS_M
        and (
            rejected_orientation is None
            or _quaternion_distance_rad(orientation, rejected_orientation)
            < NATIVE_REJECTED_POSE_ORIENTATION_EXCLUSION_RAD
        )
        for rejected_position, rejected_orientation in rejected_poses
    )
    if not reused:
        return result
    result["status"] = "rejected"
    result["blockers"] = sorted(
        set([*result.get("blockers", []), "prior_native_pose_reused"])
    )
    result["geometry_gate_digest"] = ""
    result["geometry_gate_digest"] = canonical_digest(
        result, digest_field="geometry_gate_digest"
    )
    return result


def _orientation_slew_guidance(
    *, trajectory: Mapping[str, Any], robot_id: str
) -> dict[str, Any] | None:
    """Advisory base-yaw guidance for the authored plan, or None when undecidable.

    Base yaw is the one placement degree of freedom that changes how far the
    wrist must rotate to reach an authored tool pose, so a proposer that ignores
    it is optimizing blind.  Purely advisory: the deterministic gate still
    decides, and this never widens what is acceptable.
    """

    steps = trajectory.get("maximum_steps_per_phase")
    phases = trajectory.get("phases")
    if not robot_id or not phases or steps is None:
        return None
    try:
        profile = get_robot_profile(robot_id)
    except (KeyError, ValueError):
        return None
    try:
        solved = solve_base_yaw_for_orientation(
            rest_grasp_orientation_base_xyzw=(
                profile.rest_grasp_orientation_base_xyzw
            ),
            phases=phases,
            maximum_steps_per_phase=steps,
            orientation_slew_rad_per_step=profile.orientation_slew_rad_per_step,
        )
    except RobotPlacementOrientationError:
        return None
    return {
        "recommended_base_yaw_rad": solved["best_yaw_rad"],
        "recommended_worst_phase_slew_rad": solved["best_worst_slew_rad"],
        "recommended_worst_phase_required_steps": (
            solved["best_worst_required_steps"]
        ),
        "step_budget": solved["step_budget"],
        "any_yaw_is_feasible": solved["feasible"],
        "admissible_yaw_fraction": (
            solved["feasible_yaw_count"] / solved["yaw_sample_count"]
        ),
        "advisory_only_deterministic_gate_decides": True,
    }


def _reject_infeasible_orientation_slew(
    *,
    gate: Mapping[str, Any],
    proposal: Mapping[str, Any],
    trajectory: Mapping[str, Any] | None,
    robot_id: str,
    maximum_steps_per_phase: int | None,
) -> dict[str, Any]:
    """Fail a candidate whose base orientation cannot serve the authored plan.

    Reach and support say the gripper can arrive; they do not say it can arrive
    ORIENTED.  The required wrist slew is fixed by the base orientation and the
    robot's rest grasp frame, so it is decidable here -- before the candidate is
    compiled, bundled, and executed on a rented GPU.  Eleven paid allocations on
    scene 839873 were spent rediscovering one such refusal.
    """

    result = dict(gate)
    if trajectory is None or maximum_steps_per_phase is None or not robot_id:
        return result
    phases = trajectory.get("phases")
    if not phases:
        return result
    try:
        profile = get_robot_profile(robot_id)
    except (KeyError, ValueError):
        # An unregistered embodiment has no rest grasp frame to reason from.
        # Native execution still gates it; do not invent an analytic verdict.
        return result
    orientation = _orientation_world_xyzw(proposal.get("pose"))
    if orientation is None:
        raise RobotPlacementAgentError("robot_placement_pose_invalid")
    try:
        report = evaluate_orientation_slew_feasibility(
            base_orientation_xyzw=orientation,
            rest_grasp_orientation_base_xyzw=(
                profile.rest_grasp_orientation_base_xyzw
            ),
            phases=phases,
            maximum_steps_per_phase=maximum_steps_per_phase,
            orientation_slew_rad_per_step=profile.orientation_slew_rad_per_step,
        )
    except RobotPlacementOrientationError as exc:
        raise RobotPlacementAgentError(str(exc)) from exc
    result["orientation_slew_feasibility"] = report
    if report["feasible"]:
        return result
    result["status"] = "rejected"
    result["blockers"] = sorted(
        set([*result.get("blockers", []), *report["blockers"]])
    )
    result["geometry_gate_digest"] = ""
    result["geometry_gate_digest"] = canonical_digest(
        result, digest_field="geometry_gate_digest"
    )
    return result


def _visual_passed(review: RobotPlacementVisualReviewOutput) -> bool:
    return bool(
        review.status == "passed"
        and review.robot_supported_by_declared_surface
        and review.robot_not_visibly_clipping_site_geometry
        and review.robot_faces_task_workspace
        and review.task_workspace_visually_reachable
        and review.camera_views_are_sufficient
    )


def _exact_inventory_member(
    proposal: Mapping[str, Any], *, scene_context: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Require a model selection to reproduce one immutable CPU-feasible member."""

    raw_inventory = scene_context.get(
        "deterministic_geometry_passing_candidate_inventory"
    )
    if raw_inventory is None:
        return None
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise RobotPlacementAgentError("robot_placement_candidate_inventory_invalid")
    expected_digest = scene_context.get(
        "deterministic_geometry_passing_candidate_inventory_digest"
    )
    trajectory_digest = scene_context.get(
        "deterministic_geometry_passing_candidate_inventory_trajectory_digest"
    )
    if expected_digest != canonical_digest(
        {"trajectory_digest": trajectory_digest, "candidates": raw_inventory}
    ):
        raise RobotPlacementAgentError("robot_placement_candidate_inventory_digest_mismatch")
    members: dict[str, Mapping[str, Any]] = {}
    for raw in raw_inventory:
        if not isinstance(raw, Mapping):
            raise RobotPlacementAgentError("robot_placement_candidate_inventory_invalid")
        candidate_id = str(raw.get("candidate_id") or "")
        if not candidate_id or candidate_id in members:
            raise RobotPlacementAgentError("robot_placement_candidate_inventory_invalid")
        members[candidate_id] = raw
    candidate_id = str(proposal.get("candidate_id") or "")
    member = members.get(candidate_id)
    if member is None:
        raise RobotPlacementAgentError("robot_placement_candidate_not_in_inventory")
    selected = {
        "candidate_id": candidate_id,
        "support_surface_id": proposal.get("support_surface_id"),
        "pose": proposal.get("pose"),
    }
    expected = {
        "candidate_id": candidate_id,
        "support_surface_id": member.get("support_surface_id"),
        "pose": member.get("pose"),
    }
    if canonical_digest(selected) != canonical_digest(expected):
        raise RobotPlacementAgentError(
            "robot_placement_candidate_inventory_member_mutated"
        )
    return member


def _validated_native_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    attempt = json.loads(json.dumps(dict(value), allow_nan=False))
    feedback_images = attempt.pop("feedback_images", [])
    if (
        attempt.get("schema_version")
        != "task_evaluation_robot_placement_native_attempt.v1"
        or attempt.get("status") not in {"passed", "rejected"}
        or not isinstance(attempt.get("blockers"), list)
        or attempt.get("native_attempt_digest")
        != canonical_digest(attempt, digest_field="native_attempt_digest")
    ):
        raise RobotPlacementAgentError("robot_placement_native_attempt_invalid")
    images = list(feedback_images)
    attempt["feedback_images"] = _image_metadata(images)
    attempt["_feedback_image_inputs"] = images
    return attempt


def _build_placement_receipt(
    *,
    run_id: str,
    scene_digest: str,
    task_digest: str,
    scene_context_digest: str,
    task_context_digest: str,
    overview_images: Sequence[Mapping[str, Any]],
    prior_native_attempts: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    accepted: Mapping[str, Any] | None,
    max_rounds: int,
    native_loop_enabled: bool,
    task_trajectory_digest: str | None,
) -> dict[str, Any]:
    accepted_native = dict((accepted or {}).get("native_attempt") or {})
    receipt: dict[str, Any] = {
        "schema_version": ROBOT_PLACEMENT_RECEIPT_SCHEMA_VERSION,
        "status": "accepted" if accepted is not None else "blocked",
        "run_id": run_id,
        "model": ROBOT_PLACEMENT_AGENT_MODEL,
        "reasoning_effort": ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
        "max_rounds": int(max_rounds),
        "round_count": len(history),
        "scene_binding_digest": scene_digest,
        "task_binding_digest": task_digest,
        "scene_context_digest": scene_context_digest,
        "task_context_digest": task_context_digest,
        "task_trajectory_digest": task_trajectory_digest,
        "overview_images": _image_metadata(overview_images),
        "prior_native_attempts": list(prior_native_attempts),
        "prior_native_attempt_count": len(prior_native_attempts),
        "rounds": list(history),
        "accepted_pose": accepted["proposal"]["pose"] if accepted is not None else None,
        "accepted_candidate_id": (
            accepted["proposal"]["candidate_id"] if accepted is not None else None
        ),
        "accepted_support_surface_id": (
            accepted["proposal"]["support_surface_id"] if accepted is not None else None
        ),
        "accepted_geometry_gate_digest": (
            accepted["geometry_gate"]["geometry_gate_digest"]
            if accepted is not None
            else None
        ),
        "native_agent_loop_enabled": native_loop_enabled,
        "native_attempt_count": sum(
            1 for round_record in history if round_record.get("native_attempt")
        ),
        "accepted_native_attempt_digest": (
            accepted_native.get("native_attempt_digest") if accepted_native else None
        ),
        "candidate_may_self_authorize": False,
        "physical_execution_authorized": False,
        "native_construction_required": not bool(accepted_native),
        "model_grades_controls": False,
        "claim_ceiling": ROBOT_PLACEMENT_CLAIM_CEILING,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def run_task_evaluation_robot_placement_agent(
    *,
    invoker: AgentsSDKInvoker,
    run_id: str,
    scene_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any],
    scene_context: Mapping[str, Any] | None = None,
    task_context: Mapping[str, Any] | None = None,
    overview_images: Sequence[Mapping[str, Any]],
    validate_candidate: PlacementValidator,
    render_candidate: PlacementRenderer,
    execute_candidate: PlacementExecutor | None = None,
    task_trajectory: Mapping[str, Any] | None = None,
    prior_native_attempts: Sequence[Mapping[str, Any]] = (),
    max_rounds: int = DEFAULT_MAX_PLACEMENT_ROUNDS,
    max_input_tokens: int = 300_000,
) -> dict[str, Any]:
    """Create poses and, when supplied, iterate on native execution feedback."""

    if not 1 <= int(max_rounds) <= 8:
        raise RobotPlacementAgentError("robot_placement_round_cap_invalid")
    if not overview_images:
        raise RobotPlacementAgentError("robot_placement_overview_images_missing")
    trajectory: dict[str, Any] | None = None
    if task_trajectory is not None:
        try:
            trajectory = validate_robot_placement_trajectory(task_trajectory)
        except RobotPlacementTrajectoryError as exc:
            raise RobotPlacementAgentError(str(exc)) from exc
    if execute_candidate is not None and trajectory is None:
        raise RobotPlacementAgentError("robot_placement_native_trajectory_missing")
    scene = json.loads(json.dumps(dict(scene_binding), allow_nan=False))
    task = json.loads(json.dumps(dict(task_binding), allow_nan=False))
    scene_advisory_context = json.loads(
        json.dumps(dict(scene_context or {}), allow_nan=False)
    )
    task_advisory_context = json.loads(
        json.dumps(dict(task_context or {}), allow_nan=False)
    )
    if trajectory is not None:
        task_advisory_context["native_trajectory"] = trajectory
        guidance = _orientation_slew_guidance(
            trajectory=trajectory, robot_id=str(task.get("robot_id") or "")
        )
        if guidance is not None:
            # Advisory only: the deterministic gate remains the sole authority.
            # Without this the model proposes a base orientation blind to how
            # far the wrist would have to slew, and can only learn otherwise by
            # spending a GPU allocation on the refusal.
            task_advisory_context["orientation_slew_guidance"] = guidance
    scene_digest = canonical_digest(scene)
    task_digest = canonical_digest(task)
    scene_context_digest = canonical_digest(scene_advisory_context)
    task_context_digest = canonical_digest(task_advisory_context)
    history: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    prior_attempt_records: list[dict[str, Any]] = []
    native_feedback_images: list[Mapping[str, Any]] = []
    for raw_attempt in prior_native_attempts:
        attempt = _validated_native_attempt(raw_attempt)
        if attempt.get("status") != "rejected":
            raise RobotPlacementAgentError(
                "robot_placement_prior_native_attempt_not_rejected"
            )
        native_feedback_images.extend(attempt.pop("_feedback_image_inputs"))
        prior_attempt_records.append(attempt)

    proposal_instructions = (
        "You place a fixed-base robot for one observed site and task. Use the supplied exact "
        "geometry summary, robot dimensions, overview images, immutable task trajectory, and "
        "prior native gate failures. Choose the base position and yaw so every authored "
        "precontact, contact, motion, release, retreat, and recovery tool pose is reachable at "
        "its authored orientation, not merely one task center point. "
        "Select one exact candidate_id from the deterministic feasible inventory and copy its "
        "position, orientation, and support surface byte-for-byte. Do not interpolate, invent, "
        "or mutate a pose. The robot base must sit on the declared "
        "support surface, never inside a table/counter/floor; its body must not visibly clip site "
        "geometry; and the task workspace must be reachable. Do not claim success or modify any "
        "threshold. Return only the declared structured proposal."
    )
    review_instructions = (
        "You are the visual sanity reviewer for one proposed fixed-base robot placement. Review "
        "all supplied candidate previews. Fail closed if the base appears embedded, floating, "
        "unsupported, occluded so placement cannot be judged, pointed away from the task, or the "
        "task workspace looks implausibly unreachable. Your verdict is advisory and cannot "
        "override deterministic geometry gates. Return only the declared structured verdict."
    )

    for round_index in range(int(max_rounds)):
        prompt = {
            "schema_version": ROBOT_PLACEMENT_AGENT_SCHEMA_VERSION,
            "run_id": run_id,
            "round_index": round_index,
            "scene_binding": scene,
            "task_binding": task,
            "scene_context": scene_advisory_context,
            "task_context": task_advisory_context,
            "task_trajectory": trajectory,
            "prior_native_attempts": prior_attempt_records,
            "prior_rounds": history,
            "authority_boundary": {
                "model_proposes_only": True,
                "deterministic_geometry_gate_is_authoritative": True,
                "native_construction_still_required": True,
                "model_selects_exact_inventory_member": True,
                "model_may_create_or_mutate_pose": False,
                "native_failures_must_inform_the_next_pose": True,
                "native_failure_metrics_and_images_are_authoritative_feedback": True,
                "every_trajectory_phase_requires_native_ik_and_collision_readback": True,
                "model_may_not_modify_the_trajectory": True,
                "model_may_not_change_thresholds": True,
            },
        }
        proposal_result = invoker.invoke(
            AgentsSDKAgentSpec(
                run_id=run_id,
                capability="task_aware_robot_placement_proposal",
                name="Blueprint Task-aware Robot Placement Agent",
                instructions=proposal_instructions,
                model=ROBOT_PLACEMENT_AGENT_MODEL,
                max_turns=1,
                max_output_tokens=ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS,
                max_input_tokens=max_input_tokens,
                reasoning_effort=ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
                output_type=RobotPlacementProposalOutput,
            ),
            _multimodal_input(
                prompt=prompt,
                images=[*overview_images, *native_feedback_images],
            ),
        )
        proposal = RobotPlacementProposalOutput.model_validate(
            proposal_result.output
        ).model_dump(mode="json")
        _exact_inventory_member(proposal, scene_context=scene_advisory_context)
        geometry_gate = _validated_gate(validate_candidate(proposal))
        geometry_gate = _validated_gate(
            _reject_reused_native_pose(
                gate=geometry_gate,
                proposal=proposal,
                rejected_poses=_native_rejected_poses(
                    scene_context=scene_advisory_context,
                    history=history,
                    prior_native_attempts=prior_attempt_records,
                ),
            )
        )
        geometry_gate = _validated_gate(
            _reject_infeasible_orientation_slew(
                gate=geometry_gate,
                proposal=proposal,
                trajectory=trajectory,
                robot_id=str(task.get("robot_id") or ""),
                maximum_steps_per_phase=(
                    trajectory.get("maximum_steps_per_phase")
                    if trajectory is not None
                    else None
                ),
            )
        )
        round_record: dict[str, Any] = {
            "round_index": round_index,
            "proposal": proposal,
            "proposal_provider": proposal_result.provider,
            "proposal_model": proposal_result.model,
            "proposal_sdk_version": proposal_result.sdk_version,
            "proposal_usage": dict(proposal_result.usage),
            "proposal_trace_id": proposal_result.trace_id,
            "geometry_gate": geometry_gate,
            "visual_review": None,
            "preview_images": [],
            "native_attempt": None,
        }
        if geometry_gate["status"] != "passed":
            history.append(round_record)
            continue

        preview_images = list(render_candidate(proposal, round_index))
        if not preview_images:
            raise RobotPlacementAgentError("robot_placement_candidate_previews_missing")
        preview_metadata = _image_metadata(preview_images)
        review_result = invoker.invoke(
            AgentsSDKAgentSpec(
                run_id=run_id,
                capability="task_aware_robot_placement_visual_review",
                name="Blueprint Robot Placement Visual Reviewer",
                instructions=review_instructions,
                model=ROBOT_PLACEMENT_AGENT_MODEL,
                max_turns=1,
                max_output_tokens=ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS,
                max_input_tokens=max_input_tokens,
                reasoning_effort=ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
                output_type=RobotPlacementVisualReviewOutput,
            ),
            _multimodal_input(
                prompt={
                    "schema_version": ROBOT_PLACEMENT_AGENT_SCHEMA_VERSION,
                    "run_id": run_id,
                    "round_index": round_index,
                    "scene_binding": scene,
                    "task_binding": task,
                    "scene_context": scene_advisory_context,
                    "task_context": task_advisory_context,
                    "proposal": proposal,
                    "geometry_gate": geometry_gate,
                    "authority_boundary": {
                        "visual_review_is_advisory": True,
                        "visual_review_may_veto": True,
                        "visual_review_may_not_override_geometry": True,
                    },
                },
                images=preview_images,
            ),
        )
        visual_review = RobotPlacementVisualReviewOutput.model_validate(
            review_result.output
        )
        round_record["preview_images"] = preview_metadata
        round_record["visual_review"] = visual_review.model_dump(mode="json")
        round_record["visual_review_provider"] = review_result.provider
        round_record["visual_review_model"] = review_result.model
        round_record["visual_review_sdk_version"] = review_result.sdk_version
        round_record["visual_review_usage"] = dict(review_result.usage)
        round_record["visual_review_trace_id"] = review_result.trace_id
        history.append(round_record)
        if _visual_passed(visual_review):
            if execute_candidate is None:
                accepted = round_record
                break
            provisional_receipt = _build_placement_receipt(
                run_id=run_id,
                scene_digest=scene_digest,
                task_digest=task_digest,
                scene_context_digest=scene_context_digest,
                task_context_digest=task_context_digest,
                overview_images=overview_images,
                prior_native_attempts=prior_attempt_records,
                history=history,
                accepted=round_record,
                max_rounds=max_rounds,
                native_loop_enabled=False,
                task_trajectory_digest=(
                    trajectory["trajectory_digest"] if trajectory is not None else None
                ),
            )
            native_attempt = _validated_native_attempt(
                execute_candidate(proposal, provisional_receipt, round_index)
            )
            native_feedback_images = list(
                native_attempt.pop("_feedback_image_inputs")
            )
            round_record["native_attempt"] = native_attempt
            if native_attempt["status"] == "passed":
                accepted = round_record
                break

    return _build_placement_receipt(
        run_id=run_id,
        scene_digest=scene_digest,
        task_digest=task_digest,
        scene_context_digest=scene_context_digest,
        task_context_digest=task_context_digest,
        overview_images=overview_images,
        prior_native_attempts=prior_attempt_records,
        history=history,
        accepted=accepted,
        max_rounds=max_rounds,
        native_loop_enabled=execute_candidate is not None,
        task_trajectory_digest=(
            trajectory["trajectory_digest"] if trajectory is not None else None
        ),
    )


def validate_robot_placement_receipt(
    value: Mapping[str, Any],
    *,
    expected_scene_binding_digest: str | None = None,
    expected_task_binding_digest: str | None = None,
) -> dict[str, Any]:
    """Validate an immutable accepted receipt before native compilation."""

    receipt = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        receipt.get("schema_version") != ROBOT_PLACEMENT_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "accepted"
        or receipt.get("model") != ROBOT_PLACEMENT_AGENT_MODEL
        or receipt.get("reasoning_effort") != ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        or receipt.get("candidate_may_self_authorize") is not False
        or receipt.get("physical_execution_authorized") is not False
        or not isinstance(receipt.get("native_construction_required"), bool)
        or (
            receipt.get("task_trajectory_digest") is not None
            and not (
                isinstance(receipt.get("task_trajectory_digest"), str)
                and str(receipt["task_trajectory_digest"]).startswith("sha256:")
                and len(str(receipt["task_trajectory_digest"])) == 71
            )
        )
        or receipt.get("model_grades_controls") is not False
        or receipt.get("claim_ceiling") != ROBOT_PLACEMENT_CLAIM_CEILING
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise RobotPlacementAgentError("robot_placement_receipt_invalid")
    if (
        expected_scene_binding_digest is not None
        and receipt.get("scene_binding_digest") != expected_scene_binding_digest
    ):
        raise RobotPlacementAgentError("robot_placement_scene_binding_mismatch")
    if (
        expected_task_binding_digest is not None
        and receipt.get("task_binding_digest") != expected_task_binding_digest
    ):
        raise RobotPlacementAgentError("robot_placement_task_binding_mismatch")
    pose = RobotBasePoseOutput.model_validate(receipt.get("accepted_pose"))
    rounds = receipt.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise RobotPlacementAgentError("robot_placement_receipt_rounds_missing")
    accepted = rounds[-1]
    if (
        not isinstance(accepted, Mapping)
        or (accepted.get("proposal") or {}).get("candidate_id")
        != receipt.get("accepted_candidate_id")
        or (accepted.get("geometry_gate") or {}).get("status") != "passed"
        or (accepted.get("geometry_gate") or {}).get("geometry_gate_digest")
        != receipt.get("accepted_geometry_gate_digest")
    ):
        raise RobotPlacementAgentError("robot_placement_receipt_acceptance_invalid")
    visual = RobotPlacementVisualReviewOutput.model_validate(
        accepted.get("visual_review")
    )
    if not _visual_passed(visual):
        raise RobotPlacementAgentError("robot_placement_visual_acceptance_invalid")
    if receipt.get("native_agent_loop_enabled") is True:
        native_attempt = accepted.get("native_attempt")
        if (
            not isinstance(native_attempt, Mapping)
            or native_attempt.get("status") != "passed"
            or native_attempt.get("native_attempt_digest")
            != receipt.get("accepted_native_attempt_digest")
            or receipt.get("native_construction_required") is not False
            or receipt.get("task_trajectory_digest") is None
        ):
            raise RobotPlacementAgentError("robot_placement_native_acceptance_invalid")
        _validated_native_attempt(native_attempt)
    elif receipt.get("native_construction_required") is not True:
        raise RobotPlacementAgentError("robot_placement_native_boundary_invalid")
    receipt["accepted_pose"] = pose.model_dump(mode="json")
    return receipt


__all__ = [
    "DEFAULT_MAX_PLACEMENT_ROUNDS",
    "ROBOT_PLACEMENT_AGENT_MODEL",
    "ROBOT_PLACEMENT_AGENT_MAX_OUTPUT_TOKENS",
    "ROBOT_PLACEMENT_AGENT_REASONING_EFFORT",
    "ROBOT_PLACEMENT_RECEIPT_SCHEMA_VERSION",
    "RobotPlacementAgentError",
    "RobotPlacementProposalOutput",
    "RobotPlacementVisualReviewOutput",
    "PlacementExecutor",
    "robot_placement_agents_sdk_config",
    "run_task_evaluation_robot_placement_agent",
    "validate_robot_placement_receipt",
]
