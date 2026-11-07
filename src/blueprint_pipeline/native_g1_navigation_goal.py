"""Score a G1 navigation policy against a visible goal in the same task scene.

This is a development simulator criterion for the pinned HSI vision-navigation
checkpoint. It measures root position from Isaac, not a policy self-report.
Obstacle avoidance and physical transfer are outside this score.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp_task_scoring import TaskNeutralScoringError, validate_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


SCHEMA_VERSION = "native_g1_navigation_goal.v1"
SCORE_SCHEMA_VERSION = "native_g1_navigation_goal_score.v1"
AUTHORITY_SCHEMA_VERSION = "native_g1_navigation_goal_authority.v1"
PUBLISHED_TASK_INSTRUCTION = "Avoid obstacles and move to the yellow marked area."
#: A navigation goal rides on a rigid task's scene (the published HSI setup) or on its own
#: navigation scene, compiled from the site's brief (``website_navigation_packet``).
NAVIGATION_TASK_KIND = "goal_region_navigation"
GOAL_HOST_TASK_KINDS = frozenset({"rigid_pick_place", NAVIGATION_TASK_KIND})
#: Owner decision 4 (2026-11-06 15:07Z): the site's brief plus a machine-composed authority replace the
#: named human reviewer and team, recorded as a disclosed assumption. The authority only records whether
#: the site confirmed its brief (``site_brief_confirmed``); the job holds until the site confirms its brief
#: on the website (a confirmed brief counts as consent, owner 2026-11-07), by name at the live consent
#: recheck or at the WebApp's ``task_confirmed``. ``v3``: the basis and status say whether the site
#: confirmed its brief, recomputed from its sealed context (``v2`` called every brief confirmed; none was
#: ever stored).
MACHINE_AUTHORITY_SCHEMA_VERSION = "native_g1_navigation_goal_authority.v3"
GOAL_BASIS_SCHEMA_VERSION = "native_g1_navigation_goal_basis.v1"
#: Keyed by whether the site confirmed its brief (``site_brief_confirmed``). An unconfirmed brief's basis
#: cites no owner decision: no decision makes an unconfirmed brief a basis, so it is only disclosed.
MACHINE_AUTHORITY_BASIS = {
    True: "owner_decision_2026_11_06_confirmed_brief_and_machine_composed_goal",
    False: "machine_composed_goal_site_brief_unconfirmed_disclosed",
}
MACHINE_AUTHORITY_STATUS = {
    True: "confirmed_for_development_simulation",
    False: "composed_for_development_simulation_site_brief_unconfirmed",
}
MACHINE_AUTHORITY_COMPOSER = "blueprint_task_spec_compiler"


def validate_g1_navigation_goal(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    goal = task_spec.get("g1_navigation_goal")
    marker = goal.get("visible_target_marker") if isinstance(goal, Mapping) else None
    if not isinstance(goal, Mapping) or not isinstance(marker, Mapping):
        raise ValueError("g1_navigation_goal_or_visible_marker_missing")
    try:
        center = [float(value) for value in goal["center_world_m"]]
        surface = [float(value) for value in marker["surface_position_world_m"]]
        radius = float(goal["acceptance_radius_m"])
        marker_radius = float(marker["radius_m"])
        max_height_drift = float(goal["max_root_height_drift_m"])
        settle = goal["settle_window_samples"]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("g1_navigation_goal_invalid") from exc
    if (
        task_spec.get("task_kind") not in GOAL_HOST_TASK_KINDS
        or goal.get("schema_version") != SCHEMA_VERSION
        or goal.get("task_instruction") != PUBLISHED_TASK_INSTRUCTION
        or marker.get("schema_version") != "native_task_target_marker.v1"
        or marker.get("shape") != "flat_yellow_disc"
        or marker.get("non_colliding") is not True
        or len(center) != 3 or len(surface) != 3
        or not all(math.isfinite(value) for value in (*center, *surface, radius, marker_radius, max_height_drift))
        or center != surface
        or not 0.1 <= radius <= marker_radius <= 0.5
        or not 0.05 <= max_height_drift <= 0.5
        or isinstance(settle, bool) or not isinstance(settle, int)
        or not 2 <= settle <= 100
    ):
        raise ValueError("g1_navigation_goal_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "center_world_m": center,
        "acceptance_radius_m": radius,
        "max_root_height_drift_m": max_height_drift,
        "settle_window_samples": settle,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1",
            "shape": "flat_yellow_disc",
            "non_colliding": True,
            "surface_position_world_m": surface,
            "radius_m": marker_radius,
        },
    }


def validate_g1_navigation_goal_authority(
    value: Mapping[str, Any] | None, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind team-confirmed goal arrival to one sealed G1 task/site scene.

    A ``v3`` authority (a navigation scene compiled from the site's brief, whether the site
    confirmed it disclosed) is checked by ``validate_machine_composed_goal_authority``.
    """

    if isinstance(value, Mapping) and value.get("schema_version") == MACHINE_AUTHORITY_SCHEMA_VERSION:
        return validate_machine_composed_goal_authority(value, plan=plan)
    try:
        authority = json.loads(json.dumps(value, allow_nan=False))
        if (
            plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan.get("task_kind") != "rigid_pick_place"
            or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        ):
            raise ValueError("scene_plan_mismatch")
        task_spec = plan["task_spec"]
        goal = validate_g1_navigation_goal(task_spec)
        task_contract = validate_rigid_task_success_contract(
            task_spec["task_success_contract"],
            expected_task_id=plan["task_id"],
        )
        if task_spec.get("task_success_contract_digest") != task_contract["contract_digest"]:
            raise ValueError("task_contract_digest_mismatch")
        team_id = task_contract["provenance"]["confirmed_by_team_id"]
        expected_scope = {
            "site_id": task_contract["scope"]["site_id"],
            "task_id": plan["task_id"],
            "scene_plan_digest": plan["plan_digest"],
            "task_success_contract_digest": task_contract["contract_digest"],
        }
    except (AttributeError, KeyError, TypeError, ValueError, TaskNeutralScoringError) as exc:
        raise ValueError("g1_navigation_goal_authority_invalid") from exc
    if (
        not isinstance(authority, dict)
        or set(authority) != {
            "schema_version", "status", "scope", "goal", "goal_digest",
            "confirmed_by_team_id", "human_reviewer", "reviewed_criterion",
            "obstacle_clearance_scored", "physical_outcome_claimed", "authority_digest",
        }
        or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "confirmed_for_development_simulation"
        or authority.get("scope") != expected_scope
        or authority.get("goal") != goal
        or authority.get("goal_digest") != cross_runtime_canonical_digest(goal)
        or not isinstance(team_id, str) or not team_id.strip()
        or authority.get("confirmed_by_team_id") != team_id
        or not isinstance(authority.get("human_reviewer"), str)
        or not authority["human_reviewer"].strip()
        or authority.get("reviewed_criterion") != "root_xy_goal_arrival_and_terminal_hold"
        or authority.get("obstacle_clearance_scored") is not False
        or authority.get("physical_outcome_claimed") is not False
        or authority.get("authority_digest") != cross_runtime_canonical_digest(
            authority, digest_field="authority_digest"
        )
    ):
        raise ValueError("g1_navigation_goal_authority_invalid")
    return authority


def seal_g1_navigation_goal_authority(
    *, plan: Mapping[str, Any], confirmed_by_team_id: str, human_reviewer: str
) -> dict[str, Any]:
    """Seal a recorded human decision after the caller has obtained it."""

    task_spec = plan["task_spec"]
    task_contract = validate_rigid_task_success_contract(
        task_spec["task_success_contract"], expected_task_id=plan["task_id"]
    )
    authority = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "status": "confirmed_for_development_simulation",
        "scope": {
            "site_id": task_contract["scope"]["site_id"],
            "task_id": plan["task_id"],
            "scene_plan_digest": plan["plan_digest"],
            "task_success_contract_digest": task_contract["contract_digest"],
        },
        "goal": validate_g1_navigation_goal(task_spec),
        "confirmed_by_team_id": confirmed_by_team_id,
        "human_reviewer": human_reviewer,
        "reviewed_criterion": "root_xy_goal_arrival_and_terminal_hold",
        "obstacle_clearance_scored": False,
        "physical_outcome_claimed": False,
    }
    authority["goal_digest"] = cross_runtime_canonical_digest(authority["goal"])
    authority["authority_digest"] = cross_runtime_canonical_digest(
        authority, digest_field="authority_digest"
    )
    return validate_g1_navigation_goal_authority(authority, plan=plan)


#: The checkpoints' validated goal envelope (``validate_g1_navigation_goal``) and the G1 lane's rate.
ACCEPTANCE_RADIUS_BOUNDS_M = (0.1, 0.5)
HEIGHT_DRIFT_BOUNDS_M = (0.05, 0.5)
SETTLE_SAMPLE_BOUNDS = (2, 100)
G1_CONTROL_FREQUENCY_HZ = 50.0
#: Without a time limit in the brief, an episode may last this long.
DEFAULT_EPISODE_SECONDS = 60.0


def _clamp(value: float, bounds: Sequence[float]) -> float:
    return float(min(max(value, bounds[0]), bounds[1]))


def _compiled_rows(spec: Mapping[str, Any], kind: str, predicate_id: str | None = None) -> list[dict[str, Any]]:
    try:
        rows = [dict(row) for row in spec["success"]["predicates"]
                if row.get("kind") == kind and (predicate_id is None or row.get("id") == predicate_id)]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("g1_navigation_goal_spec_invalid") from exc
    return rows


def goal_from_compiled_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The checkpoints' goal for a compiled navigation spec, and every assumption it took.

    Pure, so the authority validator can re-derive it wherever it runs: no compiler import.
    The caller has validated the spec (``website_navigation_packet``); the validator checks its seal.
    """
    goals, uprights = _compiled_rows(spec, "region_reached", "goal"), _compiled_rows(spec, "height_stable")
    if len(goals) != 1 or len(uprights) != 1:
        raise ValueError("g1_navigation_goal_spec_invalid")
    goal_row, upright = goals[0], uprights[0]
    region = goal_row.get("region") or {}
    hold = goal_row.get("hold_samples")
    if (region.get("shape") != "disc" or goal_row.get("subject") != {"kind": "robot_root"}
            or isinstance(hold, bool) or not isinstance(hold, int) or hold < 1):
        raise ValueError("g1_navigation_goal_spec_invalid")
    try:
        centre = [float(value) for value in region["center_world_m"]]
        compiled_radius, compiled_drift = float(region["radius_m"]), float(upright["max_drift_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("g1_navigation_goal_spec_invalid") from exc
    radius = _clamp(compiled_radius, ACCEPTANCE_RADIUS_BOUNDS_M)
    drift = _clamp(compiled_drift, HEIGHT_DRIFT_BOUNDS_M)
    settle = int(_clamp(hold, SETTLE_SAMPLE_BOUNDS))
    assumptions = [
        "goal_marker_restyled_flat_yellow_disc_for_the_checkpoints_trained_prompt",
        "instruction_is_the_checkpoints_trained_prompt_not_the_site_sentence",
        f"settle_window_is_{settle}_samples_at_{G1_CONTROL_FREQUENCY_HZ:g}_hz",
    ]
    if not math.isclose(radius, compiled_radius):
        assumptions.append(f"acceptance_radius_clamped_from_{compiled_radius:.3f}m_to_{radius:.3f}m")
    if not math.isclose(drift, compiled_drift):
        assumptions.append(f"root_height_drift_clamped_to_{drift:.3f}m")
    if settle != hold:
        assumptions.append(f"settle_window_clamped_from_{hold}_samples")
    assumptions.extend(f"compiled:{flag}" for flag in spec.get("flags") or [])
    if not _compiled_rows(spec, "time_limit"):
        assumptions.append(f"episode_limit_defaulted_to_{DEFAULT_EPISODE_SECONDS:g}s")
    goal = {
        "schema_version": SCHEMA_VERSION, "center_world_m": centre, "acceptance_radius_m": radius,
        "max_root_height_drift_m": drift, "settle_window_samples": settle,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {"schema_version": "native_task_target_marker.v1", "shape": "flat_yellow_disc",
                                  "non_colliding": True, "surface_position_world_m": list(centre), "radius_m": radius},
    }
    validate_g1_navigation_goal({"task_kind": NAVIGATION_TASK_KIND, "g1_navigation_goal": goal})
    return goal, assumptions


def _goal_basis(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """What a compiled navigation goal was made from: the site, the compiled spec and the brief."""
    basis = task_spec.get("navigation_goal_basis")
    if (not isinstance(basis, Mapping)
            or set(basis) != {"schema_version", "site_id", "compiled_task_spec_digest", "task_context_digest",
                              "disclosed_assumptions"}
            or basis["schema_version"] != GOAL_BASIS_SCHEMA_VERSION
            or not all(isinstance(basis[key], str) and basis[key].strip()
                       for key in ("site_id", "compiled_task_spec_digest", "task_context_digest"))
            or not isinstance(basis["disclosed_assumptions"], list)
            or not all(isinstance(item, str) and item for item in basis["disclosed_assumptions"])):
        raise ValueError("g1_navigation_goal_basis_invalid")
    return dict(basis)


def _machine_scope(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The plan's goal, basis and scope, each tied to the sealed compiled spec the plan carries.

    The spec's own seal is recomputed, and the goal and every disclosed assumption are re-derived
    from it, so neither a moved goal nor another site's basis can ride a genuine plan.
    """
    if (plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
            or plan.get("task_kind") != NAVIGATION_TASK_KIND
            or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"):
        raise ValueError("scene_plan_mismatch")
    task_spec = plan["task_spec"]
    goal = validate_g1_navigation_goal(task_spec)
    basis = _goal_basis(task_spec)
    spec = task_spec.get("compiled_task_spec")
    if (not isinstance(spec, Mapping)
            or spec.get("spec_digest") != canonical_digest(spec, digest_field="spec_digest")
            or not (spec["spec_digest"] == basis["compiled_task_spec_digest"]
                    == task_spec.get("compiled_task_spec_digest"))
            or not (spec.get("site_id") == basis["site_id"] == task_spec.get("site_id"))
            or spec.get("task_id") != plan["task_id"] or task_spec.get("task_id") != plan["task_id"]):
        raise ValueError("g1_navigation_goal_spec_binding_invalid")
    derived, assumptions = goal_from_compiled_spec(spec)
    if goal != derived or basis["disclosed_assumptions"] != assumptions:
        raise ValueError("g1_navigation_goal_spec_binding_invalid")
    scope = {"site_id": basis["site_id"], "task_id": plan["task_id"], "scene_plan_digest": plan["plan_digest"],
             "compiled_task_spec_digest": basis["compiled_task_spec_digest"],
             "task_context_digest": basis["task_context_digest"]}
    return goal, basis, scope


def _site_context(context: Any, *, basis: Mapping[str, Any]) -> dict[str, Any]:
    """The site's own task context, sealed and for the same site.

    Its digest is recomputed, so a caller cannot change a context and keep its seal. Whether the
    site confirmed its brief is not checked here: the authority records it (``site_brief_confirmed``),
    and the job holds until the site confirms its brief on the website (a confirmed brief counts as
    consent, owner 2026-11-07).
    """
    if not isinstance(context, Mapping) or context.get("context_digest") != canonical_digest(
            context, digest_field="context_digest"):
        raise ValueError("g1_navigation_goal_task_context_invalid")
    if context["context_digest"] != basis["task_context_digest"] or context.get("scene_id") != basis["site_id"]:
        raise ValueError("g1_navigation_goal_task_context_invalid")
    return dict(context)


def _site_brief_confirmed(context: Mapping[str, Any]) -> bool:
    """True only when the site confirmed its brief, with the time it did."""
    confirmed_at = context.get("confirmed_at")
    return context.get("confirmed") is True and isinstance(confirmed_at, str) and bool(confirmed_at.strip())


def validate_machine_composed_goal_authority(value: Mapping[str, Any] | None, *,
                                             plan: Mapping[str, Any]) -> dict[str, Any]:
    """Owner decision 4: the site's brief plus a machine-composed goal stands in for a named reviewer.

    Whether the site confirmed its brief is recomputed from the sealed context it carries, and the
    authority's ``site_brief_confirmed``, ``status`` and ``authority_basis`` must all say the same.
    """
    try:
        authority = json.loads(json.dumps(value, allow_nan=False))
        goal, basis, scope = _machine_scope(plan)
        if (not isinstance(authority, dict)
                or set(authority) != {"schema_version", "status", "claim_ceiling", "scope", "goal", "goal_digest",
                                      "task_context", "site_brief_confirmed", "authority_basis", "composed_by",
                                      "human_reviewer", "disclosed_assumptions", "reviewed_criterion",
                                      "obstacle_clearance_scored", "physical_outcome_claimed", "authority_digest"}
                or authority["schema_version"] != MACHINE_AUTHORITY_SCHEMA_VERSION
                or authority["claim_ceiling"] != "development_only"
                or authority["scope"] != scope
                or authority["goal"] != goal
                or authority["goal_digest"] != cross_runtime_canonical_digest(goal)
                or authority["composed_by"] != MACHINE_AUTHORITY_COMPOSER
                or authority["human_reviewer"] is not None
                or authority["disclosed_assumptions"] != basis["disclosed_assumptions"]
                or authority["reviewed_criterion"] != NAVIGATION_CRITERION
                or authority["obstacle_clearance_scored"] is not False
                or authority["physical_outcome_claimed"] is not False
                or authority["authority_digest"] != cross_runtime_canonical_digest(
                    authority, digest_field="authority_digest")):
            raise ValueError("invalid")
        context = _site_context(authority["task_context"], basis=basis)
        confirmed = _site_brief_confirmed(context)
        if (authority["site_brief_confirmed"] is not confirmed
                or authority["status"] != MACHINE_AUTHORITY_STATUS[confirmed]
                or authority["authority_basis"] != MACHINE_AUTHORITY_BASIS[confirmed]):
            raise ValueError("invalid")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("g1_navigation_goal_authority_invalid") from exc
    return authority


def compose_g1_navigation_goal_authority(*, plan: Mapping[str, Any],
                                         task_context: Mapping[str, Any]) -> dict[str, Any]:
    """Compose the goal authority for a navigation scene from the site's own task context.

    No human reviewer is named or claimed. Whether the site confirmed its brief is recorded in
    ``site_brief_confirmed``, and the ``status`` and ``authority_basis`` say the same; the job holds
    until the site confirms its brief on the website (a confirmed brief counts as consent, owner
    2026-11-07), so the authority only records it. Refuses a context that is not the one the
    goal was compiled from (``g1_navigation_goal_task_context_invalid``), a goal or basis the
    plan's compiled spec does not produce (``g1_navigation_goal_spec_binding_invalid``), or a
    plan that is not a G1 navigation scene.
    """
    goal, basis, scope = _machine_scope(plan)
    context = _site_context(task_context, basis=basis)
    confirmed = _site_brief_confirmed(context)
    authority = {
        "schema_version": MACHINE_AUTHORITY_SCHEMA_VERSION,
        "status": MACHINE_AUTHORITY_STATUS[confirmed],
        "claim_ceiling": "development_only",
        "scope": scope,
        "goal": goal,
        "goal_digest": cross_runtime_canonical_digest(goal),
        "task_context": context,
        "site_brief_confirmed": confirmed,
        "authority_basis": MACHINE_AUTHORITY_BASIS[confirmed],
        "composed_by": MACHINE_AUTHORITY_COMPOSER,
        "human_reviewer": None,
        "disclosed_assumptions": list(basis["disclosed_assumptions"]),
        "reviewed_criterion": NAVIGATION_CRITERION,
        "obstacle_clearance_scored": False,
        "physical_outcome_claimed": False,
    }
    authority["authority_digest"] = cross_runtime_canonical_digest(authority, digest_field="authority_digest")
    return validate_machine_composed_goal_authority(authority, plan=plan)


#: What a navigation scene's G1 run is scored by (``score_g1_navigation_episode``): the root inside the
#: goal for the final settle window, and its height within the drift bound at every sample. Contact off
#: the feet is recorded per step (``native_task_runtime_contract.NAVIGATION_SAMPLE_BINDING``) and is
#: never part of success. ``v2`` names both success conditions and what voids an episode (``v1``
#: named only arrival and hold, though the scorer also failed an episode on root-height drift).
NAVIGATION_CONTRACT_SCHEMA_VERSION = "native_g1_navigation_task_success_contract.v2"
NAVIGATION_CRITERION = "root_xy_goal_arrival_terminal_hold_and_root_height_stable"
#: What the scorer refuses to score (it raises): such an episode is void, never a success or a failure.
NAVIGATION_VOID_CONDITIONS = ("fewer_than_two_samples", "root_readback_invalid", "start_already_at_goal")


def navigation_task_success_contract(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """The success contract of a compiled navigation task spec, bound to its sealed spec and goal."""
    goal = validate_g1_navigation_goal(task_spec)
    spec = task_spec.get("compiled_task_spec")
    if (task_spec.get("task_kind") != NAVIGATION_TASK_KIND or not isinstance(spec, Mapping)
            or spec.get("spec_digest") != canonical_digest(spec, digest_field="spec_digest")
            or spec["spec_digest"] != task_spec.get("compiled_task_spec_digest")
            or spec.get("site_id") != task_spec.get("site_id") or spec.get("task_id") != task_spec.get("task_id")):
        raise ValueError("g1_navigation_task_success_contract_invalid")
    contacts = sorted(str(row.get("id")) for row in _compiled_rows(spec, "no_contact"))
    contract = {
        "schema_version": NAVIGATION_CONTRACT_SCHEMA_VERSION, "task_kind": NAVIGATION_TASK_KIND,
        "scope": {"site_id": spec["site_id"], "task_id": spec["task_id"]},
        "compiled_task_spec_digest": spec["spec_digest"], "goal_digest": cross_runtime_canonical_digest(goal),
        "criterion": NAVIGATION_CRITERION, "scorer": "native_g1_navigation_goal.score_g1_navigation_episode",
        # Every condition the scorer fails an episode on; both must hold for success.
        "success_conditions": [
            {"condition": "terminal_goal_hold", "measured": "robot_root_xy",
             "acceptance_radius_m": goal["acceptance_radius_m"],
             "settle_window_samples": goal["settle_window_samples"]},
            {"condition": "root_height_stable_throughout", "measured": "robot_root_z",
             "max_root_height_drift_m": goal["max_root_height_drift_m"]},
        ],
        "void_conditions": list(NAVIGATION_VOID_CONDITIONS),
        "obstacle_contact": {"recorded_per_step": True, "part_of_success": False,
                             "compiled_predicate_ids": contacts},
        "claim_ceiling": "development_only", "physical_outcome_claimed": False,
    }
    contract["contract_digest"] = canonical_digest(contract, digest_field="contract_digest")
    return contract


def validate_navigation_task_success_contract(value: Any, *, task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a navigation contract the task spec does not produce exactly (``..._invalid``)."""
    expected = navigation_task_success_contract(task_spec)
    try:
        observed = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("g1_navigation_task_success_contract_invalid") from exc
    if observed != expected:
        raise ValueError("g1_navigation_task_success_contract_invalid")
    return expected


def score_g1_navigation_episode(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    goal = validate_g1_navigation_goal(task_spec)
    if len(samples) < 2:
        raise ValueError("g1_navigation_samples_missing")
    distances: list[float] = []
    heights: list[float] = []
    for index, sample in enumerate(samples):
        try:
            position = [float(value) for value in sample["root_position_world_m"]]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("g1_navigation_root_readback_invalid") from exc
        if (
            sample.get("step_index") != index
            or len(position) != 3
            or not all(math.isfinite(value) for value in position)
        ):
            raise ValueError("g1_navigation_root_readback_invalid")
        distances.append(math.dist(position[:2], goal["center_world_m"][:2]))
        heights.append(position[2])
    radius = goal["acceptance_radius_m"]
    if distances[0] <= radius:
        raise ValueError("g1_navigation_start_already_at_goal")
    height_drifts = [abs(height - heights[0]) for height in heights]
    height_stable_throughout = all(
        drift <= goal["max_root_height_drift_m"] for drift in height_drifts
    )
    within = [
        distance <= radius
        and drift <= goal["max_root_height_drift_m"]
        for distance, drift in zip(distances, height_drifts, strict=True)
    ]
    settle = goal["settle_window_samples"]
    first_settled_step = next(
        (index for index in range(settle - 1, len(within))
         if all(within[index - settle + 1:index + 1])),
        None,
    )
    terminal_hold = len(within) >= settle and all(within[-settle:])
    result = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "scored",
        "outcome": "success" if terminal_hold and height_stable_throughout else "failure",
        "first_settled_step": first_settled_step,
        "terminal_goal_hold": terminal_hold,
        "root_height_stable_throughout": height_stable_throughout,
        "maximum_root_height_drift_observed_m": max(height_drifts),
        "initial_distance_m": distances[0],
        "minimum_distance_m": min(distances),
        "terminal_distance_m": distances[-1],
        "sample_count": len(samples),
        "goal": goal,
        "criterion": "root_xy_inside_visible_goal_with_stable_height",
        "obstacle_clearance_scored": False,
        "evidence_level": "development_only",
    }
    result["score_digest"] = canonical_digest(result, digest_field="score_digest")
    return result
