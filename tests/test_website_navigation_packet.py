"""A compiled navigation spec becomes the G1 checkpoints' goal, its basis and a machine-composed authority.

Owner request 2026-11-06: a website navigation job reaches a scored G1 run. The authority stands in
for a named human reviewer only on the site's confirmed brief (owner decision 4), and every clamp
or restyle is disclosed.
"""
from __future__ import annotations

import copy
import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_navigation_goal import (
    MACHINE_AUTHORITY_SCHEMA_VERSION,
    NAVIGATION_TASK_KIND,
    PUBLISHED_TASK_INSTRUCTION,
    compose_g1_navigation_goal_authority,
    score_g1_navigation_episode,
    validate_g1_navigation_goal,
    validate_g1_navigation_goal_authority,
)
from blueprint_pipeline.task_spec_compiler import compile_task_spec
from blueprint_pipeline.website_navigation_packet import (
    WebsiteNavigationPacketError,
    g1_navigation_goal_from_spec,
    navigation_task_spec,
)
from tests.test_task_spec_compiler import _aisle_brief, _aisle_scene

def _spec(scale="unverified", **brief):
    return compile_task_spec(brief={**_aisle_brief(), **brief}, scene=_aisle_scene(scale),
                             task_requirements={"locomote": True, "manipulate": False})


def _context(spec, *, confirmed=True):
    """The site's own sealed task context (website_site_task_context.v1)."""
    context = {"schema_version": "website_site_task_context.v1", "scene_id": spec["site_id"],
               "capture_id": "walkthrough-a", "description": spec["sentence"], "task_family": "general_task",
               "confirmed": confirmed, "confirmed_at": "2026-11-06T18:30:00Z" if confirmed else None}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _resealed(spec, radius):
    """The same spec with another goal tolerance, sealed as the compiler seals it."""
    from blueprint_pipeline.task_predicates import seal_predicates

    value = copy.deepcopy(spec)
    rows = [dict(row) for row in value["success"]["predicates"]]
    for row in rows:
        if row["id"] == "goal":
            row["region"] = {**row["region"], "radius_m": radius}
    value["success"] = seal_predicates([{k: v for k, v in row.items()} for row in rows])
    value["markers"] = [{**marker, "radius_m": radius} for marker in value["markers"]]
    value["spec_digest"] = canonical_digest(value, digest_field="spec_digest")
    return value


def _plan(task_spec):
    plan = {"task_kind": NAVIGATION_TASK_KIND, "task_id": task_spec["task_id"], "robot": {"robot_id": "unitree_g1"},
            "task_spec": task_spec}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def test_the_compiled_goal_becomes_the_checkpoints_goal_with_every_assumption_disclosed():
    spec = _spec()
    goal, assumptions = g1_navigation_goal_from_spec(spec)
    region = next(row for row in spec["success"]["predicates"] if row["id"] == "goal")["region"]
    assert goal["center_world_m"] == region["center_world_m"]
    assert goal["visible_target_marker"]["shape"] == "flat_yellow_disc"
    assert goal["visible_target_marker"]["surface_position_world_m"] == goal["center_world_m"]
    assert goal["task_instruction"] == PUBLISHED_TASK_INSTRUCTION
    assert 0.1 <= goal["acceptance_radius_m"] <= 0.5
    assert "instruction_is_the_checkpoints_trained_prompt_not_the_site_sentence" in assumptions
    assert any(item.startswith("compiled:metric_scale_") for item in assumptions)
    validate_g1_navigation_goal({"task_kind": NAVIGATION_TASK_KIND, "g1_navigation_goal": goal})


@pytest.mark.parametrize("radius,clamped", [(0.9, 0.5), (0.05, 0.1)])
def test_a_goal_outside_what_the_checkpoints_accept_is_clamped_and_said_so(radius, clamped):
    goal, assumptions = g1_navigation_goal_from_spec(_resealed(_spec(), radius))
    assert goal["acceptance_radius_m"] == clamped == goal["visible_target_marker"]["radius_m"]
    assert f"acceptance_radius_clamped_from_{radius:.3f}m_to_{clamped:.3f}m" in assumptions


def test_the_navigation_task_spec_faces_the_goal_and_carries_the_compiled_spec():
    spec = _spec()
    task = navigation_task_spec(spec, task_context=_context(spec))
    assert task["task_kind"] == NAVIGATION_TASK_KIND and task["claim_ceiling"] == "development_only"
    assert task["compiled_task_spec"] == spec and task["compiled_task_spec_digest"] == spec["spec_digest"]
    x, y = task["start_pose_world"][:2]
    gx, gy = task["g1_navigation_goal"]["center_world_m"][:2]
    qz, qw = task["start_pose_world"][5:7]
    assert math.isclose(2 * math.atan2(qz, qw), math.atan2(gy - y, gx - x), abs_tol=1e-9)
    assert task["navigation_goal_basis"]["compiled_task_spec_digest"] == spec["spec_digest"]
    assert task["task_spec_digest"] == canonical_digest(task, digest_field="task_spec_digest")


def test_an_arm_spec_is_not_a_navigation_task():
    spec = _spec()
    bad = copy.deepcopy(spec)
    bad["requirements"] = {"locomote": False, "manipulate": True}
    bad["spec_digest"] = canonical_digest(bad, digest_field="spec_digest")
    with pytest.raises(WebsiteNavigationPacketError):
        g1_navigation_goal_from_spec(bad)


def test_the_machine_composed_authority_needs_the_sites_own_confirmed_context():
    spec = _spec()
    context = _context(spec)
    plan = _plan(navigation_task_spec(spec, task_context=context))
    authority = compose_g1_navigation_goal_authority(plan=plan, task_context=context)
    assert authority["schema_version"] == MACHINE_AUTHORITY_SCHEMA_VERSION
    assert authority["human_reviewer"] is None and authority["claim_ceiling"] == "development_only"
    assert authority["task_context"] == context
    assert authority["disclosed_assumptions"] == plan["task_spec"]["navigation_goal_basis"]["disclosed_assumptions"]
    # The shared validator dispatches on the schema version.
    assert validate_g1_navigation_goal_authority(authority, plan=plan) == authority
    # An unconfirmed context refuses; so does a forged "confirmed" over an unconfirmed context's
    # real digest (the digest is recomputed), and a context for another digest.
    unconfirmed = _context(spec, confirmed=False)
    unconfirmed_plan = _plan(navigation_task_spec(spec, task_context=unconfirmed))
    with pytest.raises(ValueError, match="g1_navigation_goal_brief_not_confirmed"):
        compose_g1_navigation_goal_authority(plan=unconfirmed_plan, task_context=unconfirmed)
    forged = {**unconfirmed, "confirmed": True, "confirmed_at": "whenever"}
    with pytest.raises(ValueError, match="g1_navigation_goal_task_context_invalid"):
        compose_g1_navigation_goal_authority(plan=unconfirmed_plan, task_context=forged)
    tampered = {**authority, "obstacle_clearance_scored": True}
    with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
        validate_g1_navigation_goal_authority(tampered, plan=plan)
    other = _plan({**plan["task_spec"], "task_id": "another"})
    with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
        validate_g1_navigation_goal_authority(authority, plan=other)


def test_the_goal_and_basis_must_be_what_the_plans_compiled_spec_produces():
    spec = _spec()
    context = _context(spec)
    task = navigation_task_spec(spec, task_context=context)
    moved = copy.deepcopy(task)
    for key in ("center_world_m",):
        moved["g1_navigation_goal"][key] = [99.0, 99.0, moved["g1_navigation_goal"][key][2]]
    moved["g1_navigation_goal"]["visible_target_marker"]["surface_position_world_m"] = moved["g1_navigation_goal"]["center_world_m"]
    elsewhere = copy.deepcopy(task)
    elsewhere["navigation_goal_basis"] = {**elsewhere["navigation_goal_basis"], "site_id": "some-other-site",
                                          "compiled_task_spec_digest": "sha256:" + "e" * 64}
    hidden = copy.deepcopy(task)
    hidden["navigation_goal_basis"]["disclosed_assumptions"] = ["nothing_assumed"]
    for bad in (moved, elsewhere, hidden):
        with pytest.raises(ValueError, match="g1_navigation_goal_spec_binding_invalid"):
            compose_g1_navigation_goal_authority(plan=_plan(bad), task_context=context)


def test_the_existing_scorer_scores_a_navigation_scene_goal():
    spec = _spec("validated")
    task = navigation_task_spec(spec, task_context=_context(spec))
    goal = task["g1_navigation_goal"]
    start = task["start_pose_world"][:3]
    settle = goal["settle_window_samples"]
    points = [start, *([[*goal["center_world_m"][:2], start[2]]] * (settle + 2))]
    samples = [{"step_index": index, "root_position_world_m": point} for index, point in enumerate(points)]
    report = score_g1_navigation_episode(task_spec=task, samples=samples)
    assert report["outcome"] == "success" and report["obstacle_clearance_scored"] is False
