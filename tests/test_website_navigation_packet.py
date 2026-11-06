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

CONTEXT = {"scene_id": None, "context_digest": "sha256:" + "c" * 64}


def _spec(scale="unverified", **brief):
    return compile_task_spec(brief={**_aisle_brief(), **brief}, scene=_aisle_scene(scale),
                             task_requirements={"locomote": True, "manipulate": False})


def _context(spec):
    return {**CONTEXT, "scene_id": spec["site_id"]}


def _plan(task_spec):
    plan = {"task_kind": NAVIGATION_TASK_KIND, "task_id": task_spec["task_id"], "robot": {"robot_id": "unitree_g1"},
            "task_spec": task_spec}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _brief(spec, **changes):
    return {"context_digest": CONTEXT["context_digest"], "description": spec["sentence"], "confirmed": True,
            "confirmed_at": "2026-11-06T18:30:00Z", **changes}


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


def test_a_goal_wider_than_the_checkpoints_accept_is_clamped_and_said_so():
    spec = _spec(goal_tolerance_m=0.9)
    radius = next(row for row in spec["success"]["predicates"] if row["id"] == "goal")["region"]["radius_m"]
    goal, assumptions = g1_navigation_goal_from_spec(spec)
    if radius > 0.5:
        assert goal["acceptance_radius_m"] == 0.5
        assert any(item.startswith("acceptance_radius_clamped_from_") for item in assumptions)


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


def test_the_machine_composed_authority_needs_the_sites_confirmed_brief():
    spec = _spec()
    plan = _plan(navigation_task_spec(spec, task_context=_context(spec)))
    authority = compose_g1_navigation_goal_authority(plan=plan, brief=_brief(spec))
    assert authority["schema_version"] == MACHINE_AUTHORITY_SCHEMA_VERSION
    assert authority["human_reviewer"] is None and authority["claim_ceiling"] == "development_only"
    assert authority["disclosed_assumptions"] == plan["task_spec"]["navigation_goal_basis"]["disclosed_assumptions"]
    # The shared validator dispatches on the schema, so every existing caller accepts it.
    assert validate_g1_navigation_goal_authority(authority, plan=plan) == authority
    for unconfirmed in (_brief(spec, confirmed=False), _brief(spec, confirmed_at=None),
                        _brief(spec, context_digest="sha256:" + "d" * 64)):
        with pytest.raises(ValueError, match="g1_navigation_goal_brief_not_confirmed"):
            compose_g1_navigation_goal_authority(plan=plan, brief=unconfirmed)
    tampered = {**authority, "obstacle_clearance_scored": True}
    with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
        validate_g1_navigation_goal_authority(tampered, plan=plan)
    other = _plan({**plan["task_spec"], "task_id": "another"})
    with pytest.raises(ValueError, match="g1_navigation_goal_authority_invalid"):
        validate_g1_navigation_goal_authority(authority, plan=other)


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
