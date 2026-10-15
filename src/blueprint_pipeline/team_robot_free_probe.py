"""Record a rigid scene's robot-free success check during a team's practice run, then retain it.

Beta journey step: L5 team policy evaluation. A team's scored evaluation is
refused until the scene's robot-free probe passed (``robot_free_probe_gate``):
left untouched the task must not succeed, and carried into its goal without a
robot it must. For rigid pick-and-place scenes that probe could not run: its
Isaac launcher was never wired, and it scored the configured revision's task
specification, not the one a team is scored with, so even a pass could not
admit a scored run.

The probe now runs where the scene already exists: the first GPU session of a
team's practice run on a scene whose probe has not been decided. It is
planned here, on the control plane, against the exact task specification and
success contract the team is scored with (``scored_task_spec``); the GPU host
only executes the plan in the built scene (``team_embodiment_runtime``)
before any episode, with the robot idle at its reset pose and never driving
the object. Closeout scores the recorded trace with the rigid readiness
scorer, retains the receipt where the gate reads it (a decided receipt is
never replaced), and republishes the job's offer, so scored evaluations open
the moment the probe passes. A probe that cannot run, or fails, never breaks
the practice run that hosts it.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCENE_CONTEXT = "native_arena_scene_robot_idle_at_reset"
REQUEST_SCHEMA = "blueprint.team_robot_free_probe_request.v1"


def scored_task_spec(task_spec: Mapping[str, Any], task_success_contract: Mapping[str, Any]) -> dict[str, Any]:
    """The task specification a team is scored with: the scene's, under the job's success contract."""

    value = deepcopy(dict(task_spec))
    value["task_success_contract"] = deepcopy(dict(task_success_contract))
    return value


def probe_request(*, task_spec: Mapping[str, Any], scene_revision_digest: str) -> dict[str, Any] | None:
    """What the GPU host executes, or None when this task cannot be probed without a robot."""

    from .rigid_robot_free_probe_plan import (
        RigidRobotFreeProbeUnsupported,
        physics_plan,
        probe_context,
    )

    if task_spec.get("task_kind") != "rigid_pick_place":
        return None
    try:
        plan = physics_plan(probe_context(task_spec))
    except (RigidRobotFreeProbeUnsupported, ValueError, KeyError, TypeError):
        # Not probeable without a robot (or not a contract the rigid scorer reads): the
        # receipt stays undecided and scored evaluations stay closed, by name, at the gate.
        return None
    binding = {"configured_scene_revision_digest": scene_revision_digest,
               "task_spec_digest": canonical_digest(dict(task_spec)), "replacement_asset_digest": None,
               "scene_context": SCENE_CONTEXT, "scene_collisions_modeled": True}
    request = {"schema_version": REQUEST_SCHEMA, "plan": plan, "binding": binding, "request_digest": ""}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def retain_recorded_probe(*, profile: Mapping[str, Any], facts: Mapping[str, Any],
                          trace: Mapping[str, Any]) -> dict[str, Any]:
    """Score the recorded trace, retain the receipt beside the source launch, and republish the offer."""

    from .agent_execution_offer import publish_agent_execution_offer
    from .robot_free_readiness_retention import retain_receipt
    from .task_evaluation_scene_rigid_robot_free_readiness import score_rigid_robot_free_readiness

    scene = facts["scene"]
    receipt = score_rigid_robot_free_readiness(task_spec=scene["task_spec"], physics_probe=trace,
                                               configured_scene_revision_digest=scene["scene_revision_digest"])
    write = retain_receipt(Path(profile["robot_free_probe_root"]), receipt)
    offer = publish_agent_execution_offer(Path(profile["capture_root"]))
    return {"receipt_status": receipt["status"], "receipt_digest": receipt["receipt_digest"],
            "probe_error": receipt.get("probe_error"), "write": write, "offer_status": offer.get("status")}


__all__ = ["REQUEST_SCHEMA", "SCENE_CONTEXT", "probe_request", "retain_recorded_probe", "scored_task_spec"]
