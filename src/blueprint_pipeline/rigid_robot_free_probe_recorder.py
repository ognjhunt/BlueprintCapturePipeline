"""Execute the rigid robot-free probe plan in a live simulator; record, never score.

Beta journey step: L5 team policy evaluation. The articulated probe ramps a
door joint inside the native import run; a rigid object has no joint, so its
virtual carrier sets the object's pose directly. The scene is left untouched
first, then the object is carried (pose set every physics step, velocities
zeroed, recorded as carry so it can never count as success) to just above
the goal and let go, and physics alone decides where it rests.

The simulator is reached through a three-method seam so the same recorder
drives Isaac on a GPU host (``rigid_robot_free_probe_isaac``) and a
deterministic stand-in in hermetic tests:

- ``set_scoring_pose(pose)``: place the object's scoring frame at ``pose``
  (xyz + xyzw) with zero linear and angular velocity;
- ``step()``: advance physics by one ``1 / physics_hz`` step;
- ``read_sample()``: the native readback, ``task_scoring_pose_world`` (7),
  ``support_contact_active`` (bool) and ``scene_collision_force_n`` (float).

A failure is recorded as ``status: failed`` with its reason and never raised,
so a probe can never break the run that hosts it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_plan import PLAN_SCHEMA, interpolate_pose, carry_waypoints

TRACE_SCHEMA = "scene_rigid_robot_free_success_probe.v1"


class RigidProbeScene(Protocol):
    def set_scoring_pose(self, pose: Sequence[float]) -> None: ...

    def step(self) -> None: ...

    def read_sample(self) -> Mapping[str, Any]: ...


def _reading(scene: RigidProbeScene, *, step_index: int, phase: str) -> dict[str, Any]:
    value = scene.read_sample()
    pose = [round(float(v), 7) for v in value["task_scoring_pose_world"]]
    force = float(value.get("scene_collision_force_n", 0.0))
    support = value.get("support_contact_active")
    if len(pose) != 7 or not all(math.isfinite(v) for v in [*pose, force]) or not isinstance(support, bool):
        raise RuntimeError("rigid_robot_free_probe_readback_invalid")
    return {"step_index": step_index, "phase": phase, "pose": pose, "support_contact_active": support,
            "scene_collision_force_n": round(force, 7)}


def record_rigid_robot_free_probe(*, scene: RigidProbeScene, plan: Mapping[str, Any],
                                  binding: Mapping[str, Any]) -> dict[str, Any]:
    """Run untouched, carry and release; return the sealed raw trace.

    ``binding`` names what was probed (configured scene revision, task spec,
    replacement asset digest, scene context) and is copied into the trace.
    """

    trace: dict[str, Any] = {"schema_version": TRACE_SCHEMA, "binding": dict(binding), "plan": dict(plan)}
    try:
        if plan.get("schema_version") != PLAN_SCHEMA or plan.get("plan_digest") != canonical_digest(
                dict(plan), digest_field="plan_digest"):
            raise RuntimeError("rigid_robot_free_probe_plan_invalid")
        hz = int(plan["physics_hz"])
        rows: list[dict[str, Any]] = []
        for _ in range(round(float(plan["zero_action_seconds"]) * hz)):
            scene.step()
            rows.append(_reading(scene, step_index=len(rows), phase="zero_action"))
        # The carry starts wherever physics left the untouched object.
        waypoints = carry_waypoints(rows[-1]["pose"], plan["release_scoring_pose_world"],
                                    float(plan["carry_lift_m"]))
        for (a, b), seconds in zip(zip(waypoints, waypoints[1:]), plan["carry_seconds"], strict=True):
            count = max(1, round(float(seconds) * hz))
            for k in range(count):
                scene.set_scoring_pose(interpolate_pose(a, b, (k + 1) / count))
                scene.step()
                rows.append(_reading(scene, step_index=len(rows), phase="carry"))
        for _ in range(round(float(plan["release_seconds"]) * hz)):
            scene.step()
            rows.append(_reading(scene, step_index=len(rows), phase="release"))
        trace.update(status="recorded", physics_hz=hz, timestep_seconds=1.0 / hz, samples=rows,
                     proof_boundary={"robot_embodiment_used": False,
                                     "drive": "kinematic_state_setting_then_release", "scored_here": False})
    except Exception as exc:  # noqa: BLE001 - a probe failure is evidence, never a crash
        trace.update(status="failed", error=f"{type(exc).__name__}:{str(exc)[:200]}")
    trace["trace_digest"] = ""
    trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
    return trace


__all__ = ["TRACE_SCHEMA", "RigidProbeScene", "record_rigid_robot_free_probe"]
