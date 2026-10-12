"""Robot-free readiness: score the native probe with the exact configured success contract.

Owner decision 2026-10-12: robots never run in construction. A configured
articulated scene is ready for robot teams' evaluation when, under the same
scorer their episodes are judged by, the scene left untouched does not succeed
and the task joint driven into its goal interval and then released does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .adp_task_scoring import (
    JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD,
    TaskNeutralScoringError,
    score_articulated_task_episode,
    validate_articulated_task_spec,
)
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_rigid_relocation_native_adapter import NATIVE_PHYSICS_FREQUENCY_HZ
from .task_evaluation_scene_configuration_native_import_driver import ROBOT_FREE_PROBE_SCHEMA

SCHEMA = "task_evaluation_configured_scene_robot_free_readiness.v1"


def _scorer_trace(rows: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any],
                  native_names: Mapping[str, str | None]) -> list[dict[str, Any]]:
    """Probe readings as scorer samples. A fixed joint has no DOF and holds its reset."""

    resets, limits = spec["joint_reset_positions"], spec["joint_hard_limits"]
    trace = []
    for index, row in enumerate(rows):
        positions, velocities = {}, {}
        for joint_id, reset in resets.items():
            name = native_names.get(joint_id)
            position, velocity = (float(reset), 0.0) if name is None else (float(v) for v in row["dof"][name])
            positions[joint_id], velocities[joint_id] = position, velocity
        contact = row["phase"] == "drive"
        trace.append({
            "step_index": index, "joint_positions": positions, "joint_velocities_per_s": velocities,
            "task_contact_active": contact,
            "joint_limit_violation": any(
                not limits[joint_id][0] - JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD <= value
                <= limits[joint_id][1] + JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD
                for joint_id, value in positions.items()),
            # Nothing but the joint driver ever touches the asset, and it lets go.
            "containment_violation": False, "robot_collision_failure": False,
            "scene_collision_failure": False, "retreat_completed": not contact,
        })
    return trace


def _check(spec: Mapping[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        report = score_articulated_task_episode(task_spec=spec, samples=samples)
    except TaskNeutralScoringError as exc:
        return {"status": "unscorable", "task_succeeded": None, "errors": list(exc.args[0])[:20],
                "sample_count": len(samples)}
    return {"status": report["status"], "task_succeeded": report["task_succeeded"],
            "outcome": report["outcome"], "sample_count": len(samples)}


def score_robot_free_readiness(*, task_spec: Mapping[str, Any], task_joint_bindings: Sequence[Mapping[str, Any]],
                               native_import_result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a sealed receipt; only both checks passing admits learned-policy evaluation."""

    spec = dict(task_spec)
    normalized = validate_articulated_task_spec(spec)
    probe = native_import_result.get("robot_free_success_probe")
    base = {"schema_version": SCHEMA, "native_import_result_digest": native_import_result.get("result_digest"),
            "task_spec_digest": canonical_digest(spec)}
    boundary = {"robot_embodiment_used": False, "scene_context": "replacement_asset_on_ground_plane",
                "scene_collisions_modeled": False, "containment_modeled": False,
                "joint_limit_violation_source": "readback_against_qualified_limits",
                "simulator_evidence_is_physical_truth": False}
    if (not isinstance(probe, Mapping) or probe.get("schema_version") != ROBOT_FREE_PROBE_SCHEMA
            or probe.get("status") != "recorded"):
        reason = probe.get("error") if isinstance(probe, Mapping) else None
        receipt = {**base, "status": "probe_unavailable", "probe_error": reason, "proof_boundary": boundary,
                   "learned_policy_evaluation_admitted": False, "receipt_digest": ""}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        return receipt
    control_hz = int(spec["control_frequency_hz"])
    decimation = round(NATIVE_PHYSICS_FREQUENCY_HZ / control_hz)
    if abs(round(1.0 / float(probe["timestep_seconds"])) - NATIVE_PHYSICS_FREQUENCY_HZ) > 0:
        raise ValueError("robot_free_probe_physics_rate_mismatch")
    native_names = {str(row["joint_id"]): row.get("native_joint_name") for row in task_joint_bindings}
    samples = list(probe["samples"])
    untouched = [row for row in samples if row["phase"] == "zero_action"]
    driven = [row for row in samples if row["phase"] in {"drive", "release"}]
    # One reading per control step, as a robot team's episode is sampled; the
    # driven episode starts from the untouched scene's final reset reading.
    zero_action = _check(spec, _scorer_trace(untouched[decimation - 1::decimation], spec=normalized,
                                             native_names=native_names))
    direct_drive = _check(spec, _scorer_trace(untouched[-1:] + driven[decimation - 1::decimation],
                                              spec=normalized, native_names=native_names))
    checks = {"zero_action_not_success": zero_action["task_succeeded"] is False and zero_action["status"] == "scored",
              "direct_drive_success": direct_drive["task_succeeded"] is True}
    passed = all(checks.values())
    receipt = {**base, "status": "passed" if passed else "failed",
               "probe": {"physics_hz": NATIVE_PHYSICS_FREQUENCY_HZ, "control_frequency_hz": control_hz,
                         "plan": dict(probe["plan"])},
               "zero_action": zero_action, "direct_drive": direct_drive, "checks": checks,
               "proof_boundary": boundary, "learned_policy_evaluation_admitted": passed, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = ["SCHEMA", "score_robot_free_readiness"]
