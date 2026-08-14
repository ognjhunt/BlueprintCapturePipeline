"""Bound the Newton Robotiq drive before it can enter matched controls.

The Arena DROID asset inherits a very stiff USD position drive while MJWarp
does not enforce its authored velocity limit.  This module defines one
reversible, Newton-only identification candidate: retain the sealed stiffness
and effort/speed limits, add the minimum control-derived effective armature,
critically damp it, and rate-limit the binary target at the authored speed.
It remains comparison-ineligible until the native step trace passes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

try:  # flat provider bundle
    from decision_evidence_contracts import canonical_digest
except ImportError:  # installed package
    from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009d_newton_gripper_drive_candidate.v1"
RECEIPT_SCHEMA_VERSION = "adp009d_newton_gripper_drive_configuration.v1"
SOURCE_STIFFNESS_NM_PER_RAD = 5729.58
SOURCE_DAMPING_NM_S_PER_RAD = 0.0114592
SOURCE_EFFECTIVE_INERTIA_KG_M2 = 3.80173e-7
SOURCE_EFFORT_LIMIT_NM = 16.5
SOURCE_VELOCITY_LIMIT_RAD_S = 1.0
PHYSICS_DT_SECONDS = 1.0 / 120.0
CONTROL_DECIMATION = 8
FULL_STROKE_RAD = math.pi / 4.0
FULL_STROKE_M = 0.085
MANUFACTURER_SPEED_RANGE_M_S = (0.020, 0.150)
VELOCITY_READBACK_TOLERANCE_RAD_S = 0.05
SETTLED_VELOCITY_TOLERANCE_RAD_S = 0.05


def build_newton_gripper_drive_candidate() -> dict[str, Any]:
    """Return the provider-free drive candidate derived from sealed limits."""

    armature = (
        SOURCE_EFFORT_LIMIT_NM
        * PHYSICS_DT_SECONDS
        / SOURCE_VELOCITY_LIMIT_RAD_S
    )
    effective_inertia = SOURCE_EFFECTIVE_INERTIA_KG_M2 + armature
    damping = 2.0 * math.sqrt(SOURCE_STIFFNESS_NM_PER_RAD * effective_inertia)
    stability_ratio = (
        math.sqrt(SOURCE_STIFFNESS_NM_PER_RAD / effective_inertia)
        * PHYSICS_DT_SECONDS
    )
    target_step = (
        SOURCE_VELOCITY_LIMIT_RAD_S * PHYSICS_DT_SECONDS * CONTROL_DECIMATION
    )
    fingertip_speed = (
        SOURCE_VELOCITY_LIMIT_RAD_S * FULL_STROKE_M / FULL_STROKE_RAD
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "provider_free_candidate_ready",
        "physics_backend": "newton",
        "joint_name": "finger_joint",
        "source_drive": {
            "stiffness_nm_per_rad": SOURCE_STIFFNESS_NM_PER_RAD,
            "damping_nm_s_per_rad": SOURCE_DAMPING_NM_S_PER_RAD,
            "effective_inertia_kg_m2": SOURCE_EFFECTIVE_INERTIA_KG_M2,
            "effort_limit_nm": SOURCE_EFFORT_LIMIT_NM,
            "velocity_limit_rad_s": SOURCE_VELOCITY_LIMIT_RAD_S,
        },
        "candidate_drive": {
            "stiffness_nm_per_rad": SOURCE_STIFFNESS_NM_PER_RAD,
            "damping_nm_s_per_rad": damping,
            "armature_kg_m2": armature,
            "effective_inertia_kg_m2": effective_inertia,
            "target_rate_limit_rad_s": SOURCE_VELOCITY_LIMIT_RAD_S,
            "maximum_target_step_rad": target_step,
        },
        "derivation": {
            "physics_dt_seconds": PHYSICS_DT_SECONDS,
            "control_decimation": CONTROL_DECIMATION,
            "maximum_velocity_delta_per_max_torque_step_rad_s": (
                SOURCE_EFFORT_LIMIT_NM * PHYSICS_DT_SECONDS / effective_inertia
            ),
            "explicit_stability_ratio": stability_ratio,
            "explicit_stability_ratio_limit": 2.0,
            "damping_ratio": 1.0,
            "rated_fingertip_speed_m_s": fingertip_speed,
            "manufacturer_speed_range_m_s": list(MANUFACTURER_SPEED_RANGE_M_S),
        },
        "native_acceptance": {
            "finite_joint_state_required": True,
            "maximum_abs_joint_velocity_rad_s": (
                SOURCE_VELOCITY_LIMIT_RAD_S
                + VELOCITY_READBACK_TOLERANCE_RAD_S
            ),
            "settled_abs_joint_velocity_rad_s": SETTLED_VELOCITY_TOLERANCE_RAD_S,
            "minimum_finger_separation_travel_m": 1.0e-3,
        },
        "comparison_eligible": False,
        "claim_ceiling": "newton_native_drive_identification_candidate_only",
        "contract_digest": "",
    }
    value["contract_digest"] = canonical_digest(
        value, digest_field="contract_digest"
    )
    return value


def validate_newton_gripper_drive_candidate(value: Mapping[str, Any]) -> list[str]:
    """Reject drift from the one provider-free candidate."""

    expected = build_newton_gripper_drive_candidate()
    return (
        []
        if dict(value) == expected
        else ["adp009d_newton_gripper_drive_candidate_invalid"]
    )


def configure_newton_gripper_drive_candidate(
    embodiment: Any, *, expected_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the candidate before Newton finalizes its model."""

    if validate_newton_gripper_drive_candidate(expected_contract):
        raise RuntimeError("adp009d_newton_gripper_drive_candidate_invalid")
    actuator = embodiment.scene_config.robot.actuators.get("gripper")
    if (
        actuator is None
        or actuator.stiffness is not None
        or actuator.damping is not None
        or actuator.velocity_limit != SOURCE_VELOCITY_LIMIT_RAD_S
    ):
        raise RuntimeError("adp009d_newton_gripper_source_drive_invalid")
    candidate = expected_contract["candidate_drive"]
    actuator.stiffness = candidate["stiffness_nm_per_rad"]
    actuator.damping = candidate["damping_nm_s_per_rad"]
    actuator.armature = candidate["armature_kg_m2"]

    from isaaclab_arena.embodiments.droid.actions import (
        BinaryJointPositionZeroToOneAction,
    )

    max_step = float(candidate["maximum_target_step_rad"])

    class RateLimitedBinaryJointPositionAction(BinaryJointPositionZeroToOneAction):
        def process_actions(self, actions):
            super().process_actions(actions)
            current = self._asset.data.joint_pos[:, self._joint_ids]
            self._processed_actions = self._processed_actions.clamp(
                min=current - max_step, max=current + max_step
            )

    action_cfg = embodiment.action_config.gripper_action
    if action_cfg.class_type is not BinaryJointPositionZeroToOneAction:
        raise RuntimeError("adp009d_newton_gripper_action_source_invalid")
    action_cfg.class_type = RateLimitedBinaryJointPositionAction
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "applied_for_native_identification",
        "contract_digest": expected_contract["contract_digest"],
        "joint_name": "finger_joint",
        "configured_stiffness_nm_per_rad": actuator.stiffness,
        "configured_damping_nm_s_per_rad": actuator.damping,
        "configured_armature_kg_m2": actuator.armature,
        "target_rate_limiter_class": RateLimitedBinaryJointPositionAction.__name__,
        "comparison_eligible": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def assess_newton_gripper_drive_trace(
    *, positions_rad: list[float], velocities_rad_s: list[float]
) -> dict[str, Any]:
    """Assess one native command trace without upgrading it to fidelity proof."""

    finite = bool(positions_rad) and len(positions_rad) == len(velocities_rad_s) and all(
        math.isfinite(value) for value in [*positions_rad, *velocities_rad_s]
    )
    maximum_velocity = max((abs(value) for value in velocities_rad_s), default=math.inf)
    settled_velocity = abs(velocities_rad_s[-1]) if velocities_rad_s else math.inf
    blockers: list[str] = []
    if not finite:
        blockers.append("adp009d_newton_gripper_drive_nonfinite")
    if maximum_velocity > SOURCE_VELOCITY_LIMIT_RAD_S + VELOCITY_READBACK_TOLERANCE_RAD_S:
        blockers.append("adp009d_newton_gripper_drive_velocity_exceeded")
    if settled_velocity > SETTLED_VELOCITY_TOLERANCE_RAD_S:
        blockers.append("adp009d_newton_gripper_drive_not_settled")
    return {
        "status": "passed" if not blockers else "blocked",
        "sample_count": len(positions_rad),
        "finite_joint_state": finite,
        "maximum_abs_joint_velocity_rad_s": maximum_velocity,
        "settled_abs_joint_velocity_rad_s": settled_velocity,
        "blockers": blockers,
    }


def measure_gripper_convention_and_newton_drive(
    *, env: Any, action: Any, robot: Any, torch: Any, to_torch: Any, backend: str
) -> dict[str, Any]:
    """Measure command convention and retain Newton's native drive trace."""

    finger_pair = ("left_inner_finger", "right_inner_finger")
    body_names = list(robot.body_names)
    finger_indices = [body_names.index(name) for name in finger_pair if name in body_names]
    probe: dict[str, Any] = {
        "schema_version": "adp009d_gripper_convention_probe.v1",
        "candidate_commands": [0.0, 1.0],
        "finger_bodies": list(finger_pair),
        "settle_steps": 30,
    }
    if len(finger_indices) != 2:
        probe.update(
            status="blocked", blockers=["gripper_convention_finger_bodies_missing"]
        )
    else:
        separations: dict[str, float] = {}
        drive_traces: dict[str, dict[str, Any]] = {}
        joint_names = list(robot.joint_names)
        joint_index = joint_names.index("finger_joint") if "finger_joint" in joint_names else None
        for command in (0.0, 1.0):
            env.reset(seed=20260806)
            probe_action = torch.zeros_like(action)
            probe_action[:, :7] = to_torch(robot.data.joint_pos)[:, :7]
            probe_action[:, 7] = command
            positions: list[float] = []
            velocities: list[float] = []
            for _ in range(30):
                env.step(probe_action)
                if joint_index is not None:
                    positions.append(float(to_torch(robot.data.joint_pos)[0, joint_index]))
                    velocities.append(float(to_torch(robot.data.joint_vel)[0, joint_index]))
            poses = to_torch(robot.data.body_pose_w)[0, finger_indices, :3]
            separations[str(command)] = float(torch.linalg.vector_norm(poses[0] - poses[1]))
            if backend == "newton":
                drive_traces[str(command)] = assess_newton_gripper_drive_trace(
                    positions_rad=positions, velocities_rad_s=velocities
                )
        travel = abs(separations["0.0"] - separations["1.0"])
        probe.update(finger_separation_m=separations, separation_travel_m=travel)
        if travel < 1.0e-3:
            probe.update(status="ambiguous", blockers=["gripper_convention_travel_below_floor"])
        else:
            closes_at = 1.0 if separations["1.0"] < separations["0.0"] else 0.0
            probe.update(
                status="measured",
                blockers=[],
                closed_command=closes_at,
                open_command=1.0 - closes_at,
            )
        if backend == "newton":
            blockers = sorted(
                {blocker for trace in drive_traces.values() for blocker in trace["blockers"]}
            )
            probe["newton_drive_traces"] = drive_traces
            if joint_index is None:
                blockers.append("adp009d_newton_gripper_joint_missing")
            if blockers:
                probe.update(status="blocked", blockers=sorted(set(blockers)))
    probe["probe_digest"] = canonical_digest(probe, digest_field="probe_digest")
    return probe


def validate_newton_gripper_drive_probe(
    *, conversion: Mapping[str, Any], profile: Mapping[str, Any], trace: object
) -> list[str]:
    """Validate contract/configuration/trace binding in a native probe."""

    drive = dict(profile.get("gripper_drive_candidate") or {})
    trace_row = dict(trace) if isinstance(trace, Mapping) else {}
    receipt_digest = conversion.get("newton_gripper_drive_receipt_digest")
    if (
        conversion.get("newton_gripper_drive_contract_digest")
        != drive.get("contract_digest")
        or conversion.get("newton_gripper_drive_status")
        != "applied_for_native_identification"
        or not isinstance(receipt_digest, str)
        or not receipt_digest.startswith("sha256:")
        or trace_row.get("status") != "passed"
        or trace_row.get("blockers") != []
    ):
        return ["adp009d_newton_probe_gripper_drive_invalid"]
    return []


def build_newton_gripper_probe_fields(
    *,
    backend: str,
    profile: Mapping[str, Any],
    mapping_receipt: Mapping[str, Any] | None,
    convention_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the configuration and trace into the backend probe schema."""

    if backend != "newton":
        return {
            "asset_conversion": {
                "newton_gripper_drive_contract_digest": None,
                "newton_gripper_drive_status": None,
                "newton_gripper_drive_receipt_digest": None,
            },
            "trace": None,
        }
    configuration = dict((mapping_receipt or {}).get("gripper_drive_configuration") or {})
    trace = (
        {
            "status": "passed",
            "blockers": [],
            "commands": convention_probe["newton_drive_traces"],
        }
        if convention_probe.get("status") == "measured"
        else None
    )
    return {
        "asset_conversion": {
            "newton_gripper_drive_contract_digest": profile[
                "gripper_drive_candidate"
            ]["contract_digest"],
            "newton_gripper_drive_status": configuration.get("status"),
            "newton_gripper_drive_receipt_digest": configuration.get(
                "receipt_digest"
            ),
        },
        "trace": trace,
    }


__all__ = [
    "assess_newton_gripper_drive_trace",
    "build_newton_gripper_drive_candidate",
    "build_newton_gripper_probe_fields",
    "configure_newton_gripper_drive_candidate",
    "measure_gripper_convention_and_newton_drive",
    "validate_newton_gripper_drive_probe",
    "validate_newton_gripper_drive_candidate",
]
